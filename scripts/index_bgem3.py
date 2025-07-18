#!/usr/bin/env python3
"""
BGE-M3を使用したObsidian Vaultのインデックス作成スクリプト
Apple M2 Pro等のMetal Performance Shaders (MPS) GPUに最適化
"""
import os
import sys
from pathlib import Path
import hashlib
from datetime import datetime
import time
from tqdm import tqdm
import torch

sys.path.append(str(Path(__file__).parent))

from common.utils import setup_environment, setup_logging, get_vault_files, format_file_info
from common.chunker import MarkdownChunker
from common.db import VectorDB
from FlagEmbedding import BGEM3FlagModel


class BGEM3Indexer:
    """BGE-M3によるインデックス作成クラス"""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(config, 'index_bgem3')

        self.vault_path = Path(os.getenv('VAULT_PATH'))
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {self.vault_path}")

        embed_config = config['embeddings']['bgem3']
        self.model_name = embed_config['model_name']

        # Apple Silicon GPU (MPS) を使用
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        use_fp16 = embed_config['use_fp16'] and self.device == "mps"

        # MPSメモリ管理の最適化
        if self.device == "mps":
            torch.mps.set_per_process_memory_fraction(0.0)  # 自動管理

        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"FP16: {use_fp16}")

        self.logger.info(f"Loading model: {self.model_name}")
        self.model = BGEM3FlagModel(
            self.model_name,
            use_fp16=use_fp16,
            device=self.device,
            normalize_embeddings=True  # 正規化を明示的に有効化
        )

        # メモリ使用量を抑えるための設定
        if self.device == "mps":
            # グラディエントの蓄積を無効化
            torch.set_grad_enabled(False)

        chunk_config = config['chunking']['bgem3']
        self.chunker = MarkdownChunker(
            max_chunk_size=chunk_config['max_chunk_size'],
            chunk_overlap=config['chunking']['chunk_overlap'],
            min_chunk_size=config['chunking']['min_chunk_size']
        )

        # バッチサイズ設定（MPSもGPU扱い）
        self.batch_size = chunk_config['batch_size_gpu'] if self.device in ["cuda", "mps"] else chunk_config['batch_size_cpu']
        self.logger.info(f"Batch size: {self.batch_size}")

        self.db = VectorDB(config['database']['persist_directory'])
        self.collection_name = config['database']['collection_bgem3']

    def save_execution_stats(self, db_stats, total_time, file_count, chunk_count, 
                            total_processed_tokens, estimated_original_tokens):
        """実行統計をファイルに保存"""
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        filename_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
        stats_file = Path("results") / f"indexing_stats_bgem3_{filename_timestamp}.md"
        stats_file.parent.mkdir(exist_ok=True)

        content = f"""# BGE-M3 インデックス作成統計

## 実行情報
- 実行日時: {timestamp_str}
- モデル: {self.model_name}
- デバイス: {self.device}
- 使用FP16: {self.device == "mps"}

## 処理統計
- 処理ファイル数: {file_count}
- 生成チャンク数: {chunk_count}
- 平均チャンク数/ファイル: {chunk_count/file_count:.1f}

## トークン統計
- 総処理トークン数（オーバーラップ含む）: {total_processed_tokens:,}
- 推定元トークン数（オーバーラップ除く）: {estimated_original_tokens:,}
- 平均トークン数/チャンク: {total_processed_tokens/chunk_count:.0f}

## データベース統計
- コレクション名: {db_stats['collection_name']}
- 保存ドキュメント数: {db_stats['document_count']}
- ベクトル次元数: {db_stats['dimension']}

## パフォーマンス
- 総実行時間: {total_time:.2f}秒 ({total_time/60:.1f}分)
- ファイル処理速度: {file_count/total_time:.2f} files/秒
- チャンク処理速度: {chunk_count/total_time:.2f} chunks/秒

## 設定
- バッチサイズ: {self.batch_size}
- 最大チャンクサイズ: {self.chunker.max_chunk_size} tokens
- チャンクオーバーラップ: {self.chunker.chunk_overlap} tokens
- 最小チャンクサイズ: {self.chunker.min_chunk_size} tokens

---
"""

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.info(f"統計情報を保存しました: {stats_file}")

    def process_file(self, file_path: Path) -> list:
        """単一ファイルを処理してチャンクを生成"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            file_metadata = format_file_info(file_path, self.vault_path)
            file_metadata['indexed_at'] = datetime.now().isoformat()
            file_metadata['model'] = self.model_name

            chunks = self.chunker.chunk_text(content, file_metadata)

            return chunks

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return []

    def generate_embeddings(self, texts: list) -> list:
        """テキストのリストから埋め込みを生成"""
        with torch.no_grad():  # グラディエントを保持しない
            # バッチ処理で効率的に埋め込みを生成
            # max_lengthを適切に設定することでメモリ効率を改善
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,  # 設定されたバッチサイズを使用
                max_length=self.chunker.max_chunk_size,  # チャンクサイズと同じに設定
                convert_to_numpy=True  # メモリ効率のためNumPy配列で返す
            )

            # Dense埋め込みを取得（正規化済み）
            # encode()は常にdictを返す
            dense_vecs = embeddings['dense_vecs']

            # convert_to_numpy=Trueのため、常にNumPy配列
            result = dense_vecs.tolist()

            # メモリ解放
            del dense_vecs
            del embeddings

            return result


    def _process_chunk_batch(self, chunks, collection):
        """チャンクのバッチを処理してDBに保存"""
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]

            texts = []
            for chunk in batch_chunks:
                metadata_text = f"""
Note title: {chunk.metadata.get('relative_path', '')}
Section title: {chunk.metadata.get('heading', '')}
Last modified: {chunk.metadata.get('last_modified', '')[:10] if chunk.metadata.get('last_modified') else ''}

{chunk.content}
"""
                texts.append(metadata_text)

            metadatas = [chunk.metadata for chunk in batch_chunks]

            embeddings = self.generate_embeddings(texts)

            # IDの生成（ファイルパス + チャンクインデックスのハッシュ）
            ids = []
            for chunk in batch_chunks:
                id_string = f"{chunk.metadata['relative_path']}_{chunk.metadata['chunk_index']}"
                chunk_id = hashlib.md5(id_string.encode()).hexdigest()
                ids.append(chunk_id)

            # DBに追加（既存データがある場合は更新）
            try:
                collection.upsert(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                self.logger.error(f"DB追加エラー: {e}")
                # ChromaDBのadd_documentsメソッドを使用している場合
                self.db.add_documents(
                    collection=collection,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )

            # メモリ解放
            del embeddings, texts, metadatas, ids

    def run(self):
        """インデックス作成を実行"""
        self.logger.info("=== BGE-M3 インデックス作成開始 ===")
        start_time = time.time()

        exclude_patterns = self.config['vault'].get('exclude_patterns', [])
        files = get_vault_files(self.vault_path, exclude_patterns)
        self.logger.info(f"対象ファイル数: {len(files)}")

        # コレクションの作成/取得（BGE-M3は1024次元）
        collection = self.db.get_or_create_collection(self.collection_name, 1024)

        # メモリ効率的な処理のため、バッチごとに処理
        total_chunks = 0
        total_processed_tokens = 0  # オーバーラップを含む処理トークン数
        file_batch_size = 10

        with tqdm(total=len(files), desc="ファイル処理", unit="files") as pbar:
            batch_chunks = None
            for file_idx in range(0, len(files), file_batch_size):
                batch_files = files[file_idx:file_idx + file_batch_size]
                batch_chunks = []

                for file_path in batch_files:
                    chunks = self.process_file(file_path)
                    batch_chunks.extend(chunks)
                    pbar.update(1)

                    if chunks:
                        file_tokens = sum(c.token_count for c in chunks)
                        total_processed_tokens += file_tokens
                        self.logger.debug(f"ファイル: {file_path.name}, チャンク数: {len(chunks)}, 総トークン数: {file_tokens}")

                if batch_chunks:
                    self.logger.info(f"埋め込み生成中: {len(batch_chunks)}チャンク ({self.batch_size}個ずつ処理)")
                    self._process_chunk_batch(batch_chunks, collection)
                    total_chunks += len(batch_chunks)

            del batch_chunks
            if self.device == "mps":
                torch.mps.empty_cache()

        self.logger.info(f"総チャンク数: {total_chunks}")
        self.logger.info(f"総処理トークン数: {total_processed_tokens:,}")
        
        # オーバーラップを除いた推定元トークン数（簡易計算）
        estimated_original_tokens = total_processed_tokens - (total_chunks - len(files)) * self.chunker.chunk_overlap
        self.logger.info(f"推定元トークン数（オーバーラップ除く）: {estimated_original_tokens:,}")

        stats = self.db.get_collection_stats(self.collection_name)
        end_time = time.time()
        total_time = end_time - start_time

        self.logger.info("=== インデックス作成完了 ===")
        self.logger.info(f"コレクション: {stats['collection_name']}")
        self.logger.info(f"ドキュメント数: {stats['document_count']}")
        self.logger.info(f"次元数: {stats['dimension']}")
        self.logger.info(f"総実行時間: {total_time:.2f}秒")

        self.save_execution_stats(stats, total_time, len(files), total_chunks, 
                                 total_processed_tokens, estimated_original_tokens)


def main():
    config = setup_environment()

    indexer = BGEM3Indexer(config)
    indexer.run()


if __name__ == "__main__":
    main()
