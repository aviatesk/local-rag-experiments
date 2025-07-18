#!/usr/bin/env python3
"""
BGE-M3を使用したObsidian Vaultのインクリメンタル更新スクリプト
特定のノートのみを更新し、古いチャンクは自動的に削除
"""
import os
import sys
from pathlib import Path
import hashlib
from datetime import datetime
import time
from tqdm import tqdm
import torch
import click

# 共通モジュールのパスを追加
sys.path.append(str(Path(__file__).parent))

from common.utils import setup_environment, setup_logging, format_file_info
from common.chunker import MarkdownChunker
from common.db import VectorDB
from FlagEmbedding import BGEM3FlagModel


class BGEM3IncrementalUpdater:
    """BGE-M3によるインクリメンタル更新クラス"""

    def __init__(self, config, target_files=None):
        self.config = config
        self.logger = setup_logging(config, 'update_bgem3')
        self.target_files = target_files  # 更新対象ファイルのリスト

        # Vault設定
        self.vault_path = Path(os.getenv('VAULT_PATH'))
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {self.vault_path}")

        # モデル設定
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

        # モデル初期化
        self.logger.info(f"Loading model: {self.model_name}")
        self.model = BGEM3FlagModel(
            self.model_name,
            use_fp16=use_fp16,
            device=self.device,
            normalize_embeddings=True  # 正規化を明示的に有効化
        )

        # メモリ使用量を抑えるための設定
        if self.device == "mps":
            torch.set_grad_enabled(False)

        # チャンカー初期化
        chunk_config = config['chunking']['bgem3']
        self.chunker = MarkdownChunker(
            max_chunk_size=chunk_config['max_chunk_size'],
            chunk_overlap=config['chunking']['chunk_overlap'],
            min_chunk_size=config['chunking']['min_chunk_size']
        )

        # バッチサイズ設定
        self.batch_size = chunk_config['batch_size_gpu'] if self.device in ["cuda", "mps"] else chunk_config['batch_size_cpu']
        self.logger.info(f"Batch size: {self.batch_size}")

        # DB初期化
        self.db = VectorDB(config['database']['persist_directory'])
        self.collection_name = config['database']['collection_bgem3']

    # インクリメンタル更新の統計はbenchmark.pyで計測するため、個別ファイルとして保存しない
    # def save_execution_stats(self, stats):
    #     """実行統計をファイルに保存"""
    #     timestamp = datetime.now()
    #     timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    #     filename_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
    #     stats_file = Path("results") / f"incremental_update_bgem3_{filename_timestamp}.md"
    #     stats_file.parent.mkdir(exist_ok=True)
    #
    #     content = f"""# BGE-M3 インクリメンタル更新統計
    #
    # ## 実行情報
    # - 実行日時: {timestamp_str}
    # - モデル: {self.model_name}
    # - デバイス: {self.device}
    # - 更新モード: インクリメンタル
    #
    # ## 更新対象
    # - 更新ファイル数: {stats['updated_files']}
    # - 対象ファイル: {', '.join(stats['target_files'][:5])}{'...' if len(stats['target_files']) > 5 else ''}
    #
    # ## 処理統計
    # - 新規/更新チャンク数: {stats['new_chunks']}
    # - 削除チャンク数: {stats['deleted_chunks']}
    # - 処理前の総チャンク数: {stats['chunks_before']}
    # - 処理後の総チャンク数: {stats['chunks_after']}
    #
    # ## パフォーマンス
    # - 総実行時間: {stats['total_time']:.2f}秒 ({stats['total_time']/60:.1f}分)
    # - ファイル処理時間: {stats['processing_time']:.2f}秒
    # - DB更新時間: {stats['db_time']:.2f}秒
    # - 平均処理速度: {stats['new_chunks']/stats['total_time']:.2f} chunks/秒
    #
    # ## 設定
    # - バッチサイズ: {self.batch_size}
    # - 最大チャンクサイズ: {self.chunker.max_chunk_size} tokens
    #
    # ---
    # """
    #
    #     with open(stats_file, 'w', encoding='utf-8') as f:
    #         f.write(content)
    #
    #     self.logger.info(f"統計情報を保存しました: {stats_file}")

    def process_file(self, file_path: Path) -> list:
        """単一ファイルを処理してチャンクを生成"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # ファイルメタデータ
            file_metadata = format_file_info(file_path, self.vault_path)
            file_metadata['indexed_at'] = datetime.now().isoformat()
            file_metadata['model'] = self.model_name

            # チャンク分割
            chunks = self.chunker.chunk_text(content, file_metadata)

            return chunks

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return []

    def generate_embeddings(self, texts: list) -> list:
        """テキストのリストから埋め込みを生成"""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                max_length=self.chunker.max_chunk_size,
                convert_to_numpy=True
            )

            # encode()は常にdictを返す
            dense_vecs = embeddings['dense_vecs']

            # convert_to_numpy=Trueのため、常にNumPy配列
            result = dense_vecs.tolist()

            del dense_vecs
            del embeddings

            return result

    def update_files(self, files: list):
        """指定されたファイルを更新"""
        collection = self.db.client.get_collection(self.collection_name)

        stats = {
            'updated_files': len(files),
            'target_files': [str(f.relative_to(self.vault_path)) for f in files],
            'new_chunks': 0,
            'deleted_chunks': 0,
            'chunks_before': collection.count(),
            'processing_time': 0,
            'db_time': 0,
            'total_time': 0
        }

        start_time = time.time()

        # 各ファイルを処理
        for file_path in tqdm(files, desc="ファイル更新"):
            relative_path = str(file_path.relative_to(self.vault_path))

            # 処理開始時刻
            file_start_time = time.time()

            # 1. 既存のチャンクを削除
            db_start = time.time()
            deleted_count = self.db.delete_by_metadata(
                collection,
                {"relative_path": relative_path}
            )
            stats['deleted_chunks'] += deleted_count

            # 2. ファイルが存在する場合は新しいチャンクを生成
            if file_path.exists():
                # チャンク生成
                chunks = self.process_file(file_path)

                if chunks:
                    # バッチ処理で埋め込み生成とDB保存
                    for i in range(0, len(chunks), self.batch_size):
                        batch_chunks = chunks[i:i + self.batch_size]

                        # メタデータを含むテキストを生成
                        texts = []
                        for chunk in batch_chunks:
                            metadata_text = f"""
Note title: {chunk.metadata.get('relative_path', '')}
Section title: {chunk.metadata.get('heading', '')}
Last modified: {chunk.metadata.get('last_modified', '')[:10] if chunk.metadata.get('last_modified') else ''}

{chunk.content}"""
                            texts.append(metadata_text)

                        metadatas = [chunk.metadata for chunk in batch_chunks]

                        # 埋め込み生成
                        embeddings = self.generate_embeddings(texts)

                        # IDの生成
                        ids = []
                        for chunk in batch_chunks:
                            id_string = f"{chunk.metadata['relative_path']}_{chunk.metadata['chunk_index']}"
                            chunk_id = hashlib.md5(id_string.encode()).hexdigest()
                            ids.append(chunk_id)

                        # DBに追加
                        collection.upsert(
                            embeddings=embeddings,
                            documents=texts,
                            metadatas=metadatas,
                            ids=ids
                        )

                        stats['new_chunks'] += len(batch_chunks)

                self.logger.info(f"更新完了: {relative_path} ({deleted_count}削除, {len(chunks)}追加)")
            else:
                # ファイルが削除された場合
                self.logger.info(f"ファイル削除を検出: {relative_path} ({deleted_count}チャンク削除)")

            # 時間計測
            db_end = time.time()
            stats['processing_time'] += (db_start - file_start_time)
            stats['db_time'] += (db_end - db_start)

            # メモリ解放
            if self.device == "mps":
                torch.mps.empty_cache()

        # 最終統計
        stats['chunks_after'] = collection.count()
        stats['total_time'] = time.time() - start_time

        return stats

    def run(self):
        """インクリメンタル更新を実行"""
        self.logger.info("=== BGE-M3 インクリメンタル更新開始 ===")

        # 更新対象ファイルの決定
        if self.target_files:
            # 指定されたファイルのみ
            files = []
            for file_pattern in self.target_files:
                file_path = self.vault_path / file_pattern
                if file_path.exists():
                    files.append(file_path)
                else:
                    self.logger.warning(f"ファイルが見つかりません: {file_pattern}")
        else:
            # config.yamlのupdate_notesから読み込み
            update_notes = self.config['vault'].get('update_notes', [])
            if not update_notes:
                self.logger.error("更新対象のファイルが指定されていません")
                self.logger.info("config.yamlのvault.update_notesに更新したいファイルを指定するか、")
                self.logger.info("コマンドライン引数でファイルを指定してください")
                return

            files = []
            for note in update_notes:
                file_path = self.vault_path / note
                files.append(file_path)  # 存在しない場合も含める（削除の可能性）

        if not files:
            self.logger.error("更新対象のファイルがありません")
            return

        self.logger.info(f"更新対象: {len(files)}ファイル")

        # 更新実行
        stats = self.update_files(files)

        # 統計情報の表示
        self.logger.info("=== インクリメンタル更新完了 ===")
        self.logger.info(f"更新ファイル数: {stats['updated_files']}")
        self.logger.info(f"新規/更新チャンク: {stats['new_chunks']}")
        self.logger.info(f"削除チャンク: {stats['deleted_chunks']}")
        self.logger.info(f"総チャンク数: {stats['chunks_before']} → {stats['chunks_after']}")
        self.logger.info(f"総実行時間: {stats['total_time']:.2f}秒")


@click.command()
@click.argument('files', nargs=-1)
@click.option('--config-notes', is_flag=True, help='config.yamlのupdate_notesを使用')
def main(files, config_notes):
    """BGE-M3でのインクリメンタル更新

    使用例:

    \b
    # 特定のファイルを更新
    python scripts/update_bgem3.py "path/to/note1.md" "path/to/note2.md"

    \b
    # config.yamlのupdate_notesを使用
    python scripts/update_bgem3.py --config-notes
    """
    config = setup_environment()

    if config_notes:
        updater = BGEM3IncrementalUpdater(config)
    else:
        updater = BGEM3IncrementalUpdater(config, list(files))

    updater.run()


if __name__ == "__main__":
    main()
