#!/usr/bin/env python3
"""
Gemini Embedding APIを使用したObsidian Vaultのインデックス作成スクリプト
並列処理とバッチ処理を最適化
"""
import os
import sys
from pathlib import Path
import hashlib
from datetime import datetime
import time
from tqdm import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# 共通モジュールのパスを追加
sys.path.append(str(Path(__file__).parent))

from common.utils import setup_environment, setup_logging, get_vault_files, format_file_info
from common.chunker import MarkdownChunker
from common.db import VectorDB


class GeminiIndexer:
    """Gemini Embeddingによるインデックス作成クラス（高速版）"""

    def __init__(self, config):
        self.config = config
        self.logger = setup_logging(config)

        # Vault設定
        self.vault_path = Path(os.getenv('VAULT_PATH'))
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {self.vault_path}")

        # GCP設定
        self.project_id = os.getenv('GCP_PROJECT_ID')
        self.region = os.getenv('GCP_REGION', 'us-central1')

        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID is not set in environment")

        # モデル設定
        embed_config = config['embeddings']['gemini']
        self.model_name = embed_config['model_name']
        self.output_dimensionality = embed_config['output_dimensionality']
        self.task_type_document = embed_config['task_type_document']
        self.auto_truncate = embed_config['auto_truncate']

        # チャンカー初期化
        chunk_config = config['chunking']['gemini']
        self.chunker = MarkdownChunker(
            max_chunk_size=chunk_config['max_chunk_size'],
            chunk_overlap=config['chunking']['chunk_overlap'],
            min_chunk_size=config['chunking']['min_chunk_size']
        )

        # 並列処理設定（最適化）
        self.batch_size = chunk_config['batch_size']
        self.rate_limit_delay = chunk_config['rate_limit_delay']
        self.max_concurrent_requests = chunk_config.get('max_concurrent_requests', 5)

        # DB初期化
        self.db = VectorDB(config['database']['persist_directory'])
        self.collection_name = config['database']['collection_gemini']

        # アクセストークンのキャッシュ
        self._access_token = None
        self._token_expiry = 0

    def save_execution_stats(self, db_stats, total_time, file_count, chunk_count, api_time,
                            total_processed_tokens, estimated_original_tokens):
        """実行統計をファイルに保存"""
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        filename_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
        stats_file = Path("results") / f"indexing_stats_gemini_{filename_timestamp}.md"
        stats_file.parent.mkdir(exist_ok=True)

        content = f"""# Gemini Embedding インデックス作成統計

## 実行情報
- 実行日時: {timestamp_str}
- モデル: {self.model_name}
- プロジェクトID: {self.project_id}
- リージョン: {self.region}
- 並列リクエスト数: {self.max_concurrent_requests}

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
- API処理時間: {api_time:.2f}秒 ({api_time/60:.1f}分)
- API処理割合: {api_time/total_time*100:.1f}%
- ファイル処理速度: {file_count/total_time:.2f} files/秒
- チャンク処理速度: {chunk_count/total_time:.2f} chunks/秒

## 設定
- バッチサイズ: {self.batch_size}
- 最大チャンクサイズ: {self.chunker.max_chunk_size} tokens
- チャンクオーバーラップ: {self.chunker.chunk_overlap} tokens
- 最小チャンクサイズ: {self.chunker.min_chunk_size} tokens
- 出力次元数: {self.output_dimensionality}
- 自動トランケート: {self.auto_truncate}

---
"""

        # 新規ファイルとして保存
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.info(f"統計情報を保存しました: {stats_file}")

    def get_access_token(self):
        """アクセストークンを取得（キャッシュ付き）"""
        current_time = time.time()

        # トークンが有効な場合はキャッシュを返す
        if self._access_token and current_time < self._token_expiry:
            return self._access_token

        try:
            result = subprocess.run(
                ['gcloud', 'auth', 'print-access-token'],
                capture_output=True,
                text=True,
                check=True
            )
            self._access_token = result.stdout.strip()
            # トークンの有効期限を50分に設定（実際は60分だが余裕を持って）
            self._token_expiry = current_time + 3000
            return self._access_token
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get access token: {e}")
            raise

    def generate_embedding_api_with_timing(self, text: str, index: int):
        """埋め込み生成（タイミング測定付き）"""
        start_time = time.time()

        endpoint = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model_name}:predict"

        request_body = {
            "instances": [{
                "content": text,
                "task_type": self.task_type_document
            }],
            "parameters": {
                "outputDimensionality": self.output_dimensionality,
                "autoTruncate": self.auto_truncate
            }
        }

        try:
            access_token = self.get_access_token()

            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=utf-8"
                },
                json=request_body,
                timeout=30  # タイムアウト設定
            )

            api_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                embedding = result['predictions'][0]['embeddings']['values']

                # 768次元を使用しているため、正規化が必要
                if self.output_dimensionality != 3072:
                    import numpy as np
                    embedding_np = np.array(embedding)
                    norm = np.linalg.norm(embedding_np)
                    if norm > 0:
                        embedding = (embedding_np / norm).tolist()

                return index, embedding, api_time, None
            else:
                error_msg = f"API error: {response.status_code} - {response.text}"
                return index, None, api_time, error_msg

        except Exception as e:
            api_time = time.time() - start_time
            return index, None, api_time, str(e)

    def generate_embeddings_concurrent(self, texts: list) -> list:
        """並列処理で埋め込みを生成"""
        embeddings = [None] * len(texts)
        total_time = 0
        errors = []

        with ThreadPoolExecutor(max_workers=self.max_concurrent_requests) as executor:
            # 非同期でリクエストを送信
            future_to_index = {
                executor.submit(self.generate_embedding_api_with_timing, text, i): i
                for i, text in enumerate(texts)
            }

            # プログレスバー付きで結果を取得
            with tqdm(total=len(texts), desc="API calls", leave=False) as pbar:
                for future in as_completed(future_to_index):
                    index, embedding, api_time, error = future.result()

                    if embedding is not None:
                        embeddings[index] = embedding
                        total_time += api_time
                        self.logger.debug(f"Doc {index}: {api_time:.2f}s")
                    else:
                        errors.append((index, error))
                        self.logger.error(f"Doc {index} failed: {error}")

                    pbar.update(1)

                    # レート制限対策（必要に応じて）
                    if len(texts) > self.max_concurrent_requests:
                        time.sleep(self.rate_limit_delay)

        # 統計情報
        successful = len([e for e in embeddings if e is not None])
        avg_time = total_time / successful if successful > 0 else 0

        self.logger.info(f"並列処理完了: {successful}/{len(texts)} 成功")
        self.logger.info(f"平均API応答時間: {avg_time:.2f}秒")
        self.logger.info(f"合計時間: {total_time:.2f}秒")

        return embeddings

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

    def run(self):
        """インデックス作成を実行（高速版）"""
        self.logger.info("=== Gemini Embedding 高速インデックス作成開始 ===")
        self.logger.info(f"Project: {self.project_id}")
        self.logger.info(f"Region: {self.region}")
        self.logger.info(f"並列リクエスト数: {self.max_concurrent_requests}")
        start_time = time.time()

        # ファイル処理
        exclude_patterns = self.config['vault'].get('exclude_patterns', [])
        files = get_vault_files(self.vault_path, exclude_patterns)
        self.logger.info(f"対象ファイル数: {len(files)}")

        # コレクション作成
        collection = self.db.get_or_create_collection(
            self.collection_name,
            self.output_dimensionality
        )

        # チャンク生成
        all_chunks = []
        total_processed_tokens = 0

        for file_path in tqdm(files, desc="ファイル処理"):
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)
            if chunks:
                file_tokens = sum(c.token_count for c in chunks)
                total_processed_tokens += file_tokens

        self.logger.info(f"総チャンク数: {len(all_chunks)}")
        self.logger.info(f"総処理トークン数: {total_processed_tokens:,}")

        # オーバーラップを除いた推定元トークン数
        estimated_original_tokens = total_processed_tokens - (len(all_chunks) - len(files)) * self.chunker.chunk_overlap
        self.logger.info(f"推定元トークン数（オーバーラップ除く）: {estimated_original_tokens:,}")

        # バッチ処理で埋め込み生成
        total_api_time = 0
        successful_chunks = 0
        total_batches = (len(all_chunks) + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(0, len(all_chunks), self.batch_size),
                     desc="埋め込み生成",
                     total=total_batches):

            batch_start_time = time.time()
            batch_chunks = all_chunks[i:i + self.batch_size]

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

            # 並列処理で埋め込み生成
            embeddings = self.generate_embeddings_concurrent(texts)

            # 成功したものだけDBに追加
            valid_indices = [j for j, emb in enumerate(embeddings) if emb is not None]

            if valid_indices:
                valid_embeddings = [embeddings[j] for j in valid_indices]
                valid_texts = [texts[j] for j in valid_indices]
                valid_metadatas = [metadatas[j] for j in valid_indices]

                # ID生成
                ids = []
                seen_ids = set()
                for j in valid_indices:
                    chunk = batch_chunks[j]
                    id_string = f"{chunk.metadata['relative_path']}_{chunk.metadata['chunk_index']}"
                    chunk_id = hashlib.md5(id_string.encode()).hexdigest()
                    if chunk_id in seen_ids:
                        self.logger.warning(f"重複ID検出: {chunk_id} for {id_string}")
                    seen_ids.add(chunk_id)
                    ids.append(chunk_id)

                # DBに追加（upsertで重複を回避）
                try:
                    collection.upsert(
                        embeddings=valid_embeddings,
                        documents=valid_texts,
                        metadatas=valid_metadatas,
                        ids=ids
                    )
                    self.logger.info(f"{len(ids)}件のドキュメントを追加/更新しました")
                except Exception as e:
                    self.logger.error(f"DB追加エラー: {e}")
                    # 個別に処理を試みる
                    for idx, doc_id in enumerate(ids):
                        try:
                            collection.upsert(
                                embeddings=[valid_embeddings[idx]],
                                documents=[valid_texts[idx]],
                                metadatas=[valid_metadatas[idx]],
                                ids=[doc_id]
                            )
                        except Exception as individual_error:
                            self.logger.error(f"個別追加エラー (ID: {doc_id}): {individual_error}")

            batch_time = time.time() - batch_start_time
            total_api_time += batch_time
            if valid_indices:
                successful_chunks += len(valid_indices)
            self.logger.info(f"バッチ処理時間: {batch_time:.2f}秒 ({len(batch_chunks)}件)")

        # 統計情報
        stats = self.db.get_collection_stats(self.collection_name)
        end_time = time.time()
        total_time = end_time - start_time

        self.logger.info("=== インデックス作成完了 ===")
        self.logger.info(f"コレクション: {stats['collection_name']}")
        self.logger.info(f"ドキュメント数: {stats['document_count']}")
        self.logger.info(f"次元数: {stats['dimension']}")
        self.logger.info(f"総実行時間: {total_time:.2f}秒")

        # 実行結果をファイルに保存
        self.save_execution_stats(stats, total_time, len(files), len(all_chunks), total_api_time,
                                 total_processed_tokens, estimated_original_tokens)


def main():
    """メイン関数"""
    config = setup_environment()
    indexer = GeminiIndexer(config)
    indexer.run()


if __name__ == "__main__":
    main()
