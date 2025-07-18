#!/usr/bin/env python3
"""
Gemini Embeddingを使用したObsidian Vaultのインクリメンタル更新スクリプト
特定のノートのみを更新し、古いチャンクは自動的に削除
"""
import os
import sys
from pathlib import Path
import hashlib
from datetime import datetime
import time
from tqdm import tqdm
import click
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# 共通モジュールのパスを追加
sys.path.append(str(Path(__file__).parent))

from common.utils import setup_environment, setup_logging, format_file_info
from common.chunker import MarkdownChunker
from common.db import VectorDB


class GeminiIncrementalUpdater:
    """Gemini Embeddingによるインクリメンタル更新クラス"""

    def __init__(self, config, target_files=None):
        self.config = config
        self.logger = setup_logging(config, 'update_gemini')
        self.target_files = target_files  # 更新対象ファイルのリスト

        # Vault設定
        self.vault_path = Path(os.getenv('VAULT_PATH'))
        if not self.vault_path.exists():
            raise ValueError(f"Vault path does not exist: {self.vault_path}")

        # GCP設定
        self.project_id = os.getenv('GCP_PROJECT_ID')
        self.region = os.getenv('GCP_REGION', 'us-central1')

        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID not set in .env")

        # Gemini設定
        embed_config = config['embeddings']['gemini']
        self.model_name = embed_config['model_name']
        self.output_dimensionality = embed_config['output_dimensionality']
        self.task_type = embed_config['task_type_document']
        self.auto_truncate = embed_config.get('auto_truncate', True)

        # API並列度
        self.max_concurrent_requests = 5

        self.logger.info(f"Project: {self.project_id}")
        self.logger.info(f"Model: {self.model_name}")

        # チャンカー初期化
        chunk_config = config['chunking']['gemini']
        self.chunker = MarkdownChunker(
            max_chunk_size=chunk_config['max_chunk_size'],
            chunk_overlap=config['chunking']['chunk_overlap'],
            min_chunk_size=config['chunking']['min_chunk_size']
        )

        # バッチサイズ設定
        self.batch_size = chunk_config['batch_size']
        self.rate_limit_delay = chunk_config.get('rate_limit_delay', 0.1)

        # DB初期化
        self.db = VectorDB(config['database']['persist_directory'])
        self.collection_name = config['database']['collection_gemini']

        # アクセストークンキャッシュ
        self._access_token = None
        self._token_expiry = None

    # インクリメンタル更新の統計はbenchmark.pyで計測するため、個別ファイルとして保存しない
    # def save_execution_stats(self, stats):
    #     """実行統計をファイルに保存"""
    #     timestamp = datetime.now()
    #     timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    #     filename_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
    #     stats_file = Path("results") / f"incremental_update_gemini_{filename_timestamp}.md"
    #     stats_file.parent.mkdir(exist_ok=True)
    #
    #     content = f"""# Gemini Embedding インクリメンタル更新統計
    #
    # ## 実行情報
    # - 実行日時: {timestamp_str}
    # - モデル: {self.model_name}
    # - プロジェクトID: {self.project_id}
    # - リージョン: {self.region}
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
    # - API処理時間: {stats['api_time']:.2f}秒 ({stats['api_time']/60:.1f}分)
    # - API処理割合: {stats['api_time']/stats['total_time']*100:.1f}%
    # - 平均処理速度: {stats['new_chunks']/stats['total_time']:.2f} chunks/秒
    #
    # ## 設定
    # - バッチサイズ: {self.batch_size}
    # - 並列リクエスト数: {self.max_concurrent_requests}
    # - 最大チャンクサイズ: {self.chunker.max_chunk_size} tokens
    # - 出力次元数: {self.output_dimensionality}
    #
    # ---
    # """
    #
    #     with open(stats_file, 'w', encoding='utf-8') as f:
    #         f.write(content)
    #
    #     self.logger.info(f"統計情報を保存しました: {stats_file}")

    def get_access_token(self):
        """アクセストークンを取得（キャッシュ付き）"""
        import subprocess

        # キャッシュが有効な場合はそれを使用
        if self._access_token and self._token_expiry and time.time() < self._token_expiry:
            return self._access_token

        try:
            result = subprocess.run(
                ['gcloud', 'auth', 'print-access-token'],
                capture_output=True,
                text=True,
                check=True
            )
            self._access_token = result.stdout.strip()
            # トークンの有効期限を50分後に設定（実際は60分だが余裕を持つ）
            self._token_expiry = time.time() + 3000
            return self._access_token
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get access token: {e}")
            raise

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

    def generate_embedding_api_with_timing(self, text: str, index: int):
        """API呼び出しで埋め込みを生成（エラーハンドリング付き）"""
        start_time = time.time()

        # API endpoint
        endpoint = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model_name}:predict"

        # リクエストボディ
        request_body = {
            "instances": [{
                "content": text,
                "task_type": self.task_type
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
                timeout=30
            )

            api_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                embedding = result['predictions'][0]['embeddings']['values']

                # 768次元を使用している場合、正規化が必要
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

            # 結果を収集
            for future in as_completed(future_to_index):
                index, embedding, api_time, error = future.result()
                total_time += api_time

                if error:
                    errors.append((index, error))
                    self.logger.warning(f"Failed to generate embedding for index {index}: {error}")
                else:
                    embeddings[index] = embedding

                # レート制限対策
                time.sleep(self.rate_limit_delay)

        return embeddings, total_time, errors

    def update_files(self, files: list):
        """指定されたファイルを更新"""
        collection = self.db.client.get_collection(self.collection_name)

        # 統計情報の初期化
        stats = {
            'updated_files': len(files),
            'target_files': [str(f.relative_to(self.vault_path)) if f.exists() else str(f) for f in files],
            'new_chunks': 0,
            'deleted_chunks': 0,
            'chunks_before': collection.count(),
            'api_time': 0,
            'total_time': 0
        }

        start_time = time.time()

        # 各ファイルを処理
        for file_path in tqdm(files, desc="ファイル更新"):
            if file_path.exists():
                relative_path = str(file_path.relative_to(self.vault_path))
            else:
                # ファイルが存在しない場合（削除された場合）
                relative_path = str(file_path).replace(str(self.vault_path) + '/', '')

            # 1. 既存のチャンクを削除
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
                    # バッチ処理
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

                        # 並列処理で埋め込み生成
                        embeddings, api_time, errors = self.generate_embeddings_concurrent(texts)
                        stats['api_time'] += api_time

                        # エラーがなかったものだけをDBに保存
                        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]

                        if valid_indices:
                            valid_embeddings = [embeddings[i] for i in valid_indices]
                            valid_texts = [texts[i] for i in valid_indices]
                            valid_metadatas = [metadatas[i] for i in valid_indices]

                            # ID生成
                            ids = []
                            for idx in valid_indices:
                                chunk = batch_chunks[idx]
                                id_string = f"{chunk.metadata['relative_path']}_{chunk.metadata['chunk_index']}"
                                chunk_id = hashlib.md5(id_string.encode()).hexdigest()
                                ids.append(chunk_id)

                            # DBに保存
                            collection.upsert(
                                embeddings=valid_embeddings,
                                documents=valid_texts,
                                metadatas=valid_metadatas,
                                ids=ids
                            )

                            stats['new_chunks'] += len(valid_indices)

                self.logger.info(f"更新完了: {relative_path} ({deleted_count}削除, {len(chunks)}追加)")
            else:
                # ファイルが削除された場合
                self.logger.info(f"ファイル削除を検出: {relative_path} ({deleted_count}チャンク削除)")

        # 最終統計
        stats['chunks_after'] = collection.count()
        stats['total_time'] = time.time() - start_time

        return stats

    def run(self):
        """インクリメンタル更新を実行"""
        self.logger.info("=== Gemini Embedding インクリメンタル更新開始 ===")

        # 更新対象ファイルの決定
        if self.target_files:
            # 指定されたファイルのみ
            files = []
            for file_pattern in self.target_files:
                file_path = self.vault_path / file_pattern
                files.append(file_path)  # 存在しない場合も含める
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
                files.append(file_path)  # 存在しない場合も含める

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
        self.logger.info(f"API時間: {stats['api_time']:.2f}秒")


@click.command()
@click.argument('files', nargs=-1)
@click.option('--config-notes', is_flag=True, help='config.yamlのupdate_notesを使用')
def main(files, config_notes):
    """Gemini Embeddingでのインクリメンタル更新

    使用例:

    \b
    # 特定のファイルを更新
    python scripts/update_gemini.py "path/to/note1.md" "path/to/note2.md"

    \b
    # config.yamlのupdate_notesを使用
    python scripts/update_gemini.py --config-notes
    """
    config = setup_environment()

    if config_notes:
        updater = GeminiIncrementalUpdater(config)
    else:
        updater = GeminiIncrementalUpdater(config, list(files))

    updater.run()


if __name__ == "__main__":
    main()
