#!/usr/bin/env python3
"""
RAG質問応答スクリプト
埋め込み検索結果を使って、Ollama/Geminiモデルで質問に答える
"""
import os
import sys
from pathlib import Path
import click
import subprocess
import requests
import json
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

sys.path.append(str(Path(__file__).parent))

from common.utils import setup_environment, setup_logging
from common.db import VectorDB

from FlagEmbedding import BGEM3FlagModel

class RAGQuerySystem:
    """RAGベースの質問応答システム"""

    def __init__(self, config, embedding_model='bgem3', query_model=None):
        self.config = config
        self.logger = setup_logging(config, 'query')
        self.console = Console()

        self.db = VectorDB(config['database']['persist_directory'])

        self.embedding_model = embedding_model
        self.available_models = []

        if embedding_model in ['bgem3', 'both']:
            self.setup_bgem3()
            self.available_models.append('bgem3')

        if embedding_model in ['gemini', 'both']:
            self.setup_gemini()
            self.available_models.append('gemini')

        if not self.available_models:
            raise ValueError(f"Unsupported embedding model: {embedding_model}")

        self.query_model = query_model
        if query_model and query_model.startswith('gemini'):
            self.query_backend = 'gemini'
            self.setup_gcp_config()
        else:
            self.query_backend = 'ollama'

    def setup_bgem3(self):
        """BGE-M3の設定"""
        embed_config = self.config['embeddings']['bgem3']
        self.bgem3_model_name = embed_config['model_name']

        # インデックス作成時と同じ設定を使用
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.bgem3_model = BGEM3FlagModel(
            self.bgem3_model_name,
            use_fp16=embed_config.get('use_fp16', True) and device in ["cuda", "mps"],
            device=device,
            normalize_embeddings=embed_config.get('normalize', True)
        )
        self.bgem3_collection = self.config['database']['collection_bgem3']

    def setup_gcp_config(self):
        """GCP設定の初期化"""
        self.project_id = os.getenv('GCP_PROJECT_ID')
        self.region = os.getenv('GCP_REGION', 'us-central1')

        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID not set")

    def setup_gemini(self):
        """Geminiの設定"""
        self.setup_gcp_config()

        embed_config = self.config['embeddings']['gemini']
        self.gemini_model_name = embed_config['model_name']
        self.gemini_output_dim = embed_config['output_dimensionality']
        self.gemini_task_query = embed_config['task_type_query']
        self.gemini_collection = self.config['database']['collection_gemini']

    def generate_query_embedding_bgem3(self, query: str) -> list:
        """BGE-M3でクエリの埋め込みを生成"""
        # インデックス作成時と同じパラメータを使用
        embeddings = self.bgem3_model.encode(
            [query],
            batch_size=1,
            max_length=8192,
            convert_to_numpy=True
        )

        dense_vec = embeddings['dense_vecs'][0]

        # convert_to_numpy=Trueのため、常にNumPy配列
        return dense_vec.tolist()

    def generate_query_embedding_gemini(self, query: str) -> list:
        """Geminiでクエリの埋め込みを生成"""
        try:
            result = subprocess.run(
                ['gcloud', 'auth', 'print-access-token'],
                capture_output=True,
                text=True,
                check=True
            )
            access_token = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to get access token: {e}")
            raise

        endpoint = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.gemini_model_name}:predict"

        request_body = {
            "instances": [{
                "content": query,
                "task_type": self.gemini_task_query
            }],
            "parameters": {
                "outputDimensionality": self.gemini_output_dim,
                "autoTruncate": True
            }
        }

        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=utf-8"
            },
            json=request_body
        )

        if response.status_code == 200:
            result = response.json()
            embedding = result['predictions'][0]['embeddings']['values']

            # 768次元を使用している場合は正規化が必要
            if self.gemini_output_dim != 3072:
                import numpy as np
                embedding_np = np.array(embedding)
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    embedding = (embedding_np / norm).tolist()

            return embedding
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")

    def search_bgem3(self, query: str, top_k: int = 5):
        """BGE-M3で関連チャンクを検索"""
        try:
            collection = self.db.client.get_collection(self.bgem3_collection)
            query_embedding = self.generate_query_embedding_bgem3(query)
            results = self.db.search(collection, query_embedding, n_results=top_k)
            return results
        except Exception as e:
            self.logger.error(f"BGE-M3 search error: {e}")
            return []

    def search_gemini(self, query: str, top_k: int = 5):
        """Geminiで関連チャンクを検索"""
        try:
            collection = self.db.client.get_collection(self.gemini_collection)
            query_embedding = self.generate_query_embedding_gemini(query)
            results = self.db.search(collection, query_embedding, n_results=top_k)
            return results
        except Exception as e:
            self.logger.error(f"Gemini search error: {e}")
            return []

    def create_context(self, search_results, model_name=None):
        """検索結果からコンテキストを作成"""
        context_parts = []

        for i, result in enumerate(search_results):
            metadata = result['metadata']
            content = result['document']
            score = result['similarity']

            file_path = metadata.get('relative_path', 'Unknown')
            heading = metadata.get('heading', '')

            context_part = f"""### 参照 {i+1} (関連度: {score:.3f}){f' - {model_name}' if model_name else ''}
**ファイル**: {file_path}
**セクション**: {heading}

{content}
"""
            context_parts.append(context_part)

        return "\n---\n".join(context_parts)

    def query_ollama(self, prompt: str, model: str):
        """Ollamaモデルに問い合わせ"""
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                raise Exception("Ollama is not running. Please start Ollama first.")

            available_models = [m['name'] for m in response.json()['models']]
            if model not in available_models:
                raise Exception(f"Model '{model}' not found.")

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:
                return response.json()['response']
            else:
                raise Exception(f"Ollama error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Please run 'ollama serve' first.")

    def query_gemini(self, prompt: str, model: str):
        """Gemini APIに問い合わせ"""
        try:
            result = subprocess.run(
                ['gcloud', 'auth', 'print-access-token'],
                capture_output=True,
                text=True,
                check=True
            )
            access_token = result.stdout.strip()

            endpoint = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{model}:generateContent"
            self.logger.info(f"Gemini query endpoint: {endpoint}")

            request_body = {
                "contents": [{
                    "role": "user",
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "maxOutputTokens": 2048
                }
            }

            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=utf-8"
                },
                json=request_body
            )

            if response.status_code == 200:
                result = response.json()
                self.logger.debug(f"Gemini response: {json.dumps(result, ensure_ascii=False, indent=2)}")

                try:
                    if 'candidates' in result and result['candidates']:
                        candidate = result['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            parts = candidate['content']['parts']
                            if parts and isinstance(parts, list) and 'text' in parts[0]:
                                return parts[0]['text']

                    self.logger.error(f"Unexpected Gemini response structure: {json.dumps(result, ensure_ascii=False)}")
                    raise Exception("Gemini API returned unexpected response structure")

                except (KeyError, IndexError, TypeError) as e:
                    self.logger.error(f"Error parsing Gemini response: {e}")
                    self.logger.error(f"Response: {json.dumps(result, ensure_ascii=False)}")
                    raise Exception(f"Failed to parse Gemini response: {e}")
            else:
                raise Exception(f"Gemini API error: {response.status_code} - {response.text}")

        except Exception as e:
            self.logger.error(f"Gemini query error: {e}")
            raise

    def generate_answer(self, query: str, context: str):
        """コンテキストを使って回答を生成"""
        prompt = f"""以下の参照情報を使って、ユーザーの質問に答えてください。

## 参照情報:
{context}

## 質問:
{query}

## 回答:
参照情報に基づいて、具体的に回答してください。情報が不足している場合は、その旨を明記してください。
"""

        if self.query_backend == 'ollama':
            return self.query_ollama(prompt, self.query_model)
        else:
            return self.query_gemini(prompt, self.query_model)

    def compare_results(self, bgem3_results: list, gemini_results: list):
        """両モデルの結果を比較"""
        bgem3_files = set(r['metadata']['relative_path'] for r in bgem3_results)
        gemini_files = set(r['metadata']['relative_path'] for r in gemini_results)

        common_files = bgem3_files & gemini_files
        bgem3_only = bgem3_files - gemini_files
        gemini_only = gemini_files - bgem3_files

        self.console.print("\n[bold]Comparison Results:[/bold]")

        stats_table = Table(title="Result Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("BGE-M3", style="green")
        stats_table.add_column("Gemini", style="blue")

        stats_table.add_row("Total Results", str(len(bgem3_results)), str(len(gemini_results)))
        stats_table.add_row("Unique Files", str(len(bgem3_files)), str(len(gemini_files)))
        stats_table.add_row("Common Files", str(len(common_files)), str(len(common_files)))

        if bgem3_results and gemini_results:
            bgem3_avg_score = sum(r['similarity'] for r in bgem3_results) / len(bgem3_results)
            gemini_avg_score = sum(r['similarity'] for r in gemini_results) / len(gemini_results)
            stats_table.add_row("Avg Score", f"{bgem3_avg_score:.3f}", f"{gemini_avg_score:.3f}")

        self.console.print(stats_table)

        if bgem3_only:
            self.console.print("\n[green]Files found only by BGE-M3:[/green]")
            for f in bgem3_only:
                self.console.print(f"  - {f}")

        if gemini_only:
            self.console.print("\n[blue]Files found only by Gemini:[/blue]")
            for f in gemini_only:
                self.console.print(f"  - {f}")

    def generate_query_embedding(self, query: str):
        """指定された埋め込みモデルでクエリ埋め込みを生成"""
        if self.embedding_model == 'bgem3':
            return self.generate_query_embedding_bgem3(query)
        elif self.embedding_model == 'gemini':
            return self.generate_query_embedding_gemini(query)
        else:
            raise ValueError(f"Unsupported embedding model for single generation: {self.embedding_model}")

    def search_relevant_chunks(self, query: str, top_k: int = 5):
        """指定された埋め込みモデルで関連チャンクを検索"""
        if self.embedding_model == 'bgem3':
            return self.search_bgem3(query, top_k)
        elif self.embedding_model == 'gemini':
            return self.search_gemini(query, top_k)
        else:
            # bothの場合は両方の結果を統合
            all_results = []
            if 'bgem3' in self.available_models:
                bgem3_results = self.search_bgem3(query, top_k)
                for r in bgem3_results:
                    r['_model'] = 'BGE-M3'
                all_results.extend(bgem3_results)
            if 'gemini' in self.available_models:
                gemini_results = self.search_gemini(query, top_k)
                for r in gemini_results:
                    r['_model'] = 'Gemini'
                all_results.extend(gemini_results)
            # スコアでソートして上位k件を返す
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            return all_results[:top_k]

    def run_query(self, query: str, top_k: int = 5, compare: bool = False, verbose_results: bool = False):
        """質問応答を実行"""
        self.console.print(f"\n[bold]質問:[/bold] {query}")

        if self.embedding_model == 'bgem3':
            self.console.print(f"[dim]埋め込みモデル: BGE-M3 ({self.bgem3_model_name})[/dim]")
        elif self.embedding_model == 'gemini':
            self.console.print(f"[dim]埋め込みモデル: Gemini ({self.gemini_model_name})[/dim]")
        else:  # both
            bgem3_name = self.bgem3_model_name if hasattr(self, 'bgem3_model_name') else 'BGE-M3'
            gemini_name = self.gemini_model_name if hasattr(self, 'gemini_model_name') else 'Gemini'
            self.console.print(f"[dim]埋め込みモデル: Both (BGE-M3: {bgem3_name}, Gemini: {gemini_name})[/dim]")

        self.console.print(f"[dim]回答モデル: {self.query_model if self.query_model else 'なし（検索のみ）'}[/dim]\n")

        results_dict = {}
        all_results = []

        if 'bgem3' in self.available_models:
            with self.console.status("[cyan]BGE-M3で関連情報を検索中...[/cyan]"):
                bgem3_results = self.search_bgem3(query, top_k)
                results_dict['bgem3'] = bgem3_results

            if bgem3_results:
                self.console.print(f"[green]✓ BGE-M3: {len(bgem3_results)}件の関連情報を発見[/green]")
                if self.embedding_model == 'both':
                    for r in bgem3_results:
                        r['_model'] = 'BGE-M3'
                all_results.extend(bgem3_results)

        if 'gemini' in self.available_models:
            with self.console.status("[cyan]Geminiで関連情報を検索中...[/cyan]"):
                gemini_results = self.search_gemini(query, top_k)
                results_dict['gemini'] = gemini_results

            if gemini_results:
                self.console.print(f"[green]✓ Gemini: {len(gemini_results)}件の関連情報を発見[/green]")
                if self.embedding_model == 'both':
                    for r in gemini_results:
                        r['_model'] = 'Gemini'
                all_results.extend(gemini_results)

        if not all_results:
            self.console.print("[red]関連する情報が見つかりませんでした。[/red]")
            return

        if compare and len(results_dict) == 2:
            self.compare_results(
                results_dict.get('bgem3', []),
                results_dict.get('gemini', [])
            )

        # bothモードの場合は各モデルで別々に回答を生成
        if self.embedding_model == 'both' and self.query_model:
            # BGE-M3の結果で回答生成
            if 'bgem3' in results_dict and results_dict['bgem3']:
                bgem3_context = self.create_context(results_dict['bgem3'])

                if verbose_results:
                    self.console.print("\n")
                    self.console.print(Panel(
                        Markdown(bgem3_context),
                        title="BGE-M3 検索結果",
                        border_style="dim"
                    ))

                self.console.print("\n[cyan]BGE-M3の検索結果を使って回答を生成中...[/cyan]")
                try:
                    bgem3_answer = self.generate_answer(query, bgem3_context)
                    self.console.print("\n")
                    self.console.print(Panel(
                        Markdown(bgem3_answer),
                        title="回答 (BGE-M3による検索)",
                        border_style="green"
                    ))
                except Exception as e:
                    self.console.print(f"\n[red]BGE-M3回答生成エラー: {e}[/red]")

            # Geminiの結果で回答生成
            if 'gemini' in results_dict and results_dict['gemini']:
                gemini_context = self.create_context(results_dict['gemini'])

                if verbose_results:
                    self.console.print("\n")
                    self.console.print(Panel(
                        Markdown(gemini_context),
                        title="Gemini 検索結果",
                        border_style="dim"
                    ))

                self.console.print("\n[cyan]Geminiの検索結果を使って回答を生成中...[/cyan]")
                try:
                    gemini_answer = self.generate_answer(query, gemini_context)
                    self.console.print("\n")
                    self.console.print(Panel(
                        Markdown(gemini_answer),
                        title="回答 (Geminiによる検索)",
                        border_style="blue"
                    ))
                except Exception as e:
                    self.console.print(f"\n[red]Gemini回答生成エラー: {e}[/red]")

        # 単一モデルの場合の処理
        else:
            if self.embedding_model == 'both':
                # bothで検索のみの場合は統合結果を使用
                all_results.sort(key=lambda x: x['similarity'], reverse=True)
                all_results = all_results[:top_k]
                context = self.create_context(all_results)
            else:
                context = self.create_context(all_results)

            if verbose_results:
                self.console.print("\n")
                if self.embedding_model == 'both':
                    for r in all_results:
                        model_label = r.get('_model', '')
                        r_copy = r.copy()
                        if '_model' in r_copy:
                            del r_copy['_model']
                        context_part = self.create_context([r_copy], model_label)
                        self.console.print(Panel(
                            Markdown(context_part),
                            title=f"検索結果 - {model_label}",
                            border_style="dim"
                        ))
                else:
                    self.console.print(Panel(
                        Markdown(context),
                        title="検索結果",
                        border_style="dim"
                    ))

            if self.query_model:
                self.console.print("\n[cyan]回答を生成中...[/cyan]")
                try:
                    answer = self.generate_answer(query, context)
                    self.console.print("\n")
                    self.console.print(Panel(
                        Markdown(answer),
                        title="回答",
                        border_style="green"
                    ))
                except Exception as e:
                    self.console.print(f"\n[red]回答生成エラー: {e}[/red]")

        # 参照ファイルの表示
        if self.embedding_model == 'both' and len(results_dict) > 1:
            # bothモードの場合は各モデルの結果を別々に表示
            for model_name, results in results_dict.items():
                if results:
                    model_display_name = "BGE-M3" if model_name == 'bgem3' else "Gemini"
                    self.console.print(f"\n[bold]参照ファイル ({model_display_name}):[/bold]")
                    seen_files = set()
                    for i, result in enumerate(results[:5]):
                        file_path = result['metadata'].get('relative_path', 'Unknown')
                        if file_path not in seen_files:
                            seen_files.add(file_path)
                            score = result.get('similarity', 0)
                            self.console.print(f"  {i+1}. {file_path} (関連度: {score:.3f})")

                    if len(results) > 5:
                        self.console.print(f"  ... その他 {len(results) - 5} 件")
        else:
            # 単一モデルまたは統合結果の場合
            self.console.print("\n[bold]参照ファイル:[/bold]")
            seen_files = set()
            for i, result in enumerate(all_results[:5]):
                file_path = result['metadata'].get('relative_path', 'Unknown')
                if file_path not in seen_files:
                    seen_files.add(file_path)
                    model_label = result.get('_model', '')
                    score = result.get('similarity', 0)
                    self.console.print(f"  {i+1}. {file_path} (関連度: {score:.3f}){f' - {model_label}' if model_label else ''}")

            if len(all_results) > 5:
                self.console.print(f"  ... その他 {len(all_results) - 5} 件")


@click.command()
@click.argument('query', required=False)
@click.option('--embedding-model', '-e', type=click.Choice(['bgem3', 'gemini', 'both']), help='埋め込みモデル（bgem3: ローカル, gemini: Vertex AI, both: 両方）')
@click.option('--query-model', '-q', help='回答生成モデル（例: llama3.2, gemma2, mistral, gemini-2.5-flash）')
@click.option('--top-k', '-k', default=5, help='検索する上位結果数')
@click.option('--compare', '-c', is_flag=True, help='両モデルの結果を比較（-e both時のみ有効）')
@click.option('--verbose-results', '-v', is_flag=True, help='検索結果の詳細を表示')
def main(query, embedding_model, query_model, top_k, compare, verbose_results):
    """RAGベースの質問応答システム

    Obsidian Vaultの埋め込みデータベースを検索し、
    OllamaまたはGemini APIを使って質問に回答します。

    使用例:

    \b
    # デフォルト（Geminiで検索・回答）
    python scripts/query.py "JETLSの実装状況は？"

    \b
    # BGE-M3で検索、Ollamaで回答（完全ローカル）
    python scripts/query.py "JETLSの実装状況は？" -e bgem3 -q llama3.2

    \b
    # 両モデルで検索して比較
    python scripts/query.py "JETLSの実装状況は？" -e both --compare

    \b
    # 検索結果の詳細を表示
    python scripts/query.py "JETLSの実装状況は？" -v

    \b
    # 検索のみ（回答生成なし）
    python scripts/query.py "JETLSの実装状況は？" -q ""

    \b
    # カスタム設定
    python scripts/query.py "JETLSの実装状況は？" -e bgem3 -q gemini-pro -k 10

    \b
    # config.yamlのデフォルトクエリを使用
    python scripts/query.py
    """
    config = setup_environment()
    console = Console()

    if not embedding_model:
        embedding_model = config.get('defaults', {}).get('embedding_model', 'gemini')
    if query_model is None:
        query_model = config.get('defaults', {}).get('query_model', 'gemini-2.5-flash')

    queries = []
    if query:
        queries = [query]
    else:
        # 引数がない場合はconfigから読み込み
        queries = config.get('defaults', {}).get('queries', [])
        if not queries:
            console.print("[red]エラー: クエリを指定するか、config.yamlのdefaults.queriesにクエリを設定してください[/red]")
            return

    try:
        rag_system = RAGQuerySystem(config, embedding_model, query_model)
        for i, query in enumerate(queries):
            if i > 0:
                console.print("\n" + "="*80 + "\n")
            rag_system.run_query(query, top_k, compare, verbose_results)
    except Exception as e:
        console.print(f"[red]エラー: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
