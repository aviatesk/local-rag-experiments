#!/usr/bin/env python3
"""
両モデルで検索・比較を行うスクリプト
"""
import os
import sys
from pathlib import Path
import click
from datetime import datetime
import json
from rich.console import Console
from rich.table import Table

sys.path.append(str(Path(__file__).parent))

from common.utils import setup_environment, setup_logging
from common.db import VectorDB

from FlagEmbedding import BGEM3FlagModel
import subprocess


class EmbeddingSearcher:
    """埋め込み検索クラス"""

    def __init__(self, config, model_type='both'):
        self.config = config
        self.logger = setup_logging(config)
        self.console = Console()

        self.db = VectorDB(config['database']['persist_directory'])

        self.available_models = []

        if model_type in ['bgem3', 'both']:
            self.setup_bgem3()
            self.available_models.append('bgem3')

        if model_type in ['gemini', 'both']:
            self.setup_gemini()
            self.available_models.append('gemini')

        if not self.available_models:
            raise ValueError(f"No models available for type: {model_type}")

    def setup_bgem3(self):
        """BGE-M3の設定"""
        embed_config = self.config['embeddings']['bgem3']

        # インデックス作成時と同じ設定を使用
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self.bgem3_model = BGEM3FlagModel(
            embed_config['model_name'],
            use_fp16=embed_config.get('use_fp16', True) and device in ["cuda", "mps"],
            device=device,
            normalize_embeddings=embed_config.get('normalize', True)
        )
        self.bgem3_collection = self.config['database']['collection_bgem3']

    def setup_gemini(self):
        """Geminiの設定"""
        self.project_id = os.getenv('GCP_PROJECT_ID')
        self.region = os.getenv('GCP_REGION', 'us-central1')

        if not self.project_id:
            self.logger.warning("GCP_PROJECT_ID not set, Gemini search disabled")
            return

        embed_config = self.config['embeddings']['gemini']
        self.gemini_model_name = embed_config['model_name']
        self.gemini_output_dim = embed_config['output_dimensionality']
        self.gemini_task_query = embed_config['task_type_query']
        self.gemini_collection = self.config['database']['collection_gemini']

    def generate_query_embedding_bgem3(self, query: str) -> list:
        """BGE-M3でクエリ埋め込みを生成"""
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
        """Geminiでクエリ埋め込みを生成"""
        import requests

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

        # クエリ用タスクタイプを使用
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

            # 768次元を使用しているため、正規化が必要
            if self.gemini_output_dim != 3072:
                import numpy as np
                embedding_np = np.array(embedding)
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    embedding = (embedding_np / norm).tolist()

            return embedding
        else:
            raise Exception(f"API error: {response.status_code} - {response.text}")

    def search_bgem3(self, query: str, top_k: int = 5) -> list:
        """BGE-M3で検索"""
        try:
            collection = self.db.client.get_collection(self.bgem3_collection)
            query_embedding = self.generate_query_embedding_bgem3(query)
            results = self.db.search(collection, query_embedding, n_results=top_k)
            return results
        except Exception as e:
            self.logger.error(f"BGE-M3 search error: {e}")
            return []

    def search_gemini(self, query: str, top_k: int = 5) -> list:
        """Geminiで検索"""
        try:
            collection = self.db.client.get_collection(self.gemini_collection)
            query_embedding = self.generate_query_embedding_gemini(query)
            results = self.db.search(collection, query_embedding, n_results=top_k)
            return results
        except Exception as e:
            self.logger.error(f"Gemini search error: {e}")
            return []

    def display_results(self, results: list, model_name: str):
        """検索結果を表示"""
        if not results:
            self.console.print(f"[yellow]No results found for {model_name}[/yellow]")
            return

        table = Table(title=f"{model_name} Search Results")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Score", style="magenta", width=8)
        table.add_column("File", style="green", width=30)
        table.add_column("Heading", style="blue", width=30)
        table.add_column("Preview", style="white", width=50)

        for i, result in enumerate(results):
            score = f"{result['similarity']:.3f}"
            file_path = result['metadata'].get('relative_path', 'Unknown')
            heading = result['metadata'].get('heading', '')
            preview = result['document'][:100] + "..." if len(result['document']) > 100 else result['document']
            preview = preview.replace('\n', ' ')

            table.add_row(
                str(i + 1),
                score,
                file_path,
                heading,
                preview
            )

        self.console.print(table)

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

    def save_results(self, query: str, results_dict: dict):
        """検索結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"search_results_{timestamp}.json"
        filepath = Path("results") / filename

        filepath.parent.mkdir(exist_ok=True)

        output = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "search": self.config.get('search', {})
            },
            "results": results_dict
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self.console.print(f"\n[green]Results saved to: {filepath}[/green]")


@click.command()
@click.argument('query', required=False)
@click.option('--embedding-model', '-e', type=click.Choice(['bgem3', 'gemini', 'both']), help='埋め込みモデル（bgem3: ローカル, gemini: Vertex AI, both: 両方）')
@click.option('--top-k', '-k', default=5, help='取得する上位結果数')
@click.option('--compare', '-c', is_flag=True, help='両モデルの結果を比較')
@click.option('--save', '-s', is_flag=True, help='結果をJSONファイルに保存')
def main(query, embedding_model, top_k, compare, save):
    """埋め込みベクトル検索を実行

    BGE-M3（ローカル）とGemini（Vertex AI）の
    埋め込みモデルを使ってObsidian Vaultを検索します。

    使用例:

    \b
    # Geminiで検索（デフォルト）
    python scripts/search.py "RAGデータベース"

    \b
    # BGE-M3のみで検索
    python scripts/search.py "RAGデータベース" -e bgem3

    \b
    # 両モデルで結果を比較して保存
    python scripts/search.py "RAGデータベース" -e both --compare --save

    \b
    # config.yamlのデフォルトクエリを使用
    python scripts/search.py
    """
    config = setup_environment()
    console = Console()

    # デフォルト埋め込みモデルの取得（指定されていない場合のみconfigから読む）
    if embedding_model is None:
        embedding_model = config.get('defaults', {}).get('embedding_model', 'gemini')

    # クエリリストの作成
    queries = []
    if query:
        queries = [query]
    else:
        # 引数がない場合はconfigから読み込み
        queries = config.get('defaults', {}).get('queries', [])
        if not queries:
            console.print("[red]エラー: クエリを指定するか、config.yamlのdefaults.queriesにクエリを設定してください[/red]")
            return

    searcher = EmbeddingSearcher(config, embedding_model)
    console = Console()

    for query in queries:
        console.print(f"\n[bold]Query:[/bold] {query}")
        console.print(f"[bold]Embedding Model:[/bold] {embedding_model}")
        console.print(f"[bold]Top-K:[/bold] {top_k}\n")

        results_dict = {}

        if 'bgem3' in searcher.available_models:
            console.print("[cyan]Searching with BGE-M3...[/cyan]")
            bgem3_results = searcher.search_bgem3(query, top_k)
            searcher.display_results(bgem3_results, "BGE-M3")
            results_dict['bgem3'] = bgem3_results

        if 'gemini' in searcher.available_models:
            console.print("\n[cyan]Searching with Gemini...[/cyan]")
            gemini_results = searcher.search_gemini(query, top_k)
            searcher.display_results(gemini_results, "Gemini")
            results_dict['gemini'] = gemini_results

        if compare and len(results_dict) == 2:
            searcher.compare_results(
                results_dict.get('bgem3', []),
                results_dict.get('gemini', [])
            )

        if save:
            searcher.save_results(query, results_dict)

        if len(queries) > 1:
            console.print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
