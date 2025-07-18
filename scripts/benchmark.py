#!/usr/bin/env python3
"""
RAGシステムのベンチマークスクリプト
埋め込み生成と回答生成の時間を計測
"""
import sys
from pathlib import Path
import click
import time
import json
from datetime import datetime
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from statistics import mean, stdev
import logging
import warnings
import os

# ベンチマーク実行時はログを抑制
logging.getLogger("FlagEmbedding").setLevel(logging.WARNING)
logging.getLogger("FlagEmbedding.finetune").setLevel(logging.WARNING)
logging.getLogger("FlagEmbedding.finetune.embedder").setLevel(logging.WARNING)
logging.getLogger("FlagEmbedding.finetune.embedder.encoder_only").setLevel(logging.WARNING)
logging.getLogger("FlagEmbedding.finetune.embedder.encoder_only.m3").setLevel(logging.WARNING)
logging.getLogger("FlagEmbedding.finetune.embedder.encoder_only.m3.runner").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("common.utils").setLevel(logging.WARNING)  # common.utilsのログも抑制

class FilteredOutput:
    """特定の文字列を含む出力を抑制するラッパー"""
    def __init__(self, stream, filters):
        self.stream = stream
        self.filters = filters

    def write(self, text):
        # フィルタに該当する文字列は出力しない
        for filter_str in self.filters:
            if filter_str in text:
                return
        self.stream.write(text)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

output_filters = [
    "pre tokenize:",
    "Inference Embeddings:",
    "Fetching",
    "loading existing colbert_linear"
]

# 標準出力と標準エラー出力をフィルタリング
sys.stdout = FilteredOutput(sys.stdout, output_filters)
sys.stderr = FilteredOutput(sys.stderr, output_filters)

# トークナイザーの警告を抑制
warnings.filterwarnings("ignore", message="You're using a XLMRobertaTokenizerFast tokenizer")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(str(Path(__file__).parent))

from query import RAGQuerySystem
from common.utils import setup_environment
from update_bgem3 import BGEM3IncrementalUpdater
from update_gemini import GeminiIncrementalUpdater


class BenchmarkRunner:
    """ベンチマーク実行クラス"""

    def __init__(self, config):
        self.config = config
        self.console = Console()
        self.results = []

    def benchmark_query(self, query: str, embedding_model: str, query_model: str,
                       top_k: int = 5, runs: int = 3) -> Dict | None:
        """単一クエリのベンチマーク"""
        run_results = []

        for run_idx in range(runs):
            self.console.print(f"\n[dim]Run {run_idx + 1}/{runs}[/dim]")

            # 毎回新規作成でキャッシュの影響を排除
            rag_system = RAGQuerySystem(self.config, embedding_model, query_model)

            embed_start = time.perf_counter()
            try:
                rag_system.generate_query_embedding(query)
                embed_end = time.perf_counter()
                embed_time = embed_end - embed_start
            except Exception as e:
                self.console.print(f"[red]埋め込み生成エラー: {e}[/red]")
                continue

            search_start = time.perf_counter()
            search_results = rag_system.search_relevant_chunks(query, top_k)
            search_end = time.perf_counter()
            search_time = search_end - search_start

            # モデルが指定されている場合の回答生成時間計測
            answer_time = 0
            if query_model:
                answer_start = time.perf_counter()
                try:
                    # コンテキスト作成は計測せずに直接実行
                    context = rag_system.create_context(search_results)
                    rag_system.generate_answer(query, context)
                    answer_end = time.perf_counter()
                    answer_time = answer_end - answer_start
                except Exception as e:
                    self.console.print(f"[red]回答生成エラー: {e}[/red]")
                    answer_time = -1

            run_result = {
                'run': run_idx + 1,
                'embed_time': embed_time,
                'search_time': search_time,
                'answer_time': answer_time,
                'total_time': embed_time + search_time + (answer_time if answer_time > 0 else 0),
                'results_count': len(search_results)
            }
            run_results.append(run_result)

            self.console.print(f"  埋め込み生成: {embed_time:.3f}秒")
            self.console.print(f"  検索: {search_time:.3f}秒")
            if query_model:
                self.console.print(f"  回答生成: {answer_time:.3f}秒")
            self.console.print(f"  [bold]合計: {run_result['total_time']:.3f}秒[/bold]")

        if run_results:
            stats = self.calculate_stats(run_results)
            return {
                'query': query,
                'embedding_model': embedding_model,
                'query_model': query_model or 'なし',
                'top_k': top_k,
                'runs': runs,
                'raw_results': run_results,
                'stats': stats
            }
        return None

    def calculate_stats(self, results: List[Dict]) -> Dict:
        """統計情報を計算"""
        def get_values(key):
            return [r[key] for r in results if r[key] > 0]

        stats = {}
        for key in ['embed_time', 'search_time', 'answer_time', 'total_time']:
            values = get_values(key)
            if values:
                stats[key] = {
                    'mean': mean(values),
                    'min': min(values),
                    'max': max(values),
                    'stdev': stdev(values) if len(values) > 1 else 0
                }
            else:
                stats[key] = None

        return stats

    def benchmark_incremental_update(self, files: List[str], embedding_model: str, runs: int = 3) -> Dict | None:
        """インクリメンタル更新のベンチマーク"""
        run_results = []

        for run_idx in range(runs):
            self.console.print(f"\n[dim]Run {run_idx + 1}/{runs}[/dim]")

            # 更新クラスを初期化
            if embedding_model == 'bgem3':
                updater = BGEM3IncrementalUpdater(self.config, files)
            elif embedding_model == 'gemini':
                updater = GeminiIncrementalUpdater(self.config, files)
            else:
                self.console.print(f"[red]不明な埋め込みモデル: {embedding_model}[/red]")
                continue

            # 更新実行時間を計測
            update_start = time.perf_counter()
            try:
                # update_filesメソッドを直接呼び出し
                file_paths = [updater.vault_path / f for f in files]
                stats = updater.update_files(file_paths)
                update_end = time.perf_counter()
                update_time = update_end - update_start

                run_result = {
                    'run': run_idx + 1,
                    'update_time': update_time,
                    'updated_files': stats['updated_files'],
                    'new_chunks': stats['new_chunks'],
                    'deleted_chunks': stats['deleted_chunks'],
                    'processing_time': stats.get('processing_time', 0),
                    'db_time': stats.get('db_time', 0),
                    'api_time': stats.get('api_time', 0)
                }
                run_results.append(run_result)

                self.console.print(f"  更新時間: {update_time:.3f}秒")
                self.console.print(f"  更新ファイル数: {stats['updated_files']}")
                self.console.print(f"  新規/更新チャンク: {stats['new_chunks']}")
                self.console.print(f"  削除チャンク: {stats['deleted_chunks']}")

            except Exception as e:
                self.console.print(f"[red]更新エラー: {e}[/red]")
                continue

        if run_results:
            stats = self.calculate_update_stats(run_results)
            return {
                'files': files,
                'embedding_model': embedding_model,
                'runs': runs,
                'raw_results': run_results,
                'stats': stats
            }
        return None

    def calculate_update_stats(self, results: List[Dict]) -> Dict:
        """更新統計情報を計算"""
        def get_values(key):
            return [r[key] for r in results if key in r and r[key] >= 0]

        stats = {}
        for key in ['update_time', 'new_chunks', 'deleted_chunks', 'processing_time', 'db_time', 'api_time']:
            values = get_values(key)
            if values:
                stats[key] = {
                    'mean': mean(values),
                    'min': min(values),
                    'max': max(values),
                    'stdev': stdev(values) if len(values) > 1 else 0
                }
            else:
                stats[key] = None

        return stats

    def run_benchmark_suite(self, queries: List[str], configs: List[Dict], runs: int = 3):
        """ベンチマークスイートを実行"""
        total_tests = len(queries) * len(configs)
        test_idx = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("ベンチマーク実行中...", total=total_tests)

            for query in queries:
                for config in configs:
                    test_idx += 1
                    desc = f"[{test_idx}/{total_tests}] {config['name']}: {query[:30]}..."
                    progress.update(task, description=desc)

                    result = self.benchmark_query(
                        query=query,
                        embedding_model=config['embedding_model'],
                        query_model=config.get('query_model'),
                        top_k=config.get('top_k', 5),
                        runs=runs
                    )

                    if result:
                        result['config_name'] = config['name']
                        self.results.append(result)

                    progress.advance(task)

    def display_results(self):
        """結果を表示"""
        if not self.results:
            self.console.print("[red]結果がありません[/red]")
            return

        table = Table(title="ベンチマーク結果（平均時間）")
        table.add_column("設定", style="cyan")
        table.add_column("埋め込み", style="green", justify="right")
        table.add_column("検索", style="green", justify="right")
        table.add_column("コンテキスト", style="green", justify="right")
        table.add_column("回答生成", style="yellow", justify="right")
        table.add_column("合計", style="bold", justify="right")

        for result in self.results:
            stats = result['stats']
            query_short = result['query'][:30] + "..." if len(result['query']) > 30 else result['query']
            config_name = f"{result['config_name']}\n[dim]{query_short}[/dim]"

            def format_time(stat_dict, key):
                if stat_dict and key in stat_dict and stat_dict[key]:
                    return f"{stat_dict[key]['mean']:.3f}s"
                return "N/A"

            table.add_row(
                config_name,
                format_time(stats, 'embed_time'),
                format_time(stats, 'search_time'),
                format_time(stats, 'context_time'),
                format_time(stats, 'answer_time'),
                format_time(stats, 'total_time')
            )

        self.console.print("\n")
        self.console.print(table)

        self.console.print("\n[bold]詳細統計:[/bold]")
        for result in self.results:
            self.console.print(f"\n[cyan]{result['config_name']}[/cyan] - {result['query'][:50]}...")
            stats = result['stats']

            for phase, label in [
                ('embed_time', '埋め込み生成'),
                ('search_time', '検索'),
                ('context_time', 'コンテキスト作成'),
                ('answer_time', '回答生成'),
                ('total_time', '合計')
            ]:
                if stats.get(phase):
                    s = stats[phase]
                    self.console.print(
                        f"  {label}: "
                        f"平均={s['mean']:.3f}s, "
                        f"最小={s['min']:.3f}s, "
                        f"最大={s['max']:.3f}s, "
                        f"標準偏差={s['stdev']:.3f}s"
                    )

    def save_results(self, output_path: str):
        """結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        filepath = Path(output_path) / filename

        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': self.results
            }, f, ensure_ascii=False, indent=2)

        self.console.print(f"\n[green]結果を保存しました: {filepath}[/green]")


@click.command()
@click.option('--queries', '-q', multiple=True, help='ベンチマーク用クエリ（複数指定可）')
@click.option('--query-file', '-f', type=click.Path(exists=True), help='クエリリストファイル（1行1クエリ）')
@click.option('--update-files', '-u', multiple=True, help='インクリメンタル更新ベンチマーク用ファイル（複数指定可）')
@click.option('--runs', '-r', type=int, help='各テストの実行回数')
@click.option('--save', '-s', is_flag=True, help='結果をJSONファイルに保存')
@click.option('--mode', '-m',
              type=click.Choice(['query', 'update', 'both']),
              default='query',
              help='ベンチマークモード')
@click.option('--top-k', '-k', type=int, help='検索する上位結果数')
def main(queries, query_file, update_files, runs, save, mode, top_k):
    """RAGシステムのベンチマーク

    埋め込み生成、検索、回答生成、インクリメンタル更新の処理時間を計測します。

    使用例:

    \b
    # クエリベンチマーク（デフォルト）
    python scripts/benchmark.py -q "JETLSの実装状況は？"

    \b
    # 複数クエリでのベンチマーク
    python scripts/benchmark.py -q "JETLSの実装" -q "型推論の仕組み" -q "最適化手法"

    \b
    # インクリメンタル更新ベンチマーク
    python scripts/benchmark.py -m update -u "notes/JETLS.md" -u "notes/optimization.md"

    \b
    # 両方のベンチマークを実行
    python scripts/benchmark.py -m both -q "テストクエリ" -u "notes/test.md"

    \b
    # config.yamlの設定を使用
    python scripts/benchmark.py

    \b
    # 特定のクエリでベンチマーク
    python scripts/benchmark.py -q "テストクエリ" -r 5 --save
    """
    config = setup_environment()
    runner = BenchmarkRunner(config)
    console = Console()

    # デフォルト値の取得
    benchmark_config = config.get('benchmark', {})
    if runs is None:
        runs = benchmark_config.get('runs', 3)
    if top_k is None:
        top_k = benchmark_config.get('top_k', 5)

    # クエリリストの作成
    query_list = list(queries)
    if query_file:
        with open(query_file, 'r', encoding='utf-8') as f:
            query_list.extend([line.strip() for line in f if line.strip()])

    if not query_list:
        # 引数がない場合はconfigから読み込み
        query_list = config.get('defaults', {}).get('queries', [])
        if not query_list:
            console.print("[red]エラー: クエリを指定するか、config.yamlのdefaults.queriesにクエリを設定してください[/red]")
            return

    # ベンチマーク設定の取得
    bench_configs = benchmark_config.get('configurations', [])
    if not bench_configs:
        # デフォルト設定を使用
        default_embedding = config.get('defaults', {}).get('embedding_model', 'gemini')
        default_query = config.get('defaults', {}).get('query_model', None)
        bench_configs = [
            {'name': f'{default_embedding.upper()}（デフォルト）',
             'embedding_model': default_embedding,
             'query_model': default_query}
        ]

    # モードに応じてベンチマークを実行
    if mode in ['query', 'both']:
        console.print("\n[bold]クエリベンチマーク設定:[/bold]")
        console.print(f"- クエリ数: {len(query_list)}")
        console.print(f"- 設定数: {len(bench_configs)}")
        console.print(f"- 各テストの実行回数: {runs}")
        console.print(f"- 検索上位数: {top_k}")
        console.print(f"- 総テスト数: {len(query_list) * len(bench_configs) * runs}")

        # top_kをbench_configsに追加
        for config in bench_configs:
            if 'top_k' not in config:
                config['top_k'] = top_k

        runner.run_benchmark_suite(query_list, bench_configs, runs)
        runner.display_results()

    if mode in ['update', 'both']:
        if not update_files:
            console.print("\n[yellow]警告: インクリメンタル更新用のファイルが指定されていません[/yellow]")
        else:
            console.print("\n[bold]インクリメンタル更新ベンチマーク設定:[/bold]")
            console.print(f"- 更新ファイル数: {len(update_files)}")
            console.print(f"- 各テストの実行回数: {runs}")

            # 各埋め込みモデルでテスト
            update_models = ['bgem3', 'gemini']
            for model in update_models:
                console.print(f"\n[bold cyan]{model.upper()} インクリメンタル更新ベンチマーク[/bold cyan]")
                result = runner.benchmark_incremental_update(list(update_files), model, runs)

                if result:
                    runner.results.append({
                        'type': 'incremental_update',
                        'result': result
                    })

                    # 統計情報を表示
                    console.print(f"\n[bold]統計情報 ({model}):[/bold]")
                    for key, stat in result['stats'].items():
                        if stat:
                            console.print(
                                f"  {key}: "
                                f"平均={stat['mean']:.3f}, "
                                f"最小={stat['min']:.3f}, "
                                f"最大={stat['max']:.3f}"
                            )

    if save:
        runner.save_results("results")


if __name__ == "__main__":
    main()
