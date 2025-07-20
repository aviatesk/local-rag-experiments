#!/usr/bin/env python3
"""
ファイルのトークン数をカウントするスクリプト
tiktoken (cl100k_base)を使用してOpenAI GPT-4互換のトークン数を計算
"""
import sys
from pathlib import Path
import click
import tiktoken
from typing import List
from rich.console import Console
from rich.table import Table

console = Console()


def count_tokens(text: str) -> int:
    """テキストのトークン数をカウント"""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


def analyze_file(file_path: Path) -> dict:
    """ファイルを分析してトークン数と統計を返す"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 基本統計
        token_count = count_tokens(content)
        char_count = len(content)
        line_count = content.count('\n') + 1
        
        # 言語別の推定（簡易判定）
        japanese_chars = sum(1 for c in content if '\u3000' <= c <= '\u9fff' or '\uff00' <= c <= '\uffef')
        japanese_ratio = japanese_chars / char_count if char_count > 0 else 0
        
        return {
            'file': file_path.name,
            'path': str(file_path),
            'tokens': token_count,
            'characters': char_count,
            'lines': line_count,
            'tokens_per_char': token_count / char_count if char_count > 0 else 0,
            'chars_per_token': char_count / token_count if token_count > 0 else 0,
            'japanese_ratio': japanese_ratio,
            'exists': True,
            'error': None
        }
    except Exception as e:
        return {
            'file': file_path.name,
            'path': str(file_path),
            'exists': False,
            'error': str(e)
        }


def display_results(results: List[dict], verbose: bool = False):
    """結果を表示"""
    # エラーがあるファイルを先に表示
    errors = [r for r in results if not r.get('exists', False)]
    if errors:
        console.print("\n[red]エラー:[/red]")
        for err in errors:
            console.print(f"  {err['file']}: {err['error']}")
    
    # 成功したファイルの結果を表示
    success_results = [r for r in results if r.get('exists', False)]
    if not success_results:
        return
    
    # サマリーテーブル
    table = Table(title="トークン数カウント結果")
    table.add_column("ファイル", style="cyan")
    table.add_column("トークン数", justify="right", style="green")
    table.add_column("文字数", justify="right")
    table.add_column("行数", justify="right")
    table.add_column("文字/トークン", justify="right")
    
    if verbose:
        table.add_column("日本語率", justify="right")
    
    for result in success_results:
        row = [
            result['file'],
            f"{result['tokens']:,}",
            f"{result['characters']:,}",
            f"{result['lines']:,}",
            f"{result['chars_per_token']:.1f}"
        ]
        if verbose:
            row.append(f"{result['japanese_ratio']:.1%}")
        table.add_row(*row)
    
    # 合計行
    if len(success_results) > 1:
        total_tokens = sum(r['tokens'] for r in success_results)
        total_chars = sum(r['characters'] for r in success_results)
        total_lines = sum(r['lines'] for r in success_results)
        avg_chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
        
        row = [
            "[bold]合計[/bold]",
            f"[bold]{total_tokens:,}[/bold]",
            f"[bold]{total_chars:,}[/bold]",
            f"[bold]{total_lines:,}[/bold]",
            f"[bold]{avg_chars_per_token:.1f}[/bold]"
        ]
        if verbose:
            total_japanese = sum(r['characters'] * r['japanese_ratio'] for r in success_results)
            avg_japanese_ratio = total_japanese / total_chars if total_chars > 0 else 0
            row.append(f"[bold]{avg_japanese_ratio:.1%}[/bold]")
        
        table.add_row(*row)
    
    console.print("\n")
    console.print(table)
    
    # 詳細情報（verboseモード）
    if verbose and success_results:
        console.print("\n[bold]詳細情報:[/bold]")
        for result in success_results:
            console.print(f"\n[cyan]{result['file']}[/cyan]")
            console.print(f"  パス: {result['path']}")
            console.print(f"  トークン/文字比: {result['tokens_per_char']:.3f}")
            
            # チャンク数の推定
            console.print(f"  推定チャンク数:")
            console.print(f"    BGE-M3 (3000トークン): {result['tokens'] / 3000:.1f}チャンク")
            console.print(f"    Gemini (1500トークン): {result['tokens'] / 1500:.1f}チャンク")


@click.command()
@click.argument('files', nargs=-1, required=True, type=click.Path(exists=False))
@click.option('--verbose', '-v', is_flag=True, help='詳細情報を表示')
def main(files, verbose):
    """ファイルのトークン数をカウント
    
    tiktoken (cl100k_base)を使用してOpenAI GPT-4互換のトークン数を計算します。
    
    使用例:
    
    \b
    # 単一ファイル
    python scripts/misc/count_tokens.py README.md
    
    \b
    # 複数ファイル
    python scripts/misc/count_tokens.py notes/*.md
    
    \b
    # 詳細情報付き
    python scripts/misc/count_tokens.py -v experiments/data.txt
    """
    # ワイルドカード展開の処理
    all_files = []
    for pattern in files:
        path = Path(pattern)
        if '*' in pattern or '?' in pattern:
            # glob pattern
            parent = path.parent if path.parent != Path('.') else Path.cwd()
            matched = list(parent.glob(path.name))
            if matched:
                all_files.extend(matched)
            else:
                all_files.append(path)  # マッチしない場合もエラーとして記録
        else:
            all_files.append(path)
    
    if not all_files:
        console.print("[red]ファイルが指定されていません[/red]")
        return
    
    # 各ファイルを分析
    results = []
    for file_path in all_files:
        result = analyze_file(file_path)
        results.append(result)
    
    # 結果を表示
    display_results(results, verbose)


if __name__ == "__main__":
    main()