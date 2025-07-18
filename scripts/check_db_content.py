#!/usr/bin/env python3
"""
ChromaDBの内容を確認するスクリプト
"""
import sys
from pathlib import Path
import click

sys.path.append(str(Path(__file__).parent))

from common.utils import setup_environment
from common.db import VectorDB


@click.command()
@click.option('--collection', '-c', default='obsidian_bgem3', help='コレクション名')
@click.option('--path', '-p', help='特定のパスでフィルタ（例: JETLS/Incremental）')
@click.option('--limit', '-l', default=10, help='表示する最大件数')
@click.option('--show-content', '-s', is_flag=True, help='チャンクの内容を表示')
def main(collection, path, limit, show_content):
    """ChromaDBの内容を確認"""
    config = setup_environment()
    db = VectorDB(config['database']['persist_directory'])

    # コレクションを取得
    coll = db.client.get_collection(collection)

    # データを取得（パスフィルタは後で適用）
    if path:
        # 全件取得してからフィルタリング
        results = coll.get(
            limit=1000,  # 十分な数を取得
            include=["metadatas", "documents"] if show_content else ["metadatas"]
        )

        # パスでフィルタリング
        filtered_ids = []
        filtered_metadatas = []
        filtered_documents = [] if show_content else None

        for i, metadata in enumerate(results['metadatas']):
            if path.lower() in metadata.get('relative_path', '').lower():
                filtered_ids.append(results['ids'][i])
                filtered_metadatas.append(metadata)
                if show_content and 'documents' in results:
                    filtered_documents.append(results['documents'][i])

        # 結果を更新
        results['ids'] = filtered_ids[:limit]
        results['metadatas'] = filtered_metadatas[:limit]
        if show_content and filtered_documents:
            results['documents'] = filtered_documents[:limit]
    else:
        # フィルタなしの場合
        results = coll.get(
            limit=limit,
            include=["metadatas", "documents"] if show_content else ["metadatas"]
        )

    print(f"\nコレクション: {collection}")
    print(f"総ドキュメント数: {coll.count()}")

    if path:
        print(f"フィルタ: relative_path contains '{path}'")

    print(f"\n取得件数: {len(results['ids'])}")
    print("-" * 80)

    for i, (id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
        print(f"\n[{i+1}] ID: {id}")
        print(f"  パス: {metadata.get('relative_path', 'N/A')}")
        print(f"  見出し: {metadata.get('heading', 'N/A')}")
        print(f"  チャンク番号: {metadata.get('chunk_index', 'N/A')}")
        print(f"  インデックス日時: {metadata.get('indexed_at', 'N/A')}")

        if show_content and 'documents' in results:
            content = results['documents'][i]
            preview = content[:500] + "..." if len(content) > 500 else content
            print(f"  内容プレビュー:")
            # 改行を保持して表示
            for line in preview.split('\n')[:25]:  # 最初の25行まで
                print(f"    {line}")

    # 特定のパスが見つかったか確認
    if path:
        paths = [m.get('relative_path', '') for m in results['metadatas']]
        unique_paths = set(paths)
        print(f"\n見つかったユニークなパス:")
        for p in sorted(unique_paths):
            print(f"  - {p}")


if __name__ == "__main__":
    main()
