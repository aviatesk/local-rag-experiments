---
file-created: 2025-07-18T14:48
file-modified: 2025-07-18T14:48
---
# メタデータハンドリングの仕組み

## 現在のメタデータ構造

各チャンクには以下のメタデータが付与されています：

### ファイルレベルのメタデータ
- `name`: ファイル名（拡張子なし）= Obsidianのノートタイトル
- `absolute_path`: ファイルの絶対パス
- `relative_path`: Vault内での相対パス
- `directory`: 親ディレクトリ（Vault相対）

### インデックス時のメタデータ
- `indexed_at`: インデックス作成日時（ISO形式）
- `model`: 使用した埋め込みモデル（BGE-M3 or gemini-embedding-001）

### チャンクレベルのメタデータ
- `chunk_index`: ファイル内でのチャンク番号（0始まり）
- `is_partial`: 大きなセクションを分割したチャンクかどうか
- `heading`: セクションの見出し
- `heading_level`: 見出しレベル（1-6）
- `merged_headings`: 結合されたセクションの見出しリスト（結合時のみ）

## 検索時の活用方法

ChromaDBでは、メタデータフィルタを使った検索が可能です：

```python
# 特定のディレクトリ内のノートだけを検索
results = collection.query(
    query_texts=["検索クエリ"],
    where={"directory": {"$eq": "projects/rag-experiment"}},
    n_results=10
)

# 特定のノートタイトル（ファイル名）で絞り込み
results = collection.query(
    query_texts=["検索クエリ"],
    where={"name": {"$eq": "埋め込みモデルの比較"}},
    n_results=10
)

# 最近インデックスされたものだけを検索
results = collection.query(
    query_texts=["検索クエリ"],
    where={"indexed_at": {"$gt": "2024-01-01T00:00:00"}},
    n_results=10
)
```

## 将来的な拡張案

### 1. Obsidian Frontmatterの抽出
```yaml
---
tags: [rag, embedding, experiment]
aliases: [RAG実験, 埋め込み比較]
created: 2024-01-15
---
```

このようなfrontmatterから：
- `tags`: タグのリスト
- `aliases`: エイリアスのリスト
- カスタムプロパティ

### 2. インラインタグの抽出
本文中の `#タグ名` を抽出してメタデータに追加

### 3. リンク情報の保持
- `[[他のノート]]` へのリンクを抽出
- バックリンク情報の構築

### 実装例（frontmatter抽出）
```python
import yaml
import re

def extract_frontmatter(content: str) -> tuple[dict, str]:
    """Frontmatterを抽出して、本文と分離"""
    pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.match(pattern, content, re.DOTALL)
    
    if match:
        try:
            metadata = yaml.safe_load(match.group(1))
            content_without_fm = content[match.end():]
            return metadata, content_without_fm
        except yaml.YAMLError:
            pass
    
    return {}, content
```

## インクリメンタル更新での考慮事項

将来的にインクリメンタル更新を実装する際は：

1. **ファイルの更新日時**を追跡
2. **チャンクIDの安定性**を保証（同じ内容なら同じID）
3. **削除されたファイル**のチャンクをDBから削除
4. **メタデータの変更**（タグの追加など）も追跡

現在のID生成方式：
```python
id_string = f"{relative_path}_{chunk_index}"
chunk_id = hashlib.md5(id_string.encode()).hexdigest()
```

これにより、ファイルパスとチャンク位置が同じなら同じIDになるため、`upsert`で自然に更新される仕組みになっています。