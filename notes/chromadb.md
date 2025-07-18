---
file-created: 2025-07-18T14:53
file-modified: 2025-07-18T14:53
---
# ChromaDBのデータ保存構造

## 概要

ChromaDBは**ハイブリッドストレージ**システムを使用しています。データは以下の2つの場所に分散して保存されます：

1. **SQLite3データベース** (`chroma.sqlite3`)
2. **バイナリファイル** (各コレクションのディレクトリ内)

## データの分散

### 1. SQLiteデータベース (`chroma.sqlite3`)

メタデータと管理情報を保存：

```sql
-- 主要なテーブル
collections          -- コレクション情報（名前、次元数など）
embeddings          -- ドキュメントのメタデータ（ID、テキスト、メタデータ）
embedding_metadata  -- カスタムメタデータ
segments            -- セグメント管理情報
```

**保存される情報**：
- コレクション名と設定
- ドキュメントID
- ドキュメントテキスト
- メタデータ（file_path, heading, chunk_indexなど）
- 全文検索用インデックス

### 2. バイナリファイル（HNSWインデックス）

高速ベクトル検索用のデータ：

```
ad975352-4d66-417b-a906-067cf0fcee5f/  # コレクションのUUID
├── data_level0.bin    # HNSWグラフのレベル0データ
├── header.bin         # インデックスヘッダー情報
├── length.bin         # ベクトル長情報
└── link_lists.bin     # グラフのリンク情報
```

**保存される情報**：
- 実際の埋め込みベクトル（768次元の数値配列）
- HNSW（Hierarchical Navigable Small World）グラフ構造
- 高速近似最近傍探索用のインデックス

## データフロー

```
1. ドキュメント追加時：
   テキスト → 埋め込み生成 → 
   ├── メタデータ → SQLite
   └── ベクトル → バイナリファイル

2. 検索時：
   クエリ → 埋め込み生成 → 
   ├── ベクトル類似度検索 → バイナリファイル
   └── メタデータ取得 → SQLite
```

## 実際のデータ確認方法

### SQLiteの内容を確認

```bash
# コレクション一覧
sqlite3 data/chroma_db/chroma.sqlite3 "SELECT name, dimension FROM collections;"

# ドキュメント数
sqlite3 data/chroma_db/chroma.sqlite3 "SELECT COUNT(*) FROM embeddings;"

# メタデータのサンプル
sqlite3 data/chroma_db/chroma.sqlite3 "SELECT id, document FROM embeddings LIMIT 5;"
```

### データサイズの確認

```bash
# SQLiteのサイズ
du -h data/chroma_db/chroma.sqlite3

# バイナリファイルのサイズ
du -sh data/chroma_db/*/
```

## 更新タイミング

1. **`add_documents()`呼び出し時**：
   - SQLiteにメタデータを挿入
   - バイナリファイルにベクトルを追加
   - HNSWインデックスを更新

2. **バッチ処理**：
   - ChromaDBは内部でバッチ処理を最適化
   - 大量のドキュメント追加時も効率的

## 利点

- **高速検索**: ベクトルはバイナリ形式で効率的に保存
- **柔軟なメタデータ**: SQLiteで複雑なフィルタリングが可能
- **スケーラビリティ**: 大規模データでも性能を維持
- **永続化**: 両方のデータが保存されるため再起動後も利用可能

## 注意点

- SQLiteとバイナリファイルの**両方**が必要
- 片方だけコピーしても動作しない
- バックアップ時は`chroma_db`ディレクトリ全体をコピー