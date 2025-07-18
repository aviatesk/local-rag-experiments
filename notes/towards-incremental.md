---
file-created: 2025-07-18T14:15
file-modified: 2025-07-18T14:15
---
# インクリメンタル更新ガイド

## ChromaDBの更新機能

ChromaDBは以下の操作をサポートしており、インクリメンタルな更新が可能です：

### 基本操作

1. **追加（Add）**
   ```python
   collection.add(
       embeddings=embeddings,
       documents=documents,
       metadatas=metadatas,
       ids=ids
   )
   ```

2. **更新（Update）**
   ```python
   collection.update(
       ids=ids,
       embeddings=embeddings,  # 省略可
       documents=documents,    # 省略可
       metadatas=metadatas    # 省略可
   )
   ```

3. **削除（Delete）**
   ```python
   # IDで削除
   collection.delete(ids=["id1", "id2"])
   
   # メタデータ条件で削除
   collection.delete(where={"file_path": "/path/to/file.md"})
   ```

4. **Upsert（追加または更新）**
   ```python
   collection.upsert(
       ids=ids,
       embeddings=embeddings,
       documents=documents,
       metadatas=metadatas
   )
   ```

## 実装戦略

### 1. ファイル単位の管理

現在の実装では、各チャンクのIDを以下のように生成：

```python
id_string = f"{relative_path}_{chunk_index}"
chunk_id = hashlib.md5(id_string.encode()).hexdigest()
```

これにより：
- 同じファイルの同じチャンクは常に同じID
- ファイルが更新されても、チャンク位置が同じならIDは同じ
- `upsert`を使えば自動的に更新される

### 2. 変更検出の方法

#### 方法1: タイムスタンプベース
```python
# メタデータに最終更新時刻を保存
metadata['last_modified'] = os.path.getmtime(file_path)

# 更新が必要か判定
if current_mtime > stored_mtime:
    # 更新処理
```

#### 方法2: ハッシュベース
```python
# メタデータにファイルハッシュを保存
metadata['file_hash'] = calculate_file_hash(file_path)

# 変更検出
if current_hash != stored_hash:
    # 更新処理
```

### 3. 効率的な更新フロー

```python
def incremental_update():
    # 1. 変更ファイルの検出
    new_files, modified_files, deleted_files = detect_changes()
    
    # 2. 削除処理（削除されたファイルのチャンクを削除）
    for file in deleted_files:
        collection.delete(where={"relative_path": file})
    
    # 3. 更新処理（変更されたファイルを再インデックス）
    for file in modified_files:
        # 古いチャンクを削除
        collection.delete(where={"relative_path": file})
        # 新しいチャンクを追加
        process_and_add_file(file)
    
    # 4. 新規追加
    for file in new_files:
        process_and_add_file(file)
```

## 実用的な実装例

### Obsidianプラグインとの連携

```python
# ファイル変更を監視
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class VaultChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.md'):
            update_single_file(event.src_path)
    
    def on_created(self, event):
        if event.src_path.endswith('.md'):
            add_single_file(event.src_path)
    
    def on_deleted(self, event):
        if event.src_path.endswith('.md'):
            remove_single_file(event.src_path)
```

### バッチ更新スクリプト

```bash
# 日次で実行
python scripts/update_incremental.py --model bgem3
python scripts/update_incremental.py --model gemini
```

## パフォーマンス考慮事項

1. **大量更新時の最適化**
   - バッチ処理でまとめて更新
   - トランザクション的な処理

2. **インデックスの最適化**
   - 定期的な`optimize()`の実行
   - 不要なデータの削除

3. **並行処理**
   - 読み取りは並行可能
   - 書き込みは排他制御が必要

## 実装予定の機能

1. **変更検出の改善**
   - ファイルハッシュの保存
   - 差分チャンク検出

2. **スマート更新**
   - 変更された部分のみ再インデックス
   - チャンクの差分検出

3. **バージョン管理**
   - 過去のバージョンを保持
   - ロールバック機能