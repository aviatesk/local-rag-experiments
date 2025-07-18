# Obsidian×RAG Experiment

Obsidian VaultのRAGシステム構築実験。ローカル埋め込みモデル（BGE-M3）とクラウド埋め込みモデル（gemini-embedding-001）の比較環境。

- 目的: プライバシー重視のローカルモデルとクラウドモデルの検索精度・パフォーマンス比較
- 対象: Obsidian Vault内のMarkdownファイル（日本語含む）
- 埋め込みモデル:
  - BGE-M3: ローカル実行、日本語対応、最大8192トークン、1024次元
  - gemini-embedding-001: Vertex AI、最大2048トークン、768次元（圧縮）

```
obsidian-rag-experiment/
├── README.md              # このファイル
├── pyproject.toml        # Python依存関係（uv用）
├── .gitignore           # Git除外設定
├── config/               # 設定ファイル
│   ├── config.yaml       # 共通設定
│   ├── .env             # 環境変数（.gitignore対象）
│   └── .env.example      # 環境変数テンプレート
├── scripts/              # 実行スクリプト
│   ├── common/          # 共通モジュール
│   │   ├── __init__.py
│   │   ├── chunker.py   # Markdown対応テキスト分割
│   │   ├── db.py        # ChromaDB操作
│   │   └── utils.py     # ユーティリティ
│   ├── index_bgem3.py   # BGE-M3インデックス化
│   ├── index_gemini.py  # Geminiインデックス化（並列処理）
│   ├── search.py        # 埋め込み検索・比較
│   ├── query.py         # RAG質問応答
│   ├── update_bgem3.py  # BGE-M3インクリメンタル更新
│   ├── update_gemini.py # Geminiインクリメンタル更新
│   ├── benchmark.py     # パフォーマンス計測
│   └── clean_db.py      # DB管理
├── data/                # データ保存（.gitignore対象）
│   └── chroma_db/       # ChromaDB永続化
├── logs/                # ログファイル（.gitignore対象）
│   ├── index_bgem3_*.log
│   ├── index_gemini_*.log
│   └── ...
├── results/             # 実験結果
│   ├── indexing_stats_bgem3_*.md   # BGE-M3実行統計
│   ├── indexing_stats_gemini_*.md  # Gemini実行統計
│   ├── benchmark_results_*.json    # ベンチマーク結果
│   └── experiment_notes.md         # 実験記録
└── notes/               # ドキュメント
    ├── metadata-handling.md  # メタデータ設計
    └── benchmark_setup.md    # ベンチマーク設定
```

## セットアップ

### 1. 環境準備

```bash
uv venv

source .venv/bin/activate

uv pip sync -e .
```

### 2. 設定ファイル準備

```bash
cp config/.env.example config/.env

# .envを編集して以下を設定:
# - VAULT_PATH: Obsidian Vaultのパス
# - GCP_PROJECT_ID: Google CloudプロジェクトID（Gemini使用時）
# - GCP_REGION: Vertex AIリージョン（例: us-central1）
```

### 3. `gcloud`

```bash
gcloud init # or auth, or whatever works
```

### 4. Ollama設定

```bash
brew install ollama

ollama serve

ollama pull gemma3n:e4b
```

## 実験

### BGE-M3でインデックス作成

```bash
python scripts/index_bgem3.py
```

- 完全ローカル実行、ネットワーク不要
- Apple Silicon上ではMPS GPU使用
- 実行統計は `results/indexing_stats_bgem3.md` に保存

### `gemini-embedding-001`でインデックス作成

```bash
python scripts/index_gemini.py
```

- Vertex AI APIを使用（ネットワーク必要）
- 5並列でAPI呼び出し（高速化）
- 課金対象（使用量はGCPコンソールで確認）
- 実行統計は `results/indexing_stats_gemini.md` に保存

デフォルトのconfiguration([config.yaml](./config/config.yaml))では、
Google Cloud Consoleで使用するリージョンの
"Embed content input tokens per minute per region per base_model"
クォータを200,000から300,000以上に引き上げる必要がある
詳細は[トラブルシューティング](#トラブルシューティング)セクションを参照。

### 埋め込み検索実行

```bash
# config.yamlに記載されているモデル or Gemini (default, `gemini-embedding-001`) で検索
python scripts/search.py "検索クエリ"

# BGE-M3 (`BAAI/bge-m3`) で検索
python scripts/search.py "検索クエリ" -e bgem3

# 両モデルで検索
python scripts/search.py "検索クエリ" -e both

# config.yamlのデフォルトクエリを使用
python scripts/search.py

# 上位10件取得（デフォルト5件）
python scripts/search.py "検索クエリ" -k 10

# 結果比較表示
python scripts/search.py "検索クエリ" -e both --compare

# 結果をJSONに保存
python scripts/search.py "検索クエリ" --save
```

### RAG質問応答

```bash
# デフォルト（Geminiで検索・回答）
python scripts/query.py "JETLSのincremental analysis systemの実装状況は？"

# BGE-M3で検索、Ollamaで回答（完全ローカル）
python scripts/query.py "JETLSのincremental analysis systemの実装状況は？" -e bgem3 -q gemma3n:e4b

# Geminiで検索、別のGeminiモデルで回答
python scripts/query.py "JETLSのincremental analysis systemの実装状況は？" -q gemini-pro

# 検索のみ（回答生成なし）
python scripts/query.py "JETLSのincremental analysis systemの実装状況は？" -q ""

# 上位10件を検索して回答
python scripts/query.py "EscapeAnalysisの最新の変更点は？" -k 10 -q gemma3n:e4b

# 両モデルで検索して比較
python scripts/query.py "JETLSの実装状況は？" -e both --compare

# 検索結果の詳細を表示
python scripts/query.py "JETLSの実装状況は？" -v

# config.yamlのデフォルトクエリとモデルを使用
python scripts/query.py

# ヘルプを表示
python scripts/query.py --help
```

#### 主に使用しているQueryモデル

- Ollama（ローカル）: `gemma3n:e4b`
- Gemini API（クラウド）: `gemini-2.5-flash`

### DB管理

```bash
# コレクション一覧表示
python scripts/clean_db.py --list

# 特定コレクション削除
python scripts/clean_db.py --remove obsidian_gemini

# 全データ削除
python scripts/clean_db.py --all
```

### インクリメンタル更新

特定のファイルのみを効率的に更新:

```bash
# BGE-M3で特定ファイルを更新
python scripts/update_bgem3.py "notes/JETLS.md" "notes/optimization.md"

# Geminiで更新
python scripts/update_gemini.py "notes/JETLS.md"

# config.yamlのupdate_notesリストを使用
python scripts/update_bgem3.py --config-notes
```

削除されたファイルも自動的に検出してDBから削除。

## 埋め込み設定

`BAAI/bge-m3`:
- Dense埋め込みのみ使用（1024次元）
- バッチサイズ: 32（GPU）/ 8（CPU）
- FP16使用（MPS/CUDA時）

`gemini-embedding-001`:
- タスクタイプ: RETRIEVAL_DOCUMENT/QUERY
- 出力次元: 768（3072から圧縮）
- 自動トランケート有効
- 5並列API呼び出し
- アクセストークンキャッシュ（50分）

## パフォーマンスベンチマーク

```bash
# config.yamlの設定を使用してベンチマーク
python scripts/benchmark.py --save

# 特定のクエリでベンチマーク
python scripts/benchmark.py -q "JETLSの実装状況は？"

# 複数クエリでのベンチマーク
python scripts/benchmark.py -q "JETLSの実装" -q "型推論の仕組み" -q "最適化手法"

# インクリメンタル更新のベンチマーク
python scripts/benchmark.py -m update -u "notes/JETLS.md" -u "notes/optimization.md"

# クエリと更新の両方をベンチマーク
python scripts/benchmark.py -m both --save

# ファイルからクエリを読み込んでベンチマーク
python scripts/benchmark.py -f queries.txt -r 5 --save

# カスタム設定でベンチマーク
python scripts/benchmark.py -q "テストクエリ" -r 10 -k 10 --save

# ヘルプを表示
python scripts/benchmark.py --help
```

### ベンチマーク設定

デフォルトのベンチマーク設定（config.yaml）:

**使用クエリ**:
1. JETLSのincremental analysis systemの実装状況を教えて
2. JETLSのincremental analysis system、特にそのcode repository infrastructureの実装状況を教えて
3. Please tell me about the progress status of the EscapeAnalysis project
4. この人の仕事を一言で表すとどんなもの?
5. ChromaDBの使い方を教えて

**テスト構成**:
- 埋め込みモデル: BGE-M3（ローカル）、Gemini（クラウド）
- クエリモデル: Gemini-2.5-Flash、Gemma3n:e4b（Ollama）
- 組み合わせ: 6パターン
  - 検索のみ: BGE-M3、Gemini
  - 検索+回答: Gemini+Gemini Flash、Gemini+Gemma3n、BGE-M3+Gemini Flash、BGE-M3+Gemma3n
- 実行回数: 5回（統計計測用）
- 検索上位数: 5件

**インクリメンタル更新対象**:
- JETLS/Incremental analysis system.md
- JETLS/JETLS meeting time.md

### パフォーマンス指標

**クエリベンチマーク**:
- 埋め込み生成時間: クエリの埋め込みベクトル生成
- 検索時間: 類似チャンクの検索
- 回答生成時間: LLMによる回答生成（オプション）
- 合計時間: 全処理時間

**インクリメンタル更新ベンチマーク**:
- 更新時間: ファイルの処理と埋め込み生成
- チャンク処理速度: chunks/秒
- DB操作時間: ChromaDBへの保存時間
- API時間: Geminiの場合のAPI呼び出し時間

**統計情報**:
- 平均値、最小値、最大値、標準偏差を算出

### ベンチマークモード

- `query`: クエリベンチマーク（デフォルト）
- `update`: インクリメンタル更新ベンチマーク
- `both`: 両方を実行

## Configurations

[`config/config.yaml`](./config/config.yaml)で以下の設定が可能:

```yaml
# デフォルトクエリとモデル設定
defaults:
  queries:
    - "JETLSのincremental analysis systemの実装状況を教えて"
    - "BGE-M3の特徴と利点は？"
  embedding_model: "gemini"  # bgem3, gemini, both
  query_model: "gemini-2.5-flash"  # llama3.2, gemma2, mistral等

# ベンチマーク設定
benchmark:
  configurations:
    - name: "BGE-M3（ローカル）"
      embedding_model: "bgem3"
      query_model: null
    - name: "Gemini（クラウド）"
      embedding_model: "gemini"
      query_model: null
    - name: "BGE-M3 + Ollama"
      embedding_model: "bgem3"
      query_model: "llama3.2"
  runs: 3  # デフォルト実行回数
  top_k: 5  # デフォルト検索件数

# インクリメンタル更新対象
vault:
  update_notes:
    - "JETLS/Incremental analysis system.md"
    - "JETLS/JETLS meeting time.md"
```

## PKM specialization

### チャンク戦略

- Markdown見出しベースの階層的分割
- 短いセクションの自動結合（最大サイズの30%以下）
- 見出しレベル差2以内で結合（階層構造保持）
- BGE-M3: 最大6000トークン/チャンク
- Gemini: 最大1500トークン/チャンク（API制限）

### メタデータ設計

各チャンクに付与される情報:
- `name`: ファイル名 = Obsidianノートタイトル
- `relative_path`: Vault内相対パス
- `heading`: セクション見出し
- `heading_level`: 見出しレベル（1-6）
- `chunk_index`: チャンク番号
- `indexed_at`: インデックス作成日時
- `model`: 使用した埋め込みモデル

詳細は `notes/metadata-handling.md` 参照

### メタデータ埋め込み

検索精度向上のため、以下のメタデータを埋め込みテキストに含めています:
- ノートタイトル（ファイル名）
- セクションタイトル（見出し）
- 最終更新日

これにより、特定のノートやセクションを狙い撃ちで検索可能。

## 今後の拡張

- [x] インクリメンタル更新（ファイル変更検知）
- [x] メタデータを埋め込みに含める
- [ ] Frontmatter metadata抽出（タグ等）

## トラブルシューティング

### GPU関連
- CUDA out of memory: [config.yaml](./config/config.yaml) でバッチサイズ調整
- MPS not available: 自動的にCPUフォールバック

### API関連
- 401エラー: `gcloud auth application-default login` 再実行
- 429エラー（レート制限）:
  - **原因**: Gemini Embeddingのクォータ超過
  - **解決法1**: Google Cloud Consoleでクォータ増加申請
    1. IAM & Admin → Quotas
    2. "Embed content input tokens per minute per region per base_model" を検索
    3. 200,000 → 300,000以上に増加申請
  - **解決法2**: [config.yaml](./config/config.yaml) で調整
    ```yaml
    gemini:
      batch_size: 6  # さらに削減
      rate_limit_delay: 0.3  # さらに増加
      max_concurrent_requests: 3  # さらに削減
    ```
- タイムアウト: 30秒に設定済み

### DB関連
- コレクション削除: [scripts/clean_db.py](./scripts/clean_db.py) 使用
