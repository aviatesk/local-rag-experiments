---
file-created: 2025-07-18T14:22
file-modified: 2025-07-18T14:22
---
# Gemini Embedding パフォーマンス分析

## 現在の問題

10件のドキュメントに30秒 = **3秒/ドキュメント** は遅すぎます。

## ボトルネックの原因

### 1. 逐次処理（最大の問題）
```python
# 現在の実装：1つずつ順番に処理
for text in texts:
    embedding = self.generate_embedding_api(text)  # 3秒待つ
    time.sleep(0.1)  # さらに0.1秒待つ
```

### 2. 不要な遅延
- `rate_limit_delay: 0.1秒` × 10件 = 1秒の無駄
- `gcloud auth print-access-token` を毎回実行

### 3. API応答時間の内訳（推定）

| 処理 | 時間 |
|------|------|
| gcloud auth実行 | 0.5-1秒 |
| ネットワーク往復 | 0.1-0.3秒 |
| Vertex AI処理 | 0.5-1秒 |
| rate_limit_delay | 0.1秒 |
| **合計** | **約3秒/リクエスト** |

## 高速化ソリューション

### 1. 並列処理（5倍高速化）
```python
# 5つの同時リクエスト
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(api_call, text) for text in texts]
```

### 2. アクセストークンのキャッシュ
```python
# 50分間有効なトークンをキャッシュ
if self._token_expiry > time.time():
    return self._cached_token
```

### 3. レート制限の最適化
- 並列処理時は個別のsleepは不要
- エラー時のみリトライ

## パフォーマンス改善見込み

| 方式 | 10件の処理時間 | スループット |
|------|----------------|--------------|
| **現在（逐次）** | 30秒 | 0.33件/秒 |
| **改善後（並列5）** | 6-8秒 | 1.25-1.67件/秒 |

## 実行方法

```bash
# 高速版を使用
python scripts/index_gemini_fast.py

# または既存スクリプトを修正
chmod +x scripts/index_gemini_fast.py
```

## API制限の確認

Vertex AI Embeddingsの制限：
- **リクエスト/分**: 600（十分）
- **同時リクエスト**: 10（5並列なら問題なし）

## モニタリング

高速版では各バッチの処理時間を表示：
```
バッチ処理時間: 6.5秒 (10件)
平均API応答時間: 1.2秒
```

これで処理が遅い原因（自分側の実装）が判明し、解決策を実装しました！