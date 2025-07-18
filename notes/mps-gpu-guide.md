---
file-created: 2025-07-18T14:17
file-modified: 2025-07-18T14:17
---
# Apple Silicon GPU (MPS) 使用ガイド

## 概要

Apple M2 ProのGPUを使用してBGE-M3の埋め込み生成を高速化します。

## 設定内容

### デバイス検出
```python
# Metal Performance Shaders (MPS) を自動検出
self.device = "mps" if torch.backends.mps.is_available() else "cpu"
```

### パフォーマンス設定

1. **FP16（半精度）使用**
   - メモリ使用量を削減
   - 処理速度を向上
   - `config.yaml`で`use_fp16: true`に設定済み

2. **バッチサイズ**
   - GPU使用時: 32（大きめ）
   - CPU使用時: 8（控えめ）

## 実行時の確認

スクリプト実行時に以下のようなログが出力されます：

```
2025-07-18 14:30:00,000 - common.utils - INFO - Device: mps
2025-07-18 14:30:00,001 - common.utils - INFO - FP16: True
```

## パフォーマンス最適化

### メモリ使用量の監視

```bash
# Activity Monitorで確認
# または以下のPythonコードで確認
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

### トラブルシューティング

1. **FP16が不安定な場合**
   ```yaml
   # config.yamlで無効化
   embeddings:
     bgem3:
       use_fp16: false
   ```

2. **メモリ不足エラー**
   ```yaml
   # バッチサイズを調整
   chunking:
     bgem3:
       batch_size_gpu: 16  # 32から減らす
   ```

3. **MPSが使用されない場合**
   ```python
   # PyTorchのバージョン確認
   import torch
   print(torch.__version__)  # 2.0以降が必要
   ```

## ベンチマーク参考値

Apple M2 Proでの推定パフォーマンス：
- **MPS使用時**: 約50-100 chunks/秒
- **CPU使用時**: 約10-20 chunks/秒

実際の速度は文書の長さとバッチサイズに依存します。

## 注意事項

- MPSはApple Silicon Mac専用
- PyTorch 2.0以降が必要
- 一部の操作はCPUにフォールバックする場合がある
- 初回実行時はモデルのMPSへの最適化で時間がかかることがある