# Sinkhorn Outlier Filtering

シンクホーンマッチングの外れ値除去機能についてのドキュメント

## 概要

このリポジトリには、シンクホーンマッチングの精度を向上させるための2つの外れ値フィルタリング手法が実装されています：

1. **確率比フィルタ (Probability Ratio Filter)**: 最良マッチと次点マッチの確率比が低い曖昧なマッチを除外
2. **ダストビンマージンフィルタ (Dustbin Margin Filter)**: ダストビン確率が高い点(マッチしない方が良い点)を除外

## 実装方法

### 1. NumPy後処理版（推論後のフィルタリング）

ONNXモデルの推論結果に対してCPU上でフィルタリングを適用します。

**ファイル**: `pytorch_model/matching/outlier_filters.py`, `sample/image_matching.py`

**使用例**:
```bash
python sample/image_matching.py \
  --model model.onnx \
  --input1 img1.png \
  --input2 img2.png
```

**メリット**:
- ✅ 実装が簡単で、既存のONNXモデルをそのまま使用可能
- ✅ パラメータを実行時に自由に変更できる
- ✅ デバッグが容易

**デメリット**:
- ❌ CPU処理のためオーバーヘッドが発生
- ❌ GPU/NPUアクセラレーションを活用できない

---

### 2. PyTorchモデル統合版（ONNX組み込み）

フィルタをPyTorchモデル内に組み込み、ONNXエクスポート時に含めます。

**ファイル**:
- `pytorch_model/matching/sinkhorn.py` - `SinkhornMatcherWithFilters`クラス
- `pytorch_model/feature_detection/shi_tomasi_angle_sparse_bad_sinkhorn.py` - `ShiTomasiAngleSparseBADSinkhornMatcherWithFilters`クラス
- `onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.py` - エクスポートスクリプト

**ONNXエクスポート例**:
```bash
python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.py \
  --output model_with_filters.onnx \
  --max-keypoints 1024 \
  --ratio-threshold 2.0 \
  --dustbin-margin 0.3 \
  --height 480 \
  --width 640
```

**エクスポートされるモデル出力**:
- `keypoints1`: 画像1の特徴点 [B, K, 2]
- `keypoints2`: 画像2の特徴点 [B, K, 2]
- `matching_probs`: マッチング確率行列 [B, K+1, K+1]
- `valid_mask`: フィルタを通過したマッチのマスク [B, K] (boolean)

**ONNX推論例**:
```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model_with_filters.onnx")
results = session.run(None, {"image1": img1, "image2": img2})
keypoints1, keypoints2, matching_probs, valid_mask = results

# valid_maskを使って有効なマッチのみを抽出
valid_indices = np.where(valid_mask[0])[0]
filtered_kpts1 = keypoints1[0][valid_indices]
filtered_kpts2_indices = np.argmax(matching_probs[0][valid_indices, :-1], axis=1)
filtered_kpts2 = keypoints2[0][filtered_kpts2_indices]
```

**メリット**:
- ✅ エンドツーエンドでGPU/NPU上で実行可能
- ✅ 低レイテンシ（GPU→CPU転送不要）
- ✅ 単一のONNXモデルで完結

**デメリット**:
- ❌ パラメータ変更にはモデル再エクスポートが必要
- ❌ 実装がやや複雑

---

## フィルタの詳細

### 確率比フィルタ

**原理**: 各特徴点について、最良マッチの確率と次点マッチの確率の比を計算します。比が小さい場合、複数の候補に確率が分散しており、マッチが曖昧であることを示します。

**パラメータ**: `ratio_threshold` (推奨値: 2.0)
- `1.5`: 緩い（僅かに曖昧なマッチも許容）
- `2.0`: 標準（バランスの取れた厳しさ）
- `3.0`: 厳しい（非常に明確なマッチのみ採用）

**計算式**:
```
ratio = best_prob / second_best_prob
valid = ratio >= ratio_threshold
```

---

### ダストビンマージンフィルタ

**原理**: シンクホーンのダストビン機構を活用します。各特徴点について、最良マッチの確率とダストビン(非マッチ)の確率を比較し、ダストビン確率が高い点を除外します。

**パラメータ**: `dustbin_margin` (推奨値: 0.3)
- `0.2`: 緩い（ダストビン確率が少し高くても許容）
- `0.3`: 標準（バランスの取れた厳しさ）
- `0.5`: 厳しい（明確にマッチする点のみ採用）

**計算式**:
```
margin = best_match_prob - dustbin_prob
valid = margin >= dustbin_margin
```

---

## パラメータ推奨値

| シーン | ratio_threshold | dustbin_margin | 説明 |
|--------|-----------------|----------------|------|
| 高精度重視 | 3.0 | 0.5 | 厳格なフィルタで外れ値を最小化 |
| バランス型 | 2.0 | 0.3 | 推奨デフォルト設定 |
| マッチ数重視 | 1.5 | 0.2 | より多くのマッチを許容 |
| フィルタ無効 | None | None | フィルタリングを行わない |

---

## 実装の詳細

### NumPy版実装

**ファイル**: `pytorch_model/matching/outlier_filters.py`

```python
def probability_ratio_filter(P: np.ndarray, ratio_threshold: float = 2.0) -> np.ndarray:
    """確率比フィルタ"""
    # P: [K, K] 確率行列（ダストビン除外）
    # 戻り値: [K] boolean mask

def dustbin_margin_filter(P: np.ndarray, margin: float = 0.3) -> np.ndarray:
    """ダストビンマージンフィルタ"""
    # P: [K+1, K+1] 確率行列（ダストビン含む）
    # 戻り値: [K] boolean mask
```

### PyTorch版実装

**ファイル**: `pytorch_model/matching/sinkhorn.py`

```python
class SinkhornMatcherWithFilters(SinkhornMatcher):
    """フィルタ統合型シンクホーンマッチャ"""

    def __init__(
        self,
        iterations: int = 20,
        epsilon: float = 1.0,
        unused_score: float = 1.0,
        distance_type: str = "l2",
        ratio_threshold: float = None,  # Noneで無効化
        dustbin_margin: float = None,   # Noneで無効化
    ):
        ...

    def forward(self, desc1, desc2):
        # 戻り値: (P, valid_mask)
        # P: [B, N+1, M+1] 確率行列
        # valid_mask: [B, N] boolean tensor
```

---

## テスト

### NumPy版のテスト

```bash
# 確率比フィルタのみ
python sample/image_matching.py \
  --model model.onnx \
  --input1 test_img1.png \
  --input2 test_img2.png \
  --ratio-threshold 2.0

# ダストビンマージンフィルタのみ
python sample/image_matching.py \
  --model model.onnx \
  --input1 test_img1.png \
  --input2 test_img2.png \
  --dustbin-margin 0.3

# 両方のフィルタを併用
python sample/image_matching.py \
  --model model.onnx \
  --input1 test_img1.png \
  --input2 test_img2.png \
  --ratio-threshold 2.0 \
  --dustbin-margin 0.3
```

### PyTorch版のテスト

```bash
# ONNXエクスポート
python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn_with_filters.py \
  --output test_model_filters.onnx \
  --max-keypoints 512 \
  --ratio-threshold 2.0 \
  --dustbin-margin 0.3 \
  --height 480 \
  --width 640

# エクスポートされたモデルをONNX Runtimeでテスト
python -c "
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('test_model_filters.onnx')
img1 = np.random.randn(1, 1, 480, 640).astype(np.float32)
img2 = np.random.randn(1, 1, 480, 640).astype(np.float32)

results = session.run(None, {'image1': img1, 'image2': img2})
kpts1, kpts2, probs, valid = results

print(f'Keypoints1: {kpts1.shape}')
print(f'Keypoints2: {kpts2.shape}')
print(f'Matching probs: {probs.shape}')
print(f'Valid mask: {valid.shape}, dtype: {valid.dtype}')
print(f'Valid matches: {valid.sum()}/{valid.shape[1]}')
"
```

---

## 参考文献

1. **Sinkhorn Algorithm**:
   - Cuturi, M. "Sinkhorn distances: Lightspeed computation of optimal transport." NeurIPS 2013.

2. **SuperGlue (Dustbin Mechanism)**:
   - Sarlin et al. "SuperGlue: Learning Feature Matching with Graph Neural Networks." CVPR 2020.

3. **Lowe's Ratio Test (確率比フィルタの元となる考え方)**:
   - Lowe, D.G. "Distinctive image features from scale-invariant keypoints." IJCV 2004.

---

## 貢献

外れ値フィルタリング機能は以下のコミットで追加されました：
- 確率比フィルタ: commit 5f87ad7
- ダストビンマージンフィルタ: commit e5b9e05
- PyTorch/ONNX統合: commit (次回)
