# Angle Estimation Module

## Overview

角度推定モジュールは、Shi-Tomasiなどの特徴点検出器で検出された特徴点に対して、支配的な方向（角度）を計算するための単独モジュールです。

## 実装の詳細

### アルゴリズム

AKAZEの実装を参考にした以下のアプローチを使用：

1. **ガウシアン重み付け**: 特徴点の近傍ピクセルに対してガウシアンカーネルを適用
   ```
   w(x,y) = exp(-(x² + y²) / (2σ²))
   ```

2. **モーメント計算**: 重み付き強度モーメントを計算
   ```
   m10 = Σ(x * w(x,y) * I(x,y))  # x方向モーメント
   m01 = Σ(y * w(x,y) * I(x,y))  # y方向モーメント
   ```

3. **角度算出**: 逆正接（atan2）を使用して角度を計算
   ```
   θ = atan2(m01, m10)  # 範囲: [-π, π]
   ```

### モジュール構成

```
pytorch_model/
└── orientation/
    ├── __init__.py
    └── angle_estimation.py
        ├── AngleEstimator           # 基本的な角度推定
        └── AngleEstimatorMultiScale # マルチスケール版（実験的）
```

## 使用方法

### 基本的な使用例

```python
from pytorch_model.orientation.angle_estimation import AngleEstimator

# 角度推定器の作成
estimator = AngleEstimator(
    patch_size=15,  # パッチサイズ（奇数）
    sigma=2.5       # ガウシアンの標準偏差
)

# 画像の角度マップを計算
angles = estimator(image)  # 入力: (N, 1, H, W), 出力: (N, 1, H, W)
```

### Shi-Tomasiとの統合

```python
from pytorch_model.corner.shi_tomasi import ShiTomasiScore
from pytorch_model.orientation.angle_estimation import AngleEstimator

# 特徴点検出
detector = ShiTomasiScore(block_size=5)
scores = detector(image)

# 角度推定
angle_estimator = AngleEstimator(patch_size=15, sigma=2.5)
angles = angle_estimator(image)

# Top-Kの特徴点を選択
k = 100
scores_flat = scores.view(1, -1)
_, indices = torch.topk(scores_flat, k, dim=1)

# 対応する角度を取得
angles_flat = angles.view(1, -1)
selected_angles = angles_flat.gather(1, indices)
```

### 統合モジュールの使用

```python
from pytorch_model.feature_detection.shi_tomasi_angle import ShiTomasiWithAngle

# Shi-Tomasi + 角度推定の統合モジュール
detector = ShiTomasiWithAngle(
    block_size=5,
    patch_size=15,
    sigma=2.5
)

# 一度の呼び出しでスコアと角度を取得
scores, angles = detector(image)
```

## パイプライン統合

本モジュールは以下のパイプラインの一部として使用することを想定しています：

```
Shi-Tomasi → 角度推定 → Sparse BAD → Sinkhorn
```

### AKAZEとの比較

| モジュール | 特徴点検出 | 角度計算 | スケール |
|-----------|-----------|---------|---------|
| AKAZE | Hessian行列式 | モーメント法 | マルチスケール |
| Shi-Tomasi + AngleEstimator | 最小固有値 | モーメント法 | シングルスケール |

どちらも同じ出力形式 `(scores, angles)` を提供するため、互換性があります。

## パラメータ

### `AngleEstimator`

- **`patch_size`** (int, default=15): 角度計算に使用する局所パッチのサイズ（奇数である必要があります）
  - 小さい値: より局所的な方向、ノイズに敏感
  - 大きい値: より安定した方向、局所性が低下

- **`sigma`** (float, default=2.5): ガウシアン重みの標準偏差
  - 小さい値: 中心ピクセルにより大きな重み
  - 大きい値: より広い範囲のピクセルに重みを分散

## ONNX互換性

本モジュールは完全にONNXエクスポート可能です：

```python
import torch

model = AngleEstimator(patch_size=15, sigma=2.5)
model.eval()

dummy_input = torch.randn(1, 1, 480, 640)

torch.onnx.export(
    model,
    dummy_input,
    "angle_estimator.onnx",
    input_names=['image'],
    output_names=['angles'],
    dynamic_axes={
        'image': {0: 'batch', 2: 'height', 3: 'width'},
        'angles': {0: 'batch', 2: 'height', 3: 'width'}
    },
    opset_version=17
)
```

### ONNX互換性の特徴

- ✅ 動的制御フロー無し（if文やループ無し）
- ✅ 純粋なテンソル演算のみ使用
- ✅ 固定グラフ構造
- ✅ TensorRTと互換性あり

## 出力形式

### 角度の範囲と方向

出力される角度は `-π` から `π` の範囲のラジアン値です：

```
     -π/2 (上)
        ↑
        |
-π ←----+----→ 0 (右)
        |
        ↓
      π/2 (下)
```

- `0 rad` (0°): 右方向（正のx軸）
- `π/2 rad` (90°): 下方向（正のy軸）
- `-π/2 rad` (-90°): 上方向（負のy軸）
- `±π rad` (180°): 左方向（負のx軸）

## 実装詳細

### 効率的な実装

1. **融合カーネル**: 単一の2出力チャネル畳み込みでm10とm01を同時計算
2. **並列処理**: 全ピクセルの角度を並列計算
3. **GPU対応**: 完全にGPUアクセラレーション可能

### 計算量

- 畳み込み演算: `O(N * H * W * patch_size²)`
- atan2演算: `O(N * H * W)`
- 合計: `O(N * H * W * patch_size²)`

## 今後の拡張

### 回転不変記述子との統合

角度情報を使用してBAD記述子を回転不変にする例：

```python
# 角度に基づいてBADのペアオフセットを回転
cos_theta = torch.cos(angles)
sin_theta = torch.sin(angles)

rotated_x = offset_x * cos_theta - offset_y * sin_theta
rotated_y = offset_x * sin_theta + offset_y * cos_theta

# 回転されたオフセットで記述子を計算
# (akaze_sparse_bad_sinkhorn.pyの実装を参照)
```

## 参考文献

- AKAZE: Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces (BMVC 2013)
- Shi-Tomasi: Good Features to Track (CVPR 1994)

## ライセンス

このプロジェクトのライセンスに従います。
