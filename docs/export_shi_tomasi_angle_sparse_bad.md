# Shi-Tomasi + Angle + Sparse BAD ONNX Export

このドキュメントでは、Shi-Tomasi特徴検出、角度推定、回転不変Sparse BAD記述子を組み合わせたモデルのONNXエクスポート方法を説明します。

## エクスポートスクリプト

2つのエクスポートスクリプトが用意されています：

### 1. 記述子計算のみ (`export_shi_tomasi_angle_sparse_bad.py`)

単一画像から特徴点を検出し、回転不変記述子を計算します。

**入力:**
- `image`: (B, 1, H, W) グレースケール画像

**出力:**
- `keypoints`: (B, K, 2) キーポイント座標 (y, x) 形式
- `scores`: (B, K) キーポイントスコア
- `descriptors`: (B, K, num_pairs) 回転不変記述子

**使用例:**
```bash
# 基本的な使用
python onnx_export/export_shi_tomasi_angle_sparse_bad.py \
    --output shi_tomasi_angle_sparse_bad.onnx

# カスタムパラメータ
python onnx_export/export_shi_tomasi_angle_sparse_bad.py \
    --output model.onnx \
    --height 480 \
    --width 640 \
    --max-keypoints 512 \
    --num-pairs 256 \
    --binarization soft \
    --temperature 10.0
```

### 2. 完全なマッチングパイプライン (`export_shi_tomasi_angle_sparse_bad_sinkhorn.py`)

2つの画像間の特徴マッチングを実行します（Sinkhornマッチャー付き）。

**入力:**
- `image1`: (B, 1, H, W) 最初のグレースケール画像
- `image2`: (B, 1, H, W) 2番目のグレースケール画像

**出力:**
- `keypoints1`: (B, K, 2) 最初の画像のキーポイント
- `keypoints2`: (B, K, 2) 2番目の画像のキーポイント
- `matching_probs`: (B, K+1, K+1) マッチング確率行列

**使用例:**
```bash
# 基本的な使用
python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn.py \
    --output shi_tomasi_angle_sparse_bad_sinkhorn.onnx

# カスタムパラメータ
python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn.py \
    --output model.onnx \
    --height 480 \
    --width 640 \
    --max-keypoints 1024 \
    --num-pairs 512 \
    --binarization soft \
    --sinkhorn-iterations 20 \
    --epsilon 0.05
```

## パラメータ一覧

### 共通パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--output`, `-o` | モデル名.onnx | 出力ONNXファイルパス |
| `--height`, `-H` | 480 | 入力画像の高さ |
| `--width`, `-W` | 640 | 入力画像の幅 |
| `--max-keypoints`, `-k` | 1024 | 画像あたりの最大キーポイント数 |

### Shi-Tomasiパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--block-size` | 5 | Shi-Tomasiブロックサイズ |

### 角度推定パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--patch-size` | 15 | 角度推定パッチサイズ |
| `--sigma` | 2.5 | 角度推定のガウシアンシグマ |

### BAD記述子パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--num-pairs`, `-n` | 256 | BAD記述子のペア数 (256 or 512) |
| `--binarization` | none | 二値化モード: none, soft, hard |
| `--temperature` | 10.0 | ソフト二値化の温度パラメータ |
| `--normalize-descriptors` | True | 記述子のL2正規化 |
| `--no-normalize-descriptors` | - | 正規化を無効化 |
| `--sampling-mode` | nearest | サンプリングモード: nearest, bilinear |

### Sinkhornパラメータ（マッチング版のみ）

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--sinkhorn-iterations`, `-i` | 20 | Sinkhornイテレーション数 |
| `--epsilon`, `-e` | 0.05 | エントロピー正則化パラメータ |
| `--unused-score` | 1.0 | ダストビンエントリのスコア |
| `--distance-type` | l2 | 距離メトリック: l1, l2 |

### パイプラインパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--nms-radius` | 3 | Non-Maximum Suppression半径 |
| `--score-threshold` | 0.0 | キーポイント選択の最小スコア閾値 |

### ONNXエクスポートオプション

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `--opset-version` | 18 | ONNXオペセットバージョン |
| `--dynamic-axes` | False | 動的入力形状を有効化 |
| `--disable-dynamo` | False | Dynamoを無効化 |
| `--no-optimize` | False | ONNX最適化を無効化 |

## 推奨設定

### 高精度マッチング（低速）
```bash
python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn.py \
    --output high_accuracy.onnx \
    --max-keypoints 2048 \
    --num-pairs 512 \
    --binarization soft \
    --temperature 10.0 \
    --sinkhorn-iterations 100 \
    --epsilon 0.01 \
    --normalize-descriptors
```

### バランス型（推奨）
```bash
python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn.py \
    --output balanced.onnx \
    --max-keypoints 1024 \
    --num-pairs 256 \
    --binarization soft \
    --temperature 10.0 \
    --sinkhorn-iterations 20 \
    --epsilon 0.05 \
    --normalize-descriptors
```

### 高速マッチング（リアルタイム）
```bash
python onnx_export/export_shi_tomasi_angle_sparse_bad_sinkhorn.py \
    --output fast.onnx \
    --max-keypoints 512 \
    --num-pairs 256 \
    --binarization hard \
    --sinkhorn-iterations 10 \
    --epsilon 0.1 \
    --normalize-descriptors
```

## ONNXモデルの使用例（Python）

### 記述子計算のみ

```python
import onnxruntime as ort
import numpy as np

# ONNXモデルをロード
session = ort.InferenceSession("shi_tomasi_angle_sparse_bad.onnx")

# 入力画像（グレースケール、0-1正規化済み）
image = np.random.randn(1, 1, 480, 640).astype(np.float32)

# 推論実行
keypoints, scores, descriptors = session.run(
    None,
    {"image": image}
)

print(f"Keypoints shape: {keypoints.shape}")      # (1, K, 2)
print(f"Scores shape: {scores.shape}")            # (1, K)
print(f"Descriptors shape: {descriptors.shape}")  # (1, K, 256)
```

### マッチングパイプライン

```python
import onnxruntime as ort
import numpy as np

# ONNXモデルをロード
session = ort.InferenceSession("shi_tomasi_angle_sparse_bad_sinkhorn.onnx")

# 入力画像（グレースケール、0-1正規化済み）
image1 = np.random.randn(1, 1, 480, 640).astype(np.float32)
image2 = np.random.randn(1, 1, 480, 640).astype(np.float32)

# 推論実行
keypoints1, keypoints2, matching_probs = session.run(
    None,
    {"image1": image1, "image2": image2}
)

print(f"Keypoints1 shape: {keypoints1.shape}")        # (1, K, 2)
print(f"Keypoints2 shape: {keypoints2.shape}")        # (1, K, 2)
print(f"Matching probs shape: {matching_probs.shape}")  # (1, K+1, K+1)

# マッチング結果を取得（最も確率の高いマッチング）
matches = np.argmax(matching_probs[0, :-1, :-1], axis=1)
match_confidences = np.max(matching_probs[0, :-1, :-1], axis=1)

# 有効なマッチングのみフィルタリング
valid_mask = match_confidences > 0.5
valid_matches = matches[valid_mask]
valid_kp1 = keypoints1[0, valid_mask]
valid_kp2 = keypoints2[0, valid_matches]

print(f"Valid matches: {np.sum(valid_mask)}")
```

## C++での使用例

```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// ONNXランタイムセッション作成
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ShiTomasiAngleSparseBAD");
Ort::SessionOptions session_options;
Ort::Session session(env, "shi_tomasi_angle_sparse_bad_sinkhorn.onnx", session_options);

// 入力データ準備
cv::Mat img1_gray, img2_gray;
cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

// 正規化 (0-1範囲)
img1_gray.convertTo(img1_gray, CV_32F, 1.0/255.0);
img2_gray.convertTo(img2_gray, CV_32F, 1.0/255.0);

// ONNXテンソルに変換
std::vector<int64_t> input_shape = {1, 1, img1_gray.rows, img1_gray.cols};
std::vector<float> input1(img1_gray.begin<float>(), img1_gray.end<float>());
std::vector<float> input2(img2_gray.begin<float>(), img2_gray.end<float>());

// 推論実行
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value input_tensor1 = Ort::Value::CreateTensor<float>(
    memory_info, input1.data(), input1.size(),
    input_shape.data(), input_shape.size()
);
Ort::Value input_tensor2 = Ort::Value::CreateTensor<float>(
    memory_info, input2.data(), input2.size(),
    input_shape.data(), input_shape.size()
);

const char* input_names[] = {"image1", "image2"};
const char* output_names[] = {"keypoints1", "keypoints2", "matching_probs"};

auto outputs = session.Run(
    Ort::RunOptions{nullptr},
    input_names,
    &input_tensor1,
    2,
    output_names,
    3
);

// 結果取得
auto* keypoints1 = outputs[0].GetTensorMutableData<float>();
auto* keypoints2 = outputs[1].GetTensorMutableData<float>();
auto* matching_probs = outputs[2].GetTensorMutableData<float>();
```

## AKAZEとの比較

| 特徴 | Shi-Tomasi + Angle | AKAZE |
|------|-------------------|-------|
| 検出速度 | ⚡⚡⚡ 速い | ⚡ 遅い |
| マルチスケール | ❌ なし（単一スケール） | ✅ あり |
| 回転不変性 | ✅ 角度推定による | ✅ ネイティブサポート |
| スケール不変性 | ❌ なし | ✅ あり |
| 記述子 | Sparse BAD (回転補償) | Sparse BAD (回転補償) |
| 推奨用途 | 高速マッチング、平面物体 | 高精度、3Dマッチング |

## トラブルシューティング

### エクスポートが遅い場合
- `--disable-dynamo` オプションを試す
- `--no-optimize` で最適化をスキップ（開発時）

### メモリ不足
- `--max-keypoints` を減らす
- `--height`, `--width` を小さくする

### 精度が低い場合
- `--num-pairs 512` で記述子次元を増やす
- `--sinkhorn-iterations` を増やす
- `--epsilon` を小さくする（例: 0.01）
- `--binarization soft` を使用

### 速度が遅い場合
- `--max-keypoints` を減らす（例: 512）
- `--num-pairs 256` を使用
- `--binarization hard` を使用
- `--sinkhorn-iterations` を減らす（例: 10）

## 関連リンク

- [角度推定ドキュメント](angle_estimation.md)
- [BAD記述子ドキュメント](../pytorch_model/descriptor/bad.py)
- [Sinkhornマッチャードキュメント](../pytorch_model/matching/sinkhorn.py)
- [AKAZE版エクスポート](../onnx_export/export_akaze_sparse_bad_sinkhorn.py)
