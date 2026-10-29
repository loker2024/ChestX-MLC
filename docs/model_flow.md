# 模型流程图

## ResNet-18 @320

基于 [`configs/default.yaml`](../configs/default.yaml)。模型以一次检查（study）为预测单位，同一 study 的图片共享一组标签。

```text
                  One study (one or more images)
                              |
                              v
                  +-------------------------+
                  | Load images as RGB      |
                  +------------+------------+
                               |
                               v
                  +-------------------------+
                  | Letterbox: 320 x 320   |
                  | Training only: affine   |
                  | and color augmentation  |
                  | Tensor + ImageNet norm  |
                  +------------+------------+
                               |
                               v
                  +-------------------------+
                  | ResNet-18 per image     |
                  | Conv -> residual blocks |
                  | -> global average pool  |
                  | -> Linear(512, 10)      |
                  +------------+------------+
                               |
                               v
                  10 logits for each image
                               |
                               v
                  +-------------------------+
                  | Mean logits over images |
                  | belonging to the study  |
                  +------------+------------+
                               |
                               v
                  10 logits for the study
                               |
               +---------------+---------------+
               |                               |
               v                               v
     +---------------------+         +---------------------+
     | Training            |         | Validation / test   |
     | logits + labels     |         | Sigmoid per class   |
     | BCEWithLogitsLoss   |         | -> 10 probabilities |
     | -> backpropagation  |         +----------+----------+
     | -> AdamW update     |                    |
     +---------------------+         +----------+----------+
                                     |                     |
                                     v                     v
                              Validation metrics    submission.csv
                              Macro-AUC / F1        all 10 classes
```

- 默认训练图像增强包括随机旋转、平移、缩放以及亮度和对比度调整；默认不水平翻转。验证与推理不使用随机增强。
- 同一 study 的多张图**先聚合 logits，再执行 Sigmoid**；本图展示默认图片等权均值。正侧位加权实验及未知视角回退规则见 [调优记录](experiments.md#正侧位加权聚合配置)。训练使用 `BCEWithLogitsLoss`，不先手动执行 Sigmoid；默认配置启用 `pos_weight`。
- 默认最多训练 25 轮，按验证集 Macro-AUC 选择最优模型，连续 5 轮无提升时早停。推理通常加载 `best.pt`，默认输出全部 10 类概率。
- 模型与聚合见 [`src/model/network.py`](../src/model/network.py)，预处理见 [`src/data/transforms.py`](../src/data/transforms.py)，推理见 [`src/infer/predict.py`](../src/infer/predict.py)。
