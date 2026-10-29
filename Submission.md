# 提交记录

## B1-320-study

来源：外部提供的实验产物，并非本项目制作；实验记录来自桌面 `工作簿1.xlsx` 的 `Sheet1` 第 18 行。

### 实验名称

B1-320-Study：DenseNet121 + 320×320 输入 + Study 均衡 BCE。

### 实验性能

验证集 Study 级 Macro-AUC **0.795467**，Macro-F1 **0.293981**（阈值 0.5）；最佳 epoch 为第 6 轮，实际训练 9 轮。

### 实验参数

DenseNet121、ImageNet 预训练（权重版本待核实）；320×320 输入、Study 均衡 BCE；AdamW，backbone 学习率 `1e-4`、head 学习率 `1e-3`；2 轮 warmup 后使用 cosine 学习率衰减，并启用早停。训练集 18,513 个 study、30,344 张图；验证集 859 个 study。

### 实验分析

第 6 轮 AUC 最高，相较 B1-224 的 0.785445 提高 0.010022，相较 RUN-001 ResNet-18 320 的 0.785769 提高 0.009698。第 7–9 轮训练 loss 继续下降、验证 loss 上升，出现过拟合迹象；保留第 6 轮最佳权重。分辨率和 Study 均衡 BCE 同时变化，尚不能区分各自的贡献。

### 提交记录

- 官方提交得分：**74.41**（根据用户提供的信息记录）。
- 原文件名：`submission(2).csv`。
- 归档文件：[submission_B1-320-study_74.41.csv](outputs/submission_B1-320-study_74.41.csv)。
- 文件内容：1,414 个 study、14,140 条预测，包含 `Subject_id`、`Study_id`、`Predict_class`、`Probability` 四列。
- SHA-256：`2bbe6cf13e26bca48c0bba16742ba842aba0b0b415cdaee1de4025a74ac115d3`。
- 归档文件位于 `outputs/`，按项目规则被 Git 忽略，仅保存在本地。

## RUN-006-A

来源：本项目训练产物；实验详情见 [训练记录](docs/training_log.md)。

### 实验性能

验证集 Study 级 Macro-AUC **0.7913**、Macro-F1 **0.4103**（阈值 0.5）；最佳 epoch 为第 5 轮。

### 实验参数

DenseNet-121（`densenet121.ra_in1k`）、320×320 输入、`pos_weight: true`、cosine 学习率调度、seed 2026；同一 study 的图片 logits 等权聚合。

### 提交记录

- 官方提交得分：**74.09**；状态：已正确评分（根据用户提供的截图记录）。
- 提交 ID：`24767`；提交时间：2026-10-29 19:45:34。
- 上传文件名：`submission_RUN-006-A.csv`。
- 项目中的对应文件：[submission_densenet121_s2026.csv](outputs/submission_densenet121_s2026.csv)。
- 文件内容：1,414 个 study、14,140 条预测，包含 `Subject_id`、`Study_id`、`Predict_class`、`Probability` 四列。
- SHA-256：`b6957a111ac5844890a7d18955c1519977ee864fbe8ae02c6d30af14c82fb479`。
- 文件位于 `outputs/`，按项目规则被 Git 忽略，仅保存在本地。

## RUN-007-A

来源：本项目训练产物；实验详情见 [训练记录](docs/training_log.md)。

### 实验性能

验证集 Study 级 Macro-AUC **0.7906**、Macro-F1 **0.4198**（阈值 0.5）；最佳 epoch 为第 7 轮。

### 实验参数

DenseNet-121（`densenet121.ra_in1k`）、320×320 输入、`pos_weight: true`、cosine 学习率调度、seed 2026；正侧位 logits 按 7:3 加权聚合，单视角或含未知视角时回退等权均值。

### 提交记录

- 官方提交得分：**74.82**；状态：已正确评分（根据用户提供的截图记录）。
- 提交 ID：`24785`；提交时间：2026-10-29 19:57:22。
- 上传文件名：`submission_RUN-007-A.csv`。
- 项目中的对应文件：[submission_viewweighted_s2026.csv](outputs/submission_viewweighted_s2026.csv)。
- 文件内容：1,414 个 study、14,140 条预测，包含 `Subject_id`、`Study_id`、`Predict_class`、`Probability` 四列。
- SHA-256：`c99f3ab85468cc8cc818ee8c21bf674b2266bce6eb7c392217ed2fc4a85e3ab4`。
- 文件位于 `outputs/`，按项目规则被 Git 忽略，仅保存在本地。

两次官方得分分别为 74.09 和 74.82，与验证集 Macro-AUC 是不同的记录口径；仅凭两次提交无法确定得分差异的原因。
