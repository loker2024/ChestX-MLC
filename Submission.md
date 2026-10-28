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
