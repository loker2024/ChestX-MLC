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

## ENSEMBLE-3M

来源：本项目训练产物的融合，未重新训练；三个 checkpoint 均内嵌完整训练配置。

### 实验名称

多模型融合：A-001（ConvNeXt-Tiny @320，`view_weighted`）+ RUN-009-A（EfficientNetV2-S RW @384）+ A-001-D（ConvNeXt-Tiny @320，`mean` 聚合）。

### 实验性能

验证集 Study 级 Macro-AUC **0.8104**、Macro-F1@0.5 **0.4389**（859 个 study）；阈值 0.3 / 0.4 / 0.5 / 0.6 / 0.7 的 Macro-F1 分别为 0.3970 / 0.4221 / 0.4389 / 0.4422 / 0.4348。逐类 AUC 10/10 均高于最强单模型 A-001（0.8038），平均增益 +0.0066。

### 实验参数

- 模型与权重：A-001 = 2（归一化 0.5）、RUN-009-A = 1（0.25）、A-001-D = 1（0.25）。
- checkpoint（训练环境 `/root/ChestX-MLC/` 下）：`outputs/20261030-150749_A-001_convnext_tiny_s42/best.pt`、`outputs/20261030-140736_run009a_effv2s_384_s2026/best.pt`、`outputs/20261030-190645_A-001d_convnext_tiny_mean_s42/best.pt`。
- fusion：`probability`，即先对每个模型的 study logits 取 sigmoid，再按归一化权重加权平均。
- 选择依据：valid 上共比较 13 组融合（2 模型等权、权重 1:3 / 3:1、probability / logit，及 3 模型对照）；本方案 Macro-AUC 最高、逐类 AUC 全胜单模型，Macro-F1 高于 2 模型最佳（0.4286）。完整对比数据在训练环境 `outputs/ensemble/valid/` 各目录的 `metrics.json`、`per_class.csv`、`f1_curve.csv`。

融合与提交生成命令（在训练环境 `/root/ChestX-MLC` 执行，`--out outputs/ensemble/test` 生成的 `submission.csv` 已下载并直接改名为 `outputs/submission_ensemble_3m_prob_211_rank1.csv` 保存）：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python src/run_ensemble.py \
  --checkpoint outputs/20261030-150749_A-001_convnext_tiny_s42/best.pt \
               outputs/20261030-140736_run009a_effv2s_384_s2026/best.pt \
               outputs/20261030-190645_A-001d_convnext_tiny_mean_s42/best.pt \
  --weights 2,1,1 --fusion probability --split test --out outputs/ensemble/test
```

### 提交记录

- 状态：尚未提交官方平台（候选提交，等待正式提交）。
- 提交文件：[submission_ensemble_3m_prob_211_rank1.csv](outputs/submission_ensemble_3m_prob_211_rank1.csv)（`rank1` 标注 valid 排名第一，`211` 标注权重 2:1:1；原远程输出为 `outputs/ensemble/test/submission.csv`，下载后直接改名，无重复副本）。
- 文件内容：1,414 个 study、14,140 条预测（136 个 subject），包含 `Subject_id`、`Study_id`、`Predict_class`、`Probability` 四列；概率范围 0.002269–0.996524；训练环境内已通过 `validate_submission` 校验。
- SHA-256：`52a47fc31892386221fa63c371d49e958dd375a1e03957fc5d85b7ab5c8d98d3`。
- 文件位于 `outputs/`，按项目规则被 Git 忽略，仅保存在本地；训练环境 `/root/ChestX-MLC/outputs/ensemble/test/` 保留原始文件名 `submission.csv`，本机改名后的文件与其 md5 一致（`c388fb5ba76f61285d1b7286f808eb5a`）。

## ENSEMBLE-3M-R2

来源：本项目训练产物的融合，未重新训练；与 ENSEMBLE-3M 使用相同三个 checkpoint，差别只在权重（1:1:1 等权），为 valid 融合矩阵的**第二名**。

### 实验名称

多模型融合（等权版）：A-001（ConvNeXt-Tiny @320，`view_weighted`）+ RUN-009-A（EfficientNetV2-S RW @384）+ A-001-D（ConvNeXt-Tiny @320，`mean` 聚合），probability 等权融合。

### 实验性能

验证集 Study 级 Macro-AUC **0.8100**、Macro-F1@0.5 **0.4409**（859 个 study）；阈值 0.3 / 0.4 / 0.5 / 0.6 / 0.7 的 Macro-F1 分别为 0.4032 / 0.4237 / 0.4409 / 0.4440 / 0.4312。逐类 AUC 9/10 高于最强单模型 A-001（仅 c1 略低：0.7389 vs 0.7438），平均增益 +0.0062。valid 融合矩阵排名第 2（第 1 名 ENSEMBLE-3M：AUC 0.8104 / F1 0.4389）。

### 实验参数

- 模型与权重：A-001 = 1、RUN-009-A = 1、A-001-D = 1（归一化各 1/3）。
- checkpoint：同 ENSEMBLE-3M 的三个 `best.pt`（训练环境 `/root/ChestX-MLC/` 下）。
- fusion：`probability`。
- 选择依据：valid 共比较 13 组融合，本方案 Macro-AUC 第二（0.8100，比第 1 名低 0.0004，在噪声范围内），Macro-F1@0.5 反而更高（0.4409 vs 0.4389）；作为备选提交保留。

生成命令（在训练环境 `/root/ChestX-MLC` 执行）：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python src/run_ensemble.py \
  --checkpoint outputs/20261030-150749_A-001_convnext_tiny_s42/best.pt \
               outputs/20261030-140736_run009a_effv2s_384_s2026/best.pt \
               outputs/20261030-190645_A-001d_convnext_tiny_mean_s42/best.pt \
  --weights 1,1,1 --fusion probability --split test --out outputs/ensemble/test_w111
```

### 提交记录

- 状态：备选提交（valid 第二名），尚未提交官方平台。
- 提交文件：[submission_ensemble_3m_prob_111_rank2.csv](outputs/submission_ensemble_3m_prob_111_rank2.csv)（`rank2` 标注 valid 排名第二，`111` 标注等权；原远程输出为 `outputs/ensemble/test_w111/submission.csv`，下载后直接改名，无重复副本）。
- 文件内容：1,414 个 study、14,140 条预测（136 个 subject），包含 `Subject_id`、`Study_id`、`Predict_class`、`Probability` 四列；概率范围 0.001919–0.996157；训练环境内已通过 `validate_submission` 校验。
- SHA-256：`dcd6e4b5eb2097e418be565a99027b89545941a254e96786146093e7f3bdf6ab`。
- 文件位于 `outputs/`，按项目规则被 Git 忽略，仅保存在本地；训练环境 `/root/ChestX-MLC/outputs/ensemble/test_w111/` 保留原始文件名 `submission.csv`，本机改名后的文件与其 md5 一致（`5ebf49b66fef87dca0e1eef8462c58ed`），与 ENSEMBLE-3M 的 test 文件内容不同。
