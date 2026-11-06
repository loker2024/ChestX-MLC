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
- 归档文件：[submission_B1-320-study_74.41.csv](outputs/predictions/submissions/submission_B1-320-study_74.41.csv)。
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
- 项目中的对应文件：[submission_densenet121_s2026.csv](outputs/predictions/submissions/submission_densenet121_s2026.csv)。
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
- 项目中的对应文件：[submission_viewweighted_s2026.csv](outputs/predictions/submissions/submission_viewweighted_s2026.csv)。
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

融合与提交生成命令（在训练环境 `/root/ChestX-MLC` 执行，`--out outputs/ensemble/test` 生成的 `submission.csv` 已下载并直接改名为 `outputs/predictions/submissions/submission_ensemble_3m_prob_211_rank1.csv` 保存）：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python src/run_ensemble.py \
  --checkpoint outputs/20261030-150749_A-001_convnext_tiny_s42/best.pt \
               outputs/20261030-140736_run009a_effv2s_384_s2026/best.pt \
               outputs/20261030-190645_A-001d_convnext_tiny_mean_s42/best.pt \
  --weights 2,1,1 --fusion probability --split test --out outputs/ensemble/test
```

### 提交记录

- 状态：已提交官方平台；官方得分 **76.82**（用户反馈）。
- 提交文件：[submission_ensemble_3m_prob_211_rank1.csv](outputs/predictions/submissions/submission_ensemble_3m_prob_211_rank1.csv)（`rank1` 标注 valid 排名第一，`211` 标注权重 2:1:1；原远程输出为 `outputs/ensemble/test/submission.csv`，下载后直接改名，无重复副本）。
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

- 状态：已提交官方平台；官方得分 **76.62**（用户反馈）。
- 提交文件：[submission_ensemble_3m_prob_111_rank2.csv](outputs/predictions/submissions/submission_ensemble_3m_prob_111_rank2.csv)（`rank2` 标注 valid 排名第二，`111` 标注等权；原远程输出为 `outputs/ensemble/test_w111/submission.csv`，下载后直接改名，无重复副本）。
- 文件内容：1,414 个 study、14,140 条预测（136 个 subject），包含 `Subject_id`、`Study_id`、`Predict_class`、`Probability` 四列；概率范围 0.001919–0.996157；训练环境内已通过 `validate_submission` 校验。
- SHA-256：`dcd6e4b5eb2097e418be565a99027b89545941a254e96786146093e7f3bdf6ab`。
- 文件位于 `outputs/`，按项目规则被 Git 忽略，仅保存在本地；训练环境 `/root/ChestX-MLC/outputs/ensemble/test_w111/` 保留原始文件名 `submission.csv`，本机改名后的文件与其 md5 一致（`5ebf49b66fef87dca0e1eef8462c58ed`），与 ENSEMBLE-3M 的 test 文件内容不同。

## D-002-a-EMA（单模型）

来源：本项目训练产物，未重新训练；checkpoint 内嵌完整训练配置，推理时直接以该配置为准。

### 实验名称

单模型提交：ConvNeXt-Tiny（`convnext_tiny.fb_in1k`）@320，`view_weighted` 正位权重 0.7，EMA 权重（decay 0.999）。

### 实验性能

验证集 Study 级 Macro-AUC **0.8100**、Macro-F1@0.5 **0.4283**（859 个 study）；阈值扫描峰值在 0.6，Macro-F1 0.4422。9 类口径 Macro-AUC **0.8196**。最佳 EMA 权重取自第 5 轮（训练上限 8 轮、早停 patience 2，实际 7 轮完训，34m57s）。

### 实验参数

- 模型与权重：`outputs/20261102-145203_D-002-a/best_ema.pt`（与 A-FUSE-5 成本测试所用同一权重，md5 `e97524bb…`）。
- 训练口径：seed 42、`train.ema_enabled: true` + decay 0.999、lr 3e-4、warmup 2、cosine 至 1e-5、weight decay 0.01、`pos_weight: true`、batch_studies 16、BCE。
- 推理口径：`image.size: 320`、letterbox、ImageNet 归一化、无 hflip；不设 `--threshold`，输出每个 study 全部 10 类概率。

生成命令（在训练环境 `/root/ChestX-MLC` 执行）：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python src/run_infer.py \
  --config outputs/20261102-145203_D-002-a/config.yaml \
  --checkpoint outputs/20261102-145203_D-002-a/best_ema.pt \
  --out outputs/submission_D-002-a-EMA/submission.csv
```

### 提交记录

- 状态：已提交官方平台；官方得分 **76.91**（根据用户反馈，提交日期为 2026-11-05）。
- 提交文件：[submission_D-002-a-EMA.csv](outputs/predictions/submissions/submission_D-002-a-EMA.csv)。
- 文件内容：1,414 个 study、14,140 条预测（136 个 subject），每 study 恰好 10 类；列名 `Subject_id`、`Study_id`、`Predict_class`、`Probability`；概率范围 0.000679–0.997329；无缺失；test study 集合与 `data/test.csv` 完全一致。
- SHA-256：`d28a872b852335e5b823e3468e2be8029b58f22b6beb6520e44224c1d5c45f29`；md5 `47c597ba59ba02a7497ab940b3f311bf`（与本机下载文件一致）。
- 文件位于 `outputs/`，按项目规则被 Git 忽略，仅保存在本地；训练环境 `/root/ChestX-MLC/outputs/submission_D-002-a-EMA/submission.csv` 保留原始文件名。

## A-FUSE-6（k=4 等权，穷举选出）

来源：本项目训练产物；由 `outputs/analysis/fusion_pool_790_800.json`（25 项候选池）经 `fusion_exhaustive.py` 全穷举选出。实验详情见 [训练记录](docs/training_log.md)。

### 实验名称

A-FUSE-6：ConvNeXt 分档门槛池的 4 模型等权概率融合（D-006-b-EMA + D-002-a-EMA + C-005-c + RUN-009-A）。

### 实验性能

验证集 Study 级 Macro-AUC **0.8168**、Macro-F1 **0.4425**（阈值 0.5），859 个 study。该值由 `run_ensemble.py --split valid` 在远端复算，与穷举搜索的 top-1 完全一致（0.8168 / 0.4425），无偏差。

### 实验参数

- 成员与权重（等权 1:1:1:1，`probability` 融合）：
  - `outputs/20261104-201457_D-006-b/best_ema.pt`（单模型 valid 0.8108，池内最高）
  - `outputs/20261102-145203_D-002-a/best_ema.pt`（0.8100）
  - `outputs/20261101-202325_C-005-c/best.pt`（0.8029，正侧位加权聚合口径）
  - `outputs/20261030-140736_run009a_effv2s_384_s2026/best.pt`（0.7945，EfficientNetV2-S @384）
- 候选池口径：ConvNeXt 骨干 ≥ 0.800、其他骨干 ≥ 0.790，非重复、非 B/T 系列、非 ResNet，且同实验有 EMA 时不取 raw。
- 搜索规模：C(25,2..5) 共 68,380 组等权组合，prob 空间；k=4 的 top-1 即本方案。

### 实验分析

- 相对 A-FUSE-2（A-001 + C-005-c + RUN-009-A，0.8148）**+0.0020**；相对单模型最好 D-006-b-EMA（0.8108）**+0.0060**。
- 相对 A-FUSE-5（k3 加权 4:2:3，0.8171）**−0.0003**，即等权穷举没打过已有的加权方案；差值远小于单次 macro-AUC 的 bootstrap SE（≈0.0094），不构成性能差异证据。
- **k 的有效上限是 4**：k=5 穷举 53,130 组后的 top-1 与 k=4 完全相同（0.8168）。前 20 名区间宽度随 k 单调收窄（0.0022 → 0.0014 → 0.0008 → 0.0003），说明 k 增大只是把更多组合推上同一个平台。
- `C-005-c` 与 `RUN-009-A` 在 k≥3 的 top5 中 100% 出现，是稳定增益来源；`RUN-009-A` 单模型仅 0.7945，靠分数筛选永远进不了池，是「异构互补」而非「强者叠加」的又一例证。
- 选型偏差：top-1 是从 12,650 组里挑出的最高值，属带选择偏差的点估计，不能当作无偏性能预期；test 侧无标签，融合增益无法验证。

### 提交记录

- 状态：**已生成提交文件，尚未提交官方平台**。
- 提交文件：[submission_A-FUSE-6_k4_...csv](outputs/predictions/submissions/submission_A-FUSE-6_k4_D006bEMA-D002aEMA-C005c-RUN009A_prob_w1111.csv)。
- 文件内容：1,414 个 study、14,140 条预测，每 study 恰好 10 类；列名 `Subject_id`、`Study_id`、`Predict_class`、`Probability`；概率范围 0.001579–0.996250；无缺失；test study 集合与 `data/test.csv` 完全一致。
- SHA-256：`407d29faeacb8cac6163a31abc5c2ed17b6ff145473808165a4f72d53d2447bc`；md5 `9dfcf8fb621a455dea184207faf456dc`（与训练环境文件一致）。
- 耗时自检（`src/cost.py`，V100-SXM2-32GB，fp32，3 轮，100 study / 162 图）：单 study 均值 **93.6 ms**，中位数 91.7 ms，p95 118.0 ms，最大 134.5 ms，**未超过 100 ms 上限但余量仅 6.3%**；初始化 58.8 ms（仅元数据）；预测一致性命中 100/100 study、最大概率差 4.6e-07。
- 权重合计 **404.1 MB**（赛题上限 500 MB，余量 95.9 MB）。
- fp16 备选实测：92.1 ms（仅快 1.6 ms），但与 fp32 提交比对有 **14 个 study 不一致**（最大差 0.0020 > 1e-3 容差），且需在提交说明声明——**不采用**。耗时瓶颈在 JPEG 解码与文件 IO，不在前向。
- 清单模式复测（不带 `--checkpoint`，靠 `cost_ensemble.json` 自动发现模型与权重）：**94.3 ms**（vs 显式指定时 93.6 ms，同口径波动），一致性仍为 100/100 study、最大概率差 4.6e-07；`cost_views.csv` 被自动识别并生效。报告 `cost_run/k4_790_800/cost_report_manifest.json`。
- 提交包必需件：`cost_ensemble.json`（`fusion: probability`、`weights: [1,1,1,1]`、四个 checkpoint 用**相对清单目录**的路径 `../../<run>/best_ema.pt`，故打包时须保持 `outputs/` 目录结构不变）+ `cost_views.csv`（三列元数据 `Study_id/Dicom_id/ViewPosition`，2,218 行 / 1,414 study，取自 `data/test.csv`，不含像素）。四个成员均为 `view_weighted` 聚合，缺该文件会回退等权均值并导致一致性失败。
- 报告：`cost_run/k4_790_800/cost_report.json`、`cost_report_manifest.json`；文件位于 `outputs/`，按项目规则被 Git 忽略，仅保存在本地。
