# 训练计划 30：Distill-1 固定融合教师蒸馏实验

状态：**已执行（2026-11-04），主比较未达参考线（+0.0017，CI 跨 0）**。目标是判断固定融合教师能否改善现有单模型，并评估性能与推理耗时的取舍。实现机制详见 [train_plan29.md](train_plan29.md)；本计划使用 Distill-1 编号，取代此前计划中的 D-006-a/b，不把同一实验登记两次。

## 一、实验设计

| 编号 | 配置 | 学生 | 教师损失 | 作用 |
| --- | --- | --- | --- | --- |
| Distill-1-a | `configs/exp_Distill1a.yaml` | ConvNeXt-Tiny | 关闭 | 同代码、同预算基线 |
| Distill-1-b | `configs/exp_Distill1b.yaml` | ConvNeXt-Tiny | 开启，λ=0.5、T=1 | 蒸馏对照 |

两份 YAML 独立完整，无 `_base`；展开后唯一差异是 `train.distill_enabled`。两组均从 `convnext_tiny.fb_in1k` ImageNet 权重与新分类头开始，seed 42，320，batch 16 studies，lr 3e-4，warmup 2、cosine 8 轮、`lr_min=1e-5`、patience 2，BCE + `pos_weight`，EMA decay 0.999，亮度/对比度增强均 0.15，视角聚合正位/侧位 0.7/0.3。实际早停按 **raw valid Macro-AUC**，而 `best_ema.pt` 是已训练轮次中的 EMA 最优，主比较固定为两组 `best_ema.pt`；同时报告 raw 的 `best.pt`。不从教师或既有胸片 checkpoint 热启动学生。

教师是 A-FUSE-5：依次使用 D-002-a `best_ema.pt`、C-005-c `best.pt`、E-001 `best.pt`；各自按 checkpoint 配置做 study 聚合，再以裁剪后 logit 4:2:3 融合，教师固定。每个类别的损失为原加权硬标签 BCE，加 `0.5 ×` 无 `pos_weight` 的软目标 BCE。教师用确定性预处理，学生仍用训练随机增强。此配置是首次试验参数，不表示最优。

## 二、运行前检查

1. 训练环境确认三个教师权重实际存在，完整 SHA-256、MD5 与来源记录、内嵌配置及 raw/EMA 身份和 [cost.md §9.3](cost.md) 一致；导出程序仅自动校验已记录的 MD5 **前缀**，不代替人工核对完整身份。缺一项或不匹配则停止。确认教师原始训练划分与当前数据一致。
2. 核对 train/valid 的 `Study_id`、`Subject_id` 无交叉、标签和 ViewPosition；valid 不用于训练软标签，test 完全不参与训练。缓存的索引摘要覆盖 study、患者、图像文件名和视角，**不覆盖图像内容**。
3. 确认两组配置展开后除 `distill_enabled` 外完全一致，且 `pretrained` 权重能够正常加载；预训练加载失败则停止，不把随机骨干当作有效实验。记录代码版本、数据 CSV SHA-256、GPU、Python/PyTorch、预训练权重身份和实际运行负载。
4. 用现有教师 valid 原始预测复算 A-FUSE-5，并与记录的 valid Macro-AUC 约 **0.8171** 对照；记录精确数值及差异。现有提交使用的 logit 融合与蒸馏导出的概率裁剪再取 logit 在极端概率处可能存在数值差别，应逐 study 对齐比较并记录，不能假定完全一致。教师按训练集自身标签训练过，缓存是 in-sample 预测，不是 OOF。

## 三、生成软标签与训练

下面命令在仓库根目录、GPU 训练环境执行；三个变量须替换为实际路径，顺序不能改变。缓存目录不存在才能生成，已有缓存先核对清单，不要覆盖。

```bash
export DISTILL1_D002_EMA='outputs/<D-002-a运行目录>/best_ema.pt'
export DISTILL1_C005='outputs/<C-005-c运行目录>/best.pt'
export DISTILL1_E001='outputs/<E-001运行目录>/best.pt'
test -f "$DISTILL1_D002_EMA" && test -f "$DISTILL1_C005" && test -f "$DISTILL1_E001"
shasum -a 256 "$DISTILL1_D002_EMA" "$DISTILL1_C005" "$DISTILL1_E001"
python src/run_distill_teacher.py --checkpoint "$DISTILL1_D002_EMA" "$DISTILL1_C005" "$DISTILL1_E001" --out outputs/distill/Distill-1_teacher
```

检查 `outputs/distill/Distill-1_teacher/manifest.json` 与 `predictions.csv`，确认 train 的 study 集合、列序及概率范围，核对教师配置、哈希和索引，并抽样复核三模型融合值。再按顺序串行运行，记录起止时间和实际输出目录：

```bash
python src/run_train.py --config configs/exp_Distill1a.yaml --seed 42 --tag Distill-1-a
python src/run_train.py --config configs/exp_Distill1b.yaml --seed 42 --tag Distill-1-b
```

训练前各做小批量 GPU smoke test，核对预训练加载、有限损失和显存；如发现预训练未加载、缓存不匹配、loss 非有限或明显塌缩，停止并排查。蒸馏组逐轮记录硬标签和软标签 loss；软目标 BCE 的数值含教师熵，不与 KL 数值直接比较。每组保留配置、日志、`metrics.csv`、`best.pt`、`best_ema.pt`、`last.pt`；缓存清单和 SHA-256 归档，续训不得替换缓存。

## 四、评估与判读

用训练完成后**实际产生的目录**设置下列变量；每组固定选 `best_ema.pt`，原始 `best.pt` 另行导出作补充：

```bash
export DISTILL1_A_RUN='outputs/<实际运行目录_Distill-1-a>'
export DISTILL1_B_RUN='outputs/<实际运行目录_Distill-1-b>'
python src/run_eval.py --config configs/exp_Distill1a.yaml --checkpoint "$DISTILL1_A_RUN/best_ema.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/Distill-1-a-EMA.csv
python src/run_eval.py --config configs/exp_Distill1b.yaml --checkpoint "$DISTILL1_B_RUN/best_ema.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/Distill-1-b-EMA.csv
python src/run_eval.py --config configs/exp_Distill1a.yaml --checkpoint "$DISTILL1_A_RUN/best.pt" --pred-out outputs/predictions/valid/Distill-1-a-raw.csv
python src/run_eval.py --config configs/exp_Distill1b.yaml --checkpoint "$DISTILL1_B_RUN/best.pt" --pred-out outputs/predictions/valid/Distill-1-b-raw.csv
```

`run_eval.py` 打印逐类明细，`--pred-out` 保存逐 study 标签和概率；它不会生成 `per_class.csv`。后续比较须从原始预测计算并存入 `outputs/comparisons/Distill-1/` 的 `summary.md`、`per_class.csv` 和 bootstrap 结果，**不使用 `run_compare.py` 的默认 best.pt 结果冒充 EMA 主比较**。检查 ID、标签、类别顺序以及 `Study_id → Subject_id` 一致后：

- 主要比较 **Distill-1-b EMA − Distill-1-a EMA** 的 10 类 study 级 Macro-AUC；报告 F1@0.5、阈值曲线、逐类 AUC/support 和 class2–class10 诊断性九类 AUC。与教师 A-FUSE-5 的 valid 结果另行比较，用于描述压缩取舍。
- 按患者整簇配对 bootstrap，2,000 次、seed 2026，同一次重采样用于两组，记录类别退化和有效次数；报告 Δ、95% CI 和正向比例。+0.010 为优先复核参考线，CI 跨零不证明等价；本次单 seed、同一 valid 反复选型，bootstrap 不覆盖训练随机性及选择偏差。
- 若效果有潜力，再做固定 seed 2026 的配对复核并考虑独立验证或患者级 OOF 教师；不预设蒸馏收益或选择教师之外的模型进入融合。
- 学生若成为部署候选，以其 checkpoint 重新生成匹配的 submission，在 V100 按教师相同的完整 `cost.py` 口径比较；记录硬件、100-study/5-study、3 轮、fp32 和 5 次结果。没有 T4 卡，不能据 V100 结果声明 T4 达标。

实验完成后按仓库规则同步训练记录、实验矩阵、本计划和 `~/Desktop/loker_exp.xlsx`；Distill 系列的表格归属先核查实际工作簿再确定，不假定已有对应 sheet。当前不填写任何性能数据，不更改现有提交。

## 五、执行记录（2026-11-04）

- 代码与资产：代码版本 `529f9f1`（runner 记录 dirty_files 60，未提交内容为 cost 重构与 AUCM 改动，不涉及蒸馏路径）；教师缓存 `outputs/distill/Distill-1_teacher/`，train 18,513 studies，`cache_digest` `5d41fe72…eff28`、`predictions_sha256` `91bb6c41…baedf`、`manifest_sha256` `4d99cd2b…e5f50`、`train_csv_sha256` `3a07b4bb…f17`、`index_sha256` `9fbf9bce…a1b`；三 checkpoint SHA-256 为 D-002-a `best_ema.pt` `aeeb2106…f08`、C-005-c `best.pt` `1fb3b039…df`、E-001 `best.pt` `7d95b6d0…930`；GPU Tesla V100-SXM2-32G（32,768 MiB，driver 580.105.08），Python 3.12.3 / torch 2.10.0+cu128。
- 运行前检查：study 集合与 train 完全一致、概率范围 [0.000244, 0.998769]、融合方式 logit_clipped_probability 与权重 [4,2,3] 与记录一致；两组各做 2 batch smoke（a loss 1.2095、b 1.4874，hard 1.1449 / soft 0.6850，均有限，显存峰值 3,956.2 / 4,352.3 MiB）；两组配置展开后仅 `distill_enabled` 不同；预训练权重均确认「已加载」。**未执行**计划 §2.4 的教师 valid 逐 study 复算，与教师的比较沿用 A-FUSE-5 历史 0.8171 / 0.8249。
- 执行：Distill-1-a 13:55:13–14:29:42（34m29s，7/8 轮早停，best epoch 5）；Distill-1-b 14:29:42–15:04:18（34m36s，7/8 轮早停，best epoch 5）；串行、无并行。评测 15:04:34 起由批次脚本完成，预测保存在 `outputs/predictions/valid/Distill-1-{a,b}-{EMA,raw}.csv`，配对比较产物在 `outputs/comparisons/Distill-1/`。
- 结果：a raw 0.7987 / 0.4351、EMA 0.8092 / 0.4339（9 类 0.8190）；b raw 0.8024 / 0.4344、EMA 0.8109 / 0.4328（9 类 0.8172）；F1 峰位均在 0.5（a-EMA 0.4445、b-EMA 0.4441）；b 的 `train_hard_loss` 0.9787→0.4439、`train_soft_loss` 0.5434→0.4919（epoch 5）→0.5501（epoch 7）。
- 主比较与判读：EMA b−a 10 类 **+0.0017**（患者整簇 bootstrap 2,000 次、seed 2026、228 名患者、有效 2000/2000，CI **[-0.0037, +0.0078]**、P=0.752）；9 类 −0.0018（CI [-0.0062, +0.0032]）、F1@0.5 −0.0012（CI [-0.0068, +0.0056]）。未达 +0.010 参考线且 CI 跨 0；a/b EMA 相对历史 D-002-a-EMA 0.8100 为 −0.0008 / +0.0009，Δ 与重跑波动同量级；增益集中在 class1（+0.0337，support 24），c2 −0.0118、c3 −0.0140；与教师差 −0.0061 / −0.0078。结论：**单 seed、当前 λ/T 与 8 轮预算下蒸馏未带来可测增益，也不能判等价**。
- 未完成项：教师 valid 逐 study 复算；seed 2026 配对复核；test 集学生预测与提交；学生 `cost.py` 耗时对比（无 T4，不能声明 T4 达标）。台账归属见 `~/Desktop/loker_exp.xlsx` 新增的「Distill系列实验」sheet。
- 文档回填已同步 `docs/training_log.md`（新增 Distill-1 小节）、`docs/experiments.md`（矩阵两行 + 执行小节）与本文件。
