# P7 调优记录

> 本文保留训练环境当时的产物路径；本机整理后的分类及路径对应说明见 [架构与文件结构](architecture.md#21-本机归档的输出目录)。

当前默认配置已改为余弦调度并启用 `pos_weight`；以下历史实验记录中的配置与对照关系以当时的运行配置为准，不能直接用当前继承配置重现。

P7 的目标是把「训练什么、怎么评、交什么」固定下来。训练统一放到训练环境（远程 GPU /
赛题服务器）执行，本机只负责代码与文档验证，避免两边环境差异污染实验结果；因此本文件
记录实验矩阵、判读原则和冻结结论。执行命令见训练计划，实测数字记录在训练记录中。

## 实验环境

| 项目 | 值 |
| --- | --- |
| 数据 | 全量 train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 设备 | RTX 3090 24G（复用同一实例；本机只做代码与文档验证）；RUN-009 / A-001 系列为 V100-SXM2-32G，两种硬件耗时不可横向对照 |
| 软件 | Python 3.12、torch 2.x（CUDA）、timm 1.0.30 |
| B 组环境 | `/root/venv-b001`：Python 3.12、torch 2.8.0+cu128、torchxrayvision 1.5.7；B-001 输入 224×224 单通道、xrv 归一化（仅研究） |
| 随机种子 | 42（RUN-002-B、RUN-006-A、RUN-007-A、RUN-007-B、RUN-009-A、RUN008-A/B/C/D、A-001-B(seed 2026) 为 2026） |
| 输入 | 320×320，letterbox，ImageNet 归一化，默认无水平翻转 |
| 骨干 | timm `resnet18.a1_in1k`，ImageNet 预训练 |

## 实验矩阵

| run | tag | 配置 | epochs | pos_weight | 用途 |
| --- | --- | --- | --- | --- | --- |
| RUN-001 | `short_baseline` | `configs/exp_short.yaml` | 6 | 否 | 短周期方向筛查的基准（旧代码 59a7d15） |
| RUN-001R | `short_baseline_r2` | `configs/exp_short.yaml` | 6 | 否 | 用当前代码（e8f194e）重跑同一配置，验证训练链路回归 |
| RUN-002 | `short_pos_weight` | `configs/exp_short_pos_weight.yaml` | 6 | 是 | 与 001R 同一 schedule，只差 pos_weight |
| RUN-003 | `full_baseline` | `configs/default.yaml` | 25 | 否 | 冻结配置的完整训练，产出提交候选 |
| RUN-003-A | `run003a_cosine` | `configs/exp_cosine.yaml` | 25 | 否 | 单变量对照：只把学习率调度换成 warmup + 余弦衰减到 1e-5 |
| RUN-003-B | `run003b_cosine_pos_weight` | `configs/exp_run003b.yaml` | 25 | 是 | 当前默认口径的 ResNet-18 完整复跑（cosine + pos_weight，seed 42），给 RUN-005 / RUN-006 提供同口径基线，已完成 |
| RUN-002-A | `pos_weight_s42` | `configs/exp_pos_weight.yaml` | 25 | 是 | 完整周期 `pos_weight` 对照，seed 42（短跑 RUN-002 的复核） |
| RUN-002-B | `pos_weight_s2026` | `configs/exp_pos_weight.yaml` | 25 | 是 | 同上，seed 2026，验证 `pos_weight` 结论的 seed 稳健性 |
| RUN-004A / 004B | 待定 | `exp_full_long.yaml` / `exp_pos_weight.yaml` | 40 / 25 | 否 / 是 | 按 RUN-003 的 `best_epoch` 与 RUN-002 结论二选一 |
| RUN-005 | `resnet50_s42` | `configs/exp_resnet50.yaml` | 25 | 是 | ResNet-50 骨干与预训练权重对照，已完成 |
| RUN-006 | `densenet121_s42` | `configs/exp_densenet121.yaml` | 25 | 是 | DenseNet-121 骨干与预训练权重对照，已完成 |
| RUN-006-A | `densenet121_s2026` | `configs/exp_run006a.yaml` | 25 | 是 | RUN-006 的 seed 复核（DenseNet-121，seed 2026），已完成 |
| RUN-007 | `densenet121_viewweighted_s42` | `configs/exp_run007.yaml` | 25 | 是 | DenseNet-121 + 正侧位 7:3 加权聚合，与 RUN-006（等权）单变量对照，已完成 |
| RUN-007-A | `densenet121_viewweighted_s2026` | `configs/exp_run007a.yaml` | 25 | 是 | RUN-007 的 seed 复核（加权聚合，seed 2026），与 RUN-006-A 配对比较，已完成 |
| RUN-007-B | `densenet121_viewweighted_s2026_384` | `configs/exp_run007b.yaml` | 6 | 是 | RUN-007-A 的分辨率探针（384×384、6 轮短跑），已完成 |
| RUN008-A | `run008a_effb2_s2026` | `configs/exp_run008a.yaml` | 6 | 是 | EfficientNet-B2 骨干对照（6 轮无早停，与 RUN008-B 并行），已完成 |
| RUN008-B | `run008b_effv2s_s2026` | `configs/exp_run008b.yaml` | 6 | 是 | EfficientNetV2-S（RW）骨干对照（6 轮无早停，与 RUN008-A 并行），已完成 |
| RUN008-C | `run008c_convnext_tiny_s2026` | `configs/exp_run008c.yaml` | 6 | 是 | ConvNeXt-Tiny 骨干对照（6 轮无早停，与 RUN008-D 并行），已完成 |
| RUN008-D | `run008d_convnextv2_nano_s2026` | `configs/exp_run008d.yaml` | 6 | 是 | ConvNeXtV2-Nano 骨干对照（研究用，CC BY-NC 4.0 不能提交；训练异常），已完成 |
| RUN-009-A | `run009a_effv2s_384_s2026` | `configs/exp_run009a.yaml` | 25 | 是 | EfficientNetV2-S（RW）@384×384 完整口径复核（实际 6/25 轮后按用户要求停止，非早停），已完成 |
| RUN-010 | `freeze_head_s42` | `configs/exp_freeze_head.yaml` | 25 | 是 | 冻结 ImageNet 预训练 ResNet-18、仅训练分类头；待 GPU 环境执行 |
| A-001 | `A-001_convnext_tiny_s42` | `configs/exp_A001.yaml` | 8（早停 2，实际 6） | 是 | RUN-008-C 重跑：ConvNeXt-Tiny @320×320，epoch=8 / patience=2 / seed=42，完整展开不继承 |
| A-001-a | `A-001-a_convnext_tiny_s42` | `configs/exp_A001a.yaml` | 8（早停 2，实际 6） | 是 | A-001 的增强对照：只把 `image.hflip` 改 true（训练随机水平翻转），其余口径逐字段不变；完整展开、无 `_base` |
| A-001-B | `A-001b_convnext_tiny_asl_s42` | `configs/exp_A001b.yaml` | 8（早停 2，实际 6） | 否（ASL 与 `pos_weight` 互斥） | A-001 的损失对照：BCE + `pos_weight` → ASL（gamma_pos 0 / gamma_neg 4 / clip 0.05），其余口径全部不变 |
| A-001-C | `A-001c_convnext_tiny_freeze_s42` | `configs/exp_A001c.yaml` | 8（跑满，无早停） | 是 | A-001 的冻结对照：ConvNeXt-Tiny 骨干全部冻结、仅训练分类头（768→10，7,690 可训练参数），其余口径全部不变 |
| A-001-B(s2026) | `A-001b_convnext_tiny_asl_s2026` | `configs/exp_A001b.yaml` + `--seed 2026` | 8（早停 2，实际 7） | 否（ASL 与 `pos_weight` 互斥） | A-001-B 的 seed 复核：与 seed 42 同配置、只换 seed 到 2026 |
| A-001-D | `A-001d_convnext_tiny_mean_s42` | `configs/exp_A001d.yaml`（`_base: exp_A001.yaml`） | 8（跑满，无早停） | 是 | A-001 的聚合对照：把 `model.aggregation` 由 `view_weighted`（正位 0.7 / 侧位 0.3 加权）改为 `mean`（study 内等权均值），其余口径全部不变 |
| A-001-G1 | `A-001-G1_convnext_tiny_asl_s42` | `configs/exp_A001_G1.yaml`（`_base: exp_A001.yaml`） | 8（早停 2，实际 6） | 否（ASL 与 `pos_weight` 互斥） | A-001-B 的 ASL 负样本抑制减弱对照：`train.asl_gamma_neg: 4.0 → 2.0`（减弱负样本抑制），其余口径逐字段不变，已完成 |
| A-001-G2 | `A-001-G2_convnext_tiny_asl_s42` | `configs/exp_A001_G2.yaml`（`_base: exp_A001.yaml`） | 8（早停 2，实际 6） | 否（ASL 与 `pos_weight` 互斥） | A-001-B 的 ASL 负样本抑制加强对照：`train.asl_gamma_neg: 4.0 → 6.0`（更强负样本抑制），其余口径逐字段不变，已完成 |
| A-001-G3 | `A-001-G3_convnext_tiny_asl_s42` | `configs/exp_A001_G3.yaml`（`_base: exp_A001.yaml`） | 8（早停 2，实际 6） | 否（ASL 与 `pos_weight` 互斥） | A-001-B 的 ASL 概率移位关闭对照：`train.asl_clip: 0.05 → 0.00`（关闭概率移位），其余口径逐字段不变，已完成 |
| A-001-G4 | `A-001-G4_convnext_tiny_asl_s42` | `configs/exp_A001_G4.yaml`（`_base: exp_A001.yaml`） | 8（早停 2，实际 3） | 否（ASL 与 `pos_weight` 互斥） | A-001-B 的 ASL 概率移位增大对照：`train.asl_clip: 0.05 → 0.10`（增大概率移位），其余口径逐字段不变，已完成 |
| A-001-G5 | `A-001-G5_convnext_tiny_asl_s42` | `configs/exp_A001_G5.yaml`（`_base: exp_A001.yaml`） | 8（早停 2，实际 6） | 否（ASL 与 `pos_weight` 互斥） | A-001-B 的 ASL 正样本聚焦对照：`train.asl_gamma_pos: 0.0 → 1.0`（正样本聚焦），其余口径逐字段不变，已完成 |
| A-002 | `A-002_mobilenetv3_small050_s42` | `configs/exp_A002.yaml`（`_base: exp_A001.yaml`） | 25（实际 5，按用户要求中止） | 是 | A-001 的轻量骨干对照：ConvNeXt-Tiny → MobileNetV3-Small 0.5×（578,474 参数），轮数 25 / 早停 5；第 5 轮中止、无 `best.pt`，不能作配对对照 |
| A-003 | `A-003_mobilenetv3_small100_s42` | `configs/exp_A003.yaml`（`_base: exp_A002.yaml`） | 25（早停 5，实际 17） | 是 | A-002 的宽度对照：MobileNetV3-Small 0.5× → 1.0×（1,528,106 参数），其余口径逐字段不变；已完成（0.7808 / 0.3881，未过 0.7988 门） |
| B-001 | `B-001_xrv_densenet121_s224` | `configs/exp_B001.yaml`（`_base: default.yaml`） | 25（早停 5，实际 11） | 是 | B 组：TorchXRayVision 医学预训练 DenseNet-121（单通道 224×224 / xrv 归一化 / 6,957,834 参数）；**仅研究，不用于赛题提交、融合或候选选择**（规则第 1 条）；已完成（0.7912 / 0.4120，与预训练集重叠风险未知） |
| A-004 | `A-004_convnext_tiny_dropout_wd_s42` | `configs/exp_A004.yaml`（`_base: exp_A001.yaml`） | 8（早停 2，实际 7） | 是 | A-001 的正则化组合对照：`model.dropout` 0.0 → 0.2（分类头 Dropout）+ `train.weight_decay` 0.01 → 0.05，其余口径逐字段不变；已完成（0.7988 / 0.4233，未过 0.8088 门） |
| A-004-a | `A-004a_dropout_only_s42` | `configs/exp_A004a.yaml`（`_base: exp_A001.yaml`） | 8（跑满上限） | 是 | A-001 的分类头 dropout 单变量对照：只把 `model.dropout` 0.0 → 0.2，`train.weight_decay` 保持 0.01，其余口径逐字段不变；已完成（0.7944 / 0.4360，未过 0.8088 门） |
| A-004-b | `A-004b_weight_decay_only_s42` | `configs/exp_A004b.yaml`（`_base: exp_A001.yaml`） | 8（早停 2，实际 7） | 是 | A-001 的 weight decay 单变量对照：只把 `train.weight_decay` 0.01 → 0.05，`model.dropout` 保持 0，其余口径逐字段不变；已完成（0.7999 / 0.4238，未过 0.8088 门） |
| A-005 | `A-005` | `configs/exp_csra.yaml`（`_base: exp_A001.yaml`） | 8（早停 2，实际 5） | 是 | A-001 的分类头替换对照：把 timm 标准分类头换成单头 CSRA 头（残差权重 0.4、温度 1.0、无 bias，27,827,808 参数），其余口径逐字段不变；已完成（0.7936 / 0.4050，未过 0.8072 门） |
| A-006 | `A-006_ml_decoder_s42` | `configs/exp_ml_decoder.yaml`（`_base: exp_A001.yaml`） | 8（早停 2，实际 7） | 是 | A-001 的分类头替换对照：把 timm 标准分类头换成轻量 ML-Decoder 头（10 类查询 + 单层 4 头交叉注意力 + 逐类权重向量，28,549,226 参数），其余口径逐字段不变；已完成（0.7900 / 0.4118，未过 0.8072 门） |
| A-007 | `A-007_convnext_small_in22k_ft_s42` | `configs/exp_A007.yaml`（`_base: exp_A001.yaml`） | 8（早停 2，实际 8） | 是 | A-001 的骨干与预训练权重对照：ConvNeXt-Tiny + `fb_in1k` → ConvNeXt-Small + `fb_in22k_ft_in1k`（49,462,378 参数），其余口径逐字段不变；已完成（0.7903 / 0.4204，未过 0.8088 门） |
| A-008 | `A-008_convnext_tiny_g1_s42` | `configs/exp_A008.yaml`（`_base: exp_A001.yaml`） | 8（早停 2，实际 7） | 是 | A-001 的聚合空间对照：`model.aggregation: view_weighted → view_weighted_probability`（先逐图 sigmoid 再在概率空间按正侧位 7:3 加权），其余口径逐字段不变；已完成（0.7974 / 0.4197，未过 0.8088 门） |
| A-009 | `A-009_convnext_tiny_lr1e-3_wd0.03_s42` | `configs/exp_A009.yaml`（`_base: exp_A001.yaml`） | 10（跑满上限） | 是 | A-001 的优化超参联合放大对照：`train.lr` 0.0003 → 0.001、`train.weight_decay` 0.01 → 0.03、`train.epochs` 8 → 10、`train.early_stop_patience` 2 → 4，其余口径逐字段不变；已完成（0.7799 / 0.4101，未过 0.8088 门） |
| A-009-r2 | `A-009-r2_warmstart_convnext_tiny_lr1e-3_wd0.03_s42` | `configs/exp_A009.yaml`（`_base: exp_A001.yaml`）+ `src/run_train.py --init-from` 首轮 `best.pt` | 10（早停 4，实际 5） | 是 | A-009 的二轮 warm start 对照：以首轮 A-009 `best.pt`（epoch 10 权重）为初始权重、按 A-009 同配置重训 10 轮；已完成（0.7764 / 0.4117，未过 0.8088 门，低于首轮 0.7799 / 0.4101） |

短周期基线把 `epochs` 降到 6，并关闭 early stop 和 `pos_weight`，其余超参数继承 `default.yaml`，
这样两个 run 的差异只来自 `pos_weight`。6 个 epoch 不足以跑出最终成绩，但足以判断
「pos_weight 是帮了忙还是拖了后腿」，避免直接为 25 个 epoch 赌一把。

## 执行与结果

执行步骤和判读门见 [训练计划](train_plan01.md)，通用命令及续训注意事项见 [使用手册](manual.md)。实际运行的配置、环境、指标和耗时记录在 [training_log.md](training_log.md)，对照结果以该文件为准。

截至 2026-10-30 的对照结论（数字均来自 `metrics.csv` 与 `outputs/compare/summary.md`）：

RUN-010 尚未执行真实 GPU 训练。本机已通过配置、模型冻结、优化器参数集合和 checkpoint metadata 的合成数据测试；训练完成后再与同 seed 的全量微调基线比较 Macro-AUC、Macro-F1、逐类 AUC 和训练耗时。

- **RUN-001R 回归门通过**：末轮 macro-AUC 0.7903，对比 RUN-001 的 0.7858 为 Δ+0.0045，落在判读门 [0.7808, 0.7908] 内，曲线形状也一致（6 轮单调上升、最优 epoch 6）→ 结果达到预设回归门，但不能证明训练结果完全等价，001R 的 `best.pt` 可作为耗时自检载体。
- **`pos_weight` 未达原 AUC 增益门（RUN-002）**：最优 macro-AUC 0.7882（epoch 5），对比 001R 的 0.7903 为 Δ−0.0021，未达「+0.005 且逐类无大退化」的判读门；逐类 AUC 差呈稀有类升、多数丰富类降（class1 +0.0588，class2 −0.0328、class4 −0.0398、class8 −0.0172），正是判读门要排除的形态。当时 `train.pos_weight` 维持 `false`，RUN-004 走 004A。
- **F1 差异属阈值效应**：001R 的 F1@0.5（0.2649）低于 RUN-001（0.2796），但 001R 在阈值 0.2 处达到 0.4012；两版代码只差预处理实现且数值等价，F1 的差异来自阈值选点与训练波动，不能据此判模型退化。
- **执行顺序与计划不同**：RUN-001R 与 RUN-002 并行执行（两路各 8 workers，16 核饱和），`run_compare` 在 01:00 先出对照，随后出提交文件并跑耗时自检；RUN-003 随后从头启动（`num_workers` 16，与训练机 16 核对齐）。并行运行的耗时不可直接与独占运行比较；不能仅凭 shuffle 顺序断言对训练结果完全无影响。
- **RUN-003 正式基线完成但未增益**：25 轮上限下 epoch 8 达峰（macro-AUC 0.7887），第 13 轮因连续 5 轮无提升早停；对比 RUN-001R（0.7903）为 Δ−0.0016，未达 +0.005 增益门。F1 阈值曲线整体上移（@0.2 0.4219 vs 0.4012）但主指标 macro-AUC 未提升 → **该次实验未证明增加训练轮数有收益**；按判读门（best_epoch ≤ 15 且 RUN-002 无效）004 仍走 004A（延长策略），但预期收益需谨慎。
- **余弦学习率调度不带来 AUC 增益（RUN-003-A，2026-10-28 下午）**：best epoch 5，macro-AUC 0.7852，对比 RUN-003 的 0.7887 为 Δ−0.0035，未达门；但最佳轮次从 8 前移到 5、早停从 13 轮缩到 10 轮、总耗时从约 38 分钟降到 29m47s。逐类 AUC 常见类 9/10/5/8/4 小涨（+0.003~+0.008），稀有类 1/2 回落（−0.0279 / −0.0463，support 仅 24/37）；F1@0.5 0.2861 略高于基线的 0.2655，但阈值曲线 @0.3 更低（0.3481 vs 0.3784）。当时维持 `warmup`；现按要求将默认配置改为 `cosine`，以上实测结论不变。
- **完整周期 `pos_weight` 对照（RUN-002-A/B，2026-10-28 下午）**：seed 42 best epoch 5、macro-AUC 0.7872、F1@0.5 0.4045；seed 2026 best epoch 7、AUC 0.7871、F1@0.5 0.4198。两 seed 的 AUC 只差 0.0001（仅能说明这两次结果接近），对比 RUN-003 分别为 Δ−0.0015 / Δ−0.0016，均未达 +0.005 门。F1@0.5 的大幅提升伴随阈值曲线右移：阈值曲线峰从 RUN-003 的 0.2（0.4219）右移到 0.5–0.6（0.4182 / 0.4212），F1 上限基本不变 → 当时按 AUC 判读门维持 `false`；现按要求启用 `pos_weight`。

2026-10-29 同口径基线与骨干对照（RUN-003-B / RUN-005 / RUN-006，数字来自 `outputs/compare_backbone_s42/summary.md`；三组同为 cosine + `pos_weight: true` + seed 42，可作单变量对照）：

- **当前默认口径的 ResNet-18 基线（RUN-003-B）**：best epoch 8，macro-AUC **0.7837**、F1@0.5 **0.4040**，第 13 轮早停（33m44s）。ResNet-18 的四个口径组合现在都有实测：RUN-003（warmup + 无 `pos_weight`）0.7887、RUN-003-A（cosine + 无 `pos_weight`）0.7852、RUN-002-A（warmup + `pos_weight`）0.7872、RUN-003-B（cosine + `pos_weight`）0.7837，四组极差 0.0050 → **cosine 与 `pos_weight` 同时开启也没有 AUC 增益**，两项的单独结论方向一致。F1@0.5 的提升仍来自 `pos_weight` 的概率校准：阈值曲线峰从 RUN-003 的 0.2 右移到 0.4，上限 0.4059 低于 RUN-003 @0.2 的 0.4219。该 run 同时是 RUN-005 / RUN-006 的同口径基线。
- **ResNet-50 略高于同口径基线**：RUN-005 best epoch 9，macro-AUC 0.7857（对比 RUN-003-B Δ+0.0020）、F1@0.5 0.4193（Δ+0.0153），第 14 轮早停（43m23s）；阈值 0.2–0.7 的 F1 多数点也更高。
- **DenseNet-121 的 macro-AUC 最高、F1 仍低于 ResNet-50**：RUN-006 best epoch 8，AUC 0.7880（对比 RUN-003-B Δ+0.0043），但 F1@0.5 0.4070 只比基线高 0.0030，且低于 ResNet-50（Δ−0.0123）；逐类上 DenseNet-121 在 class3/7/9/10 更好（+0.0474/+0.0239/+0.0202/+0.0122），ResNet-50 在 class1/2/6 更好（以 005−006 计 +0.0529/+0.0151/+0.0180）。
- **骨干结论修正**：此前用「RUN-005 / RUN-006 未超过 ResNet-18 历史 0.7887 / 0.7903」判断新骨干无增益，但历史 run 是 warmup + `pos_weight: false` 口径，不能与 cosine + `pos_weight: true` 的新骨干作单变量比较。补齐 RUN-003-B 后，同口径下 ResNet-50 / DenseNet-121 的 macro-AUC 分别高 0.0020 / 0.0043，方向有利于新骨干；但单 seed、差距较小，RUN-001 与 001R 同为 seed 42 且代码版本不同，其 0.0045 差值不能作为换 seed 波动，仍**不足以判定骨干优劣** → **暂不切换 `configs/default.yaml` 的 `model.backbone`**。若后续确定候选，还需按 [train_plan04.md](train_plan04.md) 核验预训练权重链接 / 许可 / SHA-256，并在赛题指定 T4 环境实测模型大小与端到端耗时（当前 `best.pt`：ResNet-50 91 MB、DenseNet-121 28 MB、ResNet-18 44.8 MB）。
- **DenseNet-121 的 seed 复核（RUN-006-A，2026-10-29 下午）**：seed 2026 下 best epoch 5、macro-AUC **0.7913**、F1@0.5 **0.4103**，第 10 轮早停（34m14s，与 experiment4 的 EfficientNet-B2 实验并行，耗时不可与 RUN-006 的独占 44m32s 直接对照）。对比 RUN-006（seed 42）AUC 高 **+0.0033**；seed 42 的配对骨干差为 +0.0043；0.7913 比 seed 42 的 ResNet-18 高 0.0076，但 seed 不匹配，不能说明骨干优势在第二个 seed 上复现。需补跑同配置 ResNet-18 seed 2026；两次 DenseNet 的 AUC 差为 0.0033，**仍不足以判定骨干优劣**，默认骨干维持 `resnet18`，结论不变。

2026-10-29 下午 RUN-007 正侧位加权聚合对照（数字来自 `outputs/compare_run007/summary.md`；与 RUN-006 同为 DenseNet-121 + seed 42，配置只差聚合方式）：

- **正侧位 7:3 加权聚合未达 AUC 增益门**：best epoch 8，macro-AUC **0.7892**、F1@0.5 **0.4122**，第 13 轮早停（34m56s）。对比 RUN-006（等权均值，0.7880 / 0.4070）：AUC Δ**+0.0012**，未达历史 +0.005 判读门，且小于 DenseNet-121 的 seed 波动（RUN-006 与 RUN-006-A 差 0.0033）→ **单 seed 下不能判定加权聚合有效**。F1@0.5 的提升伴随阈值曲线右移：0.2–0.4 段更低（0.3596 / 0.3785 / 0.3972 vs 0.3674 / 0.3882 / 0.4069），峰从 0.6（0.4098）移到 0.5（0.4122），上限基本持平，属校准/阈值形态而非判别力提升。逐类 AUC 6 升 3 降（class2 −0.0428 最明显、class3 +0.0169、class6 +0.0147、class8 +0.0118），方向不一致。
- **口径说明**：RUN-006 跑在 `da8fef7`（无视角元信息），RUN-007 跑在 `53568fe`（`62b6464` 起增加视角元信息贯通与 `view_weighted` 聚合）；`mean` 路径经 `tests` 与 RUN-006-A 在 `a41033d` 上的实测确认数值等价，但两组仍不是同代码版本的严格单变量对照，不能把差异全部归因于聚合方式。**默认 `model.aggregation` 维持 `mean`**；若继续，需按 [train_plan06.md](train_plan06.md) 补 seed 2026 复核（RUN-006-A 已有同 seed 的等权对照可配对）。

2026-10-29 下午 RUN-007-A seed 复核（数字来自 `outputs/compare_run007a/summary.md`；按 [train_plan07.md](train_plan07.md) 执行，seed 2026 的加权版与同 seed 等权版 RUN-006-A 配对比较）：

- **加权聚合在第二个 seed 上方向反转**：RUN-007-A best epoch 7，macro-AUC **0.7906**、F1@0.5 **0.4198**，第 12 轮早停（35m01s）。与 RUN-006-A（同 seed 等权，0.7913 / 0.4103）配对比较：AUC Δ**−0.0007**、F1@0.5 +0.0095。结合 seed 42 的配对差 +0.0012，两 seed 的配对差一正一负、均值 +0.00025，绝对值均远低于 +0.005 判读门 → **正侧位 7:3 加权聚合对 macro-AUC 无稳定增益，判定无效，默认 `model.aggregation` 维持 `mean`**。
- **F1 变化同样不稳定**：seed 42 的 F1 只在 0.5–0.7 段提升（0.2–0.4 段下降），seed 2026 则 0.2–0.7 全线提升 +0.0067~+0.0196，两 seed 形态不同；逐类 AUC 上 seed 2026 为「常见类 5–10 升、稀有类 1–4 降」（class9 +0.0145、class10 +0.0147、class3 −0.0279），与 seed 42 的升降分布也不一致。F1 与逐类的这些差异不能作为判别力提升的证据。

2026-10-29 晚 RUN-007-B 分辨率探针（数字来自 `outputs/compare_run007b/summary.md`；在 RUN-007-A 上只改 384×384 与 6 轮，其余口径相同）：

- **384×384 + 6 轮短跑未见退化，但差异不可单变量归因**：best epoch 5，macro-AUC **0.7959**、F1@0.5 **0.4227**（6 轮跑满，23m42s，显存约 10.2–10.5 GB）。与 RUN-007-A（320×320、12 轮早停，0.7906 / 0.4198）相比 AUC **+0.0053**、F1@0.5 +0.0029；但 007B 的 cosine 周期压到 6 轮、第 5 轮 lr 已降到 8.25e-5 提前收敛，而 007A 第 6 轮 lr 仍有 2.87e-4，两组轮数与调度不同，**不能把 +0.0053 归因于分辨率**。逐类上稀有类 2（+0.0395）、4（+0.0171）贡献主要差值，常见类 5/6/9 略降，6 轮短跑下单类波动大。
- **下一步**：若继续分辨率方向，需用 25 轮完整口径（同 seed、同 cosine 周期、早停 5）复跑 384 并与 320 配对，才能判定分辨率是否有 +0.005 以上的稳定增益；448 需先实测显存与耗时。

2026-10-29 晚 RUN-008 骨干对照（数字来自 `outputs/compare_run008/summary.md`；按 [train_plan08.md](train_plan08.md) 执行，四组继承 RUN-007-A 的 7:3 加权聚合口径，只换骨干，并统一改为固定 6 轮、关闭早停；A+B / C+D 两两并行执行）：

- **ConvNeXt-Tiny 在 6 轮预算下 macro-AUC 最高**：RUN008-C best epoch 5，macro-AUC **0.8008**、F1@0.5 **0.4382**，比 RUN-007-A（0.7906 / 0.4198）高 AUC **+0.0102**、F1@0.5 **+0.0184**，超过 +0.005 判读门；逐类 8 升 2 降（class3 +0.0319、class4 +0.0275、class1 +0.0186，class8 −0.0113、class10 −0.0036）。但 RUN-007-A 是 25 轮上限、第 12 轮早停（峰值 epoch 7），RUN008-C 是 6 轮无早停预算，两者轮数与 cosine 周期不同、单 seed 且单组 → **+0.0102 只能作为候选方向，不能据此切换默认骨干**。
- **EfficientNet 两组接近但未达门**：RUN008-A（EfficientNet-B2）macro-AUC **0.7952**、F1@0.5 0.4095；RUN008-B（EfficientNetV2-S RW）**0.7954** / 0.4100，分别比 RUN-007-A 高 +0.0046 / +0.0048，均低于 +0.005 判读门，F1@0.5 也低于 RUN-007-A 的 0.4198。逐类上 B 的 class2/class4 提升明显（+0.0451 / +0.0425）但 class8/10 回落，形态与阈值/校准差异混杂。
- **ConvNeXtV2-Nano 组训练异常且许可不合规**：RUN008-D best epoch 2，`run_compare` 重算 macro-AUC **0.4977**、F1@0.5 0.1604（`metrics.csv` 峰值 0.5097 / 0.1604），逐类 AUC 0.46–0.54 近似随机，6 轮 train_loss 仅从 1.183 降到 1.162。已核验骨干 158 个权重张量与预训练文件数值一致（仅 `head.fc` 正常替换为 10 类），单独运行显存 5.9 GB 与 C 组同量级 → 排除权重加载失败与显存问题，原因未定位（疑与 AMP fp16 下 ConvNeXtV2 的 GRN/LayerNorm 组合或该骨干与当前 lr/优化器不匹配）。该权重为 **CC BY-NC 4.0**，赛题《比赛参赛规则》第 1 条明确不接受 NC 类许可证 → 仅研究记录，不参与骨干判读、不能用于提交。
- **口径说明**：四组均为 seed 2026、320×320、`batch_studies: 16`、`pos_weight: true`、固定 6 轮无早停；A/B 于 20:59–21:37 并行，C/D 于 21:38–22:02 并行（D 首次并行运行按用户要求于 21:52 停止并删除，22:02 单独重跑至 22:25）。并行导致耗时不可与其他 run 对照。与 RUN-007-A 的预算差异使「+0.0102」不能当作严格单变量结论；若继续 ConvNeXt 方向，需按 25 轮完整口径重跑并补 T4 端到端耗时与权重许可核验。

2026-10-30 RUN-009-A 384×384 完整口径复核（数字来自 `last.pt` 的 `extras.best_metrics` 与 `run_eval.py`；按 [train_plan09.md](train_plan09.md) 执行，本次只跑 A 组）：

- **384 + 完整轮数在 6 轮内未见增量**：best epoch 5，macro-AUC **0.7944**、F1@0.5 **0.4185**（`run_eval` 重算 0.7945 / 0.4178），比 RUN-007-A（0.7906 / 0.4198）AUC **+0.0038**、F1@0.5 −0.0020，**未达 +0.005 判读门**（门 0.7956）；比同骨干 6 轮 320 的 RUN-008-B（0.7954 / 0.4100）AUC 低 **−0.0010**、F1 +0.0078。逐类 6 升 4 降：class2 +0.0676、class3 +0.0207 提升，class8 −0.0224、class6 −0.0143 回落，无单类大幅退化。
- **只跑 6/25 轮，不能判定完整预算**：按用户要求在第 6 轮后停止（非早停、非跑满），cosine 周期未走完（第 6 轮 lr 仍 2.87e-4）；epoch 5 后 valid_loss 由 0.9507 升到 1.0356、AUC 回落，出现过拟合迹象。硬件为 V100-SXM2-32G（与 RUN-001~008 的 RTX 3090 不同实例），6 轮约 41 分钟、显存峰值约 15.9 GB，耗时不可横向对照。RUN-009-B（ConvNeXt-Tiny @384）未启动。
- **口径说明**：本组同时改变分辨率（320→384）与训练预算（6→25 轮上限 + 早停 5），与 RUN-008-B 的差值不能单变量归因；产物 `best.pt` 因提前停止缺失，由 `last.pt` 的 `extras.best_state`（epoch 5）导出后再评估。

对照时使用同一份验证数据：`run_compare.py` 优先读取 `best.pt` 内保存的配置，旧 checkpoint 无存档配置时回退到 run 目录的 `config.yaml`。`summary.md` 和 `per_class.csv` 提供逐类指标与阈值。先看 macro-AUC，再结合阈值曲线及逐类表现判断 F1 和 `pos_weight` 的影响。

2026-10-30 A-001 RUN-008-C 重跑（数字来自 `run_eval.py`；`configs/exp_A001.yaml` 完整展开、无 `_base`，与 RUN008-C 只差 epochs 8 / patience 2 / seed 42）：

- **ConvNeXt-Tiny 的 0.80 水平在第二个 seed 上复现**：best epoch 4，macro-AUC **0.8038**、F1@0.5 **0.4255**（`metrics.csv` 一致），第 6 轮早停（连续 2 轮无提升），V100 独占 34m01s。与 RUN008-C（0.8008 / 0.4382，seed 2026、6 轮固定无早停、RTX 3090）相比 AUC **+0.0030**、F1@0.5 −0.0127；但 seed、训练上限（8 vs 6）、早停（2 vs 0）与 cosine 周期长度同时不同，**不是单变量对照**，AUC 差也小于已观测 seed 波动（0.0033）→ 只能说明 RUN008-C 的水平在本组复现，仍不足以判定骨干优劣或切换默认骨干。
- **F1 差异是全阈值段形态**：A-001 在 0.2–0.7 各点 F1 均比 RUN008-C 低 0.011~0.028，两者峰值都在 0.6（0.4328 vs 0.4438），属 `pos_weight` 校准形态差异，不是判别力下降；逐类 6 升 4 降（class2 +0.0459、class4 +0.0124 提升，class9 −0.0136、class3 −0.0080 回落）。
- **口径说明**：本组与 RUN008-C 同时改变三项超参，不能把 AUC/F1 差值单变量归因；本组只用于验证「更宽轮数上限 + 早停」下 ConvNeXt-Tiny 是否稳定在 0.80 附近。若需判定轮数或早停的增益，需固定 seed 做单变量对照。

**A-001-a 训练随机水平翻转对照**（配置 `configs/exp_A001a.yaml`，数字来自 `run_eval.py`，产物
`outputs/20261030-154652_A-001-a_convnext_tiny_s42`，2026-10-30）：完整展开配置、无 `_base`，与
`configs/exp_A001.yaml` 的展开结果只差 `image.hflip: false → true`（训练随机水平翻转），其余口径
逐字段一致；6/8 轮触发早停，best epoch 4：

- valid macro-AUC **0.7960** / F1@0.5 **0.4045**；独占 V100-SXM2-32G 训练 36m58s（15:46:52–16:23:50），
  单轮约 6.2 分钟（比 A-001 的 5.7 分钟略慢，与额外随机翻转增广一致），训练期间未抽样显存。
- 与 A-001（BCE + `pos_weight`：0.8038 / 0.4255）相比 **AUC −0.0078、F1@0.5 −0.0210**：AUC 差超过
  ±0.005 判读门，约为已观测 seed 波动（0.0033）的 2.4 倍 → 训练随机水平翻转在本口径下**无增益**，
  默认 `image.hflip: false` 保持不变。
- 逐类 4 升 6 降：class8 +0.0154、class9 +0.0105、class7 +0.0084 略升；class1 −0.0458、class2 −0.0417
  退化最大（support 仅 24 / 37，低频类噪声大），class6 −0.0110。总 AUC 下降且退化集中在稀有类。
- 口径注意：本组与 A-001 只差 `image.hflip` 一个字段，属单变量对照；但单 seed，且该配置在训练开始时
  尚未提交（17:20 补提交 `ac2d2f3`），训练时 HEAD 为 `0475bc7`、`src/` 其余口径未变。

**A-001-B A-001 的 ASL 损失对照**（配置 `configs/exp_A001b.yaml`，数字来自 `run_eval.py`，产物
`outputs/20261030-165734_A-001b_convnext_tiny_asl_s42`，2026-10-30）：以 `_base: exp_A001.yaml`
只覆盖损失五项（`train.loss: asl`、`train.pos_weight: false`、`asl_gamma_pos: 0`、`asl_gamma_neg: 4`、
`asl_clip: 0.05`），6/8 轮触发早停，best epoch 4：

- valid macro-AUC **0.7970** / F1@0.5 **0.4103**；独占 V100-SXM2-32G 训练 34m41s，显存约 13.8–14.3 GB。
- 与 A-001（BCE + `pos_weight`：0.8038 / 0.4255）相比 **AUC −0.0068、F1@0.5 −0.0152**：AUC 差超过
  ±0.005 判读门，约为已观测 seed 波动（0.0033）的 2 倍 → A-001 口径下 ASL 无增益，不切换默认损失。
- 逐类 3 升 7 降：class1（support 24）−0.0352、class4 −0.0191 退化最大，class9 +0.0080、
  class3 +0.0025、class8 +0.0026 略升；退化集中在低频类。
- 口径注意：ASL 与 `pos_weight` 在配置层互斥（`TrainConfig` 校验），本次是「损失方案」替换而非
  纯损失单变量；ASL 的 `train_loss` / `valid_loss` 量纲与 BCE 不同，不可横向比较 loss，只看 AUC/F1；
  单 seed 不能据此断言 ASL 在所有骨干/口径下无效。

**A-001-C 冻结骨干对照（仅训练分类头）**（配置 `configs/exp_A001c.yaml`，数字来自 `run_eval.py`，
产物 `outputs/20261030-173249_A-001c_convnext_tiny_freeze_s42`，2026-10-30）：以 `_base: exp_A001.yaml`
只覆盖 `model.freeze_backbone: true`，ConvNeXt-Tiny 骨干全部冻结，训练时骨干走 `eval()` 模式（BN 统计不更新），
仅分类头（768→10 Linear）可训练——checkpoint metadata 实测 `total_params: 27827818` /
`trainable_params: 7690`，可训练比例约 0.03%；8 轮跑满未触发早停，best epoch 8：

- valid macro-AUC **0.7264** / F1@0.5 **0.3602**（F1 峰值在 0.5；0.4 为 0.3551、0.6 为 0.3524）；
  独占 V100-SXM2-32G 训练约 38m01s（17:32:49–18:10:50），显存约 6.0 GB——比 A-001 的约 13.8 GB 低一半以上。
- 与 A-001（BCE + `pos_weight` 全量微调：0.8038 / 0.4255）相比 **AUC −0.0774、F1@0.5 −0.0653**，
  远超 ±0.005 判读门（约为已观测 seed 波动 0.0033 的 23 倍）：冻结 ImageNet 特征 + 线性头在胸片上
  信息量不足，不能替代全量微调，不作为提交候选。
- 逐轮单调爬升但全程低平（0.6819→0.7034→0.7175→0.7208→0.7238→0.7256→0.7262→0.7264），每轮增量
  从 +0.0215 衰减到 +0.0002，说明瓶颈在固定特征而非头部容量或训练时长——继续延长轮数或单独调分类头
  学习率的收益有限；逐类 class5/class9/class10 AUC 0.8041/0.8131/0.8282 尚可（高频类），
  class1/class4 仅 0.6672/0.6614，低频类退化最明显。
- 口径注意：本组与 A-001 的差异是「可训练参数范围 + 骨干 `eval()` 模式（BN 不更新）」，不是纯单变量；
  train_loss 与 A-001 同为 BCE 量纲（1.09→0.98）可比，但因可训练参数极少、过拟合风险低，8 轮未触发早停；
  单 seed、单轮数上限，结论只针对本口径。

**A-001-B（seed 2026）ASL 的 seed 复核**（配置 `configs/exp_A001b.yaml` + `--seed 2026`，数字来自
`run_eval.py`，产物 `outputs/20261030-181110_A-001b_convnext_tiny_asl_s2026`，2026-10-30）：与
A-001-B（seed 42）完全同配置、只换 seed 到 2026（本仓库既有 seed 复核惯例），7/8 轮触发早停，best epoch 5：

- valid macro-AUC **0.7952** / F1@0.5 **0.4182**，F1 峰值在 0.6（0.4254）；独占 V100-SXM2-32G 训练
  约 37m29s（18:11:10–18:48:39），显存约 12.2 GB；checkpoint metadata 确认 `seed: 2026`、`freeze_backbone: false`。
- 与 A-001-B（seed 42：0.7970 / 0.4103）的 **seed 配对差 AUC −0.0018、F1@0.5 +0.0079**：AUC 差在已观测
  seed 波动（±0.0033）内 → ASL 的 0.795–0.797 水平在第二个 seed 上复现；F1 峰位与 s42 同为 0.6，
  说明 F1 的 seed 差属校准形态波动而非判别力变化。
- 与 A-001（BCE，0.8038 / 0.4255）相比 AUC −0.0086、F1@0.5 −0.0073：AUC 差仍超过 ±0.005 判读门
  （约 1.7 倍）→ **A-001 口径下 ASL 无增益的结论在两 seed 上复现，确认不切换默认损失**；剩余差异
  仍可能来自 `pos_weight` 被移除（ASL 与 pos_weight 配置层互斥），本组不能把差异归因到 ASL 本身。
- 口径注意：seed 2026 同时改变数据划分与初始化/增强随机性，本组与 A-001-B(s42) 构成 seed 配对，
  与 A-001 的对照仍受损失方案差异混杂。

**A-002 / A-003 轻量骨干对照（MobileNetV3-Small 宽度）**（配置 `configs/exp_A002.yaml` / `configs/exp_A003.yaml`，数字来自 `run_eval.py`，产物 `outputs/20261030-223911_A-002_mobilenetv3_small050_s42` / `outputs/20261030-230725_A-003_mobilenetv3_small100_s42`，2026-10-30/31）：

- A-002（0.5×，578,474 参数）22:39:11 启动，23:03:03 第 5 轮后按用户要求中止（非早停），只有 `last.pt` 与 5 轮 metrics、无 `best.pt` → 与 A-003 的配对 `run_compare` 不成立；前 5 轮最佳 epoch 4（0.7634 / 0.3918，仅历史参考）。
- A-003（1.0×，1,528,106 参数，相对 A-002 只换骨干）23:07:25 启动，00:24:44 第 17 轮早停（best epoch 12），共 1h17m19s、单轮约 4.5 分钟、显存 1.4–1.6 GB；`best.pt` metadata 确认预训练权重已加载、全量微调（6,249,691 B）。
- valid macro-AUC **0.7808** / F1@0.5 **0.3881**（F1 峰值在 0.6：0.3965）；与 A-001 的 0.8038 / 0.4255 相比 **AUC −0.0230、F1@0.5 −0.0374**，距 plan11 的方向性筛选门 0.7988 **−0.0180 → 未过门**。
- 逐类 AUC 10 类全部低于 A-001（退化最大 class2 −0.0924、class3 −0.0301、class4 −0.0264，class8 仅 −0.0008）；阈值 0.2–0.7 全段低于 A-001。
- 判读：1.0× 宽度在本口径下仍明显低于 ConvNeXt-Tiny 记录值且未过门 → 不补 25+5 ConvNeXt 基线、不补 seed 2026、不改默认骨干；A-002 中止使「宽度 vs 0.5×」的配对问题保持未答，要回答需重跑 A-002。单 seed、单预算，差值不作统计显著性结论。

**A-001-D 取消正侧位加权对照**（配置 `configs/exp_A001d.yaml`，数字来自 `run_eval.py`，产物
`outputs/20261030-190645_A-001d_convnext_tiny_mean_s42`，2026-10-30）：以 `_base: exp_A001.yaml`
只覆盖 `model.aggregation: view_weighted → mean`，study 内多张图片改为等权均值，其余口径逐字段
不变（同 seed 42、同 8 轮上限与早停）；8 轮跑满无早停，best epoch 6：

- valid macro-AUC **0.7931** / F1@0.5 **0.4340**（F1 峰值在 0.5；0.2 为 0.4044、0.3 为 0.4168、
  0.4 为 0.4294、0.6 为 0.4293、0.7 为 0.4021）；独占 V100-SXM2-32G 训练约 40m03s
  （19:06:45–19:46:48），显存约 14.0 GB；checkpoint metadata 确认 `aggregation: mean`、
  `freeze_backbone: false`、`trainable_params: 27,827,818`（全量微调）。
- 与 A-001（`view_weighted`：0.8038 / 0.4255）相比 **AUC −0.0107、F1@0.5 +0.0085**：AUC 差超
  ±0.005 判读门、约 3.2 倍 seed 波动 → 取消正侧位加权在 A-001 口径下判别力下降；F1 提升来自
  F1 峰值由 0.6 左移到 0.5 的校准形态变化（0.2–0.5 段 F1 高于 A-001、0.6–0.7 段更低），
  不是排序能力提升。逐类 4 升 6 降，class2 −0.0431、class1 −0.0362 退化最大，高频 class9/class10
  基本持平（+0.0025 / −0.0060）。
- 判读：A-001 系列继续使用 `view_weighted`，不改成等权。结合 RUN-007 / RUN-007-A（DenseNet-121 上
  等权 vs 加权的两 seed 配对差方向相反、均未达 +0.005 门），正侧位加权在两套骨干上的收益都不稳定——
  A-001-D 直接证明「取消加权不优于保留加权」；要判定加权本身是否有效，仍需同骨干两 seed 配对。
- 口径注意：本组与 A-001 只差 `model.aggregation` 一个字段（`_base` 继承，其余逐字段一致），
  属单变量对照；单 seed；硬件、代码版本与 A-001 相同，耗时可直接横向对照。

**A-001-G 组 ASL 超参扫描**（配置 `configs/exp_A001_G1.yaml`…`configs/exp_A001_G5.yaml`，
数字来自 `run_eval.py`，产物 `outputs/20261030-201129_A-001-G1_convnext_tiny_asl_s42` …
`outputs/20261030-220518_A-001-G5_convnext_tiny_asl_s42`，2026-10-30）：五组均以
`_base: exp_A001.yaml` 只覆盖损失五项，以 A-001-B（ASL gamma_pos 0 / gamma_neg 4 / clip 0.05：
0.7970 / 0.4103）为参照；在 autodl V100-SXM2-32G 上按 G1→G5 串行执行（链式脚本，
20:11:24–22:36:29，共 2h25m05s），代码版本统一 `78ce901`、seed 42、数据与 A-001 同一份：

- **G1（gamma_neg 2，减弱负样本抑制）**：best epoch 4、6/8 轮早停，macro-AUC **0.7949** / F1@0.5
  **0.3956**（30m17s）；与 A-001-B 差 AUC −0.0021、F1@0.5 −0.0147（F1 峰左移到 0.4 的 0.4222）→
  减弱负样本抑制无增益，AUC 差在 seed 波动内。
- **G2（gamma_neg 6，更强负样本抑制）**：best epoch 4、6/8 轮，AUC **0.7999** / F1@0.5 **0.3391**
  （31m42s）；vs A-001-B 差 AUC +0.0029、F1@0.5 −0.0712：AUC 微升但概率整体推高（0.5 阈值处逐类
  recall 0.90–0.99、precision 很低），F1 峰值右移到 0.6（0.4211），固定 0.5 阈值下不可用。
- **G3（clip 0.00，关闭概率移位）**：best epoch 4、6/8 轮，AUC **0.7971** / F1@0.5 **0.4320**
  （30m51s）；vs A-001-B 差 AUC +0.0001、F1@0.5 **+0.0217**（也高于 A-001 的 0.4255）：关闭概率移位
  只改校准不改排序，F1 峰值就在 0.5，是五组中 F1@0.5 最高的一组。
- **G4（clip 0.10，增大概率移位）**：best epoch 1、3/8 轮即早停，AUC **0.7855** / F1@0.5 **0.3695**
  （18m14s）；vs A-001-B 差 AUC −0.0115、F1@0.5 −0.0408 → 过大的概率移位使训练早期退化，方向否定。
- **G5（gamma_pos 1，正样本聚焦）**：best epoch 4、6/8 轮，AUC **0.8002** / F1@0.5 **0.4202**
  （30m33s）；vs A-001-B 差 AUC +0.0032（seed 波动内）、F1@0.5 +0.0099；五组 AUC 最高、逐类 8 升 2 降
  （class4 +0.0297、class3 +0.0117 提升最大），但仍低于 A-001 的 0.8038 → 方向偏正面、未达判定门。
- **组结论**：gamma_neg 2→4→6 时 AUC 单调升（0.7949→0.7970→0.7999），但 F1@0.5 只由校准峰决定
  （三组峰值 F1 0.41–0.42 基本持平）；clip 在本扫描中只影响校准（0.00 略优于 0.05、0.10 破坏训练）；
  gamma_pos=1 方向偏正面、差值在 seed 波动内。五组 AUC 0.7855–0.8002 均未超过 A-001（0.8038）→
  **ASL 超参扫描没有改变「A-001 口径下 ASL 不优于 BCE + `pos_weight`」的结论，默认损失与 ASL 默认超参
  不切换**；若继续，应把 clip 0 + gamma_pos 1 组合另开单变量组并补 seed 复核，不在单 seed 上继续扫参数。

**B 组：TorchXRayVision 医学预训练 DenseNet-121 独立研究对照**（配置 `configs/exp_B001.yaml`，数字来自 `run_eval.py`，产物
`outputs/20261031-112106_B-001_xrv_densenet121_s224`，2026-10-31）：

- 目的与边界：验证 TorchXRayVision `densenet121-res224-all`（18 类胸片预训练）在本地数据上的可复现差异；
  **仅研究，不用于赛题提交、模型融合或候选选择**（规则第 1 条），计划见 [train_plan12.md](train_plan12.md)。
- 口径：单通道 224×224 + xrv 归一化 [-1024,1024]、letterbox、全量微调、BCE + `pos_weight`、`view_weighted`
  0.7/0.3、25 轮上限 + 早停 5、seed 42；与 A 组同数据划分、同聚合、同 seed。
- 结果：第 11 轮早停（best epoch 6），valid macro-AUC **0.7912** / F1@0.5 **0.4120**（F1 峰值在 0.6：0.4228）；
  参考 RUN-006（ImageNet DenseNet-121 @320、3 通道，0.7880 / 0.4070）与之差 AUC +0.0032、F1@0.5 +0.0050。
- 判读：差值同时来自分辨率、通道、归一化与初始化来源，**不可归因医学预训练**；本地数据来源不可确证、
  疑似 MIMIC-CXR-JPG 衍生，与预训练集重叠风险未知（疑似高）→ 指标可能乐观，不声称无泄漏；不并入冻结结论。

**A-004：分类头 dropout + weight decay 组合对照**（配置 `configs/exp_A004.yaml`，数字来自 `run_eval.py`，
产物 `outputs/20261031-121344_A-004_convnext_tiny_dropout_wd_s42`，2026-10-31）：

- 目的与口径：A-001 上只改两项——`model.dropout: 0.0 → 0.2`（分类头 Linear 前的 Dropout，训练启用 /
  评估关闭）与 `train.weight_decay: 0.01 → 0.05`；其余（ConvNeXt-Tiny @320、epoch 8 / patience 2、
  seed 42、`view_weighted` 0.7/0.3、BCE + `pos_weight`、cosine、batch 16 studies）逐字段一致；
  autodl 独占 V100-SXM2-32G，`check_data.py --full` 与 A-001 同一份划分。
- 结果：第 7 轮早停（best epoch 5），valid macro-AUC **0.7988** / F1@0.5 **0.4233**（F1 峰值在 0.6：
  0.4340），32m16s、显存约 14.0 GB；与 A-001 本次独立重算（0.8038 / 0.4255）差 **AUC −0.0050、
  F1@0.5 −0.0022**，距方向性门 0.8088 差 −0.0100 → 未过门。
- 训练行为：A-004 各轮 train_loss 高于 A-001（同轮 +0.005~+0.026），valid_loss 前 4 轮略高
  （+0.006~+0.013）、第 5–6 轮略低（−0.017 / −0.009），但走势未变，仍在第 4 轮见底后反弹
  （0.8990→0.9166→1.0102→1.1973）→ 正则化只抬高了训练损失水平、没有推迟过拟合；
  阈值曲线在 0.2–0.4 段高于 A-001、0.5–0.7 段更低，峰值仅 +0.0012（属校准形态变化）；逐类 4 升 6 降，
  class2 −0.0348、class6 −0.0186 退化最大，class3 +0.0276 提升最大。
- 统计判读：配对 bootstrap（859 studies 重采样 2,000 次）ΔAUC（A-001 − A-004）95% CI
  [−0.0061, +0.0167]、P(Δ>0)=0.803，ΔF1@0.5 95% CI [−0.0084, +0.0133] → 方向为 A-001 略优，
  单 seed、CI 跨 0，不能判定真实退化，但没有增益证据。
- 判读：同时改 dropout 与 weight_decay 两项，只能评价组合、不能分开归因；−0.0050 约为 A 组 seed
  波动 ±0.0033 的 1.5 倍 → **A-001 口径保持不变，默认配置不切换**；若继续，应把 dropout 与
  weight_decay 拆成两个单变量组并补 seed 2026 配对。

**A-004-a / A-004-b：dropout 与 weight decay 的单变量分解**（配置 `configs/exp_A004a.yaml` /
`configs/exp_A004b.yaml`，数字来自 `run_eval.py`，产物 `outputs/20261031-130923_A-004a_dropout_only_s42` /
`outputs/20261031-134732_A-004b_weight_decay_only_s42`，2026-10-31，计划见 [train_plan14.md](train_plan14.md)）：

- 目的与口径：把 A-004 同时改的两项拆开——A-004-a 只开分类头 dropout 0.2（`weight_decay` 保持 0.01），
  A-004-b 只把 `train.weight_decay` 提到 0.05（dropout 保持 0）；其余（ConvNeXt-Tiny @320、epoch 8 /
  patience 2、seed 42、`view_weighted` 0.7/0.3、BCE + `pos_weight`、cosine、batch 16 studies）与 A-001
  逐字段一致，两组串行在 autodl 独占 V100-SXM2-32G 执行。
- 结果：A-004-a 跑满 8 轮（best epoch 6）、37m16s，macro-AUC **0.7944** / F1@0.5 **0.4360**（F1 峰在
  0.4：0.4400）；A-004-b 第 7 轮早停（best epoch 5）、31m59s，macro-AUC **0.7999** / F1@0.5 **0.4238**
  （F1 峰在 0.6：0.4364）。
- 相对 A-001（0.8038 / 0.4255）：A-004-a 差 AUC **−0.0094**、F1@0.5 **+0.0105**；A-004-b 差 AUC
  **−0.0039**、F1@0.5 **−0.0017**；距方向性门 0.8088 分别差 −0.0144 / −0.0089 → 均未过门。
- 归因：两项单独都为负——dropout 单独 −0.0094（配对 bootstrap 95% CI [−0.0027, +0.0215]、P=0.934，
  约 2.8 倍 seed 波动）、weight decay 单独 −0.0039（95% CI [−0.0056, +0.0138]、P=0.785，在 seed 波动
  量级）；组合 −0.0050 接近两项单独差的算术平均（−0.0067），无协同放大；A-004 vs 单 wd 的配对差仅
  −0.0012（P=0.378）→ **组合的 AUC 水平主要由 weight decay 决定，dropout 是主要负向项**。
- 训练行为：A-004-a 各轮 train_loss 高 A-001 0.003–0.006、valid_loss 高 0.004–0.095；A-004-b train_loss
  高 0.0004–0.0093、valid_loss 前 4 轮高 0.007–0.027、第 5 轮低 0.018；两组 valid_loss 都在第 4 轮见底
  后反弹（A-004-a 0.9158→0.9305→1.1141→1.3354→1.4158；A-004-b 0.9126→0.9153→1.0621→1.2732）→
  两项都没有推迟过拟合。
- 判读：dropout 单独把排序变差、但把 F1 峰从 0.6 左移到 0.4 并抬高阈值曲线低段 → 只改校准形态、
  不改排序；weight decay 单独的差值接近噪声；两组均未过门 → **A-001 口径与默认训练超参不切换，
  不在单 seed 上继续扫正则化参数**。若继续，应先补 seed 2026 配对复核，再考虑与其他增强的组合。

**A-005：CSRA 分类头对照**（配置 `configs/exp_csra.yaml`，数字来自 `run_eval.py`，产物
`outputs/20261031-164123_A-005/`，配对基线 `outputs/20261031-151114_A-005_baseline_s42`，
2026-10-31，计划见 [训练计划 15](train_plan15.md)）：

- 目的与口径：在 A-001 训练预算内只把标准分类头换成单头 CSRA（残差权重 0.4、温度 1.0、
  无 bias，27,827,808 参数），其余口径逐字段不变；与 A-006 共用同一配对基线（同版本重训
  标准头 0.8022 / 0.4210），同数据（train 30,344 图 / 18,513 studies、valid 1,422 / 859）、
  同设备（V100-SXM2-32G 独占）、同预算（8 轮 / patience 2 / seed 42）、同 320×320 /
  `view_weighted` / BCE + `pos_weight`。
- 结果：第 5 轮早停（best epoch 3）、24m28s，macro-AUC **0.7936** / F1@0.5 **0.4050**（F1 峰
  在 0.6：0.4191）；`run_eval.py` 重算与 `metrics.csv` 一致。
- 相对配对基线：ΔAUC **−0.0086**、ΔF1@0.5 **−0.0161**，距方向性门 0.8072 差 **−0.0136**
  → 未过门；对历史 A-001（0.8038 / 0.4255）差 AUC −0.0103、F1@0.5 −0.0205。
- 配对 bootstrap（859 studies × 2,000 次，按 study 有放回重采样、seed 2026、退化类跳过）：
  ΔAUC（基线 − A-005）95% CI [−0.0030, +0.0210]、P=0.928；ΔF1@0.5 95% CI [+0.0061, +0.0266]、
  P=0.998 → 排序能力无可靠提升，F1@0.5 的下降有统计支持。
- 训练行为：峰值更早（epoch 3 vs 基线 4）且更低，valid_loss 第 3 轮见底 0.9035 后反弹到
  0.9403（基线为第 4 轮见底 0.9084 后反弹），train_loss 0.9887 → 0.7726；逐类 1 升 9 降，
  仅 class9 +0.0077，低支持类退化最大（class3 −0.0310、class2 −0.0194、class1 −0.0105）。
- 判读：CSRA 头未过门 → `model.head: standard` 保持默认，不切换、不补 seed 2026 复核、不
  进入候选提交；与 A-006（ML-Decoder 0.7900）合并看，两种分类头结构均低于同版本标准头
  配对基线 0.8022，分类头结构方向否定。

**A-006：ML-Decoder 分类头对照**（配置 `configs/exp_ml_decoder.yaml`，数字来自 `run_eval.py`，产物
`outputs/20261031-155701_A-006_ml_decoder_s42`，配对基线 `outputs/20261031-151114_A-005_baseline_s42`，
2026-10-31，计划见 [train_plan16.md](train_plan16.md)）：

- 目的与口径：ConvNeXt-Tiny @320、epoch 8 / patience 2、seed 42、`view_weighted` 0.7/0.3、BCE +
  `pos_weight`、cosine、batch 16 studies 全部与 A-001 一致，只把 `model.head` 由 `standard` 换成
  `ml_decoder`（10 个类别查询对 768 维特征图做单层 4 头交叉注意力，再按逐类权重向量投影出 logits；
  参数量 27,827,818 → 28,549,226）。按计划 16 的要求，主对照用同版本、同数据、同预算重训的标准头
  （`A-005_baseline_s42`，15:11:14–15:40:35、best epoch 4、**0.8022 / 0.4210**、27,827,818 参数），
  历史 A-001（0.8038）只作参考；方向性筛选门 = 0.8022 + 0.005 = **0.8072**。
- 结果：A-006 15:57:01–16:29:40（32m39s）、第 7 轮早停（best epoch 5），macro-AUC **0.7900** /
  F1@0.5 **0.4118**（F1 峰在 0.6：0.4173；基线 0.4383）。
- 相对配对基线：ΔAUC **−0.0122**、ΔF1@0.5 **−0.0092**，距门 −0.0172 → 未过门；逐类 2 升 8 降
  （class3 +0.0097、class7 +0.0076；class1 −0.0559、class2 −0.0455、class4 −0.0151 退化最大，
  高支持类 class9 / class10 基本持平）。
- 配对 bootstrap（859 studies × 2,000 次、按 study 重采样、seed 2026）：ΔAUC（基线 − A-006）=
  **+0.0123**，95% CI **[+0.0001, +0.0245]**、P(Δ>0) = **0.976**；ΔF1@0.5 = +0.0092，
  95% CI [−0.0018, +0.0205]、P = 0.952 → 标准头 AUC 显著更优，F1 方向一致但未跨零。
- 训练行为：收敛更慢（epoch 1 0.7533 vs 基线 0.7736）、峰值更晚更低（epoch 5 / 0.7900 vs epoch 4 /
  0.8022）；valid_loss 第 5 轮见底后反弹到 1.0157，与基线相同的「见顶即过拟合」形态，没有推迟过拟合。
- 判读：分类头方向否定——10 类共享查询在低支持度类上退化最重（假设与类别查询冷启动有关，未单独
  验证）；默认 `model.head: standard` 不切换，不补 seed 2026 配对，不进入候选提交。若继续注意力头，
  应先解决稀疏类的类别先验/初始化，而不是在单 seed 上改结构。

**A-007：ConvNeXt-Small 骨干 + ImageNet-22k 预训练权重对照**（配置 `configs/exp_A007.yaml`，数字来自
`run_eval.py`，产物 `outputs/20261031-172305_A-007_convnext_small_in22k_ft_s42`，2026-10-31，计划见
[train_plan17.md](train_plan17.md)）：

- 目的与口径：数据、预处理、`view_weighted` 0.7/0.3、BCE + `pos_weight`、cosine、batch 16 studies、
  epoch 8 / patience 2、seed 42 全部与 A-001 一致，只把 `model.backbone` 由 `convnext_tiny.fb_in1k`
  换成 `convnext_small.fb_in22k_ft_in1k`（ConvNeXt-Small 骨干 + ImageNet-22k 预训练后在 1k 微调的
  权重；参数量 27,827,818 → 49,462,378，+77.7%）。训练代码与 A-001 相同（`ea555c0` 只新增配置与计划
  文档），因此直接以 A-001（0.8038 / 0.4255，同数据、同设备、同预算、同 seed）为主对照，不再重训
  配对基线；方向性筛选门 = 0.8038 + 0.005 = **0.8088**。
- 结果：A-007 17:23:05–18:00:08（37m03s，含启动与首轮数据索引）、8/8 轮跑满后第 8 轮早停（best
  epoch 6），macro-AUC **0.7903** / F1@0.5 **0.4204**（F1 峰在 0.6：0.4225；A-001 峰 0.4328）。
- 相对 A-001：ΔAUC **−0.0135**、ΔF1@0.5 **−0.0051**，距门 −0.0185 → 未过门。
- 配对 bootstrap（859 studies × 2,000 次、按 study 重采样、seed 2026）：ΔAUC（A-001 − A-007）=
  **+0.0136**，95% CI **[+0.0006, +0.0260]**、P(Δ>0) = **0.983**；ΔF1@0.5 = +0.0051，
  95% CI [−0.0065, +0.0172]、P = 0.807 → A-001 的 AUC 优势正区间占优，F1 差方向一致但跨零。
- 逐类：3 升 7 降（class9 +0.0082、class8 +0.0032、class5 +0.0031；class2 −0.1079 断崖、class4
  −0.0147、class6 −0.0114），class2（support 37）是 AUC 差的主要来源（10 类平均拉低约 0.0108）。
- 训练行为：每轮 AUC 均低于 A-001 同轮（epoch 1 0.7452 vs 0.7738，epoch 6 0.7903 vs 0.8012），
  峰值更晚更低（epoch 6 / 0.7903 vs epoch 4 / 0.8038）；valid_loss 第 5 轮见底 0.9267 后反弹到
  1.3911 / 1.8042（A-001 为 0.8856 → 0.9333 / 1.0194），过拟合更陡；train_loss 到第 8 轮仍降到
  0.4471 → 更大容量在固定 8 轮预算内没有学得更好。
- 资源：参数 +21,634,560（+77.7%）、显存峰值 14,702 MiB（A-001 约 11.7 GB，+约 3.0 GB）、耗时
  37m03s / 8 轮 vs 34m01s / 6 轮（+8.9%）；预训练权重 200,928,946 B 约为 A-001 `fb_in1k` 权重
  114,374,272 B 的 1.76 倍。
- 判读：骨干/预训练权重方向否定（单 seed）——本组同时变骨干结构与预训练数据，差值不能单变量归因；
  ConvNeXt-Tiny（`fb_in1k`）保持 A 组默认候选，不补 seed 2026 配对，不进入候选提交。若继续更大骨干，
  应先解决固定 8 轮预算下的过拟合（更早停止 / 更强正则）或改用更长预算，并另开单变量实验
  （Small + `fb_in1k` 或 Tiny + `fb_in22k_ft_in1k`）把结构与权重拆开。


**A-008：聚合空间（logits G0 → 概率空间 G1）对照**（配置 `configs/exp_A008.yaml`，产物
`outputs/20261031-190629_A-008_convnext_tiny_g1_s42`，2026-10-31，计划见
[train_plan18.md](train_plan18.md)）

- 配置：以 A-001 为基准只改 `model.aggregation: view_weighted → view_weighted_probability`，正侧位仍 7:3；展开配置逐字段核对仅此一项不同。
- 结果：A-008 19:06:29–19:38:48（32m19s）、第 7 轮早停（best epoch 5），macro-AUC **0.7974** / F1@0.5 **0.4197**。
- 对比 A-001（0.8038 / 0.4255）：ΔAUC **−0.0064**、ΔF1@0.5 −0.0058，距 +0.005 方向性门差 −0.0114，**未过门**。
- 阈值曲线（0.2–0.7）：A-008 0.3735 / 0.3973 / 0.4120 / 0.4197 / 0.4359 / 0.4327，A-001 0.3641 / 0.3835 / 0.4046 / 0.4255 / 0.4328 / 0.4205；低阈值段（0.2–0.4）与 0.6–0.7 段 A-008 略高、0.5 附近更低，为概率空间平均的校准差异，F1@0.5 不作为达门依据。
- 判读：概率空间聚合训练方向否定（单 seed，差值超过 ±0.0033 波动带）；默认聚合保持 `view_weighted`，不做 seed 2026 复核、不进入候选提交。

**A-009：优化超参联合放大（lr 1e-3 / wd 0.03 / 10 轮 / patience 4）对照**（配置 `configs/exp_A009.yaml`，产物
`outputs/20261031-194952_A-009_convnext_tiny_lr1e-3_wd0.03_s42`，2026-10-31，计划见
[train_plan19.md](train_plan19.md)）

- 配置：以 A-001 为基准只改 `train.lr` 0.0003 → 0.001、`train.weight_decay` 0.01 → 0.03、`train.epochs` 8 → 10、`train.early_stop_patience` 2 → 4，其余口径逐字段不变；四项同改，差值不可拆单项。
- 结果：A-009 19:49:52–20:33:27（43m35s）、10/10 轮跑满无早停（best epoch 10），macro-AUC **0.7799** / F1@0.5 **0.4101**。
- 对比 A-001（0.8038 / 0.4255）：ΔAUC **−0.0239**、ΔF1@0.5 −0.0154，距 +0.005 方向性门差 −0.0289，**未过门**。
- 阈值曲线（0.2–0.7）：A-009 0.3825 / 0.3991 / 0.4053 / 0.4101 / 0.4083 / 0.3925，A-001 0.3641 / 0.3835 / 0.4046 / 0.4255 / 0.4328 / 0.4205；低阈值段（0.2–0.3）A-009 略高，0.5–0.7 段明显更低，F1 峰位由 0.6 左移到 0.5。
- 逐类：A-001 在 10 类中 9 类 AUC 高于 A-009，只有 class7 低 0.0027；降幅最大为 class2（−0.0867，support 37）、class4（−0.0414）、class3（−0.0391）。
- 配对 bootstrap（859 studies × 2,000 次，按 study 有放回重采样，seed 2026）：ΔAUC（A-001 − A-009）95% CI [+0.0121, +0.0360] P=1.000、ΔF1@0.5 95% CI [+0.0027, +0.0285] P=0.991，差距不是 valid 内采样噪声。
- 判读：与「更大 lr + 更强 wd + 更长预算」的组合假设相反（单 seed）；A-004-b 显示 wd 0.01 → 0.05 单独只差 AUC −0.0039（seed 波动内），本组 −0.0239 更可能主要来自 lr 放大后的收敛推迟与过拟合加剧，但四项同改不能单变量归因；默认超参保持 A-001 口径，不补 seed 2026，若继续调参应改开 lr / wd 单变量实验。

**A-009-r2：A-009 的二轮 warm start（以首轮 best.pt 为初始权重、同配置重训 10 轮）**（配置 `configs/exp_A009.yaml` + `src/run_train.py --init-from`，产物
`outputs/20261031-204914_A-009-r2_warmstart_convnext_tiny_lr1e-3_wd0.03_s42`，2026-10-31，计划见
[train_plan20.md](train_plan20.md)）

- 配置：与首轮 A-009 逐字段相同（lr 1e-3 / wd 0.03 / 10 轮 / patience 4），唯一区别是初始化——不加载 ImageNet 预训练权重，改为加载首轮 `best.pt`（epoch 10，SHA-256 `6988006910b824a83c277764695d0db4a3bdeb3ab0b1b029d22dc9e3ee0f1b15`）；训练从第 1 轮重新开始，不恢复优化器 / 调度器 / 进度。
- 结果：20:49:14–21:13:48（24m34s，V100 与另一容器共享）、第 5 轮早停（连续 4 轮无提升，best epoch 1），macro-AUC **0.7764** / F1@0.5 **0.4117**；逐轮 AUC 0.7764 / 0.7739 / 0.7587 / 0.7692 / 0.7664。
- 对比首轮 A-009（0.7799 / 0.4101）：ΔAUC **−0.0035**、ΔF1@0.5 +0.0016；对比 A-001（0.8038 / 0.4255）：ΔAUC **−0.0274**、ΔF1@0.5 −0.0138，距 +0.005 方向性门差 −0.0324，**未过门**。
- 阈值曲线（0.2–0.7）：A-009-r2 0.3684 / 0.3833 / 0.3966 / 0.4117 / 0.4137 / 0.4022，首轮 A-009 0.3825 / 0.3991 / 0.4053 / 0.4101 / 0.4083 / 0.3925；低阈值段（0.2–0.3）二轮更低、0.5–0.7 段略高，属校准形态变化。
- 逐类：class5 0.8579 / class9 0.8942 / class10 0.8678 相对稳定，class1 0.6993 / class4 0.6764 / class8 0.6973 偏低；未做与首轮 A-009 逐类差异的显著性检验。
- 判读：以首轮 ep10 权重初始化后，lr 重新从 5e-4 warmup 到 1e-3 破坏了已收敛的权重——train_loss 由首轮的 0.6562 反弹到 0.7556，valid AUC 由源权重 0.7799 掉到 0.7764 后连续 4 轮未回升，5 轮即早停；**warm start 二轮无增益**，A-009 首轮的否定结论不变。本轮耗时受 GPU 共享影响（每轮约 6–7 分钟 vs 首轮 4.4 分钟/轮），不可横向比较。

## 冻结结论

1. **提交格式**：`run_infer.py` 默认输出全部 10 类概率，不做阈值过滤——赛题的
   macro-AUC 需要排序信息，全 10 类输出更稳妥；`--threshold 0.5` 只用于对齐赛题示例
   的过滤形态。
2. **配置**：`configs/default.yaml` 即最终配置（25 epoch、`pos_weight: true`、
   `eval.threshold: 0.5`，骨干维持 `resnet18`——RUN-003-B 补齐同口径基线后，RUN-005 /
   RUN-006 的 macro-AUC 只高 0.0020 / 0.0043，单 seed 差距不足以判定骨干优劣；
   RUN-006-A 用 seed 2026 复核 DenseNet-121 得 0.7913，但缺少同 seed 的 ResNet-18
   对照，尚不能判断骨干优势是否在第二个 seed 复现；RUN-008 的 ConvNeXt-Tiny 在 6 轮
   预算下 0.8008 超过 +0.005 门，但预算与 RUN-007-A 不同且单 seed，同口径复核前
   不切换；A-001（2026-10-30，seed 42、epoch 8、早停 2）复现 0.8038，
   与 RUN008-C 同量级、差异混杂，仍不切换；A-001-a（训练随机水平翻转，AUC 0.7960）、A-001-C（冻结骨干，0.7264）与 A-004 系列（dropout + weight_decay 组合 0.7988、单 dropout 0.7944、单 weight_decay 0.7999）、A-006 的 ML-Decoder 分类头（0.7900 vs 同版本重训标准头 0.8022）与 A-005 的 CSRA 分类头（0.7936 vs 同一配对基线 0.8022）与 A-007 的 ConvNeXt-Small + `fb_in22k_ft_in1k` 骨干/权重（0.7903 vs A-001 0.8038）也未带来 AUC 增益，默认 `image.hflip: false`、`freeze_backbone: false` 与训练超参保持不变）；
   实验配置一律用 `_base` 继承它，不在实验文件里复制超参。
   在对比表出来之前不无依据地改默认值，要改必须先附 `run_compare.py` 的结果；`train.pos_weight` 的 AUC 未达原增益门，但 F1@0.5 显著提高，现按要求设为 `true`；`train.lr_scheduler` 经 RUN-003-A 复核无 AUC 增益，现按要求改为 `cosine`。
3. **复现口径**：默认 seed 为 42，seed 复核另有 2026；指标统一由 `run_compare.py` 在同一份 valid 上重算，
   训练设备与预训练加载情况记在 checkpoint 的 `metadata` 里。
4. **分工**：本机只跑 `python -m pytest` 和文档校验；训练相关命令都在训练环境执行，
   实测结果回填训练记录，对照结论记录在本文件。

## 后续待验证优化项（未排期）

记录想法，不承诺优先级；后续实验应在已有基线的基础上单独做对照，
一次只改一个变量，并保存对应的验证报告。

| 优化项 | 预期收益 | 改动点 | 备注 |
| --- | --- | --- | --- |
| 新骨干对照 | 已验证（未达切换门槛） | `configs/exp_densenet121.yaml` / `configs/exp_resnet50.yaml` / `configs/exp_run008{a,b,c,d}.yaml` | **RUN-003-B / RUN-005 / RUN-006 / RUN-006-A 已完成（同口径）**：ResNet-18 基线 0.7837、ResNet-50 0.7857、DenseNet-121 0.7880（seed 42）/ 0.7913（seed 2026）；新骨干 macro-AUC 高 0.0020 / 0.0043，seed 2026 尚缺 ResNet-18 配对结果，仍不足以判定骨干优劣，暂不切换默认骨干。**RUN-008 骨干对照（2026-10-29 晚，6 轮无早停）**：ConvNeXt-Tiny 0.8008（比 RUN-007-A 高 +0.0102）、EfficientNet-B2 0.7952（+0.0046）、EfficientNetV2-S 0.7954（+0.0048）、ConvNeXtV2-Nano 0.4977（训练异常）；ConvNeXt-Tiny 超过判读门但预算与 RUN-007-A（25 轮、12 轮早停）不同且单 seed，只作候选方向；ConvNeXtV2-Nano 权重为 CC BY-NC 4.0，赛题规则不接受。详见 [train_plan04.md](train_plan04.md)、[train_plan05.md](train_plan05.md)、[train_plan08.md](train_plan08.md) 与训练记录。T4 端到端耗时与权重许可 / SHA-256 核验仍待做（仅当选新骨干时）。A-001（2026-10-30）在 seed 42、epoch 8、早停 2 下复现 0.8038，与 RUN008-C 同量级、差异混杂；**A-003（2026-10-30/31）**：MobileNetV3-Small 1.0×（1,528,106 参数）@320、25 轮上限 + 早停 5 口径在第 17 轮早停（best epoch 12），macro-AUC 0.7808 / F1@0.5 0.3881，低于 A-001 的 0.8038 且距 0.7988 门差 −0.0180 → 轻量 CNN 在本预算下未过门，默认骨干不切换；A-002（0.5×）第 5 轮按用户要求中止、无 `best.pt`，两者配对比较不成立（详见 [train_plan11.md](train_plan11.md)）；**A-007（2026-10-31）**：ConvNeXt-Small + `fb_in22k_ft_in1k`（49,462,378 参数）@320、8 轮 / patience 2 / seed 42 口径在第 8 轮跑满后早停（best epoch 6），macro-AUC 0.7903 / F1@0.5 0.4204，低于 A-001（0.8038 / 0.4255）ΔAUC −0.0135、距 0.8088 门 −0.0185，配对 bootstrap ΔAUC 95% CI [+0.0006, +0.0260] P=0.983 → 更大骨干 + 更强预训练在本预算下未过门，且同时变结构与权重不可单变量归因（详见 [train_plan17.md](train_plan17.md)） |
| **视图元信息（ViewPosition）** | 已验证（无效） | 已实现 `StudyRecord` / `collate_studies` 传递视角及正侧位加权聚合 | **RUN-007 / RUN-007-A 已判无效**：seed 42 加权 macro-AUC 0.7892 vs 等权 0.7880（Δ+0.0012），seed 2026 加权 0.7906 vs 等权 0.7913（Δ−0.0007），两 seed 配对差方向相反且均低于 +0.005 门；F1 与逐类变化在两 seed 间也不一致。**A-001-D（2026-10-30）在 ConvNeXt-Tiny 上做等权 vs 加权**：等权 0.7931 / F1@0.5 0.4340 vs 加权 A-001 0.8038 / 0.4255，AUC **−0.0107**（超 ±0.005 门、约 3.2 倍 seed 波动）、F1 +0.0085（峰位 0.6→0.5 的校准差）→ 等权不优于加权；A-001 系列口径维持 `view_weighted`，`configs/default.yaml` 的默认聚合未改动。两套骨干结论方向一致：正侧位加权的收益未被证明，但取消加权同样不是改善方向。`Rows`/`Columns` 不建议用（易学设备偏差），`Subject_id` 严禁入特征 |
| 阈值曲线选点 | 中 | 仅推理侧：`run_compare.py` 扫 0.2~0.7，必要时改 `eval.threshold` | RUN-003 正式模型已复核：F1 峰值仍在 0.2（0.4219，@0.5 为 0.2655），与 RUN-001R 形态一致；RUN-002-A/B 的 pos_weight 模型把峰右移到 0.5–0.6（0.4182 / 0.4212）、RUN-003-B（cosine + pos_weight）峰在 0.4（0.4059），上限均与基线 @0.2 持平，这些选点仅用于验证 F1；默认提交输出全部 10 类概率，不按阈值过滤 |
| 训练轮数 | 已验证 | 在正式基线之后调整 `train.epochs` | **RUN-003 已判无效**：epoch 8 达峰（0.7887）后过拟合、第 13 轮早停，25 轮上限未带来 AUC 增益（Δ−0.0016 vs RUN-001R），该次实验未证明增加训练轮数有收益 |
| 学习率调度 | 已验证 | `train.lr_scheduler` / `train.lr_min` | **RUN-003-A 判无效**：余弦调度 best epoch 5、macro-AUC 0.7852 vs RUN-003 的 0.7887（Δ−0.0035，未达门）；但最佳轮次前移（8→5）、早停更快（13→10 轮）、耗时约 30 分钟 vs 38 分钟。当前默认配置为 `cosine`；该对照未证明 AUC 增益 |
| `pos_weight` | 已验证 | `train.pos_weight: true` | **RUN-002-A/B 未达原 AUC 增益门**：完整周期双 seed macro-AUC 0.7872 / 0.7871，vs RUN-003 0.7887 为 Δ−0.0015 / −0.0016，均未达 +0.005 门；F1@0.5 提升（0.4045 / 0.4198）伴随阈值曲线右移，已测峰值与基线接近，当前默认开启 |
| 数据增强（训练水平翻转） | 已验证（无效） | `image.hflip` | **A-001-a 已做（2026-10-30）**：A-001 口径下只开训练随机水平翻转，best epoch 4 macro-AUC 0.7960 / F1@0.5 0.4045 vs A-001 0.8038 / 0.4255（ΔAUC −0.0078、ΔF1@0.5 −0.0210，超 ±0.005 门且约 2.4 倍 seed 波动）；逐类 4 升 6 降、退化集中在低频类 → `image.hflip: false` 保持默认。其它增强（旋转、颜色抖动）未测，若要继续应单独开单变量对照 |
| 损失函数设计（ASL / focal） | 已验证（未达增益门） | `train.loss` / `train.asl_gamma_pos` / `train.asl_gamma_neg` / `train.asl_clip` | **A-001-B 已做（2026-10-30）**：A-001 口径下用 ASL（gamma_pos 0 / gamma_neg 4 / clip 0.05）替代 BCE + `pos_weight`，best epoch 4 macro-AUC 0.7970 vs A-001 0.8038（Δ−0.0068，超过 ±0.005 门，F1@0.5 也低 −0.0152）；ASL 与 `pos_weight` 互斥（替换非纯单变量）且单 seed → 暂不切换默认损失；继续验证应换 support 更充分的骨干或更长轮数；**seed 2026 复核（2026-10-30 晚）**：A-001-B(s2026) 0.7952 vs A-001-B(s42) 0.7970（Δ−0.0018，在 seed 波动内）、vs A-001 0.8038（Δ−0.0086 仍超门）→ ASL 无增益在两 seed 复现，默认损失不切换；**A-001-G1~G5 超参扫描（2026-10-30 晚）**：五组均以 `_base: exp_A001.yaml` 只覆盖 ASL 五项（参照 A-001-B 0.7970 / 0.4103）——G1 gamma_neg 2（0.7949 / 0.3956）、G2 gamma_neg 6（0.7999 / 0.3391，F1 峰右移到 0.6）、G3 clip 0.00（0.7971 / 0.4320，F1@0.5 组内最高）、G4 clip 0.10（0.7855 / 0.3695，best epoch 1 退化）、G5 gamma_pos 1（0.8002 / 0.4202，组内 AUC 最高）；五组 AUC 均未超过 A-001（0.8038 / 0.4255），gamma_neg 方向只改校准、clip 0 略改善 F1@0.5、gamma_pos 1 方向偏正面但差值在 seed 波动内 → 默认损失仍不切换；后续可把 clip 0 + gamma_pos 1 组合另开单变量组并补 seed 复核 |
| 聚合方式消融 | 已验证（等权不优于加权） | 比较平均 logits、平均概率及最大聚合 | 当前使用平均 logits 后 sigmoid；**A-001-D（2026-10-30）**：ConvNeXt-Tiny 上把正侧位加权（`view_weighted`）换成等权 `mean`，macro-AUC 0.7931 vs A-001 的 0.8038（−0.0107，超 ±0.005 门），等权不优于加权；平均概率与最大聚合仍未实现或实测 |
| 更高分辨率 / TTA | 已验证（方向性） | `image.size`、`run_infer.py` 加翻转 TTA | **RUN-007-B 已做 384×384 + 6 轮短跑（macro-AUC 0.7959 / F1@0.5 0.4227）**：比 320 的 12 轮早停 0.7906 高 +0.0053，但轮数与 cosine 周期不同、不可单变量归因；需 25 轮完整对照再判定。**RUN-009-A（2026-10-30）**：EfficientNetV2-S RW @384×384 + 25 轮上限口径只跑 6/25 轮（用户要求停止），macro-AUC 0.7944，比同骨干 320×320 6 轮的 RUN-008-B（0.7954）低 −0.0010，未显示增量；25 轮完整对照仍待做。TTA 未做；448 显存未实测（3090 24G） |
| 冻结骨干、仅训练分类头 | 已验证（大幅退化） | `configs/exp_freeze_head.yaml` / `configs/exp_A001c.yaml`、`model.freeze_backbone` | **A-001-C 已做（2026-10-30）**：ConvNeXt-Tiny @320 冻结骨干、仅训分类头（7,690 / 27,827,818 可训练参数），8 轮跑满 best epoch 8 macro-AUC 0.7264 / F1@0.5 0.3602 vs A-001 0.8038 / 0.4255（ΔAUC −0.0774、ΔF1@0.5 −0.0653）；逐轮增量衰减到 +0.0002 说明瓶颈在固定特征 → 该口径不作提交候选。RUN-010（ResNet-18 冻结）仍待执行。 |
| 正则化（dropout / weight_decay） | 已验证（未达增益门） | `model.dropout` / `train.weight_decay` | **A-004（2026-10-31）**：A-001 口径下分类头 dropout 0.2 + weight_decay 0.05 组合，valid macro-AUC 0.7988 vs A-001 0.8038（Δ−0.0050）、F1@0.5 0.4233 vs 0.4255（Δ−0.0022），距方向性门 0.8088 差 −0.0100 → 未过门；各轮 train_loss 高于 A-001、valid_loss 走势未变（第 4 轮后仍反弹）；单 seed、两项同改不可分开归因，若继续应拆成两个单变量组并补 seed 复核；**A-004-a / A-004-b（2026-10-31）**：把组合拆成两个单变量组——A-004-a 只开 dropout （0.7944 / 0.4360，ΔAUC −0.0094、F1@0.5 +0.0105，F1 峰左移到 0.4 的 0.4400）、A-004-b 只提 weight_decay（0.7999 / 0.4238，ΔAUC −0.0039、F1@0.5 −0.0017）；两项单独均为负向，组合 −0.0050 主要由 weight decay 决定（组合 vs 单 wd 差 −0.0012），dropout 是主要负向项 → 正则化方向无增益，默认训练超参不变，后续应补 seed 2026 配对后再换档位 |

| 分类头结构（ML-Decoder / CSRA） | 已验证（两种注意力头均未达增益门） | `model.head` / `configs/exp_ml_decoder.yaml` / `configs/exp_csra.yaml` | **A-006（2026-10-31）**：A-001 口径下把标准头换成轻量 ML-Decoder（10 类查询 + 单层 4 头交叉注意力，28,549,226 参数），valid macro-AUC **0.7900** / F1@0.5 **0.4118** vs 同版本重训标准头 0.8022 / 0.4210（ΔAUC **−0.0122**、ΔF1@0.5 −0.0092），距方向性门 0.8072 差 −0.0172；配对 bootstrap ΔAUC（基线 − A-006）95% CI [+0.0001, +0.0245]、P=0.976、ΔF1@0.5 CI [−0.0018, +0.0205]；逐类 2 升 8 降、低支持类退化最大；收敛更慢（峰值 epoch 5 vs 4）、未推迟过拟合。**A-005（2026-10-31）**：同一配对基线下把标准头换成单头 CSRA（残差权重 0.4 / 温度 1.0 / 无 bias，27,827,808 参数），valid macro-AUC **0.7936** / F1@0.5 **0.4050**（ΔAUC **−0.0086**、ΔF1@0.5 **−0.0161**），距方向性门 0.8072 差 **−0.0136**；配对 bootstrap ΔAUC 95% CI [−0.0030, +0.0210]、P=0.928，ΔF1@0.5 CI [+0.0061, +0.0266]、P=0.998；逐类仅 class9 +0.0077、9 类降，低支持类退化最大；收敛更早（峰值 epoch 3 vs 4）且更低 → 两种分类头结构均无增益，`model.head: standard` 保持默认，不补 seed 2026 配对 |

### 正侧位加权聚合配置

`configs/exp_view_weighted.yaml` 在默认骨干上启用加权聚合；RUN-007 系列使用 DenseNet-121。正位（AP、PA）和侧位（LATERAL、LL）先分别求 logits 均值，再按 0.7 / 0.3 加权；单视角或含未知视角时回退全部图片等权均值。推理及耗时检查须使用相同配置；缺少完整测试索引时，多图加权一致性校验会报错。实测结果见上文及 [训练记录](training_log.md)。

### RUN-010 执行命令

```bash
python src/run_train.py \
  --config configs/exp_freeze_head.yaml \
  --seed 42 \
  --tag freeze_head_s42
```

启动前确认日志包含“预训练权重: 已加载”，并在 `best.pt` 的 `metadata` 中核对 `freeze_backbone: true`、`trainable_params` 小于 `total_params`；实际指标与耗时回填 [training_log.md](training_log.md) 后再做结论。
