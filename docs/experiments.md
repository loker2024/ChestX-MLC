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
| 随机种子 | 42（RUN-002-B、RUN-006-A、RUN-007-A、RUN-007-B、RUN-009-A、RUN008-A/B/C/D、A-001-B(seed 2026) 为 2026、A-001(s216) 为 216） |
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
| A-001(s216) | `A-001_convnext_tiny_s216` | `configs/exp_A001_seed216.yaml`（`_base: exp_A001.yaml`） | 12（早停 4，实际 10） | 是 | A-001 的 seed 216 复核 + 预算/正则三项覆盖：`seed` 42 → 216、`train.epochs` 8 → 12、`train.early_stop_patience` 2 → 4、`train.weight_decay` 0.01 → 0.02，其余口径逐字段不变；已完成（0.7947 / 0.4235，未过 0.8088 门） |
| A-FUSE-1 | `test_fuse4_prob_w1111`（推理融合，无新训练） | `src/run_ensemble.py --fusion probability`，四模型等权 1:1:1:1（A-001 / A-004-b / RUN-009-A / A-001-D） | — | — | 既有 A 组 `best.pt` 的 valid 概率融合选型与提交通道：手动 9 组 + 6 模型权重网格 249 组，最终四模型等权融合 valid 0.8116 / F1@0.5 0.4416（+0.0078 vs A-001，过 0.8088 门），产出 test 提交文件；排除 B-001 / RUN-008-D |
| A-FUSE-2 | `test_fuse3_final`（推理融合，无新训练） | `src/run_ensemble.py --fusion probability`，三模型等权 1:1:1（A-001 / C-005-c / RUN-009-A） | — | — | C 系列入池后的融合重选型：33 候选、贪心三空间 + 穷举 + 权重精修 + 患者整簇 bootstrap，三条路径独立收敛到同一三元组；valid 0.8148 / F1@0.5 0.4420，相对 A-001 +0.0109（区间不跨 0，显著）、相对 A-FUSE-1 +0.0033（跨 0）；S4 加 A-003 边际 +0.0008 且 F1 掉 0.008 → 不采纳 |
| A-FUSE-3 | `test_fuse4_c005c`（推理融合，无新训练） | `src/run_ensemble.py --fusion probability`，四模型等权 1:1:1:1（A-001 / A-004-b / RUN-009-A / C-005-c） | — | — | A-FUSE-1 的成员替换选型：A-001-D → C-005-c、保留 A-004-b；valid 0.8148 / F1@0.5 0.4417（+0.0032 vs A-FUSE-1，区间跨 0）；追加式第五成员 0.8138 / 0.8127 均更低，说明替换优于追加；与 A-FUSE-2 精确差 −0.000038、bootstrap P(Δ>0)=0.487 → 等价替代而非改进 |
| A-FUSE-4 | `test_fuse3_ema_top3`（推理融合，无新训练） | `src/run_ensemble.py --fusion probability`，三模型等权 1:1:1（D-002-a-EMA / D-004-b-EMA / D-004-c-EMA） | — | — | 直接取当日单模型 valid AUC 前三名组合的检验：valid 0.8121 / F1@0.5 0.4289（9 类 0.8198），平台 test **77.36**；相对 A-FUSE-2 valid −0.0029（患者整簇 bootstrap CI [−0.0103, +0.0044]、P(Δ>0)=0.205，跨 0）→ 三成员同为 ConvNeXt-Tiny + EMA 同源变体（相关 0.971–0.979），增益几乎全落在 class1，不作为提交方案 |
| A-FUSE-5 | `k3_D002a_C005c_E001`（推理融合，无新训练） | `src/run_ensemble.py --fusion logit`，三模型权重 4:2:3（D-002-a-EMA / C-005-c / E-001） | — | — | k3 导出包（成本优化后的三模型组合）的 test 融合提交：valid 0.8171 / F1@0.5 0.4333（9 类 0.8249，阈值峰值 0.61 / 0.4526）；同成员等权概率 0.8168 → 权重精修仅 +0.0003 且 F1@0.5 反降 0.0019；相对 A-FUSE-2 +0.0023（9 类 +0.0036）、相对 A-FUSE-4 +0.0050（9 类 +0.0051），首个 10 类与 9 类口径同时超 A-FUSE-2 的融合；**平台 test 77.34**，与 A-FUSE-4 的 77.36 差 −0.02 → valid 提升未传导到 test；未做患者整簇配对 bootstrap，权重在同一 valid 上精修带选型偏差 |
| A-FUSE-5 耗时优化 | `cost_run/afuse5`（工程测试，无新训练） | `src/cost.py` 的 O1 线程池复用；`--precision fp32 --preprocess-workers 6`，V100-SXM2-32G | — | — | 第 7 条耗时的 O1–O4 对照（见 [cost.md §9.3](cost.md)）：基线 82.19 ms/study → O1 **78.69**（中位 −3.5 ms、−4.3%，5 次全部 ≤90、最慢 78.85）；O2 相对 O1 −1.67 ms 未过 2 ms 门已回退；O3 跳过（D2H 0.12 ms）；线程 1/2/4/6/8 为 101.51/81.94/78.45/78.69/78.72，4–8 等价、默认保持 6；35 次运行一致性全 0、最大概率差恒 2.887e-4；**T4 未验证**，数字不可外推 |
| A-FUSE-6 | `test_fuse4_k4_790_800`（推理融合，无新训练） | `src/run_ensemble.py --fusion probability`，四模型等权 1:1:1:1（D-006-b-EMA / D-002-a-EMA / C-005-c / RUN-009-A）；成员由 `outputs/fusion_exhaustive.py` 从 25 项池（ConvNeXt ≥0.80 / 其他 ≥0.79、去重复与 B/T、同实验有 EMA 不取 raw）穷举 k=2..5 选出 | — | — | k=4 等权 prob 融合的穷举 top-1：valid 0.8168 / F1@0.5 0.4425，`--split valid` 复算逐位一致；**k=5 穷举 53,130 组后 top-1 与 k=4 同为 0.8168 → k 有效上限为 4**；相对 A-FUSE-2 +0.0020、相对 A-FUSE-5 加权 4:2:3 −0.0003（等权未打过加权）；耗时 93.6 ms/study（V100 fp32，余量 6.3%）、权重 404.1/500 MB；已出 test 提交，未提交平台 |
| C-001-a | `C-001-a` | `configs/exp_C001a.yaml`（完整展开） | 12（早停 4，实际 7） | 是 | 四骨干低学习率对照之一：ConvNeXt-Tiny `fb_in1k`，骨干 `train.lr=5e-5` / 分类头 `train.head_lr=1.5e-4`；已完成（0.7919 / 0.4191，未过 0.8088 门） |
| C-001-b | `C-001-b` | `configs/exp_C001b.yaml`（完整展开） | 12（早停 4，实际 7） | 是 | 同 C-001-a 口径，骨干换 EfficientNet-B2 `ra_in1k`（唯一变量）；已完成（0.7774 / 0.3996，未过 0.8088 门） |
| C-001-c | `C-001-c` | `configs/exp_C001c.yaml`（完整展开） | 12（早停 4，实际 7） | 是 | 同 C-001-a 口径，骨干换 EfficientNetV2-S RW `ra2_in1k`（唯一变量）；已完成（0.7807 / 0.4055，未过 0.8088 门） |
| C-001-d | `C-001-d` | `configs/exp_C001d.yaml`（完整展开） | 12（早停 4，实际 7） | 是 | 同 C-001-a 口径，骨干换 DenseNet-121 `ra_in1k`（唯一变量）；已完成（0.7849 / 0.4049，未过 0.8088 门） |
| C-002-a | `C-002-a` | `configs/exp_C002a.yaml`（完整展开） | 12（早停 4，实际 9） | 是 | ConvNeXt-Tiny 统一学习率基线 `lr=3e-4`、seed 42；已完成（0.7962 / 0.4218） |
| C-002-b | `C-002-b` | `configs/exp_C002b.yaml`（完整展开） | 12（早停 4，实际 8） | 是 | 同 C-002-a 仅 `lr=1.5e-4`（seed 42 配对）；已完成（0.7976 / 0.4249） |
| C-002-c | `C-002-c` | `configs/exp_C002c.yaml`（完整展开） | 12（早停 4，实际 9） | 是 | 同 C-002-a 仅 seed 2026；已完成（0.7958 / 0.4300） |
| C-002-d | `C-002-d` | `configs/exp_C002d.yaml`（完整展开） | 12（早停 4，实际 8） | 是 | 同 C-002-c 仅 `lr=1.5e-4`（seed 2026 配对）；已完成（0.7956 / 0.4275） |
| C-002-b-EXTRA | `C-002-b-EXTRA` | `configs/exp_C002b_extra.yaml`（完整展开） | 8（早停 2，实际 7） | 是 | C-002-b 的短预算复核：`epochs` 12→8、`patience` 4→2；已完成（0.7958 / 0.4431），改 epochs 同时缩短 cosine 周期，非纯 patience 对照 |
| C-002-e | `C-002-e` | `configs/exp_C002e.yaml`（完整展开） | 12（早停 2，实际 9） | 是 | C-002-a 的学习率上扫：`lr` 3e-4→6e-4、`patience` 4→2；已完成（0.7844 / 0.4132），方向否决 |
| D-001-a | `D-001-a` | `configs/exp_D001a.yaml`（完整展开） | 8（早停 2，实际 7） | 是 | A-001 的五项组合对照：`size` 320→384、`brightness`/`contrast` 0.15→0.25、`frontal_weight` 0.7→0.8、`lr` 3e-4→1.5e-4，seed 42；已完成（0.7988 / **0.4401**，未过 0.8088 门） |
| D-001-b | `D-001-b` | `configs/exp_D001b.yaml`（完整展开） | 8（早停 2，实际 7） | 是 | D-001-a 的 seed 复核：同配置、`seed` 42→2026，其余逐字段一致；已完成（0.7973 / 0.4321，未过 0.8088 门） |
| D-002-a | `D-002-a` | `configs/exp_D002a.yaml`（完整展开） | 8（早停 2，实际 7） | 是 | A-001 的 EMA 对照：`train.ema_enabled: true`、decay 0.999，seed 42；原始 `best.pt` 已完成（0.8005 / 0.4222，未过门） |
| D-002-a-EMA | `D-002-a` / `best_ema.pt` | 同 D-002-a，独立复评 EMA 权重 | 同 D-002-a（EMA best 5） | 是 | D-002-a 的独立 EMA 实验：与原始权重共享一次训练，不重复计时；已完成（**0.8100** / 0.4283，相对 A-001 +0.0062，过方向性门） |
| D-002-b | `D-002-b` | `configs/exp_D002b.yaml`（完整展开） | 8（早停 2，实际 6） | 是 | D-002-a 的 seed 2026 复核；原始 `best.pt` 已完成（0.7999 / 0.4150，未过门） |
| D-002-b-EMA | `D-002-b` / `best_ema.pt` | 同 D-002-b，独立复评 EMA 权重 | 同 D-002-b（EMA best 5） | 是 | D-002-b 的独立 EMA 实验：与原始权重共享一次训练，不重复计时；已完成（0.8074 / 0.4269，相对 A-001 +0.0035，未过门） |
| D-003-a | `D-003-a` | `configs/exp_D003a.yaml`（完整展开） | 8（实际 8，best 7） | 是 | A-001 的 Study 注意力聚合对照：`model.aggregation: study_attention`，seed 42；已完成（0.7958 / **0.4494**，AUC 未过门） |
| D-003-b | `D-003-b` | `configs/exp_D003b.yaml`（完整展开） | 8（早停 2，实际 4） | 是 | D-003-a 的 seed 2026 复核；已完成（0.7868 / 0.4068，best 2，AUC 明显退化） |
| D-002-c | `D-002-c` | `configs/exp_D002c.yaml`（完整展开） | 8（实际 8，跑满） | 是 | D-002 EMA 的第三 seed（216），补三点均值；已完成（原始 0.7961 / 0.4331，**EMA 0.8043** / 0.4329，三点均值 0.8072 未过门） |
| D-004-a | `D-004-a` | `configs/exp_D004a.yaml`（完整展开） | 8（早停 2，实际 6） | 是 | EMA 基线上单项复核 `frontal_weight` 0.7→0.8；已完成（原始 0.7971 / 0.4203，EMA 0.8084 / 0.4293，vs 基线 −0.0016 未过门） |
| D-004-b | `D-004-b` | `configs/exp_D004b.yaml`（完整展开） | 8（早停 2，实际 7） | 是 | EMA 基线上单项复核 `image.size` 320→384；已完成（原始 0.7961 / 0.4273，EMA 0.8085 / 0.4323，vs 基线 −0.0015 未过门） |
| D-004-c | `D-004-c` | `configs/exp_D004c.yaml`（完整展开） | 8（早停 2，实际 7） | 是 | EMA 基线上单项复核 `image.contrast` 0.15→0.25；已完成（原始 0.7945 / 0.4287，EMA **0.8096** / 0.4277，vs 基线 −0.0004 未过门） |
| D-004-d | `D-004-d` | `configs/exp_D004d.yaml`（完整展开） | 8（早停 2，实际 6） | 是 | EMA 基线上单项复核 `image.brightness` 0.15→0.25；已完成（原始 0.7961 / 0.4128，EMA 0.8058 / 0.4280，vs 基线 −0.0042 未过门） |
| D-004-e | `D-004-e` | `configs/exp_D004e.yaml`（完整展开） | 8（早停 2，实际 7） | 是 | EMA 基线上单项复核 `frontal_weight` 0.7→0.9（2026-11-02 曾取消，11-04 补跑）；已完成（原始 0.7982 / 0.4289，EMA 0.8079 / 0.4326，vs 基线 −0.0021 未过门） |
| D-002-d | `D-002-d` | `configs/exp_D002d.yaml`（完整展开） | 8（早停 2，实际 8） | 是 | D-002-a 口径仅换骨干预训练权重 `convnext_tiny.fb_in1k`→`fb_in22k_ft_in1k`；已完成（原始 0.7964 / 0.4258，EMA 0.8037 / 0.4284，vs 基线 −0.0063，为 D-002/D-004 六组 EMA 最低） |
| D-005-a | `D-005-a` | `configs/exp_D005a.yaml`（完整展开） | 8（早停 2，实际 3） | 是 | ImageNet 初始化后全程 AUCM（PESG adam lr 3e-4、margin 1.0、v1、类别均衡采样）；**失败**（原始 0.5233 / 0.0000，EMA 0.5403 / 0.0000，输出塌缩） |
| D-005-b | `D-005-b` | `configs/exp_D005b.yaml`（完整展开） | 8（早停 2，实际 3） | 是 | 从 D-002-a BCE `best_ema.pt` warm start 后转 AUCM 微调；**失败**（原始 0.5411 / 0.0000，EMA 0.5825 / 0.0000，起点 0.8100 一个 epoch 掉到 0.5422） |
| E-001 | `E-001` | `configs/exp_E001.yaml`（完整展开） | 8（早停 2，实际 5） | 是 | 换骨干 `efficientnetv2_rw_s.ra2_in1k` + EMA，320 档；已完成（原始 0.7970 / 0.4112，EMA 0.8038 / 0.4307，vs 基线 −0.0062 未过门） |
| E-001-a | `E-001-a` | `configs/exp_E001a.yaml`（完整展开） | 8（早停 2，实际 3） | 是 | 同 E-001 骨干，`image.size` 320→384；已完成（原始 0.7865 / 0.4000，EMA 0.7969 / 0.4137，vs 基线 −0.0131，明显退化） |
| Distill-1-a | `Distill-1-a` | `configs/exp_Distill1a.yaml`（独立完整配置） | 8（早停 2，实际 7） | 是 | 固定融合教师离线蒸馏的配对基线：同代码、同预算、`distill_enabled: false`，seed 42；已完成（raw 0.7987 / 0.4351，EMA 0.8092 / 0.4339，9 类 0.8190） |
| Distill-1-b | `Distill-1-b` | `configs/exp_Distill1b.yaml`（独立完整配置） | 8（早停 2，实际 7） | 是 | 同 a 仅开 `distill_enabled: true`（λ=0.5、T=1，教师 A-FUSE-5 裁剪 logit 4:2:3 缓存）；已完成（raw 0.8024 / 0.4344，EMA 0.8109 / 0.4328，9 类 0.8172；vs a EMA +0.0017，CI [-0.0037, +0.0078] 跨 0，未达 +0.010） |
| D-006-a | `D-006-a` | `configs/exp_D006a.yaml`（独立完整配置） | 8（早停 2，实际 7） | 是 | EMA 早停口径下的 DropPath 对照：`model.drop_path_rate` 0→0.1（其余同 D-002-a）；已完成（raw 0.8009 / 0.4331，EMA 0.8082 / 0.4271，9 类 0.8185；vs D-002-a-EMA −0.0019 / 9 类 −0.0012，CI [-0.0082, +0.0055] 跨 0） |
| D-006-b | `D-006-b` | `configs/exp_D006b.yaml`（独立完整配置） | 8（早停 2，实际 7） | 是 | EMA 早停口径下的学习率终点对照：`train.cosine_end_epoch` null→6，第 6 轮 LR 到 1e-5 并保持；已完成（raw 0.8071 / 0.4425，EMA **0.8108** / 0.4297，9 类 0.8189；vs 基线 EMA +0.0008、raw +0.0067，EMA CI [-0.0042, +0.0059] 跨 0） |
| D-006-c | `D-006-c` | `configs/exp_D006c.yaml`（独立完整配置） | 8（早停 2，实际 7） | 是 | EMA 早停口径下的 Study Mixup 对照：`mixup_alpha` 0.2、`mixup_prob` 0.3，同批内按视角签名配对（**实际覆盖率 0.2448**，低于配置概率）；已完成（raw 0.7988 / 0.4262，EMA 0.8076 / 0.4306，9 类 0.8159；vs 基线 −0.0017 / −0.0024，CI [-0.0077, +0.0038] 跨 0） |
| D-007-a | `D-007-a` | `configs/exp_D007a.yaml`（独立完整配置） | 8（早停 2，实际 6） | 是 | cosine 终点 6 的三 seed 复核（a）：seed 2026，与 D-006-b 同配置只换 seed；已完成（raw 0.7976 / 0.4282，EMA 0.8055 / 0.4218，9 类 0.8147；vs D-002-a-EMA −0.0046，CI [-0.0125, +0.0036] 跨 0） |
| D-007-b | `D-007-b` | `configs/exp_D007b.yaml`（独立完整配置） | 8（早停 2，实际 7） | 是 | cosine 终点 6 的三 seed 复核（b）：seed 216；已完成（raw 0.7960 / 0.4302，EMA 0.8044 / 0.4381，9 类 0.8143；vs 基线 −0.0057，CI [-0.0126, +0.0017] 跨 0）；与 a 的同配置换 seed 配对差 CI [-0.0054, +0.0080]，即 seed 噪声宽 0.0134 |
| D-008-a | `D-008-a` | `configs/exp_D008a.yaml`（独立完整配置） | 8（早停 2，实际 7） | 是 | 骨干容量阶梯 1/3：Tiny→**ConvNeXt-Small `fb_in1k`**，seed 42，早停改 `ema_auc`；已完成（raw 0.7927 / 0.4209，EMA 0.8057 / 0.4333，9 类 0.8161；vs D-002-a-EMA −0.0043，CI [-0.0110, +0.0032] 跨 0） |
| D-008-b | `D-008-b` | `configs/exp_D008b.yaml`（独立完整配置） | 8（早停 2，实际 7） | 是 | 骨干容量阶梯 2/3：Tiny→**ConvNeXt-Base `fb_in1k`**，seed 42；已完成（raw 0.7909 / 0.4181，EMA 0.8021 / 0.4337，9 类 0.8121；**vs 基线 −0.0079，CI [-0.0131, −0.0025] 不跨 0、P=0.002，本项目首个可测负向**；逐类 9 降 1 升） |
| D-008-c | `D-008-c` | `configs/exp_D008c.yaml`（独立完整配置） | 8（早停 2，实际 7） | 是 | 骨干容量阶梯 3/3：Tiny→**ConvNeXt-Large `fb_in1k`**，seed 42；首次尝试 16:29–16:50 被容器 cgroup 43 GiB 上限 SIGKILL（`rc=137`），17:06:35 起由 page-cache 驱逐守护配合重跑，60m53s 完成；已完成（raw 0.7976 / 0.4187，EMA 0.8022 / 0.4385，9 类 0.8137；**vs 基线 −0.0078，CI [-0.0142, −0.0005] 不跨 0、P=0.019**，但 9 类 CI [-0.0123, +0.0010] 跨 0；raw 最优仅 epoch 3） |
| D-009-a | `D-009-a` | `configs/exp_D009a.yaml`（独立完整配置） | 8（早停 2，实际 4） | 是 | D-002-a 的唯一变量对照：训练采样 `study_sampler: patient_count`、`study_sampler_threshold: 5`、`study_sampler_target_share: 0.665`（多检查患者 study 期望占比 18.5%→66.5%），seed 42，早停口径 `ema_auc`（基线 raw）；已完成（raw 0.7729 / 0.3985，EMA 0.7907 / 0.4200，9 类 0.8021；**vs D-002-a-EMA −0.0193，CI [−0.0311, −0.0058]、P=0.003 不跨 0**；valid 分层 ge5/ge10 −0.0167/−0.0105，设计目标未兑现；4 轮即早停，峰值前移到 ep1–2） |
| D-009-b | `D-009-b` | `configs/exp_D009b.yaml`（独立完整配置） | 8（早停 2，实际 4） | 是 | 同 D-009-a，seed 2026（配对 D-002-b）；已完成（raw 0.7619 / 0.3956，EMA 0.7830 / 0.4061，9 类 0.7918；**vs D-002-b-EMA −0.0244，CI [−0.0397, −0.0105]、P=0.002 不跨 0**；分层 ge5/ge10 −0.0194/−0.0152；10 类逐类全降；双 seed 同向） |

短周期基线把 `epochs` 降到 6，并关闭 early stop 和 `pos_weight`，其余超参数继承 `default.yaml`，
这样两个 run 的差异只来自 `pos_weight`。6 个 epoch 不足以跑出最终成绩，但足以判断
「pos_weight 是帮了忙还是拖了后腿」，避免直接为 25 个 epoch 赌一把。

## 执行与结果

执行步骤和判读门见 [训练计划](../archive/train_plan01.md)，通用命令及续训注意事项见 [使用手册](manual.md)。实际运行的配置、环境、指标和耗时记录在 [training_log.md](training_log.md)，对照结果以该文件为准。

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
- **骨干结论修正**：此前用「RUN-005 / RUN-006 未超过 ResNet-18 历史 0.7887 / 0.7903」判断新骨干无增益，但历史 run 是 warmup + `pos_weight: false` 口径，不能与 cosine + `pos_weight: true` 的新骨干作单变量比较。补齐 RUN-003-B 后，同口径下 ResNet-50 / DenseNet-121 的 macro-AUC 分别高 0.0020 / 0.0043，方向有利于新骨干；但单 seed、差距较小，RUN-001 与 001R 同为 seed 42 且代码版本不同，其 0.0045 差值不能作为换 seed 波动，仍**不足以判定骨干优劣** → **暂不切换 `configs/default.yaml` 的 `model.backbone`**。若后续确定候选，还需按 [train_plan04.md](../archive/train_plan04.md) 核验预训练权重链接 / 许可 / SHA-256，并在赛题指定 T4 环境实测模型大小与端到端耗时（当前 `best.pt`：ResNet-50 91 MB、DenseNet-121 28 MB、ResNet-18 44.8 MB）。
- **DenseNet-121 的 seed 复核（RUN-006-A，2026-10-29 下午）**：seed 2026 下 best epoch 5、macro-AUC **0.7913**、F1@0.5 **0.4103**，第 10 轮早停（34m14s，与 experiment4 的 EfficientNet-B2 实验并行，耗时不可与 RUN-006 的独占 44m32s 直接对照）。对比 RUN-006（seed 42）AUC 高 **+0.0033**；seed 42 的配对骨干差为 +0.0043；0.7913 比 seed 42 的 ResNet-18 高 0.0076，但 seed 不匹配，不能说明骨干优势在第二个 seed 上复现。需补跑同配置 ResNet-18 seed 2026；两次 DenseNet 的 AUC 差为 0.0033，**仍不足以判定骨干优劣**，默认骨干维持 `resnet18`，结论不变。

2026-10-29 下午 RUN-007 正侧位加权聚合对照（数字来自 `outputs/compare_run007/summary.md`；与 RUN-006 同为 DenseNet-121 + seed 42，配置只差聚合方式）：

- **正侧位 7:3 加权聚合未达 AUC 增益门**：best epoch 8，macro-AUC **0.7892**、F1@0.5 **0.4122**，第 13 轮早停（34m56s）。对比 RUN-006（等权均值，0.7880 / 0.4070）：AUC Δ**+0.0012**，未达历史 +0.005 判读门，且小于 DenseNet-121 的 seed 波动（RUN-006 与 RUN-006-A 差 0.0033）→ **单 seed 下不能判定加权聚合有效**。F1@0.5 的提升伴随阈值曲线右移：0.2–0.4 段更低（0.3596 / 0.3785 / 0.3972 vs 0.3674 / 0.3882 / 0.4069），峰从 0.6（0.4098）移到 0.5（0.4122），上限基本持平，属校准/阈值形态而非判别力提升。逐类 AUC 6 升 3 降（class2 −0.0428 最明显、class3 +0.0169、class6 +0.0147、class8 +0.0118），方向不一致。
- **口径说明**：RUN-006 跑在 `da8fef7`（无视角元信息），RUN-007 跑在 `53568fe`（`62b6464` 起增加视角元信息贯通与 `view_weighted` 聚合）；`mean` 路径经 `tests` 与 RUN-006-A 在 `a41033d` 上的实测确认数值等价，但两组仍不是同代码版本的严格单变量对照，不能把差异全部归因于聚合方式。**默认 `model.aggregation` 维持 `mean`**；若继续，需按 [train_plan06.md](../archive/train_plan06.md) 补 seed 2026 复核（RUN-006-A 已有同 seed 的等权对照可配对）。

2026-10-29 下午 RUN-007-A seed 复核（数字来自 `outputs/compare_run007a/summary.md`；按 [train_plan07.md](../archive/train_plan07.md) 执行，seed 2026 的加权版与同 seed 等权版 RUN-006-A 配对比较）：

- **加权聚合在第二个 seed 上方向反转**：RUN-007-A best epoch 7，macro-AUC **0.7906**、F1@0.5 **0.4198**，第 12 轮早停（35m01s）。与 RUN-006-A（同 seed 等权，0.7913 / 0.4103）配对比较：AUC Δ**−0.0007**、F1@0.5 +0.0095。结合 seed 42 的配对差 +0.0012，两 seed 的配对差一正一负、均值 +0.00025，绝对值均远低于 +0.005 判读门 → **正侧位 7:3 加权聚合对 macro-AUC 无稳定增益，判定无效，默认 `model.aggregation` 维持 `mean`**。
- **F1 变化同样不稳定**：seed 42 的 F1 只在 0.5–0.7 段提升（0.2–0.4 段下降），seed 2026 则 0.2–0.7 全线提升 +0.0067~+0.0196，两 seed 形态不同；逐类 AUC 上 seed 2026 为「常见类 5–10 升、稀有类 1–4 降」（class9 +0.0145、class10 +0.0147、class3 −0.0279），与 seed 42 的升降分布也不一致。F1 与逐类的这些差异不能作为判别力提升的证据。

2026-10-29 晚 RUN-007-B 分辨率探针（数字来自 `outputs/compare_run007b/summary.md`；在 RUN-007-A 上只改 384×384 与 6 轮，其余口径相同）：

- **384×384 + 6 轮短跑未见退化，但差异不可单变量归因**：best epoch 5，macro-AUC **0.7959**、F1@0.5 **0.4227**（6 轮跑满，23m42s，显存约 10.2–10.5 GB）。与 RUN-007-A（320×320、12 轮早停，0.7906 / 0.4198）相比 AUC **+0.0053**、F1@0.5 +0.0029；但 007B 的 cosine 周期压到 6 轮、第 5 轮 lr 已降到 8.25e-5 提前收敛，而 007A 第 6 轮 lr 仍有 2.87e-4，两组轮数与调度不同，**不能把 +0.0053 归因于分辨率**。逐类上稀有类 2（+0.0395）、4（+0.0171）贡献主要差值，常见类 5/6/9 略降，6 轮短跑下单类波动大。
- **下一步**：若继续分辨率方向，需用 25 轮完整口径（同 seed、同 cosine 周期、早停 5）复跑 384 并与 320 配对，才能判定分辨率是否有 +0.005 以上的稳定增益；448 需先实测显存与耗时。

2026-10-29 晚 RUN-008 骨干对照（数字来自 `outputs/compare_run008/summary.md`；按 [train_plan08.md](../archive/train_plan08.md) 执行，四组继承 RUN-007-A 的 7:3 加权聚合口径，只换骨干，并统一改为固定 6 轮、关闭早停；A+B / C+D 两两并行执行）：

- **ConvNeXt-Tiny 在 6 轮预算下 macro-AUC 最高**：RUN008-C best epoch 5，macro-AUC **0.8008**、F1@0.5 **0.4382**，比 RUN-007-A（0.7906 / 0.4198）高 AUC **+0.0102**、F1@0.5 **+0.0184**，超过 +0.005 判读门；逐类 8 升 2 降（class3 +0.0319、class4 +0.0275、class1 +0.0186，class8 −0.0113、class10 −0.0036）。但 RUN-007-A 是 25 轮上限、第 12 轮早停（峰值 epoch 7），RUN008-C 是 6 轮无早停预算，两者轮数与 cosine 周期不同、单 seed 且单组 → **+0.0102 只能作为候选方向，不能据此切换默认骨干**。
- **EfficientNet 两组接近但未达门**：RUN008-A（EfficientNet-B2）macro-AUC **0.7952**、F1@0.5 0.4095；RUN008-B（EfficientNetV2-S RW）**0.7954** / 0.4100，分别比 RUN-007-A 高 +0.0046 / +0.0048，均低于 +0.005 判读门，F1@0.5 也低于 RUN-007-A 的 0.4198。逐类上 B 的 class2/class4 提升明显（+0.0451 / +0.0425）但 class8/10 回落，形态与阈值/校准差异混杂。
- **ConvNeXtV2-Nano 组训练异常且许可不合规**：RUN008-D best epoch 2，`run_compare` 重算 macro-AUC **0.4977**、F1@0.5 0.1604（`metrics.csv` 峰值 0.5097 / 0.1604），逐类 AUC 0.46–0.54 近似随机，6 轮 train_loss 仅从 1.183 降到 1.162。已核验骨干 158 个权重张量与预训练文件数值一致（仅 `head.fc` 正常替换为 10 类），单独运行显存 5.9 GB 与 C 组同量级 → 排除权重加载失败与显存问题，原因未定位（疑与 AMP fp16 下 ConvNeXtV2 的 GRN/LayerNorm 组合或该骨干与当前 lr/优化器不匹配）。该权重为 **CC BY-NC 4.0**，赛题《比赛参赛规则》第 1 条明确不接受 NC 类许可证 → 仅研究记录，不参与骨干判读、不能用于提交。
- **口径说明**：四组均为 seed 2026、320×320、`batch_studies: 16`、`pos_weight: true`、固定 6 轮无早停；A/B 于 20:59–21:37 并行，C/D 于 21:38–22:02 并行（D 首次并行运行按用户要求于 21:52 停止并删除，22:02 单独重跑至 22:25）。并行导致耗时不可与其他 run 对照。与 RUN-007-A 的预算差异使「+0.0102」不能当作严格单变量结论；若继续 ConvNeXt 方向，需按 25 轮完整口径重跑并补 T4 端到端耗时与权重许可核验。

2026-10-30 RUN-009-A 384×384 完整口径复核（数字来自 `last.pt` 的 `extras.best_metrics` 与 `run_eval.py`；按 [train_plan09.md](../archive/train_plan09.md) 执行，本次只跑 A 组）：

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
  **仅研究，不用于赛题提交、模型融合或候选选择**（规则第 1 条），计划见 [train_plan12.md](../archive/train_plan12.md)。
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
`outputs/20261031-134732_A-004b_weight_decay_only_s42`，2026-10-31，计划见 [train_plan14.md](../archive/train_plan14.md)）：

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
2026-10-31，计划见 [训练计划 15](../archive/train_plan15.md)）：

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
2026-10-31，计划见 [train_plan16.md](../archive/train_plan16.md)）：

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
[train_plan17.md](../archive/train_plan17.md)）：

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
[train_plan18.md](../archive/train_plan18.md)）

- 配置：以 A-001 为基准只改 `model.aggregation: view_weighted → view_weighted_probability`，正侧位仍 7:3；展开配置逐字段核对仅此一项不同。
- 结果：A-008 19:06:29–19:38:48（32m19s）、第 7 轮早停（best epoch 5），macro-AUC **0.7974** / F1@0.5 **0.4197**。
- 对比 A-001（0.8038 / 0.4255）：ΔAUC **−0.0064**、ΔF1@0.5 −0.0058，距 +0.005 方向性门差 −0.0114，**未过门**。
- 阈值曲线（0.2–0.7）：A-008 0.3735 / 0.3973 / 0.4120 / 0.4197 / 0.4359 / 0.4327，A-001 0.3641 / 0.3835 / 0.4046 / 0.4255 / 0.4328 / 0.4205；低阈值段（0.2–0.4）与 0.6–0.7 段 A-008 略高、0.5 附近更低，为概率空间平均的校准差异，F1@0.5 不作为达门依据。
- 判读：概率空间聚合训练方向否定（单 seed，差值超过 ±0.0033 波动带）；默认聚合保持 `view_weighted`，不做 seed 2026 复核、不进入候选提交。

**A-009：优化超参联合放大（lr 1e-3 / wd 0.03 / 10 轮 / patience 4）对照**（配置 `configs/exp_A009.yaml`，产物
`outputs/20261031-194952_A-009_convnext_tiny_lr1e-3_wd0.03_s42`，2026-10-31，计划见
[train_plan19.md](../archive/train_plan19.md)）

- 配置：以 A-001 为基准只改 `train.lr` 0.0003 → 0.001、`train.weight_decay` 0.01 → 0.03、`train.epochs` 8 → 10、`train.early_stop_patience` 2 → 4，其余口径逐字段不变；四项同改，差值不可拆单项。
- 结果：A-009 19:49:52–20:33:27（43m35s）、10/10 轮跑满无早停（best epoch 10），macro-AUC **0.7799** / F1@0.5 **0.4101**。
- 对比 A-001（0.8038 / 0.4255）：ΔAUC **−0.0239**、ΔF1@0.5 −0.0154，距 +0.005 方向性门差 −0.0289，**未过门**。
- 阈值曲线（0.2–0.7）：A-009 0.3825 / 0.3991 / 0.4053 / 0.4101 / 0.4083 / 0.3925，A-001 0.3641 / 0.3835 / 0.4046 / 0.4255 / 0.4328 / 0.4205；低阈值段（0.2–0.3）A-009 略高，0.5–0.7 段明显更低，F1 峰位由 0.6 左移到 0.5。
- 逐类：A-001 在 10 类中 9 类 AUC 高于 A-009，只有 class7 低 0.0027；降幅最大为 class2（−0.0867，support 37）、class4（−0.0414）、class3（−0.0391）。
- 配对 bootstrap（859 studies × 2,000 次，按 study 有放回重采样，seed 2026）：ΔAUC（A-001 − A-009）95% CI [+0.0121, +0.0360] P=1.000、ΔF1@0.5 95% CI [+0.0027, +0.0285] P=0.991，差距不是 valid 内采样噪声。
- 判读：与「更大 lr + 更强 wd + 更长预算」的组合假设相反（单 seed）；A-004-b 显示 wd 0.01 → 0.05 单独只差 AUC −0.0039（seed 波动内），本组 −0.0239 更可能主要来自 lr 放大后的收敛推迟与过拟合加剧，但四项同改不能单变量归因；默认超参保持 A-001 口径，不补 seed 2026，若继续调参应改开 lr / wd 单变量实验。

**A-009-r2：A-009 的二轮 warm start（以首轮 best.pt 为初始权重、同配置重训 10 轮）**（配置 `configs/exp_A009.yaml` + `src/run_train.py --init-from`，产物
`outputs/20261031-204914_A-009-r2_warmstart_convnext_tiny_lr1e-3_wd0.03_s42`，2026-10-31，计划见
[train_plan20.md](../archive/train_plan20.md)）

- 配置：与首轮 A-009 逐字段相同（lr 1e-3 / wd 0.03 / 10 轮 / patience 4），唯一区别是初始化——不加载 ImageNet 预训练权重，改为加载首轮 `best.pt`（epoch 10，SHA-256 `6988006910b824a83c277764695d0db4a3bdeb3ab0b1b029d22dc9e3ee0f1b15`）；训练从第 1 轮重新开始，不恢复优化器 / 调度器 / 进度。
- 结果：20:49:14–21:13:48（24m34s，V100 与另一容器共享）、第 5 轮早停（连续 4 轮无提升，best epoch 1），macro-AUC **0.7764** / F1@0.5 **0.4117**；逐轮 AUC 0.7764 / 0.7739 / 0.7587 / 0.7692 / 0.7664。
- 对比首轮 A-009（0.7799 / 0.4101）：ΔAUC **−0.0035**、ΔF1@0.5 +0.0016；对比 A-001（0.8038 / 0.4255）：ΔAUC **−0.0274**、ΔF1@0.5 −0.0138，距 +0.005 方向性门差 −0.0324，**未过门**。
- 阈值曲线（0.2–0.7）：A-009-r2 0.3684 / 0.3833 / 0.3966 / 0.4117 / 0.4137 / 0.4022，首轮 A-009 0.3825 / 0.3991 / 0.4053 / 0.4101 / 0.4083 / 0.3925；低阈值段（0.2–0.3）二轮更低、0.5–0.7 段略高，属校准形态变化。
- 逐类：class5 0.8579 / class9 0.8942 / class10 0.8678 相对稳定，class1 0.6993 / class4 0.6764 / class8 0.6973 偏低；未做与首轮 A-009 逐类差异的显著性检验。
- 判读：以首轮 ep10 权重初始化后，lr 重新从 5e-4 warmup 到 1e-3 破坏了已收敛的权重——train_loss 由首轮的 0.6562 反弹到 0.7556，valid AUC 由源权重 0.7799 掉到 0.7764 后连续 4 轮未回升，5 轮即早停；**warm start 二轮无增益**，A-009 首轮的否定结论不变。本轮耗时受 GPU 共享影响（每轮约 6–7 分钟 vs 首轮 4.4 分钟/轮），不可横向比较。

**A-001(s216)：A-001 口径下的 seed 216 复核（+12 轮 / patience 4 / wd 0.02）**（配置 `configs/exp_A001_seed216.yaml`，产物
`outputs/20261031-213405_A-001_convnext_tiny_s216`，2026-10-31）：
- 配置：相对 A-001 同时改四项——`seed` 42 → 216、`train.epochs` 8 → 12、`train.early_stop_patience` 2 → 4、`train.weight_decay` 0.01 → 0.02，其余逐字段不变。
- 结果：21:34:05–22:17:50（43m45s）、第 10 轮后连续 4 轮无提升早停（best epoch 6），valid macro-AUC **0.7947** / F1@0.5 **0.4235**（`run_eval.py` 重算）。
- 阈值曲线（0.2–0.7）：0.3768 / 0.3992 / 0.4127 / 0.4235 / 0.4245 / 0.3960，F1 峰在 0.6；低阈值段高于 A-001、0.4 以上低于 A-001，属校准形态变化。
- 逐类：4 升（class4 +0.0264 最大）6 降（class1 −0.0494 最大，support 24），没有单类崩掉。
- 判读：AUC 相对 A-001 差 **−0.0091**、F1@0.5 差 **−0.0020**，距 0.8088 门差 −0.0141 → **未过门**；ep6 达峰后 valid_loss 由 0.9631 反弹到 2.0729，12 轮预算未带来后段增益、wd 0.02 未推迟过拟合；四项同改不可单变量归因，方向否定、默认口径不变（seed 42 / 8 轮 / patience 2 / wd 0.01）。

**A-FUSE-1：valid 多模型概率融合选型与提交（推理融合，不新增训练）**（脚本 `src/run_ensemble.py`，valid 产物 `outputs/ensemble/valid/`，test 产物 `outputs/ensemble/test_fuse4/submission.csv`，2026-10-31）：
- 目的：不再新增训练，直接用既有 A 组 `best.pt` 做融合，检验多模型融合能否跨过 0.8088 判读门并产出提交文件；按赛题规则排除 B-001（TorchXRayVision 医学预训练，仅研究用、不接入融合）与 RUN-008-D（ConvNeXtV2-Nano，权重 CC BY-NC 4.0）。
- 候选与单模型 valid 基准（本轮统一在 valid 859 studies 上重算、宏观 AUC）：A-001 0.8038、A-005_baseline 0.8022、A-004-b 0.7999、A-008 0.7974、A-001-B(ASL, s42) 0.7970、A-001(s216) 0.7947、RUN-009-A 0.7945、A-001-D(mean 聚合) 0.7931；其余 A 组模型本轮未做融合（无逐 study 概率缓存），A-001-C 冻结骨干 0.7264 不作候选。
- 概率缓存：`outputs/ensemble/valid/single_<tag>/predictions.csv` 共 11 个，含 Study_id + p1…p10；单模型 valid 推理约 50 秒。
- 手动组合（概率空间加权平均，权重按模型顺序给出，valid macro-AUC / F1@0.5）：A-001+RUN-009-A+A-001-D w111 0.8100 / 0.4409；同组 w211 0.8104 / 0.4389；A-001+A-005_baseline+RUN-009-A+A-001-D w1111 0.8106 / 0.4401；A-001+A-005_baseline+RUN-009-A+A-001-D+A-001(s216) w11111 0.8108 / 0.4446；A-001+A-004-b+RUN-009-A+A-001-D w1111 **0.8116 / 0.4416**（最终采用）；A-001+A-008+RUN-009-A+A-001-D w1111 0.8112 / 0.4455；A-001+A-005_baseline+A-004-b+RUN-009-A+A-001-D w11111 0.8115 / 0.4411；六模型等权 0.8113 / 0.4406；A-001+A-001-B(ASL)+RUN-009-A+A-001-D 0.8107 / 0.4447。
- 权重网格：6 模型（A-001 / A-005_baseline / RUN-009-A / A-001-D / A-004-b / A-008）、权重 0/1/2、至少 2 个模型参与，共 249 组合法组合；top1 0.8124（A-001:2、A-005_baseline:1、RUN-009-A:2、A-004-b:2、A-008:1）/ F1@0.5 0.4402，相对等权四模型 0.8116 只高 **0.0008**，在 valid 噪声内（单 seed 波动约 ±0.0033）→ **不采用网格最优权重**，避免在 valid 上过拟合权重。
- 判读：四模型等权融合 valid macro-AUC **0.8116** / F1@0.5 **0.4416**，相对 A-001（0.8038 / 0.4255）ΔAUC **+0.0078**（过 0.8088 门 +0.0028）、ΔF1@0.5 +0.0161；相对历史最佳融合 fuse3（A001+run009a+A001d w211，0.8104 / 0.4389）ΔAUC +0.0012；四模型覆盖两种骨干（ConvNeXt-Tiny @320、EfficientNetV2-S @384）与两种正则/聚合差异（wd 0.05、mean 聚合），等权比权重搜索更稳健 → 融合作为**提交通道**可用，单模型仍是口径基线。
- 限制：融合增益未做配对 bootstrap 显著性检验；权重网格与选型都在同一份 valid 上做，top1 数字不是无偏估计；test 侧只有提交文件、无标签，融合在 test 上的增益无法验证；四模型融合推理成本（test 1414 studies 约 112 秒）随模型数线性增加。
- 产物：`outputs/ensemble/test_fuse4/submission.csv`（14,140 行 = 1414 studies × 10 类，概率区间 0.0019–0.9955，UTF-8），`outputs/eval_reports/` 下不另存。

**A-FUSE-2：C 系列入池后的融合重选型与配对 bootstrap（推理融合，无新训练）**（脚本 `outputs/analysis_fusion_search.py`、`outputs/analysis_final_choice.py`，test 产物 `outputs/ensemble/test_fuse3_final/submission.csv`，2026-11-01）：

- 目的：C-001～C-006 共 21 组新模型完成后，重做融合选型；同时补上 A-FUSE-1 遗留的两项——**配对 bootstrap 显著性检验**与**穷举搜索**，避免贪心局部最优与手工挑组合。
- 候选池：33 个（A 组 15 + C 组 18），全部在 valid 859 studies 上重算。按规则排除 B-001（TorchXRayVision 医学预训练，仅研究用）与 A-001-C（冻结骨干 0.7264，不作候选）。C-006-d 在选型时尚未完成（11-01 首跑被实例重启打断），不入池；它后来于 2026-11-02 重跑完成（0.8009 / 0.4371），**本节及 A-FUSE-3 的 test 提交均基于重跑前的 33 池，未包含 d**，纳入需重跑一次筛选。C-006-a/b/c 为本次新导出。
- 单模型头名（AUC / F1@0.5）：A-001 0.8038 / 0.4255、C-005-c 0.8029 / 0.4291、A-006_baseline 与 A-005_baseline 并列 0.8022、C-003-a 0.8020 / 0.4405、C-005-d 0.8019、C-006-b 0.8014。C 系列没有单模型越过 A-001。
- 方法：① 贪心前向选择（prob / logit / rank 三种空间，接受增益门 0.0005）；② 穷举 2 模型 630 组、3 模型 7140 组等权；③ 坐标上升精修权重（网格 0 / 0.5 / 1 / 1.5 / 2 / 3）；④ 患者整簇配对 bootstrap（228 名患者，2000 次，种子 2026）。
- 关键结果（valid macro-AUC / F1@0.5）：

  | 方案 | 成员 | AUC | F1@0.5 |
  | --- | --- | --- | --- |
  | S1 | A-001（单模型基线） | 0.8038 | 0.4255 |
  | S2 | A-001 + C-005-c | 0.8122 | 0.4389 |
  | **S3（采用）** | **A-001 + C-005-c + RUN-009-A** | **0.8148** | **0.4420** |
  | S4 | S3 + A-003 | 0.8156 | 0.4340 |
  | S5 | A-FUSE-1（历史提交，A-001 + A-004-b + RUN-009-A + A-001-D） | 0.8116 | 0.4416 |
  | S6 | fuse3 w211（A-001 + RUN-009-A + A-001-D，2:1:1） | 0.8104 | 0.4389 |

- 穷举与贪心一致：3 模型最优为 A-001 + C-005-c + RUN-009-A（0.8148），2 模型最优为 A-001 + C-005-c（0.8122）；贪心在三种融合空间都收敛到同一三元组，说明不是单一路径的偶然。
- 权重精修无效：坐标上升只把 0.8156 抬到 0.8161（+0.0005），最优解仍退回等权附近（0.3/0.3/0.2/0.2）→ **不采用权重搜索**，等权保持。
- 显著性（S3 相对各基线，患者整簇 bootstrap）：vs S1 单模型 ΔAUC **+0.0109**，95% CI [+0.0031, +0.0182]，P(Δ>0)=0.996，**不跨 0**；vs S5 历史提交 ΔAUC +0.0033，CI [−0.0018, +0.0076]，P=0.906，**跨 0**；vs S4 加 A-003 ΔAUC −0.0010，CI 跨 0。
- 逐类（S3 vs S5）：7 升 3 降，class1 +0.0156（support 24）、class3 +0.0092、class5 +0.0082 领先，class4 −0.0055 最大跌幅，无单类崩掉。
- 判读：**S3 采用为提交方案**。融合相对单模型的增益是显著的（+0.0109，区间不跨 0），相对 A-FUSE-1 的 +0.0033 方向一致但**未达显著**——只能表述为「不劣于历史提交、点估计更高」，不能声称确定改进。S4 多出的 A-003（单模型 0.7808）边际仅 +0.0008 且 F1 掉 0.008、bootstrap 区间跨 0 → 不采纳，模型数维持 3 个以控制推理成本。C-005-c（正侧位聚合权重 0.8）与 RUN-009-A（EfficientNetV2-S @384）是两个主要增益来源，前者说明**聚合口径差异**与骨干差异同样有效。
- 限制：全部选型在同一份 valid 上完成，S3 的 0.8148 是**带选择偏差的点估计**，不是无偏性能预期；bootstrap 只覆盖 valid 内抽样不确定性，不含训练随机性；A-003 / C-006 等弱模型的入池判决同样受该偏差影响；test 侧无标签，融合增益无法验证；三模型 test 推理耗时 97 秒（1414 studies）。
- 产物：`outputs/ensemble/test_fuse3_final/submission.csv` 与 `outputs/predictions/submissions/submission_A-FUSE-2_A001-C005c-RUN009A_prob_w111.csv`（14,140 行 = 1414 studies × 10 类，概率区间 0.0021–0.9968，UTF-8），正式提交件为 `outputs/submission_A-FUSE-2_A001-C005c-RUN009A_prob_w111.csv`，`outputs/submission.csv` 保留为指向它的软链以兼容 `run_infer.py` 默认路径；校验：列序 `Subject_id, Study_id, Predict_class, Probability`、无重复键、无缺失、study 集合与 A-FUSE-1 提交一致；与 A-FUSE-1 提交的逐类 Spearman 均值 0.9667（最小 0.9334），属同源模型的合理偏移。

**A-FUSE-3：A-001-D → C-005-c 的成员替换与 C-003-a 对照（推理融合，无新训练）**（脚本 `src/run_ensemble.py`，test 产物 `outputs/ensemble/test_fuse4_c005c/submission.csv`，2026-11-01）：

- 目的：A-FUSE-2 的搜索是「往池子里加成员」，本组改测「换成员」——A-FUSE-1 四个成员中哪个可被 C-005-c 顶替。基线固定为 A-FUSE-1（A-001 / A-004-b / RUN-009-A / A-001-D），四个方案概率空间等权。
- 结果表（valid 859 studies，患者整簇配对 bootstrap 2000 次、种子 2026，全部相对原方案）：

  | 方案 | 成员 | valid AUC | F1@0.5 | ΔAUC | 95% CI |
  | --- | --- | ---: | ---: | ---: | --- |
  | 原方案 A-FUSE-1 | A-001 / A-004-b / RUN-009-A / A-001-D | 0.8116 | 0.4416 | — | — |
  | 新四模型（采用） | A-001 / A-004-b / RUN-009-A / C-005-c | 0.8148 | 0.4417 | +0.0032 | [−0.0016, +0.0071] |
  | 新五模型 | 原四模型 + C-005-c | 0.8138 | 0.4426 | +0.0022 | [−0.0003, +0.0047] |
  | C-003 对照 | 原四模型 + C-003-a | 0.8127 | 0.4447 | +0.0011 | [−0.0013, +0.0035] |

- 判读：**替换优于追加**。C-005-c 顶替 A-001-D 得 0.8148，而把 C-005-c 或 C-003-a 作为等权第五成员只得到 0.8138 / 0.8127——多一个成员会摊薄其余模型权重，A-001-D 正是那个应当让位的成员（与 A-FUSE-2 中 A-001-D 在 fuse3 w211 里权重被下调的方向一致）。相对 A-FUSE-1 的 +0.0032 区间跨 0，只能表述为不劣于历史提交。
- 与已提交方案的关系：A-FUSE-2 的三模型精确值 0.814837，本方案 0.814798，差 −0.000038；配对 bootstrap CI [−0.0026, +0.0026]、P(Δ>0)=0.487 → **统计上不可区分**。两者四舍五入同为 0.8148，因此本方案不构成对 A-FUSE-2 的改进，属等价替代（多一个成员、多一份推理成本）。
- 产物：`outputs/ensemble/test_fuse4_c005c/submission.csv`，正式提交件 `outputs/submission_A-FUSE-3_A001-A004b-RUN009A-C005c_prob_w1111.csv`，本机副本置于 `outputs/predictions/submissions/`，`outputs/submission.csv` 软链改指本文件；三处 md5 一致。校验：14,140 行 = 1414 studies × 10 类、概率 0.002011–0.995954、无缺失无重复键、study 集合与顺序同 A-FUSE-1 提交；逐类 Spearman 对 A-FUSE-1 均值 0.9613（最小 0.9134）、对 A-FUSE-2 均值 0.9921（最小 0.9859）。四模型 test 推理耗时 117 秒。

**A-FUSE-4：当日单模型 AUC 前三名的直接组合与首个带 test 反馈的融合提交**（脚本 `src/run_ensemble.py`，产物
`outputs/ensemble/test_fuse3_ema_top3/submission.csv`，2026-11-02）

- 目的：不做搜索，直接取 D-002-c / D-004 / E-001 七组完成后 valid 单模型 AUC 的前三名（D-002-a-EMA 0.8100、D-004-c-EMA 0.8096、D-004-b-EMA 0.8085）等权概率融合，检验「强者叠加」是否成立，并产出可提交文件。
- 结果：valid macro-AUC **0.8121** / F1@0.5 **0.4289**（阈值峰值在 0.6，0.4496），9 类口径 **0.8198**。相对单模型最好（D-002-a-EMA 0.8100 / 0.4283）仅 +0.0021；相对现行 A-FUSE-2（0.8148 / 0.4420 / 9 类 0.8213）**−0.0029**。患者整簇配对 bootstrap（228 患者、2000 次、种子 2026）：ΔAUC **−0.0029**、95% CI [−0.0103, +0.0044]、P(Δ>0)=0.205 → 两者统计上不可区分，方向不利于本方案。
- **平台 test 反馈 77.36**（百分制，即 0.7736）。这是本仓库首个记录到平台分数的融合提交，此前 A-FUSE-1 / 2 / 3 只有 valid 与提交文件、无反馈分数，因此无法据此判断 valid 排序在 test 上是否成立。
- 判读：**「AUC 前三名直接相加」不成立。** 三个成员都是 ConvNeXt-Tiny + EMA，只差 `image.contrast`（0.15→0.25）、`image.size`（320→384）两项单一扰动，成员间逐元素预测相关 **0.971–0.979**，融合吃到的是训练抖动而非结构互补。逐类 ΔAUC 显示增益几乎全部来自 class1（**+0.0192**），其余 9 类合计 **−0.0021**（class2 −0.0081 最差）——9 类口径从 0.8196 到 0.8198，等于没有提升。这与 C-005-c、C-006-d、D-004-c 三次「组级正向、剔除 class1 后归零」是同一机制的第四次观测。
- 与 A-FUSE-2 的差距来源可定位：A-FUSE-2 的成员覆盖三种结构差异（ConvNeXt-Tiny 基线 / 正侧位聚合权重 0.8 / EfficientNetV2-S @384），而本方案的三个成员在骨干、聚合口径、损失、增强策略上几乎一致。**融合的增益来自口径多样性，不是单模型分数。**
- 产物：`outputs/ensemble/test_fuse3_ema_top3/submission.csv`，正式提交件 `outputs/submission_A-FUSE-4_D002a-D004b-D004c-EMA_prob_w111.csv`，本机副本置于 `outputs/predictions/submissions/`；三处 md5 一致（`0c79610a81be…`）。校验：14,140 行 = 1414 studies × 10 类、概率区间 0.000730–0.995435、无缺失无重复键、study 集合与顺序同 A-FUSE-2 提交一致；与 A-FUSE-2 逐类 Spearman 均值 0.9289（最小 0.8718）。三模型 test 推理耗时约 100 秒。
- 限制：三成员的选择依据是同一份 valid 的单模型 AUC 排名，带选择偏差；结论只在 valid 859 studies 与本次 test 上成立，不能外推为「同源融合一律无效」——若换入 C-005-c 或 RUN-009-A 这类结构差异成员，同一套三模型框架的 valid 为 0.8154（贪心与穷举一致），仍高于本方案 0.0033。

**A-FUSE-5：k3 导出包的三模型 logit 加权融合与 test 提交**（脚本 `src/run_ensemble.py`，产物
`outputs/ensemble_k3_D002a_C005c_E001/submission.csv`，2026-11-03）

- 目的：`k3_D002a_C005c_E001.tar.gz` 是成本优化后导出的三模型组合包（内含 `cost_ensemble.json` 与 `MANIFEST.json`），本次把包上传到 sjy、同步代码后跑 test 融合推理并回传 `submission.csv`；同时用 `--split valid` 复现导出包声明的 valid 指标，检验 4:2:3 logit 加权能否超过 A-FUSE-2 / A-FUSE-4，并补记平台反馈。
- 结果：valid macro-AUC **0.8171** / F1@0.5 **0.4333**（阈值峰值在 0.61，0.4526），9 类口径 **0.8249**；与 `MANIFEST.json` 的 refined_logit_auc 0.8171 / refined_logit_9class 0.8249 逐位一致。同成员等权概率 0.8168（F1@0.5 0.4352、峰 0.62 / 0.4553）→ 4:2:3 精修只把 AUC 抬了 **+0.0003**，F1@0.5 反而低 0.0019。
- 对比：相对 A-FUSE-2（0.8148 / 0.4420 / 9 类 0.8213）**+0.0023 / −0.0087 / +0.0036**；相对 A-FUSE-4（0.8121 / 0.4289 / 9 类 0.8198）**+0.0050 / +0.0044 / +0.0051**；相对单模型最好 D-002-a-EMA（0.8100 / 0.4283）**+0.0071 / +0.0050**。这是首个在 10 类与 9 类两个口径上同时超过 A-FUSE-2 的融合。
- **平台 test 反馈 77.34**（百分制，即 0.7734）。A-FUSE-4 同平台为 77.36，两者差 **−0.02**（绝对 −0.0002）；而 A-FUSE-5 的 valid 比 A-FUSE-4 高 0.0050 → **valid 上 0.005 量级的提升没有传导到 test**，与「小差距在 test 上不可分辨」的判断一致。A-FUSE-2 仍无平台分数，无法完成 valid→test 的三点校验。
- 判读：成员结构是 A-FUSE-5 相对 A-FUSE-4 的关键差异——D-002-a-EMA 提供 EMA 稳定化、C-005-c 提供正侧位聚合权重 0.8 的口径差异、E-001 换 EfficientNetV2-S RW 骨干；三者分别覆盖稳定性、聚合口径、骨干三类差异，多样性优于 A-FUSE-4 的三个 ConvNeXt-Tiny + EMA 同源变体。valid 0.8171 是三模型组合里的最高值，9 类口径同步到 0.8249（A-FUSE-4 0.8198），说明增益不是 class1 假象（class1 逐类 AUC 0.7463 仍是 10 类最低，但 9 类 macro 高于 10 类 macro）。但 test 77.34 与 A-FUSE-4 的 77.36 打平 → **valid 上的选型收益在 test 上无法体现，融合提交的决策应按多样性与耗时，而非 valid 小数点第三位**。
- 产物：sjy `outputs/ensemble_k3_D002a_C005c_E001/submission.csv`（14,140 行 = 1414 studies × 10 类、概率 0.001211–0.996982、UTF-8），本机副本 `outputs/submission_k3_D002a_C005c_E001.csv`；同目录 `cost_ensemble.json`（`fusion: logit`、`weights: [4,2,3]`、三 ckpt）与 `MANIFEST.json`（三 ckpt md5 与 valid 指标）。校验：14,140 行 + header、字段 `Subject_id,Study_id,Predict_class,Probability` 与历史提交一致、无缺失/重复键、每 study 10 类 1–10、study 集合与 `data/test.csv` 完全一致（1414）。三模型 test 推理约 1 分钟（RTX 3090，非 `cost.py` 计时口径）。
- 限制：未做患者整簇配对 bootstrap（A-FUSE-2/4 的显著性结论不能直接沿用）；4:2:3 权重与成员选择都在同一份 valid 上得到，带选型偏差；`k3` 包是为第 7 条耗时优化导出的组合，最终提交选择还需结合 `docs/cost.md` 的三模型 93.6 ms/study 与四模型临界值一起判断。

**A-FUSE-5 耗时优化：O1–O4 逐项对照**（工程测试，无新训练，产物在远端 `cost_run/afuse5/`，2026-11-04，计划见 [cost_optimization_plan.md](cost_optimization_plan.md)）

- 目的：在保持 submission 预测不变、完整计时口径不变的前提下，把 A-FUSE-5 的单 study 耗时压到 90 ms 以内，为第 7 条的 100 ms 留余量。固定对象为三模型 logit 4:2:3（三 ckpt md5 与 k3 包 `MANIFEST.json` 一致），样本 100 study / 162 图 + 5 个预热 study，V100-SXM2-32G、fp32、workers 6、3 轮。
- 结果：基线中位 **82.19 ms**（5 次 81.63–84.37）→ 最终 **78.69 ms**（5 次 77.40–78.85），中位改善 **−3.5 ms（−4.3%）**，5 次全部 ≤90 ms、余量 21.3 ms。唯一纳入的改动是 **O1 线程池复用**（整次运行一个池，10 对正反序全部同向、中位 −4.3 ms）。O2 设备张量复用相对 O1 中位 −1.67 ms、区间跨 0，未过 2 ms 门已回退；O3 按前置条件跳过（D2H 仅 0.12 ms）；O4 线程 1/2/4/6/8 中位为 101.51 / 81.94 / 78.45 / 78.69 / 78.72，4–8 等价、默认保持 6。
- 数值一致性：35 次运行与 submission 比对全部 0 个 study 不一致，最大概率差恒 2.887e-4（sjy 与 V100 的 GPU 浮点差，编辑前后同值）→ 优化没有改变数值路径。
- 分段诊断：解码占 42%（JPEG，单图 study 无法并行）、三模型前向 37%、预处理 18%、回传与融合合计 0.4%。因此本轮没有动模型与精度。
- 限制：全部在 V100 上测得，**不能外推 T4**（T4 fp32 约为 V100 的 52%、CPU 更少），官方第 7 条以 T4 复测为准；CUDA Graph、`torch.compile`、fp16、量化未评估。完整口径见 [cost.md §9.3](cost.md)。

**A-FUSE-6：25 项池的 k=2..5 全穷举与 k=4 等权融合提交**（脚本 `outputs/fusion_exhaustive.py`、`src/run_ensemble.py`，产物 `outputs/ensemble/test_fuse4_k4_790_800/`，2026-11-06）

- 目的：在「ConvNeXt 骨干 ≥ 0.800 / 其他骨干 ≥ 0.790、非重复、非 B/T 系列、非 ResNet、同实验有 EMA 不取 raw」的 25 项候选池上做全组合穷举，回答两个问题——k 的有效上限在哪、等权融合能否超过已有的加权方案（A-FUSE-5 的 4:2:3，valid 0.8171），并据此产出可提交的 test 融合文件。
- 搜索规模：C(25,2)=300、C(25,3)=2,300、C(25,4)=12,650、C(25,5)=53,130，合计 68,380 组，`prob` 空间、等权、859 studies。
- 结果：k=2 **0.8140**（D-006-b-EMA + C-005-c）、k=3 **0.8160**（+ RUN-009-A）、k=4 **0.8168**（+ D-002-a-EMA）、k=5 **0.8168**（再加 C-003-a，与前一名持平）。前 20 名区间宽度随 k 单调收窄：0.0022 → 0.0014 → 0.0008 → 0.0003。
- 判读：**k 的有效上限是 4**——k=4 与 k=5 的上界完全相同，k=5 那 53,130 组没有带来增益。组合数从 300 涨到 53,130（177 倍）只把 top1 抬了 0.0028，且 k 越大前 20 名越挤在同一平台，是典型的选择偏差平台形态。`C-005-c` 与 `RUN-009-A` 在 k≥3 的 top5 中 100% 出现，为必选成员；`RUN-009-A` 单模型仅 0.7945，靠分数筛选永远进不了前 25，是「异构互补」而非「强者叠加」的又一例证。
- 对比：相对 A-FUSE-2（0.8148）+0.0020、相对单模型最好 D-006-b-EMA（0.8108）+0.0060；相对 A-FUSE-5（0.8171）**−0.0003**，即**等权穷举没打过已有的加权方案**，差值远小于单次 macro-AUC 的 bootstrap SE（≈0.0094），不构成性能差异证据。
- 产物：`outputs/ensemble/test_fuse4_k4_790_800/submission.csv`（14,140 行 = 1414 studies × 10 类、概率 0.001579–0.996250、UTF-8），归档 `outputs/predictions/submissions/submission_A-FUSE-6_k4_D006bEMA-D002aEMA-C005c-RUN009A_prob_w1111.csv`（SHA-256 `407d29fa…`、md5 `9dfcf8fb…`，与训练环境一致）；同目录 `cost_ensemble.json`（`probability`、`[1,1,1,1]`、四 ckpt 用相对清单目录的路径）与 `cost_views.csv`（2,218 行三列元数据）。`run_ensemble.py --split valid` 复算 0.8168 / 0.4425，与穷举 top-1 逐位一致。
- 耗时：`src/cost.py`（V100-SXM2-32G、fp32、3 轮、100 study / 162 图）单 study **93.6 ms**（中位 91.7、p95 118.0、最大 134.5），未超 100 ms 但**余量仅 6.3%**；不带 `--checkpoint` 靠清单自动发现复测 94.3 ms、一致性 100/100。fp16 实测 92.1 ms（仅快 1.6 ms）且与 fp32 提交有 14 个 study 不一致（最大差 0.0020 > 1e-3），**不采用**——瓶颈在 JPEG 解码与文件 IO，压精度无用。权重合计 404.1 MB（上限 500 MB）。
- 限制：top-1 是从 12,650 组里挑出的最高值，带选择偏差，不是无偏性能预期；test 侧无标签，融合增益无法验证；未做患者整簇配对 bootstrap。**93.6 ms 的余量偏小，且 A-FUSE-5 优化记录称「T4 fp32 约为 V100 的 52%」，若官方以 T4 复测，四模型有超时风险，届时应退到 k=3（valid 0.8160、少一个模型的前向）。**

**C-001：四骨干低学习率 + 分类头差异学习率对照**（配置 `configs/exp_C001{a,b,c,d}.yaml`，产物
`outputs/20261101-114223_C-001-a` 等四组，2026-11-01，计划见 [train_plan21.md](../archive/train_plan21.md)）

- 设计：四份配置逐字段一致，唯一变量是 `model.backbone`（ConvNeXt-Tiny `fb_in1k` / EfficientNet-B2 `ra_in1k` / EfficientNetV2-S RW `ra2_in1k` / DenseNet-121 `ra_in1k`）；骨干峰值 `train.lr=5e-5`、分类头 `train.head_lr=1.5e-4`，12 轮上限、patience 4、seed 42 串行独占 V100。
- 结果：四组**全部在 best epoch 3 达峰、epoch 7 触发早停**，无一跑满 12 轮。C-001-a 0.7919 / 0.4191（31m33s）、C-001-b 0.7774 / 0.3996（33m01s）、C-001-c 0.7807 / 0.4055（29m52s）、C-001-d 0.7849 / 0.4049（27m04s）。
- 对比 A-001（0.8038 / 0.4255）：ΔAUC **−0.0119 / −0.0264 / −0.0231 / −0.0189**，距 0.8088 方向性门 −0.0169 / −0.0314 / −0.0281 / −0.0239，**四组均未过门**。注意 A-001 与 C-001 之间同时改了学习率方案、轮数与 patience，差值不能单变量归因于骨干。
- 逐类：C-001-a 3 升 7 降（class8 +0.0185）、C-001-b 1 升 9 降、C-001-c 0 升 10 降、C-001-d 1 升 9 降（class8 +0.0162）；降幅集中在 class1（support 24）、class4、class6、class7，无单类崩掉。
- 融合补充价值：按 A-FUSE-1 公式重算基线得 0.8116 / 0.4416（与记录一致），四个候选作等权第五成员后 ΔAUC 为 −0.0004 / +0.0003 / +0.0005 / −0.0007，患者整簇 bootstrap（2000 次、种子 2026）区间全部跨 0。
- 判读：**方向否定**——分层低学习率没有换来更长训练窗口，反而把峰值压在 warmup 结束的第 3 轮，epoch 4 起 valid_loss 单调反弹（C-001-c 0.9331 → 1.5712）。C-001 不产生提交候选，不改默认骨干与学习率；四组同受早峰压制，本轮不能判定骨干绝对优劣。未执行：第二 seed、逐类 bootstrap、test 推理与 T4 耗时自检、DenseNet-121 权重 SHA-256 单独记录。

**C-002：ConvNeXt-Tiny 统一学习率双 seed 配对**（配置 `configs/exp_C002{a,b,c,d}.yaml`，产物
`outputs/20261101-135010_C-002-a` 等四组，2026-11-01，计划见 [train_plan22.md](../archive/train_plan22.md)）

- 设计：a（3e-4, seed 42）↔ b（1.5e-4, seed 42）、c（3e-4, seed 2026）↔ d（1.5e-4, seed 2026）两组 seed 内配对，每对仅 `train.lr` 不同；AdamW 单参数组（`head_lr: null`）、320×320、12 轮上限、patience 4、warmup 2 + cosine，V100-SXM2-32G 串行独占。
- 结果：0.7962 / 0.4218（a，40m39s）、0.7976 / 0.4249（b，35m48s）、0.7958 / 0.4300（c，39m53s）、0.7956 / 0.4275（d，36m10s）；四组均在 best 后按 `patience 4` 早停，峰值集中在第 4–5 轮。
- 配对判读：Δ42 = **+0.0014**（95% CI [−0.0100, +0.0133]，P>0 = 0.599），Δ2026 = **−0.0002**（[−0.0135, +0.0102]，0.465），平均 **+0.0006**（[−0.0097, +0.0098]，0.538）；患者整簇 bootstrap 2000 次、种子 2026。
- 判读：**学习率不构成有效杠杆**。plan22 / plan23 预置的三项切换条件全部未满足——方向不一致、平均增益仅为 0.005 阈值的 1/8、class1（support 24）在两个 seed 均降 ≥0.02（−0.0618 / −0.0262）。seed 间波动（同学习率换 seed 为 +0.0004 / −0.0020）大于学习率间效应，故 **3e-4 保留为 ConvNeXt-Tiny 320 分辨率默认值**，不改提交口径。四组同受早峰压制，与 C-001 形态一致：train_loss 持续下降而 valid_loss 第 5 轮起单调反弹，说明瓶颈在正则化与数据侧，不在学习率。C-003 分辨率对照暂缓（配置待修订），其学习率取值须重新确认。
- 补充两组：C-002-b-EXTRA（b 的 `epochs` 12→8、`patience` 4→2，31m57s，0.7958 / **0.4431**）与 C-002-e（a 的 `lr` 3e-4→6e-4、`patience` 4→2，39m38s，0.7844 / 0.4132）。短预算组 AUC 与 b 差 −0.0018、在 seed 噪声内，耗时省 3m51s，但其 F1 抬升主要来自 best.pt 恰好落在 AUC 几乎相同的第 5 轮（选点效应），且 `epochs` 变短同时缩短了 cosine 周期，**不是纯 patience 对照**；上扫组从 epoch 1 即退化，比 a 低 0.0118 → **1.5e-4–3e-4 是合理区间，3e-4 为该区间上界**。

**口径告警：C-003~C-006 的组级 ΔAUC 由 class1 主导**（2026-11-01 补算，适用于其后四个小节）

- 现象：以 C-002-b 为基线（10 类 macro-AUC 0.7976），C-003~C-006 共 10 组里有 6 组高于基线，最大 +0.0053。但**剔除 class1（support 24）、只对 9 类求平均后，10 组全部低于基线**（Δ9 从 −0.0006 到 −0.0090）。
- 原因：C-002-b 的 class1 AUC = **0.6609**，是 C-002-b/d、C-003-a/b、C-004、C-005-a~d、C-006-a~c 共 12 组里最低的一个，其余落在 0.6676–0.7464、中位约 0.710。单类提升 0.05 除以 10 恰为 0.005，**与判读门同量级**，因此任何以 C-002-b 为基线的组级 ΔAUC 都不具备机制含义。
- 同一效应对「A-001 是单模型最高」也成立：A-001 的 10 类 0.8038 高于 C-002-b（+0.0062），但 9 类只有 0.8105、**低于 C-002-b 的 0.8128**（−0.0023）；A-005_baseline（0.8101）、C-003-a（0.8122）、C-005-c（0.8092）同样如此。差距全部来自 class1（A-001 0.7438 vs C-002-b 0.6609）。
- 口径：主指标仍是赛题定义的 10 类 macro-AUC，各节按 10 类排序的结论不变；但**不得把 +0.004~+0.005 的组级差异解读为「分辨率 / 聚合权重 / 增强强度带来了机制收益」**——它在 9 类口径下反号。判断机制应看 9 类 AUC 与逐类稳定性；若要抬 10 类指标，小 support 类才是杠杆，而 859 studies 的单次评估不足以支撑切换决策。
- 影响边界：不影响 A-FUSE-2 / A-FUSE-3 的融合结论——融合增益是相对 A-001 的实测患者整簇配对 bootstrap（+0.0109，区间不跨 0），多模型平均本身会压低单类噪声。但融合选型若只按 10 类 macro-AUC 挑成员，会系统性偏好 class1 偶然偏高的模型，后续选型须用 9 类口径复核。

**C-003：ConvNeXt-Tiny 输入分辨率 320 → 384 对照**（配置 `configs/exp_C003{a,b}.yaml`，产物
`outputs/20261101-174052_C-003-a`、`outputs/20261101-181438_C-003-b`，2026-11-01，计划见 [train_plan23.md](../archive/train_plan23.md)）

- 设计：`image.size` 320→384，两 seed 各一组（a seed 42、b seed 2026），其余按 C-002-b 口径固定：AdamW 单参数组、峰值 lr 1.5e-4、12 轮上限、patience 2、warmup 2 + cosine、view_weighted 0.7/0.3。首轮 `train_plan23.md` 定的 3e-4 口径在学习率判读后由用户改为 C-002-b 口径，故两组均从 1.5e-4 起训，非从 A-001 checkpoint 热启动。
- 结果：C-003-a **0.8020 / 0.4405**（best epoch 5，7 轮早停，17:40:52–18:14）、C-003-b **0.7969 / 0.4448**（best epoch 5，7 轮早停，18:14:38–18:48）。阈值曲线峰均落在 **0.5**（a 0.3846/0.4092/0.4239/**0.4405**/0.4397/0.4364，b 0.3940/0.4163/0.4311/**0.4448**/0.4351/0.4153）。
- 判读：以同 seed 的 320 组为对照，ΔAUC 为 **+0.0044**（a vs C-002-b 0.7976）与 **+0.0013**（b vs C-002-d 0.7956），均未达 +0.005 方向性门；但对 A-001（0.8038）a 仍低 0.0018。**真正显著的是 F1@0.5：+0.0156 / +0.0173**，幅度是 AUC 变化的 3–13 倍，且两组峰位都稳定在 0.5、无校准漂移 → 384 抬高的是阈值处的判别质量（precision/recall 平衡），不是整体排序能力。（9 类口径：a **−0.0006**、b **−0.0041**，见上方告警——AUC 层面的「提升」不成立；F1@0.5 的抬升不受 class1 影响，仍是真实观察。）
- 限制：对照组 patience 4、本组 patience 2，属**非纯分辨率对照**（`train_plan23.md` 已预先声明该差异）；每 seed 仅一组，无重复；未做配对 bootstrap；384 的显存峰值与单轮耗时未单独记录，不能据此估算算力代价；同受早峰压制（第 5 轮达峰、第 7 轮早停），与 320 组形态一致。

**C-004：ConvNeXt-Tiny 平方根类别权重**（配置 `configs/exp_C004.yaml`，产物 `outputs/20261101-185032_C-004`，2026-11-01，计划见 [train_plan24.md](../archive/train_plan24.md)）

- 设计：`train.pos_weight_power` 1.0 → **0.5**（BCE 正样本权重由 `负/正` 改为其平方根，权重只按训练集 study 标签统计），其余同 C-002-b 口径。配置首行注释仍写「与 A-001 展开配置仅 `pos_weight_power` 不同」，但字段值已是 lr 1.5e-4 / 12 轮 / patience 2 —— **以字段值为准，注释属未同步**。
- 结果：**0.7950 / 0.4125**（best epoch 4，6 轮早停，18:50:32–19:18）。阈值曲线 **0.3967 / 0.4198 / 0.4362 / 0.4125 / 0.3894 / 0.3269**，峰在 **0.4** 而非 0.5。
- 判读：AUC 相对 C-002-b（0.7976）为 **−0.0026**，排序能力未改善。但 F1 曲线形态发生整体左移：F1@0.5 从 0.4249 掉到 0.4125（−0.0124），而 F1@0.4 达 0.4362、比基线的 F1@0.5 还高 **+0.0113**。**这是校准变化，不是判别能力变化**——把 0.5 处的下降读成「变差」、或把 0.4 处的升高读成「变好」，都是同一类误读；判据只能是 AUC 与逐类排序。（9 类口径 **−0.0090**，见上方告警：平方根权重在排序上是净损失，并非换一个阈值就能追平。）
- 限制：单 seed；未做配对 bootstrap；校准结论仅由阈值曲线峰位与两侧陡降推得，未出可靠性图或 ECE，也没有 logit 分布对比；峰位左移对提交无实质影响（提交输出全 10 类概率、不做阈值过滤）。

**C-005：ConvNeXt-Tiny 正侧位聚合权重四组对照**（配置 `configs/exp_C005{a,b,c,d}.yaml`，产物
`outputs/20261101-192302_C-005-a` 等四组，2026-11-01，计划见 [train_plan25.md](../archive/train_plan25.md)）

- 设计：`model.frontal_weight` 0.7 → **0.5 / 0.6 / 0.8 / 0.9** 四组（侧位权重取补数），其余同 C-002-b 口径，全部从原始预训练权重全量训练、不热启动。四份配置展开后逐字段一致，唯一变量是聚合权重。
- 结果（valid 859 studies，宏平均）：

  | 组 | frontal:lateral | macro-AUC | F1@0.5 | 峰值阈值 / F1 | best epoch | 轮数 |
  | --- | --- | ---: | ---: | --- | ---: | ---: |
  | C-005-a | 0.5 : 0.5 | 0.7991 | **0.4365** | 0.6 / 0.4412 | 5 | 7 |
  | C-005-b | 0.6 : 0.4 | 0.7981 | 0.4218 | 0.6 / 0.4375 | 4 | 6 |
  | **C-005-c** | **0.8 : 0.2** | **0.8029** | 0.4291 | 0.6 / 0.4365 | 5 | 7 |
  | C-005-d | 0.9 : 0.1 | 0.8019 | 0.4191 | 0.6 / 0.4309 | 4 | 6 |
  | C-002-b（0.7 : 0.3 基线） | 0.7 : 0.3 | 0.7976 | 0.4249 | 0.6 / — | 4 | 6 |

- 判读：AUC 与权重**不单调**（0.5→0.7991、0.6→0.7981、0.7→0.7976、0.8→0.8029、0.9→0.8019），峰值落在 0.8 与 0.9 两个高正位权重、且 0.8 最高。C-005-c 相对 C-002-b **+0.0053，过 +0.005 方向性门**，但对 A-001（0.8038）仍 **−0.0009**，不能算超过历史最佳单模型。（9 类口径 **−0.0036**，见上方告警：这次「过门」由 class1 造成，不能作为机制证据；C-005-c 的实际价值在融合实测中，不在单模型 AUC。）F1@0.5 与权重呈反向松散关系（0.5 最高 0.4365、0.9 最低 0.4191），四组峰位统一在 0.6、无校准漂移。
- 后续价值：C-005-c 是后续融合的关键成员——它贡献的是**聚合口径差异**（正侧位加权不同）而非骨干差异，与 RUN-009-A（EfficientNetV2-S @384）一起构成 A-FUSE-2 的两个主要增益来源。A-FUSE-2 / A-FUSE-3 的成员选择见本文件相应小节。
- 限制：单 seed 四组，权重-性能关系可能被 seed 噪声主导（0.8 与 0.9 只差 0.0010）；逐类对比见 `outputs/comparisons/compare_C005/`（`summary.md` 的 per-run 表与逐类表），四组 `valid_ids_sha256` 一致可核；未做统一 bootstrap 的多重比较校正。

**C-006：ConvNeXt-Tiny 亮度与对比度扰动四组对照**（配置 `configs/exp_C006{a,b,c,d}.yaml`，产物
`outputs/20261101-212740_C-006-a` 等三组 + `outputs/20261102-122314_C-006-d`（d 为重跑），2026-11-01/02，计划见 [train_plan26.md](../archive/train_plan26.md)）

- 设计：以 `image.brightness=0.15`、`image.contrast=0.15` 为历史参考，a 亮度 0.15→0.05、b 亮度 0.15→0.25、c 对比度 0.15→0.05、d 对比度 0.15→0.25，每组只改一个参数，其余同 C-002-b 口径。
- 结果：C-006-a **0.7957 / 0.4348**（best 5，7 轮）、C-006-b **0.8014 / 0.4263**（best 4，6 轮）、C-006-c **0.7927 / 0.4156**（best 4，6 轮）、C-006-d **0.8009 / 0.4371**（best 5，7 轮）。d 的 11-01 首跑被实例重启打断（仅 1 轮、无 `best.pt`），2026-11-02 12:23:14–12:55:15 在同一 HEAD `64d03da`、seed 42、同型号 V100 上重跑完成，中断残留归档到远端 `outputs/_aborted/`。四组已按 `run_compare.py` 统一重算（`outputs/comparisons/compare_C006/`），`valid_ids_sha256` 全部一致（`81e8be7f…`）、859 studies。阈值曲线：a 峰 0.6（0.4411）、b 峰 0.6（0.4382）、c 峰 **0.7**（0.4350）、d 峰 **0.6**（0.4493，四组最高）。
- 判读：相对 C-002-b（0.7976 / 0.4249），a **−0.0019**、b **+0.0038**、d **+0.0033**、c **−0.0049**；b 与 d 都落在 `train_plan26.md` 的「弱正向」区间（0 < ΔAUC < 0.005），**均未过 +0.005 门**，且 9 类口径分别为 **−0.0011** / **−0.0035**，正向增益全部来自基线 class1 偏低，剔除即转负。四组相对 A-001（0.8038）为 −0.0081 / −0.0024 / **−0.0029** / −0.0111，无一接近。**逐类降幅门（任何类别 AUC 降 ≥0.02）本轮补完**：a 触发（class4 −0.0331）、c 触发（class2 −0.0251）、b 通过（−0.0108）、**d 通过（−0.0164）**——d 是唯一「组级正向 + 逐类不触门」的组合，但幅度不足，只算方向性线索。**对比度这一对互为机制证据**：c（0.05）峰右移到 0.7 且仍在上升，d（0.25）峰回到 0.6 并给出四组最高 F1@0.5（0.4371）与最高峰值（0.4493），说明峰位是对比度扰动的单调响应，而非单组噪声。四组均受早峰压制（第 4–5 轮达峰、第 6–7 轮早停），d 后段过拟合最重（valid_loss 0.9128→1.4312，AUC 连掉两轮 −0.0129 / −0.0144）。
- 限制：四组各单 seed；未做配对 bootstrap，也未校正「在同一 valid 上挑最高分」的多重比较；`brightness` 与 `contrast` 未做交互扫描，因此只能判断这两个参数的**边缘**效应，不能排除两参数联合时的互补；d 与 a/b/c 相隔约 14 小时重跑，代码 HEAD、seed、设备、数据与训练口径一致，唯一配置差异是 `model.xrv_weights` 由残留值改为 `null`（`arch: timm` 下不被读取，ImageNet 预训练加载已在启动日志确认）。

**T-001：外部 ENS-02 双权重在官方 valid 的复评**（配置 `configs/exp_T001{a,b}.yaml`，权重
`outputs/ensemble/ENS02_R1_R2_384_models_v1/`，2026-11-02）

- 设计：外部团队交付的 ENS-02 两个权重（TorchVision DenseNet-121 / 384 / letterbox / ImageNet 归一 / 10 类）先经 `src/import_checkpoint.py` 转成本仓库 checkpoint 格式（外层键由 `state_dict` 改为 `model`，并写入存档配置与来源 sha256），再用 `run_eval.py` 在官方 valid 复评。两份配置完整独立、不继承 `_base`，逐字段相同，唯一差别是 `train.loss` 与 ASL 参数（a=BCE、b=ASL γ+=1/γ−=4/clip=0.05）；`model.aggregation` 固定 `mean`、`image.size=384`、`head=standard`、`eval.threshold=0.5`。权重在 RTX 4060 + torch 2.11 上训练，本次只做前向、不复训。
- 结果（valid 859 studies / 1422 图 / 228 患者，与历史各表同一份数据；两场串行 25 秒，含建模型与 CSV 写出，非 `cost.py` 计时口径）：

  | 组 | macro-AUC（10 类） | 9 类口径 | F1@0.5 | F1 峰位 / 峰值 | 逐类 AUC（class1→10） |
  | --- | ---: | ---: | ---: | --- | --- |
  | T-001-a（R1，BCE） | 0.7866 | 0.7971 | 0.2609 | **0.2 / 0.3996** | .6923 .8199 .7742 .7027 .8686 .7496 .7689 .7236 .9003 .8662 |
  | T-001-b（R2，ASL） | 0.7848 | 0.7948 | 0.3939 | 0.5 / 0.3939 | .6951 .8300 .7692 .7006 .8650 .7488 .7575 .7219 .8987 .8616 |
  | ENS02（0.5/0.5 概率平均，本机复算） | 0.7897 | — | 0.3516 | 0.5 / 0.3516 | — |

- 与来源声明的对照（这是本节的主要目的）：交付方 `docs/MODEL_CARD.md`（`yun` 分支）声明官方 valid 为 R1 **0.786109**、R2 **0.784789**、ENS02 **0.790110**。复评差值为 R1 **+0.000519**、R2 **+0.000051**、ENS02 **−0.000455**。R2 的差恰好是「一对样本排序翻转」的量级——class1 只有 24 正 / 835 负，单对翻转即 1/(24×835) ≈ **5.0e-5**，与该差值同阶；R1 的 5.2e-4 约合 10 对翻转，ENS02 的 4.6e-4 同理。三处都在并列样本的数值抖动范围内，说明**权重装填、384 letterbox、ImageNet 归一、study 内均值聚合四项口径与来源一致**，误差来源是浮点累加顺序与近阈值并列判定，而非流程错配。反过来看，若聚合方式（logits 均值 ↔ 概率均值）或归一化口径错了，差值会落在 1e-3 以上并伴随逐类系统性偏移，本表没有出现该形态。
- 判读：
  - 水平：两者均低于本仓库单模型头名 A-001（0.8038 / 0.4255）**0.0172 / 0.0190**，也低于 C-003-a（0.8020）与刚完成的 C-006-d（0.8009）。9 类口径 0.7971 / 0.7948 对 C-002-b 的 0.8128，差距扩大到 **−0.0157 / −0.0180**——差距不是 class1 的偶然性造出来的，剔除小 support 类后反而更大。
  - 损失不是排序杠杆：R1 与 R2 的 macro-AUC 只差 **0.0018**，逐类 8 升 2 降，最大单类差 c7 **+0.0114**、c2 **−0.0101**。同骨干同分辨率同预处理下只换损失，AUC 层面的位移落在 seed 噪声量级内。
  - **校准差异远大于排序差异**：R1 在 0.5 阈值下 F1 仅 0.2609，class1 / class3 / class4 直接零预测（precision = recall = 0），F1 曲线单调左移、峰在 **0.2**（0.3996）；R2 相反，0.5 阈值下各类 recall 普遍 0.87–0.99、F1 0.3939、峰正好在 0.5。同一队列、AUC 只差 0.0018，阈值行为却差一个量级——这是「study-balanced BCE 未做正样本加权」与「ASL 负样本抑制」的直接后果。**实操结论：这两个权重的 F1 不能按统一阈值横向比较**，比较 F1 必须先各自对齐峰位阈值；本仓库的 `f1@0.5` 口径对 R1 不公平，但这不影响 AUC 结论。
  - 融合空间窄：ENS02 相对 R2 仅 **+0.0048**（0.7897 vs 0.7848），低于其声明值 +0.0053，更远低于本仓库 A-FUSE-3 的 +0.0109。两模型同骨干、同分辨率、同预处理，只差损失与训练随机性，误差高度相关，融合能吃到的是抖动而不是互补。
- 限制：单次前向、无 seed 重复；外部权重的训练划分（其内部 v2：9396 患者训练 / 1659 患者验证）与本仓库口径不同，本表只能做「同一份 valid 上的横向比较」，不能归因于训练配方差异；未做患者整簇配对 bootstrap，ENS02 的 +0.0048 没有置信区间；`docs/MODEL_CARD.md` 里 v2 划分的 0.8030 / 0.8059 与本表的 859 Study 不可比，勿混用。

**D-001：384 + 正位权重 0.8 + 亮/对比 0.25 + lr 1.5e-4 的组合双 seed 对照**（配置 `configs/exp_D001{a,b}.yaml`，产物
`outputs/20261102-130616_D-001-a`、`outputs/20261102-134231_D-001-b`，2026-11-02，训练记录见 [training_log.md](training_log.md)）

- 设计：以 A-001 展开口径为参照，**同时**改动五项——`image.size` 320→384、`image.brightness` 0.15→0.25、`image.contrast` 0.15→0.25、`model.frontal_weight` 0.7→0.8、`train.lr` 3e-4→1.5e-4，预算收到 8 轮 / patience 2；两份配置除 `seed`（42 / 2026）外逐字段一致，`model.xrv_weights` 由 A-001 的残留值清为 `null`（`arch: timm` 下不被读取）。目标是检验「A 系列单项结论能否叠加成组合增益」，不是单变量归因。
- 结果：D-001-a **0.7988 / 0.4401**（best 5，7 轮，36m14s）、D-001-b **0.7973 / 0.4321**（best 5，7 轮，35m04s）；阈值峰均在 0.6（0.4462 / 0.4357）。`outputs/comparisons/compare_D001/` 已把两组与 A-001、C-006-b、C-006-d 统一重算，五个 run 的 `valid_ids_sha256` 一致（`81e8be7f…`）、均 859 studies；AMP 与 fp32 复评差值 ≤ 0.0001。
- 判读：
  - **双 seed 都没过门，方向一致。** 相对 A-001（0.8038 / 0.4255）ΔAUC **−0.0050 / −0.0065**；相对 C-002-b（0.7976 / 0.4249）**+0.0012 / −0.0003**。两 seed 相差 0.0015，远小于与 A-001 的差，结论不是单 seed 噪声。
  - **组合不叠加，低于任何一项单项。** 同骨干 320 口径的 C-006-b（brightness 0.25）0.8014、C-006-d（contrast 0.25）0.8009、C-005-c（frontal 0.8）0.8029 都高于 D-001 双 seed。在 A-001 口径上把这几项一起改并同时降 lr、提分辨率，成绩反而回到基线附近——**单项的「弱正向」不能相加**，这是本组最有价值的信息。
  - **9 类口径同样无增益**：a **0.8099**、b **0.8050**，对 A-001 的 0.8105 分别为 −0.0006 / −0.0055，对 C-002-b 的 0.8128 为 −0.0029 / −0.0078。a 的 class1 由 0.7438 降到 0.6991（−0.0447），组级下滑不是小 support 类造成的假象。
  - **F1@0.5 的抬升来自短预算。** 0.4401 / 0.4321 高于 A-001 的 0.4255，但同预算口径的 C-002-b-EXTRA（320、增强 0.15、8 轮 / patience 2）F1@0.5 已是 0.4431、AUC 0.7958——早峰形态本身就会抬高 F1，本组没有额外贡献。**不要把 0.4401 读成 384 或增强的收益。**
  - 逐类方向两 seed 不一致：a 4 升 6 降（c1 −0.0447、c4 −0.0255），b 5 升 5 降（c3 −0.0401、c2 −0.0326）；仅 c8、c9 两 seed 同向。收敛形态一致：均第 5 轮达峰、第 7 轮早停，valid_loss 后段冲高（a 1.3329、b 1.4448）。
- 限制：五项同改，**只能判「该组合无效」，不能把差异归给其中任何一项**；单 seed 各自一次、未做配对 bootstrap、未拆项重训。单项证据见 C-005（frontal）、C-006（亮/对比）、RUN-007-B / RUN-009-A（384）。

**D-002：EMA 三 seed 对照，EMA 权重独立记为 `原实验名-EMA`**（配置 `configs/exp_D002{a,b,c}.yaml`，产物
`outputs/20261102-145203_D-002-a`、`outputs/20261102-160442_D-002-b`、`outputs/20261102-192442_D-002-c`，2026-11-02）

- 设计：相对 A-001 只启用 `train.ema_enabled: true`，`ema_decay=0.999`，其余训练口径不变；a/b/c 仅 seed 42 / 2026 / 216 不同，三份配置完整展开、逐字段一致。每次训练分别保存按原始验证 AUC 选出的 `best.pt` 与按 EMA 验证 AUC 选出的 `best_ema.pt`，两者在 valid 859 studies 上独立 fp32 复评。**`D-002-a-EMA` / `D-002-b-EMA` / `D-002-c-EMA` 是独立实验条目，但与对应原始组共享训练过程，不重复计算训练耗时。** c 组是为把双 seed 均值推过 +0.005 门而补的第三点，属事后追加，不做多重比较校正。
- 结果：

  | 实验 | 权重 | best | 实际轮数 | macro-AUC | F1@0.5 | F1 峰位 / 峰值 | 训练耗时 |
  | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
  | D-002-a | `best.pt` | 5 | 7/8 | 0.8005 | 0.4222 | 0.7 / 0.4344 | 34m57s |
  | D-002-a-EMA | `best_ema.pt` | 5 | 同 a | **0.8100** | 0.4283 | 0.6 / 0.4422 | 同 a，不重复计 |
  | D-002-b | `best.pt` | 4 | 6/8 | 0.7999 | 0.4150 | 0.6 / 0.4233 | 30m50s |
  | D-002-b-EMA | `best_ema.pt` | 5 | 同 b | 0.8074 | 0.4269 | 0.6 / 0.4423 | 同 b，不重复计 |
  | D-002-c | `best.pt` | 6 | 8/8（跑满） | 0.7961 | **0.4331** | 0.5 / 0.4331 | 39m33s |
  | D-002-c-EMA | `best_ema.pt` | 5 | 同 c | 0.8043 | 0.4329 | 0.6 / 0.4397 | 同 c，不重复计 |

- 判读：
  - **EMA 在三个 seed 上都改善各自原始权重**：配对差 **+0.0095 / +0.0075 / +0.0082**，三次方向完全一致，平均提升 +0.0084。这是本组最稳定的信号。
  - **补第三 seed 后，「EMA 稳定超过 A-001」这一判断被否掉。** 三点 EMA 为 0.8100 / 0.8074 / **0.8043**，均值 **0.8072**，相对 A-001（0.8038）仅 **+0.0034**，落在 +0.005 门内；极差 0.0057、样本标准差 0.0029，与要检测的效应同量级。c 的 0.8043 是三点最低，**新增的一点把均值从 0.8087 拉回到 0.8072**——说明 a 的 +0.0062 是上尾而非稳定水平。
  - 相对 A-001，原始三点为 −0.0034 / −0.0040 / −0.0077（均值 −0.0050），EMA 三点为 +0.0062 / +0.0035 / +0.0005。**只有 a 单次过门，b/c 均未过门且逐点递降**，与 seed 无单调关系，更像抽签结果。
  - **9 类口径同向，增益不是 class1 假象**：a 原始 / EMA 为 0.8088 / **0.8196**，b 为 0.8100 / **0.8159**，c 为 0.8052 / **0.8155**，A-001 为 0.8105。c 的 class1 为 0.7719，剔除后宏平均仍高于原始，EMA 的改善来自其余类别。
  - EMA 的 F1@0.5 分别提高 +0.0061 / +0.0119 / −0.0002；c 的原始权重 F1 峰恰在 0.5（0.4331），EMA 把峰推到 0.6（0.4397），是三点里唯一「峰值 > 0.44」的组。F1 与 AUC 同向但不改变结论。
- 产物与限制：a/b 与 D-001、A-001 的统一重算见 `outputs/comparisons/compare_D002_D003/`；c 与 D-004、E-001 一起在 `outputs/comparisons/compare_D004_E001/`（七个 run 的 `valid_ids_sha256` 均为 `81e8be7f…`、859 studies）。EMA 组使用独立 `run_eval.py` 报告与预测 `outputs/eval_reports/D-002-{a,b,c}-EMA.txt`、`outputs/predictions/valid/D-002-{a,b,c}-EMA.csv`（本地按 `-EMA` 命名，远端对应文件名为小写 `-ema`）。未做患者整簇配对 bootstrap、EMA decay 扫描或 test/融合评估；三点均值仍低于门，**不据此改默认配置或生成新提交**。

**D-003：Study 注意力聚合双 seed 对照**（配置 `configs/exp_D003{a,b}.yaml`，产物
`outputs/20261102-152705_D-003-a`、`outputs/20261102-163536_D-003-b`，2026-11-02）

- 设计：相对 A-001 只把 `model.aggregation` 从 `view_weighted` 改为 `study_attention`，其余训练口径不变；a/b 仅 seed 42 / 2026 不同。D-002 与 D-003 均在 HEAD `e3ec4d6`、同一张 V100-SXM2-32G 上串行执行，预检确认各对配置仅 seed 不同，启动时均确认 ImageNet 预训练加载成功。
- 结果：D-003-a **0.7958 / 0.4494**（best 7，跑满 8 轮，37m32s，F1 峰就在 0.5）；D-003-b **0.7868 / 0.4068**（best 2，4 轮早停，23m09s，F1 峰在 0.5）。统一比较见 `outputs/comparisons/compare_D002_D003/`。
- 判读：相对 A-001，a/b 的 ΔAUC 为 **−0.0080 / −0.0170**，9 类口径分别为 0.8033 / 0.8018，对 A-001 的 0.8105 也为 **−0.0073 / −0.0087**，退化不是 class1 造成。a 的 F1@0.5 虽提高 +0.0239，但 AUC 与 9 类口径同时下降，不能把阈值处的 F1 收益当成判别力改善；b 的 AUC/F1 均下降。两 seed 相差 0.0090、best epoch 分别为 7 与 2，收敛形态也不一致。**Study 注意力没有可复现的 AUC 收益，不切换 `view_weighted` 默认聚合。**
- 限制：未做配对 bootstrap、第三 seed 或注意力超参扫描；本组只能否定当前实现的这两次训练，不证明所有注意力聚合结构都无效。

**D-004：在 EMA 基线上重测 C 系列五项单项**（配置 `configs/exp_D004{a,b,c,d}.yaml`，产物
`outputs/20261102-200415_D-004-a`、`outputs/20261102-211906_D-004-b`、`outputs/20261102-215558_D-004-c`、`outputs/20261102-223054_D-004-d`，2026-11-02）

- 设计：把基线整体换成**已启用 EMA 的 `D-002-a` 口径**（A-001 训练口径 + `ema_enabled`、decay 0.999、8 轮上限、patience 2、lr 3e-4），四项各改一项、单 seed 42：a `model.frontal_weight` 0.7→0.8、b `image.size` 320→384、c `image.contrast` 0.15→0.25、d `image.brightness` 0.15→0.25。**这不是 C 系列重复实验**：C 系列测的是「相对 A-001 是否有增益」，本组测的是「相对更强基线是否仍有增益」。第五项 `frontal_weight` 0.9（D-004-e）原于 2026-11-02 由用户取消，2026-11-04 补跑（`configs/exp_D004e.yaml`，其余口径不变）。五组均产出 `best.pt` 与 `best_ema.pt`，同口径与 `D-002-a-EMA` 配对。
- 结果：

  | 实验 | 权重 | best | 实际轮数 | macro-AUC | F1@0.5 | F1 峰位 / 峰值 | 训练耗时 |
  | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
  | D-002-a-EMA（基线） | `best_ema.pt` | 5 | 7/8 | **0.8100** | 0.4283 | 0.6 / 0.4422 | 34m57s |
  | D-004-a | `best.pt` | 4 | 6/8 | 0.7971 | 0.4203 | 0.6 / 0.4397 | 32m19s |
  | D-004-a-EMA | `best_ema.pt` | 5 | 同 a | 0.8084 | 0.4293 | 0.6 / 0.4408 | 同 a，不重复计 |
  | D-004-b | `best.pt` | 5 | 7/8 | 0.7961 | 0.4273 | 0.6 / 0.4363 | 36m53s |
  | D-004-b-EMA | `best_ema.pt` | 5 | 同 b | 0.8085 | 0.4323 | 0.6 / 0.4417 | 同 b，不重复计 |
  | D-004-c | `best.pt` | 5 | 7/8 | 0.7945 | 0.4287 | 0.6 / 0.4365 | 34m57s |
  | D-004-c-EMA | `best_ema.pt` | 5 | 同 c | **0.8096** | 0.4277 | 0.6 / 0.4435 | 同 c，不重复计 |
  | D-004-d | `best.pt` | 4 | 6/8 | 0.7961 | 0.4128 | 0.6 / 0.4321 | 30m10s |
  | D-004-d-EMA | `best_ema.pt` | 5 | 同 d | 0.8058 | 0.4280 | 0.6 / **0.4451** | 同 d，不重复计 |
  | D-004-e | `best.pt` | 5 | 7/8 | 0.7982 | 0.4289 | 0.6 / 0.4350 | 34m41s |
  | D-004-e-EMA | `best_ema.pt` | 5 | 同 e | 0.8079 | 0.4326 | 0.6 / 0.4373 | 同 e，不重复计 |

- 判读：
  - **四项都没过门，最接近的也只是打平。** 相对 `D-002-a-EMA`（0.8100 / 0.4283），EMA 权重 ΔAUC 为 a **−0.0016**、b **−0.0015**、c **−0.0004**、d **−0.0042**，全部为负。c 的 −0.0004 只是数值上贴着基线，不能读成「对比度加强仍然有用」——它既没超过基线，也没超过噪声水平。
  - **c 的贴线是 class1 补偿换来的。** c-EMA 的 9 类口径为 **0.8142**，相对基线 0.8196 为 **−0.0054**，且 class2 单独下降 **0.0254**（超过 0.02 逐类门）。组级几乎持平而剔除 class1 后明显变差，说明 c 的组级数值靠小 support 类撑住，不是判别力提升。这与 C-006-d 在 A-001 口径上的结论（组级 +0.0033、9 类 −0.0035、增益来自 class1）是同一机制的第二次观测。
  - **d 是唯一明确负向的一项**（−0.0042，9 类 −0.0033，class2 −0.0135）。C-006-b 在 A-001 口径上曾记为「弱正向」+0.0038 但被 9 类证伪；本组在更强基线上直接给出负号，**亮度加强可以判定为无效**。
  - **a 与 b 落在中性区**（−0.0016 / −0.0015，9 类 −0.0034 / −0.0021，逐类降幅均未触 0.02 门）。它们既没有增益也没有明显损害，与 C-005-c（+0.0053 但 9 类 −0.0036）、C-003-a（+0.0044 但 9 类 −0.0006）的「表面正向、剔除 class1 后归零」相互印证：**在 EMA 基线上，正位权重 0.8 与 384 分辨率都不再提供可测量的排序收益。**
  - 四组的原始权重反而全部低于 D-002-a（0.8005）：a −0.0034、b −0.0044、c −0.0060、d −0.0044，与 EMA 权重方向一致。EMA 对四组各自的提升为 +0.0113 / +0.0124 / +0.0151 / +0.0097，再次复现 D-002 的组内正向作用，但作用幅度不足以让任何一组翻越基线。
  - F1 侧无一致信号：d-EMA 峰值 **0.4451** 为全批最高（优于基线 0.4422），但其 AUC 最低，属阈值处形态差异，不作为选型依据。
  - **e（`frontal_weight` 0.9）方向为负但不可分辨**：EMA 0.8079，相对基线 −0.0021（9 类 0.8178，−0.0018），逐类降幅未触 0.02 门。三档 0.7 / 0.8 / 0.9 的 EMA 为 0.8100 / 0.8084 / 0.8079，单调略降但全在单次 SE 0.0094 内，只能判「没有证据支持提高正位权重」；其原始权重 0.7982 是本批五组最高，属训练抖动，不构成 0.9 更优的证据。
- 限制：单 seed、每项一次训练，±0.002 量级不能区分个体差异与 seed 噪声；共享同一份 valid 反复选型，最高分带选择偏差；未做配对 bootstrap 与多重比较校正。

**E-001：EfficientNetV2-S RW 换骨干双分辨率对照**（配置 `configs/exp_E001.yaml`、`configs/exp_E001a.yaml`，产物
`outputs/20261102-203634_E-001`、`outputs/20261102-210126_E-001-a`，2026-11-02）

- 设计：把骨干从 `convnext_tiny.fb_in1k`（27.83M 参数）换成 `efficientnetv2_rw_s.ra2_in1k`（22.17M 参数），其余保持 `D-002-a` 口径（含 EMA、8 轮上限、patience 2、lr 3e-4、seed 42）。E-001 为 320 档，E-001-a 在其余不变的前提下把 `image.size` 提到 384。两组均确认预训练权重加载成功（`HF_HUB_OFFLINE=1` 下命中缓存）。
- 结果：

  | 实验 | 权重 | best | 实际轮数 | macro-AUC | F1@0.5 | F1 峰位 / 峰值 | 训练耗时 |
  | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
  | D-002-a-EMA（基线） | `best_ema.pt` | 5 | 7/8 | **0.8100** | 0.4283 | 0.6 / 0.4422 | 34m57s |
  | E-001 | `best.pt` | 3 | 5/8 | 0.7970 | 0.4112 | 0.5 / 0.4112 | 24m52s |
  | E-001-EMA | `best_ema.pt` | 5 | 同 E-001 | 0.8038 | 0.4307 | 0.6 / 0.4343 | 同 E-001，不重复计 |
  | E-001-a | `best.pt` | 1 | 3/8 | 0.7865 | 0.4000 | 0.6 / 0.4110 | 17m39s |
  | E-001-a-EMA | `best_ema.pt` | 3 | 同 E-001-a | 0.7969 | 0.4137 | 0.6 / 0.4195 | 同 E-001-a，不重复计 |

- 判读：
  - **换骨干没有收益，两档都低于 convnext_tiny 基线。** EMA 权重相对 `D-002-a-EMA`：320 档 **−0.0062**、384 档 **−0.0131**，均已越过 +0.005 门的下侧；9 类口径同向（0.8139 / 0.8053 对基线 0.8196，−0.0057 / −0.0143）。
  - **E-001 的 0.8038 与 A-001 同值属巧合，勿误读为「追平基线」。** 本批基线已换成启 EMA 的 0.8100，该组实际落后 0.0062；它与 A-001 的相等只是两个不同口径数值的偶然重合。
  - **384 档明显更差且收敛不稳**：best epoch 1、第 3 轮即早停，训练只用了 17m39s（320 档 5 轮 / 24m52s），说明分辨率提高后该骨干在这个学习率预算下第 1 轮之后就没再改进。这更像**优化配置不匹配**（有效 batch 与 lr 未随分辨率调整），而不是「高分辨率对胸片无用」的证据。
  - 逐类层面 320 档 class3 降 **0.0231**、384 档 class3 降 **0.0466**，均触发 0.02 逐类门；两档的退化集中在 class3，不是全局均匀下滑。
  - **归因限制**：E-001 与 E-001-a 同时换了骨干与（384 档）分辨率，不能拆项判断是骨干容量、感受野还是优化预算导致下滑。本组只能给出「在当前训练配方下 EfficientNetV2-S RW 不优于 ConvNeXt-Tiny」这一结论。
- 限制：单 seed、每档一次训练；未做 `efficientnetv2_rw_s` 的 lr / 分辨率专项调参，也未测 384 档配更小 lr 或更大 batch 的变体；未做配对 bootstrap。本批原以七组收尾，D-004-e 已于 2026-11-04 补跑（见上一节）。

**D-002-d：换 ImageNet-22k 预训练权重**（配置 `configs/exp_D002d.yaml`，产物 `outputs/20261104-102710_D-002-d`，2026-11-04）

- 设计：D-002-a 口径完全不变，只把 `model.backbone` 从 `convnext_tiny.fb_in1k` 换成 `convnext_tiny.fb_in22k_ft_in1k`（ImageNet-22k 预训练后在 1k 微调），检验更强 ImageNet 预训练在当前预算下是否有增益。单 seed 42、8 轮上限、patience 2。首次启动（10:22:53）因 HF 离线缓存缺该权重退化为随机初始化，已终止并归档到 `outputs/_aborted/20261104-102253_D-002-d_pretrained_failed`；补下载（`HF_ENDPOINT=https://hf-mirror.com`，28,589,128 参数）并验证离线命中后重跑，日志确认「预训练权重: 已加载」。
- 结果：raw **0.7964 / 0.4258**（9 类 0.8060，best 6，8/8 轮，38m23s）；EMA **0.8037 / 0.4284**（9 类 0.8142，F1 峰 0.6 / 0.4405）。
- 判读：相对 `D-002-a`（raw 0.8005 / EMA 0.8100），raw −0.0041、EMA −0.0063（9 类 −0.0054），方向为负；**0.8037 是 D-002/D-004 六组 EMA 中最低**（0.8100 / 0.8084 / 0.8085 / 0.8096 / 0.8058 / 0.8037），没有任何一档支持「22k 权重更好」。9 类口径同步下降说明不是 class1 造成。结论限于当前 8 轮、lr 3e-4 预算：换更强的 ImageNet 预训练没有带来可测收益，不支持替换默认骨干。
- 限制：单 seed，不能排除 seed 效应；未做配对 bootstrap；22k 权重可能需要更小 lr 或更长训练才能体现优势，本轮未扫。

**D-005：多标签 AUCM 对照（初筛失败）**（配置 `configs/exp_D005a.yaml`、`configs/exp_D005b.yaml`，产物 `outputs/20261104-110534_D-005-a`、`outputs/20261104-112249_D-005-b`，2026-11-04）

- 设计：依据 `train_plan28.md`。D-005-a 为 ImageNet 初始化后全程 AUCM；D-005-b 从 `20261102-145203_D-002-a/best_ema.pt`（source epoch 5，sha256 `aeeb2106…`）warm start 后转 AUCM 微调。两组均关闭 `pos_weight`、`loss: aucm`、AUCM margin 1.0 / version v1、PESG（adam、lr 3e-4、weight decay 0.01、epoch_decay 0）、`batch_studies: 16`，按类别均衡采样（每批单类别、正负各半）。
- 依赖：`libauc==2.0.0` 用 uv 装入主 `.venv`（新增 40 包、无版本升降级，torch 保持 2.10.0+cu128）；`tests/test_aucm.py` 7 passed。
- 结果：

  | 实验 | 权重 | best | 实际轮数 | macro-AUC | 9 类口径 | F1@0.5 | 训练耗时 |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | D-002-a-EMA（起点/基线） | `best_ema.pt` | 5 | 7/8 | **0.8100** | 0.8196 | 0.4283 | 34m57s |
  | D-005-a | `best.pt` | 1 | 3/8 | 0.5233 | 0.5326 | 0.0000 | 17m15s |
  | D-005-a-EMA | `best_ema.pt` | 1 | 同 a | 0.5403 | 0.5538 | 0.0000 | 同 a |
  | D-005-b | `best.pt` | 1 | 3/8 | 0.5411 | 0.5352 | 0.0000 | 17m13s |
  | D-005-b-EMA | `best_ema.pt` | 1 | 同 b | 0.5825 | 0.5816 | 0.0000 | 同 b |

- 判读：
  - **输出塌缩，不是学得慢。** 两组 AUC 落在 0.52–0.58、F1@0.5 全 0，概率分布几乎恒定（D-005-a raw 均值 0.0459 / std 0.0026；D-005-b-EMA std 0.0524），排序信息消失。
  - **D-005-b 是定性证据**：起点 BCE-EMA 0.8100，一个 epoch 后 0.5422，第三轮 0.5000 → AUCM 的更新在破坏已收敛权重，问题在优化/损失侧而非初始化。
  - 根因未定，候选按可疑度：(a) PESG 的 lr 3e-4 相对 1157 步/epoch 过大，配合默认 `clip_value=1.0` 快速把权重带离预训练解；(b) `TaskPESG` 对 `a/b/alpha` 的逐类 mask 恢复在单类别 batch 下可能让 alpha 与动量状态累积偏差；(c) AUCM 需要更长预算与更小 lr，8 轮、3e-4 的初筛设置本身不匹配。三种假设均未验证。
  - **结论**：AUCM 在当前实现与初筛超参下不可用，不能作为提交候选；两组记为负结果，不构成「AUCM 方法无效」的证据。后续若继续，先固定 D-005-b 的 warm start 对 lr ∈ {3e-5, 1e-5} 做单变量对照，确认不塌缩后再看全程 AUCM。

**Distill-1：固定融合教师离线蒸馏配对对照**（配置 `configs/exp_Distill1a.yaml`、`configs/exp_Distill1b.yaml`，产物 `outputs/20261104-135517_Distill-1-a`、`outputs/20261104-142947_Distill-1-b`，教师缓存 `outputs/distill/Distill-1_teacher`，2026-11-04）

- 设计：依据 `train_plan29.md` / `train_plan30.md`。教师固定为 A-FUSE-5 的三模型裁剪 logit 4:2:3 融合（D-002-a `best_ema.pt`、C-005-c `best.pt`、E-001 `best.pt`），先把训练集 18,513 个 study 的软标签落盘并校验（study 集合、概率范围、三 checkpoint SHA-256、融合权重），再由学生读取；缓存为 in-sample 预测，不是 OOF。两份配置独立完整、无 `_base`，展开后唯一差异是 `train.distill_enabled`：a 关闭，b 开启（λ=0.5、T=1，软目标 BCE 不带 `pos_weight`）。学生同从 `convnext_tiny.fb_in1k` 初始化（确认已加载），seed 42、320、batch 16 studies、lr 3e-4、warmup 2、cosine 8 轮、patience 2、BCE + `pos_weight`、EMA 0.999。
- 结果：

  | 实验 | 权重 | best | 实际轮数 | 10 类 Macro-AUC | 9 类口径 | F1@0.5 | 训练耗时 |
  | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | Distill-1-a | `best.pt` | 5 | 7/8 | 0.7987 | 0.8069 | 0.4351 | 34m29s |
  | Distill-1-a-EMA | `best_ema.pt` | 4 | 同 a | 0.8092 | 0.8190 | 0.4339 | 同 a |
  | Distill-1-b | `best.pt` | 5 | 7/8 | 0.8024 | 0.8069 | 0.4344 | 34m36s |
  | Distill-1-b-EMA | `best_ema.pt` | 4 | 同 b | 0.8109 | 0.8172 | 0.4328 | 同 b |
  | D-002-a-EMA（历史基线） | `best_ema.pt` | 5 | 7/8 | **0.8100** | 0.8196 | 0.4283 | 34m57s |

- 判读：
  - **主比较未达参考线。** b−a 的 EMA 10 类 Δ **+0.0017**，患者整簇配对 bootstrap（2,000 次、seed 2026、228 名患者、有效 2000/2000）95% CI **[-0.0037, +0.0078]**、P(Δ>0)=0.752；9 类 **−0.0018**、F1@0.5 **−0.0012**。未达 +0.010 优先复核参考线且 CI 跨 0，单 seed 下既不能判增益，也不能判等价。
  - **与重跑噪声同量级**：a / b 的 EMA 相对 D-002-a-EMA 0.8100 为 −0.0008 / +0.0009，说明同配置换一次运行的波动已在 ±0.001；本次 Δ 不能与噪声区分。
  - **增益集中在 class1**：逐类只有 c1 +0.0337（support 24），c2 −0.0118、c3 −0.0140；9 类口径由正转负，与 D-002 系列「组级提升靠 class1」的模式一致。
  - **压缩损失点估计 0.006–0.008**：b-EMA 对教师 0.8171 / 0.8249 为 −0.0061（CI [-0.0125, +0.0009]）/ −0.0078；教师为 in-sample 三模型融合，单学生落后符合预期，但比较仅用历史记录值，不宣称非劣。
  - **软目标信息量有限**：b 的 `train_soft_loss` 0.5434→0.4919（epoch 5）→0.5501（epoch 7），全程仅降 0.05 量级，而 `train_hard_loss` 0.9787→0.4439；软 BCE 后程回升与 valid_loss 上升同步，说明教师与硬标签高度一致、软标签未提供额外排序信息。
- 限制：教师软标签为 in-sample 且教师也在同一 valid 上选型；单 seed；同一 valid 反复选型；bootstrap 不覆盖训练随机性与选择偏差；未做 seed 2026 配对、test 评估、教师 valid 独立复算与学生耗时对比。结论限于「当前 λ/T、8 轮预算、单 seed 下无可测增益」，不外推到其他温度、权重或更长训练。

**D-006：单模型 AUC 三方向初筛（已完成）**（配置 `configs/exp_D006a.yaml`、`configs/exp_D006b.yaml`、`configs/exp_D006c.yaml`，产物 `outputs/20261104-193131_D-006-a`、`outputs/20261104-201457_D-006-b`、`outputs/20261104-205718_D-006-c`，2026-11-04）

- 依据 [20261104-train-plan-01.md](20261104-train-plan-01.md)。三组各跑一次、全部 seed 42，不重跑基线、不做早停对照、不预排额外 seed、不做融合或蒸馏。共同口径：`convnext_tiny.fb_in1k` 预训练（日志确认「预训练权重: 已加载」）、320 letterbox、hflip 关闭、batch 16 studies、AdamW lr 3e-4、warmup 2、BCE + `pos_weight`、EMA 0.999、8 轮上限、patience 2。
- **早停依据由历史 D-002-a 的 raw AUC 改为 EMA AUC**（`train.early_stop_monitor: ema_auc`）。与历史分数的差值因此属低成本探索结果，不能把全部差值严格归因于研究变量。
- 参考基线：D-002-a-EMA（valid 0.8100 / 9 类 0.8196 / F1@0.5 0.4283），同一 859 study 集合、同一份 valid。

**D-006-a：DropPath 0.1**（配置 `configs/exp_D006a.yaml`，产物 `outputs/20261104-193131_D-006-a`）

- 设计：相对 D-002-a 唯一变量是 `model.drop_path_rate` 0→0.1，其余逐字段一致（含 EMA、8 轮、patience 2、lr 3e-4、seed 42）。
- 结果：raw **0.8009 / 0.4331**（峰值 0.6 / 0.4404，best epoch 6）；EMA **0.8082 / 0.4271**（9 类 0.8185，峰值 0.7 / 0.4403，best epoch 5）；7/8 轮早停，训练 **34m15s**；`cost.py` fp32 单 study **54.5 ms**（100 study / 162 图，连续 3 轮，0 个 study 不一致，init 59.1 ms，162 图折算 33.6 ms/图）。
- 实现核验：17 个残差块的 `drop_path` 概率 0.0059–0.1000（线性递增到 0.1），`eval()` 下同一输入两次前向完全一致（`allclose=True`）。
- 判读：相对基线 **Δ10 −0.0019 / Δ9 −0.0012**，患者整簇配对 bootstrap（2000 次、seed 2026、228 患者、2000/2000 有效）95% CI **[-0.0082, +0.0055]**、P(Δ>0)=0.330 → 跨 0，**不可分辨**。raw 侧 0.8009 vs 0.8005 近乎同值（+0.0005），但 9 类口径 −0.0033。逐类最大跌幅 class3 −0.0142，最大涨幅 class6 +0.0102 / class4 +0.0092，均未触 0.02 逐类门，没有哪一类出现方向明确的退化或增益。F1@0.5 0.4271 略低于基线 0.4283，峰值位置从 0.6 右移到 0.7，属阈值形态差异而非排序差异。
- 结论：DropPath 0.1 在当前 8 轮、lr 3e-4 预算下**没有可测增益，也没有明确损害**。本轮早停依据已改为 EMA AUC，且 EMA best epoch（5）与基线相同，可比性较好；但单 seed、单次训练不足以支持方法级结论，若继续该方向应做多 seed 配对，而不是在同一 valid 上继续调强度。

**D-006-b：cosine 第 6 轮到达最低学习率**（配置 `configs/exp_D006b.yaml`，产物 `outputs/20261104-201457_D-006-b`）

- 设计：唯一变量是 `train.cosine_end_epoch` null→6，衰减分母由 `8-1-2=5` 改为 `6-1-2=3`，使第 6 轮即到 `lr_min` 1e-5 并在第 7–8 轮保持。逐轮 LR 已用真实 optimizer/scheduler 核验：`1.5e-4 / 3e-4 / 3e-4 / 2.275e-4 / 8.25e-5 / 1e-5 / 1e-5 / 1e-5`，与计划表逐位一致，第 6–8 轮无反弹。
- 结果：raw **0.8071 / 0.4425**（峰值 0.6 / 0.4513，best epoch 5）；EMA **0.8108 / 0.4297**（9 类 0.8189，峰值 0.6 / 0.4491，best epoch 5）；7/8 轮早停，训练 **34m31s**；`cost.py` fp32 单 study **53.9 ms**（100 study / 162 图、0 个 study 不一致、init 59.9 ms）。
- 判读：EMA 侧 **Δ10 +0.0008 / Δ9 −0.0007**，患者整簇配对 bootstrap 95% CI **[-0.0042, +0.0059]**、P(Δ>0)=0.635 → **完全不可分辨**，0.8108 不能读成「超过基线」。raw 侧 **Δ10 +0.0067 / Δ9 +0.0059**，CI **[-0.0025, +0.0158]**、P(Δ>0)=0.915，下界贴 0 但未跨 0；逐类 raw 增益集中在 **class4 +0.0407**（68 正），超过 0.02 逐类门，class7 −0.0096 为最大跌幅。EMA 逐类则是 6 升 4 降、最大涨 class4 +0.0154 / 最大跌 class2 −0.0142，无一类触门。
- **低 LR 保持阶段（第 6–7 轮，lr 1e-5）没有带来提升**：EMA 0.8102 → 0.8074、raw 0.8054 → 0.8041，同期 train_loss 0.573 → 0.542 继续下降而 valid_loss 0.921 → 0.982 上升，属典型过拟合。因此本组的价值不在「更长的低 LR 训练」，而在**第 5 轮的位置**：b 的第 5 轮 lr 是 8.25e-5（a 为 2.0e-4），更早进入低学习率使该轮权重明显好于 a 的同一轮。计划第 3.3 条要求「未进入第 6 轮则标注未充分观察」——本轮实际进入了第 6、7 轮，观察到的是**无增益**，不属观察不足。
- 结论：提前完成 cosine 使 EMA 权重**持平**（+0.0008，噪声内）、raw 权重**小幅正向**（+0.0067，CI 下界贴 0）；F1 峰值 0.4491（EMA）/ 0.4513（raw）为本批最高，但峰值位置不能替代 AUC 判读。单 seed 下不能判定「提前衰减有效」，可作为后续「更短 cosine + 更小 lr」组合的线索，不宜直接改默认配置。

**D-006-c：Study Mixup（alpha 0.2、概率 0.3）**（配置 `configs/exp_D006c.yaml`，产物 `outputs/20261104-205718_D-006-c`）

- 设计：开启按视角签名分组、同批内两两配对的 Study Mixup（`train.mixup_alpha: 0.2`、`train.mixup_prob: 0.3`），其余同 D-006-a 口径（含 `ema_auc` 早停、drop_path 0）。图像、标签、offsets 与视角数在混合后保持不变，软标签走原 `pos_weight` 的 BCE。
- 执行事实：**实际混合覆盖率 0.2392–0.2488、7 轮均值 0.2448**（每轮约 4,428–4,606 / 18,513 study），为配置概率 0.3 的 **81.6%**。缺口来自同批次内找不到同视角签名配对的 study（单图正位、单图侧位、含 unknown 的 study 按设计不参与），属计划已预期的执行事实，不是实现失效。
- 结果：raw **0.7988 / 0.4262**（峰值 0.6 / 0.4399，best epoch 6）；EMA **0.8076 / 0.4306**（9 类 0.8159，峰值 0.6 / 0.4358，best epoch 5）；7/8 轮早停，训练 **34m52s**；`cost.py` fp32 单 study **56.0 ms**（100 study / 162 图、0 个 study 不一致、init 81.4 ms）。
- 判读：EMA **Δ10 −0.0024 / Δ9 −0.0037**，患者整簇配对 bootstrap 95% CI **[-0.0077, +0.0038]**、P(Δ>0)=0.233 → 跨 0，**不可分辨**，方向略负。raw **Δ10 −0.0017 / Δ9 −0.0040**，CI **[-0.0129, +0.0104]** 跨 0。EMA 逐类 4 升 6 降，最大涨 class1 +0.0100 / 最大跌 class2 −0.0118，无一类触 0.02 门；raw 逐类出现两个明确信号：**class2 −0.0485**（37 正）与 **class8 −0.0209**（214 正）超逐类门退化，同时 class4 +0.0361 上升，呈「类间重新分配」而非整体下滑。
- 结论：在当前配方与 0.2448 的实际覆盖率下，Study Mixup **没有增益证据，且 9 类口径与 raw 的类级退化方向一致偏负**。由于 F1@0.5 由 0.4283 → 0.4306、峰值 0.4358 略高，可见混合起到了轻微正则作用但没有转化为排序能力提升。单 seed、单次训练、单一 alpha/prob 组合，不足以判定 Mixup 方法无效；若要继续，应先按签名统计可配对比例、再决定是否放宽到同签名内跨 batch 配对或提高 batch size。

**D-006 跨组汇总**

| 组 | 研究变量 | raw AUC | EMA AUC | 9 类（EMA） | Δ10（EMA vs 基线） | 95% CI | cost |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| D-002-a-EMA | 基线 | 0.8005 | 0.8100 | 0.8196 | — | — | — |
| D-006-a | DropPath 0.1 | 0.8009 | 0.8082 | 0.8185 | −0.0019 | [-0.0082, +0.0055] | 54.5 ms |
| D-006-b | cosine 终点 6 | 0.8071 | **0.8108** | 0.8189 | **+0.0008** | [-0.0042, +0.0059] | 53.9 ms |
| D-006-c | Study Mixup | 0.7988 | 0.8076 | 0.8159 | −0.0024 | [-0.0077, +0.0038] | 56.0 ms |

- **三项均未过 +0.010 优先复核参考线，三个 CI 全部跨 0。** 本批只能判定「在 8 轮、lr 3e-4、EMA 早停口径下，三者都没有可测的组级增益」，不能判定任何一项方法无效。
- 三个 9 类口径全为负（−0.0012 / −0.0007 / −0.0037），与 10 类方向一致，说明差值不是 class1（24 正）撑出来的假象——这一点与本仓库历史上「组级正向、9 类反号」的多次观测不同。
- b 是唯一在两个口径上都为正/接近零的组（10 类 +0.0008、9 类 −0.0007），且 raw 侧 +0.0067、class4 +0.0407 是全批最强信号；但 EMA 是主权重，其 +0.0008 完全落在噪声内。
- 耗时三组 53.9–56.0 ms/study，差异 ≤2.1 ms，与推理路径完全相同（都是单 ConvNeXt-Tiny + view_weighted）一致，说明 DropPath/Mixup 只影响训练、不增加推理成本；三组都距 100 ms 上限有 >44 ms 余量，但 V100 数字不可外推到 T4。
- 训练耗时 34m15s / 34m31s / 34m52s，三组几乎相同，说明本批的单变量改动没有改变训练吞吐。
- 后续：不自动追加训练（计划第 6 节）。若要继续，优先级依次是 ① b 方向补 3 seed 配对，确认 raw +0.0067 与 class4 +0.0407 是否可重复；② c 方向先统计视角签名可配对比例、再决定是否调整配对粒度；③ a 方向证据最弱，暂不投入。

**D-007：cosine 终点 6 的三 seed 复核（已完成）**（配置 `configs/exp_D007a.yaml`、`configs/exp_D007b.yaml`，产物 `outputs/20261106-140614_D-007-a`、`outputs/20261106-143742_D-007-b`，2026-11-06）

- 设计：执行 D-006 汇总后续第 ① 条。seed 42 复用 D-006-b 不重跑，本批补 seed 2026（a）与 seed 216（b），唯一研究变量是 `seed`。
- 执行：a 14:06:09–14:37:37（31m28s，6/8 轮）、b 14:37:37–15:12:19（34m42s，7/8 轮），V100-SXM2-32G 独占，`--no-save-last`。
- 结果：a raw 0.7976 / 0.4282、EMA **0.8055** / 0.4218（9 类 0.8147）；b raw 0.7960 / 0.4302、EMA **0.8044** / 0.4381（9 类 0.8143）。
- 判读：三 seed 相对基线的 Δ10 为 +0.0008（42）/ −0.0046（2026）/ −0.0057（216），**均值 −0.0032，三个 CI 全部跨 0**。**提前衰减到最低学习率不构成可复现增益**，D-006-b 的 +0.0008 属 seed 42 个例；其 raw +0.0067 与 class4 +0.0407 在本批未复现（a/b 的 raw Δ −0.0024 / −0.0040，class4 +0.0014 / +0.0067）。
- 副产品：**同配置只换 seed 的配对差 CI 宽 0.0134**（[−0.0054, +0.0080]），再次实测出 seed 噪声量级，印证 ±0.005 门在本项目不可判读。
- 形态观察：D-007-a 的 EMA 峰值在第 4 轮、raw 峰值在第 5 轮，与 D-002-a 同形态，说明「EMA 最优点早于 raw」稳定，为导出 `best_ema.pt` 时按 EMA 早停提供第二组证据。

**D-008：骨干容量阶梯 Small / Base / Large（已完成）**（配置 `configs/exp_D008{a,b,c}.yaml`，产物 `outputs/20261106-151321_D-008-a`、`outputs/20261106-155019_D-008-b`、`outputs/20261106-170640_D-008-c`，2026-11-06）

- 设计：以 D-002-a（Tiny `fb_in1k`，seed 42）为基线，同 seed 换 Small / Base / Large，兼作蒸馏教师候选。显式改用 `ema_auc` 早停（基线是 raw），**因此不是严格单变量对照**，差值混入早停口径；理由见 D-007 的 EMA/raw 峰值形态。原 ConvNeXt V2 三档于 15:21 删除（权重 CC-BY-NC，赛题不接受）。
- 执行：a（Small）15:13:16–15:50:15（37m00s，7/8 轮）、b（Base）15:50:15–16:28:53（38m38s，7/8 轮）。**c（Large）首跑 16:29–16:50 在 epoch 3 被容器 cgroup 43 GiB 内存上限 SIGKILL（rc=137）**，根因是 47 GiB train 图片的 page cache 计入 cgroup、前几组累积到 27.4 GiB 后撞顶；`drop_caches` 无权限，改用 `posix_fadvise(DONTNEED)` 用户态逐出（27.4 → 2.0 GiB）并每 150 秒守护，未改训练配置，17:06:35 重跑。失败 run 归档 `outputs/_aborted/20261106-162948_D-008-c_cgroup_oom/`。
- 结果：a（Small）raw 0.7927 / 0.4209、EMA **0.8057** / 0.4333（9 类 0.8161）；b（Base）raw 0.7909 / 0.4181、EMA **0.8021** / 0.4337（9 类 0.8121）；c（Large）raw 0.7976 / 0.4187、EMA **0.8022** / 0.4385（9 类 0.8137，60m53s，7/8 轮）。
- 判读：**扩容骨干在 8 轮预算下没有收益，且规模越大越早过拟合。** 容量阶梯单调下降：Tiny 0.8100 → Small 0.8057（Δ −0.0043，CI [−0.0110, +0.0032] 跨 0）→ Base 0.8021（Δ −0.0079，CI [−0.0131, −0.0025]，P=0.002）→ Large 0.8022（Δ −0.0078，CI [−0.0142, −0.0005]，P=0.019）。**Base 与 Large 是本项目首次出现 CI 不跨 0 的负向；Large 未优于 Base，两档持平且都低于 Tiny。** 三点限制必须一起引用：①早停口径与基线不同（`ema_auc` vs raw）；②单 seed；③最优轮次随规模提前（Base 第 4 轮、Large 第 3 轮），Large 的 valid_loss 从 0.89 涨到 1.93、train_loss 降到 0.22，8 轮明显过长。另需披露 **c 的 9 类口径 CI 跨 0**（−0.0059，[−0.0123, +0.0010]），只有 10 类口径不跨 0。因此只能读作「本预算与本早停口径下扩容无收益、Base/Large 有可测损害、8 轮对 Large 过长」，不能外推为架构优劣。
- F1 侧无补偿：a/b/c 的 F1@0.5（0.4333 / 0.4337 / 0.4385）略高于基线 0.4283，但峰值（0.4423 / 0.4354 / 0.4385）未超基线 0.4422，属阈值处形态变化，不作选型依据。**c 的 F1 峰值落在阈值 0.5 而非 0.6**，是六组里唯一如此的一档，说明 Large 概率分布整体偏低（校准偏移）。
- 教师用途收窄：a/b/c 的 EMA 权重都不如 Tiny 基线 0.8100，用它们做蒸馏教师的先验价值不高；若仍要蒸，应改用更短预算（Large 第 3 轮即达 raw 峰）或直接以 Tiny 的 EMA 权重为教师。

**D-009：患者检查次数重加权采样 patient_count（已完成）**（配置 `configs/exp_D009a.yaml`、`configs/exp_D009b.yaml`，产物 `outputs/20261107-112733_D-009-a`、`outputs/20261107-114205_D-009-b`，2026-11-07）

- 设计：D-002-a 的唯一变量对照。训练集多检查患者（检查数 ≥5）只占 study 的 18.51%，valid 却占 571/859 = 66.47%，训练分布与评测分布不一致；D-009 用有放回 `WeightedRandomSampler` 把多检查组 study 的期望占比拉到 66.5%（`train.study_sampler: patient_count`、阈值 5、目标占比 0.665），每 epoch 样本数不变。a seed 42 / b seed 2026 分别配对 D-002-a / D-002-b，其余逐字段同基线。**早停口径与基线不同**（本组未设 `early_stop_monitor`，按 `early_stop_metric` 默认 `ema_auc`；基线按 raw AUC），差值不严格归因于重采样。
- 执行：a 19:27:29–19:42:01（14m32s，4/8 轮）、b 19:42:01–19:57:00（14m59s，4/8 轮），sjy RTX 3090 24G 独占，`--no-save-last`。两组启动日志均确认预训练权重已加载。
- 结果：a raw 0.7729 / 0.3985、EMA **0.7907** / 0.4200（9 类 0.8021）；b raw 0.7619 / 0.3956、EMA **0.7830** / 0.4061（9 类 0.7918）。两组 EMA 峰值都在第 2 轮、第 4 轮 stale 2 早停；raw 峰值 a 第 2 轮、b 第 1 轮。
- 判读：**四组配对 Δ 全部为负且 CI 全部不跨 0，是本项目迄今量级最大的一次负向。** 10 类口径 a-EMA **−0.0193**（CI [−0.0311, −0.0058]、P=0.003）、b-EMA **−0.0244**（CI [−0.0397, −0.0105]、P=0.002），raw 侧 a −0.0276、b −0.0379 同向；9 类诊断口径 a −0.0176、b −0.0241，方向一致。**设计目标未兑现**：valid 分层多检查患者（ge5/ge10）的 Δ 为 a −0.0167/−0.0105、b −0.0194/−0.0152，多检查组与整体同步退化，raw 侧同样为负（a −0.0280/−0.0184、b −0.0319/−0.0337）——「训练分布向评测分布对齐」的机制假设被直接否定。两 seed 都与基线同向且幅度达 seed 噪声的 4–5 倍（D-007 实测 CI 宽 0.0134）。限制：①早停口径差异（D-008 同口径对照仅 −0.0043/−0.0079，不足解释 −0.02）；②双 seed 单次训练，b 的效应量比 a 更大；③两组都只跑 4 轮、峰值前移到第 1–2 轮，结论限于 8 轮预算；④未逐 epoch 记录实际多检查占比（期望 0.665）。综合判读：**在当前预算下按患者检查次数重加权采样明确有害，不进入提交候选，方向暂不追加训练。**

## 冻结结论

### 新实验判读协议（2026-11-04 起）

本协议以 [实验复盘 §3.0.2](optimization_review.md) 为起点，经复核后采用以下分层口径；适用于此后新开展的对照与后续判读。复盘中的原建议、下文及历史计划中的 ±0.005 门和结论保留当时口径，不自动追认或改写为本协议下的结论。

1. **先固定对照**：预先写明基线、候选、差异、seed、数据划分、训练预算和选点规则；同一 seed 尽量配对，保持其余条件一致。主指标为同一 valid 上的 **Study 级 10 类 Macro-AUC**，报告双方数值和 Δ（候选减基线）。同时报告 F1 所用阈值与阈值曲线，不能用 F1 替代 AUC。
2. **证据分层**：初筛可用一个配对 seed，报告 Δ 和患者整簇配对 bootstrap 95% CI，作为是否继续投入的依据，不宣称跨 seed 稳定增益。复核至少使用 **3 个预先指定的配对 seed**，报告每个 seed 的双方 AUC、Δ、方向和配对 Δ 均值，观察训练随机性。要支持更强的推广结论，还须对平均 Δ 计算患者配对 CI，并在独立患者验证集或其他独立复核中检验，同时说明适用范围。单 seed 的 CI 和多 seed 对应不同不确定性，不能互相替代。
3. **+0.010 的用途**：将正向 Δ **+0.010** 作为当前资源和验证规模下优先投入复核的经验参考线，并非统计显著性门或有效／无效的分界。小于该值的候选仍可根据 CI、seed 方向、耗时、模型大小及融合互补性决定是否复核；超过该值也须检查 CI、逐类结果和选型偏差。不得由单模型 AUC 的标准误直接推断配对 Δ 的不确定性。
4. **逐类复核**：每次同时报告剔除 class1 后的 **class2–class10 诊断性 9 类 Macro-AUC**、其 Δ、每类 AUC 和 Δ，以及各类正样本数；class1 单独列出，不以其单次变化作为机制结论的充分依据，应结合其余类别、多 seed、支持数和不确定性分析。此 9 类是诊断口径，须明确写出类别，不能与其他 9 类或平台评分口径混称。若 10 类与诊断性 9 类方向不一致，或存在明显逐类退化，须披露并谨慎判读，不能仅凭 10 类均值宣称普遍改善或判定方案无价值。
5. **配对 bootstrap**：按患者有放回抽样并携带该患者全部 studies，在同一重采样上计算双方 Study 级 Macro-AUC 的差；记录抽样次数、随机种子、CI 方法和有效／无效重采样数，并说明无效原因。多 seed 的平均 Δ CI：每次对所有 seed 使用同一批重采样患者，分别计算各配对 Δ，再对这些 Δ 求均值；它不是先融合模型预测再计算 AUC。该 CI 主要反映固定模型的患者抽样不确定性，不覆盖训练随机性和同一 valid 反复选型偏差；逐 seed 差值仍须单独报告。仅有单一标签的一类无法计算 AUC 时，不得静默改变该次重采样的类别集合；该次视为无效并报告，若无效率明显影响结果则不得作可靠的 CI 结论。
6. **结论表述**：同时报告效应、CI、seed 稳定性和实际成本。CI 全部大于 0 可描述为该固定模型比较的正向证据；全部小于 0 可描述为负向证据；包含 0 时说明方向尚不确定，并给出区间所允许的幅度。未达到参考线不等于「无效」或「不可分辨」；CI 跨 0 也不等于两者等价。只有预先定义可接受差异范围且 CI 足够窄时，才能进一步讨论差异是否足够小。同一 valid 的反复选型可能使结果乐观，多 seed 不能消除这一偏差。

### 历史冻结结论

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
   与 RUN008-C 同量级、差异混杂，仍不切换；A-001-a（训练随机水平翻转，AUC 0.7960）、A-001-C（冻结骨干，0.7264）与 A-004 系列（dropout + weight_decay 组合 0.7988、单 dropout 0.7944、单 weight_decay 0.7999）、A-006 的 ML-Decoder 分类头（0.7900 vs 同版本重训标准头 0.8022）与 A-005 的 CSRA 分类头（0.7936 vs 同一配对基线 0.8022）与 A-007 的 ConvNeXt-Small + `fb_in22k_ft_in1k` 骨干/权重（0.7903 vs A-001 0.8038）也未带来 AUC 增益；C-001 的四种骨干在骨干 5e-5 / 头 1.5e-4 下全部于 epoch 3 达峰、valid macro-AUC 0.7774–0.7919、均未过门，同样不构成切换依据；C-002 的四组 ConvNeXt-Tiny 统一学习率配对落在 0.7956–0.7976，3e-4 与 1.5e-4 的配对差为 +0.0014 / −0.0002、区间跨 0，学习率维持 3e-4；默认 `image.hflip: false`、`freeze_backbone: false` 与训练超参保持不变）；
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
| 新骨干对照 | 已验证（未达切换门槛） | `configs/exp_densenet121.yaml` / `configs/exp_resnet50.yaml` / `configs/exp_run008{a,b,c,d}.yaml` | **RUN-003-B / RUN-005 / RUN-006 / RUN-006-A 已完成（同口径）**：ResNet-18 基线 0.7837、ResNet-50 0.7857、DenseNet-121 0.7880（seed 42）/ 0.7913（seed 2026）；新骨干 macro-AUC 高 0.0020 / 0.0043，seed 2026 尚缺 ResNet-18 配对结果，仍不足以判定骨干优劣，暂不切换默认骨干。**RUN-008 骨干对照（2026-10-29 晚，6 轮无早停）**：ConvNeXt-Tiny 0.8008（比 RUN-007-A 高 +0.0102）、EfficientNet-B2 0.7952（+0.0046）、EfficientNetV2-S 0.7954（+0.0048）、ConvNeXtV2-Nano 0.4977（训练异常）；ConvNeXt-Tiny 超过判读门但预算与 RUN-007-A（25 轮、12 轮早停）不同且单 seed，只作候选方向；ConvNeXtV2-Nano 权重为 CC BY-NC 4.0，赛题规则不接受。详见 [train_plan04.md](../archive/train_plan04.md)、[train_plan05.md](../archive/train_plan05.md)、[train_plan08.md](../archive/train_plan08.md) 与训练记录。T4 端到端耗时与权重许可 / SHA-256 核验仍待做（仅当选新骨干时）。A-001（2026-10-30）在 seed 42、epoch 8、早停 2 下复现 0.8038，与 RUN008-C 同量级、差异混杂；**A-003（2026-10-30/31）**：MobileNetV3-Small 1.0×（1,528,106 参数）@320、25 轮上限 + 早停 5 口径在第 17 轮早停（best epoch 12），macro-AUC 0.7808 / F1@0.5 0.3881，低于 A-001 的 0.8038 且距 0.7988 门差 −0.0180 → 轻量 CNN 在本预算下未过门，默认骨干不切换；A-002（0.5×）第 5 轮按用户要求中止、无 `best.pt`，两者配对比较不成立（详见 [train_plan11.md](../archive/train_plan11.md)）；**A-007（2026-10-31）**：ConvNeXt-Small + `fb_in22k_ft_in1k`（49,462,378 参数）@320、8 轮 / patience 2 / seed 42 口径在第 8 轮跑满后早停（best epoch 6），macro-AUC 0.7903 / F1@0.5 0.4204，低于 A-001（0.8038 / 0.4255）ΔAUC −0.0135、距 0.8088 门 −0.0185，配对 bootstrap ΔAUC 95% CI [+0.0006, +0.0260] P=0.983 → 更大骨干 + 更强预训练在本预算下未过门，且同时变结构与权重不可单变量归因（详见 [train_plan17.md](../archive/train_plan17.md)） |
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
