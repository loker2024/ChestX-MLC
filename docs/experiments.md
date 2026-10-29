# P7 调优记录

当前默认配置已改为余弦调度并启用 `pos_weight`；以下历史实验记录中的配置与对照关系以当时的运行配置为准，不能直接用当前继承配置重现。

P7 的目标是把「训练什么、怎么评、交什么」固定下来。训练统一放到训练环境（远程 GPU /
赛题服务器）执行，本机只负责代码与文档验证，避免两边环境差异污染实验结果；因此本文件
记录实验矩阵、判读原则和冻结结论。执行命令见训练计划，实测数字记录在训练记录中。

## 实验环境

| 项目 | 值 |
| --- | --- |
| 数据 | 全量 train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 设备 | RTX 3090 24G（复用同一实例；本机只做代码与文档验证） |
| 软件 | Python 3.12、torch 2.x（CUDA）、timm 1.0.30 |
| 随机种子 | 42（RUN-002-B、RUN-006-A、RUN-007-A、RUN008-A/B/C/D 为 2026） |
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

短周期基线把 `epochs` 降到 6，并关闭 early stop 和 `pos_weight`，其余超参数继承 `default.yaml`，
这样两个 run 的差异只来自 `pos_weight`。6 个 epoch 不足以跑出最终成绩，但足以判断
「pos_weight 是帮了忙还是拖了后腿」，避免直接为 25 个 epoch 赌一把。

## 执行与结果

执行步骤和判读门见 [训练计划](train_plan01.md)，通用命令及续训注意事项见 [使用手册](manual.md)。实际运行的配置、环境、指标和耗时记录在 [training_log.md](training_log.md)，对照结果以该文件为准。

截至 2026-10-28 的对照结论（数字均来自 `metrics.csv` 与 `outputs/compare/summary.md`）：

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

对照时使用同一份验证数据：`run_compare.py` 优先读取 `best.pt` 内保存的配置，旧 checkpoint 无存档配置时回退到 run 目录的 `config.yaml`。`summary.md` 和 `per_class.csv` 提供逐类指标与阈值。先看 macro-AUC，再结合阈值曲线及逐类表现判断 F1 和 `pos_weight` 的影响。

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
   不切换）；
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
| 新骨干对照 | 已验证（未达切换门槛） | `configs/exp_densenet121.yaml` / `configs/exp_resnet50.yaml` / `configs/exp_run008{a,b,c,d}.yaml` | **RUN-003-B / RUN-005 / RUN-006 / RUN-006-A 已完成（同口径）**：ResNet-18 基线 0.7837、ResNet-50 0.7857、DenseNet-121 0.7880（seed 42）/ 0.7913（seed 2026）；新骨干 macro-AUC 高 0.0020 / 0.0043，seed 2026 尚缺 ResNet-18 配对结果，仍不足以判定骨干优劣，暂不切换默认骨干。**RUN-008 骨干对照（2026-10-29 晚，6 轮无早停）**：ConvNeXt-Tiny 0.8008（比 RUN-007-A 高 +0.0102）、EfficientNet-B2 0.7952（+0.0046）、EfficientNetV2-S 0.7954（+0.0048）、ConvNeXtV2-Nano 0.4977（训练异常）；ConvNeXt-Tiny 超过判读门但预算与 RUN-007-A（25 轮、12 轮早停）不同且单 seed，只作候选方向；ConvNeXtV2-Nano 权重为 CC BY-NC 4.0，赛题规则不接受。详见 [train_plan04.md](train_plan04.md)、[train_plan05.md](train_plan05.md)、[train_plan08.md](train_plan08.md) 与训练记录。T4 端到端耗时与权重许可 / SHA-256 核验仍待做（仅当选新骨干时） |
| **视图元信息（ViewPosition）** | 已验证（无效） | 已实现 `StudyRecord` / `collate_studies` 传递视角及正侧位加权聚合 | **RUN-007 / RUN-007-A 已判无效**：seed 42 加权 macro-AUC 0.7892 vs 等权 0.7880（Δ+0.0012），seed 2026 加权 0.7906 vs 等权 0.7913（Δ−0.0007），两 seed 配对差方向相反且均低于 +0.005 门 → 默认 `aggregation: mean` 不变；F1 与逐类变化在两 seed 间也不一致。`Rows`/`Columns` 不建议用（易学设备偏差），`Subject_id` 严禁入特征 |
| 阈值曲线选点 | 中 | 仅推理侧：`run_compare.py` 扫 0.2~0.7，必要时改 `eval.threshold` | RUN-003 正式模型已复核：F1 峰值仍在 0.2（0.4219，@0.5 为 0.2655），与 RUN-001R 形态一致；RUN-002-A/B 的 pos_weight 模型把峰右移到 0.5–0.6（0.4182 / 0.4212）、RUN-003-B（cosine + pos_weight）峰在 0.4（0.4059），上限均与基线 @0.2 持平，这些选点仅用于验证 F1；默认提交输出全部 10 类概率，不按阈值过滤 |
| 训练轮数 | 已验证 | 在正式基线之后调整 `train.epochs` | **RUN-003 已判无效**：epoch 8 达峰（0.7887）后过拟合、第 13 轮早停，25 轮上限未带来 AUC 增益（Δ−0.0016 vs RUN-001R），该次实验未证明增加训练轮数有收益 |
| 学习率调度 | 已验证 | `train.lr_scheduler` / `train.lr_min` | **RUN-003-A 判无效**：余弦调度 best epoch 5、macro-AUC 0.7852 vs RUN-003 的 0.7887（Δ−0.0035，未达门）；但最佳轮次前移（8→5）、早停更快（13→10 轮）、耗时约 30 分钟 vs 38 分钟。当前默认配置为 `cosine`；该对照未证明 AUC 增益 |
| `pos_weight` | 已验证 | `train.pos_weight: true` | **RUN-002-A/B 未达原 AUC 增益门**：完整周期双 seed macro-AUC 0.7872 / 0.7871，vs RUN-003 0.7887 为 Δ−0.0015 / −0.0016，均未达 +0.005 门；F1@0.5 提升（0.4045 / 0.4198）伴随阈值曲线右移，已测峰值与基线接近，当前默认开启 |
| 聚合方式消融 | 待评估 | 比较平均 logits、平均概率及最大聚合 | 当前使用平均 logits 后 sigmoid；其他方式尚未实现或实测 |
| 更高分辨率 / TTA | 已验证（方向性） | `image.size`、`run_infer.py` 加翻转 TTA | **RUN-007-B 已做 384×384 + 6 轮短跑（macro-AUC 0.7959 / F1@0.5 0.4227）**：比 320 的 12 轮早停 0.7906 高 +0.0053，但轮数与 cosine 周期不同、不可单变量归因；需 25 轮完整对照再判定。TTA 未做；448 显存未实测（3090 24G） |

### 正侧位加权聚合配置

`configs/exp_view_weighted.yaml` 在默认骨干上启用加权聚合；RUN-007 系列使用 DenseNet-121。正位（AP、PA）和侧位（LATERAL、LL）先分别求 logits 均值，再按 0.7 / 0.3 加权；单视角或含未知视角时回退全部图片等权均值。推理及耗时检查须使用相同配置；缺少完整测试索引时，多图加权一致性校验会报错。实测结果见上文及 [训练记录](training_log.md)。
