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
| 随机种子 | 42（RUN-002-B、RUN-006-A 为 2026） |
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

对照时使用同一份验证数据：`run_compare.py` 优先读取 `best.pt` 内保存的配置，旧 checkpoint 无存档配置时回退到 run 目录的 `config.yaml`。`summary.md` 和 `per_class.csv` 提供逐类指标与阈值。先看 macro-AUC，再结合阈值曲线及逐类表现判断 F1 和 `pos_weight` 的影响。

## 冻结结论

1. **提交格式**：`run_infer.py` 默认输出全部 10 类概率，不做阈值过滤——赛题的
   macro-AUC 需要排序信息，全 10 类输出更稳妥；`--threshold 0.5` 只用于对齐赛题示例
   的过滤形态。
2. **配置**：`configs/default.yaml` 即最终配置（25 epoch、`pos_weight: true`、
   `eval.threshold: 0.5`，骨干维持 `resnet18`——RUN-003-B 补齐同口径基线后，RUN-005 /
   RUN-006 的 macro-AUC 只高 0.0020 / 0.0043，单 seed 差距不足以判定骨干优劣；
   RUN-006-A 用 seed 2026 复核 DenseNet-121 得 0.7913，但缺少同 seed 的 ResNet-18
   对照，尚不能判断骨干优势是否在第二个 seed 复现）；
   实验配置一律用 `_base` 继承它，不在实验文件里复制超参。
   在对比表出来之前不无依据地改默认值，要改必须先附 `run_compare.py` 的结果；`train.pos_weight` 的 AUC 未达原增益门，但 F1@0.5 显著提高，现按要求设为 `true`；`train.lr_scheduler` 经 RUN-003-A 复核无 AUC 增益，现按要求改为 `cosine`。
3. **复现口径**：默认 seed 为 42，seed 复核另有 2026；指标统一由 `run_compare.py` 在同一份 valid 上重算，
   训练设备与预训练加载情况记在 checkpoint 的 `metadata` 里。
4. **分工**：本机只跑 `python -m pytest` 和文档校验；训练相关命令都在训练环境执行，
   实测结果回填训练记录，对照结论记录在本文件。

## 后续待验证优化项（未排期）

记录想法，不承诺优先级；每项都要等基线（25 轮 `default.yaml`）出结果后再单独做对照，
一次只改一个变量，并附 `run_compare.py` 结果。

| 优化项 | 预期收益 | 改动点 | 备注 |
| --- | --- | --- | --- |
| 新骨干对照 | 已验证（未达切换门槛） | `configs/exp_densenet121.yaml` / `configs/exp_resnet50.yaml` | **RUN-003-B / RUN-005 / RUN-006 / RUN-006-A 已完成（同口径）**：ResNet-18 基线 0.7837、ResNet-50 0.7857、DenseNet-121 0.7880（seed 42）/ 0.7913（seed 2026）；新骨干 macro-AUC 高 0.0020 / 0.0043，seed 2026 尚缺 ResNet-18 配对结果，仍不足以判定骨干优劣，暂不切换默认骨干；详见 [train_plan04.md](train_plan04.md)、[train_plan05.md](train_plan05.md) 与训练记录。T4 端到端耗时与权重许可 / SHA-256 核验仍待做（仅当选新骨干时） |
| **视图元信息（ViewPosition）** | 已验证（未达切换门槛） | 已实现 `StudyRecord` / `collate_studies` 传递视角及正侧位加权聚合 | **RUN-007 已完成**：7:3 加权聚合 macro-AUC 0.7892 vs RUN-006 等权 0.7880（Δ+0.0012），未达 +0.005 门且在 seed 波动内，默认 `aggregation` 维持 `mean`；F1@0.5 提升属阈值曲线右移。`Rows`/`Columns` 不建议用（易学设备偏差），`Subject_id` 严禁入特征 |
| 阈值曲线选点 | 中 | 仅推理侧：`run_compare.py` 扫 0.2~0.7，必要时改 `eval.threshold` | RUN-003 正式模型已复核：F1 峰值仍在 0.2（0.4219，@0.5 为 0.2655），与 RUN-001R 形态一致；RUN-002-A/B 的 pos_weight 模型把峰右移到 0.5–0.6（0.4182 / 0.4212）、RUN-003-B（cosine + pos_weight）峰在 0.4（0.4059），上限均与基线 @0.2 持平，这些选点仅用于验证 F1；默认提交输出全部 10 类概率，不按阈值过滤 |
| 训练轮数 | 已验证 | 在正式基线之后调整 `train.epochs` | **RUN-003 已判无效**：epoch 8 达峰（0.7887）后过拟合、第 13 轮早停，25 轮上限未带来 AUC 增益（Δ−0.0016 vs RUN-001R），该次实验未证明增加训练轮数有收益 |
| 学习率调度 | 已验证 | `train.lr_scheduler` / `train.lr_min` | **RUN-003-A 判无效**：余弦调度 best epoch 5、macro-AUC 0.7852 vs RUN-003 的 0.7887（Δ−0.0035，未达门）；但最佳轮次前移（8→5）、早停更快（13→10 轮）、耗时约 30 分钟 vs 38 分钟。当前默认配置为 `cosine`；该对照未证明 AUC 增益 |
| `pos_weight` | 已验证 | `train.pos_weight: true` | **RUN-002-A/B 未达原 AUC 增益门**：完整周期双 seed macro-AUC 0.7872 / 0.7871，vs RUN-003 0.7887 为 Δ−0.0015 / −0.0016，均未达 +0.005 门；F1@0.5 提升（0.4045 / 0.4198）伴随阈值曲线右移，已测峰值与基线接近，当前默认开启 |
| 聚合方式消融 | 低—中 | `aggregate_logits` 增加「平均概率 / 最大 / 平均 logits」三种 | 当前实现是平均 logits 后 sigmoid（等价 G2 方案），可作为对照项记录 |
| 更高分辨率 / TTA | 待评估 | `image.size`、`run_infer.py` 加翻转 TTA | 先测显存与耗时，3090 24G 下 448 可能吃紧 |

### 正侧位加权聚合配置

`configs/exp_view_weighted.yaml` 使用与默认配置相同的训练设置，仅将 `model.aggregation` 设为 `view_weighted`、`model.frontal_weight` 设为 0.7；`configs/exp_run007.yaml` 以 `_base: exp_densenet121.yaml` 把同一加权口径挂到 RUN-006 的 DenseNet-121 上，用于单变量对照。**RUN-007 实测**（`outputs/compare_run007/summary.md`，明细见训练记录）：macro-AUC 0.7892 vs RUN-006 等权 0.7880，Δ+0.0012 未达判读门，且小于 seed 波动，默认 `aggregation: mean` 不变。推理、评估及耗时检查须传入相同配置。

每个 study 先按正位（AP、PA）和侧位（LATERAL、LL）分别求 logits 均值，再按 0.7 / 0.3 聚合，最后 sigmoid；只有一种已知视角或包含未知视角时，对该 study 全部图片等权平均。空值及未识别值（包括 `PA LLD`）视为未知。默认 `mean` 保持原有图片等权逻辑。没有完整测试索引时，视角加权的多图耗时一致性校验会报错，避免把未知视角结果当作有效校验。
