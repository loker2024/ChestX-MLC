# P7 调优记录

P7 的目标是把「训练什么、怎么评、交什么」固定下来。训练统一放到训练环境（远程 GPU /
赛题服务器）执行，本机只负责代码与文档验证，避免两边环境差异污染实验结果；因此本文件
记录实验矩阵、判读原则和冻结结论。执行命令见训练计划，实测数字记录在训练记录中。

## 实验环境

| 项目 | 值 |
| --- | --- |
| 数据 | 全量 train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 设备 | RTX 3090 24G（复用同一实例；本机只做代码与文档验证） |
| 软件 | Python 3.12、torch 2.x（CUDA）、timm 1.0.30 |
| 随机种子 | 42（所有 run 相同） |
| 输入 | 320×320，letterbox，ImageNet 归一化，默认无水平翻转 |
| 骨干 | timm `resnet18.a1_in1k`，ImageNet 预训练 |

## 实验矩阵

| run | tag | 配置 | epochs | pos_weight | 用途 |
| --- | --- | --- | --- | --- | --- |
| RUN-001 | `short_baseline` | `configs/exp_short.yaml` | 6 | 否 | 短周期方向筛查的基准（旧代码 59a7d15） |
| RUN-001R | `short_baseline_r2` | `configs/exp_short.yaml` | 6 | 否 | 用当前代码（e8f194e）重跑同一配置，验证训练链路回归 |
| RUN-002 | `short_pos_weight` | `configs/exp_short_pos_weight.yaml` | 6 | 是 | 与 001R 同一 schedule，只差 pos_weight |
| RUN-003 | `full_baseline` | `configs/default.yaml` | 25 | 否 | 冻结配置的完整训练，产出提交候选 |
| RUN-004A / 004B | 待定 | `exp_full_long.yaml` / `exp_pos_weight.yaml` | 40 / 25 | 否 / 是 | 按 RUN-003 的 `best_epoch` 与 RUN-002 结论二选一 |

短周期配置只把 `epochs` 降到 6、关掉 early stop，其余超参与 `default.yaml` 完全一致，
这样两个 run 的差异只来自 `pos_weight`。6 个 epoch 不足以跑出最终成绩，但足以判断
「pos_weight 是帮了忙还是拖了后腿」，避免直接为 25 个 epoch 赌一把。

## 执行与结果

执行步骤和判读门见 [训练计划](train_plan01.md)，通用命令及续训注意事项见 [使用手册](manual.md)。实际运行的配置、环境、指标和耗时记录在 [training_log.md](training_log.md)，对照结果以该文件为准。

截至 2026-10-28 的对照结论（数字均来自 `metrics.csv` 与 `outputs/compare/summary.md`）：

- **RUN-001R 回归门通过**：末轮 macro-AUC 0.7903，对比 RUN-001 的 0.7858 为 Δ+0.0045，落在判读门 [0.7808, 0.7908] 内，曲线形状也一致（6 轮单调上升、最优 epoch 6）→ 当前代码的训练链路与旧代码等价，001R 的 `best.pt` 可作为耗时自检载体。
- **`pos_weight` 判无效（RUN-002）**：最优 macro-AUC 0.7882（epoch 5），对比 001R 的 0.7903 为 Δ−0.0021，未达「+0.005 且逐类无大退化」的判读门；逐类 AUC 差呈稀有类升、多数丰富类降（class1 +0.0588，class2 −0.0328、class4 −0.0398、class8 −0.0172），正是判读门要排除的形态。`train.pos_weight` 维持 `false`，RUN-004 走 004A。
- **F1 差异属阈值效应**：001R 的 F1@0.5（0.2649）低于 RUN-001（0.2796），但 001R 在阈值 0.2 处达到 0.4012；两版代码只差预处理实现且数值等价，F1 的差异来自阈值选点与训练波动，不能据此判模型退化。
- **执行顺序与计划不同**：RUN-001R 与 RUN-002 并行执行（两路各 8 workers，16 核饱和），`run_compare` 在 01:00 先出对照，随后出提交文件并跑耗时自检，RUN-003 于 01:13 启动、epoch 1 后暂停待续训；并行只影响墙钟耗时，不影响 AUC（shuffle 顺序由主进程决定）。

对照时使用同一份验证数据：`run_compare.py` 优先读取 `best.pt` 内保存的配置，旧 checkpoint 无存档配置时回退到 run 目录的 `config.yaml`。`summary.md` 和 `per_class.csv` 提供逐类指标与阈值。先看 macro-AUC，再结合阈值曲线及逐类表现判断 F1 和 `pos_weight` 的影响。

## 冻结结论

1. **提交格式**：`run_infer.py` 默认输出全部 10 类概率，不做阈值过滤——赛题的
   macro-AUC 需要排序信息，全 10 类输出更稳妥；`--threshold 0.5` 只用于对齐赛题示例
   的过滤形态。
2. **配置**：`configs/default.yaml` 即最终配置（25 epoch、`pos_weight: false`、
   `eval.threshold: 0.5`）；实验配置一律用 `_base` 继承它，不在实验文件里复制超参。
   在对比表出来之前不无依据地改默认值，要改必须先附 `run_compare.py` 的结果；`train.pos_weight` 经 RUN-002 对照判无效（2026-10-28），维持 `false`。
3. **复现口径**：seed 固定 42，指标统一由 `run_compare.py` 在同一份 valid 上重算，
   训练设备与预训练加载情况记在 checkpoint 的 `metadata` 里。
4. **分工**：本机只跑 `python -m pytest` 和文档校验；训练相关命令都在训练环境执行，
   实测结果回填训练记录，对照结论记录在本文件。

## 后续待验证优化项（未排期）

记录想法，不承诺优先级；每项都要等基线（25 轮 `default.yaml`）出结果后再单独做对照，
一次只改一个变量，并附 `run_compare.py` 结果。

| 优化项 | 预期收益 | 改动点 | 备注 |
| --- | --- | --- | --- |
| **视图元信息（ViewPosition）** | 中—高 | `StudyRecord` 加 `views`、`collate_studies` 透传、分类头拼接 one-hot 或改用视图条件聚合 | 正位 / 侧位显影不同，当前无差别均值会互相稀释；`Rows`/`Columns` 不建议用（易学设备偏差），`Subject_id` 严禁入特征 |
| 阈值曲线选点 | 中 | 仅推理侧：`run_compare.py` 扫 0.3~0.7，必要时改 `eval.threshold` | RUN-001R 已扫 0.2~0.7：F1 峰值在 0.2（0.4012，@0.5 为 0.2649）；仍需在 25 轮正式模型上复核 |
| 训练轮数 | 中 | 在正式基线之后调整 `train.epochs` | RUN-001R 最优仍在末轮（0.7903），6 轮未收敛，等 RUN-003（25 轮）验证 |
| `pos_weight` | 已验证 | `train.pos_weight: true` | **RUN-002 判无效**：AUC 0.7882 vs RUN-001R 0.7903（Δ−0.0021，未达 +0.005 门），逐类稀有类升、多数类降；默认维持 `false` |
| 聚合方式消融 | 低—中 | `aggregate_logits` 增加「平均概率 / 最大 / 平均 logits」三种 | 当前实现是平均 logits 后 sigmoid（等价 G2 方案），可作为对照项记录 |
| 更高分辨率 / TTA | 待评估 | `image.size`、`run_infer.py` 加翻转 TTA | 先测显存与耗时，3090 24G 下 448 可能吃紧 |
