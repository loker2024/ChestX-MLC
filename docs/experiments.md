# P7 调优记录

P7 的目标是把「训练什么、怎么评、交什么」固定下来。训练统一放到训练环境（远程 GPU /
赛题服务器）执行，本机只负责代码与文档验证，避免两边环境差异污染实验结果；因此本文件
提供可直接复制的命令、实验矩阵和结果回填表格，数字一律来自 `src/run_compare.py` 生成的
`summary.md`，不手抄训练日志。

## 实验环境

| 项目 | 值 |
| --- | --- |
| 数据 | 全量 train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 设备 | 训练环境 GPU（本机 RTX 4060 Laptop 只跑 pytest，不训练） |
| 软件 | Python 3.12、torch 2.x（CUDA）、timm 1.0.30 |
| 随机种子 | 42（所有 run 相同） |
| 输入 | 320×320，letterbox，ImageNet 归一化，默认无水平翻转 |
| 骨干 | timm `resnet18.a1_in1k`，ImageNet 预训练 |

## 实验矩阵

| run | 配置 | epochs | pos_weight | 用途 |
| --- | --- | --- | --- | --- |
| short_baseline | `configs/exp_short.yaml` | 6 | 否 | 短周期方向筛查的基准 |
| short_pos_weight | `configs/exp_short_pos_weight.yaml` | 6 | 是 | 与基准同一 schedule，只差 pos_weight |
| full_baseline | `configs/default.yaml` | 25 | 否 | 冻结配置的完整训练 |

短周期配置只把 `epochs` 降到 6、关掉 early stop，其余超参与 `default.yaml` 完全一致，
这样两个 run 的差异只来自 `pos_weight`。6 个 epoch 不足以跑出最终成绩，但足以判断
「pos_weight 是帮了忙还是拖了后腿」，避免直接为 25 个 epoch 赌一把。

## 复现命令

在训练环境的仓库根目录执行；先确认 `data/images` 指向全量数据（或直接把
`configs/default.yaml` 里的 `data.<split>.images` 改成本机路径）。

```bash
# 0) 数据体检（可选，确认路径、标签与划分没问题）
python src/check_data.py --config configs/default.yaml --full

# 1) 两个短周期对照：同一 seed，只差 pos_weight
python src/run_train.py --config configs/exp_short.yaml --seed 42 --tag short_baseline
python src/run_train.py --config configs/exp_short_pos_weight.yaml --seed 42 --tag short_pos_weight

# 2) 在同一份 valid 上重算 macro-AUC 与多阈值 macro-F1
python src/run_compare.py \
    --runs outputs/<ts>_short_baseline outputs/<ts>_short_pos_weight \
    --thresholds 0.3,0.4,0.5,0.6,0.7

# 3) 按冻结配置训练最终模型并生成提交
python src/run_train.py --config configs/default.yaml --seed 42
python src/run_infer.py --config configs/default.yaml \
    --checkpoint outputs/<final>/best.pt --out outputs/submission.csv
```

`run_compare.py` 优先读取 `best.pt` 内保存的配置，旧 checkpoint 没有配置时才回退到
run 目录的 `config.yaml`。续训后因此不会误用初次训练的配置。它分别按各 run 的配置读取
valid，比较前应确认各配置指向同一份验证数据；`summary.md` 和 `per_class.csv` 还提供
逐类 AUC、precision、recall、F1 与阈值，便于检查稀有类和常见类是否同时改善。

## 结果

> 待训练环境跑完两组短周期对照后回填：把 `outputs/compare/summary.md` 的表格原样贴到
> 这里，再补一列每轮耗时。下面先保留空行，方便直接填。

| run | pos_weight | seed | best_epoch | macro-AUC | f1@0.3 | f1@0.4 | f1@0.5 | f1@0.6 | f1@0.7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `short_baseline` | 否 | 42 | | | | | | | |
| `short_pos_weight` | 是 | 42 | | | | | | | |

判读方法：

- 先看 macro-AUC：它不受阈值影响，是模型排序能力的主指标；
- 再看 `f1@0.5` 与整条阈值曲线：只有当别的阈值在各 run 上都更好时，才值得调整
  `eval.threshold`；
- `pos_weight` 打开后如果只有少数类 recall 上升、其余类掉得更多，macro-F1 不升反降，
  就维持关闭。

## 冻结结论

1. **提交格式**：`run_infer.py` 默认输出全部 10 类概率，不做阈值过滤——赛题的
   macro-AUC 需要排序信息，全 10 类输出更稳妥；`--threshold 0.5` 只用于对齐赛题示例
   的过滤形态。
2. **配置**：`configs/default.yaml` 即最终配置（25 epoch、`pos_weight: false`、
   `eval.threshold: 0.5`）；实验配置一律用 `_base` 继承它，不在实验文件里复制超参。
   在对比表出来之前不无依据地改默认值，要改必须先附 `run_compare.py` 的结果。
3. **复现口径**：seed 固定 42，指标统一由 `run_compare.py` 在同一份 valid 上重算，
   训练设备与预训练加载情况记在 checkpoint 的 `metadata` 里。
4. **分工**：本机只跑 `python -m pytest` 和文档校验；训练相关命令都在训练环境执行，
   本文件随结果回填。

## 后续待验证优化项（未排期）

记录想法，不承诺优先级；每项都要等基线（25 轮 `default.yaml`）出结果后再单独做对照，
一次只改一个变量，并附 `run_compare.py` 结果。

| 优化项 | 预期收益 | 改动点 | 备注 |
| --- | --- | --- | --- |
| **视图元信息（ViewPosition）** | 中—高 | `StudyRecord` 加 `views`、`collate_studies` 透传、分类头拼接 one-hot 或改用视图条件聚合 | 正位 / 侧位显影不同，当前无差别均值会互相稀释；`Rows`/`Columns` 不建议用（易学设备偏差），`Subject_id` 严禁入特征 |
| 阈值曲线选点 | 中 | 仅推理侧：`run_compare.py` 扫 0.3~0.7，必要时改 `eval.threshold` | RUN-001 出现 AUC 0.786 而 F1@0.5 仅 0.280，阈值偏高的可能性大 |
| 训练轮数 | 中 | `train.epochs` 加到 10~25 | RUN-001 最优在最后一轮，6 轮未收敛 |
| `pos_weight` | 待验证 | `train.pos_weight: true` | 已配好 `configs/exp_short_pos_weight.yaml`，等 RUN-002 |
| 聚合方式消融 | 低—中 | `aggregate_logits` 增加「平均概率 / 最大 / 平均 logits」三种 | 当前实现是平均 logits 后 sigmoid（等价 G2 方案），可作为对照项记录 |
| 更高分辨率 / TTA | 待评估 | `data.image_size`、`run_infer.py` 加翻转 TTA | 先测显存与耗时，3090 24G 下 448 可能吃紧 |
