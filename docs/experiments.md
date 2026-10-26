# P7 调优记录

P7 的目标是把「训练什么、怎么评、交什么」固定下来。本文件记录实验环境、对照实验的
矩阵与复现命令，以及最终的冻结结论；数字一律来自 `src/run_compare.py` 重算的结果
（`outputs/compare/summary.md`），而不是手抄训练日志。

## 实验环境

| 项目 | 值 |
| --- | --- |
| 数据 | 全量 train 30,344 图 / valid 1,422 图（study 级，`data/images` 指向本地副本） |
| 设备 | NVIDIA GeForce RTX 4060 Laptop（WSL Ubuntu 24.04，CUDA 12.8） |
| 软件 | Python 3.12.3、torch 2.10.0+cu128、timm 1.0.30 |
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

```bash
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

`run_compare.py` 读的是每个 run 目录里存档的 `config.yaml`，所以即使仓库里的默认配置
后来改了，旧 run 的指标也不会被「穿越式」改写。

## 结果

> 等实验跑完后填写：直接贴 `outputs/compare/summary.md` 的表格，并记录每轮耗时。

| run | pos_weight | seed | best_epoch | macro-AUC | f1@0.3 | f1@0.4 | f1@0.5 | f1@0.6 | f1@0.7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待填 | | | | | | | | | |

## 冻结结论

1. **提交格式**：`run_infer.py` 默认输出全部 10 类概率，不再做阈值过滤——赛题的
   macro-AUC 需要排序信息，全 10 类输出更稳妥；`--threshold 0.5` 只用于对齐赛题
   示例的过滤形态。
2. **配置**：`configs/default.yaml` 即最终配置，实验配置一律用 `_base` 继承它，
   不在实验文件里复制一份超参。
3. **复现口径**：seed 固定 42，指标统一由 `run_compare.py` 在同一份 valid 上重算，
   训练设备与预训练加载情况记在 checkpoint 的 `metadata` 里。

（阈值与 pos_weight 的取舍等结果出来后补在这里。）
