# 训练计划 07：RUN-007-A（DenseNet-121 正侧位加权聚合的 seed 复核）

## 目的与对照

RUN-007 在 seed 42 下观察到正侧位 7:3 加权聚合相对 RUN-006 等权均值的
macro-AUC 差为 +0.0012，未达 +0.005 判读门，也小于 DenseNet-121 的 seed
波动（RUN-006 与 RUN-006-A 相差 0.0033）。本组用 seed 2026 复跑 RUN-007 的
口径，检验加权聚合的方向是否在第二个 seed 上稳定；与 RUN-006-A（同 seed 的
等权对照）可直接配对比较，差异只应来自聚合方式。

| 实验 | 配置 | 骨干 | 聚合 | seed |
| --- | --- | --- | --- | --- |
| RUN-006-A | `configs/exp_run006a.yaml` | `densenet121.ra_in1k` | 图片等权均值 | 2026 |
| RUN-007-A | `configs/exp_run007a.yaml` | `densenet121.ra_in1k` | 正侧位 7:3 | 2026 |

RUN-007-A 继承 RUN-007 配置：320×320、`pos_weight: true`、warmup + cosine、
最多 25 轮、早停耐心值 5。加权聚合先分别求正位（AP、PA）与侧位（LATERAL、LL）
的 logits 均值，再按 0.7 / 0.3 合并；只有一种已知视角或包含未知视角时，回退为
该 study 全部图片的等权均值。聚合同时用于训练和验证。

## 启动前核对

在仓库根目录核对解析后的配置：

```bash
PYTHONPATH=src python - <<'PY'
from common.config import load_config

run007 = load_config('configs/exp_run007.yaml').to_dict()
run007a = load_config('configs/exp_run007a.yaml').to_dict()
differences = {key: (run007[key], run007a[key]) for key in run007 if run007[key] != run007a[key]}
assert differences == {'seed': (42, 2026)}, differences
print('核对通过：除 seed 外与 RUN-007 完全一致。')
PY
```

确认全量数据体检通过，并与 RUN-006-A 使用同一份 train / valid 数据：

```bash
python src/check_data.py --config configs/exp_run007a.yaml --full
```

## 训练与比较

```bash
python src/run_train.py --config configs/exp_run007a.yaml \
    --seed 2026 --tag densenet121_viewweighted_s2026
```

检查日志中的预训练权重是否已加载；未加载则排查后重新启动。保存运行目录中的
`config.yaml`、`train.log`、`metrics.csv`、`best.pt` 和 `last.pt`。

```bash
python src/run_compare.py \
    --runs 'outputs/<run006a_run>' 'outputs/<run007a_run>' \
    --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 \
    --out outputs/compare_run007a
```

主要比较 study 级 macro-AUC，并查看逐类 AUC 和各阈值下的 macro-F1；记录最优
epoch、运行时间及代码版本。两个 seed 的「加权−等权」配对差应同号，才可作为
方向性证据；若 seed 42 为正、seed 2026 为负，则说明差异在 seed 波动内。完成后
将实测回填 `docs/training_log.md`，将对照结论记录到 `docs/experiments.md`。
