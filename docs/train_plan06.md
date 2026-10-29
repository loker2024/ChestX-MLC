# 训练计划 06：RUN-007（DenseNet-121 + 正侧位加权聚合）

## 目的与对照

以 seed 42 的 RUN-006 为对照，观察正侧位 7:3 加权聚合对 DenseNet-121 的影响。
两组使用相同的训练、验证数据及训练设置，计划只改变聚合方式。

| 实验 | 配置 | 骨干 | 聚合 | seed |
| --- | --- | --- | --- | --- |
| RUN-006 | `configs/exp_densenet121.yaml` | `densenet121.ra_in1k` | 图片等权均值 | 42 |
| RUN-007 | `configs/exp_run007.yaml` | `densenet121.ra_in1k` | 正侧位 7:3 | 42 |

RUN-007 继承 RUN-006 配置：320×320、`pos_weight: true`、warmup + cosine、最多
25 轮、早停耐心值 5。加权聚合先分别求正位（AP、PA）与侧位（LATERAL、LL）
的 logits 均值，再按 0.7 / 0.3 合并。只有一种已知视角或包含未知视角时，回退为
该 study 全部图片的等权均值。聚合同时用于训练和验证。

## 启动前核对

在仓库根目录核对解析后的配置：

```bash
PYTHONPATH=src python - <<'PY'
from common.config import load_config

baseline = load_config('configs/exp_densenet121.yaml').to_dict()
run007 = load_config('configs/exp_run007.yaml').to_dict()
assert baseline['model']['aggregation'] == 'mean'
assert run007['model']['aggregation'] == 'view_weighted'
assert run007['model']['frontal_weight'] == 0.7
baseline['model']['aggregation'] = run007['model']['aggregation']
assert baseline == run007, '除聚合方式外配置不一致'
print('配置核对通过')
PY
```

确认全量数据体检通过，并与 RUN-006 使用同一份 train / valid 数据。对照 RUN-006
保存的配置、代码版本和运行环境；若代码或环境存在差异，记录差异，不能将结果变化
全部归因于聚合方式。

```bash
python src/check_data.py --config configs/exp_run007.yaml --full
```

## 训练与比较

```bash
python src/run_train.py --config configs/exp_run007.yaml \
    --seed 42 --tag densenet121_viewweighted_s42
```

检查日志中的预训练权重是否已加载；未加载则排查后重新启动。保存运行目录中的
`config.yaml`、`train.log`、`metrics.csv`、`best.pt` 和 `last.pt`。

```bash
python src/run_compare.py \
    --runs 'outputs/<run006_run>' 'outputs/<run007_run>' \
    --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 \
    --out outputs/compare_run007
```

主要比较 study 级 macro-AUC，并查看逐类 AUC 和各阈值下的 macro-F1；记录最优
epoch、运行时间及代码版本。单个 seed 的小幅差异只能作为初步结果，若结果值得
继续，再做匹配 seed 的复核。完成后将实测回填 `docs/training_log.md`，将对照结论
记录到 `docs/experiments.md`。
