# 训练计划 05：RUN-006-A（DenseNet-121 的 seed 复核）

RUN-006（DenseNet-121 骨干对照）只有 seed 42 一个种子，与同口径 ResNet-18 基线
RUN-003-B 的 macro-AUC 差为 +0.0043。本计划用 seed 2026 复跑 RUN-006 的口径，
观察 DenseNet-121 的第二次结果；要检验骨干差异在第二个 seed 下的方向，还需
补跑相同训练口径的 ResNet-18 seed 2026。RUN-001 与 RUN-001R 均为 seed 42，
其 0.0045 差值不能作为换 seed 的波动。

训练在远程 GPU 环境执行；实际结果回填 [training_log.md](training_log.md)，对照结论
记录在 [experiments.md](experiments.md)。

## 实验配置

| 实验 | 配置 | 骨干 | seed | 最大 epoch | 早停耐心值 | num_workers |
| --- | --- | --- | --- | --- | --- | --- |
| RUN-006 | `configs/exp_densenet121.yaml` | `densenet121.ra_in1k` | 42 | 25 | 5 | 16 |
| RUN-006-A | `configs/exp_run006a.yaml` | `densenet121.ra_in1k` | 2026 | 25 | 5 | 16 |

`configs/exp_run006a.yaml` 以 `_base: exp_densenet121.yaml` 继承 RUN-006 的全部设置，
只覆盖顶层 `seed: 2026`；其余字段（320×320 letterbox、ImageNet 归一化、
`batch_studies: 16`、`pos_weight: true`、学习率 0.0003、前 2 轮 warmup 后余弦衰减到
0.00001、CUDA AMP、`pretrained: true`、连续 5 轮无提升早停）完全一致。与 RUN-006
的配置差异仅为随机种子；两次实际代码版本和运行负载也不同，不能断言结果差异只由 seed 造成；`tests/test_config.py` 的
`test_run006a_experiment_config_loads` 已断言两份配置「除 seed 外零差异」。

启动前可在仓库根目录复核一次：

```bash
PYTHONPATH=src python - <<'PY'
from common.config import load_config

run006 = load_config('configs/exp_densenet121.yaml').to_dict()
run006a = load_config('configs/exp_run006a.yaml').to_dict()
differences = {key: (run006[key], run006a[key]) for key in run006 if run006[key] != run006a[key]}
assert differences == {'seed': (42, 2026)}, differences
print('核对通过：除 seed 外与 RUN-006 完全一致。')
PY
```

确认全量数据体检通过（`python src/check_data.py --config configs/default.yaml --full`），
并确认与 RUN-006 使用同一份 train / valid 数据。

## 训练命令

```bash
python src/run_train.py --config configs/exp_run006a.yaml \
    --seed 2026 --tag densenet121_s2026
```

启动后检查终端或 `train.log` 中的 `预训练权重: 已加载`；未加载就停止该组，排查权重
缓存后从头启动，不把该次结果当作预训练实验。正常完成的 run 保存在
`outputs/<时间戳>_<tag>/`，包含解析后的 `config.yaml`、`train.log`、`metrics.csv`、
`best.pt` 和 `last.pt`。

## 结果比较

```bash
python src/run_compare.py \
    --runs 'outputs/<run006_run>' 'outputs/<run006a_run>' \
    --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 \
    --out outputs/compare_run006a
```

判读时先比较 RUN-006 与 RUN-006-A 的 study 级 macro-AUC、阈值曲线与逐类 AUC，
这只能描述 DenseNet 的两次结果。骨干对照需在同一验证集、相同训练口径下，
分别比较 seed 42 和 seed 2026 的 DenseNet 与 ResNet-18；目前后者尚缺配对结果。
即使两个配对差值方向一致，两个 seed 仍不足以证明优势稳定。

RUN-006-A 与 experiment4 的 EfficientNet-B2 实验并行执行（共用 3090 与 16 核 CPU），
耗时不可与 RUN-006 的独占 44m32s 直接比较；两组 `num_workers` 均为 16；不能仅凭此断言并行运行不会影响数值结果。
