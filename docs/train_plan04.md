# 训练计划 04：ResNet-50 与 DenseNet-121

两组实验各训练一次，在相同训练设置下比较模型及其 ImageNet 预训练权重的组合。训练在远程 GPU 环境执行；实际结果回填 [training_log.md](training_log.md)，对照结论记录在 [experiments.md](experiments.md)。

## 实验配置

| 实验 | 配置 | 骨干 | seed | 最大 epoch | 早停耐心值 | num_workers |
| --- | --- | --- | --- | --- | --- | --- |
| RUN-005 | `configs/exp_resnet50.yaml` | `resnet50` | 42 | 25 | 5 | 16 |
| RUN-006 | `configs/exp_densenet121.yaml` | `densenet121.ra_in1k` | 42 | 25 | 5 | 16 |

两份配置继承 `configs/default.yaml`。当前共同设置包括：320×320 letterbox、ImageNet 归一化、`batch_studies: 16`、`pos_weight: true`、学习率 0.0003、前 2 轮 warmup 后余弦衰减至 0.00001、CUDA AMP，以及 `pretrained: true`。实际图片数取决于每个 study 的图片数。连续 5 个 epoch 的验证集 Macro-AUC 没有严格提升即早停，因此不保证跑满 25 轮。

在仓库根目录核对两组当前生效配置；如果断言失败，先核查配置变化，不要直接启动实验：

```bash
PYTHONPATH=src python - <<'PY'
from common.config import load_config

resnet = load_config('configs/exp_resnet50.yaml').to_dict()
densenet = load_config('configs/exp_densenet121.yaml').to_dict()
assert resnet['model'].pop('backbone') == 'resnet50'
assert densenet['model'].pop('backbone') == 'densenet121.ra_in1k'
assert resnet == densenet, '除骨干外的配置不一致'
assert resnet['model']['pretrained'] is True
assert resnet['seed'] == 42
assert resnet['image']['size'] == 320
assert resnet['train']['epochs'] == 25
assert resnet['train']['early_stop_patience'] == 5
assert resnet['train']['num_workers'] == 16
assert resnet['train']['batch_studies'] == 16
assert resnet['train']['pos_weight'] is True
assert resnet['train']['lr'] == 0.0003
assert resnet['train']['warmup_epochs'] == 2
assert resnet['train']['lr_scheduler'] == 'cosine'
assert resnet['train']['lr_min'] == 0.00001
print('两组配置核对通过；除骨干外相同。')
PY
```

确认全量数据体检通过，且两次运行使用同一份 train、valid 数据和同一代码版本。两组在同一训练机上串行执行，避免同时占用 GPU 和两组 16 workers。

## 训练命令

```bash
python src/check_data.py --config configs/default.yaml --full

python src/run_train.py --config configs/exp_resnet50.yaml \
    --seed 42 --tag resnet50_s42

python src/run_train.py --config configs/exp_densenet121.yaml \
    --seed 42 --tag densenet121_s42
```

每次启动后检查终端或 `train.log` 中的 `预训练权重: 已加载`。程序可能在加载失败时回退到随机初始化；如果显示未加载，停止该组，排查权重下载或缓存后从头启动，不把该次结果当作预训练实验。若显存不足，也停止该组并记录设备、配置和错误；排除问题后再运行，勿把改变训练设置后的结果当成同设置对照。

正常完成的 run 保存在 `outputs/<时间戳>_<tag>/`，包含解析后的 `config.yaml`、`train.log`、`metrics.csv`、`best.pt` 和 `last.pt`。检查归档配置及 checkpoint 的 `metadata.pretrained_loaded`，确认实际使用的设置和权重加载状态。

## 结果比较

把下面两个占位路径替换为实际 run 目录，再在同一验证集上重算两组各自 `best.pt` 的指标：

```bash
python src/run_compare.py \
    --runs 'outputs/<resnet50_run>' 'outputs/<densenet121_run>' \
    --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 \
    --out outputs/compare_backbones
```

查看 `outputs/compare_backbones/summary.md` 和 `per_class.csv`：先比较 study 级 Macro-AUC，再结合 Macro-F1 阈值曲线、逐类 AUC/F1 与样本数判断。记录每组最佳 epoch、实际轮数、停止原因、设备、运行时间及预训练加载情况。每组只有一个 seed，结果不能证明不同随机种子下的稳定性。

历史 ResNet-18 结果可以作为参考；此前实验的学习率调度或 `pos_weight` 与当前设置不同，不能将差异直接归因于骨干。两组新实验的权重也采用不同预训练配方，比较的是模型和权重的组合，而非仅隔离网络架构的影响：参见 [ResNet-50 模型卡](https://huggingface.co/timm/resnet50.a1_in1k)与 [DenseNet-121 模型卡](https://huggingface.co/timm/densenet121.ra_in1k)。

最终候选需核验实际使用的预训练权重公开链接、许可、上游数据说明与 SHA-256，并在赛题指定 T4 环境实测模型文件大小和端到端耗时。多图 study 的最终耗时口径尚未明确，按 [cost.md](cost.md) 记录限制，不能由单图候选耗时直接认定合规。
