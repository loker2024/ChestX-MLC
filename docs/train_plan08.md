# 训练计划 08：RUN008-A / B / C / D 新骨干对照

## 目的与基线

在 RUN-007-A 的正侧位 7:3 加权聚合口径上，比较四组骨干及其对应的 ImageNet 预训练权重。RUN-007-A 为已有对照：`densenet121.ra_in1k`、seed 2026，验证集 study 级 Macro-AUC **0.7906**、Macro-F1@0.5 **0.4198**（重算结果，见 [训练记录](training_log.md)）。这不是官方得分。

四组继承 `configs/exp_run007a.yaml`，覆盖 `model.backbone` 并统一改为固定 6 轮、关闭早停；比较的是**骨干与对应权重的组合**，不能将差异全部归因于架构。与 RUN-007-A 相比还多出训练轮数与早停策略差异，见「对照与判读」。RUN-007-A 曾与同 seed 等权聚合对照得到 AUC −0.0007，两个 seed 的加权差值方向不一致；本计划按要求保留 7:3 口径，不将其作为聚合有效性的证明。[实验记录](experiments.md)

| 实验 | 待创建配置 | timm 骨干及权重标识 | tag |
| --- | --- | --- | --- |
| RUN008-A | `configs/exp_run008a.yaml` | `efficientnet_b2.ra_in1k` | `run008a_effb2_s2026` |
| RUN008-B | `configs/exp_run008b.yaml` | `efficientnetv2_rw_s.ra2_in1k` | `run008b_effv2s_s2026` |
| RUN008-C | `configs/exp_run008c.yaml` | `convnext_tiny.fb_in1k` | `run008c_convnext_tiny_s2026` |
| RUN008-D | `configs/exp_run008d.yaml` | `convnextv2_nano.fcmae_ft_in1k` | `run008d_convnextv2_nano_s2026` |

固定条件：seed 2026、320×320 letterbox、ImageNet 归一化、10 类、`pretrained: true`、`batch_studies: 16`、`pos_weight: true`、学习率 0.0003、2 轮 warmup 后 cosine 至 0.00001、CUDA AMP、固定 6 epoch、无早停（`early_stop_patience: 0`）、`num_workers: 16`。四组按 A+B、C+D 两两并行执行、共享同一块 GPU，训练耗时不可横向比较。每个 batch 的实际图片数由 study 图片数决定。聚合在训练和验证均使用：先分别对正位 / 侧位 logits 求均值，再按 0.7 / 0.3 加权；仅一种已知视角或含未知视角时回退到全部图片等权 logits 均值。

## 权重和数据许可

执行前逐项核验实际下载的权重文件、上游预训练数据、许可及赛题规则，并记录精确模型标识、链接、文件 SHA-256 和环境版本；许可或预训练数据是否符合赛题要求无法确认时，暂不执行该组。以下模型卡是具体版本的来源入口，模型卡标注不替代赛题规则核验：

- RUN008-A：[EfficientNet-B2](https://huggingface.co/timm/efficientnet_b2.ra_in1k)，模型卡标注 Apache-2.0。
- RUN008-B：[EfficientNetV2-RW-S](https://huggingface.co/timm/efficientnetv2_rw_s.ra2_in1k)，timm 变体，模型卡标注 Apache-2.0。
- RUN008-C：[ConvNeXt-Tiny](https://huggingface.co/timm/convnext_tiny.fb_in1k)，模型卡标注 Apache-2.0。
- RUN008-D：[ConvNeXt V2-Nano](https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in1k)，模型卡标注 **CC BY-NC 4.0**，且采用 FCMAE 预训练后微调；须单独核对用途及比赛规则，不能假定适用。

不要因权重暂不可用而改用其他 tag 或随机初始化；记录实际使用文件的 SHA-256，而不是用模型名称代替权重身份。模型卡中的参数量、输入尺寸与论文训练配方不能证明本项目效果或 T4 耗时。

## 启动前准备

训练在远程 GPU 环境执行。先确认 RUN-007-A 的 run 目录、`best.pt`、训练代码版本、train / valid 数据及其内容与当前环境一致；记录 `git rev-parse HEAD`、工作区改动、Python / torch / timm 版本、GPU 和数据文件校验值。原 RUN-007-A 的代码版本见 [训练记录](training_log.md)；若代码、数据或环境有变化，比较时明确标注差异，不能认定与旧 run 构成严格单变量对照。为获得严格配对证据，可在相同环境和代码下额外重跑 RUN-007-A；它不计入本计划四组。

本计划只定义配置，以下四个配置文件需在执行前创建；**若文件已存在，先检查内容，不覆盖已有文件**。新增配置时同步更新 `CHANGELOG.md`。

```bash
python - <<'PY'
from pathlib import Path

models = {
    'a': 'efficientnet_b2.ra_in1k',
    'b': 'efficientnetv2_rw_s.ra2_in1k',
    'c': 'convnext_tiny.fb_in1k',
    'd': 'convnextv2_nano.fcmae_ft_in1k',
}
paths = [Path(f'configs/exp_run008{k}.yaml') for k in models]
existing = [str(path) for path in paths if path.exists()]
if existing:
    raise SystemExit(f'配置已存在，请先核查，未写入：{existing}')
for key, name in models.items():
    path = Path(f'configs/exp_run008{key}.yaml')
    with path.open('x', encoding='utf-8') as f:
        f.write(f'_base: exp_run007a.yaml\n\nmodel:\n  backbone: {name}\n')
    print(path, name)
PY
```

配置继承可能随父配置改变；每次训练前检查四组解析值与当前 RUN-007-A 仅差 `model.backbone` 与统一的 `train.epochs` / `train.early_stop_patience` 覆盖，并核对关键固定条件：

```bash
PYTHONPATH=src python - <<'PY'
import copy

from common.config import load_config

base = load_config('configs/exp_run007a.yaml').to_dict()
assert base['seed'] == 2026
assert base['image']['size'] == 320 and base['image']['letterbox'] is True
assert base['image']['normalize'] == 'imagenet'
assert base['model']['backbone'] == 'densenet121.ra_in1k'
assert base['model']['pretrained'] is True and base['model']['num_classes'] == 10
assert base['model']['aggregation'] == 'view_weighted'
assert base['model']['frontal_weight'] == 0.7
train = base['train']
assert train['batch_studies'] == 16 and train['pos_weight'] is True
assert train['lr'] == 0.0003 and train['warmup_epochs'] == 2
assert train['lr_scheduler'] == 'cosine' and train['lr_min'] == 0.00001
assert train['amp'] is True and train['epochs'] == 25
assert train['early_stop_patience'] == 5 and train['num_workers'] == 16
expected = {
    'a': 'efficientnet_b2.ra_in1k', 'b': 'efficientnetv2_rw_s.ra2_in1k',
    'c': 'convnext_tiny.fb_in1k', 'd': 'convnextv2_nano.fcmae_ft_in1k',
}
reference = copy.deepcopy(base)
reference['model'].pop('backbone')
assert reference['train'].pop('epochs') == 25
assert reference['train'].pop('early_stop_patience') == 5
for key, name in expected.items():
    config = load_config(f'configs/exp_run008{key}.yaml').to_dict()
    assert config['model'].pop('backbone') == name
    assert config['train'].pop('epochs') == 6
    assert config['train'].pop('early_stop_patience') == 0
    assert config == reference, f'RUN008-{key.upper()} 除骨干与训练预算外的配置不一致'
    print(f'RUN008-{key.upper()} 配置通过：{name}')
PY
```

上面的检查只证明当前解析的配置相同，不证明与历史 run 完全相同。还须读取 RUN-007-A 的 `best.pt` 内存档配置（缺失时用该 run 的 `config.yaml`）、数据及日志，与当前解析配置逐项核对；若路径不同，核实实际数据内容一致并记录路径差异。`run_compare.py` 优先使用各 `best.pt` 的存档配置和其中的 valid 路径，不会自动强制使用同一验证集。

确认完整数据体检通过：

```bash
python src/check_data.py --config configs/exp_run007a.yaml --full
```

在训练机用已安装的 timm 验证四个名称、320×320 的 10 类输出及预训练权重能实际加载；可逐组运行以下检查，将 `MODEL` 设为上表对应的完整标识：

```bash
MODEL=efficientnet_b2.ra_in1k python - <<'PY'
import os
import timm
import torch

name = os.environ['MODEL']
model = timm.create_model(name, pretrained=True, num_classes=10).eval()
with torch.inference_mode():
    output = model(torch.zeros(1, 3, 320, 320))
assert output.shape == (1, 10), output.shape
print(name, '预训练加载和输出形状通过', tuple(output.shape))
PY
```

该检查可能下载权重，需在有网络或已有缓存的环境执行；成功不代替记录权重来源、许可和哈希。预检查未通过、显存不足或配置不符时停止该组，记录原因；不要自行改 batch、输入尺寸或其他参数并继续称为同口径对照。

## 两两并行训练

四组按 A+B、C+D 两批两两并行执行：同一块 GPU 上每批两个进程共享算力与各自的 16 workers，训练耗时不可横向比较，也不能用于与其他 run 的耗时对照。先同时启动 A、B，两组都结束后再同时启动 C、D；某组失败先排查，不改变其他组条件。每组固定 6 轮、关闭早停，跑满 6 轮后结束。

```bash
# 第一批：A + B 同时启动
nohup python src/run_train.py --config configs/exp_run008a.yaml --seed 2026 --tag run008a_effb2_s2026 > outputs/run008a.nohup.log 2>&1 &
nohup python src/run_train.py --config configs/exp_run008b.yaml --seed 2026 --tag run008b_effv2s_s2026 > outputs/run008b.nohup.log 2>&1 &

# 第二批：A、B 都结束后，C + D 同时启动
nohup python src/run_train.py --config configs/exp_run008c.yaml --seed 2026 --tag run008c_convnext_tiny_s2026 > outputs/run008c.nohup.log 2>&1 &
nohup python src/run_train.py --config configs/exp_run008d.yaml --seed 2026 --tag run008d_convnextv2_nano_s2026 > outputs/run008d.nohup.log 2>&1 &
```

每组启动后检查 `train.log` 的 `预训练权重: 已加载`；加载失败可能回退随机初始化，须停止并从头重跑。完成后核对 `best.pt` 的 `metadata.pretrained_loaded` 为 `True`，并保存 `config.yaml`、`train.log`、`metrics.csv`、`best.pt`、`last.pt`，记录实际权重文件哈希。配置中 `pretrained: true` 或该标志为真，不能单独证明权重文件的具体来源及哈希。

## 对照与判读

确认五组访问的是同一份 valid 数据后，将下面路径替换为实际 run 目录并重算：

```bash
python src/run_compare.py \
    --runs 'outputs/<run007a>' 'outputs/<run008a>' 'outputs/<run008b>' \
           'outputs/<run008c>' 'outputs/<run008d>' \
    --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 \
    --out outputs/compare_run008
```

查看 `summary.md`、`per_class.csv`：以 study 级 Macro-AUC 为主要指标，记录每组相对 RUN-007-A 的差值、逐类 AUC 与样本数、各阈值 Macro-F1、最佳 epoch、实际轮数、停止原因、训练设备 / 耗时、代码版本及权重状态。阈值改变 F1，不改变使用连续概率计算的 AUC；验证指标与官方线上得分分别记录。

四组固定 6 轮且关闭早停，RUN-007-A 为 25 轮上限、第 12 轮早停（峰值 epoch 7），两者训练预算不同；与 RUN-007-A 的差值只作方向性参考，不能当作严格的单变量对照。

可沿用历史 **AUC ≥ +0.005 且逐类无明显退化**作为筛选参考，不把它当作统计显著性结论。四组均只有 seed 2026；对有希望的候选用另一 seed 与同口径 DenseNet 配对复核，确认数据、配置、代码和环境差异，不能凭单次验证结果宣布胜出。不同模型所用预训练配方不同，结果属于模型与权重组合的比较；等效的配置不保证不同代码版本训练过程完全相同。

完成后将实际运行事实回填 [training_log.md](training_log.md)，将对照结论写入 [experiments.md](experiments.md)，并按规则更新 `CHANGELOG.md`。最终候选需在赛题指定 T4 上按 [cost.md](cost.md) 实测模型大小及端到端耗时；多图 study 的最终聚合耗时口径仍有文档所述限制，不能直接据单图耗时断言最终合规。
