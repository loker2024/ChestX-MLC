# ChestX-MLC 训练记录

本文件记录模型冻结之后的**全部训练 run**：一个骨干一个二级标题，一次 run 一条三级标题记录，末尾汇总。

规则：

- 数字只从 `outputs/<run>/metrics.csv` 与 `run_compare` 生成的 `outputs/compare/summary.md` 复制，不手抄训练日志。
- 每次 run 把实际使用的配置文件原样贴进代码块，并给出 `configs/` 链接；正文只写「相对父配置改了什么」。
- 训练产物（`best.pt` / `last.pt` / `metrics.csv` / `train.log`）按约定不入库，只记录路径。
- 换骨干或输入尺寸时新增二级标题；数据流和模块职责见 [architecture.md](architecture.md)，配置及运行命令见 [manual.md](manual.md)。

## ResNet-18 @320（timm `resnet18.a1_in1k`）

### 记录模板

~~~markdown
### RUN-00x <tag>（<配置文件>）

配置 [configs/xxx.yaml](../configs/xxx.yaml)，完整内容：

```yaml
<该配置文件原样贴入>
```

| 项 | 值 |
|---|---|
| 相对父配置的差异 | <只写改动的字段> |
| 运行命令 | `<命令>` |
| 设备 / 环境 | <GPU> / <torch> / <timm> / seed |
| 数据 | <train 图数与 study 数；valid 图数与 study 数> |
| 预训练 | <是否加载 `resnet18.a1_in1k`> |
| 产物 | `outputs/<run>/` |
| 耗时 | <总时长 / 每轮> |

metrics.csv（原样复制）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|

**结论**：<AUC/F1 谁赢、下一步是否动默认值>
~~~

### RUN-001 short_baseline（`configs/exp_short.yaml`）

配置 [configs/exp_short.yaml](../configs/exp_short.yaml)，完整内容：

```yaml
# 快速筛查配置：把 epoch 数压短，只用来比较「哪个选项方向更好」，不追求最终成绩。
#
# 用法（在仓库根目录执行）：
#     python src/run_train.py --config configs/exp_short.yaml --tag short_baseline
#
# 为什么不在 default.yaml 上直接改：默认配置是最终提交用的完整配置，
# 筛查只是阶段性的，单独放一份、通过 _base 继承默认值，两边的改动就不会互相污染。
_base: default.yaml

train:
  epochs: 6
  warmup_epochs: 2
  early_stop_patience: 0   # 只跑 6 轮，不早停，保证两组对照跑满同样的轮数
```

| 项 | 值 |
|---|---|
| 相对父配置的差异 | `train.epochs: 25 → 6`、`train.early_stop_patience: 5 → 0` |
| 运行命令 | `python src/run_train.py --config configs/exp_short.yaml --seed 42 --tag short_baseline` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 42 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 预训练 | 已加载 `resnet18.a1_in1k` |
| 产物 | `outputs/20261027-050417_short_baseline/`（远端 `/root/ChestX-MLC`，UTC 05:04:17 / 北京 13:04 启动） |
| 耗时 | 28m41s（05:04:17 → 05:32:58 UTC，每轮约 4m47s，含验证与落盘） |

metrics.csv（原样复制）：

| epoch | lr | train_loss | valid_loss | macro_auc | macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.3572 | 0.3417 | 0.7162 | 0.1310 |
| 2 | 3.00e-04 | 0.3313 | 0.3280 | 0.7579 | 0.2164 |
| 3 | 3.00e-04 | 0.3186 | 0.3370 | 0.7665 | 0.2317 |
| 4 | 3.00e-04 | 0.3106 | 0.3308 | 0.7719 | 0.2134 |
| 5 | 3.00e-04 | 0.3030 | 0.3293 | 0.7818 | 0.2562 |
| 6 | 3.00e-04 | 0.2944 | 0.3199 | 0.7858 | 0.2796 |

**结论**：macro-AUC 六轮单调上升、最优即最后一轮（0.7858），表明 6 轮时仍在改善，短周期适合筛查方向，是否收敛需结合更长训练验证；F1@0.5 为 0.2796；AUC 和 F1 衡量不同性质，不能凭两者数值差判断阈值高低，需结合 `run_compare` 的阈值曲线判断。

### RUN-002 short_pos_weight（`configs/exp_short_pos_weight.yaml`）

配置 [configs/exp_short_pos_weight.yaml](../configs/exp_short_pos_weight.yaml)，完整内容：

```yaml
# 筛查用的 pos_weight 配置：继承 exp_short.yaml（短周期），再打开 pos_weight。
#
# 用法（在仓库根目录执行）：
#     python src/run_train.py --config configs/exp_short_pos_weight.yaml --tag short_pos_weight
#
# 和 exp_short.yaml 用同一套短周期，两组结果才有可比性；正式对比时换成
# exp_pos_weight.yaml（完整 25 轮）。
_base: exp_short.yaml

train:
  pos_weight: true
```

| 项 | 值 |
|---|---|
| 相对父配置的差异 | `train.pos_weight: false → true`，其余同 RUN-001 |
| 运行命令 | `python src/run_train.py --config configs/exp_short_pos_weight.yaml --seed 42 --tag short_pos_weight` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 42 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 预训练 | 已加载 `resnet18.a1_in1k` |
| 产物 | 待填 |
| 耗时 | 待填 |

metrics.csv（跑完从 run 目录复制）：

| epoch | lr | train_loss | valid_loss | macro_auc | macro_f1 |
|---|---|---|---|---|---|

**结论**：待填（AUC 与整条 F1 阈值曲线同 RUN-001 对比）。

### RUN-003 full_baseline（`configs/default.yaml`）

配置 [configs/default.yaml](../configs/default.yaml)，完整内容：

```yaml
data:
  train:
    csv: data/train.csv
    images: data/images/train
  valid:
    csv: data/valid.csv
    images: data/images/valid
  test:
    csv: data/test.csv
    images: data/images/test

image:
  size: 320
  letterbox: true
  normalize: imagenet
  hflip: false

model:
  backbone: resnet18
  pretrained: true
  num_classes: 10

train:
  epochs: 25
  batch_studies: 16
  lr: 0.0003
  weight_decay: 0.01
  pos_weight: false
  warmup_epochs: 2
  amp: true
  grad_clip: 5.0
  num_workers: 8
  early_stop_patience: 5

eval:
  batch_studies: 32
  threshold: 0.5

output:
  run_dir: outputs

seed: 42
device: auto
```

（以上为该文件的超参数部分；完整文件含逐字段中文注释，见 [configs/default.yaml](../configs/default.yaml)。）

| 项 | 值 |
|---|---|
| 相对父配置的差异 | 无（冻结配置本体） |
| 运行命令 | `python src/run_train.py --config configs/default.yaml --seed 42 --tag full_baseline` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 42 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 预训练 | 待确认 |
| 产物 | 待填 |
| 耗时 | 待填 |

metrics.csv（跑完从 run 目录复制）：

| epoch | lr | train_loss | valid_loss | macro_auc | macro_f1 |
|---|---|---|---|---|---|

**结论**：待填（冻结配置的正式成绩，用于生成 `submission.csv`）。

### 汇总

| run | 配置 | 关键差异 | epochs | best_epoch | macro-AUC | F1@0.5 | 耗时 | 结论 |
|---|---|---|---|---|---|---|---|---|
| RUN-001 | `exp_short.yaml` | 短周期基准 | 6 | 6 | 0.7858 | 0.2796 | 28m41s | 末轮仍改善，仅作方向基准 |
| RUN-002 | `exp_short_pos_weight.yaml` | +pos_weight | 6 | | | | | 待运行 |
| RUN-003 | `default.yaml` | 25 轮冻结版 | 25 | | | | | 待运行 |

判读口径：先看 macro-AUC（不受阈值影响）；再看 `f1@0.5` 与整条阈值曲线（`run_compare` 的 `summary.md`）；`pos_weight` 只有在多数类同时不退化时才算有效。

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 3090 24G（驱动 535.247.01，CUDA 12.2） |
| Python / 框架 | Python 3.12、torch 2.10.0+cu128、timm 1.0.30、torchvision 0.25.0+cu128 |
| 数据根目录 | `/root/sj-tmp`（`data/` 下以软链接挂入，数据本身不入库） |
| 随机种子 | 42（所有 run 相同） |
