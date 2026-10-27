# ChestX-MLC 训练记录

本文件记录模型冻结之后的**全部训练 run**：一个骨干一个二级标题，一次 run 一条三级标题记录，末尾汇总。

规则：

- 数字只从 `outputs/<run>/metrics.csv` 与 `run_compare` 生成的 `outputs/compare/summary.md` 复制，不手抄训练日志。
- 每次 run 把实际使用的配置文件原样贴进代码块，并给出 `configs/` 链接；正文只写「相对父配置改了什么」。
- 训练产物（`best.pt` / `last.pt` / `metrics.csv` / `train.log`）按约定不入库，只记录路径。
- 换骨干或换输入尺寸就新增一个二级标题，各自带流程图、推理/评估/对比、调参索引，互不覆盖。

## ResNet-18 @320（timm `resnet18.a1_in1k`）

### 流程图

```text
data/{train,valid}.csv + data/images/{train,valid}
   │ read_split() 按 Study_id 聚合、校验同 study 标签一致、缺图报错
   │ src/data/index.py:41 index_images / :70 read_split
   ▼
StudyRecord(study_id, subject_id, images[n], labels[10])
   │ StudyDataset.__getitem__                       src/data/dataset.py:44
   │ Letterbox(320) + ImageNet 归一化          src/data/transforms.py:21 / :60
   ▼
collate_studies（首尾相接，不补假图）               src/data/dataset.py:67
   images[N,3,320,320]  offsets[B+1]  labels[B,10]  study_ids[B]
   N = 本批 B 个 study 的真实图片数（B=16，平均约 26 张）
   │
   ▼ build_model                                      src/model/network.py:26
resnet18.a1_in1k（11,181,642 参数）
   conv1 → layer1..layer4 → avgpool → fc(512→10)
   ▼ 逐图 forward ⇒ logits[N,10]
   ▼ aggregate_logits：同 study 多图取均值           src/model/network.py:58
study logits[B,10]
   ├─ 训练：BCEWithLogitsLoss（pos_weight 可选）     src/model/losses.py:20
   │        train_one_epoch：autocast → backward    src/train/engine.py:102
   │        → clip_grad_norm_(5.0) → scaler.step     src/train/engine.py:129
   └─ 验证：sigmoid → macro_auc / macro_f1      src/common/metrics.py:95 / :111
            evaluate                                src/train/engine.py:147
   ▼
fit：AdamW(3e-4, 0.01) + warmup 2 + AMP + 早停     src/train/engine.py:189
   ▼
outputs/<YYYYmmdd-HHMMSS>_<tag>/
   config.yaml · train.log · metrics.csv · last.pt（每轮）· best.pt（最优）
```

### 推理 / 评估 / 对比

```text
① 推理（出提交）
run_infer.py --config … --checkpoint outputs/<run>/best.pt --out outputs/submission.csv
  build_loader("test")                     src/data/dataset.py:93
  load_checkpoint_model(pretrained=False)  src/infer/predict.py:28
  predict_loader → Study_id + p1..p10      src/infer/predict.py:47
  build_submission_frame(threshold=None)   src/infer/submission.py:32
  write_submission → validate_submission   src/infer/submission.py:83 / :95

② 单 run 评估（阈值网格 + 逐类明细）
run_eval.py --checkpoint … --thresholds 0.3,0.4,0.5,0.6,0.7
  per_class_report + macro_f1_curve        src/common/metrics.py:27 / :142

③ 多 run 对比（本文件的数字来源）
run_compare.py --runs <run_a> <run_b> --thresholds …
  evaluate_run → write_summary            src/run_compare.py:70 / :133
  → outputs/compare/summary.{csv,md}
```

### 参数 → 生效位置（调参索引）

| 想调的东西 | 配置项 | 生效位置 |
|---|---|---|
| 轮数 | `train.epochs` | `src/train/engine.py:189` |
| 批大小 / 显存 | `train.batch_studies` | `src/data/dataset.py:117` |
| 分辨率 / 速度 | `image.size` | `src/data/transforms.py:21` |
| 损失加权 | `train.pos_weight` | `src/model/losses.py:20` |
| 学习率 / 权重衰减 | `train.lr` / `train.weight_decay` | `src/train/engine.py:219` |
| warmup | `train.warmup_epochs` | `src/train/engine.py:80` |
| 混合精度 | `train.amp` | `src/train/engine.py:224` |
| 梯度裁剪 | `train.grad_clip` | `src/train/engine.py:129` |
| 数据吞吐（核数） | `train.num_workers` | `src/data/dataset.py:129` |
| 早停 | `train.early_stop_patience` | `src/train/engine.py:189` |
| 判定阈值 | `eval.threshold` | `src/common/metrics.py:111` |
| 骨干 | `model.backbone` | `src/model/network.py:26` |
| 预训练开关 | `model.pretrained` | `src/model/network.py:26` |

### 记录模板

~~~markdown
### RUN-00x <tag>（<配置文件>）

配置 [configs/xxx.yaml](configs/xxx.yaml)，完整内容：

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

| epoch | lr | train_loss | valid_loss | macro_auc | macro_f1 |
|---|---|---|---|---|---|

**结论**：<AUC/F1 谁赢、下一步是否动默认值>
~~~

### RUN-001 short_baseline（`configs/exp_short.yaml`）

配置 [configs/exp_short.yaml](configs/exp_short.yaml)，完整内容：

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

**结论**：macro-AUC 六轮单调上升、最优即最后一轮（0.7858），说明 6 轮远未跑满收敛，短周期只适合判方向、不能当最终成绩；F1@0.5 仅 0.2796，与 macro-AUC 差距大，阈值默认 0.5 明显偏高（后续用 `run_compare` 的阈值曲线看最优点）。

### RUN-002 short_pos_weight（`configs/exp_short_pos_weight.yaml`）

配置 [configs/exp_short_pos_weight.yaml](configs/exp_short_pos_weight.yaml)，完整内容：

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

配置 [configs/default.yaml](configs/default.yaml)，完整内容：

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

（以上为该文件的超参数部分；完整文件含逐字段中文注释，见 [configs/default.yaml](configs/default.yaml)。）

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
| RUN-001 | `exp_short.yaml` | 短周期基准 | 6 | 6 | 0.7858 | 0.2796 | 28m41s | 未收敛，仅作方向基准 |
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
