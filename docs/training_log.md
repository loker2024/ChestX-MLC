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

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.3572 | 0.3417 | 0.7162 | 0.1310 |
| 2 | 3.00e-04 | 0.3313 | 0.3280 | 0.7579 | 0.2164 |
| 3 | 3.00e-04 | 0.3186 | 0.3370 | 0.7665 | 0.2317 |
| 4 | 3.00e-04 | 0.3106 | 0.3308 | 0.7719 | 0.2134 |
| 5 | 3.00e-04 | 0.3030 | 0.3293 | 0.7818 | 0.2562 |
| 6 | 3.00e-04 | 0.2944 | 0.3199 | 0.7858 | 0.2796 |

**结论**：macro-AUC 六轮单调上升、最优即最后一轮（0.7858），表明 6 轮时仍在改善，短周期适合筛查方向，是否收敛需结合更长训练验证；F1@0.5 为 0.2796；AUC 和 F1 衡量不同性质，不能凭两者数值差判断阈值高低，需结合 `run_compare` 的阈值曲线判断。

### RUN-001R short_baseline_r2（`configs/exp_short.yaml`）

配置与 RUN-001 完全相同（`_base: default.yaml` + 6 轮、warmup 2、不早停），见 RUN-001 段；本 run 的目的是**用当前代码（e8f194e）重跑同一配置，验证新代码没有改变训练结果**（回归门 [0.7808, 0.7908]）。

| 项 | 值 |
|---|---|
| 相对父配置的差异 | 无（与 RUN-001 同配置；差异是代码版本从 59a7d15 升到 e8f194e） |
| 运行命令 | `python src/run_train.py --config configs/exp_short.yaml --seed 42 --tag short_baseline_r2` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 42 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 预训练 | 已加载 `resnet18.a1_in1k` |
| 产物 | `outputs/20261028-002013_short_baseline_r2/` |
| 耗时 | 38m05s（00:20:13 → 00:58:18 Asia/Shanghai，与 RUN-002 并行；每轮约 6m20s） |

metrics.csv（原样复制）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.3571 | 0.3411 | 0.7178 | 0.1355 |
| 2 | 3.00e-04 | 0.3306 | 0.3330 | 0.7551 | 0.1880 |
| 3 | 3.00e-04 | 0.3185 | 0.3344 | 0.7557 | 0.2004 |
| 4 | 3.00e-04 | 0.3110 | 0.3252 | 0.7795 | 0.2239 |
| 5 | 3.00e-04 | 0.3037 | 0.3252 | 0.7822 | 0.2666 |
| 6 | 3.00e-04 | 0.2958 | 0.3206 | 0.7903 | 0.2651 |

**结论**：末轮 macro-AUC 0.7903，对比 RUN-001 的 0.7858，Δ+0.0045，落在判读门 [0.7808, 0.7908] 内 → **回归门通过**，该次结果达到预设回归门，但不能证明两版训练结果完全一致；F1@0.5 0.2649 略低于旧 0.2796，阈值曲线峰值在 0.2（=0.4012），见 `outputs/compare/summary.md`。耗时口径：本次与 RUN-002 并行（两路各 8 workers，16 核饱和），单轮 6m20s 不可与 RUN-001 串行 4m47s/轮直接对照；主进程负责 shuffle，但不能仅凭此断言 `num_workers` 不会影响训练结果。口径说明：表中 `valid_macro_f1` 列（console 打印为 `macro_f1`）为训练时按 0.5 阈值算出的 0.2651，`run_compare` 用同一份 valid 重算为 0.2649（差 0.0002，属重算实现差异）；本段与汇总表统一采用 `run_compare` 值，便于与 RUN-002 同口径比较。

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
| 产物 | `outputs/20261028-002512_short_pos_weight/` |
| 耗时 | 33m06s（00:25:12 → 00:58:18 Asia/Shanghai，与 RUN-001R 并行；每轮约 5m31s） |

metrics.csv（原样复制）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0309 | 0.9865 | 0.7308 | 0.3702 |
| 2 | 3.00e-04 | 0.9752 | 0.9454 | 0.7600 | 0.3810 |
| 3 | 3.00e-04 | 0.9344 | 0.9813 | 0.7522 | 0.3848 |
| 4 | 3.00e-04 | 0.9046 | 0.9448 | 0.7735 | 0.3945 |
| 5 | 3.00e-04 | 0.8761 | 0.9188 | 0.7882 | 0.4048 |
| 6 | 3.00e-04 | 0.8443 | 0.9348 | 0.7792 | 0.3890 |

**结论**：最优 macro-AUC 0.7882（epoch 5），对比 RUN-001R 的 0.7903 为 Δ−0.0021，**未达判读门 AUC ≥ +0.005**；逐类 AUC 呈「稀有类升、多数丰富类降」（见下表），正是判读门要排除的形态。F1 阈值曲线峰值在 0.6（0.4129）高于 RUN-001R 的 0.4012，但主指标 macro-AUC 不达标 → **pos_weight 判无效**，`train.pos_weight` 维持 `false`，RUN-004 走 004A（40 轮延长训练），不走 004B。口径说明：0.7882 取自 `metrics.csv`（epoch 5 训练时值），`run_compare` 用同一份 valid 重算为 0.7881（差 0.0001，Δ 按重算值为 −0.0022），结论不变。

逐类 AUC（`outputs/compare/per_class.csv`，Δ 为四位小数相减）：

| class_id | RUN-001R | RUN-002 | Δ（002−001R） |
|---|---|---|---|
| 1 | 0.7060 | 0.7648 | +0.0588 |
| 2 | 0.8477 | 0.8149 | −0.0328 |
| 3 | 0.7577 | 0.7508 | −0.0069 |
| 4 | 0.7250 | 0.6852 | −0.0398 |
| 5 | 0.8701 | 0.8697 | −0.0004 |
| 6 | 0.7706 | 0.7784 | +0.0078 |
| 7 | 0.7543 | 0.7631 | +0.0088 |
| 8 | 0.7175 | 0.7003 | −0.0172 |
| 9 | 0.8888 | 0.8884 | −0.0004 |
| 10 | 0.8656 | 0.8657 | +0.0001 |

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
  num_workers: 16
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
| 预训练 | 已加载 `resnet18.a1_in1k` |
| 产物 | `outputs/20261028-013241_full_baseline/`（`best.pt` 44.8 MB / `last.pt` 179 MB / `metrics.csv` / `train.log`）；提交文件 `outputs/submission_run003.csv`（1,414 studies × 10 类 = 14,140 行） |
| 耗时 | 2026-10-28 09:32:41 → 10:11 CST，**13 轮即早停**（未跑满 25 轮），共约 **38 分钟**；单轮约 3.3 分钟（`num_workers: 16`，较 8 worker 的并行口径快约 20%） |

metrics.csv（13 轮原样复制，epoch 13 后因连续 5 轮无提升早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.3576 | 0.3417 | 0.7189 | 0.1338 |
| 2 | 3.00e-04 | 0.3317 | 0.3329 | 0.7649 | 0.2099 |
| 3 | 3.00e-04 | 0.3197 | 0.3313 | 0.7660 | 0.2221 |
| 4 | 3.00e-04 | 0.3117 | 0.3234 | 0.7733 | 0.2141 |
| 5 | 3.00e-04 | 0.3055 | 0.3255 | 0.7827 | 0.2656 |
| 6 | 3.00e-04 | 0.2973 | 0.3202 | 0.7865 | 0.2713 |
| 7 | 3.00e-04 | 0.2903 | 0.3241 | 0.7768 | 0.2820 |
| 8 | 3.00e-04 | 0.2817 | 0.3256 | **0.7887** | 0.2655 |
| 9 | 3.00e-04 | 0.2723 | 0.3288 | 0.7787 | 0.2994 |
| 10 | 3.00e-04 | 0.2603 | 0.3449 | 0.7759 | 0.3187 |
| 11 | 3.00e-04 | 0.2475 | 0.3708 | 0.7762 | 0.2673 |
| 12 | 3.00e-04 | 0.2339 | 0.3546 | 0.7626 | 0.3092 |
| 13 | 3.00e-04 | 0.2193 | 0.3907 | 0.7550 | 0.2863 |

**结论**：最优 epoch 8，macro-AUC **0.7887**、F1@0.5 **0.2655**（`run_compare` 重算与训练时值一致）。epoch 8 后 valid_loss 由 0.326 升至 0.391、AUC 回落至 0.755，连续 5 轮无提升，`early_stop_patience: 5` 于第 13 轮触发早停，未跑满 25 轮。对比 RUN-001R（0.7903）Δ **−0.0016**，未达增益门槛；`run_compare` 重算的 F1 阈值曲线整体上移（0.2 处 0.4219 vs RUN-001R 0.4012，Δ+0.0207；0.5 处 0.2655 vs 0.2649），但主指标 macro-AUC 未提升 → **25 轮上限 + 早停未带来 AUC 增益，该次实验未证明增加训练轮数有收益**。逐类 AUC 与 RUN-001R 互有升降（class7 +0.0213、class2 −0.0203），总体持平；class 1/3/4 在 0.5 阈值下 F1 为 0（support 24/46/68，稀有类），若以类别 F1 为目标，可另行研究类别阈值；默认提交输出全部类别概率。`num_workers` 由 8 提高到 16 可能改变数据增强随机序列；不能据此保证训练指标不变。

逐类 AUC（`outputs/compare/per_class.csv`，Δ 为四位小数相减）：

| class_id | RUN-001R | RUN-003 | Δ（003−001R） |
|---|---|---|---|
| 1 | 0.7060 | 0.7087 | +0.0027 |
| 2 | 0.8477 | 0.8274 | −0.0203 |
| 3 | 0.7577 | 0.7577 | 0.0000 |
| 4 | 0.7250 | 0.7138 | −0.0112 |
| 5 | 0.8701 | 0.8628 | −0.0073 |
| 6 | 0.7706 | 0.7763 | +0.0057 |
| 7 | 0.7543 | 0.7756 | +0.0213 |
| 8 | 0.7175 | 0.7078 | −0.0097 |
| 9 | 0.8888 | 0.8923 | +0.0035 |
| 10 | 0.8656 | 0.8642 | −0.0014 |

### RUN-003-A run003a_cosine（`configs/exp_cosine.yaml`）

配置 [configs/exp_cosine.yaml](../configs/exp_cosine.yaml)，完整内容：

```yaml
# RUN-003-A 对照配置：在冻结配置上只改学习率调度策略。
#
# 差异只有两项：前 2 轮线性 warmup（沿用 default.yaml 的 warmup_epochs: 2），
# 之后按余弦衰减到 1e-5；其余（25 轮、early-stop patience 5、pos_weight 关闭、
# num_workers 16）全部继承 default.yaml，用来和 RUN-003 做单变量对照。
#
# 用法（在仓库根目录执行）：
#     python src/run_train.py --config configs/exp_cosine.yaml --seed 42 --tag run003a_cosine
_base: default.yaml

train:
  lr_scheduler: cosine
  lr_min: 0.00001
```

| 项 | 值 |
|---|---|
| 相对父配置的差异 | `train.lr_scheduler: warmup → cosine`、新增 `train.lr_min: 0.00001`，其余同 RUN-003 |
| 运行命令 | `python src/run_train.py --config configs/exp_cosine.yaml --seed 42 --tag run003a_cosine` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 42 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 预训练 | 已加载 `resnet18.a1_in1k` |
| 产物 | `outputs/20261028-035123_run003a_cosine/`（远端 `/root/ChestX-MLC`，UTC 03:51:23 / 北京 11:51 启动） |
| 耗时 | 29m47s（UTC 03:51:23 → 04:21:10，北京 11:51 → 12:21）；**10 轮即早停**，单轮约 3.3 分钟 |

metrics.csv（10 轮原样复制，epoch 10 后因连续 5 轮无提升早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.3575 | 0.3437 | 0.7067 | 0.1371 |
| 2 | 3.00e-04 | 0.3311 | 0.3342 | 0.7566 | 0.2194 |
| 3 | 3.00e-04 | 0.3193 | 0.3338 | 0.7562 | 0.2088 |
| 4 | 2.99e-04 | 0.3111 | 0.3219 | 0.7719 | 0.2198 |
| 5 | 2.94e-04 | 0.3038 | 0.3227 | **0.7852** | 0.2855 |
| 6 | 2.87e-04 | 0.2959 | 0.3212 | 0.7788 | 0.2583 |
| 7 | 2.77e-04 | 0.2879 | 0.3222 | 0.7770 | 0.2849 |
| 8 | 2.65e-04 | 0.2789 | 0.3286 | 0.7781 | 0.2768 |
| 9 | 2.50e-04 | 0.2680 | 0.3336 | 0.7773 | 0.2934 |
| 10 | 2.33e-04 | 0.2542 | 0.3407 | 0.7781 | 0.3327 |

**结论**：最优 epoch 5，macro-AUC **0.7852**、F1@0.5 **0.2855**（`run_eval` 在 valid 重算 0.2861）。对比 RUN-003（0.7887，best epoch 8）为 **Δ−0.0035**，未达 +0.005 增益门 → **余弦学习率调度未带来 AUC 增益**；但最佳轮次从 8 前移到 5、早停从 13 轮缩到 10 轮、总耗时从约 38 分钟降到约 30 分钟。valid_loss 在 epoch 4 触底（0.3219）后缓慢回升，epoch 7 起明显过拟合。逐类 AUC 见下表：常见类 9/10/5/8/4 小涨，稀有类 1/2 回落（support 仅 24/37），macro 的下降主要来自这两类。`run_eval --thresholds` 阈值曲线：F1@0.5 0.2861 高于 RUN-003 的 0.2655（Δ+0.0206），但 @0.3 为 0.3481 低于 RUN-003 的 0.3784，峰值仍不在 0.5。

逐类 AUC（`run_eval.py --config configs/default.yaml` 在 valid 上重算，Δ 为四位小数相减）：

| class_id | support | RUN-003 | RUN-003-A | Δ（003A−003） |
|---|---|---|---|---|
| 1 | 24 | 0.7087 | 0.6808 | −0.0279 |
| 2 | 37 | 0.8274 | 0.7811 | −0.0463 |
| 3 | 46 | 0.7577 | 0.7796 | +0.0219 |
| 4 | 68 | 0.7138 | 0.7217 | +0.0079 |
| 5 | 115 | 0.8628 | 0.8691 | +0.0063 |
| 6 | 177 | 0.7763 | 0.7724 | −0.0039 |
| 7 | 193 | 0.7756 | 0.7668 | −0.0088 |
| 8 | 214 | 0.7078 | 0.7162 | +0.0084 |
| 9 | 260 | 0.8923 | 0.8972 | +0.0049 |
| 10 | 306 | 0.8642 | 0.8672 | +0.0030 |

### RUN-003-B run003b_cosine_pos_weight（`configs/exp_run003b.yaml`）

配置 [configs/exp_run003b.yaml](../configs/exp_run003b.yaml)，完整内容：

```yaml
# RUN-003-B：当前默认口径的 ResNet-18 完整复跑（cosine 调度 + pos_weight，seed 42）。
#
# 用途：RUN-003 / RUN-003-A / RUN-002-A 跑在旧口径上（warmup 调度，或 pos_weight
# 关闭），而 RUN-005 / RUN-006 继承的是当前 default.yaml（cosine 调度 +
# pos_weight: true）。为了给骨干对照提供同口径的 ResNet-18 基线，本配置把当前
# 默认口径的关键字段固定下来，单独复跑一次。
# 下面显式写出的字段取值与 default.yaml 当前完全一致；写明是为了让 RUN-003-B 的
# 配置快照以后还能查——default.yaml 之后再变，也不会影响本组实验的口径。
#
# 用法（在仓库根目录执行）：
#     python src/run_train.py --config configs/exp_run003b.yaml --seed 42 --tag run003b_cosine_pos_weight
_base: default.yaml

model:
  backbone: resnet18

train:
  pos_weight: true
  lr_scheduler: cosine
  lr_min: 0.00001

seed: 42
```

| 项 | 值 |
|---|---|
| 相对父配置的差异 | 无配置差异：显式固定的 `backbone: resnet18`、`pos_weight: true`、`lr_scheduler: cosine`、`lr_min: 1e-5`、`seed: 42` 与当前 `default.yaml` 取值一致，其余继承 `default.yaml`（25 轮、早停 5、`num_workers` 16）。本 run 的意义是把当前默认口径冻结成可复现的实验快照，作为 RUN-005 / RUN-006 的同口径基线 |
| 运行命令 | `python src/run_train.py --config configs/exp_run003b.yaml --seed 42 --tag run003b_cosine_pos_weight` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 42 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 预训练 | 已加载 `resnet18.a1_in1k`（checkpoint `metadata.pretrained_loaded = true`） |
| 产物 | `outputs/20261029-041949_run003b_cosine_pos_weight/`（`best.pt` 42.7 MiB / `last.pt` 170.8 MiB / `metrics.csv` / `train.log`） |
| 耗时 | 2026-10-29 12:19:49 → 12:53:33 CST，**13 轮即早停**（未跑满 25 轮），共 **33 分 44 秒**；单轮约 2.6 分钟 |

metrics.csv（13 轮原样复制，epoch 13 后因连续 5 轮无提升早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0321 | 0.9963 | 0.7185 | 0.3531 |
| 2 | 3.00e-04 | 0.9754 | 0.9621 | 0.7534 | 0.3861 |
| 3 | 3.00e-04 | 0.9351 | 0.9647 | 0.7630 | 0.3913 |
| 4 | 2.99e-04 | 0.9035 | 0.9268 | 0.7769 | 0.4010 |
| 5 | 2.94e-04 | 0.8755 | 0.9473 | 0.7776 | 0.3910 |
| 6 | 2.87e-04 | 0.8447 | 0.9285 | 0.7822 | 0.3987 |
| 7 | 2.77e-04 | 0.8054 | 0.9701 | 0.7797 | 0.3967 |
| 8 | 2.65e-04 | 0.7653 | 0.9580 | **0.7837** | 0.4042 |
| 9 | 2.50e-04 | 0.7187 | 1.0101 | 0.7788 | 0.4099 |
| 10 | 2.33e-04 | 0.6767 | 1.0912 | 0.7721 | 0.4009 |
| 11 | 2.15e-04 | 0.6287 | 1.1417 | 0.7704 | 0.4013 |
| 12 | 1.96e-04 | 0.5833 | 1.2110 | 0.7644 | 0.3909 |
| 13 | 1.76e-04 | 0.5460 | 1.3812 | 0.7563 | 0.3994 |

**结论**：最优 epoch 8，macro-AUC **0.7837**、F1@0.5 **0.4040**（`run_compare` 重算；与训练时 `metrics.csv` 的 0.7837 / 0.4042 一致）。epoch 8 后 valid_loss 由 0.96 升至 1.38、AUC 回落，连续 5 轮无提升，`early_stop_patience: 5` 于第 13 轮触发早停，总耗时 33m44s、单轮约 2.6 分钟（比 RUN-003 的 3.3 分钟/轮快，训练时机器空闲）。

ResNet-18 的四个口径组合（seed 42、25 轮上限、早停 5）现在全部有实测：

| 学习率调度 | `pos_weight` | run | macro-AUC | F1@0.5 |
|---|---|---|---|---|
| warmup | 否 | RUN-003 | 0.7887 | 0.2655 |
| cosine | 否 | RUN-003-A | 0.7852 | 0.2855 |
| warmup | 是 | RUN-002-A | 0.7872 | 0.4037 |
| cosine | 是 | RUN-003-B | 0.7837 | 0.4040 |

四组极差 0.0050（RUN-003 与 RUN-003-B）：对比 RUN-003 为 Δ−0.0050、对比 RUN-002-A 为 Δ−0.0035、对比 RUN-003-A 为 Δ−0.0015 → **cosine 与 `pos_weight` 同时开启也没有 AUC 增益**，四组中反而最低；两项的单独结论（RUN-003-A、RUN-002-A）方向一致。F1@0.5 的提升仍来自 `pos_weight` 的概率校准：阈值曲线 @0.2/0.3/0.4/0.5/0.6/0.7 为 0.3657/0.3846/0.4059/0.4040/0.3984/0.3731，峰从 RUN-003 的 0.2 右移到 0.4，但上限 0.4059 低于 RUN-003 @0.2 的 0.4219。逐类 AUC 上，class1/3/5/8 高于 RUN-003，class2/4/6/7/9/10 更低（class4 −0.0412 最明显，support 68）。

本 run 的第二个用途是给骨干对照提供同口径基线：RUN-005（ResNet-50）与 RUN-006（DenseNet-121）同为 cosine + `pos_weight: true` + seed 42，可直接与 RUN-003-B 作单变量比较。同口径下 ResNet-18 的 0.7837 比 ResNet-50 的 0.7857 低 0.0020、比 DenseNet-121 的 0.7880 低 0.0043，详见 `outputs/compare_backbone_s42/summary.md`。

逐类 AUC（`run_compare.py` 在 valid 上重算，Δ 为四位小数相减）：

| class_id | support | RUN-003 | RUN-003-A | RUN-002-A | RUN-003-B | Δ（003B−003） |
|---|---|---|---|---|---|---|
| 1 | 24 | 0.7087 | 0.6808 | 0.7386 | 0.7374 | +0.0287 |
| 2 | 37 | 0.8274 | 0.7811 | 0.8285 | 0.8100 | −0.0174 |
| 3 | 46 | 0.7577 | 0.7796 | 0.7687 | 0.7713 | +0.0136 |
| 4 | 68 | 0.7138 | 0.7217 | 0.6971 | 0.6726 | −0.0412 |
| 5 | 115 | 0.8628 | 0.8691 | 0.8693 | 0.8693 | +0.0065 |
| 6 | 177 | 0.7763 | 0.7724 | 0.7621 | 0.7659 | −0.0104 |
| 7 | 193 | 0.7756 | 0.7668 | 0.7630 | 0.7539 | −0.0217 |
| 8 | 214 | 0.7078 | 0.7162 | 0.6946 | 0.7129 | +0.0051 |
| 9 | 260 | 0.8923 | 0.8972 | 0.8873 | 0.8843 | −0.0080 |
| 10 | 306 | 0.8642 | 0.8672 | 0.8630 | 0.8600 | −0.0042 |

### RUN-002-A pos_weight_s42（`configs/exp_pos_weight.yaml`）

配置 [configs/exp_pos_weight.yaml](../configs/exp_pos_weight.yaml)，完整内容：

```yaml
# 对照实验：按训练集类别频率给正样本加权（pos_weight）。
#
# 用法（在仓库根目录执行）：
#     python src/run_train.py --config configs/exp_pos_weight.yaml --tag pos_weight
#
# 为什么值得试：10 类标签里稀有类别的正样本很少，普通 BCE 容易把它们一直压成
# 低概率；pos_weight 会让稀有类的正样本被错分时付出更大的 loss。
# 是否真的更好要拿 valid 的 macro-AUC 和逐类 F1 说话，见 docs/experiments.md。
_base: default.yaml

train:
  pos_weight: true
```

| 项 | 值 |
|---|---|
| 相对父配置的差异 | `train.pos_weight: false → true`，其余同 RUN-003（完整 25 轮配置） |
| 运行命令 | `python src/run_train.py --config configs/exp_pos_weight.yaml --seed 42 --tag pos_weight_s42` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 42 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 预训练 | 已加载 `resnet18.a1_in1k` |
| 产物 | `outputs/20261028-042115_pos_weight_s42/`（远端 `/root/ChestX-MLC`，UTC 04:21:15 / 北京 12:21 启动） |
| 耗时 | 30m34s（UTC 04:21:15 → 04:51:49，北京 12:21 → 12:51）；**10 轮即早停**，单轮约 3.3 分钟 |

metrics.csv（10 轮原样复制）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0329 | 0.9948 | 0.7200 | 0.3589 |
| 2 | 3.00e-04 | 0.9747 | 0.9570 | 0.7584 | 0.3826 |
| 3 | 3.00e-04 | 0.9334 | 0.9602 | 0.7598 | 0.3892 |
| 4 | 3.00e-04 | 0.9053 | 0.9448 | 0.7724 | 0.3990 |
| 5 | 3.00e-04 | 0.8736 | 0.9140 | **0.7872** | 0.4045 |
| 6 | 3.00e-04 | 0.8452 | 0.9442 | 0.7807 | 0.3990 |
| 7 | 3.00e-04 | 0.8110 | 0.9449 | 0.7820 | 0.4032 |
| 8 | 3.00e-04 | 0.7751 | 0.9399 | 0.7825 | 0.4028 |
| 9 | 3.00e-04 | 0.7350 | 0.9945 | 0.7729 | 0.4055 |
| 10 | 3.00e-04 | 0.6958 | 1.0459 | 0.7714 | 0.4019 |

**结论**：最优 epoch 5，macro-AUC **0.7872**、F1@0.5 **0.4045**。对比 RUN-003（0.7887）为 **Δ−0.0015**，未达 +0.005 增益门 → **完整周期下 pos_weight 对 AUC 仍无增益**（与短跑 RUN-002 的结论一致）。F1@0.5 从 0.2655 升到 0.4045，但 `run_eval` 阈值曲线显示：pos_weight 的曲线峰在 **0.6（0.4182）**，而 RUN-003 在 **0.2（0.4219）**——上限几乎相同，F1@0.5 的提升主要是概率校准让工作点右移，不是判别能力提高。valid_loss 在 epoch 5 之后持续回升（0.914 → 1.046），过拟合比 RUN-003 更早。注意 `train_loss` 量级（约 0.87–1.03）与 RUN-003（约 0.30）不可直接比较，pos_weight 改写了 loss 尺度。

### RUN-002-B pos_weight_s2026（`configs/exp_pos_weight.yaml`）

配置与 RUN-002-A 相同（[configs/exp_pos_weight.yaml](../configs/exp_pos_weight.yaml)），仅随机种子不同：

| 项 | 值 |
|---|---|
| 相对父配置的差异 | 同上；seed 42 → 2026 |
| 运行命令 | `python src/run_train.py --config configs/exp_pos_weight.yaml --seed 2026 --tag pos_weight_s2026` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 2026 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 预训练 | 已加载 `resnet18.a1_in1k` |
| 产物 | `outputs/20261028-045153_pos_weight_s2026/`（远端 `/root/ChestX-MLC`，UTC 04:51:53 / 北京 12:51 启动） |
| 耗时 | 45m58s（UTC 04:51:53 → 05:37:51，北京 12:51 → 13:37）；**12 轮早停**，单轮约 3.3 分钟（最后一轮含验证略长） |

metrics.csv（12 轮原样复制）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0339 | 0.9724 | 0.7448 | 0.3700 |
| 2 | 3.00e-04 | 0.9664 | 0.9460 | 0.7586 | 0.3854 |
| 3 | 3.00e-04 | 0.9314 | 0.9130 | 0.7810 | 0.4049 |
| 4 | 3.00e-04 | 0.9012 | 0.9419 | 0.7691 | 0.3964 |
| 5 | 3.00e-04 | 0.8711 | 0.9261 | 0.7821 | 0.4066 |
| 6 | 3.00e-04 | 0.8384 | 0.9585 | 0.7762 | 0.4026 |
| 7 | 3.00e-04 | 0.8067 | 0.9358 | **0.7871** | 0.4198 |
| 8 | 3.00e-04 | 0.7707 | 0.9970 | 0.7794 | 0.4119 |
| 9 | 3.00e-04 | 0.7262 | 1.0334 | 0.7823 | 0.4017 |
| 10 | 3.00e-04 | 0.6871 | 1.0830 | 0.7612 | 0.3913 |
| 11 | 3.00e-04 | 0.6467 | 1.0999 | 0.7635 | 0.4050 |
| 12 | 3.00e-04 | 0.6127 | 1.2143 | 0.7771 | 0.4131 |

**结论**：最优 epoch 7，macro-AUC **0.7871**、F1@0.5 **0.4198**。与 RUN-002-A（0.7872 / 0.4045）相比，两个 seed 的 AUC 只差 0.0001，说明 pos_weight 的 AUC 结果稳定；F1@0.5 有约 0.015 的 seed 波动。对比 RUN-003 为 Δ−0.0016，同样未达增益门。阈值曲线峰在 0.6（0.4212）与 RUN-003 的 0.2 峰值（0.4219）基本持平，F1 提升同属阈值/校准效应。两 seed 均从 epoch 8 起 valid_loss 加速上升（1.0 → 1.21），早停触发晚于 RUN-002-A。

## ResNet-50 @320（timm `resnet50.a1_in1k`）

### RUN-005 resnet50_s42（`configs/exp_resnet50.yaml`）

配置 [configs/exp_resnet50.yaml](../configs/exp_resnet50.yaml)，完整内容：

```yaml
# 待排期的骨干对照：只切换到 ImageNet 预训练 ResNet-50。
_base: default.yaml

model:
  backbone: resnet50
```

（`timm` 的 `resnet50` 别名默认加载 `resnet50.a1_in1k` 权重；其余超参数全部继承 [configs/default.yaml](../configs/default.yaml)。）

| 项 | 值 |
|---|---|
| 相对父配置的差异 | 仅 `model.backbone: resnet50`；沿用 `default.yaml` 的 320 letterbox、`batch_studies: 16`、`pos_weight: true`、lr 3e-4、前 2 轮 warmup 后余弦衰减至 1e-5、CUDA AMP、`num_workers: 16`、25 轮上限与早停 patience 5 |
| 运行命令 | `python src/run_train.py --config configs/exp_resnet50.yaml --seed 42 --tag resnet50_s42` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 42 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 预训练 | 已加载 `resnet50.a1_in1k`（checkpoint `metadata.pretrained_loaded = true`） |
| 产物 | `outputs/20261029-022055_resnet50_s42/`（`best.pt` 91 MB / `last.pt` 360 MB / `metrics.csv` / `train.log`） |
| 耗时 | 2026-10-29 10:20:50 → 11:04:13 CST，**14 轮即早停**（未跑满 25 轮），共 **43 分 23 秒**；单轮约 3.1 分钟 |

metrics.csv（14 轮原样复制，epoch 14 后因连续 5 轮无提升早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0267 | 0.9767 | 0.7375 | 0.3801 |
| 2 | 3.00e-04 | 0.9617 | 0.9525 | 0.7626 | 0.3892 |
| 3 | 3.00e-04 | 0.9168 | 1.0336 | 0.7702 | 0.4042 |
| 4 | 2.99e-04 | 0.8835 | 0.9695 | 0.7764 | 0.4096 |
| 5 | 2.94e-04 | 0.8485 | 0.9692 | 0.7726 | 0.3990 |
| 6 | 2.87e-04 | 0.8115 | 0.9915 | 0.7775 | 0.4051 |
| 7 | 2.77e-04 | 0.7664 | 1.0351 | 0.7819 | 0.4235 |
| 8 | 2.65e-04 | 0.7116 | 1.0387 | 0.7720 | 0.4087 |
| 9 | 2.50e-04 | 0.6618 | 1.0395 | **0.7857** | 0.4206 |
| 10 | 2.33e-04 | 0.6139 | 1.1785 | 0.7676 | 0.4127 |
| 11 | 2.15e-04 | 0.5687 | 1.3125 | 0.7634 | 0.4067 |
| 12 | 1.96e-04 | 0.5232 | 1.4319 | 0.7724 | 0.4252 |
| 13 | 1.76e-04 | 0.4822 | 1.6130 | 0.7733 | 0.4228 |
| 14 | 1.55e-04 | 0.4460 | 1.6828 | 0.7718 | 0.4117 |

**结论**：最优 epoch 9，macro-AUC **0.7857**、F1@0.5 **0.4193**（`run_compare` 重算；训练时 `metrics.csv` 记录 0.4206，差异来自评估时的批与加载口径）。epoch 9 后 valid_loss 由 1.04 升至 1.68、AUC 回落，连续 5 轮无提升，`early_stop_patience: 5` 于第 14 轮触发早停。对 DenseNet-121（RUN-006，0.7880）AUC 低 0.0023，但在阈值 0.2–0.6 的 F1 曲线多数更高（@0.5 0.4193 vs 0.4070，Δ+0.0123）。与历史 ResNet-18 结果相比（RUN-003 0.7887、RUN-001R 0.7903）略低，但当时为 warmup 调度 + `pos_weight: false`，不能把差异归因于骨干本身。逐类 AUC 见 RUN-006 一节。

## DenseNet-121 @320（timm `densenet121.ra_in1k`）

### RUN-006 densenet121_s42（`configs/exp_densenet121.yaml`）

配置 [configs/exp_densenet121.yaml](../configs/exp_densenet121.yaml)，完整内容：

```yaml
# 待排期的骨干对照：只切换到 ImageNet 预训练 DenseNet-121。
_base: default.yaml

model:
  backbone: densenet121.ra_in1k
```

（其余超参数全部继承 [configs/default.yaml](../configs/default.yaml)，与 RUN-005 相同，仅骨干不同。）

| 项 | 值 |
|---|---|
| 相对父配置的差异 | 仅 `model.backbone: densenet121.ra_in1k`；其余与 RUN-005 完全一致 |
| 运行命令 | `python src/run_train.py --config configs/exp_densenet121.yaml --seed 42 --tag densenet121_s42` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 42 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 预训练 | 已加载 `densenet121.ra_in1k`（checkpoint `metadata.pretrained_loaded = true`） |
| 产物 | `outputs/20261029-030417_densenet121_s42/`（`best.pt` 28 MB / `last.pt` 108 MB / `metrics.csv` / `train.log`） |
| 耗时 | 2026-10-29 11:04:13 → 11:48:45 CST，**13 轮即早停**（未跑满 25 轮），共 **44 分 32 秒**；单轮约 3.4 分钟 |

metrics.csv（13 轮原样复制，epoch 13 后因连续 5 轮无提升早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9951 | 0.9734 | 0.7591 | 0.3857 |
| 2 | 3.00e-04 | 0.9580 | 0.9721 | 0.7595 | 0.3894 |
| 3 | 3.00e-04 | 0.9259 | 0.9305 | 0.7765 | 0.4025 |
| 4 | 2.99e-04 | 0.9048 | 0.9888 | 0.7649 | 0.4005 |
| 5 | 2.94e-04 | 0.8817 | 0.9355 | 0.7742 | 0.3896 |
| 6 | 2.87e-04 | 0.8550 | 0.9243 | 0.7765 | 0.4060 |
| 7 | 2.77e-04 | 0.8287 | 0.9349 | 0.7876 | 0.4169 |
| 8 | 2.65e-04 | 0.8001 | 0.9610 | **0.7880** | 0.4073 |
| 9 | 2.50e-04 | 0.7585 | 0.9873 | 0.7773 | 0.4137 |
| 10 | 2.33e-04 | 0.7030 | 1.0673 | 0.7763 | 0.4239 |
| 11 | 2.15e-04 | 0.6546 | 1.0831 | 0.7831 | 0.4178 |
| 12 | 1.96e-04 | 0.5948 | 1.2888 | 0.7764 | 0.4306 |
| 13 | 1.76e-04 | 0.5419 | 1.3928 | 0.7760 | 0.4199 |

**结论**：最优 epoch 8，macro-AUC **0.7880**、F1@0.5 **0.4070**（`run_compare` 重算；训练时 `metrics.csv` 记录 0.4073）。epoch 8 后 valid_loss 由 0.96 升至 1.39、AUC 回落，连续 5 轮无提升，于第 13 轮早停。对比 RUN-005：AUC 高 0.0023（两组中最高），但 F1@0.5 低 0.0123，阈值 0.2–0.6 的 F1 曲线多数略低（0.2 处 0.3674 vs 0.3808；0.6 处 0.4098 vs 0.4147），仅 0.7 处略高（0.3908 vs 0.3870）。单 seed 下 0.0023 的 AUC 差不足以证明骨干优劣；两组均未超过历史 ResNet-18 的 0.7887 / 0.7903（配置口径不同，见 RUN-005）。

逐类 AUC（`outputs/compare_backbones/per_class.csv`，Δ 为四位小数相减）：

| class_id | RUN-005 ResNet-50 | RUN-006 DenseNet-121 | Δ（006−005） |
|---|---|---|---|
| 1 | 0.7477 | 0.6948 | −0.0529 |
| 2 | 0.8341 | 0.8191 | −0.0151 |
| 3 | 0.7211 | 0.7684 | +0.0474 |
| 4 | 0.7103 | 0.7073 | −0.0031 |
| 5 | 0.8635 | 0.8706 | +0.0071 |
| 6 | 0.7826 | 0.7646 | −0.0180 |
| 7 | 0.7615 | 0.7854 | +0.0239 |
| 8 | 0.6939 | 0.6951 | +0.0012 |
| 9 | 0.8829 | 0.9031 | +0.0202 |
| 10 | 0.8591 | 0.8713 | +0.0122 |

### RUN-006-A densenet121_s2026（`configs/exp_run006a.yaml`）

RUN-006 的 seed 复核：只把随机种子从 42 换成 2026，其余继承 RUN-006。配置
[configs/exp_run006a.yaml](../configs/exp_run006a.yaml)，完整内容：

```yaml
# RUN-006-A：RUN-006（DenseNet-121 骨干对照）的 seed 复现，只把随机种子从 42 换成 2026，
# 其余配置完全继承 RUN-006 的口径（cosine 调度 + pos_weight: true，25 轮，早停 5）。
#
# 用途：RUN-003-B / RUN-005 / RUN-006 的骨干对照只有单 seed，DenseNet-121 比同口径
# ResNet-18 基线高 0.0043，但差距落在历史同配置 seed 波动范围内（RUN-001 与 001R
# 相差 0.0045）。本组用第二个 seed 复核 DenseNet-121 的 macro-AUC 是否稳定，给
# 骨干结论补 seed 稳健性证据；与 RUN-006 的差异只应来自随机种子。
#
# 用法（在仓库根目录执行）：
#     python src/run_train.py --config configs/exp_run006a.yaml --seed 2026 --tag densenet121_s2026
_base: exp_densenet121.yaml

seed: 2026
```

（其余字段全部继承 [configs/exp_densenet121.yaml](../configs/exp_densenet121.yaml)，
即 cosine 调度 + `pos_weight: true` + 25 轮 + 早停 5 + `num_workers` 16。）

上方保留原配置注释；其中 RUN-001 与 RUN-001R 均为 seed 42，0.0045 不是换 seed 的波动。

| 项 | 值 |
|---|---|
| 相对父配置的差异 | 仅顶层 `seed: 2026`；其余与 RUN-006 完全一致 |
| 运行命令 | `python src/run_train.py --config configs/exp_run006a.yaml --seed 2026 --tag densenet121_s2026` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 2026；代码 `a41033d`（含视图元信息支持，`aggregation: mean` 路径与 RUN-006 的 `da8fef7` 数值等价，`tests` 覆盖） |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies |
| 预训练 | 已加载 `densenet121.ra_in1k`（checkpoint `metadata.pretrained_loaded = true`） |
| 产物 | `outputs/20261029-051514_densenet121_s2026/`（`best.pt` / `last.pt` / `metrics.csv` / `train.log`） |
| 耗时 | 2026-10-29 13:15:14 → 13:49:28 CST，**10 轮即早停**（未跑满 25 轮），共 **34 分 14 秒**；与 experiment4 的 EfficientNet-B2 实验并行执行（共用 3090 与 16 核 CPU），耗时不可与 RUN-006 的独占 44m32s 直接比较 |

metrics.csv（10 轮原样复制，epoch 10 后因连续 5 轮无提升早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9992 | 0.9595 | 0.7561 | 0.3769 |
| 2 | 3.00e-04 | 0.9616 | 0.9481 | 0.7664 | 0.3990 |
| 3 | 3.00e-04 | 0.9237 | 0.9369 | 0.7764 | 0.4023 |
| 4 | 2.99e-04 | 0.9032 | 0.9235 | 0.7851 | 0.4079 |
| 5 | 2.94e-04 | 0.8816 | 0.9046 | **0.7913** | 0.4102 |
| 6 | 2.87e-04 | 0.8608 | 0.9200 | 0.7881 | 0.3936 |
| 7 | 2.77e-04 | 0.8304 | 0.9170 | 0.7911 | 0.4139 |
| 8 | 2.65e-04 | 0.7981 | 0.9623 | 0.7903 | 0.4305 |
| 9 | 2.50e-04 | 0.7512 | 0.9830 | 0.7863 | 0.4212 |
| 10 | 2.33e-04 | 0.7073 | 1.0248 | 0.7842 | 0.4168 |

**结论**：最优 epoch 5，macro-AUC **0.7913**、F1@0.5 **0.4103**（`run_compare`
重算；`metrics.csv` 训练时记录 F1 0.4102，AUC 一致）。epoch 5 后 valid_loss 由 0.90
回升至 1.02、AUC 回落，连续 5 轮无提升，于第 10 轮早停。对比 RUN-006（seed 42）：
AUC 高 **+0.0033**（0.7913 vs 0.7880），F1@0.5 高 +0.0033（0.4103 vs 0.4070）；
最佳轮次由 8 前移到 5，早停由 13 轮提前到 10 轮。阈值曲线上 RUN-006-A 峰在 0.5
（0.4103）、RUN-006 峰在 0.6（0.4098），曲线主体接近但 RUN-006-A 在 0.2–0.4 段略低
（0.2 处 0.3614 vs 0.3674）。与同口径 ResNet-18 基线 RUN-003-B（0.7837）相比，
RUN-006-A 高 **+0.0076**；DenseNet-121 的两个 seed（0.7880 / 0.7913）都高于基线，
其中只有 seed 42 的差值是配对骨干对照；seed 2026 缺少 ResNet-18 对照，
不能认定骨干优势在第二个 seed 复现。两次 DenseNet 训练的代码版本与负载也不同，
不能将 +0.0033 全部归因于 seed；**仍不足以判定骨干优劣**。

逐类 AUC（`outputs/compare_run006a/per_class.csv`，Δ 为 RUN-006-A − RUN-006）：

| class_id | RUN-006 DenseNet-121 s42 | RUN-006-A DenseNet-121 s2026 | Δ（006-A − 006） |
|---|---|---|---|
| 1 | 0.6948 | 0.7405 | +0.0457 |
| 2 | 0.8191 | 0.8212 | +0.0022 |
| 3 | 0.7684 | 0.7929 | +0.0245 |
| 4 | 0.7073 | 0.7023 | −0.0050 |
| 5 | 0.8706 | 0.8640 | −0.0067 |
| 6 | 0.7646 | 0.7684 | +0.0038 |
| 7 | 0.7854 | 0.7637 | −0.0216 |
| 8 | 0.6951 | 0.7070 | +0.0119 |
| 9 | 0.9031 | 0.8894 | −0.0137 |
| 10 | 0.8713 | 0.8631 | −0.0082 |

### RUN-007 densenet121_viewweighted_s42（`configs/exp_run007.yaml`）

配置 [configs/exp_run007.yaml](../configs/exp_run007.yaml)，完整内容：

```yaml
# RUN-007：在 RUN-006 的 DenseNet-121 配置上比较正侧位 7:3 加权聚合。
_base: exp_densenet121.yaml

model:
  aggregation: view_weighted
  frontal_weight: 0.7
```

（其余字段全部继承 [configs/exp_densenet121.yaml](../configs/exp_densenet121.yaml)，
即 320×320 letterbox、`pos_weight: true`、学习率 3e-4、前 2 轮 warmup 后余弦衰减到
1e-5、CUDA AMP、25 轮上限、早停 5、`num_workers` 16。）

| 项 | 值 |
|---|---|
| 相对父配置的差异 | 仅 `model.aggregation: mean → view_weighted`、新增 `model.frontal_weight: 0.7`；其余与 RUN-006 完全一致（`load_config` 断言除聚合方式外零差异） |
| 运行命令 | `python src/run_train.py --config configs/exp_run007.yaml --seed 42 --tag densenet121_viewweighted_s42` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 42；代码 `53568fe`（RUN-006 为 `da8fef7`，`62b6464` 起增加视角元信息贯通与 `view_weighted` 聚合；`mean` 路径由 `tests` 与 RUN-006-A 在 `a41033d` 上的实测确认数值等价，但两组并非同代码版本的严格单变量对照） |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 RUN-006 同一份 split；`check_data.py --full` 全量通过，跨 split 无 Subject / Study / Dicom 重叠） |
| 预训练 | 已加载 `densenet121.ra_in1k`（checkpoint `metadata.pretrained_loaded = true`） |
| 产物 | `outputs/20261029-062002_densenet121_viewweighted_s42/`（`best.pt` 28 MB / `last.pt` 108 MB / `metrics.csv` / `train.log`） |
| 耗时 | 2026-10-29 14:20:02 → 14:54:58 CST，**13 轮即早停**（未跑满 25 轮），共 **34 分 56 秒**；单轮约 2.7 分钟（独占运行，GPU 利用率约 46–48%） |

metrics.csv（13 轮原样复制，epoch 13 后因连续 5 轮无提升早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9994 | 0.9412 | 0.7659 | 0.3936 |
| 2 | 3.00e-04 | 0.9584 | 0.9584 | 0.7702 | 0.4053 |
| 3 | 3.00e-04 | 0.9245 | 0.9310 | 0.7755 | 0.4006 |
| 4 | 2.99e-04 | 0.9037 | 0.9465 | 0.7781 | 0.4025 |
| 5 | 2.94e-04 | 0.8809 | 0.9068 | 0.7834 | 0.4073 |
| 6 | 2.87e-04 | 0.8550 | 0.9268 | 0.7783 | 0.4082 |
| 7 | 2.77e-04 | 0.8357 | 0.9464 | 0.7854 | 0.4181 |
| 8 | 2.65e-04 | 0.8053 | 0.9155 | **0.7892** | 0.4120 |
| 9 | 2.50e-04 | 0.7669 | 0.9551 | 0.7856 | 0.4223 |
| 10 | 2.33e-04 | 0.7175 | 1.0233 | 0.7868 | 0.4256 |
| 11 | 2.15e-04 | 0.6636 | 1.0940 | 0.7768 | 0.4169 |
| 12 | 1.96e-04 | 0.6078 | 1.3703 | 0.7689 | 0.4207 |
| 13 | 1.76e-04 | 0.5519 | 1.4413 | 0.7665 | 0.4194 |

**结论**：最优 epoch 8，macro-AUC **0.7892**、F1@0.5 **0.4122**（`run_compare` 重算；
`metrics.csv` 训练时记录 AUC 0.7892 / F1 0.4120，AUC 一致）。epoch 8 后 valid_loss 由
0.92 升至 1.44、AUC 回落，连续 5 轮无提升，于第 13 轮早停。对比 RUN-006（等权均值，
0.7880 / 0.4070）：AUC 高 **+0.0012**、F1@0.5 高 +0.0052。AUC 差未达历史判读门的
+0.005，也小于 DenseNet-121 两个 seed 的差值（RUN-006 与 RUN-006-A 差 0.0033）→
**单 seed 下不能判定正侧位加权聚合有效**。F1@0.5 的提升伴随阈值曲线整体右移：0.2–0.4
段反而更低（0.3596 / 0.3785 / 0.3972，RUN-006 为 0.3674 / 0.3882 / 0.4069），峰由
RUN-006 的 0.6（0.4098）移到 0.5（0.4122），上限基本持平，属校准/阈值形态而非判别力
提升。逐类 AUC 6 升 3 降 1 平（见下表），class2 −0.0428（support 37）最明显、class3
+0.0169，方向不一致。RUN-006 与 RUN-007 的代码版本不同（`da8fef7` → `53568fe`，新增
视角元信息贯通），不能把差异全部归因于聚合方式；若要继续，需按
[train_plan06.md](train_plan06.md) 补 seed 2026 复核（RUN-006-A 已有同 seed 的等权
对照可配对）。**默认 `model.aggregation` 维持 `mean`。**

逐类 AUC（`outputs/compare_run007/per_class.csv`，Δ 为四位小数相减）：

| class_id | support | RUN-006 等权 | RUN-007 加权 7:3 | Δ（007−006） |
|---|---|---|---|---|
| 1 | 24 | 0.6948 | 0.7044 | +0.0096 |
| 2 | 37 | 0.8191 | 0.7763 | −0.0428 |
| 3 | 46 | 0.7684 | 0.7853 | +0.0169 |
| 4 | 68 | 0.7073 | 0.7187 | +0.0114 |
| 5 | 115 | 0.8706 | 0.8725 | +0.0019 |
| 6 | 177 | 0.7646 | 0.7793 | +0.0147 |
| 7 | 193 | 0.7854 | 0.7740 | −0.0114 |
| 8 | 214 | 0.6951 | 0.7069 | +0.0118 |
| 9 | 260 | 0.9031 | 0.9002 | −0.0029 |
| 10 | 306 | 0.8713 | 0.8742 | +0.0029 |

### RUN-007-A densenet121_viewweighted_s2026（`configs/exp_run007a.yaml`）

RUN-007 的 seed 复核：只把随机种子从 42 换成 2026，其余继承 RUN-007。配置
[configs/exp_run007a.yaml](../configs/exp_run007a.yaml)，完整内容：

```yaml
# RUN-007-A：RUN-007（DenseNet-121 + 正侧位 7:3 加权聚合）的 seed 复核，
# 只把随机种子从 42 换成 2026，其余配置完全继承 RUN-007 的口径。
#
# 用途：RUN-007 单 seed 下正侧位加权聚合的 macro-AUC 比 RUN-006 等权高 0.0012，
# 未达 +0.005 判读门，也小于 DenseNet-121 的 seed 波动（RUN-006 与 RUN-006-A
# 相差 0.0033）。本组用第二个 seed 复核加权聚合的方向是否稳定；与 RUN-006-A
# （同 seed 的等权对照）可直接配对比较，差异只应来自聚合方式。
#
# 用法（在仓库根目录执行）：
#     python src/run_train.py --config configs/exp_run007a.yaml --seed 2026 --tag densenet121_viewweighted_s2026
_base: exp_run007.yaml

seed: 2026
```

（其余字段全部继承 [configs/exp_run007.yaml](../configs/exp_run007.yaml)，即再继承
`exp_densenet121.yaml` 的全部口径：320×320 letterbox、`pos_weight: true`、warmup 后
余弦衰减、CUDA AMP、25 轮上限、早停 5、`num_workers` 16；`tests/test_config.py` 的
`test_run007a_experiment_config_loads` 断言与 RUN-007 除 seed 外零差异。）

| 项 | 值 |
|---|---|
| 相对父配置的差异 | 仅顶层 `seed: 42 → 2026`；其余与 RUN-007（含加权聚合与正位权重 0.7）完全一致 |
| 运行命令 | `python src/run_train.py --config configs/exp_run007a.yaml --seed 2026 --tag densenet121_viewweighted_s2026` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 2026；代码 `27dbe56`（在 `53568fe` 基础上仅新增本配置、测试与计划；`src/` 未变） |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 RUN-006-A 同一份 split；`check_data.py --full` 全量通过） |
| 预训练 | 已加载 `densenet121.ra_in1k`（checkpoint `metadata.pretrained_loaded = true`） |
| 产物 | `outputs/20261029-070733_densenet121_viewweighted_s2026/`（`best.pt` / `last.pt` / `metrics.csv` / `train.log`） |
| 耗时 | 2026-10-29 15:07:33 → 15:42:34 CST，**12 轮即早停**（未跑满 25 轮），共 **35 分 01 秒**；单轮约 2.9 分钟（独占运行） |

metrics.csv（12 轮原样复制，epoch 12 后因连续 5 轮无提升早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9966 | 0.9555 | 0.7636 | 0.3866 |
| 2 | 3.00e-04 | 0.9553 | 0.9612 | 0.7690 | 0.4022 |
| 3 | 3.00e-04 | 0.9228 | 0.9498 | 0.7778 | 0.4140 |
| 4 | 2.99e-04 | 0.9007 | 0.9292 | 0.7836 | 0.4153 |
| 5 | 2.94e-04 | 0.8791 | 0.9046 | 0.7869 | 0.4032 |
| 6 | 2.87e-04 | 0.8615 | 0.9325 | 0.7751 | 0.3911 |
| 7 | 2.77e-04 | 0.8306 | 0.9143 | **0.7906** | 0.4194 |
| 8 | 2.65e-04 | 0.8031 | 0.9605 | 0.7842 | 0.4203 |
| 9 | 2.50e-04 | 0.7579 | 0.9600 | 0.7845 | 0.4261 |
| 10 | 2.33e-04 | 0.7143 | 1.0379 | 0.7748 | 0.4178 |
| 11 | 2.15e-04 | 0.6664 | 1.0880 | 0.7807 | 0.4157 |
| 12 | 1.96e-04 | 0.6056 | 1.2269 | 0.7708 | 0.4249 |

**结论**：最优 epoch 7，macro-AUC **0.7906**、F1@0.5 **0.4198**（`run_compare`
重算；`metrics.csv` 训练时记录 F1 0.4194，AUC 一致）。epoch 7 后 valid_loss 由 0.91
升至 1.23、AUC 回落，连续 5 轮无提升，于第 12 轮早停。与同 seed 的等权对照
RUN-006-A（0.7913 / 0.4103）配对比较：**AUC Δ−0.0007**、F1@0.5 +0.0095。

两个 seed 的「加权−等权」配对差方向相反：

| seed | 等权对照 | 加权 | Δ（加权−等权） |
|---|---|---|---|
| 42 | RUN-006 0.7880 | RUN-007 0.7892 | **+0.0012** |
| 2026 | RUN-006-A 0.7913 | RUN-007-A 0.7906 | **−0.0007** |

两 seed 配对差的均值为 +0.00025，绝对值均远低于 +0.005 判读门，且方向不一致 →
**加权正侧位 7:3 聚合对 macro-AUC 无稳定增益，判定无效，默认 `model.aggregation`
维持 `mean`。** F1 表现为：seed 42 仅 0.5–0.7 段提升（0.2–0.4 段下降），seed 2026
则全线提升 +0.0067~+0.0196（0.2–0.7），两 seed 形态不同，说明 F1 变化同样不稳定，
不能作为判别力提升的证据。逐类 AUC 上 seed 2026 呈现「常见类 5–10 升、稀有类 1–4
降」（class9 +0.0145、class10 +0.0147、class3 −0.0279），与 seed 42 的方向也不完全
一致。

逐类 AUC（`outputs/compare_run007a/per_class.csv`，Δ 为四位小数相减）：

| class_id | support | RUN-006-A 等权 | RUN-007-A 加权 7:3 | Δ（007A−006A） |
|---|---|---|---|---|
| 1 | 24 | 0.7405 | 0.7208 | −0.0197 |
| 2 | 37 | 0.8212 | 0.7944 | −0.0268 |
| 3 | 46 | 0.7929 | 0.7650 | −0.0279 |
| 4 | 68 | 0.7023 | 0.6975 | −0.0048 |
| 5 | 115 | 0.8640 | 0.8752 | +0.0112 |
| 6 | 177 | 0.7684 | 0.7794 | +0.0110 |
| 7 | 193 | 0.7637 | 0.7718 | +0.0081 |
| 8 | 214 | 0.7070 | 0.7205 | +0.0135 |
| 9 | 260 | 0.8894 | 0.9039 | +0.0145 |
| 10 | 306 | 0.8631 | 0.8778 | +0.0147 |

## 汇总

| run | 配置 | 关键差异 | epochs | best_epoch | macro-AUC | F1@0.5 | 耗时 | 结论 |
|---|---|---|---|---|---|---|---|---|
| RUN-001 | `exp_short.yaml` | 短周期基准（旧代码 59a7d15） | 6 | 6 | 0.7858 | 0.2796 | 28m41s | 末轮仍改善，仅作方向基准 |
| RUN-001R | `exp_short.yaml` | 新代码回归（e8f194e） | 6 | 6 | 0.7903 | 0.2649 | 38m05s（并行） | 回归门通过（Δ+0.0045，在 0.7808~0.7908 内） |
| RUN-002 | `exp_short_pos_weight.yaml` | +pos_weight | 6 | 5 | 0.7882 | 0.4048 | 33m06s（并行） | AUC Δ−0.0021 未达门，pos_weight 无效 |
| RUN-003 | `default.yaml` | 25 轮冻结版（`num_workers` 16） | 13（早停）/25 | 8 | 0.7887 | 0.2655 | 38m | AUC Δ−0.0016 未达增益门，该次实验未证明增加训练轮数有收益；提交文件 `outputs/submission_run003.csv` |
| RUN-003-A | `exp_cosine.yaml` | +余弦调度（warmup 后衰减到 1e-5） | 10（早停）/25 | 5 | 0.7852 | 0.2855 | 29m47s | AUC Δ−0.0035 未达门，但最佳轮次前移、耗时缩短；调度不带来 AUC 增益 |
| RUN-003-B | `exp_run003b.yaml` | 当前默认口径（+cosine +pos_weight），seed 42 | 13（早停）/25 | 8 | 0.7837 | 0.4040 | 33m44s | 同口径 ResNet-18 基线；四组 ResNet-18 口径中 macro-AUC 最低（vs RUN-003 Δ−0.0050），cosine + pos_weight 组合无增益；供 RUN-005 / RUN-006 单变量对照 |
| RUN-002-A | `exp_pos_weight.yaml` | +pos_weight，seed 42 | 10（早停）/25 | 5 | 0.7872 | 0.4045 | 30m34s | AUC Δ−0.0015 未达门；F1@0.5 大涨但阈值峰右移到 0.6，上限与基线的 @0.2 持平 |
| RUN-002-B | `exp_pos_weight.yaml` | +pos_weight，seed 2026 | 12（早停）/25 | 7 | 0.7871 | 0.4198 | 45m58s | 与 002-A 的 AUC 仅差 0.0001，两次结果接近；未达原 AUC 增益门 |
| RUN-005 | `exp_resnet50.yaml` | ResNet-50 + a1_in1k，+cosine +pos_weight | 14（早停）/25 | 9 | 0.7857 | 0.4193 | 43m23s | 同口径下 AUC 高于 RUN-003-B（Δ+0.0020）、F1@0.5 高 0.0153；低于 DenseNet-121（Δ−0.0023） |
| RUN-006 | `exp_densenet121.yaml` | DenseNet-121 + ra_in1k，+cosine +pos_weight | 13（早停）/25 | 8 | 0.7880 | 0.4070 | 44m32s | 同口径下 macro-AUC 最高（vs RUN-003-B Δ+0.0043），F1@0.5 仅高 0.0030；单 seed 差不足以判定骨干优劣 |
| RUN-006-A | `exp_run006a.yaml` | DenseNet-121 + ra_in1k，+cosine +pos_weight，seed 2026 | 10（早停）/25 | 5 | 0.7913 | 0.4103 | 34m14s（并行） | seed 复核：AUC 比 RUN-006 高 +0.0033；缺少 seed 2026 的 ResNet-18 配对对照，尚不足以判定骨干优劣 |
| RUN-007 | `exp_run007.yaml` | DenseNet-121 + 正侧位 7:3 加权聚合，seed 42 | 13（早停）/25 | 8 | 0.7892 | 0.4122 | 34m56s | 对比 RUN-006（等权）AUC Δ+0.0012，未达 +0.005 门且小于 seed 波动（0.0033）；F1@0.5 提升伴随阈值曲线右移 → 暂不改默认聚合 |
| RUN-007-A | `exp_run007a.yaml` | DenseNet-121 + 正侧位 7:3 加权聚合，seed 2026 | 12（早停）/25 | 7 | 0.7906 | 0.4198 | 35m01s | 与同 seed 等权 RUN-006-A 配对 Δ−0.0007，与 seed 42 的 +0.0012 方向相反、均在 seed 波动内 → 加权聚合无稳定增益，默认 `mean` 不变 |

注：RUN-001R / 002 的 macro-AUC 与 F1@0.5 取 `outputs/compare/summary.md` 的重算值（与 `metrics.csv` 的差异 ≤0.0003，见各 run 的「口径说明」）。
注：RUN-003-B 与 RUN-003 / 003-A / 002-A 的 macro-AUC 与 F1@0.5 取 `outputs/compare_run003b/summary.md` 的重算值（RUN-003-B 与训练时 `metrics.csv` 的差异 ≤0.0002）。
注：RUN-005 / RUN-006 的 macro-AUC 与 F1@0.5 取 `outputs/compare_backbones/summary.md` 的重算值（与 `metrics.csv` 的差异 ≤0.0013）；两组与 RUN-003-B 同为 cosine 调度 + `pos_weight: true` + seed 42，可与 RUN-003-B 直接作单变量骨干对照（`outputs/compare_backbone_s42/summary.md`），与 RUN-003 系列的 warmup / `pos_weight: false` 口径不同，历史数字只作参考。
注：RUN-001R 与 RUN-002 并行执行（两路各 8 workers，16 核饱和），单轮耗时不可与 RUN-001 的串行口径直接对照；主进程负责 shuffle，但 `num_workers` 等运行差异可能影响训练结果。
注：RUN-003 为独占串行运行（16 workers 独占 16 核），单轮约 3.3 分钟，可与 RUN-001 的串行口径对照；13 轮总耗时约 38 分钟（含启动与验证），未跑满 25 轮是因早停，不是被中断。
注：RUN-003-B 同为独占串行运行，单轮约 2.6 分钟、13 轮共 33m44s；与 RUN-003 的差异主要来自机器负载（训练时 GPU 空闲），不影响 AUC 口径。

注：RUN-006-A 的 macro-AUC 与 F1@0.5 取 `outputs/compare_run006a/summary.md` 的重算值（`metrics.csv` 的 F1 差 0.0001、AUC 一致）；与 RUN-006 的配置仅 seed 不同，但代码版本和运行负载不同，不能将结果差异完全归因于 seed。RUN-006-A 与 experiment4 的 EfficientNet-B2 实验并行执行（共用 3090 与 16 核 CPU），单轮耗时不可与 RUN-006 的独占 44m32s 对照。
注：RUN-007 的 macro-AUC 与 F1@0.5 取 `outputs/compare_run007/summary.md` 的重算值（与 `metrics.csv` 的 AUC 一致、F1 差 0.0002）；与 RUN-006 仅差 `model.aggregation`，但代码版本不同（`53568fe` vs `da8fef7`，后者无视角元信息），且 0.0012 的 AUC 差小于 DenseNet-121 的 seed 波动 0.0033，不能将差异全部归因于聚合方式。
注：RUN-007-A 的 macro-AUC 与 F1@0.5 取 `outputs/compare_run007a/summary.md` 的重算值（与 `metrics.csv` 的 AUC 一致、F1 差 0.0004）；与 RUN-006-A 的配置仅差聚合方式，但代码版本不同（`27dbe56` vs `a41033d`，`src/` 等价），单 seed 配对差 −0.0007 与 seed 42 的 +0.0012 方向相反，两 seed 配对差绝对值均小于 +0.005 判读门 → 加权聚合判定无效。

判读口径：先看 macro-AUC（不受阈值影响）；再看 `f1@0.5` 与整条阈值曲线（`run_compare` 的 `summary.md`）；`pos_weight` 只有在多数类同时不退化时才算有效。

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 3090 24G（驱动 535.247.01，CUDA 12.2） |
| Python / 框架 | Python 3.12、torch 2.10.0+cu128、timm 1.0.30、torchvision 0.25.0+cu128 |
| 数据根目录 | `/root/sj-tmp`（`data/` 下以软链接挂入，数据本身不入库） |
| 随机种子 | 42（RUN-002-B / RUN-006-A 为 2026，其余 run 均为 42） |
