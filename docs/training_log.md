# ChestX-MLC 训练记录

> 本文保留训练环境当时的产物路径；本机整理后的分类及路径对应说明见 [架构与文件结构](architecture.md#21-本机归档的输出目录)。

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

### RUN-007-B densenet121_viewweighted_s2026_384（`configs/exp_run007b.yaml`）

RUN-007-A 的分辨率探针：只把输入尺寸从 320 提到 384、训练轮数从 25 缩到 6，
其余继承 RUN-007-A，用于看更高分辨率在 6 轮短跑里的趋势。配置
[configs/exp_run007b.yaml](../configs/exp_run007b.yaml)，完整内容：

```yaml
# RUN-007-B：在 RUN-007-A 的基础上只改两处——输入分辨率 320→384、训练 25→6 轮，
# 其余口径（DenseNet-121、正侧位 7:3 加权聚合、pos_weight、warmup + cosine、
# seed 2026、batch_studies 16）完全继承 RUN-007-A。
#
# 用途：RUN-007 / RUN-007-A 已在 320×320 下判定「正侧位 7:3 加权聚合」无稳定
# 增益；本组把输入分辨率提到 384×384，用 6 轮短跑看更高分辨率能否带来更优的
# valid macro-AUC 趋势。6 轮会把 cosine 周期压缩到 6 轮（warmup 2 轮占 1/3），
# 指标不能与 RUN-007-A 的 12 轮早停结果直接比较，只用于判断方向。
#
# 用法（在仓库根目录执行）：
#     python src/run_train.py --config configs/exp_run007b.yaml --tag densenet121_viewweighted_s2026_384
_base: exp_run007a.yaml

image:
  size: 384

train:
  epochs: 6
```

（其余字段全部继承 [configs/exp_run007a.yaml](../configs/exp_run007a.yaml)，即再继承
`exp_run007.yaml` / `exp_densenet121.yaml`：正侧位 7:3 加权聚合、`pos_weight: true`、
warmup 2 轮后余弦衰减、CUDA AMP、早停 5、`num_workers` 16、seed 2026；
`tests/test_config.py` 的 `test_run007b_experiment_config_loads` 断言与 RUN-007-A
除 `image.size` 与 `train.epochs` 外零差异。）

| 项 | 值 |
|---|---|
| 相对父配置的差异 | `image.size: 320 → 384`、`train.epochs: 25 → 6`；其余与 RUN-007-A 完全一致 |
| 运行命令 | `python src/run_train.py --config configs/exp_run007b.yaml --tag densenet121_viewweighted_s2026_384` |
| 设备 / 环境 | RTX 3090 24G / torch 2.10.0+cu128 / timm 1.0.30 / seed 2026；代码 `bfa6d45`（在 `27dbe56` 基础上仅新增 RUN-008 评估工具，训练路径未变） |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 RUN-007-A 同一份 split） |
| 预训练 | 已加载 `densenet121.ra_in1k`（checkpoint `metadata.pretrained_loaded = true`） |
| 产物 | `outputs/20261029-123158_densenet121_viewweighted_s2026_384/`（`best.pt` / `last.pt` / `metrics.csv` / `train.log`） |
| 耗时 | 2026-10-29 20:31:58 → 20:55:40 CST，6 轮跑满（未早停），共 **23 分 42 秒**；单轮约 3.9 分钟；显存峰值约 10.2–10.5 GB（独占运行） |

metrics.csv（6 轮原样复制）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9999 | 0.9617 | 0.7594 | 0.3824 |
| 2 | 3.00e-04 | 0.9600 | 0.9802 | 0.7529 | 0.3786 |
| 3 | 3.00e-04 | 0.9224 | 0.9313 | 0.7729 | 0.4067 |
| 4 | 2.27e-04 | 0.8845 | 0.9252 | 0.7868 | 0.4120 |
| 5 | 8.25e-05 | 0.8139 | 0.9059 | **0.7959** | 0.4227 |
| 6 | 1.00e-05 | 0.7456 | 0.9261 | 0.7945 | 0.4263 |

**结论**：最优 epoch 5，macro-AUC **0.7959**、F1@0.5 **0.4227**（`run_compare`
重算，与 `metrics.csv` 一致）；epoch 6 学习率衰减到 1e-5 后 AUC 基本走平。与
RUN-007-A（320×320、12 轮早停，0.7906 / 0.4198）相比 AUC **+0.0053**、F1@0.5
+0.0029，但两者轮数与 cosine 周期不同（007B 的 6 轮周期让第 5 轮 lr 已降到
8.25e-5、模型提前收敛；007A 第 6 轮 lr 仍有 2.87e-4），**不能把差异单变量归因于
分辨率**。仅作方向筛查：384 + 6 轮短跑在 6 轮内即达 0.7959、未见分辨率提升带来
的退化，值得后续用 25 轮完整对照（同 seed、同调度与早停口径）再判定。

逐类 AUC（`outputs/compare_run007b/per_class.csv`，Δ 为四位小数相减）：

| class_id | support | RUN-007-A 320 | RUN-007-B 384 | Δ（007B−007A） |
|---|---|---|---|---|
| 1 | 24 | 0.7208 | 0.7188 | −0.0020 |
| 2 | 37 | 0.7944 | 0.8339 | +0.0395 |
| 3 | 46 | 0.7650 | 0.7642 | −0.0008 |
| 4 | 68 | 0.6975 | 0.7146 | +0.0171 |
| 5 | 115 | 0.8752 | 0.8694 | −0.0058 |
| 6 | 177 | 0.7794 | 0.7737 | −0.0057 |
| 7 | 193 | 0.7718 | 0.7778 | +0.0060 |
| 8 | 214 | 0.7205 | 0.7259 | +0.0054 |
| 9 | 260 | 0.9039 | 0.9027 | −0.0012 |
| 10 | 306 | 0.8778 | 0.8781 | +0.0003 |

差异主要来自稀有类 2（+0.0395）与 4（+0.0171），常见类 5/6/9 略降；6 轮短跑下
单类波动大，逐类方向不足以作为分辨率结论。


## RUN-008 骨干对照 @320（EfficientNet-B2 / EfficientNetV2-S / ConvNeXt-Tiny / ConvNeXtV2-Nano）

按 [train_plan08.md](train_plan08.md) 执行：继承 RUN-007-A 的正侧位 7:3 加权聚合口径
（`configs/exp_run007a.yaml`），四组只覆盖 `model.backbone`，并把训练预算统一改为
**固定 6 轮、关闭早停**。seed 2026、320×320 letterbox、ImageNet 归一化、`batch_studies: 16`、
`pos_weight: true`、warmup 2 轮后 cosine 衰减到 1e-5、CUDA AMP、`num_workers: 16`。
四组按 A+B、C+D 两两并行执行，共享 GPU 使训练耗时不可横向比较；RUN-007-A 是 25 轮上限、
第 12 轮早停（峰值 epoch 7），训练预算与本计划不同，口径说明见本节末尾。

四组权重来源与许可（实际下载文件经 hf-mirror 镜像，SHA-256 与 HF 仓库 commit 记录如下）：

| 组 | timm 标识 | HF 仓库 / commit | 权重文件字节 | SHA-256 | 模型卡许可 |
|---|---|---|---|---|---|
| RUN008-A | `efficientnet_b2.ra_in1k` | `timm/efficientnet_b2.ra_in1k` / `3577c4a7d84723645311bb5a9e5086f1b62ec8e2` | 36,757,206 | `e9adbcce7e5d5055c571c4cafdcc7f920b6a6ec42e643c49dff1aabe1d5f53c5` | Apache-2.0 |
| RUN008-B | `efficientnetv2_rw_s.ra2_in1k` | `timm/efficientnetv2_rw_s.ra2_in1k` / `ee38fe1c4e8ca084db9b6ec22c3d6a7d02990704` | 96,483,448 | `81379566b6b222f182fa010a993ce396e97b08568bb4f87241b61b064294a65e` | Apache-2.0 |
| RUN008-C | `convnext_tiny.fb_in1k` | `timm/convnext_tiny.fb_in1k` / `b43a6303c9fcf176d2d707478a128c2c91e93528` | 114,374,272 | `08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276` | Apache-2.0 |
| RUN008-D | `convnextv2_nano.fcmae_ft_in1k` | `timm/convnextv2_nano.fcmae_ft_in1k` / `fd83d9c986153f1bac0fc78368a17a66eb9c584e` | 62,511,014 | `c9b5106b30a4593916f1ad5a3fa4f659d48e8ce0107bb9901c457a3686d27a44` | **CC BY-NC 4.0** |

赛题《比赛参赛规则》第 1 条明确不接受 NC 类许可证，RUN008-D 仅作研究记录、不用于最终提交。
四组使用同一份 train / valid 数据（train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies），
启动前 `check_data.py --full` 全部通过。

公共口径（四组相同）：

| 项 | 值 |
|---|---|
| 设备 / 环境 | RTX 3090 24G（驱动 535.247.01）/ torch 2.10.0+cu128 / timm 1.0.30 / Python 3.12.15 / seed 2026 |
| 代码版本 | `b516936`（本计划只新增配置、测试与文档；`src/` 与 RUN-007-A 的 `27dbe56` 在训练路径上无改动） |
| 预训练 | 四组均加载成功（`train.log` 打印「预训练权重: 已加载」，checkpoint `metadata.pretrained_loaded = true`） |
| 训练预算 | 固定 6 轮、`early_stop_patience: 0`（无早停）；A+B 并行、C+D 并行 |
| 推理尺寸 / 聚合 | 320×320 letterbox、ImageNet 归一化、`view_weighted` 7:3 |

### RUN008-A run008a_effb2_s2026（`configs/exp_run008a.yaml`）

配置 [configs/exp_run008a.yaml](../configs/exp_run008a.yaml)，完整内容：

```yaml
_base: exp_run007a.yaml

model:
  backbone: efficientnet_b2.ra_in1k

train:
  epochs: 6
  early_stop_patience: 0
```

| 项 | 值 |
|---|---|
| 相对父配置的差异 | `model.backbone: densenet121.ra_in1k → efficientnet_b2.ra_in1k`、`train.epochs: 25 → 6`、`train.early_stop_patience: 5 → 0`；其余继承 RUN-007-A |
| 运行命令 | `python src/run_train.py --config configs/exp_run008a.yaml --seed 2026 --tag run008a_effb2_s2026` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 RUN-007-A 同一份） |
| 预训练 | 已加载 `efficientnet_b2.ra_in1k`（SHA-256 `e9adbcce…`，Apache-2.0） |
| 产物 | `outputs/20261029-125907_run008a_effb2_s2026/`（`best.pt` 31,314,907 B） |
| 耗时 | 2026-10-29 20:59:07 → 21:33:40 CST，共 **34 分 33 秒**；前约 3 分钟独占、其余与 RUN008-B 并行，耗时不可横向比较 |

metrics.csv（6 轮原样复制）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0254 | 0.9162 | 0.7810 | 0.3921 |
| 2 | 3.00e-04 | 0.9357 | 0.9097 | 0.7913 | 0.3913 |
| 3 | 3.00e-04 | 0.8805 | 0.9309 | 0.7948 | 0.4249 |
| 4 | 2.27e-04 | 0.8199 | 0.9103 | **0.7952** | 0.4095 |
| 5 | 8.25e-05 | 0.6775 | 1.0625 | 0.7752 | 0.4172 |
| 6 | 1.00e-05 | 0.5538 | 1.2204 | 0.7784 | 0.4148 |

**结论**：最优 epoch 4，macro-AUC **0.7952**、F1@0.5 **0.4095**（`run_compare`
重算）。比 RUN-007-A（0.7906 / 0.4198）AUC **+0.0046**、F1 −0.0103，未达 +0.005 判读门；
epoch 5–6 valid_loss 由 0.91 抬升到 1.22、AUC 回落，6 轮预算内已在 epoch 4 达峰。逐类 AUC
class2 +0.0507、class7 +0.0115 提升，class9 −0.0110、class4 −0.0084 回落，方向不一致。

### RUN008-B run008b_effv2s_s2026（`configs/exp_run008b.yaml`）

配置 [configs/exp_run008b.yaml](../configs/exp_run008b.yaml)，完整内容：

```yaml
_base: exp_run007a.yaml

model:
  backbone: efficientnetv2_rw_s.ra2_in1k

train:
  epochs: 6
  early_stop_patience: 0
```

| 项 | 值 |
|---|---|
| 相对父配置的差异 | `model.backbone: densenet121.ra_in1k → efficientnetv2_rw_s.ra2_in1k`、`train.epochs: 25 → 6`、`train.early_stop_patience: 5 → 0`；其余继承 RUN-007-A |
| 运行命令 | `python src/run_train.py --config configs/exp_run008b.yaml --seed 2026 --tag run008b_effv2s_s2026` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 RUN-007-A 同一份） |
| 预训练 | 已加载 `efficientnetv2_rw_s.ra2_in1k`（SHA-256 `81379566…`，Apache-2.0） |
| 产物 | `outputs/20261029-130144_run008b_effv2s_s2026/`（`best.pt` 89,597,451 B） |
| 耗时 | 2026-10-29 21:01:44 → 21:36:48 CST，共 **35 分 04 秒**；与 RUN008-A 并行，耗时不可横向比较 |

metrics.csv（6 轮原样复制）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0011 | 0.9218 | 0.7787 | 0.3997 |
| 2 | 3.00e-04 | 0.9274 | 0.8950 | **0.7955** | 0.4102 |
| 3 | 3.00e-04 | 0.8737 | 0.9253 | 0.7912 | 0.4287 |
| 4 | 2.27e-04 | 0.7942 | 0.9321 | 0.7824 | 0.4085 |
| 5 | 8.25e-05 | 0.6151 | 1.0340 | 0.7951 | 0.4298 |
| 6 | 1.00e-05 | 0.4705 | 1.2541 | 0.7902 | 0.4402 |

**结论**：最优 epoch 2，macro-AUC **0.7954**（`metrics.csv` 0.7955）、F1@0.5 **0.4100**
（重算）。比 RUN-007-A AUC **+0.0048**、F1 −0.0098，未达 +0.005 门；epoch 2 之后 AUC 在
0.78–0.795 间波动，epoch 6 valid_loss 抬升到 1.25。逐类 class2 +0.0451、class4 +0.0425
提升，class5 −0.0142、class8 −0.0140 回落。

### RUN008-C run008c_convnext_tiny_s2026（`configs/exp_run008c.yaml`）

配置 [configs/exp_run008c.yaml](../configs/exp_run008c.yaml)，完整内容：

```yaml
_base: exp_run007a.yaml

model:
  backbone: convnext_tiny.fb_in1k

train:
  epochs: 6
  early_stop_patience: 0
```

| 项 | 值 |
|---|---|
| 相对父配置的差异 | `model.backbone: densenet121.ra_in1k → convnext_tiny.fb_in1k`、`train.epochs: 25 → 6`、`train.early_stop_patience: 5 → 0`；其余继承 RUN-007-A |
| 运行命令 | `python src/run_train.py --config configs/exp_run008c.yaml --seed 2026 --tag run008c_convnext_tiny_s2026` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 RUN-007-A 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（SHA-256 `08b9dc9c…`，Apache-2.0） |
| 产物 | `outputs/20261029-133817_run008c_convnext_tiny_s2026/`（`best.pt` 111,384,747 B） |
| 耗时 | 2026-10-29 21:38:17 → 22:02:35 CST，共 **24 分 18 秒**；前 13 分 43 秒与 RUN008-D 并行、之后独占，耗时不可横向比较 |

metrics.csv（6 轮原样复制）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9854 | 0.9388 | 0.7774 | 0.3926 |
| 2 | 3.00e-04 | 0.9303 | 0.9169 | 0.7868 | 0.3979 |
| 3 | 3.00e-04 | 0.8893 | 0.9224 | 0.7930 | 0.4077 |
| 4 | 2.27e-04 | 0.8270 | 0.9095 | 0.7960 | 0.4147 |
| 5 | 8.25e-05 | 0.6883 | 0.9605 | **0.8007** | 0.4381 |
| 6 | 1.00e-05 | 0.5576 | 1.0395 | 0.7980 | 0.4490 |

**结论**：最优 epoch 5，macro-AUC **0.8008**（`metrics.csv` 0.8007）、F1@0.5 **0.4382**
（重算）。比 RUN-007-A AUC **+0.0102**、F1 **+0.0184**，超过 +0.005 判读门；逐类 8 升 2 降
（class3 +0.0319、class4 +0.0275、class1 +0.0186 提升最大，class8 −0.0113、class10 −0.0036
回落），没有明显退化类。但训练预算（6 轮无早停 vs 12 轮早停）、单 seed、并行执行三项差异
仍在，只能作为候选方向，不能据此切换默认骨干。

### RUN008-D run008d_convnextv2_nano_s2026（`configs/exp_run008d.yaml`，仅研究）

配置 [configs/exp_run008d.yaml](../configs/exp_run008d.yaml)，完整内容：

```yaml
_base: exp_run007a.yaml

model:
  backbone: convnextv2_nano.fcmae_ft_in1k

train:
  epochs: 6
  early_stop_patience: 0
```

| 项 | 值 |
|---|---|
| 相对父配置的差异 | `model.backbone: densenet121.ra_in1k → convnextv2_nano.fcmae_ft_in1k`、`train.epochs: 25 → 6`、`train.early_stop_patience: 5 → 0`；其余继承 RUN-007-A |
| 运行命令 | `python src/run_train.py --config configs/exp_run008d.yaml --seed 2026 --tag run008d_convnextv2_nano_s2026` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 RUN-007-A 同一份） |
| 预训练 | 已加载 `convnextv2_nano.fcmae_ft_in1k`（SHA-256 `c9b5106b…`，**CC BY-NC 4.0**；骨干 158 个张量逐一核对与权重文件数值一致） |
| 产物 | `outputs/20261029-140248_run008d_convnextv2_nano_s2026/`（`best.pt` 60,021,051 B） |
| 耗时 | 2026-10-29 22:02:48 → 22:25:27 CST，共 **22 分 39 秒**（单独运行；此前 21:38–21:52 的并行运行按用户要求停止并删除） |

metrics.csv（6 轮原样复制）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.1834 | 1.1637 | 0.4777 | 0.1128 |
| 2 | 3.00e-04 | 1.1780 | 1.1624 | **0.5097** | 0.1604 |
| 3 | 3.00e-04 | 1.1639 | 1.1633 | 0.4857 | 0.1307 |
| 4 | 2.27e-04 | 1.1635 | 1.1582 | 0.4956 | 0.0892 |
| 5 | 8.25e-05 | 1.1621 | 1.1588 | 0.4975 | 0.0832 |
| 6 | 1.00e-05 | 1.1619 | 1.1582 | 0.5004 | 0.1272 |

**结论（异常 run，不参与判读）**：6 轮 train_loss 仅从 1.183 降到 1.162、valid_loss 基本不变，
`run_compare` 重算 macro-AUC **0.4977**、F1@0.5 **0.1604**，逐类 AUC 0.46–0.54（近似随机）。
已核验：骨干 158 个权重张量全部与预训练文件数值一致（仅 `head.fc` 正常替换为 10 类）、
单独运行显存 5.9 GB（与 RUN008-C 同量级）→ 排除权重加载失败与显存不足；具体原因未定位
（怀疑 AMP fp16 下 ConvNeXtV2 的 GRN / LayerNorm 组合或该骨干与当前 lr / 优化器不匹配）。
该组权重为 CC BY-NC 4.0，赛题第 1 条不接受，**不能用于提交**；本组仅研究记录。

### RUN-008 组间对照

`outputs/compare_run008/summary.md`（本地重算，valid 859 studies，阈值网格 0.2–0.7）：

| run | 骨干 | epochs | best_epoch | macro_auc | f1@0.2 | f1@0.3 | f1@0.4 | f1@0.5 | f1@0.6 | f1@0.7 |
|---|---|---|---|---|---|---|---|---|---|---|
| RUN-007-A | `densenet121.ra_in1k` | 25 | 7 | 0.7906 | 0.3701 | 0.3872 | 0.4061 | 0.4198 | 0.4270 | 0.3972 |
| RUN008-A | `efficientnet_b2.ra_in1k` | 6 | 4 | 0.7952 | 0.3569 | 0.3791 | 0.3994 | 0.4095 | 0.4096 | 0.3974 |
| RUN008-B | `efficientnetv2_rw_s.ra2_in1k` | 6 | 2 | 0.7954 | 0.3467 | 0.3694 | 0.3886 | 0.4100 | 0.4185 | 0.3710 |
| RUN008-C | `convnext_tiny.fb_in1k` | 6 | 5 | **0.8008** | 0.3914 | 0.4116 | 0.4259 | 0.4382 | 0.4438 | 0.4416 |
| RUN008-D | `convnextv2_nano.fcmae_ft_in1k` | 6 | 2 | 0.4977 | 0.2719 | 0.2719 | 0.2719 | 0.1604 | 0.0000 | 0.0000 |

相对 RUN-007-A：A +0.0046、B +0.0048、C **+0.0102**、D −0.2929（异常）。F1@0.5 上 A/B 低于
RUN-007-A（−0.0103 / −0.0098），C 高 +0.0184，且 C 在 0.2–0.7 全段更高。

逐类 AUC（`outputs/compare_run008/per_class.csv`，Δ 为四位小数相减）：

| class_id | support | RUN-007-A | RUN008-A | RUN008-B | RUN008-C |
|---|---|---|---|---|---|
| 1 | 24 | 0.7208 | 0.7244 | 0.7137 | 0.7394 |
| 2 | 37 | 0.7944 | 0.8451 | 0.8395 | 0.8078 |
| 3 | 46 | 0.7650 | 0.7708 | 0.7669 | 0.7969 |
| 4 | 68 | 0.6975 | 0.6891 | 0.7400 | 0.7250 |
| 5 | 115 | 0.8752 | 0.8757 | 0.8610 | 0.8782 |
| 6 | 177 | 0.7794 | 0.7781 | 0.7786 | 0.7823 |
| 7 | 193 | 0.7718 | 0.7833 | 0.7708 | 0.7854 |
| 8 | 214 | 0.7205 | 0.7195 | 0.7065 | 0.7092 |
| 9 | 260 | 0.9039 | 0.8929 | 0.9070 | 0.9091 |
| 10 | 306 | 0.8778 | 0.8731 | 0.8700 | 0.8742 |

口径说明：

1. **训练预算不同**：RUN-007-A 为 25 轮上限、第 12 轮早停（峰值 epoch 7）；RUN-008 四组固定
   6 轮、无早停。四组的 AUC 是 6 轮预算内的峰值，与 RUN-007-A 的差值不能当作严格单变量对照；
   若要严格配对，应按同样 6 轮预算重跑 DenseNet-121（本计划未做）。
2. **并行与耗时**：A+B 并行（20:59–21:37）、C+D 并行（21:38–22:02，D 于 21:52 按用户要求
   停止并删除该次异常目录、22:02 起单独重跑）。单轮耗时受共享 GPU 影响，不可与其他 run 的
   串行耗时对照，也不能用于耗时合规判断。
3. **D 组异常**：见上，权重加载与显存已排除，原因未定位；同时 CC BY-NC 4.0 不合规，该组
   不参与骨干优劣判读。
4. **判读**：A/B 未达 +0.005 判读门；C 超过门且逐类无明显退化，但受预算不同、单 seed、
   并行执行限制，只能作为候选方向。若要切换默认骨干，需补同预算（6 轮）的 DenseNet-121
   对照、第二 seed 复核，并完成权重许可 / SHA-256 材料与赛题指定 T4 端到端耗时实测。

## RUN-009 骨干 @384×384 完整口径复核（EfficientNetV2-S RW）

按 [train_plan09.md](train_plan09.md) 执行：在 RUN-008 的 6 轮 320×320 骨干对照基础上，把输入
分辨率提到 384×384、训练预算恢复为 25 轮上限 + 早停 5，复核 RUN-008-B 的 EfficientNetV2-S RW。
原计划串行先跑 RUN-009-B（ConvNeXt-Tiny）、后跑 RUN-009-A；实际训练环境换为新实例
（V100-SXM2-32G），原串行脚本 `/root/run009.sh` 不在该实例上，遂先启动 RUN-009-A，并在第 6 轮
结束后按用户要求停止，**RUN-009-B 未执行**。

### RUN-009-A run009a_effv2s_384_s2026（`configs/exp_run009a.yaml`）

配置 [configs/exp_run009a.yaml](../configs/exp_run009a.yaml)，完整内容：

```yaml
_base: exp_run008b.yaml

image:
  size: 384

train:
  epochs: 25
  early_stop_patience: 5
```

（其余字段全部继承 [configs/exp_run008b.yaml](../configs/exp_run008b.yaml) → `exp_run007a.yaml`：
`efficientnetv2_rw_s.ra2_in1k`、正侧位 7:3 加权聚合、`pos_weight: true`、warmup 2 轮后 cosine
衰减到 1e-5、`batch_studies: 16`、`num_workers: 16`、CUDA AMP、letterbox、ImageNet 归一化、
seed 2026。）

| 项 | 值 |
|---|---|
| 相对父配置的差异 | `image.size: 320 → 384`、`train.epochs: 6 → 25`、`train.early_stop_patience: 0 → 5`；其余继承 RUN-008-B |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_run009a.yaml --seed 2026 --tag run009a_effv2s_384_s2026` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / 96 核 CPU / 375 GB 内存 / seed 2026；代码 `bf7e276` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 RUN-007-A / RUN-008 同一份；`check_data.py --full` 全部通过） |
| 预训练 | 已加载 `efficientnetv2_rw_s.ra2_in1k`（HF commit `ee38fe1c…`、96,483,448 B、SHA-256 `81379566…`、Apache-2.0；本地缓存校验通过后离线加载，`train.log` 打印「预训练权重: 已加载」） |
| 产物 | `outputs/20261030-140736_run009a_effv2s_384_s2026/`（`best.pt` 89,586,379 B / `last.pt` / `metrics.csv` / `train.log`） |
| 耗时 | 2026-10-30 14:07:36 → 14:50 CST，**6/25 轮后按用户要求停止**（非早停、非跑满），共约 **41 分钟**；单轮约 6.7 分钟；显存峰值约 15.9 GB（独占运行） |

启动前一次失败与处理：首次 14:03 启动时 HuggingFace 直连失败（`[Errno 99] Cannot assign
requested address ... huggingface.co/timm/efficientnetv2_rw_s.ra2_in1k/...`，重试 5 次后
`src/model/network.py` 捕获异常并回退随机初始化），已终止并删除该次 run 目录；改用
`HF_ENDPOINT=https://hf-mirror.com` 预下载权重（`/root/.cache/huggingface/hub/models--timm--efficientnetv2_rw_s.ra2_in1k/`，
文件 96,483,448 B、SHA-256 `81379566b6b222f182fa010a993ce396e97b08568bb4f87241b61b064294a65e`，
与 RUN-008-B 记录一致），再以 `HF_HUB_OFFLINE=1` 重启，`train.log` 确认预训练权重加载成功。

metrics.csv（6 轮原样复制，epoch 6 结束后停止）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0076 | 0.9082 | 0.7805 | 0.3984 |
| 2 | 3.00e-04 | 0.9244 | 0.9107 | 0.7914 | 0.3977 |
| 3 | 3.00e-04 | 0.8679 | 0.9361 | 0.7865 | 0.4128 |
| 4 | 2.99e-04 | 0.8190 | 0.9323 | 0.7900 | 0.4040 |
| 5 | 2.94e-04 | 0.7482 | 0.9507 | **0.7944** | 0.4185 |
| 6 | 2.87e-04 | 0.6699 | 1.0356 | 0.7930 | 0.4270 |

**结论**：最优 epoch 5，macro-AUC **0.7944**、F1@0.5 **0.4185**（取自 `last.pt` 的
`extras.best_metrics`；`run_eval.py` 在 valid 上重算 macro-AUC **0.7945**、F1@0.5 **0.4178**，
差 ≤0.0007）。比 RUN-007-A（0.7906 / 0.4198）AUC **+0.0038**、F1@0.5 −0.0020，**未达 +0.005
判读门**（门槛 0.7956）；比同骨干 320×320 固定 6 轮的 RUN-008-B（0.7954 / 0.4100）AUC 低
**−0.0010**，即本组没有超过该骨干 6 轮 320 短跑的水平。epoch 5 之后 valid_loss 由 0.9507 升到
1.0356、AUC 回落，train_loss 仍单调下降，出现过拟合迹象。**本组只跑 6/25 轮即按用户要求
停止，cosine 周期未走完（第 6 轮 lr 仍有 2.87e-4），不能据此判定 25 轮完整预算下的最终水平**；
停止原因是人工中断，不是早停触发。

产物说明：训练脚本 `src/run_train.py:256-267` 只在训练完整结束后才把 `best_state` 落盘为
`best.pt`，本次提前停止后该文件缺失；已用临时脚本从 `last.pt` 的 `extras.best_state`（epoch 5）
导出 `best.pt`（复用 `common.checkpoint.save_checkpoint`，epoch / metrics / config / seed /
metadata 的写法与训练脚本一致），上表指标与 `run_eval.py` 结果均基于该文件。

逐类 AUC（`run_eval.py`，Δ 为四位小数相减）：

| class_id | support | RUN-007-A 320 | RUN-008-B 320 / 6 轮 | RUN-009-A 384 | Δ（009A−007A） |
|---|---|---|---|---|---|
| 1 | 24 | 0.7208 | 0.7137 | 0.7274 | +0.0066 |
| 2 | 37 | 0.7944 | 0.8395 | 0.8620 | +0.0676 |
| 3 | 46 | 0.7650 | 0.7669 | 0.7857 | +0.0207 |
| 4 | 68 | 0.6975 | 0.7400 | 0.6864 | −0.0111 |
| 5 | 115 | 0.8752 | 0.8610 | 0.8700 | −0.0052 |
| 6 | 177 | 0.7794 | 0.7786 | 0.7651 | −0.0143 |
| 7 | 193 | 0.7718 | 0.7708 | 0.7729 | +0.0011 |
| 8 | 214 | 0.7205 | 0.7065 | 0.6981 | −0.0224 |
| 9 | 260 | 0.9039 | 0.9070 | 0.9013 | −0.0026 |
| 10 | 306 | 0.8778 | 0.8700 | 0.8759 | −0.0019 |

差异集中在稀有类 2/3 提升（+0.0676 / +0.0207）与常见类 6/8 回落（−0.0143 / −0.0224）；
macro-AUC 仅高 +0.0038，没有达到判读门，也没有单类大幅退化。

## A-001：RUN-008-C 重跑（epoch=8 / patience=2 / 320×320 / seed=42）

用户要求在 RUN008-C 基础上重跑：训练上限改为 8 轮、早停 patience 2、输入 320×320、seed 42，
配置独立完整展开（不继承），实验命名 A-001。除上述三项外，其余字段与 `configs/exp_run008c.yaml`
展开后的最终配置逐字段一致（`tests/test_config.py::test_a001_experiment_config_loads` 断言）。

配置 [configs/exp_A001.yaml](../configs/exp_A001.yaml)：完整展开、无 `_base`，内容与
`exp_run008c.yaml` 展开结果相比仅三处不同：`train.epochs: 8`、`train.early_stop_patience: 2`、
顶层 `seed: 42`；其余字段（骨干 `convnext_tiny.fb_in1k`、`image.size: 320`、letterbox、ImageNet
归一化、无水平翻转、正侧位 7:3 加权聚合、`pos_weight: true`、warmup 2 轮后 cosine 衰减到 1e-5、
`batch_studies: 16`、`lr: 0.0003`、`weight_decay: 0.01`、CUDA AMP、`grad_clip: 5.0`、
`num_workers: 16` 等）与 RUN008-C 同值。

| 项 | 值 |
|---|---|
| 相对 RUN008-C 的差异 | `train.epochs: 6 → 8`、`train.early_stop_patience: 0 → 2`、`seed: 2026 → 42`；其余全部一致（完整展开，无 `_base`） |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A001.yaml --seed 42 --tag A-001_convnext_tiny_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / 96 核 CPU / seed 42；代码 `0475bc7` + 本 run 的配置与测试改动 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 RUN-007-A / RUN-008 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（HF commit `b43a6303…`、114,374,272 B、SHA-256 `08b9dc9c…`、Apache-2.0）；本环境无法直连 HuggingFace 与 hf-mirror，从 ModelScope 下载 `timm/convnext_tiny.fb_in1k` 后按 HF 缓存结构放置（SHA-256 与 RUN008-C 记录一致），`HF_HUB_OFFLINE=1` 离线加载 |
| 产物 | `outputs/20261030-150749_A-001_convnext_tiny_s42/`（`best.pt` 111,384,683 B / `last.pt` / `metrics.csv` / `train.log`） |
| 耗时 | 2026-10-30 15:07:49 → 15:41:50 CST，共 **34 分 01 秒**（6 轮后早停触发，单轮约 5.7 分钟）；显存峰值约 11.7 GB（独占运行） |

metrics.csv（6 轮原样复制，epoch 6 结束后连续 2 轮无提升、触发早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9916 | 0.9412 | 0.7738 | 0.3932 |
| 2 | 3.00e-04 | 0.9332 | 0.9315 | 0.7881 | 0.4076 |
| 3 | 3.00e-04 | 0.8894 | 0.9057 | 0.7940 | 0.4137 |
| 4 | 2.72e-04 | 0.8466 | 0.8856 | **0.8038** | 0.4255 |
| 5 | 2.00e-04 | 0.7613 | 0.9333 | 0.8016 | 0.4289 |
| 6 | 1.10e-04 | 0.6161 | 1.0194 | 0.8012 | 0.4308 |

**结论**：最优 epoch 4，macro-AUC **0.8038**、F1@0.5 **0.4255**（`run_eval.py` 在 valid 上
重算，与 `metrics.csv` 一致）；第 5、6 轮 AUC 连续低于 epoch 4，`early_stop_patience: 2` 于第
6 轮后触发早停（训练日志「连续 2 个 epoch 没有提升，提前结束训练」）。与 RUN008-C
（0.8008 / 0.4382，6 轮固定无早停、seed 2026、RTX 3090）相比：macro-AUC **+0.0030**、
F1@0.5 **−0.0127**；但两者同时差 seed（42 vs 2026）、训练上限（8 vs 6 轮，cosine 周期长度也
随之不同：epoch 4 的 lr 为 2.72e-4 vs RUN008-C 的 2.27e-4）与早停（2 vs 0）三项，**不是单变量
对照**，AUC 差（+0.0030）也小于已观测的 seed 波动（DenseNet-121 的 seed 波动 0.0033）。
F1@0.5 的 −0.0127 在 0.2–0.7 全阈值段均存在（Δ −0.011~−0.028），与 RUN008-C 的 `pos_weight`
校准形态不同，不能据此判定 F1 变差；F1 峰值两 run 都在 0.6（A-001 0.4328 / RUN008-C 0.4438）。

阈值曲线（`run_eval.py`）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-001 | 0.3641 | 0.3835 | 0.4046 | 0.4255 | 0.4328 | 0.4205 |
| RUN008-C | 0.3914 | 0.4116 | 0.4259 | 0.4382 | 0.4438 | 0.4416 |

逐类 AUC（A-001 来自 `run_eval.py`，RUN008-C 来自 `outputs/compare_run008/per_class.csv` 的
重算值；Δ 为四位小数相减）：

| class_id | support | RUN008-C | A-001 | Δ（A-001−RUN008-C） |
|---|---|---|---|---|
| 1 | 24 | 0.7394 | 0.7438 | +0.0044 |
| 2 | 37 | 0.8078 | 0.8537 | +0.0459 |
| 3 | 46 | 0.7969 | 0.7889 | −0.0080 |
| 4 | 68 | 0.7250 | 0.7374 | +0.0124 |
| 5 | 115 | 0.8782 | 0.8719 | −0.0063 |
| 6 | 177 | 0.7823 | 0.7867 | +0.0044 |
| 7 | 193 | 0.7854 | 0.7801 | −0.0053 |
| 8 | 214 | 0.7092 | 0.7048 | −0.0044 |
| 9 | 260 | 0.9091 | 0.8955 | −0.0136 |
| 10 | 306 | 0.8742 | 0.8754 | +0.0012 |

逐类 6 升 4 降：class2 +0.0459（support 仅 37）、class4 +0.0124 提升最大，class9 −0.0136、
class3 −0.0080 回落；没有单类大幅退化。**判读**：A-001 在更宽的轮数上限 + 早停下，6 轮内达到
0.8038，与 RUN008-C 的 0.8008 同量级；由于三项配置差异同时改变、且只用单 seed，只能说明
「RUN008-C 的 ConvNeXt-Tiny 水平在本组复现」，不能归因于扩展轮数或早停策略。

## A-001-B：A-001 的 ASL 损失对照（320×320 / seed=42）

用户要求以 A-001 为基准把损失函数由 BCE（`pos_weight: true`）换成 ASL（非对称损失），其余口径
逐字段保持不变，实验命名 A-001-B。ASL 与 `pos_weight` 在配置层互斥（`TrainConfig` 校验），故
`pos_weight` 显式关闭；`configs/exp_A001b.yaml` 以 `_base: exp_A001.yaml` 只覆盖损失相关五项
（`train.loss: asl`、`train.pos_weight: false`、`train.asl_gamma_pos: 0.0`、`train.asl_gamma_neg: 4.0`、
`train.asl_clip: 0.05`），其余（ConvNeXt-Tiny、320×320 letterbox、ImageNet 归一化、无水平翻转、
正侧位 7:3 加权聚合、warmup 2 轮后 cosine 衰减到 1e-5、`batch_studies: 16`、`lr: 0.0003`、
`weight_decay: 0.01`、CUDA AMP、`grad_clip: 5.0`、`num_workers: 16`、8 轮上限、早停 2、seed 42）
与 A-001 同值（`tests/test_config.py::test_a001b_experiment_config_loads` 断言仅五项不同）。

| 项 | 值 |
|---|---|
| 相对 A-001 的差异 | 损失方案整体替换：BCE + `pos_weight` → ASL（gamma_pos 0 / gamma_neg 4 / clip 0.05，论文默认值）；其余全部一致 |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A001b.yaml --seed 42 --tag A-001b_convnext_tiny_asl_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / 96 核 CPU / seed 42；代码 `9bde5a8` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 / RUN-008 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（HF commit `b43a6303…`、114,374,272 B、SHA-256 `08b9dc9c…`、Apache-2.0）；`HF_HUB_OFFLINE=1` 离线加载 |
| 产物 | `outputs/20261030-165734_A-001b_convnext_tiny_asl_s42/`（`best.pt` 111,384,875 B / `last.pt` 445,536,235 B / `metrics.csv` / `train.log` / `config.yaml`） |
| 耗时 | 2026-10-30 16:57:34 → 17:32:15 CST，共 **34 分 41 秒**（6 轮后早停触发，单轮约 5.8 分钟）；训练期间 `nvidia-smi` 抽样显存约 13.8–14.3 GB（独占运行） |

metrics.csv（6 轮原样复制，epoch 6 结束后连续 2 轮无提升、触发早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.1067 | 0.1051 | 0.7889 | 0.3878 |
| 2 | 3.00e-04 | 0.1030 | 0.1089 | 0.7816 | 0.3903 |
| 3 | 3.00e-04 | 0.1000 | 0.1055 | 0.7889 | 0.4050 |
| 4 | 2.72e-04 | 0.0964 | 0.1057 | **0.7970** | 0.4103 |
| 5 | 2.00e-04 | 0.0892 | 0.1085 | 0.7876 | 0.4272 |
| 6 | 1.10e-04 | 0.0740 | 0.1231 | 0.7710 | 0.4131 |

**结论**：最优 epoch 4，macro-AUC **0.7970**、F1@0.5 **0.4103**（`run_eval.py` 在 valid 上
重算，与 `metrics.csv` 一致）；第 5、6 轮 AUC 连续低于 epoch 4，`early_stop_patience: 2` 于第
6 轮后触发早停（训练日志「连续 2 个 epoch 没有提升，提前结束训练」）。与 A-001
（0.8038 / 0.4255，同 6 轮早停、同 best epoch 4）相比：macro-AUC **−0.0068**、F1@0.5
**−0.0152**。ASL 的 `train_loss` / `valid_loss` 量纲与 BCE 不同（focal + 概率移位后 loss 幅度
显著变小），两组 loss 数值不可横向比较，判别力只看 AUC / F1。−0.0068 超过本项目「±0.005
判读门」，且约为已观测 seed 波动（0.0033）的 2 倍，在这组 A-001 口径下 ASL 没有带来增益；
但 ASL 替换同时移除了 `pos_weight`（两者在配置层互斥），严格说是「损失方案」整体替换而非纯
损失单变量，且仍为单 seed，不能据此断言 ASL 在全部口径下无效。

阈值曲线（`run_eval.py`）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-001-B | 0.2828 | 0.2996 | 0.3415 | 0.4103 | 0.4018 | 0.2688 |
| A-001 | 0.3641 | 0.3835 | 0.4046 | 0.4255 | 0.4328 | 0.4205 |

A-001-B 的 F1 峰值在 0.5（0.4103），0.6 为 0.4018；A-001 峰值在 0.6（0.4328）。A-001-B
在 0.2–0.7 全段均低于 A-001，区别不是校准右移，而是判别力整体略降。

逐类 AUC（两组均来自 `run_eval.py`）：

| class_id | support | A-001 | A-001-B | Δ（B−A-001） |
|---|---|---|---|---|
| 1 | 24 | 0.7438 | 0.7086 | −0.0352 |
| 2 | 37 | 0.8537 | 0.8458 | −0.0079 |
| 3 | 46 | 0.7889 | 0.7914 | +0.0025 |
| 4 | 68 | 0.7374 | 0.7183 | −0.0191 |
| 5 | 115 | 0.8719 | 0.8696 | −0.0023 |
| 6 | 177 | 0.7867 | 0.7781 | −0.0086 |
| 7 | 193 | 0.7801 | 0.7736 | −0.0065 |
| 8 | 214 | 0.7048 | 0.7074 | +0.0026 |
| 9 | 260 | 0.8955 | 0.9035 | +0.0080 |
| 10 | 306 | 0.8754 | 0.8733 | −0.0021 |

逐类 3 升 7 降：class1（support 仅 24）−0.0352、class4 −0.0191 退化最大，class9 +0.0080、
class3 +0.0025、class8 +0.0026 略升；退化集中在低频类，与 ASL「抑制简单负样本、强调难正样本」
的机制在有效正样本极少的类上反而更不利的预期方向一致，但单 seed 不能确认。
**判读**：A-001 口径下 ASL 未达增益门（−0.0068），不切换默认损失；如需继续验证 ASL，应换
support 更充分的骨干或更长轮数另开对照，而不是在单 seed 上继续调 gamma。


## A-001-a：A-001 的训练随机水平翻转对照（320×320 / seed=42）

用户要求以 A-001 为基准只开启训练随机水平翻转，其余口径逐字段保持不变，实验命名 A-001-a。
配置 `configs/exp_A001a.yaml` 为完整展开、无 `_base`，与 `configs/exp_A001.yaml` 的展开结果仅差
`image.hflip: false → true` 一处（该配置在训练后于 17:20 补提交 `ac2d2f3`；run 目录 `config.yaml`
与仓库文件逐行一致）。

| 项 | 值 |
|---|---|
| 相对 A-001 的差异 | `image.hflip: false → true`（训练随机水平翻转）；其余全部一致（完整展开，无 `_base`） |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A001a.yaml --seed 42 --tag A-001-a_convnext_tiny_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / seed 42；代码 `0475bc7` + 本 run 的配置改动 |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（SHA-256 `08b9dc9c…`、Apache-2.0）；`HF_HUB_OFFLINE=1` 离线加载 |
| 产物 | `outputs/20261030-154652_A-001-a_convnext_tiny_s42/`（`best.pt` 111,384,683 B / `last.pt` 445,536,107 B / `metrics.csv` / `train.log` / `config.yaml`） |
| 耗时 | 2026-10-30 15:46:52 → 16:23:50 CST，共 **36 分 58 秒**（6 轮后早停触发，单轮约 6.2 分钟，略慢于 A-001 的 5.7 分钟/轮，与随机翻转的额外增强开销一致）；训练期间未抽样显存 |

metrics.csv（6 轮原样复制，epoch 6 结束后连续 2 轮无提升、触发早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9980 | 0.9452 | 0.7666 | 0.3844 |
| 2 | 3.00e-04 | 0.9414 | 0.9599 | 0.7674 | 0.3991 |
| 3 | 3.00e-04 | 0.9006 | 0.9138 | 0.7855 | 0.4149 |
| 4 | 2.72e-04 | 0.8656 | 0.9060 | **0.7960** | 0.4047 |
| 5 | 2.00e-04 | 0.8042 | 0.9335 | 0.7951 | 0.4213 |
| 6 | 1.10e-04 | 0.6973 | 0.9785 | 0.7932 | 0.4267 |

**结论**：最优 epoch 4，macro-AUC **0.7960**、F1@0.5 **0.4045**（`run_eval.py` 在 valid 上重算；
`metrics.csv` 同轮为 0.7960 / 0.4047，F1 差 0.0002）；第 5、6 轮 AUC 连续低于 epoch 4，
`early_stop_patience: 2` 于第 6 轮后触发早停（训练日志「连续 2 个 epoch 没有提升，提前结束训练」）。
与 A-001（0.8038 / 0.4255，同 seed、同 8 轮上限与早停，唯一差异是 `hflip`）相比：macro-AUC
**−0.0078**、F1@0.5 **−0.0210**。−0.0078 超过本项目「±0.005 判读门」，约为已观测 seed 波动
（0.0033）的 2.4 倍 → **在本口径下训练随机水平翻转没有增益、AUC 略降**；胸片左右不对称（心影、
膈肌位置），水平翻转可能引入与标签不一致的解剖形态，方向与预期一致，但单 seed 不能据此断言
翻转在所有骨干/口径下都无效。

阈值曲线（`run_eval.py`）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-001-a | 0.3481 | 0.3653 | 0.3832 | 0.4045 | 0.4271 | 0.4365 |
| A-001 | 0.3641 | 0.3835 | 0.4046 | 0.4255 | 0.4328 | 0.4205 |

A-001-a 在 0.2–0.6 段均低于 A-001（−0.0057~−0.0210），0.7 处反超（+0.0160）；峰值从 A-001 的
0.6（0.4328）右移到 0.7（0.4365）。曲线右移说明部分差异来自概率校准，但 AUC（不受阈值影响）
本身低 0.0078，判别力下降是主证据。

逐类 AUC（A-001 来自 `run_eval.py`，A-001-a 为本 run 重算值）：

| class_id | support | A-001 | A-001-a | Δ（a−A-001） |
|---|---|---|---|---|
| 1 | 24 | 0.7438 | 0.6980 | −0.0458 |
| 2 | 37 | 0.8537 | 0.8120 | −0.0417 |
| 3 | 46 | 0.7889 | 0.7804 | −0.0085 |
| 4 | 68 | 0.7374 | 0.7280 | −0.0094 |
| 5 | 115 | 0.8719 | 0.8783 | +0.0064 |
| 6 | 177 | 0.7867 | 0.7757 | −0.0110 |
| 7 | 193 | 0.7801 | 0.7885 | +0.0084 |
| 8 | 214 | 0.7048 | 0.7202 | +0.0154 |
| 9 | 260 | 0.8955 | 0.9060 | +0.0105 |
| 10 | 306 | 0.8754 | 0.8732 | −0.0022 |

逐类 4 升 6 降：常见类 8 / 9 / 7 略升（+0.0154 / +0.0105 / +0.0084），稀有类 1 / 2 退化最大
（−0.0458 / −0.0417，support 仅 24 / 37）、class6 −0.0110；升降方向不集中，但总 AUC 下降，且退化
落在低 support 类上会放大逐类噪声。**判读**：训练随机水平翻转未达增益门（−0.0078），默认
`image.hflip: false` 保持不变；若要继续验证增强方向，应换 seed 或更换增强强度（如小角度旋转）
另开单变量对照，而不是在这一组上叠加多种增强。

## A-001-C：A-001 的冻结骨干对照（320×320 / seed=42）

用户要求以 A-001 为基准把骨干全部冻结、只训练分类头，其余口径逐字段保持不变，实验命名 A-001-C。
配置 `configs/exp_A001c.yaml` 以 `_base: exp_A001.yaml` 只覆盖 `model.freeze_backbone: true`；
训练循环同时把冻结骨干固定为 `eval()`（BatchNorm running statistics 不更新），分类头保持
`train()`。`best.pt` metadata 确认冻结生效：`freeze_backbone: true`、`total_params: 27,827,818` /
`trainable_params: 7,690`（仅 timm 分类头），与提交 `1acc229` 时的合成模型验证一致。

| 项 | 值 |
|---|---|
| 相对 A-001 的差异 | `model.freeze_backbone: false → true`（骨干冻结、仅训练分类头）；其余口径逐字段一致 |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A001c.yaml --seed 42 --tag A-001c_convnext_tiny_freeze_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / seed 42；代码 `1acc229` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（SHA-256 `08b9dc9c…`、Apache-2.0）；`HF_HUB_OFFLINE=1` 离线加载 |
| 产物 | `outputs/20261030-173249_A-001c_convnext_tiny_freeze_s42/`（`best.pt` 111,384,939 B / `last.pt` 222,818,891 B——只存分类头的优化器状态，明显小于全量微调 run 的 445 MB / `metrics.csv` / `train.log` / `config.yaml`） |
| 耗时 | 2026-10-30 17:32:49 → 18:10:50 CST，共 **38 分 01 秒**（8/8 轮跑满、未触发早停，单轮约 4.8 分钟，快于 A-001 的 5.7 分钟/轮，与骨干不反向传播一致）；训练期间未抽样显存 |

metrics.csv（8 轮原样复制，8/8 轮跑满）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0926 | 1.0686 | 0.6819 | 0.3379 |
| 2 | 3.00e-04 | 1.0388 | 1.0420 | 0.7034 | 0.3468 |
| 3 | 3.00e-04 | 1.0120 | 1.0203 | 0.7175 | 0.3575 |
| 4 | 2.72e-04 | 0.9972 | 1.0120 | 0.7208 | 0.3600 |
| 5 | 2.00e-04 | 0.9896 | 1.0079 | 0.7238 | 0.3600 |
| 6 | 1.10e-04 | 0.9830 | 1.0061 | 0.7256 | 0.3617 |
| 7 | 3.77e-05 | 0.9804 | 1.0050 | 0.7262 | 0.3593 |
| 8 | 1.00e-05 | 0.9791 | 1.0045 | **0.7264** | 0.3596 |

**结论**：最优 epoch 8（跑满 8 轮），macro-AUC **0.7264**、F1@0.5 **0.3602**（`run_eval.py` 重算；
`metrics.csv` 同轮为 0.7264 / 0.3596，F1 差 0.0006）。与 A-001（0.8038 / 0.4255，同 seed、同 8 轮
上限与早停，唯一差异是 `freeze_backbone`）相比：macro-AUC **−0.0774**、F1@0.5 **−0.0653**；
−0.0774 约为已观测 seed 波动（0.0033）的 23 倍，远超「±0.005 判读门」→ **冻结骨干在 A-001 口径下
明确不可行**。8 轮内 AUC 仍在缓慢上升（0.6819 → 0.7264），但末轮增量已收窄到 +0.0002，且 cosine
周期结束时 lr 已降到 1e-5，继续训练分类头也难以回到 0.80 量级：ImageNet 预训练特征与胸片多标签
任务的特征需求差异过大，7,690 个分类头参数不足以完成适配。默认 `model.freeze_backbone: false`
保持不变；RUN-010（ResNet-18 冻结骨干）可按此结果降优先级——两者都只训练分类头，预期同样明显低于
全量微调，若仍要执行，应把结论定位为「训练成本对照」而非「精度候选」。

阈值曲线（`run_eval.py`）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-001-c | 0.3085 | 0.3341 | 0.3551 | 0.3602 | 0.3524 | 0.2907 |
| A-001 | 0.3641 | 0.3835 | 0.4046 | 0.4255 | 0.4328 | 0.4205 |

A-001-c 在 0.2–0.7 全段低于 A-001（−0.0556~−0.1298），峰值在 0.5（0.3602），0.7 处掉到 0.2907；
判别力整体下降，不是阈值选点问题。

逐类 AUC（A-001 来自 `run_eval.py`，A-001-c 为本 run 重算值）：

| class_id | support | A-001 | A-001-c | Δ（c−A-001） |
|---|---|---|---|---|
| 1 | 24 | 0.7438 | 0.6672 | −0.0766 |
| 2 | 37 | 0.8537 | 0.7253 | −0.1284 |
| 3 | 46 | 0.7889 | 0.6963 | −0.0926 |
| 4 | 68 | 0.7374 | 0.6614 | −0.0760 |
| 5 | 115 | 0.8719 | 0.8041 | −0.0678 |
| 6 | 177 | 0.7867 | 0.6783 | −0.1084 |
| 7 | 193 | 0.7801 | 0.7085 | −0.0716 |
| 8 | 214 | 0.7048 | 0.6811 | −0.0237 |
| 9 | 260 | 0.8955 | 0.8131 | −0.0824 |
| 10 | 306 | 0.8754 | 0.8282 | −0.0472 |

逐类 **10 类全降**：class2 −0.1284、class6 −0.1084、class3 −0.0926 退化最大，support 最充足的
class8 / class10 降幅最小（−0.0237 / −0.0472）。**判读**：冻结骨干未达任何增益预期，默认配置不采用；
该结果只说明「A-001 口径下仅训分类头不可行」，不能据此否定在更大数据集或不同分类头结构上的线性探测实验。

## A-001-B（seed 2026）：A-001-B 的 seed 复核（320×320 / seed=2026）

用户要求补 A-001-B 的第二个 seed 复核：沿用 `configs/exp_A001b.yaml`（ASL 损失方案：`train.loss: asl`、
`pos_weight: false`、gamma_pos 0 / gamma_neg 4 / clip 0.05）与 A-001-B 同口径，只把 seed 由 42 改为
2026；其余字段（ConvNeXt-Tiny fb_in1k、320×320 letterbox、正侧位 7:3 加权聚合、8 轮上限、早停 2）不变。
A-001-B 单 seed 的 AUC −0.0068 与判读门 ±0.005 只差 0.0018，落在 seed 波动量级附近，先补第二 seed
确认「ASL 水平是否复现」，再决定 ASL 判定。

| 项 | 值 |
|---|---|
| 相对 A-001-B 的差异 | `seed: 42 → 2026`；其余口径逐字段一致（配置仍为 `configs/exp_A001b.yaml`，seed 由命令行传入） |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A001b.yaml --seed 2026 --tag A-001b_convnext_tiny_asl_s2026` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / seed 2026；代码 `e25d73e` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001-B 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（SHA-256 `08b9dc9c…`、Apache-2.0）；`HF_HUB_OFFLINE=1` 离线加载 |
| 产物 | `outputs/20261030-181110_A-001b_convnext_tiny_asl_s2026/`（`best.pt` 111,384,939 B / `last.pt` 445,536,363 B / `metrics.csv` / `train.log` / `config.yaml`）；metadata `freeze_backbone: false`、`total_params` = `trainable_params` = 27,827,818 |
| 耗时 | 2026-10-30 18:11:10 → 18:48:39 CST，共 **37 分 29 秒**（7 轮后早停触发，单轮约 5.4 分钟，与 A-001-B 的 5.8 分钟/轮接近）；训练期间未抽样显存 |

metrics.csv（7 轮原样复制，epoch 7 结束后连续 2 轮无提升、触发早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.1072 | 0.1057 | 0.7875 | 0.3913 |
| 2 | 3.00e-04 | 0.1037 | 0.1062 | 0.7869 | 0.4007 |
| 3 | 3.00e-04 | 0.1005 | 0.1059 | 0.7915 | 0.4179 |
| 4 | 2.72e-04 | 0.0972 | 0.1072 | 0.7903 | 0.4147 |
| 5 | 2.00e-04 | 0.0902 | 0.1079 | **0.7952** | 0.4181 |
| 6 | 1.10e-04 | 0.0753 | 0.1191 | 0.7828 | 0.4173 |
| 7 | 3.77e-05 | 0.0565 | 0.1471 | 0.7700 | 0.4082 |

**结论**：最优 epoch 5，macro-AUC **0.7952**、F1@0.5 **0.4182**（`run_eval.py` 重算；`metrics.csv`
同轮为 0.7952 / 0.4181）；第 6、7 轮 AUC 连续低于 epoch 5，`early_stop_patience: 2` 于第 7 轮后
触发早停。与 A-001-B（seed 42：0.7970 / 0.4103，唯一差异是 seed）相比：macro-AUC **−0.0018**、
F1@0.5 **+0.0079**；AUC 差远小于已观测 seed 波动（0.0033），F1 差来自两 seed 的概率校准形态（两 run
都是 ASL + `pos_weight: false`，F1 峰值都在 0.6）→ **A-001-B 的 ASL 水平在第二个 seed 上复现**
（0.7952 与 0.7970 相距 0.0018）。ASL 与 BCE 的判定仍缺 seed 2026 的 BCE 配对（A-001 只跑了 seed 42；
RUN008-C 虽为 seed 2026 + BCE，但固定 6 轮预算、无早停，不能当作同口径配对），因此维持 A-001-B 的
结论：**A-001 口径下 ASL 没有证明出增益，不切换默认损失**；若要把 ASL 判定做实，应补 A-001（BCE）的
seed 2026 对照，而不是继续调 gamma。

阈值曲线（`run_eval.py`）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-001-B(seed 2026) | 0.2864 | 0.3116 | 0.3666 | 0.4182 | 0.4254 | 0.3298 |
| A-001-B(seed 42) | 0.2828 | 0.2996 | 0.3415 | 0.4103 | 0.4018 | 0.2688 |

seed 2026 在 0.2–0.7 全段略高（+0.0036~+0.0610），峰值从 0.5 右移到 0.6（0.4254 vs 0.4103 @0.5）；
两 seed 的 F1 曲线形态接近，差异幅度小于 A-001-B 与 A-001 在 0.5 处的差（−0.0152）。F1 的 +0.0079
属于阈值选点带来的小差异，不足以改变 AUC 层面的判读。

逐类 AUC（A-001-B 来自 `run_eval.py`，seed 2026 为本 run 重算值）：

| class_id | support | A-001-B | A-001-b s2026 | Δ（s2026−s42） |
|---|---|---|---|---|
| 1 | 24 | 0.7086 | 0.7230 | +0.0144 |
| 2 | 37 | 0.8458 | 0.8261 | −0.0197 |
| 3 | 46 | 0.7914 | 0.8080 | +0.0166 |
| 4 | 68 | 0.7183 | 0.6921 | −0.0262 |
| 5 | 115 | 0.8696 | 0.8743 | +0.0047 |
| 6 | 177 | 0.7781 | 0.7732 | −0.0049 |
| 7 | 193 | 0.7736 | 0.7804 | +0.0068 |
| 8 | 214 | 0.7074 | 0.7083 | +0.0009 |
| 9 | 260 | 0.9035 | 0.9012 | −0.0023 |
| 10 | 306 | 0.8733 | 0.8657 | −0.0076 |

逐类 5 升 5 降：class4 −0.0262、class2 −0.0197 降幅最大，class3 +0.0166、class1 +0.0144 升幅最大；
升类集中在低 support 类（1 / 3），降类包含 class4（support 68）与 class2（37）→ 两 seed 的逐类
形态不一致，与「ASL 在稀有类上不稳定」的既有观察一致。**判读**：这组 seed 复核只用于确认 A-001-B
的水平可复现，不改变「A-001 口径下 ASL 未达增益门」的结论；默认损失维持 BCE + `pos_weight`。

## A-001-D：A-001 的取消正侧位加权对照（320×320 / seed=42）

用户要求以 A-001 为基准取消正侧位加权，其余口径逐字段保持不变，实验命名 A-001-D。配置
`configs/exp_A001d.yaml` 以 `_base: exp_A001.yaml` 只覆盖 `model.aggregation: view_weighted → mean`：
study 内全部图片的 logits 改为等权均值；`frontal_weight: 0.7` 保留 A-001 原值但不参与计算（训练、
验证与推理链路只在 `aggregation: view_weighted` 时传入视角）。

| 项 | 值 |
|---|---|
| 相对 A-001 的差异 | `model.aggregation: view_weighted → mean`（正位 0.7 / 侧位 0.3 加权 → study 内等权均值）；其余口径逐字段一致 |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A001d.yaml --seed 42 --tag A-001d_convnext_tiny_mean_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / seed 42；代码 `64a6525` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（`HF_HUB_OFFLINE=1` 离线加载） |
| 产物 | `outputs/20261030-190645_A-001d_convnext_tiny_mean_s42/`（`best.pt` 111,384,875 B / `last.pt` 445,536,427 B / `metrics.csv` / `train.log` / `config.yaml`） |
| 耗时 | 2026-10-30 19:06:45 → 19:46:48 CST，共 **40 分 03 秒**（8/8 轮跑满、未触发早停，单轮约 5.0 分钟）；训练末段抽样显存约 14.0 GB（14,344–14,350 MiB） |

metrics.csv（8 轮原样复制，8/8 轮跑满）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9954 | 0.9555 | 0.7649 | 0.3829 |
| 2 | 3.00e-04 | 0.9404 | 0.9525 | 0.7722 | 0.3941 |
| 3 | 3.00e-04 | 0.8971 | 0.9294 | 0.7800 | 0.4108 |
| 4 | 2.72e-04 | 0.8510 | 0.8988 | 0.7924 | 0.4078 |
| 5 | 2.00e-04 | 0.7695 | 0.9343 | 0.7905 | 0.4201 |
| 6 | 1.10e-04 | 0.6148 | 1.0859 | **0.7931** | 0.4340 |
| 7 | 3.77e-05 | 0.4717 | 1.3156 | 0.7893 | 0.4276 |
| 8 | 1.00e-05 | 0.4058 | 1.3976 | 0.7871 | 0.4239 |

**结论**：最优 epoch 6（8/8 轮跑满；训练日志结尾「连续 2 个 epoch 没有提升，提前结束训练」是轮数到期后的
正常退出），macro-AUC **0.7931**、F1@0.5 **0.4340**（`run_eval.py` 重算，与 `metrics.csv` 同轮一致）。
与 A-001（0.8038 / 0.4255，同 seed、同 8 轮上限与早停、同代码版本，唯一差异是 `model.aggregation`）相比：
macro-AUC **−0.0107**、F1@0.5 **+0.0085**。AUC 差超过 ±0.005 判读门、约为已观测 seed 波动（0.0033）的
3.2 倍 → **在 A-001 口径下取消正侧位加权让判别力下降**；F1@0.5 的 +0.0085 属校准形态变化（等权后输出
概率整体下移，F1 峰值由 A-001 的 0.6 左移到 0.5），不能视为排序能力提升。8 轮 train_loss 单调降到
0.41、valid_loss 从 e6 起回升到 1.40，过拟合明显，与 A-001 在 6 轮早停的形态一致。

阈值曲线（`run_eval.py`）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-001-D | 0.4044 | 0.4168 | 0.4294 | 0.4340 | 0.4293 | 0.4021 |
| A-001 | 0.3641 | 0.3835 | 0.4046 | 0.4255 | 0.4328 | 0.4205 |

A-001-D 在 0.2–0.5 段高于 A-001（+0.0403 / +0.0333 / +0.0248 / +0.0085），0.6–0.7 段回落
（−0.0035 / −0.0184）；峰值在 0.5，A-001 峰值在 0.6。整条曲线不是简单平移，判别力方向以
阈值无关的 AUC 为准。

逐类 AUC（A-001 来自 `run_eval.py`，A-001-D 为本 run 重算值）：

| class_id | support | A-001 | A-001-D | Δ（D−A-001） |
|---|---|---|---|---|
| 1 | 24 | 0.7438 | 0.7076 | −0.0362 |
| 2 | 37 | 0.8537 | 0.8106 | −0.0431 |
| 3 | 46 | 0.7889 | 0.7989 | +0.0100 |
| 4 | 68 | 0.7374 | 0.7305 | −0.0069 |
| 5 | 115 | 0.8719 | 0.8570 | −0.0149 |
| 6 | 177 | 0.7867 | 0.7696 | −0.0171 |
| 7 | 193 | 0.7801 | 0.7822 | +0.0021 |
| 8 | 214 | 0.7048 | 0.7072 | +0.0024 |
| 9 | 260 | 0.8955 | 0.8980 | +0.0025 |
| 10 | 306 | 0.8754 | 0.8694 | −0.0060 |

逐类 **4 升 6 降**，退化集中在 support 最少的 class2（−0.0431）与 class1（−0.0362），以及中频的
class5 / class6（−0.0149 / −0.0171）；高频的 class9 / class10 基本持平（+0.0025 / −0.0060）。
**判读**：正位视图在加权聚合中占 0.7 权重，去掉后低频类信号被侧位稀释，与本表退化模式一致；取消
正侧位加权在 A-001 口径下不带来增益（AUC −0.0107），A-001 系列继续使用 `view_weighted`，不因本组
结果改成等权。该结论与 RUN-007 / RUN-007-A（DenseNet-121 上等权 vs 加权的两 seed 配对差方向相反、
均未达 +0.005 门）不矛盾：两套骨干都只说明正侧位加权未证明稳定增益，而本组进一步说明「取消加权」
也不是改善方向。

## A-001-G 组：A-001-B 的 ASL 超参扫描（320×320 / seed=42）

用户要求在 A-001 基础上依次跑五组 ASL 超参对照（命名 A-001-G1…A-001-G5），检查负样本抑制强度
（`asl_gamma_neg`）、概率移位量（`asl_clip`）与正样本聚焦（`asl_gamma_pos`）三个方向的敏感性。
五组配置 `configs/exp_A001_G1.yaml`…`configs/exp_A001_G5.yaml` 均以 `_base: exp_A001.yaml` 继承
A-001，只覆盖损失五项（`train.loss: asl`、`train.pos_weight: false`、`train.asl_gamma_pos`、
`train.asl_gamma_neg`、`train.asl_clip`），其余口径（ConvNeXt-Tiny `convnext_tiny.fb_in1k`、
320×320 letterbox、正侧位 7:3 加权聚合、warmup 2 轮 + cosine、8 轮上限、早停 2、
`batch_studies: 16`、seed 42）逐字段沿用 A-001；参照为 A-001-B（ASL gamma_pos 0 / gamma_neg 4 /
clip 0.05：macro-AUC 0.7970 / F1@0.5 0.4103）。五组在 autodl V100-SXM2-32G 上按 G1→G5 串行执行
（链式脚本，2026-10-30 20:11:24–22:36:29，共 2 小时 25 分 05 秒），代码版本统一 `78ce901`，数据与
A-001 同一份（train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies）。五组只在 ASL 五项上不同、其余逐字段一致，`tests/test_config.py::
test_a001g_experiment_config_loads` 对五组做该断言。

| 组 | 目的 | gamma_pos | gamma_neg | clip | 相对 A-001-B 的差异 |
|---|---|---|---|---|---|
| A-001-G1 | 检查负样本抑制是否过强 | 0.0 | 2.0 | 0.05 | gamma_neg 4 → 2 |
| A-001-G2 | 检查更强负样本抑制 | 0.0 | 6.0 | 0.05 | gamma_neg 4 → 6 |
| A-001-G3 | 关闭概率移位 | 0.0 | 4.0 | 0.00 | clip 0.05 → 0.00 |
| A-001-G4 | 增大概率移位 | 0.0 | 4.0 | 0.10 | clip 0.05 → 0.10 |
| A-001-G5 | 检查正样本聚焦是否有帮助 | 1.0 | 4.0 | 0.05 | gamma_pos 0 → 1 |


### A-001-G1：gamma_neg 4 → 2（减弱负样本抑制）

`configs/exp_A001_G1.yaml` 以 `_base: exp_A001.yaml` 只覆盖损失五项，相对 A-001-B 的唯一差异是
`train.asl_gamma_neg: 4.0 → 2.0`（减弱负样本抑制）；其余口径与 A-001 / A-001-B 逐字段一致（该配置在训练前已提交 `78ce901`）。

| 项 | 值 |
|---|---|
| 相对 A-001-B 的差异 | `train.asl_gamma_neg: 4.0 → 2.0`（减弱负样本抑制）；其余口径逐字段一致 |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A001_G1.yaml --seed 42 --tag A-001-G1_convnext_tiny_asl_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / 96 核 CPU / seed 42；代码 `78ce901` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 / A-001-B 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（与 A-001-B 同一份缓存，`HF_HUB_OFFLINE=1` 离线加载） |
| 产物 | `20261030-201129_A-001-G1_convnext_tiny_asl_s42/`（`best.pt` 111,384,939 B / `last.pt` 445,536,299 B / `metrics.csv` / `train.log` / `config.yaml`） |
| 耗时 | 2026-10-30 20:11:24 → 20:41:41 CST，共 **30 分 17 秒**（6 轮后早停，单轮约 5.0 分钟）；训练期间未逐组抽样显存（G1 抽样约 14,346 MiB，与 A-001-B 的 13.8–14.3 GB 同量级） |

metrics.csv（6 轮原样复制，epoch 6 结束后连续 2 轮无提升、触发早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.1524 | 0.1502 | 0.7830 | 0.3350 |
| 2 | 3.00e-04 | 0.1468 | 0.1550 | 0.7834 | 0.2938 |
| 3 | 3.00e-04 | 0.1422 | 0.1510 | 0.7866 | 0.3285 |
| 4 | 2.72e-04 | 0.1365 | 0.1512 | **0.7949** | 0.3959 |
| 5 | 2.00e-04 | 0.1250 | 0.1552 | 0.7940 | 0.4286 |
| 6 | 1.10e-04 | 0.1014 | 0.1736 | 0.7841 | 0.4047 |

**结论**：最优 epoch 4，macro-AUC **0.7949**、F1@0.5 **0.3956**（`run_eval.py` 重算，与 `metrics.csv` 同轮一致）；F1 峰值在 0.4（0.4222），0.5 为 0.3956。与 A-001-B（0.7970 / 0.4103）相比 AUC **−0.0021**、F1@0.5 **−0.0147**：AUC 差在 seed 波动（±0.0033）内，减弱负样本抑制没有带来可辨别的排序增益；F1@0.5 下降来自概率整体下移、最优阈值由 0.5 左移到 0.4。逐类 6 升 4 降，class1 −0.0295、class2 −0.0317 退化最大（support 24 / 37），class4 +0.0194、class8 +0.0111 略升。


### A-001-G2：gamma_neg 4 → 6（更强负样本抑制）

`configs/exp_A001_G2.yaml` 以 `_base: exp_A001.yaml` 只覆盖损失五项，相对 A-001-B 的唯一差异是
`train.asl_gamma_neg: 4.0 → 6.0`（更强负样本抑制）；其余口径与 A-001 / A-001-B 逐字段一致（该配置在训练前已提交 `78ce901`）。

| 项 | 值 |
|---|---|
| 相对 A-001-B 的差异 | `train.asl_gamma_neg: 4.0 → 6.0`（更强负样本抑制）；其余口径逐字段一致 |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A001_G2.yaml --seed 42 --tag A-001-G2_convnext_tiny_asl_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / 96 核 CPU / seed 42；代码 `78ce901` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 / A-001-B 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（与 A-001-B 同一份缓存，`HF_HUB_OFFLINE=1` 离线加载） |
| 产物 | `20261030-204225_A-001-G2_convnext_tiny_asl_s42/`（`best.pt` 111,384,939 B / `last.pt` 445,536,299 B / `metrics.csv` / `train.log` / `config.yaml`） |
| 耗时 | 2026-10-30 20:42:21 → 21:14:03 CST，共 **31 分 42 秒**（6 轮后早停，单轮约 5.3 分钟）；训练期间未逐组抽样显存（G1 抽样约 14,346 MiB，与 A-001-B 的 13.8–14.3 GB 同量级） |

metrics.csv（6 轮原样复制，epoch 6 结束后连续 2 轮无提升、触发早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.0833 | 0.0822 | 0.7795 | 0.3326 |
| 2 | 3.00e-04 | 0.0805 | 0.0855 | 0.7790 | 0.3517 |
| 3 | 3.00e-04 | 0.0782 | 0.0832 | 0.7824 | 0.3521 |
| 4 | 2.72e-04 | 0.0757 | 0.0820 | **0.7999** | 0.3391 |
| 5 | 2.00e-04 | 0.0705 | 0.0846 | 0.7965 | 0.3795 |
| 6 | 1.10e-04 | 0.0599 | 0.0911 | 0.7836 | 0.3882 |

**结论**：最优 epoch 4，macro-AUC **0.7999**、F1@0.5 **0.3391**；F1 峰值右移到 0.6（0.4211），0.5 只有 0.3391。与 A-001-B 相比 AUC **+0.0029**（在 seed 波动内）、F1@0.5 **−0.0712**：0.5 阈值处逐类 recall 0.90–0.99 而 precision 很低（class10 0.520、class9 0.407，其余 ≤0.27），更大的 gamma_neg 把输出概率整体推高，排序略好但固定阈值 F1 不可用；三组峰值 F1（0.41–0.42）基本持平。逐类 7 升 3 降，class4 +0.0224、class6 +0.0169 提升，class1 −0.0153、class2 −0.0101 回落。


### A-001-G3：clip 0.05 → 0.00（关闭概率移位）

`configs/exp_A001_G3.yaml` 以 `_base: exp_A001.yaml` 只覆盖损失五项，相对 A-001-B 的唯一差异是
`train.asl_clip: 0.05 → 0.00`（关闭概率移位）；其余口径与 A-001 / A-001-B 逐字段一致（该配置在训练前已提交 `78ce901`）。

| 项 | 值 |
|---|---|
| 相对 A-001-B 的差异 | `train.asl_clip: 0.05 → 0.00`（关闭概率移位）；其余口径逐字段一致 |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A001_G3.yaml --seed 42 --tag A-001-G3_convnext_tiny_asl_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / 96 核 CPU / seed 42；代码 `78ce901` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 / A-001-B 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（与 A-001-B 同一份缓存，`HF_HUB_OFFLINE=1` 离线加载） |
| 产物 | `20261030-211449_A-001-G3_convnext_tiny_asl_s42/`（`best.pt` 111,384,939 B / `last.pt` 445,536,299 B / `metrics.csv` / `train.log` / `config.yaml`） |
| 耗时 | 2026-10-30 21:14:44 → 21:45:35 CST，共 **30 分 51 秒**（6 轮后早停，单轮约 5.1 分钟）；训练期间未逐组抽样显存（G1 抽样约 14,346 MiB，与 A-001-B 的 13.8–14.3 GB 同量级） |

metrics.csv（6 轮原样复制，epoch 6 结束后连续 2 轮无提升、触发早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.1209 | 0.1194 | 0.7868 | 0.3658 |
| 2 | 3.00e-04 | 0.1170 | 0.1225 | 0.7824 | 0.3811 |
| 3 | 3.00e-04 | 0.1138 | 0.1192 | 0.7870 | 0.3716 |
| 4 | 2.72e-04 | 0.1098 | 0.1186 | **0.7971** | 0.4323 |
| 5 | 2.00e-04 | 0.1017 | 0.1242 | 0.7908 | 0.4246 |
| 6 | 1.10e-04 | 0.0848 | 0.1412 | 0.7789 | 0.4299 |

**结论**：最优 epoch 4，macro-AUC **0.7971**、F1@0.5 **0.4320**；F1 峰值就在 0.5（0.4320），0.6 为 0.3535。与 A-001-B 相比 AUC **+0.0001**、F1@0.5 **+0.0217**：关闭概率移位后 AUC 不变、F1@0.5 明显改善，说明概率移位在本扫描中只影响校准；0.4320 是五组最高的 F1@0.5，比 A-001 的 0.4255 高 0.0065，但 AUC 仍比 A-001 低 0.0067。逐类 4 升 6 降，class4 +0.0344、class7 +0.0135 提升，class1 −0.0313、class2 −0.0131 回落。


### A-001-G4：clip 0.05 → 0.10（增大概率移位）

`configs/exp_A001_G4.yaml` 以 `_base: exp_A001.yaml` 只覆盖损失五项，相对 A-001-B 的唯一差异是
`train.asl_clip: 0.05 → 0.10`（增大概率移位）；其余口径与 A-001 / A-001-B 逐字段一致（该配置在训练前已提交 `78ce901`）。

| 项 | 值 |
|---|---|
| 相对 A-001-B 的差异 | `train.asl_clip: 0.05 → 0.10`（增大概率移位）；其余口径逐字段一致 |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A001_G4.yaml --seed 42 --tag A-001-G4_convnext_tiny_asl_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / 96 核 CPU / seed 42；代码 `78ce901` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 / A-001-B 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（与 A-001-B 同一份缓存，`HF_HUB_OFFLINE=1` 离线加载） |
| 产物 | `20261030-214622_A-001-G4_convnext_tiny_asl_s42/`（`best.pt` 111,384,939 B / `last.pt` 445,536,043 B / `metrics.csv` / `train.log` / `config.yaml`） |
| 耗时 | 2026-10-30 21:46:17 → 22:04:31 CST，共 **18 分 14 秒**（3 轮后早停，单轮约 6.1 分钟）；训练期间未逐组抽样显存（G1 抽样约 14,346 MiB，与 A-001-B 的 13.8–14.3 GB 同量级） |

metrics.csv（3 轮原样复制，epoch 2、3 连续低于 epoch 1，第 3 轮后触发早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.0936 | 0.0919 | **0.7855** | 0.3695 |
| 2 | 3.00e-04 | 0.0902 | 0.0948 | 0.7771 | 0.3814 |
| 3 | 3.00e-04 | 0.0873 | 0.0924 | 0.7761 | 0.3933 |

**结论**：最优 epoch 1（0.7855 / 0.3695），epoch 2、3 的 AUC（0.7771 / 0.7761）连续更低，第 3 轮后即早停——五组里唯一在训练早期就退化的组。macro-AUC **0.7855**、F1@0.5 **0.3695**；与 A-001-B 相比 AUC **−0.0115**、F1@0.5 **−0.0408**，超出 ±0.005 判读门（约 3.5 倍 seed 波动）。clip 0.10 把负样本概率压得过狠，模型更早锁进低置信解 → 增大概率移位方向否定。逐类 2 升 8 降，class2 −0.0814 退化最大、class1 −0.0262，class4 +0.0147 是少数回升之一。


### A-001-G5：gamma_pos 0 → 1（正样本聚焦）

`configs/exp_A001_G5.yaml` 以 `_base: exp_A001.yaml` 只覆盖损失五项，相对 A-001-B 的唯一差异是
`train.asl_gamma_pos: 0.0 → 1.0`（正样本聚焦）；其余口径与 A-001 / A-001-B 逐字段一致（该配置在训练前已提交 `78ce901`）。

| 项 | 值 |
|---|---|
| 相对 A-001-B 的差异 | `train.asl_gamma_pos: 0.0 → 1.0`（正样本聚焦）；其余口径逐字段一致 |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A001_G5.yaml --seed 42 --tag A-001-G5_convnext_tiny_asl_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / 96 核 CPU / seed 42；代码 `78ce901` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 / A-001-B 同一份） |
| 预训练 | 已加载 `convnext_tiny.fb_in1k`（与 A-001-B 同一份缓存，`HF_HUB_OFFLINE=1` 离线加载） |
| 产物 | `20261030-220518_A-001-G5_convnext_tiny_asl_s42/`（`best.pt` 111,384,939 B / `last.pt` 445,536,299 B / `metrics.csv` / `train.log` / `config.yaml`） |
| 耗时 | 2026-10-30 22:05:14 → 22:35:47 CST，共 **30 分 33 秒**（6 轮后早停，单轮约 5.1 分钟）；训练期间未逐组抽样显存（G1 抽样约 14,346 MiB，与 A-001-B 的 13.8–14.3 GB 同量级） |

metrics.csv（6 轮原样复制，epoch 6 结束后连续 2 轮无提升、触发早停）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.0594 | 0.0582 | 0.7833 | 0.3874 |
| 2 | 3.00e-04 | 0.0570 | 0.0605 | 0.7847 | 0.3948 |
| 3 | 3.00e-04 | 0.0550 | 0.0580 | 0.7917 | 0.3853 |
| 4 | 2.72e-04 | 0.0525 | 0.0580 | **0.8002** | 0.4203 |
| 5 | 2.00e-04 | 0.0475 | 0.0620 | 0.7803 | 0.4360 |
| 6 | 1.10e-04 | 0.0371 | 0.0725 | 0.7792 | 0.4245 |

**结论**：最优 epoch 4，macro-AUC **0.8002**、F1@0.5 **0.4202**；F1 峰值在 0.5（0.4202）。与 A-001-B 相比 AUC **+0.0032**（在 seed 波动内）、F1@0.5 **+0.0099**：正样本聚焦没有带来可判定的提升，但方向偏正面，且是五组里 AUC 最高的一组；与 A-001（0.8038 / 0.4255）相比 AUC −0.0036、F1@0.5 −0.0053。逐类 8 升 2 降，class4 +0.0297、class3 +0.0117 提升最大，class1 −0.0245、class2 −0.0113 回落。


### A-001-G 组间对照

阈值曲线（`run_eval.py`，0.2–0.7；五组为对各自 `best.pt` 重新执行
`--thresholds 0.2,0.3,0.4,0.5,0.6,0.7` 的结果，保存在 `outputs/eval_reports/A-001-G<g>_full.txt`）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-001-G1 | 0.3097 | 0.3661 | 0.4222 | 0.3956 | 0.3139 | 0.1950 |
| A-001-G2 | 0.2724 | 0.2782 | 0.2933 | 0.3391 | 0.4211 | 0.3494 |
| A-001-G3 | 0.2866 | 0.3103 | 0.3673 | 0.4320 | 0.3535 | 0.1953 |
| A-001-G4 | 0.2719 | 0.2787 | 0.3105 | 0.3695 | 0.3472 | 0.3017 |
| A-001-G5 | 0.2789 | 0.3009 | 0.3491 | 0.4202 | 0.3759 | 0.2009 |
| A-001-B | 0.2828 | 0.2996 | 0.3415 | 0.4103 | 0.4018 | 0.2688 |
| A-001 | 0.3641 | 0.3835 | 0.4046 | 0.4255 | 0.4328 | 0.4205 |

逐类 AUC（本组与 A-001-B 均来自 `run_eval.py`；A-001 数值取自 A-001 小节）：

| class_id | support | A-001 | A-001-B | G1 | G2 | G3 | G4 | G5 |
|---|---|---|---|---|---|---|---|---|
| 1 | 24 | 0.7438 | 0.7086 | 0.6791 | 0.6933 | 0.6773 | 0.6824 | 0.6841 |
| 2 | 37 | 0.8537 | 0.8458 | 0.8141 | 0.8357 | 0.8327 | 0.7644 | 0.8345 |
| 3 | 46 | 0.7889 | 0.7914 | 0.7863 | 0.7855 | 0.7842 | 0.7839 | 0.8031 |
| 4 | 68 | 0.7374 | 0.7183 | 0.7377 | 0.7407 | 0.7527 | 0.7330 | 0.7480 |
| 5 | 115 | 0.8719 | 0.8696 | 0.8732 | 0.8720 | 0.8674 | 0.8669 | 0.8746 |
| 6 | 177 | 0.7867 | 0.7781 | 0.7882 | 0.7950 | 0.7774 | 0.7713 | 0.7867 |
| 7 | 193 | 0.7801 | 0.7736 | 0.7763 | 0.7738 | 0.7871 | 0.7671 | 0.7758 |
| 8 | 214 | 0.7048 | 0.7074 | 0.7185 | 0.7162 | 0.7123 | 0.7130 | 0.7088 |
| 9 | 260 | 0.8955 | 0.9035 | 0.8995 | 0.9066 | 0.9077 | 0.9029 | 0.9092 |
| 10 | 306 | 0.8754 | 0.8733 | 0.8756 | 0.8800 | 0.8724 | 0.8701 | 0.8770 |
| **macro-AUC** | — | 0.8038 | 0.7970 | 0.7949 | 0.7999 | 0.7971 | 0.7855 | 0.8002 |

**判读**：

- **gamma_neg 方向（G1 / A-001-B / G2，clip 固定 0.05）**：AUC 0.7949 → 0.7970 → 0.7999 单调上升，
  但 F1@0.5 为 0.3956 → 0.4103 → 0.3391，G2 的 F1 峰值右移到 0.6（0.4211）。增大 gamma_neg 把输出
  概率整体推高（逐类 recall 高、precision 低），排序略好而固定 0.5 阈值的 F1 不可用；三组峰值 F1 都在
  0.41–0.42，与 A-001-B 持平 → 负样本抑制在 2–6 区间主要改变校准形态，未形成稳定判别力增益（AUC 差
  ≤0.0032，均在 seed 波动 ±0.0033 内）。
- **clip 方向（G3 / A-001-B / G4，gamma_pos 0 / gamma_neg 4）**：clip 0.00 的 AUC 0.7971 与 A-001-B 的
  0.7970 持平、F1@0.5 升到 **0.4320**（+0.0217，也高于 A-001 的 0.4255），F1 峰值就在 0.5；clip 0.10
  时训练早期即退化（best epoch 1、3 轮早停，AUC 0.7855 / F1 0.3695）→ 概率移位量在本扫描中只影响校准
  （0 略优于 0.05），加大到 0.10 会破坏训练，0.10 方向否定。
- **gamma_pos 方向（G5 vs A-001-B）**：gamma_pos=1 的 AUC **0.8002** 为五组最高（较 A-001-B +0.0032，
  在 seed 波动内；较 A-001 −0.0036），F1@0.5 0.4202（+0.0099），逐类 8 升 2 降（class4 +0.0297、
  class3 +0.0117 提升最大）。方向偏正面，但单 seed、差值在噪声内，不能判有效。
- **组结论**：五组 macro-AUC 0.7855–0.8002 全部低于 A-001 的 0.8038；组内最好的 G5（0.8002）与
  A-001-B 只差 +0.0032（seed 波动内）→ **ASL 超参扫描没有改变「A-001 口径下 ASL 不优于 BCE +
  `pos_weight`」的结论，默认损失与 ASL 三项默认超参不切换**。若继续该方向，建议把 G3（clip 0）与
  G5（gamma_pos 1）组合另开单变量组，并对候选组补 seed 2026 复核，而不是在单 seed 上继续扫参数。

## A-002 / A-003：MobileNetV3-Small 轻量骨干宽度对照（320×320 / seed=42）

按 [train_plan10.md](train_plan10.md) 与 [train_plan11.md](train_plan11.md) 执行：先以 A-002 检验
MobileNetV3-Small 0.5× 的轻量骨干表现，再以同配方换 1.0× 宽度做 A-003 对照。A-002 于 2026-10-30
22:39:11 启动，23:03:03 第 5 轮写盘后按用户要求中止（非早停、非收敛），因此没有 `best.pt`，无法与
A-003 做 `run_compare` 配对比较；本小节只把 A-002 的 5 轮指标作为历史记录，A-003 是完整 run。

### A-002：MobileNetV3-Small 0.5×（第 5 轮按用户要求中止）

配置 [configs/exp_A002.yaml](../configs/exp_A002.yaml)，以 `_base: exp_A001.yaml` 继承 A-001：

| 项 | 值 |
|---|---|
| 相对 A-001 的差异 | `model.backbone: convnext_tiny.fb_in1k → mobilenetv3_small_050.lamb_in1k`；`train.epochs: 8 → 25`、`train.early_stop_patience: 2 → 5`；其余口径逐字段一致 |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A002.yaml --seed 42 --tag A-002_mobilenetv3_small050_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08）/ torch 2.10.0+cu128 / timm 1.0.30 / seed 42；代码 `2fea31a` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 同一份） |
| 预训练 | 已加载 `mobilenetv3_small_050.lamb_in1k`（`HF_HUB_OFFLINE=1` 离线缓存加载），10 类模型 578,474 参数 |
| 产物 | `outputs/20261030-223911_A-002_mobilenetv3_small050_s42/`（`last.pt` 9,592,203 B / `metrics.csv` / `train.log` / `config.yaml`；**无 `best.pt`**） |
| 耗时 | 22:39:11 → 23:03:03 CST，约 24 分钟（5 轮写盘，单轮约 4.8 分钟）；显存抽样约 1.2 GB |

metrics.csv（5 轮原样复制，第 5 轮写盘后中止）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0846 | 1.0582 | 0.7033 | 0.3455 |
| 2 | 3.00e-04 | 1.0045 | 1.0108 | 0.7229 | 0.3603 |
| 3 | 3.00e-04 | 0.9707 | 0.9869 | 0.7562 | 0.3733 |
| 4 | 2.99e-04 | 0.9517 | 0.9511 | **0.7634** | 0.3918 |
| 5 | 2.94e-04 | 0.9381 | 0.9661 | 0.7591 | 0.3937 |

**结论**：第 5 轮按用户要求中止（非早停、未收敛），没有 `best.pt`，不能作为配对对照，也未进入提交候选。
中止前最佳 epoch 4（macro-AUC 0.7634 / F1@0.5 0.3918），第 5 轮 AUC 回落、valid_loss 抬头。与 A-003 前 5 轮
（0.7113 / 0.7340 / 0.7465 / 0.7497 / 0.7499）相比，A-002 在 epoch 3–5 领先（epoch 4 高 0.0137），但两者都未
收敛且 A-002 没有最终权重，**不构成有效对照**。

### A-003：MobileNetV3-Small 1.0×（完整 run，第 17 轮早停）

配置 [configs/exp_A003.yaml](../configs/exp_A003.yaml)，以 `_base: exp_A002.yaml` 只覆盖骨干：

| 项 | 值 |
|---|---|
| 相对 A-002 的差异 | `model.backbone: mobilenetv3_small_050.lamb_in1k → mobilenetv3_small_100.lamb_in1k`，其余口径逐字段一致（`tests/test_config.py::test_a003_experiment_config_loads` 校验） |
| 运行命令 | `HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A003.yaml --seed 42 --tag A-003_mobilenetv3_small100_s42` |
| 设备 / 环境 | V100-SXM2-32G（驱动 580.105.08，96 核 / 375 GB 内存）/ torch 2.10.0+cu128 / timm 1.0.30 / seed 42；代码 `2fea31a` |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies；`check_data.py --full` 复核 train / valid / test 无 Subject / Study / Dicom 交叉（日志 `outputs/A-003_check_data.log`） |
| 预训练 | 已加载 `mobilenetv3_small_100.lamb_in1k`（固定 revision `76b303ec28c466e9df37f4745dfb7aef20ab87ae`；`model.safetensors` 10,241,912 B、SHA-256 `46d2c063b18125884c48937afa4c49e18128869e52e8db96df48bf0a4d7ff697`；许可 Apache-2.0、上游训练数据 ImageNet-1k；证据 `outputs/A-003_weight_evidence/`）；10 类模型 1,528,106 参数全量微调 |
| 产物 | `outputs/20261030-230725_A-003_mobilenetv3_small100_s42/`（`best.pt` 6,249,691 B / `last.pt` 24,833,035 B / `metrics.csv` / `train.log` / `config.yaml`）；评测全文 `outputs/eval_reports/A-003_full.txt`；逐 Study 预测 `outputs/A-003_valid_predictions.csv`（859 行，`Study_id` + `p1..p10` + `label1..label10`） |
| 耗时 | 2026-10-30 23:07:25 → 2026-10-31 00:24:44 CST，共 **1 小时 17 分 19 秒**；17/25 轮（epoch 12 后连续 5 轮无提升触发早停），单轮约 4.5 分钟；显存抽样 1.4–1.6 GB |

metrics.csv（17 轮原样复制；第 17 轮后日志「连续 5 个 epoch 没有提升，提前结束训练」）：

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.1193 | 1.2303 | 0.7113 | 0.2838 |
| 2 | 3.00e-04 | 1.0611 | 1.0137 | 0.7340 | 0.3642 |
| 3 | 3.00e-04 | 0.9964 | 1.0107 | 0.7465 | 0.3844 |
| 4 | 2.99e-04 | 0.9872 | 1.0108 | 0.7497 | 0.3609 |
| 5 | 2.94e-04 | 0.9661 | 1.0383 | 0.7499 | 0.3477 |
| 6 | 2.87e-04 | 0.9557 | 1.0092 | 0.7474 | 0.3649 |
| 7 | 2.77e-04 | 0.9448 | 0.9494 | 0.7719 | 0.4061 |
| 8 | 2.65e-04 | 0.9260 | 0.9377 | 0.7766 | 0.4058 |
| 9 | 2.50e-04 | 0.9182 | 0.9528 | 0.7769 | 0.3912 |
| 10 | 2.33e-04 | 0.9046 | 0.9721 | 0.7710 | 0.3772 |
| 11 | 2.15e-04 | 0.8933 | 0.9895 | 0.7722 | 0.4080 |
| 12 | 1.96e-04 | 0.8782 | 0.9540 | **0.7808** | 0.3881 |
| 13 | 1.76e-04 | 0.8597 | 0.9730 | 0.7728 | 0.3889 |
| 14 | 1.55e-04 | 0.8478 | 0.9617 | 0.7787 | 0.4048 |
| 15 | 1.34e-04 | 0.8217 | 1.0122 | 0.7745 | 0.3979 |
| 16 | 1.14e-04 | 0.8073 | 0.9969 | 0.7714 | 0.3892 |
| 17 | 9.48e-05 | 0.7853 | 1.0317 | 0.7768 | 0.4120 |

**结论**：best epoch 12，macro-AUC **0.7808** / F1@0.5 **0.3881**（`run_eval.py` 重算一致）；
checkpoint metadata 确认 `pretrained_loaded: true`、`total_params: 1,528,106`、`trainable_params: 1,528,106`、
`freeze_backbone: false`。

- 与 A-001（ConvNeXt-Tiny，0.8038 / 0.4255；A-001 为 8 轮上限、早停 2，只作方向性比较）相比：
  AUC **-0.0230**、F1@0.5 **-0.0374**；相对 plan11 的方向性筛选门
  **0.7988** 差 **-0.0180** → **未过门**。
- A-002 中止前没有 `best.pt`，plan11 要求的 A-002 / A-003 配对 `run_compare` 无法执行，也没有生成
  `summary.md` / `summary.csv` / `per_class.csv`；本记录不宣称完成配对比较，未过门也不补 25+5 的
  ConvNeXt-Tiny 基线。
- 曲线形态：epoch 7 起 macro-AUC 进入 0.77–0.78 平台（第 12 轮达峰），train_loss 持续降到 0.785，
  而 valid_loss 自第 12 轮起回升到 1.03、第 13–17 轮无提升 → 过拟合与容量 / 预算限制并存；1.0× 宽度
  （1.53M 参数）相对 0.5×（0.58M）没有表现出追赶 A-001 的势头，轻量 CNN 方向在本口径下不宜替代默认骨干。

阈值曲线（`run_eval.py`；A-001 列为记录值）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-003 | 0.3410 | 0.3601 | 0.3736 | 0.3881 | 0.3965 | 0.3701 |
| A-001 | 0.3641 | 0.3835 | 0.4046 | 0.4255 | 0.4328 | 0.4205 |

A-003 的 F1 峰值在 0.6（0.3965），整条曲线在 0.2–0.7 全段低于 A-001（Δ = -0.0231/-0.0234/-0.0310/-0.0374/-0.0363/-0.0504）。

逐类 AUC（A-001 为记录值，A-003 为本次重算）：

| class_id | support | A-001 | A-003 | Δ（A-003−A-001） |
|---|---|---|---|---|
| 1 | 24 | 0.7438 | 0.7268 | -0.0170 |
| 2 | 37 | 0.8537 | 0.7613 | -0.0924 |
| 3 | 46 | 0.7889 | 0.7588 | -0.0301 |
| 4 | 68 | 0.7374 | 0.7110 | -0.0264 |
| 5 | 115 | 0.8719 | 0.8504 | -0.0215 |
| 6 | 177 | 0.7867 | 0.7741 | -0.0126 |
| 7 | 193 | 0.7801 | 0.7686 | -0.0115 |
| 8 | 214 | 0.7048 | 0.7040 | -0.0008 |
| 9 | 260 | 0.8955 | 0.8891 | -0.0064 |
| 10 | 306 | 0.8754 | 0.8643 | -0.0111 |

10 类**全部低于 A-001**：退化最大的是低频 class2（−0.0924）与 class3 / class4（−0.0301 / −0.0264），
class8 基本持平（−0.0008），高频 class9 / class10 低 −0.0064 / −0.0111。

## B-001：B 组医学预训练 DenseNet-121 对照（224×224 / 单通道 / seed=42，仅研究）

B 组独立研究对照，计划与边界见 [train_plan12.md](train_plan12.md)。仅作研究记录，**不用于赛题提交、模型融合或候选选择**（外部医学影像预训练，违反赛题规则第 1 条）。

- 配置：`configs/exp_B001.yaml`（`_base: default.yaml`），完整展开见 run 目录 `config.yaml`；数据划分、聚合与 seed 与 A 组一致（同 train/valid CSV、`view_weighted` 正位 0.7 / 侧位 0.3、seed 42）。
- 骨干：TorchXRayVision `densenet121-res224-all`（DenseNet-121，6,957,834 参数；18 类分类头替换为 10 类、置空 `op_threshs`，输出原始 logits，先聚合再算 loss），单通道 224×224、letterbox、归一化到 [-1024, 1024]（`image.normalize: xrv`、`image.in_channels: 1`、`image.xrv_maxval: 255`）；全量微调、BCE + `pos_weight`、关闭 `image.hflip`。
- 权重：`https://github.com/mlmed/torchxrayvision/releases/download/v1/nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt`，28,382,008 B，SHA-256 `56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899`，许可证 Apache-2.0（仅覆盖 TxRV 产物，上游数据集另有条款）；预训练数据 5 源（NIH ChestX-ray14 / PadChest / CheXpert / MIMIC-CXR / OpenI）。
- 环境：独立 venv `/root/venv-b001`（Python 3.12、torch 2.8.0+cu128、torchxrayvision 1.5.7、timm 1.0.30），V100-SXM2-32G，`num_workers: 16`。
- 命令：`/root/venv-b001/bin/python src/run_train.py --config configs/exp_B001.yaml --seed 42 --tag B-001_xrv_densenet121_s224`，2026-10-31 11:21:06 启动，12:01 第 11 轮后早停（连续 5 轮无提升），共约 40 分钟、单轮约 3.5 分钟；训练 18,513 studies、验证 859 studies。

| epoch | lr | train_loss | valid_loss | valid_macro_auc | valid_macro_f1 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0027 | 0.9593 | 0.7694 | 0.3794 |
| 2 | 3.00e-04 | 0.9364 | 0.9504 | 0.7781 | 0.3843 |
| 3 | 3.00e-04 | 0.9113 | 0.9131 | 0.7859 | 0.3940 |
| 4 | 2.99e-04 | 0.8941 | 0.9180 | 0.7842 | 0.4052 |
| 5 | 2.94e-04 | 0.8766 | 0.9186 | 0.7876 | 0.3970 |
| 6 | 2.87e-04 | 0.8661 | 0.9029 | **0.7913** | **0.4116** |
| 7 | 2.77e-04 | 0.8518 | 0.9108 | 0.7887 | 0.4073 |
| 8 | 2.65e-04 | 0.8375 | 0.9149 | 0.7867 | 0.4043 |
| 9 | 2.50e-04 | 0.8249 | 0.9270 | 0.7895 | 0.4121 |
| 10 | 2.33e-04 | 0.8063 | 0.9645 | 0.7832 | 0.4077 |
| 11 | 2.15e-04 | 0.7948 | 0.9340 | 0.7841 | 0.4033 |

**结论**：best epoch 6，valid macro-AUC **0.7912** / F1@0.5 **0.4120**（`run_eval.py` 在 valid 859 studies 上重算；`metrics.csv` 的 0.7913 / 0.4116，差异来自阈值口径，≤0.0002）；`best.pt` 28,456,243 B，metadata 确认 `pretrained_loaded: true`、`weights_sha256` 与上表一致、6,957,834 参数全量微调。

- 阈值曲线（`run_eval.py`）：0.2 0.3562 / 0.3 0.3806 / 0.4 0.3965 / 0.5 0.4120 / 0.6 **0.4228（峰值）** / 0.7 0.3848；F1 峰值出现在 0.6，与 A-001（0.6 峰 0.4328）形态一致。
- 逐类 AUC（valid）：class1 0.6881 / class2 0.8113 / class3 0.7895 / class4 0.7433 / class5 0.8632 / class6 0.7913 / class7 0.7697 / class8 0.6909 / class9 0.8956 / class10 0.8694；support 24/37/46/68/115/177/193/214/260/306。
- 与参考 RUN-006（ImageNet DenseNet-121 @320×320、3 通道、imagenet 归一化、seed 42，0.7880 / 0.4070）相比：AUC **+0.0032**、F1@0.5 **+0.0050**；但分辨率（224 vs 320）、输入通道（1 vs 3）、归一化（xrv vs imagenet）与初始化来源同时不同，**差值不可归因于医学预训练**。
- 与 A-001（ConvNeXt-Tiny，0.8038 / 0.4255）不可直接比较：骨干、预训练来源、输入分辨率与归一化均不同，仅记录数值。
- **重叠风险**：本地数据来源不可确证，指纹（JPEG 量化表 q95、PatientOrientation 取值、规模占比）疑似 MIMIC-CXR-JPG 衍生子集；`densenet121-res224-all` 预训练集含 MIMIC-CXR（约 40%）→ 与本地训练集重叠风险**未知（疑似高）**，valid 指标可能乐观，不能声称无泄漏；唯一确证途径是向主办方询问数据来源。
- 产物：`outputs/20261031-112106_B-001_xrv_densenet121_s224/`（`best.pt` 28,456,243 B / `last.pt` 112,857,109 B / `metrics.csv` / `train.log` / `config.yaml`）、`outputs/eval_reports/B-001_full.txt`（阈值曲线与逐类明细）、`outputs/B-001_valid_predictions.csv`（859 个 valid study 的 Study_id / label_1..label_10 / p1..p10）。
- 判读：本组仅回答「同划分、同聚合、同 seed 下医学预训练权重是否带来可复现差异」的前置研究问题；由于上述混杂与重叠风险，**不并入冻结结论、不参与骨干优劣判定、不进入提交候选**。若后续要推进，应先获得数据来源书面确认、再设计分辨率/归一化对齐的单变量实验。


## A-004：A-001 的分类头 dropout 与 weight decay 组合对照（320×320 / seed=42）

背景：A-001 在 ConvNeXt-Tiny @320 上以 8 轮上限 + 早停 2 跑出 valid macro-AUC 0.8038 / F1@0.5
0.4255，但训练与验证损失从 epoch 4 起明显分离（train 0.8466 → 0.7613，valid 0.8856 → 0.9333），
valid_loss 在 epoch 4 见底后反弹。本组验证「分类头 dropout + 更强 weight decay」这组正则化组合能否
推迟过拟合、把 macro-AUC 推过方向性门（A-001 重算值 + 0.005 = 0.8088），计划见 [train_plan13.md](train_plan13.md)。

| 项 | 值 |
|---|---|
| 相对 A-001 差异 | `model.dropout: 0.0 → 0.2`（分类头 Dropout 位于 `ResetClassifier` 的 Linear 之前，作用于 768 维特征，训练启用 / 评估关闭，不是随机深度）；`train.weight_decay: 0.01 → 0.05`（AdamW 全参数同一参数组） |
| 运行命令 | `HF_HUB_OFFLINE=1 .venv/bin/python src/run_train.py --config configs/exp_A004.yaml --seed 42 --tag A-004_convnext_tiny_dropout_wd_s42` |
| 代码版本 | `ad336f2`（`exp/A-004` 分支）：`src/common/config.py` 新增并校验 `model.dropout`，`src/model/network.py` 两处 `timm.create_model` 传入 `drop_rate=cfg.model.dropout`；默认 0，对历史配置逐字段无影响 |
| 设备环境 | autodl 独占 Tesla V100-SXM2-32G（驱动 580.105.08），训练期间无其他 GPU 占用；显存峰值约 14.0 GB（14,346 MiB） |
| 数据 | `data/train.csv` 30,344 图 / 18,513 studies、`data/valid.csv` 1,422 图 / 859 studies；与 A-001 同一份划分，`check_data.py --full` 核验跨 split Subject/Study/Dicom_id 无重叠 |
| 预训练 | `convnext_tiny.fb_in1k`（HF commit `b43a6303…`、114,374,272 B、SHA-256 `08b9dc9c…`、Apache-2.0），与 A-001 共用同一份本地缓存，`HF_HUB_OFFLINE=1` 离线加载；`train.log` 确认「预训练权重: 已加载」，`best.pt` metadata `pretrained_loaded: true` |
| 产物 | `outputs/20261031-121344_A-004_convnext_tiny_dropout_wd_s42/`（`best.pt` 111,385,131 B、`last.pt` 445,536,555 B、`metrics.csv`、`train.log`、`config.yaml` 确认 `dropout: 0.2` / `weight_decay: 0.05`）、`outputs/eval_reports/A-004_full.txt`、`outputs/A-004_valid_predictions.csv`（859 个 valid study）、`outputs/compare_A001_A004/` |
| 耗时 | 32m16s（12:13:44–12:46:00），第 7 轮后早停（上限 8）；单轮约 3 分钟，epoch 1 因首轮冷缓存读盘约 10.9 分钟 |

| epoch | lr | train_loss | valid_loss | valid macro-AUC | valid F1@0.5 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9966 | 0.9469 | 0.7712 | 0.3835 |
| 2 | 3.00e-04 | 0.9407 | 0.9432 | 0.7816 | 0.4033 |
| 3 | 3.00e-04 | 0.8963 | 0.9171 | 0.7862 | 0.4092 |
| 4 | 2.72e-04 | 0.8543 | 0.8990 | 0.7960 | 0.4079 |
| 5 | 2.00e-04 | 0.7769 | 0.9166 | **0.7988** | 0.4232 |
| 6 | 1.10e-04 | 0.6419 | 1.0102 | 0.7969 | 0.4300 |
| 7 | 3.77e-05 | 0.4967 | 1.1973 | 0.7928 | 0.4324 |

第 5 轮后连续 2 轮无提升 → 第 7 轮早停。

**结论**：best epoch 5，valid macro-AUC **0.7988** / F1@0.5 **0.4233**（`run_eval.py` 在 valid 859
studies 上重算，与 `metrics.csv` 的 epoch 5 0.7988 / 0.4232 一致，F1 差 0.0001）；相对 A-001
（0.8038 / 0.4255）差 **AUC −0.0050、F1@0.5 −0.0022**，距方向性门 0.8088 差 **−0.0100 → 未过门**。

阈值曲线（`outputs/compare_A001_A004/summary.md`；A-001 为本次独立重算）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-001 | 0.3641 | 0.3835 | 0.4046 | 0.4255 | **0.4328** | 0.4205 |
| A-004 | 0.3698 | 0.3967 | 0.4127 | 0.4233 | **0.4340** | 0.4178 |
| Δ（A-004 − A-001） | +0.0057 | +0.0132 | +0.0081 | −0.0022 | +0.0012 | −0.0027 |

A-004 在 0.2–0.4 段 F1 高于 A-001、0.5–0.7 段更低，F1 峰值同为 0.6（0.4340 vs 0.4328，+0.0012）→
属阈值附近的校准形态变化，排序指标 macro-AUC 才是主判据。

逐类 AUC（`outputs/compare_A001_A004/per_class.csv`；class_id 1–10 按赛题标签顺序）：

| class_id | support | A-001 | A-004 | Δ |
|---|---|---|---|---|
| 1 | 24 | 0.7438 | 0.7479 | +0.0041 |
| 2 | 37 | 0.8537 | 0.8189 | −0.0348 |
| 3 | 46 | 0.7889 | 0.8165 | +0.0276 |
| 4 | 68 | 0.7374 | 0.7295 | −0.0079 |
| 5 | 115 | 0.8719 | 0.8652 | −0.0067 |
| 6 | 177 | 0.7867 | 0.7681 | −0.0186 |
| 7 | 193 | 0.7801 | 0.7750 | −0.0051 |
| 8 | 214 | 0.7048 | 0.7053 | +0.0005 |
| 9 | 260 | 0.8955 | 0.8918 | −0.0037 |
| 10 | 306 | 0.8754 | 0.8696 | −0.0058 |

逐类 4 升 6 降：class3 +0.0276 提升最大，class2 −0.0348、class6 −0.0186 退化最大；升幅集中在
support 较小的 class1 / class3（24 / 46 例，单类 AUC 的 seed 噪声大），降幅集中在 support 更大的
class2 / class6，与整体 −0.0050 的方向一致。

**判读**：

- 正则化组合没有推迟过拟合：A-004 各轮 train_loss 均高于 A-001（同轮 +0.005~+0.026），valid_loss
  前 4 轮也高于 A-001（+0.006~+0.013）、第 5–6 轮转为略低（−0.017 / −0.009），但整体走势未变，仍在
  epoch 4 见底后上行（0.8990 → 0.9166 → 1.0102 → 1.1973），且第 7 轮即触发早停（A-001 为第 6 轮后
  早停、best epoch 4；A-004 best epoch 5）→ dropout 与 weight decay 只把训练损失水平抬高、没有
  缩小 train/valid 差距。
- 配对 bootstrap（valid 859 studies 按 study 重采样 2,000 次，基于两份逐 study 预测 CSV）：
  ΔAUC（A-001 − A-004）= +0.0051，95% CI [−0.0061, +0.0167]、P(Δ>0) = 0.803；ΔF1@0.5 = +0.0023，
  95% CI [−0.0084, +0.0133] → 差异方向为 A-001 略优，但 CI 跨 0，且 bootstrap 只覆盖 valid 内
  重采样（不含训练随机性）→ 单 seed 下不能判定 A-004 真实退化，可确定的是**没有增益**。
- 口径注意：本组同时改 dropout 与 weight decay 两项，只能评价组合、不能分开归因；单 seed；
  −0.0050 约为 A 组 seed 波动 ±0.0033 的 1.5 倍 → A-001 口径保持不变。

## A-004-a：仅分类头 dropout 0.2 单变量对照（320×320 / seed=42）

背景：A-004 同时打开 dropout 0.2 与 weight_decay 0.05，组合相对 A-001 差 AUC −0.0050，但无法归因到
具体一项。本组只开分类头 dropout、weight_decay 保持 A-001 的 0.01，其余口径逐字段一致，回答
「dropout 单独是否退化、退化多少」，计划见 [train_plan14.md](train_plan14.md)。

| 项 | 值 |
|---|---|
| 相对 A-001 差异 | `model.dropout: 0.0 → 0.2`（分类头 `ResetClassifier` 中 Linear 前的 Dropout，作用于 768 维特征，训练启用 / 评估关闭，不是随机深度）；`train.weight_decay` 保持 0.01 |
| 运行命令 | `HF_HUB_OFFLINE=1 .venv/bin/python src/run_train.py --config configs/exp_A004a.yaml --seed 42 --tag A-004a_dropout_only_s42` |
| 代码版本 | `3231c3f`（`exp/A-004` 分支）：新增 `configs/exp_A004a.yaml`（`_base: exp_A001.yaml` 只覆盖 `model.dropout`）、`docs/train_plan14.md` 与 `tests/test_a004_decomp.py`；模型与训练代码与 A-001 相同 |
| 设备环境 | autodl 独占 Tesla V100-SXM2-32G（驱动 580.105.08），训练期间无其他 GPU 占用；显存占用与 A-004 同量级（约 14 GB） |
| 数据 | `data/train.csv` 30,344 图 / 18,513 studies、`data/valid.csv` 1,422 图 / 859 studies；与 A-001 同一份划分，`check_data.py --full` 通过 |
| 预训练 | `convnext_tiny.fb_in1k`（与 A-001 共用同一份本地缓存，HF commit `b43a6303…`、SHA-256 `08b9dc9c…`），`HF_HUB_OFFLINE=1` 离线加载；`train.log` 确认「预训练权重: 已加载」 |
| 产物 | `outputs/20261031-130923_A-004a_dropout_only_s42/`（`best.pt` 111,385,131 B、`last.pt` 445,536,555 B、`metrics.csv`、`train.log`）、`outputs/eval_reports/A-004a_full.txt`、`outputs/A-004a_valid_predictions.csv`（859 个 valid study）、`outputs/compare_A001_A004_decomp/` |
| 耗时 | 37m16s（13:09:23–13:46:39）；跑满 8 轮上限，第 6 轮达峰后第 7、8 轮无提升，训练在 epoch 8 结束时按早停逻辑收尾 |

| epoch | lr | train_loss | valid_loss | valid macro-AUC | valid F1@0.5 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9944 | 0.9523 | 0.7673 | 0.3818 |
| 2 | 3.00e-04 | 0.9381 | 0.9514 | 0.7761 | 0.3993 |
| 3 | 3.00e-04 | 0.8940 | 0.9099 | 0.7897 | 0.4120 |
| 4 | 2.72e-04 | 0.8494 | 0.9158 | 0.7888 | 0.4143 |
| 5 | 2.00e-04 | 0.7656 | 0.9305 | 0.7939 | 0.4173 |
| 6 | 1.10e-04 | 0.6221 | 1.1141 | **0.7944** | 0.4356 |
| 7 | 3.77e-05 | 0.4801 | 1.3354 | 0.7907 | 0.4253 |
| 8 | 1.00e-05 | 0.4175 | 1.4158 | 0.7889 | 0.4206 |

**结论**：best epoch 6，valid macro-AUC **0.7944** / F1@0.5 **0.4360**（`run_eval.py` 在 valid 859
studies 上重算，与 `metrics.csv` 的 epoch 6 0.7944 / 0.4356 一致，F1 差 0.0004）；相对 A-001
（0.8038 / 0.4255）差 **AUC −0.0094、F1@0.5 +0.0105**，距方向性门 0.8088 差 **−0.0144 → 未过门**。

阈值曲线（`outputs/compare_A001_A004_decomp/summary.md`；A-001 为 A-004 阶段独立重算值）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-001 | 0.3641 | 0.3835 | 0.4046 | 0.4255 | **0.4328** | 0.4205 |
| A-004-a | 0.4090 | 0.4234 | **0.4400** | 0.4360 | 0.4254 | 0.3992 |
| Δ（A-004-a − A-001） | +0.0449 | +0.0399 | +0.0354 | +0.0105 | −0.0074 | −0.0213 |

A-004-a 的 F1 峰从 A-001 的 0.6 左移到 0.4（0.4400 vs 0.4328，+0.0072），0.2–0.5 段全段高于 A-001、
0.6–0.7 段更低 → dropout 只改变阈值附近的校准形态，排序指标 macro-AUC 反而变低。

逐类 AUC（`outputs/compare_A001_A004_decomp/per_class.csv`；class_id 1–10 按赛题标签顺序）：

| class_id | support | A-001 | A-004-a | Δ |
|---|---|---|---|---|
| 1 | 24 | 0.7438 | 0.7397 | −0.0041 |
| 2 | 37 | 0.8537 | 0.8136 | −0.0401 |
| 3 | 46 | 0.7889 | 0.7655 | −0.0234 |
| 4 | 68 | 0.7374 | 0.7214 | −0.0160 |
| 5 | 115 | 0.8719 | 0.8589 | −0.0130 |
| 6 | 177 | 0.7867 | 0.7702 | −0.0165 |
| 7 | 193 | 0.7801 | 0.7889 | +0.0088 |
| 8 | 214 | 0.7048 | 0.7066 | +0.0018 |
| 9 | 260 | 0.8955 | 0.9026 | +0.0071 |
| 10 | 306 | 0.8754 | 0.8768 | +0.0014 |

逐类 5 升 5 降：class2 −0.0401、class3 −0.0234 退化最大，class7 +0.0088、class9 +0.0071 提升最大；
退化集中在 class2–class6 的中低支持类，与整体 −0.0094 一致。

**判读**：

- dropout 单独是主要负向项：AUC −0.0094 约 2.8 倍 A 组 seed 波动（±0.0033）；配对 bootstrap
  （valid 859 studies 重采样 2,000 次）ΔAUC（A-001 − A-004-a）= +0.0094，95% CI [−0.0027, +0.0215]、
  P(Δ>0) = 0.934 → CI 仍跨 0，单 seed 不能判定统计显著，但方向与 A-004 的 −0.0050 一致。
- 没有推迟过拟合：valid_loss 第 1–6 轮均高于 A-001（+0.011 / +0.020 / +0.004 / +0.030 / −0.003 /
  +0.095），第 4 轮见底 0.9158 后仍反弹到 1.4158；best epoch 6 晚于 A-001 的 4，但峰值更低。
- 口径注意：单 seed；与 A-004 一样，dropout 只作用于 768 维分类头特征（不是随机深度），
  因此「dropout 无效」不能外推到骨干内的 stochastic depth。

## A-004-b：仅 weight decay 0.05 单变量对照（320×320 / seed=42）

背景：A-004 同时打开 dropout 0.2 与 weight_decay 0.05，组合相对 A-001 差 AUC −0.0050。本组只把
`train.weight_decay` 提到 0.05、dropout 保持 0，其余口径与 A-001 逐字段一致，回答「weight decay
单独是否退化」，与 A-004-a 一起把组合差拆分到两项。计划见 [train_plan14.md](train_plan14.md)。

| 项 | 值 |
|---|---|
| 相对 A-001 差异 | `train.weight_decay: 0.01 → 0.05`（AdamW 全参数同一参数组）；`model.dropout` 保持 0.0 |
| 运行命令 | `HF_HUB_OFFLINE=1 .venv/bin/python src/run_train.py --config configs/exp_A004b.yaml --seed 42 --tag A-004b_weight_decay_only_s42` |
| 代码版本 | `3231c3f`（`exp/A-004` 分支）：新增 `configs/exp_A004b.yaml`（`_base: exp_A001.yaml` 只覆盖 `train.weight_decay`）、`docs/train_plan14.md` 与 `tests/test_a004_decomp.py`；模型与训练代码与 A-001 相同 |
| 设备环境 | autodl 独占 Tesla V100-SXM2-32G（驱动 580.105.08），训练期间无其他 GPU 占用；训练中观测显存 14,350 MiB |
| 数据 | `data/train.csv` 30,344 图 / 18,513 studies、`data/valid.csv` 1,422 图 / 859 studies；与 A-001 同一份划分 |
| 预训练 | `convnext_tiny.fb_in1k`（与 A-001 共用同一份本地缓存，HF commit `b43a6303…`、SHA-256 `08b9dc9c…`），`HF_HUB_OFFLINE=1` 离线加载；`train.log` 确认「预训练权重: 已加载」 |
| 产物 | `outputs/20261031-134732_A-004b_weight_decay_only_s42/`（`best.pt` 111,385,131 B、`last.pt` 445,536,555 B、`metrics.csv`、`train.log`）、`outputs/eval_reports/A-004b_full.txt`、`outputs/A-004b_valid_predictions.csv`（859 个 valid study）、`outputs/compare_A001_A004_decomp/` |
| 耗时 | 31m59s（13:47:32–14:19:31），第 7 轮后早停（上限 8） |

| epoch | lr | train_loss | valid_loss | valid macro-AUC | valid F1@0.5 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9920 | 0.9489 | 0.7699 | 0.3833 |
| 2 | 3.00e-04 | 0.9350 | 0.9389 | 0.7814 | 0.4049 |
| 3 | 3.00e-04 | 0.8898 | 0.9224 | 0.7849 | 0.4085 |
| 4 | 2.72e-04 | 0.8480 | 0.9126 | 0.7936 | 0.4133 |
| 5 | 2.00e-04 | 0.7654 | 0.9153 | **0.7999** | 0.4240 |
| 6 | 1.10e-04 | 0.6254 | 1.0621 | 0.7910 | 0.4427 |
| 7 | 3.77e-05 | 0.4811 | 1.2732 | 0.7850 | 0.4363 |

第 5 轮后连续 2 轮无提升 → 第 7 轮早停。

**结论**：best epoch 5，valid macro-AUC **0.7999** / F1@0.5 **0.4238**（`run_eval.py` 在 valid 859
studies 上重算，与 `metrics.csv` 的 epoch 5 0.7999 / 0.4240 一致，F1 差 0.0002）；相对 A-001
（0.8038 / 0.4255）差 **AUC −0.0039、F1@0.5 −0.0017**，距方向性门 0.8088 差 **−0.0089 → 未过门**。

阈值曲线（`outputs/compare_A001_A004_decomp/summary.md`；A-001 为 A-004 阶段独立重算值）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-001 | 0.3641 | 0.3835 | 0.4046 | 0.4255 | **0.4328** | 0.4205 |
| A-004-b | 0.3678 | 0.3878 | 0.4085 | 0.4238 | **0.4364** | 0.4278 |
| Δ（A-004-b − A-001） | +0.0037 | +0.0043 | +0.0039 | −0.0017 | +0.0036 | +0.0073 |

A-004-b 的 F1 峰仍在 0.6（0.4364 vs 0.4328，+0.0036），0.2–0.4 与 0.6–0.7 段略高于 A-001、0.5 段略低
→ 形态与 A-001 相同，只有小幅校准偏移。

逐类 AUC（`outputs/compare_A001_A004_decomp/per_class.csv`；class_id 1–10 按赛题标签顺序）：

| class_id | support | A-001 | A-004-b | Δ |
|---|---|---|---|---|
| 1 | 24 | 0.7438 | 0.7262 | −0.0176 |
| 2 | 37 | 0.8537 | 0.8162 | −0.0375 |
| 3 | 46 | 0.7889 | 0.8077 | +0.0188 |
| 4 | 68 | 0.7374 | 0.7452 | +0.0078 |
| 5 | 115 | 0.8719 | 0.8650 | −0.0069 |
| 6 | 177 | 0.7867 | 0.7792 | −0.0075 |
| 7 | 193 | 0.7801 | 0.7732 | −0.0069 |
| 8 | 214 | 0.7048 | 0.7214 | +0.0166 |
| 9 | 260 | 0.8955 | 0.8935 | −0.0020 |
| 10 | 306 | 0.8754 | 0.8715 | −0.0039 |

逐类 4 升 6 降：class3 +0.0188、class8 +0.0166、class4 +0.0078 提升，class2 −0.0375、class1 −0.0176
退化最大；高支持类 class9 / class10 基本持平（−0.0020 / −0.0039）。

**判读**：

- weight decay 单独接近噪声：AUC −0.0039 在 A 组 seed 波动（±0.0033）量级；配对 bootstrap
  （valid 859 studies 重采样 2,000 次）ΔAUC（A-001 − A-004-b）= +0.0039，95% CI [−0.0056, +0.0138]、
  P(Δ>0) = 0.785；F1@0.5 −0.0017，F1 峰仍在 0.6（0.4364，+0.0036）。
- 没有推迟过拟合：train_loss 每轮仅高 A-001 0.0004–0.0093，valid_loss 前 4 轮高 0.007–0.027、
  第 5 轮低 0.018，第 4 轮见底 0.9126 后仍反弹到 1.2732；best epoch 5 与 A-004 相同。
- 归因：两项单独差（−0.0094 / −0.0039）的算术平均为 −0.0067，组合实测 −0.0050 略好于线性平均
  → 无协同放大；配对 bootstrap 中 A-004 vs A-004-a 的 ΔAUC = +0.0043（95% CI [−0.0088, +0.0169]、
  P = 0.737）、A-004 vs A-004-b = −0.0012（95% CI [−0.0092, +0.0068]、P = 0.378）→ **组合的 AUC 水平
  主要由 weight decay 决定，dropout 是主要负向项**。
- 口径注意：单 seed；方向性门 0.8088 两组均未过；若要继续正则化方向，应先补 seed 2026 配对，
  再考虑更高/更低档位或与数据增强组合，不在单 seed 上继续扫参。

## A-005：CSRA 分类头对照（320×320 / seed=42）

A-006 同批的分类头结构对照：在 A-001 口径下把 timm 原分类头换成单头 CSRA（类别空间分数经
L2 归一化后按残差权重 0.4 加权，温度 1.0，无 bias）。与 A-006 共用同一配对基线
`A-005_baseline`（标准头同版本重训），两者可直接横向比较。

| 项 | 值 |
| --- | --- |
| 相对 A-001 差异 | 仅分类头：`model.head: csra`（`csra_residual_weight: 0.4`、`csra_temperature: 1.0`）；其余逐字段同 `exp_A001.yaml` |
| 分类头结构 | 单头 CSRA：每类空间均值 + 0.4 × 空间注意力加权分数，池化前按位置 L2 归一化；类别投影与 bias 均与标准头不同；27,827,808 参数（标准头 27,827,818，少 10） |
| 运行命令 | `HF_HUB_OFFLINE=1 .venv/bin/python src/run_train.py --config configs/exp_csra.yaml --seed 42 --tag A-005` |
| 代码版本 | `exp/A-004` @ `b5b14d9`，工作区干净（启动前 `git status --short` 无输出） |
| 设备环境 | autodl 独占 Tesla V100-SXM2-32G（驱动 580.105.08）；峰值显存未单独记录，与标准头同量级（A-006 同类记录 11,906 MiB） |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 同一份 `data/valid.csv`） |
| 预训练 | `convnext_tiny.fb_in1k`，SHA-256 `08b9dc9c…`；`metadata.pretrained_loaded: true`、`freeze_backbone: false` |
| 产物 | `outputs/20261031-164123_A-005/`（`best.pt` 111,384,739 B、`last.pt` 445,548,435 B）、`outputs/A-005_valid.csv`、`outputs/A-005_baseline_valid.csv`、`outputs/compare_A005/`、`outputs/compare_A001_A005/`、`outputs/plan15_bootstrap.py` |
| 配对基线 | `outputs/20261031-151114_A-005_baseline_s42`：15:11:14–15:40:35、29m21s、best epoch 4、macro-AUC 0.8022 / F1@0.5 0.4210（与 A-006 共用） |
| 耗时 | 24m28s（16:41:23–17:05:51，独占，未并行） |

逐轮：

| epoch | lr | train_loss | valid_loss | valid macro-AUC | valid F1@0.5 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.50e-04 | 0.9887 | 0.9349 | 0.7717 | 0.3987 |
| 2 | 3.00e-04 | 0.9400 | 0.9138 | 0.7870 | 0.4131 |
| 3 | 3.00e-04 | 0.8924 | 0.9035 | **0.7936** | 0.4050 |
| 4 | 2.72e-04 | 0.8540 | 0.9084 | 0.7912 | 0.4121 |
| 5 | 2.00e-04 | 0.7726 | 0.9403 | 0.7889 | 0.4219 |

第 5 轮触发早停（连续 2 轮无提升），best epoch 3；`run_eval.py` 重算 valid macro-AUC 0.7936 /
F1@0.5 0.4050，与训练输出一致。

阈值曲线：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
| --- | --- | --- | --- | --- | --- | --- |
| 配对基线 F1 | 0.3594 | 0.3819 | 0.4020 | 0.4210 | 0.4383 | 0.4262 |
| A-005 F1 | 0.3400 | 0.3691 | 0.3931 | 0.4050 | 0.4191 | 0.3527 |
| ΔF1 | −0.0194 | −0.0128 | −0.0089 | −0.0160 | −0.0192 | −0.0735 |

逐类 AUC：

| class_id | support | 配对基线 | A-005 | Δ |
| --- | --- | --- | --- | --- |
| 1 | 24 | 0.7311 | 0.7207 | −0.0105 |
| 2 | 37 | 0.8611 | 0.8417 | −0.0194 |
| 3 | 46 | 0.7939 | 0.7628 | −0.0310 |
| 4 | 68 | 0.7248 | 0.7196 | −0.0052 |
| 5 | 115 | 0.8741 | 0.8680 | −0.0061 |
| 6 | 177 | 0.7835 | 0.7769 | −0.0066 |
| 7 | 193 | 0.7787 | 0.7762 | −0.0025 |
| 8 | 214 | 0.7076 | 0.6999 | −0.0077 |
| 9 | 260 | 0.8964 | **0.9041** | +0.0077 |
| 10 | 306 | 0.8710 | 0.8657 | −0.0053 |

配对 bootstrap（`outputs/plan15_bootstrap.py`，对 859 个 study 有放回重采样 2,000 次、seed
2026、阈值 0.5、单类正负样本缺失时跳过该次）：ΔAUC（配对基线 − A-005）= +0.0086、95% CI
[−0.0030, +0.0210]、P(Δ>0)=0.928；ΔF1@0.5 = +0.0161、95% CI [+0.0061, +0.0266]、
P=0.998（2,000 / 2,000 次有效）。对历史 A-001（0.8038 / 0.4255）：ΔAUC = +0.0103、95% CI
[−0.0010, +0.0217]、P=0.962；ΔF1@0.5 = +0.0205、95% CI [+0.0105, +0.0303]、P=1.000。

**判读**

- 方向性门：配对基线 + 0.005 = **0.8072**；A-005 差 **−0.0136**（对历史 A-001 参考门 0.8088
  差 −0.0152）→ **未过门**。
- 配对 bootstrap：ΔAUC 的 95% CI 跨 0、P=0.928 未达 0.975 判据；只有 ΔF1@0.5 的 CI 整体为
  正 → 排序能力无可靠提升，F1@0.5 的下降有统计支持。
- 学习曲线：峰值更早（epoch 3 vs 基线 4）且更低；valid_loss 第 3 轮见底 0.9035 后反弹到
  0.9403（基线为第 4 轮见底 0.9084 后反弹）→ 相同预算下更早触及容量上限或过拟合，未推迟
  过拟合。
- 逐类形态：10 类中仅 class9 上升（+0.0077），9 类下降，低支持类退化最大（class3 −0.0310、
  class2 −0.0194、class1 −0.0105）→ 注意力残差头没有改善稀疏类别。
- 资源：参数量与标准头基本持平（−10），单轮耗时同量级，未单独记录峰值显存。
- 结论：CSRA 未过门 → **`model.head: standard` 保持默认，不切换、不补 seed 2026 复核、
  不进入候选提交**。与 A-006（ML-Decoder 0.7900）合并看，两个分类头结构候选均低于同版本
  标准头配对基线 0.8022，分类头结构方向否定。
- 口径注意：A-005 与 A-006 共用 A-005_baseline；A-006 训练 15:57–16:29、A-005 训练
  16:41–17:05，二者串行且各自独占 V100，无并发干扰。

## A-006：轻量 ML-Decoder 分类头对照（320×320 / seed=42）

背景：A-001 的 timm 标准分类头（LayerNorm → Dropout → Linear 768→10）在 ConvNeXt-Tiny @320 上达到
0.8038 / 0.4255。本组保持数据、预处理、聚合、损失、优化器、轮数与 seed 全部不变，只把分类头换成
轻量 ML-Decoder（10 个类别查询 + 单层 4 头交叉注意力 + 逐类权重向量，内部维度 256），检验「用类别
查询从特征图自适应抽取类证据」能否改善 study 级 Macro-AUC。计划见 [train_plan16.md](train_plan16.md)，
结构见 [ml_decoder_implementation.md](ml_decoder_implementation.md)。

| 项 | 值 |
|---|---|
| 相对 A-001 差异 | `model.head: standard → ml_decoder`（`configs/exp_ml_decoder.yaml` 以 `_base: exp_A001.yaml` 只覆盖该字段；展开配置逐字段核对仅此一项不同） |
| 分类头结构 | `src/model/ml_decoder.py`：ConvNeXt 768 维特征图 → `project` Linear(768→256) → 10 个可学习类别查询（10×256）对特征序列做单层 4 头交叉注意力 → LayerNorm → 逐类权重向量（10×256）点积 + `class_bias`(10) → 10 类 logits；参数量 27,827,818 → **28,549,226**（+721,408，+2.6%） |
| 运行命令 | `HF_HUB_OFFLINE=1 .venv/bin/python src/run_train.py --config configs/exp_ml_decoder.yaml --seed 42 --tag A-006_ml_decoder_s42` |
| 代码版本 | `97505f0`（`exp/A-004` 分支）：新增 `src/model/ml_decoder.py`、`configs/exp_ml_decoder.yaml`、`tests/test_ml_decoder.py`、`docs/ml_decoder_implementation.md` 与 `docs/train_plan16.md`；`src/model/network.py` 按 `model.head` 分派注意力头，预训练加载失败直接报错（不回退随机初始化） |
| 设备环境 | autodl 独占 Tesla V100-SXM2-32GB（驱动 580.105.08），训练期间无其他 GPU 占用；训练中观测显存 11,906 MiB |
| 数据 | `data/train.csv` 30,344 图 / 18,513 studies、`data/valid.csv` 1,422 图 / 859 studies；与 A-001 同一份划分，执行前 `check_data.py --full` 复核图片齐全、三划分 Subject/Study/Dicom 零重叠，CSV SHA-256 与 A-001 记录一致 |
| 预训练 | `convnext_tiny.fb_in1k`（本地缓存 HF commit `b43a6303…`、safetensors 114,374,272 B、SHA-256 `08b9dc9c…`，与 A-001 记录一致），`HF_HUB_OFFLINE=1` 离线加载；`train.log` 与 checkpoint metadata 均确认 `pretrained_loaded: true` |
| 产物 | `outputs/20261031-155701_A-006_ml_decoder_s42/`（`best.pt` 114,276,003 B、`last.pt` 457,115,795 B、`metrics.csv`、`train.log`、`config.yaml`）、`outputs/eval_reports/A-006_decoder_full.txt`、`outputs/A-006_decoder_valid.csv`（859 个 valid study）、`outputs/compare_A006/`（`summary.md` / `per_class.csv` / `bootstrap_A006.csv`） |
| 配对基线 | `outputs/20261031-151114_A-005_baseline_s42/`（tag `A-005_baseline_s42`）：`exp_A001.yaml` 标准头同版本重训，15:11:14–15:40:35（29m21s）、第 6 轮早停（best epoch 4）、valid macro-AUC **0.8022** / F1@0.5 **0.4210**、27,827,818 参数、`best.pt` 111,385,195 B（目录内未保留 `last.pt`）；与 A-006 同数据、同设备、同预算，训练时工作区为 `50620f3` + 未提交改动（备份 `/root/plan16_backup/wip.patch`，与 `97505f0` 已跟踪 src 改动仅差「允许 `model.head=ml_decoder`」及新增文件），standard 头代码路径一致 |
| 耗时 | 32m39s（15:57:01–16:29:40），第 7 轮后早停（上限 8） |

| epoch | lr | train_loss | valid_loss | valid macro-AUC | valid F1@0.5 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0128 | 0.9758 | 0.7533 | 0.3823 |
| 2 | 3.00e-04 | 0.9711 | 0.9730 | 0.7616 | 0.3819 |
| 3 | 3.00e-04 | 0.9300 | 0.9164 | 0.7799 | 0.4122 |
| 4 | 2.72e-04 | 0.8940 | 0.9164 | 0.7848 | 0.4110 |
| 5 | 2.00e-04 | 0.8394 | 0.9218 | **0.7900** | 0.4118 |
| 6 | 1.10e-04 | 0.7518 | 0.9849 | 0.7861 | 0.4215 |
| 7 | 3.77e-05 | 0.6506 | 1.0157 | 0.7861 | 0.4353 |

第 5 轮后连续 2 轮无提升 → 第 7 轮早停；valid_loss 自第 5 轮起反弹（0.9218 → 0.9849 → 1.0157）。

**结论**：best epoch 5，valid macro-AUC **0.7900** / F1@0.5 **0.4118**（`run_eval.py` 在 valid 859
studies 上重算，与 `metrics.csv` 的 epoch 5 一致）；相对配对基线（0.8022 / 0.4210）差 **AUC −0.0122、
F1@0.5 −0.0092**，距方向性门 0.8072 差 **−0.0172 → 未过门**；相对历史 A-001（0.8038 / 0.4255）差
AUC −0.0138、F1@0.5 −0.0137。

阈值曲线（`outputs/compare_A006/summary.md`；配对基线为同版本重训、本轮独立重算值）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| 配对基线 | 0.3594 | 0.3819 | 0.4020 | 0.4210 | **0.4383** | 0.4262 |
| A-006 | 0.3612 | 0.3811 | 0.3999 | 0.4118 | **0.4173** | 0.4018 |
| Δ（A-006 − 基线） | +0.0018 | −0.0008 | −0.0021 | −0.0092 | −0.0210 | −0.0244 |

A-006 的 F1 峰仍在 0.6（0.4173 vs 0.4383，−0.0210），0.2 段略高、0.3 以上逐段走低 → 头替换没有改变
阈值形态，只是整体区分度下降。

逐类 AUC（`outputs/compare_A006/per_class.csv`；class_id 1–10 按赛题标签顺序；support 与基线一致）：

| class_id | support | 配对基线 | A-006 | Δ |
|---|---|---|---|---|
| 1 | 24 | 0.7311 | 0.6752 | −0.0559 |
| 2 | 37 | 0.8611 | 0.8156 | −0.0455 |
| 3 | 46 | 0.7939 | 0.8036 | +0.0097 |
| 4 | 68 | 0.7248 | 0.7097 | −0.0151 |
| 5 | 115 | 0.8741 | 0.8670 | −0.0071 |
| 6 | 177 | 0.7835 | 0.7807 | −0.0028 |
| 7 | 193 | 0.7787 | 0.7863 | +0.0076 |
| 8 | 214 | 0.7076 | 0.7041 | −0.0035 |
| 9 | 260 | 0.8964 | 0.8929 | −0.0035 |
| 10 | 306 | 0.8710 | 0.8649 | −0.0061 |

逐类 2 升 8 降：class3 +0.0097、class7 +0.0076 小幅提升，class1 −0.0559、class2 −0.0455、class4
−0.0151 退化最大；退化集中在低支持度类（class1 / class2 / class4 的正例数分别 24 / 37 / 68），
高支持类 class9 / class10 基本持平（−0.0035 / −0.0061）。

**判读**：

- 方向性门：0.7900 距门 0.8072 差 −0.0172（约为 A 组 seed 波动 ±0.0033 的 5.2 倍）→ 明确未过门；
  相对 A-001 也只低不高（−0.0138）。
- 配对 bootstrap（`outputs/plan16_bootstrap.py`；859 studies 按 study 有放回重采样 2,000 次、
  seed 2026、退化类跳过，2,000 次均保持 10 类有效）：ΔAUC（基线 − A-006）= **+0.0123**，
  95% CI **[+0.0001, +0.0245]**、P(Δ>0) = **0.976**；ΔF1@0.5 = +0.0092，95% CI [−0.0018, +0.0205]、
  P = 0.952 → AUC 差的可信区间整体在 0 以上（标准头更优），F1@0.5 差方向一致但区间跨 0；该检验
  只覆盖 valid 内重采样、不含训练随机性，且单 seed。
- 学习曲线：收敛更慢——epoch 1 只有 0.7533（基线 0.7736），峰值更晚（epoch 5 vs 基线 epoch 4）、
  峰值更低（0.7900 vs 0.8022）；valid_loss 第 5 轮见底 0.9218 后反弹到 1.0157，train_loss 仍降到
  0.6506 → 与基线相同的「AUC 见顶后过拟合」形态，注意力头没有把过拟合推迟，只是在预算内没追上。
- 逐类形态：低支持度类退化最严重，高支持类基本持平 → 10 个类别查询共享同一套注意力参数，在
  class1 / class2 这类极稀疏类上缺少足够的梯度信号（假设，未单独验证）；若继续该方向，应先解决
  类别查询的冷启动问题，而不是在单 seed 上继续改结构。
- 资源：参数量 +721,408（+2.6%），实测显存 11.9 GB 与基线同量级（注意力头相对骨干开销可忽略）；
  耗时 32m39s vs 基线 29m21s（+11%，含首轮索引构建差异）。
- 结论：**分类头方向否定**——`model.head: standard` 保持默认，不做 seed 2026 配对复核、不进入
  候选提交；本组唯一变量是分类头（同数据、同设备、同预算、同 seed），差值可归因头结构。
- 口径注意：单 seed；A-006 与配对基线使用同一 valid 选型（可能高估收益，本组为负向结论，该偏差
  不影响「未过门」判断）；history 原始值见 `metrics.csv`。

## A-007：ConvNeXt-Small 骨干与 ImageNet-22k 预训练权重对照（320×320 / seed=42）

背景：A-001 的 ConvNeXt-Tiny（`fb_in1k`）@320 在 8 轮预算内达到 0.8038 / 0.4255，是 A 组当前最高
的 valid 水平。本组保持数据、预处理、聚合、损失、优化器、轮数与 seed 全部不变，只把骨干换成
ConvNeXt-Small 并改用 `fb_in22k_ft_in1k` 权重（ImageNet-22k 预训练后在 ImageNet-1k 微调），检验
「更大骨干 + 更强预训练」能否在训练预算不变的前提下提升 study 级 Macro-AUC。计划见
[train_plan17.md](train_plan17.md)。注意这是「骨干结构 + 预训练数据」同时变的混合对照，不能把差异
单变量归因于结构或权重之一。

| 项 | 值 |
|---|---|
| 相对 A-001 差异 | `model.backbone: convnext_tiny.fb_in1k → convnext_small.fb_in22k_ft_in1k`（`configs/exp_A007.yaml` 以 `_base: exp_A001.yaml` 只覆盖该字段；展开配置逐字段核对仅此一项不同） |
| 模型规模 | ConvNeXt-Small + 10 类头 **49,462,378** 参数（A-001 的 ConvNeXt-Tiny 为 27,827,818，+21,634,560 / +77.7%）；checkpoint metadata 记录 `total_params: 49462378`、`trainable_params: 49462378`（全量微调）、`pretrained_loaded: true` |
| 运行命令 | `HF_HUB_OFFLINE=1 .venv/bin/python src/run_train.py --config configs/exp_A007.yaml --seed 42 --tag A-007_convnext_small_in22k_ft_s42` |
| 代码版本 | `ea555c0`（`exp/A-007` 分支）：新增 `configs/exp_A007.yaml`、`docs/train_plan17.md` 与 CHANGELOG 条目；训练/模型/数据代码与 A-001 相同（`_base` 继承，仅骨干字段不同） |
| 设备环境 | autodl 独占 Tesla V100-SXM2-32GB（驱动 580.105.08），训练期间无其他 GPU 占用；训练中 `nvidia-smi` 抽样显存峰值 14,702 MiB（A-001 记录约 11.7 GB，+约 3.0 GB） |
| 数据 | `data/train.csv` 30,344 图 / 18,513 studies、`data/valid.csv` 1,422 图 / 859 studies；与 A-001 同一份划分，执行前 `check_data.py --full` 复核图片齐全、三划分 Subject/Study/Dicom 零重叠 |
| 预训练 | `convnext_small.fb_in22k_ft_in1k`（训练机本地缓存 HF commit `fe6b358a…`、`model.safetensors` 200,928,946 B、SHA-256 `2457488b…`；来源 `timm/convnext_small.fb_in22k_ft_in1k`），`HF_HUB_OFFLINE=1` 离线加载；`train.log` 与 checkpoint metadata 均确认 `pretrained_loaded: true` |
| 产物 | `outputs/20261031-172305_A-007_convnext_small_in22k_ft_s42/`（`best.pt` 197,987,899 B、`last.pt` 791,962,333 B、`metrics.csv`、`train.log`、`config.yaml`）、`outputs/eval_reports/A-007_full.txt`、`outputs/A-007_convnext_small_valid.csv`（859 个 valid study）、`outputs/compare_A007/`（`summary.md` / `per_class.csv` / `bootstrap_A007.csv` / `bootstrap_A007.log`） |
| 配对基线 | A-001（`outputs/20261030-150749_A-001_convnext_tiny_s42/`，0.8038 / 0.4255；同数据、同设备、同预算、同 seed 42）。本组只改配置 YAML，训练代码路径与 A-001 完全一致（`_base` 逐字段继承），因此直接以 A-001 记录值作对照、复用其 valid 预测 `outputs/A-001_ref_valid.csv` 做配对，不另重训基线 |
| 耗时 | 37m03s（17:23:05–18:00:08），8/8 轮跑满后第 8 轮触发早停（best epoch 6），平均约 4.6 分钟/轮（含启动与首轮数据索引开销） |

| epoch | lr | train_loss | valid_loss | valid macro-AUC | valid F1@0.5 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 1.0074 | 0.9870 | 0.7452 | 0.3799 |
| 2 | 3.00e-04 | 0.9544 | 0.9699 | 0.7672 | 0.3910 |
| 3 | 3.00e-04 | 0.9238 | 0.9950 | 0.7782 | 0.3795 |
| 4 | 2.72e-04 | 0.8834 | 0.9371 | 0.7837 | 0.4070 |
| 5 | 2.00e-04 | 0.8208 | 0.9267 | 0.7883 | 0.4131 |
| 6 | 1.10e-04 | 0.7023 | 0.9885 | **0.7903** | 0.4209 |
| 7 | 3.77e-05 | 0.5453 | 1.3911 | 0.7863 | 0.4361 |
| 8 | 1.00e-05 | 0.4471 | 1.8042 | 0.7776 | 0.4333 |

第 6 轮后连续 2 轮无提升 → 第 8 轮结束触发早停（正好跑满上限 8 轮，非提前终止）；valid_loss 第 5
轮见底 0.9267 后反弹到 1.3911 / 1.8042，A-001 同期为第 4 轮见底 0.8856 → 0.9333 / 1.0194，过拟合
明显更陡。

**结论**：best epoch 6，valid macro-AUC **0.7903** / F1@0.5 **0.4204**（`run_eval.py` 在 valid 859
studies 上重算；`metrics.csv` 的 epoch 6 为 0.7903 / 0.4209，F1 差 0.0005）；相对 A-001
（0.8038 / 0.4255）差 **AUC −0.0135、F1@0.5 −0.0051**，距方向性门 0.8088 差 **−0.0185 → 未过门**。
逐轮 AUC 全部低于 A-001 同轮（差值 −0.0133～−0.0286），收敛更慢（峰值 epoch 6 vs 4）且更低。

阈值曲线（`outputs/compare_A007/summary.md`；A-001 与 A-007 均本轮独立重算）：

| threshold | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|
| A-001 | 0.3641 | 0.3835 | 0.4046 | 0.4255 | **0.4328** | 0.4205 |
| A-007 | 0.3750 | 0.3957 | 0.4112 | 0.4204 | **0.4225** | 0.4097 |
| Δ（A-007 − A-001） | +0.0109 | +0.0122 | +0.0067 | −0.0051 | −0.0102 | −0.0108 |

A-007 的 F1 峰仍在 0.6（0.4225 vs A-001 0.4328，−0.0102），0.2–0.4 段略高、0.5 以上逐段走低 →
与 A-001 相同的阈值形态，只是高阈值段区分度下降；低阈值段的 +0.011～+0.012 只是校准形态差异。

逐类 AUC（`outputs/compare_A007/per_class.csv`；class_id 1–10 按赛题标签顺序；support 与 A-001 一致）：

| class_id | support | A-001 | A-007 | Δ（A-007 − A-001） |
|---|---|---|---|---|
| 1 | 24 | 0.7438 | 0.7368 | −0.0069 |
| 2 | 37 | 0.8537 | 0.7458 | −0.1079 |
| 3 | 46 | 0.7889 | 0.7895 | +0.0007 |
| 4 | 68 | 0.7374 | 0.7227 | −0.0147 |
| 5 | 115 | 0.8719 | 0.8749 | +0.0031 |
| 6 | 177 | 0.7867 | 0.7753 | −0.0114 |
| 7 | 193 | 0.7801 | 0.7767 | −0.0035 |
| 8 | 214 | 0.7048 | 0.7080 | +0.0032 |
| 9 | 260 | 0.8955 | 0.9037 | +0.0082 |
| 10 | 306 | 0.8754 | 0.8697 | −0.0058 |

逐类 3 升 7 降：class9 +0.0082、class8 +0.0032、class5 +0.0031 小幅提升，class2 −0.1079 断崖式
退化（support 仅 37）、class4 −0.0147、class6 −0.0114；高支持类 class9 / class10 一升一降基本持平。
class2 是 A-007 掉出判读门的主要来源（10 类平均上拉低约 0.0108，AUC 总差 −0.0135）。

**判读**：

- 方向性门：0.7903 距门 0.8088 差 −0.0185（约 5.6 倍 A 组 seed 波动 ±0.0033）→ 明确未过门；
  相对 A-001（0.8038 / 0.4255）也低 −0.0135 / −0.0051（约 4.1 倍 seed 波动）。
- 配对 bootstrap（`outputs/plan17_bootstrap.py`；859 studies 按 study 有放回重采样 2,000 次、
  seed 2026、退化类跳过）：ΔAUC（A-001 − A-007）= **+0.0136**，95% CI **[+0.0006, +0.0260]**、
  P(Δ>0) = **0.983**；ΔF1@0.5 = +0.0051，95% CI [−0.0065, +0.0172]、P = 0.807 → AUC 差的可信
  区间整体在 0 以上（A-001 更优），F1@0.5 差方向一致但区间跨 0；该检验只覆盖 valid 内重采样、
  不含训练随机性，且单 seed。
- 学习曲线：每轮 AUC 都低于 A-001 同轮（epoch 1 0.7452 vs 0.7738，epoch 6 0.7903 vs 0.8012），
  峰值更晚（epoch 6 vs 4）且更低；valid_loss 第 5 轮见底 0.9267 后反弹到 1.3911 / 1.8042，反弹
  幅度明显大于 A-001（0.9333 / 1.0194）→ 更大骨干在固定 8 轮预算内没有学得更好，反而过拟合更陡；
  train_loss 到第 8 轮仍降到 0.4471 → 训练集拟合更快、验证集泛化更差，与参数量 +77.7% 的容量
  不匹配当前数据规模（假设，未单独验证）。
- 逐类形态：class2（support 37）断崖退化 −0.1079，是 AUC 差的主要来源；class9 提升 +0.0082 →
  差异不是全类均匀平移，而是个别低支持类被大模型带偏（假设：更大容量的分类头对稀疏类过拟合，
  未单独验证）。
- 资源：参数量 +21,634,560（+77.7%）；实测显存 14,702 MiB vs A-001 的 11.7 GB（约 +3.0 GB）；
  耗时 37m03s / 8 轮 vs A-001 34m01s / 6 轮（总时长 +8.9%，轮数多 2 轮）；预训练权重为 FB
  ImageNet-22k→1k 两阶段（200,928,946 B，约为 A-001 的 `fb_in1k` 权重 114,374,272 B 的 1.76 倍）。
- 结论：**骨干/预训练权重方向否定（单 seed）**——ConvNeXt-Tiny（`fb_in1k`）保持 A 组默认候选，
  ConvNeXt-Small + `in22k_ft` 不进入候选提交、不补 seed 2026 复核；若继续更大骨干方向，应先解决
  固定 8 轮预算下 Small 的过拟合（如更早停止 / 更强正则）或改用更长训练预算，并另开单变量实验
  （Small + `fb_in1k` 或 Tiny + `in22k_ft`）拆分结构与权重的影响。
- 口径注意：本组同时变骨干结构与预训练数据，差值不能单变量归因；单 seed；A-001 与 A-007 各自
  用同一 valid 选型（best epoch 按 valid AUC 选择，可能高估收益）；history 原始值见 `metrics.csv`。

## A-008：A-001 的聚合空间对照（logits G0 → 概率空间 G1，320×320 / seed=42）

背景与判读门见 [train_plan18.md](train_plan18.md)；配置 `configs/exp_A008.yaml`（`_base: exp_A001.yaml`，
只覆盖 `model.aggregation: view_weighted → view_weighted_probability`，`frontal_weight: 0.7` 不变）。

| 项 | 值 |
|---|---|
| 运行命令 | `HF_HUB_OFFLINE=1 .venv/bin/python src/run_train.py --config configs/exp_A008.yaml --seed 42 --tag A-008_convnext_tiny_g1_s42` |
| 代码版本 | `c57ddf0`（分支 `exp/A-008`） |
| 设备环境 | autodl 独占 Tesla V100-SXM2-32G（驱动 580.105.08），训练中显存约 14.3 GB |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies；`check_data.py --full` 全通过 |
| 预训练 | `convnext_tiny.fb_in1k`，`HF_HUB_OFFLINE=1` 离线加载 |
| 产物 | `outputs/20261031-190629_A-008_convnext_tiny_g1_s42/`（`best.pt` 111,385,259 B / `last.pt` / `metrics.csv` / `train.log` / `config.yaml`）、`outputs/eval_reports/A-008_full.txt`、`outputs/A-008_valid_predictions.csv` |
| 耗时 | 2026-10-31 19:06:29 → 19:38:48 CST，共 32m19s（含启动与首轮数据索引），第 7 轮后早停（best epoch 5） |

| epoch | lr | train_loss | valid_loss | valid macro-AUC | valid F1@0.5 |
|---|---|---|---|---|---|
| 1 | 1.50e-04 | 0.9924 | 0.9489 | 0.7722 | 0.3870 |
| 2 | 3.00e-04 | 0.9366 | 0.9397 | 0.7776 | 0.4010 |
| 3 | 3.00e-04 | 0.8916 | 0.9329 | 0.7794 | 0.4098 |
| 4 | 2.72e-04 | 0.8468 | 0.9181 | 0.7909 | 0.4218 |
| 5 | 2.00e-04 | 0.7660 | 0.9328 | **0.7974** | 0.4197 |
| 6 | 1.10e-04 | 0.6242 | 1.0375 | 0.7942 | 0.4352 |
| 7 | 3.77e-05 | 0.4841 | 1.2658 | 0.7867 | 0.4338 |

**结论**：best epoch 5，valid macro-AUC **0.7974** / F1@0.5 **0.4197**；相对 A-001（0.8038 / 0.4255）
差 **AUC −0.0064、F1@0.5 −0.0058**，距方向性门 0.8088 差 **−0.0114 → 未过门**。A-008 的 AUC 在
ep5 见顶（A-001 为 ep4），ep6–7 过拟合更陡（valid_loss 1.0375 / 1.2658）；F1@0.5 逐轮升至 ep6
0.4352（高于 A-001 全程峰值 0.4308），属概率空间平均带来的校准形态差异。单 seed、差值超过
±0.0033 波动带，结论：概率空间聚合训练方向否定，默认聚合保持 `view_weighted`。

## A-009：A-001 的优化超参联合放大（lr 1e-3 / wd 0.03 / 10 轮 / patience 4，320×320 / seed=42）

背景与判读门见 [train_plan19.md](train_plan19.md)；配置 `configs/exp_A009.yaml`（`_base: exp_A001.yaml`，
只覆盖 `train.lr: 0.0003 → 0.001`、`train.weight_decay: 0.01 → 0.03`、`train.epochs: 8 → 10`、
`train.early_stop_patience: 2 → 4`，其余逐字段继承 A-001）。

| 项 | 值 |
|---|---|
| 运行命令 | `HF_HUB_OFFLINE=1 .venv/bin/python -u src/run_train.py --config configs/exp_A009.yaml --seed 42 --tag A-009_convnext_tiny_lr1e-3_wd0.03_s42` |
| 代码版本 | `8a449c7`（分支 `exp/A-009`；本组只新增配置，未改代码） |
| 设备环境 | autodl 独占 Tesla V100-SXM2-32G（驱动 580.105.08） |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 同一份，未重跑 `check_data --full`） |
| 预训练 | `convnext_tiny.fb_in1k`，`HF_HUB_OFFLINE=1` 离线加载（日志「预训练权重: 已加载」） |
| 产物 | `outputs/20261031-194952_A-009_convnext_tiny_lr1e-3_wd0.03_s42/`（`best.pt` 111,385,195 B / `last.pt` 445,550,459 B / `metrics.csv` / `train.log` / `config.yaml`）、`outputs/eval_reports/A-009_full.txt`、`outputs/A-009_valid_predictions.csv`、`outputs/compare_A009/bootstrap_A009.csv` |
| 耗时 | 2026-10-31 19:49:52 → 20:33:27 CST，共 **43m35s**（10/10 轮跑满、早停未触发，含启动与数据索引） |

| epoch | lr | train_loss | valid_loss | valid macro-AUC | valid F1@0.5 |
|---|---|---|---|---|---|
| 1 | 5.00e-04 | 1.0425 | 0.9723 | 0.7387 | 0.3697 |
| 2 | 1.00e-03 | 1.0060 | 1.0090 | 0.7319 | 0.3775 |
| 3 | 1.00e-03 | 0.9727 | 0.9896 | 0.7397 | 0.3820 |
| 4 | 9.51e-04 | 0.9541 | 0.9777 | 0.7556 | 0.3680 |
| 5 | 8.14e-04 | 0.9282 | 0.9573 | 0.7590 | 0.3829 |
| 6 | 6.15e-04 | 0.8966 | 0.9384 | 0.7730 | 0.3948 |
| 7 | 3.95e-04 | 0.8516 | 0.9396 | 0.7789 | 0.4072 |
| 8 | 1.96e-04 | 0.7791 | 0.9757 | 0.7780 | 0.4078 |
| 9 | 5.90e-05 | 0.6969 | 1.0241 | 0.7797 | 0.4107 |
| 10 | 1.00e-05 | 0.6562 | 1.0955 | **0.7799** | **0.4102** |

**结论**：`metrics.csv` 的 best epoch 10（10 轮全部跑满，`early_stop_patience: 4` 未触发），
valid macro-AUC **0.7799** / F1@0.5 **0.4102**；`run_eval.py` 重算为 **0.7799** / **0.4101**
（F1 差 0.0001），阈值 0.2–0.7 曲线为 0.3825 / 0.3991 / 0.4053 / **0.4101** / 0.4083 / 0.3925，
F1 峰位在 0.5。相对 A-001（0.8038 / 0.4255）差 **AUC −0.0239、F1@0.5 −0.0154**，距方向性门
0.8088 差 **−0.0289 → 未过门**。

训练行为上，A-009 的 AUC 从 ep1 0.7387 逐轮升到 ep10 0.7799，但 valid_loss 在 ep6 触底
0.9384 后连续反弹（ep8 0.9757 / ep9 1.0241 / ep10 1.0955），10 轮内没有出现 A-001 那种 ep4
见顶的早峰（A-001 ep4 0.8038，ep5 / ep6 为 0.8016 / 0.8012）；也就是 lr 放大到 1e-3 后 AUC 的
上升被推迟到训练末尾，而验证损失已经明显过拟合。逐阈值看，A-009 在 0.2–0.3 段高于 A-001
（0.3825 vs 0.3641、0.3991 vs 0.3835），0.5–0.7 段全面更低，属概率分布右移、判正更激进的
校准形态，不是单纯的 F1 变差。

逐类 AUC（A-001 与 A-009 均由 `run_eval.py` 在 valid 859 studies 上重算）：A-001 在 10 类中
9 类高于 A-009，只有 class7 低 0.0027；降幅最大是 class2（0.8537 → 0.7670，−0.0867，support
仅 37）、class4（−0.0414）与 class3（−0.0391），class9 / class10 基本持平（−0.0027 / −0.0014），
没有单类崩掉。配对 bootstrap（859 studies × 2,000 次，按 study 有放回重采样、seed 2026、退化类
跳过）ΔAUC（A-001 − A-009）95% CI **[+0.0121, +0.0360]** P=1.000、ΔF1@0.5 95% CI
**[+0.0027, +0.0285]** P=0.991，差距不是 valid 内采样噪声。

本组四项超参同改（lr ×3.3、wd ×3、epochs +2、patience ×2），差值不能单变量归因；A-004-b 已
显示 wd 0.01 → 0.05 单独只差 AUC −0.0039（seed 波动内），因此 −0.0239 的下降更可能主要来自
lr 放大后的收敛推迟与过拟合加剧，但不能据此排除 wd 与轮数的影响。结论：**优化超参联合放大
方向否定（单 seed）**，默认超参保持 A-001 口径（lr 0.0003 / wd 0.01 / 8 轮 / patience 2），
不补 seed 2026；若继续调参，应另开 lr、wd 单变量实验。
## A-009-r2：A-009 的二轮 warm start（以首轮 best.pt 为初始权重、同配置重训 10 轮，320×320 / seed=42）

背景与判读门见 [train_plan20.md](train_plan20.md)；配置 `configs/exp_A009.yaml`（与首轮 A-009 逐字段相同，
只把初始化改为首轮 `best.pt`，epoch 10 权重，SHA-256 `6988006910b824a83c277764695d0db4a3bdeb3ab0b1b029d22dc9e3ee0f1b15`）。
训练从第 1 轮重新开始，不恢复优化器 / 调度器 / 进度。

| 项 | 值 |
|---|---|
| 运行命令 | `HF_HUB_OFFLINE=1 .venv/bin/python -u src/run_train.py --config configs/exp_A009.yaml --seed 42 --init-from outputs/20261031-194952_A-009_convnext_tiny_lr1e-3_wd0.03_s42/best.pt --tag A-009-r2_warmstart_convnext_tiny_lr1e-3_wd0.03_s42` |
| 代码版本 | `exp/A-009` 分支（含本轮 `--init-from` 改动） |
| 设备环境 | autodl Tesla V100-SXM2-32G（与另一容器共享 GPU，宿主机侧约 14.3 GiB 被占，训练每轮约 6–7 分钟，慢于首轮独占的 4.4 分钟/轮） |
| 数据 | train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies（与 A-001 同一份） |
| 预训练 | 不加载 ImageNet；`--init-from` 从首轮 A-009 `best.pt` 加载权重，训练日志「warm start: 从 … 加载模型权重 (source epoch 10, sha256 6988006910b8…)」 |
| 产物 | `outputs/20261031-204914_A-009-r2_warmstart_convnext_tiny_lr1e-3_wd0.03_s42/`（`best.pt` 111,385,579 B / `last.pt` 445,550,395 B / `metrics.csv` / `train.log` / `config.yaml`）、`outputs/eval_reports/A-009-r2_full.txt`、`outputs/A-009-r2_valid_predictions.csv` |
| 耗时 | 2026-10-31 20:49:14 → 21:13:48 CST，共 **24m34s**（5 轮早停，GPU 共享，不可横向比较） |

| epoch | lr | train_loss | valid_loss | valid macro-AUC | valid F1@0.5 |
|---|---|---|---|---|---|
| 1 | 5.00e-04 | 0.7556 | 0.9977 | **0.7764** | **0.4117** |
| 2 | 1.00e-03 | 0.8260 | 0.9703 | 0.7739 | 0.4077 |
| 3 | 1.00e-03 | 0.8197 | 1.0172 | 0.7587 | 0.3980 |
| 4 | 9.51e-04 | 0.8100 | 0.9863 | 0.7692 | 0.3901 |
| 5 | 8.14e-04 | 0.7586 | 1.0525 | 0.7664 | 0.3985 |

**结论**：第 5 轮结束后连续 4 个 epoch 没有提升（`early_stop_patience: 4` 触发），best epoch 1，
valid macro-AUC **0.7764** / F1@0.5 **0.4117**；`run_eval.py` 重算一致，阈值 0.2–0.7 曲线
0.3684 / 0.3833 / 0.3966 / 0.4117 / **0.4137**（峰 0.6）/ 0.4022。相对首轮 A-009（0.7799 / 0.4101）
差 **AUC −0.0035、F1@0.5 +0.0016**；相对 A-001（0.8038 / 0.4255）差 **AUC −0.0274、F1@0.5 −0.0138**，
距方向性门 0.8088 差 **−0.0324 → 未过门**。训练行为：从首轮 ep10 权重出发，ep1 训练后 valid AUC
0.7764（低于源权重的 0.7799），train_loss 由首轮的 0.6562 反弹到 0.7556——lr 重新 warmup 到 1e-3
破坏了首轮已收敛的权重；之后 4 轮 AUC 都没有回到 0.7764 以上（0.7739 / 0.7587 / 0.7692 / 0.7664），
早停直接结束。逐类 AUC 与首轮同批有效类大致同形（class5 0.8579 / class9 0.8942 / class10 0.8678，
class1 0.6993 / class4 0.6764 / class8 0.6973），没有单类崩掉，也没有单类明显抬高。
判读：**warm start 二轮无增益**，A-009 首轮「lr 放大 + wd 放大 + 预算延长」的否定结论不变；
继续在已收敛解上跑同配置第二轮只会继续过拟合。本轮耗时受 GPU 共享影响（每轮约 6–7 分钟 vs
首轮 4.4 分钟/轮），不可与首轮横向比较。

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
| RUN008-A | `exp_run008a.yaml` | EfficientNet-B2 + ra_in1k，6 轮无早停 | 6 | 4 | 0.7952 | 0.4095 | 34m33s（与 B 并行） | 与 RUN-007-A 差 AUC +0.0046、F1 −0.0103，未达 +0.005 门 |
| RUN008-B | `exp_run008b.yaml` | EfficientNetV2-S RW + ra2_in1k，6 轮无早停 | 6 | 2 | 0.7954 | 0.4100 | 35m04s（与 A 并行） | 与 RUN-007-A 差 AUC +0.0048、F1 −0.0098，未达门 |
| RUN008-C | `exp_run008c.yaml` | ConvNeXt-Tiny + fb_in1k，6 轮无早停 | 6 | 5 | 0.8008 | 0.4382 | 24m18s（前 14 分钟与 D 并行） | 与 RUN-007-A 差 AUC +0.0102、F1 +0.0184，超过门；但训练预算不同、单 seed，仅作候选方向 |
| RUN008-D | `exp_run008d.yaml` | ConvNeXtV2-Nano + fcmae_ft_in1k，6 轮无早停（仅研究） | 6 | 2 | 0.4977 | 0.1604 | 22m39s（单独） | 训练异常（逐类 AUC≈随机，原因未定位）；CC BY-NC 4.0 不合规，不参与判读、不用于提交 |
| RUN-007-B | `exp_run007b.yaml` | DenseNet-121 + 正侧位 7:3 加权聚合 + 384×384，seed 2026，6 轮短跑 | 6 | 5 | 0.7959 | 0.4227 | 23m42s | 探针：6 轮内达 0.7959（比 007A 的 12 轮 0.7906 高 +0.0053），但轮数与调度不同、不可单变量归因分辨率；待 25 轮完整对照 |
| RUN-009-A | `exp_run009a.yaml` | EfficientNetV2-S RW + ra2_in1k，384×384，25 轮上限 | 6（用户要求停止）/25 | 5 | 0.7944 | 0.4185 | 约 41m（V100，独占） | 未达门：vs RUN-007-A AUC +0.0038（门 0.7956）、F1 −0.0020；比 RUN-008-B 低 −0.0010；6/25 轮提前停止（非早停），不能判定完整预算 |
| A-001 | `exp_A001.yaml` | ConvNeXt-Tiny + fb_in1k，epoch=8 / patience=2 / seed=42（RUN008-C 重跑） | 6（早停）/8 | 4 | **0.8038** | 0.4255 | 34m01s（独占，V100） | 与 RUN008-C（0.8008 / 0.4382）差 AUC +0.0030、F1 −0.0127；seed/轮数/早停三项同时不同，非单变量对照，AUC 差在 seed 波动内 → 视为 RUN008-C 水平复现 |
| A-001-a | `exp_A001a.yaml` | A-001 的训练随机水平翻转对照：唯一差异 `image.hflip: true` | 6（早停）/8 | 4 | 0.7960 | 0.4045 | 36m58s（独占，V100） | 与 A-001（0.8038 / 0.4255）差 AUC −0.0078、F1@0.5 −0.0210；AUC 差超 ±0.005 门（约 2.4 倍 seed 波动）→ 训练水平翻转无增益，默认 `image.hflip: false` 保持不变 |
| A-001-B | `exp_A001b.yaml` | A-001 的损失方案替换：BCE + `pos_weight` → ASL（gamma_pos 0 / gamma_neg 4 / clip 0.05） | 6（早停）/8 | 4 | 0.7970 | 0.4103 | 34m41s（独占，V100） | 与 A-001（0.8038 / 0.4255）差 AUC −0.0068、F1@0.5 −0.0152；超过 ±0.005 判读门且约 2 倍 seed 波动，单 seed + 同时移除 `pos_weight` → A-001 口径下 ASL 无增益，不切换默认损失 |
| A-001-C | `exp_A001c.yaml` | A-001 的冻结骨干对照：ConvNeXt-Tiny 骨干全部冻结、仅训练分类头（768→10，7,690 / 27,827,818 可训练参数） | 8（跑满）/8 | 8 | **0.7264** | 0.3602 | 38m01s（独占，V100） | 与 A-001（0.8038 / 0.4255）差 AUC −0.0774、F1@0.5 −0.0653，远超 ±0.005 门 → 冻结 ImageNet 特征不可替代全量微调；checkpoint metadata 确认 `freeze_backbone: true`、`trainable_params: 7690` |
| A-001-B(s2026) | `exp_A001b.yaml` + `--seed 2026` | A-001-B 的 seed 复核：与 seed 42 同配置、只换 seed 2026 | 7（早停）/8 | 5 | **0.7952** | 0.4182 | 37m29s（独占，V100） | 与 A-001-B(s42) 配对差 AUC −0.0018（seed 波动 ±0.0033 内）、F1@0.5 +0.0079；与 A-001（BCE，0.8038 / 0.4255）差 AUC −0.0086 仍超 ±0.005 门 → ASL 无增益在两 seed 复现，不切换默认损失 |
| A-001-D | `exp_A001d.yaml` | A-001 的聚合对照：`model.aggregation: view_weighted → mean`（取消正侧位 0.7/0.3 加权，study 内等权均值） | 8（跑满）/8 | 6 | 0.7931 | 0.4340 | 40m03s（独占，V100） | 与 A-001（0.8038 / 0.4255）差 AUC −0.0107、F1@0.5 +0.0085；AUC 差超 ±0.005 门（约 3.2 倍 seed 波动）→ 取消正侧位加权在 A-001 口径下判别力下降；F1 峰值由 0.6 左移到 0.5，属校准形态变化 |
| A-001-G1 | `exp_A001_G1.yaml` | A-001-B 的 ASL 负样本抑制减弱对照：`train.asl_gamma_neg: 4.0 → 2.0`（减弱负样本抑制） | 6（早停）/8 | 4 | 0.7949 | 0.3956 | 30m17s（串行，V100） | 与 A-001-B（0.7970 / 0.4103）差 AUC −0.0021、F1@0.5 −0.0147：减弱负样本抑制无增益，F1 峰左移到 0.4（0.4222） |
| A-001-G2 | `exp_A001_G2.yaml` | A-001-B 的 ASL 负样本抑制加强对照：`train.asl_gamma_neg: 4.0 → 6.0`（更强负样本抑制） | 6（早停）/8 | 4 | 0.7999 | 0.3391 | 31m42s（串行，V100） | 与 A-001-B 差 AUC +0.0029（seed 波动内）、F1@0.5 −0.0712：AUC 微升但概率整体推高，F1 峰右移到 0.6（0.4211） |
| A-001-G3 | `exp_A001_G3.yaml` | A-001-B 的 ASL 概率移位关闭对照：`train.asl_clip: 0.05 → 0.00`（关闭概率移位） | 6（早停）/8 | 4 | 0.7971 | 0.4320 | 30m51s（串行，V100） | 与 A-001-B 差 AUC +0.0001、F1@0.5 +0.0217（也高于 A-001 的 0.4255）：关闭概率移位只改校准，F1 峰在 0.5，组内 F1@0.5 最高 |
| A-001-G4 | `exp_A001_G4.yaml` | A-001-B 的 ASL 概率移位增大对照：`train.asl_clip: 0.05 → 0.10`（增大概率移位） | 3（早停）/8 | 1 | 0.7855 | 0.3695 | 18m14s（串行，V100） | 与 A-001-B 差 AUC −0.0115、F1@0.5 −0.0408：best epoch 1、3 轮即早停 → 增大概率移位破坏训练，方向否定 |
| A-001-G5 | `exp_A001_G5.yaml` | A-001-B 的 ASL 正样本聚焦对照：`train.asl_gamma_pos: 0.0 → 1.0`（正样本聚焦） | 6（早停）/8 | 4 | 0.8002 | 0.4202 | 30m33s（串行，V100） | 与 A-001-B 差 AUC +0.0032（seed 波动内）、F1@0.5 +0.0099：五组 AUC 最高，但仍低于 A-001 的 0.8038 → 方向偏正面未达判定 |
| A-009 | `A-009_convnext_tiny_lr1e-3_wd0.03_s42` | `configs/exp_A009.yaml`（`_base: exp_A001.yaml`） | 10（跑满上限） | 10 | 0.7799 | 0.4101 | 43m35s（独占，V100） | 与 A-001（0.8038 / 0.4255）差 AUC −0.0239、F1@0.5 −0.0154，距 0.8088 门差 −0.0289 → 未过门；lr 放大 3.3 倍后收敛右移、valid_loss 末段反弹，四项同改不可单变量归因 |
| A-009-r2 | `A-009-r2_warmstart_convnext_tiny_lr1e-3_wd0.03_s42` | A-009 的二轮 warm start：`configs/exp_A009.yaml` + `src/run_train.py --init-from` 首轮 `best.pt`（epoch 10 权重） | 10（早停 4，实际 5） | 1 | 0.7764 | 0.4117 | 24m34s（GPU 共享） | 与首轮 A-009（0.7799 / 0.4101）差 AUC −0.0035、F1@0.5 +0.0016，与 A-001（0.8038 / 0.4255）差 AUC −0.0274、F1@0.5 −0.0138，距 0.8088 门差 −0.0324 → 未过门；lr 重新 warmup 破坏首轮收敛权重（ep1 0.7764 后 4 轮无提升），warm start 无增益 |
| A-002 | `exp_A002.yaml` | MobileNetV3-Small 0.5×（578,474 参数）；epoch=25 / patience=5 / seed=42 | 5（用户要求中止）/25 | — | — | — | 约 24m（V100） | 第 5 轮按用户要求停止（非早停），无 `best.pt`，不能与 A-003 配对；前 5 轮最佳 epoch 4（0.7634 / 0.3918）仅作历史参考 |
| A-003 | `exp_A003.yaml` | MobileNetV3-Small 1.0×（1,528,106 参数）；epoch=25 / patience=5 / seed=42 | 17（早停）/25 | 12 | 0.7808 | 0.3881 | 1h17m19s（独占，V100） | 未过门：vs A-001（0.8038 / 0.4255）AUC -0.0230、F1@0.5 -0.0374，距方向性筛选门 0.7988 差 -0.0180；A-002 中止无 `best.pt` → 配对比较不成立 |
| B-001 | `exp_B001.yaml` | B 组：TxRV `densenet121-res224-all` 医学预训练 + 单通道 224×224 + xrv 归一化（6,957,834 参数）；epoch=25 / patience=5 / seed=42，**仅研究** | 11（早停）/25 | 6 | 0.7912 | 0.4120 | 约 40m（独占，V100） | 与 RUN-006（0.7880 / 0.4070）差 AUC +0.0032、F1@0.5 +0.0050，但分辨率/归一化/通道/权重来源同时不同，差值不可归因医学预训练；本地数据与预训练集重叠风险未知 → 指标可能乐观；不用于提交/融合/候选选择 |
| A-004 | `exp_A004.yaml` | A-001 的分类头 dropout 与 weight decay 组合对照：`model.dropout: 0.0 → 0.2` + `train.weight_decay: 0.01 → 0.05` | 7（早停）/8 | 5 | 0.7988 | 0.4233 | 32m16s（独占，V100） | 与 A-001（0.8038 / 0.4255）差 AUC −0.0050、F1@0.5 −0.0022，距方向性门 0.8088 差 −0.0100 → 未过门；正则化只抬高训练损失水平、未推迟过拟合（valid loss 第 4 轮后仍反弹），配对 bootstrap ΔAUC 95% CI [−0.0061, +0.0167] 跨 0 → 无增益证据，A-001 口径不变 |
| A-004-a | `exp_A004a.yaml` | A-001 的分类头 dropout 单变量对照（把 A-004 的组合拆开）：`model.dropout: 0.0 → 0.2`，`train.weight_decay` 保持 0.01 | 8（跑满）/8 | 6 | 0.7944 | 0.4360 | 37m16s（独占，V100） | 与 A-001（0.8038 / 0.4255）差 AUC −0.0094、F1@0.5 +0.0105，距方向性门 0.8088 差 −0.0144 → 未过门；dropout 单独让排序变差、F1 峰由 0.6 左移到 0.4（0.4400），valid loss 未推迟过拟合（第 4 轮后仍反弹）；配对 bootstrap ΔAUC 95% CI [−0.0027, +0.0215] P=0.934 → dropout 是主要负向项，A-001 口径不变 |
| A-004-b | `exp_A004b.yaml` | A-001 的 weight decay 单变量对照（把 A-004 的组合拆开）：`train.weight_decay: 0.01 → 0.05`，`model.dropout` 保持 0.0 | 7（早停）/8 | 5 | 0.7999 | 0.4238 | 31m59s（独占，V100） | 与 A-001（0.8038 / 0.4255）差 AUC −0.0039、F1@0.5 −0.0017，距方向性门 0.8088 差 −0.0089 → 未过门；差值在 seed 波动（±0.0033）量级，配对 bootstrap ΔAUC 95% CI [−0.0056, +0.0138] P=0.785 → weight decay 单独接近噪声、无增益，A-001 口径不变 |
| A-005 | `exp_csra.yaml` | 与 A-006 同批的分类头对照：标准头换单头 CSRA（残差权重 0.4 / 温度 1.0 / 无 bias，27,827,808 参数，比标准头少 10）；其余逐字段同 A-001（16:41:23 启动；工作区 `b5b14d9`） | 5（早停）/8 | 3 | 0.7936 | 0.4050 | 24m28s（独占，V100） | 与配对基线 A-005_baseline（0.8022 / 0.4210）差 AUC −0.0086、F1@0.5 −0.0161，距方向性门 0.8072 差 −0.0136 → 未过门，`model.head: standard` 保持默认、不补 seed |
| A-005_baseline | `exp_A001.yaml` | A-006 的配对基线：标准头同版本重训（15:11:14 启动；工作区 `50620f3` + 未提交改动） | 6（早停）/8 | 4 | 0.8022 | 0.4210 | 29m21s（独占，V100） | 与 A-001（0.8038 / 0.4255）差 AUC −0.0016、F1@0.5 −0.0045，在 seed 波动（±0.0033）量级；为 A-006 提供同数据、同设备、同预算的配对基线 |
| A-006 | `exp_ml_decoder.yaml` | A-001 的分类头替换对照：timm 标准头 → 轻量 ML-Decoder（10 类查询 + 单层 4 头交叉注意力；28,549,226 参数，+721,408） | 7（早停）/8 | 5 | 0.7900 | 0.4118 | 32m39s（独占，V100） | 未过门：vs 配对基线（0.8022 / 0.4210）差 AUC **−0.0122**、F1@0.5 **−0.0092**，距方向性门 0.8072 差 −0.0172；配对 bootstrap ΔAUC（基线 − A-006）= +0.0123、95% CI [+0.0001, +0.0245]、P=0.976 → 标准头 AUC 显著更优；逐类 2 升 8 降、低支持类退化最大 → 分类头方向否定，默认 `standard` 不切换、不补 seed 2026 |
| A-007 | `exp_A007.yaml` | A-001 的骨干与预训练权重对照：ConvNeXt-Tiny + `fb_in1k` → ConvNeXt-Small + `fb_in22k_ft_in1k`（49,462,378 参数，+21,634,560 / +77.7%） | 8（跑满后早停）/8 | 6 | 0.7903 | 0.4204 | 37m03s（独占，V100） | 未过门：vs A-001（0.8038 / 0.4255）差 AUC **−0.0135**、F1@0.5 **−0.0051**，距方向性门 0.8088 差 −0.0185；配对 bootstrap ΔAUC（A-001 − A-007）= +0.0136、95% CI [+0.0006, +0.0260]、P=0.983 → A-001 AUC 显著更优；逐类 3 升 7 降、class2 −0.1079 断崖退化为主因 → 骨干/预训练权重方向否定，ConvNeXt-Tiny `fb_in1k` 保持 A 组候选、不补 seed 2026 |

注：RUN-001R / 002 的 macro-AUC 与 F1@0.5 取 `outputs/compare/summary.md` 的重算值（与 `metrics.csv` 的差异 ≤0.0003，见各 run 的「口径说明」）。
注：RUN-003-B 与 RUN-003 / 003-A / 002-A 的 macro-AUC 与 F1@0.5 取 `outputs/compare_run003b/summary.md` 的重算值（RUN-003-B 与训练时 `metrics.csv` 的差异 ≤0.0002）。
注：RUN-005 / RUN-006 的 macro-AUC 与 F1@0.5 取 `outputs/compare_backbones/summary.md` 的重算值（与 `metrics.csv` 的差异 ≤0.0013）；两组与 RUN-003-B 同为 cosine 调度 + `pos_weight: true` + seed 42，可与 RUN-003-B 直接作单变量骨干对照（`outputs/compare_backbone_s42/summary.md`），与 RUN-003 系列的 warmup / `pos_weight: false` 口径不同，历史数字只作参考。
注：RUN-001R 与 RUN-002 并行执行（两路各 8 workers，16 核饱和），单轮耗时不可与 RUN-001 的串行口径直接对照；主进程负责 shuffle，但 `num_workers` 等运行差异可能影响训练结果。
注：RUN-003 为独占串行运行（16 workers 独占 16 核），单轮约 3.3 分钟，可与 RUN-001 的串行口径对照；13 轮总耗时约 38 分钟（含启动与验证），未跑满 25 轮是因早停，不是被中断。
注：RUN-003-B 同为独占串行运行，单轮约 2.6 分钟、13 轮共 33m44s；与 RUN-003 的差异主要来自机器负载（训练时 GPU 空闲），不影响 AUC 口径。

注：RUN-006-A 的 macro-AUC 与 F1@0.5 取 `outputs/compare_run006a/summary.md` 的重算值（`metrics.csv` 的 F1 差 0.0001、AUC 一致）；与 RUN-006 的配置仅 seed 不同，但代码版本和运行负载不同，不能将结果差异完全归因于 seed。RUN-006-A 与 experiment4 的 EfficientNet-B2 实验并行执行（共用 3090 与 16 核 CPU），单轮耗时不可与 RUN-006 的独占 44m32s 对照。
注：RUN-007 的 macro-AUC 与 F1@0.5 取 `outputs/compare_run007/summary.md` 的重算值（与 `metrics.csv` 的 AUC 一致、F1 差 0.0002）；与 RUN-006 仅差 `model.aggregation`，但代码版本不同（`53568fe` vs `da8fef7`，后者无视角元信息），且 0.0012 的 AUC 差小于 DenseNet-121 的 seed 波动 0.0033，不能将差异全部归因于聚合方式。
注：RUN-007-A 的 macro-AUC 与 F1@0.5 取 `outputs/compare_run007a/summary.md` 的重算值（与 `metrics.csv` 的 AUC 一致、F1 差 0.0004）；与 RUN-006-A 的配置仅差聚合方式，但代码版本不同（`27dbe56` vs `a41033d`，`src/` 等价），单 seed 配对差 −0.0007 与 seed 42 的 +0.0012 方向相反，两 seed 配对差绝对值均小于 +0.005 判读门 → 加权聚合判定无效。
注：RUN008 四组的 macro-AUC 与 F1@0.5 取 `outputs/compare_run008/summary.md` 的重算值（A/B/C 与 `metrics.csv` 的差异 ≤0.0001，D 的 AUC 差 0.0120 因输出近似随机）；四组固定 6 轮无早停，与 RUN-007-A 的 25 轮上限 / 12 轮早停不同，差值只作方向性参考。
注：RUN008 四组按 A+B / C+D 两两并行执行，单轮耗时受共享 GPU 影响，不可与串行 run 对照；D 组第一次并行运行于 21:52 按用户要求停止并删除，22:02 起单独重跑。
注：RUN008-D 权重为 CC BY-NC 4.0，按赛题第 1 条不能用于提交；该组训练异常（逐类 AUC 0.46–0.54），不参与骨干判读。
注：RUN-009-A 的 macro-AUC 与 F1@0.5 取 `last.pt` 的 `extras.best_metrics`（`run_eval.py` 重算 0.7945 / 0.4178，差 ≤0.0007）；与 RUN-007-A / RUN-008-B 的差值同时来自分辨率（384 vs 320）与训练预算（25 轮上限 vs 固定 6 轮）改变，且本组只跑 6/25 轮即按用户要求停止，不能作单变量归因；硬件为 V100-SXM2-32G（与 RUN-001~008 的 RTX 3090 是不同实例），耗时不可横向对照。
注：A-001 的 macro-AUC 与 F1@0.5 由 `run_eval.py` 在 valid 上重算（与 `metrics.csv` 一致）；与 RUN008-C 同时差 seed（42 vs 2026）、训练上限（8 vs 6 轮）与早停（2 vs 0），且 cosine 周期长度随 epochs 改变，差值不能单变量归因；硬件为 V100-SXM2-32G（与 RUN-001~008 的 RTX 3090 不同实例），耗时不可横向对照；RUN008-C 的逐类与阈值曲线取自 `outputs/compare_run008/`（RTX 3090 环境重算，同 valid split、同指标函数）。
注：A-001-a 的 macro-AUC 与 F1@0.5 由 `run_eval.py` 在 valid 上重算（0.7960 / 0.4045；`metrics.csv` 的 best epoch 4 为 0.7960 / 0.4047，F1 差 0.0002）；与 A-001 仅差 `image.hflip`（训练随机水平翻转），其余口径逐字段一致，属单变量对照；单 seed，翻转带来的额外增强开销使单轮耗时由 5.7 分钟升到约 6.2 分钟（36m58s / 6 轮），硬件同为 V100-SXM2-32G。
注：A-001-C 的 macro-AUC 与 F1@0.5 由 `run_eval.py` 在 valid 上重算（0.7264 / 0.3602；`metrics.csv` 的 best epoch 8 为 0.7264 / 0.3596，F1 差 0.0006）；checkpoint metadata 记录 `freeze_backbone: true`、`total_params: 27827818`、`trainable_params: 7690`；与 A-001 的差异同时来自可训练参数范围与骨干 `eval()` 模式（BN 统计不更新），非单一变量；硬件同为 V100-SXM2-32G。
注：A-001-B(seed 2026) 的 macro-AUC 与 F1@0.5 由 `run_eval.py` 重算（0.7952 / 0.4182，与 `metrics.csv` 的 best epoch 5 一致）；与 A-001-B(s42) 是 seed 配对（同代码版本、同配置，仅 seed 不同），F1 峰位在 0.6（0.4254）与 s42 一致。
注：A-001-D 的 macro-AUC 与 F1@0.5 由 `run_eval.py` 在 valid 上重算（0.7931 / 0.4340；`metrics.csv` 的 best epoch 6 为 0.7931 / 0.4340，一致）；`configs/exp_A001d.yaml` 以 `_base: exp_A001.yaml` 只覆盖 `model.aggregation`，与 A-001 属单变量对照（同代码 `64a6525`、同 seed 42、同为 V100-SXM2-32G 独占）；F1 峰值 0.5（0.4340）vs A-001 的 0.6（0.4328），阈值曲线在 0.2–0.5 高于 A-001、0.6–0.7 低于 A-001。
注：A-001-G 组五组由同一链式脚本在 autodl V100-SXM2-32G 上串行执行（G1→G5，20:11:24–22:36:29，共 2h25m05s），代码版本统一 `78ce901`、seed 42、数据与 A-001 同一份；每组只覆盖 ASL 五项，其余口径与 A-001 逐字段一致，相对 A-001-B 属单变量对照（G1/G2 只改 gamma_neg、G3/G4 只改 clip、G5 只改 gamma_pos）；单 seed，差值按 ±0.0033 seed 波动与 ±0.005 判读门解读。
注：A-001-G 组 macro-AUC 与 F1@0.5 由 `run_eval.py` 在 valid 上重算（与 `metrics.csv` 的 best epoch 同轮一致）；阈值曲线 0.2–0.7 与逐类 AUC 取自 `outputs/eval_reports/A-001-G<g>_full.txt`。G4 的 best epoch 1 与 3 轮早停说明 clip 0.10 在训练早期即退化。

注：A-002 / A-003 的结果取 A-003 `run_eval.py` 在 valid 上的重算值（与 best epoch 的 `metrics.csv` 一致）；A-002 于第 5 轮按用户要求中止，只有 `last.pt` 与 5 轮 metrics、没有 `best.pt`，不能与 A-003 做配对对照，A-002 一行的 AUC / F1 留空。A-003 与 A-001 同时差骨干、训练预算（25+5 vs 8+2）与 cosine 周期长度，只作方向性比较；单 seed，差值不作统计显著性结论。硬件同为 V100-SXM2-32G，与 RUN-001~008 的 RTX 3090 不同实例，耗时不可横向对照。

注：B-001 为独立研究对照（B 组），权重为 TorchXRayVision `densenet121-res224-all`（Apache-2.0，SHA-256 56524913…8899），预训练数据含 NIH ChestX-ray14、PadChest、CheXpert、MIMIC-CXR、OpenI；本地数据来源不可确证、疑似 MIMIC-CXR-JPG 衍生子集，与预训练集重叠风险未知（疑似高），因此 valid 指标可能乐观，不能据此声称无泄漏；与 RUN-006（ImageNet DenseNet-121 @320、3 通道、imagenet 归一化）的差值同时来自分辨率、通道、归一化与初始化来源，不可单变量归因医学预训练；按赛题规则第 1 条，本组仅作研究，不接入提交、模型融合或候选选择。单 seed。

注：A-004 的 macro-AUC 与 F1@0.5 由 `run_eval.py` 在 valid 上重算（0.7988 / 0.4233；`metrics.csv` 的 best epoch 5 为 0.7988 / 0.4232，F1 差 0.0001）；`configs/exp_A004.yaml` 以 `_base: exp_A001.yaml` 只覆盖两个字段（`model.dropout` 0.2、`train.weight_decay` 0.05），与 A-001 属单 seed 同数据、同划分、同为 V100-SXM2-32G 独占；因同时改两项，只能评价组合、不能分开归因 dropout 与 weight decay 各自的作用；配对 bootstrap（859 studies × 2,000 次）ΔAUC（A-001 − A-004）95% CI [−0.0061, +0.0167]，只覆盖 valid 内重采样、不含训练随机性，不作为显著性结论。

注：A-004-a / A-004-b 的 macro-AUC 与 F1@0.5 由 `run_eval.py` 在 valid 上重算（A-004-a 0.7944 / 0.4360，`metrics.csv` 的 best epoch 6 为 0.7944 / 0.4356，F1 差 0.0004；A-004-b 0.7999 / 0.4238，`metrics.csv` 的 best epoch 5 为 0.7999 / 0.4240，F1 差 0.0002）；两个配置各以 `_base: exp_A001.yaml` 只覆盖一个字段（`model.dropout` / `train.weight_decay`），与 A-001 属单 seed 同数据、同划分、同为 V100-SXM2-32G 独占，代码 `3231c3f` 只新增配置与计划文档、模型与训练代码与 A-001 相同；四个 run（A-001 / A-004 / A-004-a / A-004-b）的逐 study 预测两两做配对 bootstrap（859 studies × 2,000 次）：ΔAUC（A-001 − A-004-a）95% CI [−0.0027, +0.0215]、ΔAUC（A-001 − A-004-b）95% CI [−0.0056, +0.0138]、ΔAUC（A-004 − A-004-a）95% CI [−0.0088, +0.0169]、ΔAUC（A-004 − A-004-b）95% CI [−0.0092, +0.0068]，A-001 − A-004 复算为 +0.0050 / [−0.0064, +0.0163]，与既有记录 +0.0051 / [−0.0061, +0.0167] 仅差实现级别；各组 CI 均跨 0，只覆盖 valid 内重采样、不含训练随机性，不作为显著性结论；A-004 组合的 AUC 水平主要由 weight decay 决定（vs 单 wd 差 −0.0012），dropout 是主要负向项。

注：A-005 一行与 A-006 共用同一条配对基线 `A-005_baseline`：两者相对 A-001 的唯一差异都是 `model.head`，同数据、同设备、同预算、同 seed 42，因此 A-005 与 A-006 的差值也可直接横向比较（0.7936 vs 0.7900）；A-005 的 macro-AUC 与 F1@0.5 由 `run_eval.py` 在 valid 859 studies 上重算（与 `metrics.csv` 的 best epoch 3 一致 0.7936 / 0.4050）；配对 bootstrap（859 studies × 2,000 次，按 study 有放回重采样、seed 2026、退化类跳过，2,000 次均 10 类有效）ΔAUC（基线 − A-005）95% CI [−0.0030, +0.0210]、ΔF1@0.5 95% CI [+0.0061, +0.0266]。

注：A-005_baseline 一行是 A-006 的配对基线（tag 沿用计划 15 的命名），不是独立候选实验；它用 `exp_A001.yaml` 同口径重训标准头，与 A-001（0.8038 / 0.4255）差 AUC −0.0016 / F1@0.5 −0.0045，属 A 组 seed 波动量级，说明 A-001 的水平可由同代码快照复现。A-006 的 macro-AUC 与 F1@0.5 由 `run_eval.py` 在 valid 859 studies 上重算（与 `metrics.csv` 的 best epoch 5 一致 0.7900 / 0.4118），配对基线重算值 0.8022 / 0.4210 与其 `metrics.csv` 的 best epoch 4 一致；配对 bootstrap（859 studies × 2,000 次，按 study 有放回重采样、seed 2026、退化类跳过，2,000 次均 10 类有效）ΔAUC（基线 − A-006）95% CI [+0.0001, +0.0245]、ΔF1@0.5 95% CI [−0.0018, +0.0205]，只覆盖 valid 内重采样、不含训练随机性；A-006 与配对基线同数据、同设备（V100-SXM2-32G）、同预算（8 轮 / patience 2 / seed 42），唯一配置差异为 `model.head`，属单变量对照；参数量由 `best.pt` 权重统计（27,827,818 → 28,549,226）。

注：A-007 的 macro-AUC 与 F1@0.5 由 `run_eval.py` 在 valid 859 studies 上重算（与 `metrics.csv` 的 best epoch 6 一致 0.7903 / 0.4209，F1 差 0.0005）；`configs/exp_A007.yaml` 以 `_base: exp_A001.yaml` 只覆盖 `model.backbone`（`convnext_tiny.fb_in1k → convnext_small.fb_in22k_ft_in1k`），训练代码与 A-001 相同（代码 `ea555c0` 只新增配置、计划文档与 CHANGELOG），属同数据、同划分、同设备（V100-SXM2-32G 独占）、同预算（8 轮 / patience 2 / seed 42）的对照；但相对 A-001 同时变骨干结构与预训练数据，差值不能单变量归因；配对 bootstrap（859 studies × 2,000 次，按 study 有放回重采样、seed 2026、退化类跳过）ΔAUC（A-001 − A-007）95% CI [+0.0006, +0.0260] P=0.983、ΔF1@0.5 95% CI [−0.0065, +0.0172] P=0.807，只覆盖 valid 内重采样、不含训练随机性；参数量由 `best.pt` 权重统计（27,827,818 → 49,462,378）。

注：A-009 的 macro-AUC 与 F1@0.5 由 `run_eval.py` 在 valid 上重算（0.7799 / 0.4101；`metrics.csv` 的 best epoch 10 为 0.7799 / 0.4102，F1 差 0.0001）；`configs/exp_A009.yaml` 以 `_base: exp_A001.yaml` 只覆盖四项（`train.lr` 0.0003 → 0.001、`train.weight_decay` 0.01 → 0.03、`train.epochs` 8 → 10、`train.early_stop_patience` 2 → 4），训练代码与 A-001 相同（代码 `8a449c7` 只新增配置与计划文档），属同数据、同划分、同设备（V100-SXM2-32G 独占）、同 seed 42 的组合对照；四项同改，差值不能单变量归因；配对 bootstrap（859 studies × 2,000 次，按 study 有放回重采样、seed 2026、退化类跳过，2,000 次均 10 类有效）ΔAUC（A-001 − A-009）95% CI [+0.0121, +0.0360] P=1.000、ΔF1@0.5 95% CI [+0.0027, +0.0285] P=0.991，只覆盖 valid 内重采样、不含训练随机性。

判读口径：先看 macro-AUC（不受阈值影响）；再看 `f1@0.5` 与整条阈值曲线（`run_compare` 的 `summary.md`）；`pos_weight` 只有在多数类同时不退化时才算有效。

## 环境

| 项 | 值 |
|---|---|
| GPU | RTX 3090 24G（驱动 535.247.01，CUDA 12.2）；RUN-009 / A-001~A-001-G / A-002 / A-003 / A-004（含 A-004-a / A-004-b）、A-005 / A-005_baseline / A-006 / A-007 系列为 V100-SXM2-32G（驱动 580.105.08），两种硬件的耗时不可横向对照 |
| Python / 框架 | Python 3.12、torch 2.10.0+cu128、timm 1.0.30、torchvision 0.25.0+cu128 |
| B 组独立环境 | `/root/venv-b001`：Python 3.12、torch 2.8.0+cu128、torchxrayvision 1.5.7、timm 1.0.30（仅 B-001） |
| 数据根目录 | `/root/sj-tmp`（`data/` 下以软链接挂入，数据本身不入库） |
| 随机种子 | 42（RUN-002-B / RUN-006-A / RUN-007-A / RUN-007-B / RUN008-A~D / RUN-009-A / A-001-B(seed 2026) 为 2026，其余 run 均为 42） |
