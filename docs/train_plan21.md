# 训练计划 21：C-001 四骨干低学习率与分类头差异学习率

**状态：四组 GPU 训练、四组评估与融合补充价值分析已完成（2026-11-01 11:42–13:44 训练，13:53–13:55 评估，13:56–14:04 融合分析）。** 四组全部在 epoch 7 早停（best epoch 3），macro-AUC 0.7774–0.7919，均未过 0.8088 方向性门；以等权第五成员加入 A-FUSE-1 后 ΔAUC 在 ±0.0007 内、bootstrap 区间全部跨 0。实际结果见文末「执行记录」。

## 目标与实验矩阵

用统一低学习率训练四种 ImageNet-1k 预训练骨干，检验单模型及其对现有融合的补充价值。四组同 seed 42、同训练数据和配置，仅模型骨干及其预训练权重不同；结构与预训练配方同时变化，不能将结果纯粹归因于架构。每个编号、训练 tag 和预测文件一一对应：

| 编号 / tag | 完整独立配置 | timm 骨干及权重 |
| --- | --- | --- |
| C-001-a | `configs/exp_C001a.yaml` | `convnext_tiny.fb_in1k` |
| C-001-b | `configs/exp_C001b.yaml` | `efficientnet_b2.ra_in1k` |
| C-001-c | `configs/exp_C001c.yaml` | `efficientnetv2_rw_s.ra2_in1k` |
| C-001-d | `configs/exp_C001d.yaml` | `densenet121.ra_in1k` |

四份 YAML 均完整展开，不使用 `_base`；除 `model.backbone` 外字段一致。`model.arch=timm`，全量微调，标准 10 类分类头；ImageNet 预训练权重，并非 B-001 的医学预训练权重。沿用 A-001 的 train/valid/test 路径、320 letterbox、ImageNet 归一化、无水平翻转、正侧位 logits 聚合 0.7/0.3、BCE + `pos_weight`、batch 16 studies、AdamW weight decay 0.01、AMP、grad clip 5.0。

训练上限 12 epoch、早停 patience 4，以 valid Macro-AUC 选 `best.pt`。四组各一 seed，12 轮改变余弦周期；和 A-001 的 8 轮、patience 2、统一 3e-4 学习率相比有多项差异，不能单独归因于学习率、分组或轮数。A-001 历史 valid AUC 0.8038 及四模型融合 0.8116 仅作参考，不作为独立测试集证据。

## 两组学习率的准确口径

- 以 `model.get_classifier()` 返回的模块的可训练参数作为分类头组，其余可训练参数作为骨干组；分类器之前的投影和归一化层属于骨干组。两组非空、互不重叠并覆盖全部可训练参数；无法识别分类头时直接报错。
- `train.lr=0.00005` 是骨干峰值，`train.head_lr=0.00015` 是分类头峰值；所有参数组权重衰减均为 0.01。该方案是两组差异学习率，不是逐 stage 衰减。
- `train.lr_min=0.000005` 是骨干末轮学习率；分类头按同一倍率衰减，末轮为 0.000015。两组采用共同倍率，因此每轮学习率比始终为 1:3。warmup 2 轮，epoch 1 为 0.000025 / 0.000075，epoch 2、3 为 0.00005 / 0.00015；epoch 12 为 0.000005 / 0.000015，早停时不一定到达末轮。

## 实现与本机验证

1. `src/common/config.py` 已加入可选 `head_lr: float | None = None`，默认 `None` 保留历史单参数组行为；非空时校验有限且大于 0，保存与恢复配置包含该字段。`lr`、`lr_min` 分别表示骨干峰值和最低值。
2. `src/train/engine.py` 已按分类器参数对象身份建立分组 AdamW，拒绝空组和无法识别的分类头；`head_lr=None` 保持原单组语义。
3. `_build_scheduler` 对各组应用相同倍率，`lr_min / lr` 为共同末轮倍率。控制台、`metrics.csv` 和已启用的 TensorBoard 分别记录两组实际学习率；续训恢复优化器与调度器状态并检查配置一致性。
4. `tests/test_head_lr.py` 覆盖四个 `pretrained=False` timm 骨干的参数归属、12 轮学习率与比例、非法值和分类头、续训配置差异、连续训练与中断恢复的模型权重和学习率；不下载权重。
5. 使用 `.venv/bin/python -m pytest tests/test_config.py tests/test_engine.py tests/test_resume.py`，再运行 `.venv/bin/python -m pytest` 和 `git diff --check`。检查 C-001 四个配置完整展开及值一致性。修复已发现问题后重新运行受影响检查；训练前固定代码版本及未提交 diff。

当前代码的标准模型路径在预训练加载异常时可能回退随机初始化：训练前确认缓存和日志，训练后检查每个 `best.pt` 的 `metadata.pretrained_loaded == True`；不满足则该次作废，不能作为本计划结果。RUN008 的 ConvNeXt-Tiny、EfficientNet-B2 和 EfficientNetV2-S 权重版本、哈希及许可见 [训练记录](training_log.md) 的 RUN-008 表；正式运行前逐项核对实际文件。DenseNet-121 的准确 revision、文件 SHA-256、模型卡许可与预训练来源须在运行前补齐并确认符合赛题规则；缺证据时暂不训练该组。

## 训练准备及命令（已于 2026-11-01 执行）

在 GPU 训练环境仓库根目录执行；先记录代码 HEAD、`git status --short`、完整未提交 diff、环境版本、权重来源及 SHA-256，核对患者、study、图片划分和数据哈希。确认四组使用同一代码快照与数据，预训练权重缓存可用。训练建议串行，单组失败先定位原因，不将失败或随机初始化结果混入比较。以下为四组实际使用的命令（2026-11-01 已按此串行执行完毕，实际目录、耗时与指标见文末「执行记录」）：

```bash
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C001a.yaml --seed 42 --tag C-001-a
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C001b.yaml --seed 42 --tag C-001-b
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C001c.yaml --seed 42 --tag C-001-c
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C001d.yaml --seed 42 --tag C-001-d
```

每组记录实际运行目录、设备、起止时间、best epoch、停止原因、显存峰值与每轮两组学习率。保留 `config.yaml`、`best.pt`、`last.pt`、`metrics.csv`、`train.log`，不提交权重或数据。预计串行跑满约 2.7–4.6 小时，仅根据历史每轮时间粗估；实际以设备、骨干和早停为准。

## 评估与融合

对每组分别用该次运行目录内的 `config.yaml` 和 `best.pt` 执行评估，例如将下面的 `<run>` 换成 C-001-a 的实际路径，其余三组替换配置路径及输出编号：

```bash
python src/run_eval.py --config outputs/<run>/config.yaml --checkpoint outputs/<run>/best.pt \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-001-a.csv
```

报告各组 Macro-AUC、F1@0.5、阈值曲线、逐类 AUC、类别支持数和最佳 epoch。保留 study ID、标签及连续概率，评估与融合前核验 `Study_id`、`Subject_id`、标签及数据版本一致；不以阈值 F1 代替 AUC。

以已记录的 A-FUSE-1 四模型等权结果为参考，先用同一 valid 和相同脚本重算原融合；分别将每个新候选加入原融合并等权比较，记录逐类变化、配对 bootstrap 和推理成本，再对有价值候选评估替换成员。避免在同一 valid 上进行大规模权重搜索；在独立验证或 OOF 之前，这些选型结果存在乐观偏差，bootstrap 不覆盖训练随机性，也不能消除选型偏差。test 无标签，不报告 test AUC。若要提交，先固定模型和融合公式，并按赛题要求核验模型许可、资源和 T4 端到端耗时。

四组均为单 seed；若某候选值得继续，后续另行安排同配置第二 seed 与相应对照，不把本计划写成已完成的 seed 复核。执行后将实际结果和未完成项回填 `docs/experiments.md`、`docs/training_log.md`、本计划与 `CHANGELOG.md`。

## 执行记录（2026-11-01）

**代码与环境**：远端 `/root/ChestX-MLC`，分支 `loker`，训练前 HEAD `4076647`（四组共用同一快照；C-001 只新增配置与 `train.head_lr` 支持，训练期间未改代码）。autodl Tesla V100-SXM2-32G，`HF_HUB_OFFLINE=1` 离线加载，四组均打印「预训练权重: 已加载」，`best.pt` 的 `metadata.pretrained_loaded` 均为 `True`。数据沿用官方 train 18,513 studies / valid 859 studies，未重划分。四组串行独占 GPU，中间无其他训练或评估进程。

四份配置经 `diff` 逐字段核对，除 `model.backbone` 与首行注释外完全一致，单变量（骨干+预训练权重）设计成立。

| 编号 | 骨干 | 可训练参数 | 运行目录 | 起止（CST） | 耗时 | epochs | best epoch | macro-AUC | F1@0.5 | ΔAUC vs A-001 |
| --- | --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| C-001-a | `convnext_tiny.fb_in1k` | 27,827,818 | `20261101-114223_C-001-a` | 11:42:23–12:13:56 | 31m33s | 7/12（早停） | 3 | 0.7919 | 0.4191 | −0.0119 |
| C-001-b | `efficientnet_b2.ra_in1k` | 7,715,084 | `20261101-121402_C-001-b` | 12:14:02–12:47:03 | 33m01s | 7/12（早停） | 3 | 0.7774 | 0.3996 | −0.0264 |
| C-001-c | `efficientnetv2_rw_s.ra2_in1k` | 22,166,226 | `20261101-124710_C-001-c` | 12:47:10–13:17:02 | 29m52s | 7/12（早停） | 3 | 0.7807 | 0.4055 | −0.0231 |
| C-001-d | `densenet121.ra_in1k` | 6,964,106 | `20261101-131708_C-001-d` | 13:17:08–13:44:12 | 27m04s | 7/12（早停） | 3 | 0.7849 | 0.4049 | −0.0189 |

参照 A-001（ConvNeXt-Tiny / 3e-4 单参数组 / 8 轮 / patience 2）valid macro-AUC **0.8038** / F1@0.5 **0.4255**，方向性门 0.8088。四组均未过门，最大者 C-001-a 距门 −0.0169。

**关键观察**：四组**全部在 best epoch 3 达峰、epoch 7 触发 `patience 4` 早停**，无一跑满 12 轮。三组非 ConvNeXt 骨干的 epoch 1 AUC 只有 0.726–0.739，靠 warmup 后第 2–3 轮补到峰值；epoch 4 起 train_loss 继续下降而 valid_loss 单调反弹（C-001-c 由 0.9331 升到 1.5712），过拟合在所有骨干上同步发生。骨干 5e-5 / 分类头 1.5e-4 的分层低学习率没有推迟过拟合，反而把峰值固定在 warmup 刚结束的第 3 轮——与「低学习率更稳、可跑更长」的预期相反。

**评估**：四组分别用各自运行目录的 `config.yaml` + `best.pt` 在 valid 859 studies 上重算，逐 study 概率写入 `outputs/predictions/valid/C-001-{a,b,c,d}.csv`，报告存 `outputs/eval_reports/C-001-{a,b,c,d}.txt`。F1 阈值曲线（0.2 / 0.3 / 0.4 / 0.5 / 0.6 / 0.7）：a 0.3546 / 0.3813 / 0.4020 / 0.4191 / **0.4277** / 0.4021；b 0.3583 / 0.3770 / 0.3895 / 0.3996 / **0.4017** / 0.3756；c 0.3529 / 0.3775 / 0.3926 / **0.4055** / 0.4050 / 0.3722；d 0.3573 / 0.3806 / 0.3960 / 0.4049 / **0.4198** / 0.3854。四组峰位都在 0.5–0.6，与 A-001 一致，未出现校准整体偏移。

**融合补充价值**：按 A-FUSE-1 口径（A-001 + A-004-b + RUN-009-A + A-001-D 概率空间等权）先重算基线，得 **0.8116 / 0.4416**，与记录值完全一致；再把每个候选作等权第五成员：

| 五模型组合 | macro-AUC | ΔAUC | F1@0.5 | ΔAUC 95% 区间（患者整簇 bootstrap，2000 次） | P(Δ>0) |
| --- | ---: | ---: | ---: | --- | ---: |
| 基线四模型 | 0.8116 | — | 0.4416 | — | — |
| + C-001-a | 0.8112 | −0.0004 | 0.4410 | [−0.0026, +0.0017] | 0.368 |
| + C-001-b | 0.8119 | +0.0003 | 0.4424 | [−0.0029, +0.0032] | 0.603 |
| + C-001-c | 0.8121 | +0.0005 | 0.4403 | [−0.0028, +0.0038] | 0.639 |
| + C-001-d | 0.8109 | −0.0007 | 0.4404 | [−0.0034, +0.0019] | 0.291 |

四个区间的宽度（约 0.005–0.007）远大于点估计，全部跨 0；最大正向 C-001-c 的 +0.0005 仅为其区间下限绝对值的 1/5，不构成补充价值证据。

**判读**：单模型层面四组方向否定——同口径低学习率 + 分层学习率下，四种 ImageNet-1k 骨干都没有超过 A-001 的 0.8038，且全部在 epoch 3 达峰；融合层面四组与基线不可区分。结论：**C-001 不产生提交候选，不改默认骨干与默认学习率**。本轮不能判定某个骨干绝对更差——四组都受同一早峰问题压制，骨干差异被训练预算与学习率组合掩盖；若继续骨干方向，应先消除「best epoch 3」的早峰（调大学习率或缩短 cosine 周期），再重做对照。

**未执行项**：未补第二 seed；未做逐类 AUC 的配对 bootstrap（C-001 不是单变量方向性对照，且四组均未过门）；未做 test 侧推理与 T4 端到端耗时自检；未验证 DenseNet-121 权重的 revision 与 SHA-256（训练日志确认预训练加载成功，但文件哈希未单独记录）。
