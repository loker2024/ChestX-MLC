# 训练计划 21：C-001 四骨干低学习率与分类头差异学习率

**状态：计划、独立配置和两组学习率代码已准备；GPU 训练、评估和融合均未执行。** 运行前仍须完成权重来源、许可和数据核验，固定实际使用的代码及环境。

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

## 训练准备及命令（仅实现和验证完成后）

在 GPU 训练环境仓库根目录执行；先记录代码 HEAD、`git status --short`、完整未提交 diff、环境版本、权重来源及 SHA-256，核对患者、study、图片划分和数据哈希。确认四组使用同一代码快照与数据，预训练权重缓存可用。训练建议串行，单组失败先定位原因，不将失败或随机初始化结果混入比较。以下是未来执行命令，并不代表已经授权或执行训练：

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
