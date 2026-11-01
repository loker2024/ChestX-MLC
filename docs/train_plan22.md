# 训练计划 22：C-002 ConvNeXt-Tiny 学习率双 seed 配对实验

**状态：计划与独立配置已准备；GPU 训练和评估均未执行。** 本计划检验统一学习率从 `3e-4` 降至 `1.5e-4` 对 valid Macro-AUC 的影响；原四种骨干的 C-002 方案暂不执行。

## 实验矩阵

| 实验编号 / tag | 配置 | 角色 | 峰值学习率 | 训练 seed | 同 seed 对照 |
| --- | --- | --- | --- | --- | --- |
| C-002-a | `configs/exp_C002a.yaml` | 基线 | 0.0003 | 42 | C-002-b |
| C-002-b | `configs/exp_C002b.yaml` | 低学习率候选 | 0.00015 | 42 | C-002-a |
| C-002-c | `configs/exp_C002c.yaml` | 基线 | 0.0003 | 2026 | C-002-d |
| C-002-d | `configs/exp_C002d.yaml` | 低学习率候选 | 0.00015 | 2026 | C-002-c |

四份 YAML 完整展开、不使用 `_base`，编号与训练 tag、预测和比较目录保持一致。所有组均为 `convnext_tiny.fb_in1k` ImageNet-1k 预训练、全量微调、标准 10 类分类头、AdamW 单参数组（`head_lr: null`）。同 seed 的两组配置仅 `train.lr` 不同；同学习率的两个 seed 仅顶层 `seed` 不同。

固定原始 train / valid，不重划分（`data.resplit.enabled: false`）；320×320 letterbox、ImageNet 归一化、无水平翻转、正侧位 logits 聚合 0.7/0.3、BCE + `pos_weight`、batch 16 studies、weight decay 0.01、AMP、grad clip 5.0、12 epoch 上限、patience 4、warmup 2、cosine、`lr_min: 0.00001`。`seed` 控制训练随机性，不改变这份固定数据划分。

## 预期学习率及对照边界

两组 `lr_min` 都固定为 A-001 的 `1e-5`，仅改变 `train.lr`；它控制整条 warmup / cosine 曲线，而不只是单一时刻的学习率：

| epoch | 基线 a / c | 候选 b / d |
| --- | ---: | ---: |
| 1 | 0.00015 | 0.000075 |
| 2、3 | 0.00030 | 0.00015 |
| 12（若跑满） | 0.00001 | 0.00001 |

末轮相同，所以低学习率曲线不是基线曲线全程减半；早停时未必跑到末轮。对比 a vs b、c vs d 评估该学习率曲线变化，须固定代码、数据、预训练文件、训练预算与环境并记录实际配置。相同 seed 并不保证不同超参数产生相同随机训练轨迹；配对是实验设计，不代表两次梯度过程完全一致。

A-001 使用 8 轮上限、patience 2；本计划 12 轮、patience 4，cosine 周期也变了。A-001 0.8038 仅为历史参考，C-002-a 不称为与 A-001 完全同配置的重跑，也不能把二者的差异单独归因于 LR。

## 训练前核验

1. 固定并记录代码 HEAD、`git status --short`、未提交 diff、依赖版本、设备和 GPU 占用；核对四组展开配置符合矩阵和单变量条件，代码与四组保持一致。
2. 核验原 train / valid 的患者、study、标签及图像索引和数据版本哈希；确认无患者泄漏，所有组使用相同划分。保留已有 valid，不合并或重划分。
3. 核验 `convnext_tiny.fb_in1k` 模型卡许可、预训练数据来源、实际权重 revision 和 SHA-256。历史记录见 [A-001 训练记录](training_log.md) 与 RUN-008 权重表；四组须使用同一权重文件。
4. 当前标准模型路径可能在预训练加载失败后回退随机初始化；训练启动时检查日志，结束后核对 `best.pt` 的 `metadata.pretrained_loaded == True`。未加载预训练的 run 不能当作有效实验，排查后重新运行。
5. 使用项目 `.venv` 验证四份配置、参数组及学习率调度，运行 `.venv/bin/python -m pytest tests/test_config.py tests/test_engine.py tests/test_head_lr.py -q` 和 `git diff --check`。检查完成不代表 GPU 训练已执行。

## 训练

训练环境在仓库根目录串行执行，编号、seed 和配置一一对应；失败先排查，不使用无效 run 进入比较。以下是未来命令：

```bash
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C002a.yaml --seed 42 --tag C-002-a
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C002b.yaml --seed 42 --tag C-002-b
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C002c.yaml --seed 2026 --tag C-002-c
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C002d.yaml --seed 2026 --tag C-002-d
```

记录实际运行路径、起止时间、设备、显存、是否独占、实际 epoch、每轮学习率、最优 epoch 与停止原因。保留每组 `config.yaml`、`best.pt`、`last.pt`、`metrics.csv` 和 `train.log`。历史 A-001 约 5.7 分钟/轮；四组各跑满 12 轮粗估约 4.6 小时，实际耗时受硬件、早停和负载影响；评估时间另计。

## 评估与判读

用每次实际运行目录的配置和 checkpoint 评估；以下以 C-002-a 为例，`<run>` 替换为真实目录，其余组同步替换编号：

```bash
python src/run_eval.py --config outputs/<run>/config.yaml --checkpoint outputs/<run>/best.pt \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-002-a.csv
```

保留每组 valid 的 Study_id、标签及连续概率，核验 `Study_id -> Subject_id`、标签及数据版本一致。汇总每组 Macro-AUC、F1@0.5、阈值曲线、逐类 AUC / 支持数、训练和验证 loss、最佳 epoch 与实际轮数；AUC 使用 study 级连续概率，F1 使用阈值。

分别计算 **Δ42 = AUC(C-002-b) − AUC(C-002-a)** 和 **Δ2026 = AUC(C-002-d) − AUC(C-002-c)**，报告各 seed 的差值、两者平均差及逐类变化；在同一 valid 上以相同 study 有放回重采样做配对 bootstrap，保留随机种子、重采样次数和退化类处理。该区间仅反映验证集抽样不确定性，不涵盖训练随机性，也不能消除用同一 valid 多次选型造成的偏差。

两次同向且平均增益达到约 +0.005，可作为优先复核的方向性筛选条件，不是显著性证明；若两次均小幅正向、低于 +0.005，结合区间和逐类表现保留进一步验证价值；若方向不一致或出现明显类别退化，则先补充独立验证 / 更多配对 seed，不据单次最佳分数切换。两 seed 不足以证明泛化；valid 已被多次使用，后续须独立验证或 OOF。若作为融合候选，仅在核验数据和标签一致后与现有融合比较，固定公式后再考虑 test 提交。

完成训练后将实际结果及未完成项回填 `docs/experiments.md`、`docs/training_log.md`、本计划及 `CHANGELOG.md`；不填写未执行的成绩。
