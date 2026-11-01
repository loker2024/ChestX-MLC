# 训练计划 24：C-004 平方根类别权重实验

**状态：计划及独立配置已准备；尚未进行 GPU 训练或评估。**

## 目标与假设

检验在 A-001 的 ConvNeXt-Tiny 训练口径中，把 BCE 正样本权重由 `负 study 数 / 正 study 数` 改为其平方根，是否改善 study 级 Macro-AUC。权重只由训练集的 study 标签统计，不按图像数统计；全正或全负类别仍设为 1。该方案不预设一定改善结果。

| 实验 | 角色 | 配置 | seed | `pos_weight_power` |
| --- | --- | --- | ---: | ---: |
| A-001 | 历史参考 | `configs/exp_A001.yaml` / 历史运行配置 | 42 | 1.0（旧配置未写入时的兼容默认值） |
| C-004 | 平方根权重候选 | `configs/exp_C004.yaml` | 42 | 0.5 |

C-004 配置完整展开，不含 `_base`。按当前代码展开后，除 `train.pos_weight_power` 外与 C-002-b 配置完全一致。固定 320×320 letterbox、ImageNet 归一化、无水平翻转、ConvNeXt-Tiny ImageNet-1k 预训练、全量微调和标准头；正侧位 logits 聚合 0.7/0.3；BCE + pos_weight、AdamW、lr=0.00015、weight_decay=0.01、batch 16 studies、12 epoch 上限、patience 2、warmup 2、cosine、lr_min=1e-5、AMP。实际轮数可以不同。

训练口径于 2026-11-01 统一：`train.epochs=12`、`train.early_stop_patience=2`、`train.lr=0.00015`（与 C-002-b 相同），因此 C-004 不再沿用 A-001 的 8 轮 / patience 2 / 3e-4 口径。类别权重之外的干净对照是 320 分辨率下的 C-002-b（其 patience 为 4），A-001 的 0.8038 只作历史参考，不能作为同口径基线直接相减。

## 训练前核验

1. 在仓库根目录记录 `git rev-parse HEAD`、`git status --short`、相关未提交 diff、依赖版本、设备以及 GPU 占用，保留 C-004 训练的代码快照。A-001 是历史运行，尤其应对照其日志、记录和历史代码，检查数据加载、增强、损失、优化器及调度等路径是否变化；新增指数参数本身为预期变化。无法核实历史差异时，比较只能称历史参考，不能严格归因。
2. 固定原始 train/valid，不重划分；核对患者、study、标签、图像索引和数据版本，保留哈希，确认无患者泄漏。检查 `Study_id -> Subject_id`、标签与 A-001 验证集一致。
3. 核对 `convnext_tiny.fb_in1k` 预训练文件来源、许可、revision 与 SHA-256 是否同 A-001；核对历史 A-001 日志、运行配置、权重来源和训练设备。训练日志及 C-004 的 best.pt 中 `metadata.pretrained_loaded` 必须表明预训练加载成功，否则该 run 无效。
4. 运行以下配置验证，预期只输出 `train.pos_weight_power: 1.0 -> 0.5`（比对基准为同口径的 C-002-b，不再是 A-001），失败则先排查；不得把该本地验证当作 GPU 结果。

```bash
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src').resolve()))
from common.config import load_config
base = load_config('configs/exp_C002b.yaml').to_dict()
candidate = load_config('configs/exp_C004.yaml').to_dict()
assert base['train']['pos_weight_power'] == 1.0
assert candidate['train']['pos_weight_power'] == 0.5
candidate['train']['pos_weight_power'] = 1.0
assert candidate == base, '除指数外存在其他配置差异'
print('train.pos_weight_power: 1.0 -> 0.5')
PY
.venv/bin/python -m pytest tests/test_pos_weight_power.py tests/test_config.py tests/test_model.py tests/test_resume.py -q
git diff --check
```

## 训练

训练环境在仓库根目录执行；新训练从原始预训练权重开始，不使用 A-001 checkpoint 热启动。运行一次，不默认重跑历史 A-001 或追加其他 seed。

```bash
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C004.yaml --seed 42 --tag C-004
```

记录真实运行目录、启动和结束时间、GPU 型号、显存峰值、每轮学习率/耗时、最优 epoch 和早停原因。保留 `config.yaml`、`best.pt`、`last.pt`、`metrics.csv`、`train.log` 和可用的 `tb/`。预训练未加载、OOM 或数值异常时排查并记录，失败 run 不进入正式比较。

预算：最多 12 轮；参考历史 A-001 每轮约 5.7 分钟，跑满约 70 分钟，不含评估和启动，实际以当前设备为准。

## 评估与产物

设置 `RUN_C004` 为 C-004 的真实运行目录，使用该 run 自身配置和 checkpoint 评估：

```bash
: "${RUN_C004:?请设置 C-004 的真实运行目录}"
python src/run_eval.py --config "$RUN_C004/config.yaml" --checkpoint "$RUN_C004/best.pt" \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 \
  --pred-out outputs/predictions/valid/C-004.csv
```

本机历史 A-001 归档含 `config.yaml`、`metrics.csv`、`train.log` 和 `outputs/predictions/valid/A-001_valid_predictions.csv`，**不含 best.pt**。先核验历史预测的来源、ID、标签和 859 个 valid study；若可用，使用保存的 study 预测与 C-004 作同一口径重算和配对比较。不得要求 `run_compare.py` 使用无 checkpoint 的本机目录。如果训练环境有经核验的 A-001 best.pt，才使用以下可选命令，并记录真实目录：

```bash
: "${RUN_A001:?请设置包含 config.yaml 和 best.pt 的 A-001 真实运行目录}"
: "${RUN_C004:?请设置 C-004 的真实运行目录}"
python src/run_compare.py --runs "$RUN_A001" "$RUN_C004" \
  --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 \
  --out outputs/comparisons/compare_A001_C004
```

若历史预测、标签映射或数据版本无法核验，暂停配对分析；单独报告 C-004 的指标和限制，不虚构严格对照。两份预测都保存 `Study_id`、10 类真实标签及连续概率；通过 valid 索引核验 `Study_id -> Subject_id`，ID 当作字符串处理，不静默取交集。

## 判读

- 主指标：study 级连续概率的 Macro-AUC；辅助报告 F1@0.5、阈值曲线、逐类 AUC / F1 / 正样本支持数、最佳 epoch、训练与验证 loss、耗时和显存。相同阈值的 F1 用于辅助，不把阈值改善等同于 AUC 改善。
- ΔAUC = AUC(C-004) − AUC(A-001)。仅当 ΔAUC ≥ +0.005，且没有任何类别 AUC 相比 A-001 下降 ≥0.02，才进入优先复核候选；0 < ΔAUC < 0.005 为弱正向，其他结果不切换默认。类别退化需结合支持数和不确定性解释；这些门槛是方向性筛选，并非显著性检验。
- 配对 bootstrap：验证预测与患者映射一致后，seed=2026，目标 2,000 次有效重采样。若同一患者有多个 study，按患者簇抽样并保留其全部 study；否则按 study 抽样。每次两模型使用同一批抽样索引；任一类别抽样后仅有单一标签时丢弃该次，最多尝试 20,000 次。报告有效次数、ΔAUC 及其 2.5%/97.5% 分位区间；不足 2,000 次时不报告稳定区间。现有 `run_compare.py` 不实现此分析，执行时编写并保留分析脚本、输入文件及结果。
- 此区间只反映验证集抽样不确定性，不覆盖训练随机性；同一 valid 多次选型可能产生偏差。单 seed 不足以证明泛化，若进入候选，再考虑独立验证或 OOF；不自动添加 seed 或执行额外训练。

训练完成后，将实际配置、日志、指标、对照资格和未完成项回填 `docs/experiments.md`、`docs/training_log.md`、本文及 `CHANGELOG.md`，保留预测和比较产物。未经验证不填写结果，也不生成 test 提交。

## 执行记录（2026-11-01）

单组训练，18:50:32 启动、19:18:27 结束（27m55s），autodl Tesla V100-SXM2-32G 独占，`HF_HUB_OFFLINE=1`，预训练权重已加载。

```bash
python src/run_train.py --config configs/exp_C004.yaml --seed 42   # 18:50:32–19:18:27
```

| epochs | best | macro-AUC | F1@0.4 | F1@0.5 | F1@0.6 | F1@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6/12 | 4 | 0.7950 | **0.4362** | 0.4125 | 0.3894 | 0.3269 |

**判读：这是校准左移，不是判别能力变化。**

1. 阈值曲线峰从其余各组的 0.5/0.6 移到 **0.4**，0.5 处骤降、0.7 处崩到 0.3269。平方根权重压低正类权重，输出概率整体下移约一个档位。
2. 两个方向都会误读：F1@0.5 比基线 C-002-b 低 0.0124（看似变差），但 F1@0.4 = 0.4362 又比基线的 F1@0.5 高 0.0113（看似变好）。二者是同一位移的两面。按 AUC 判：−0.0026；剔除 class1 的 9 类口径 **−0.0090** → 排序是净损失。
3. 逐类 5 升 5 降，class1 +0.0550（吃基线 class1 偏低的红利）、class2 −0.0396（两个口径都降，是本组最实的退化点）。
4. 对提交无影响：赛题按概率排序算 macro-AUC，`run_infer.py` 默认输出全 10 类概率、不做阈值过滤。

**未完成**：配对 bootstrap、可靠性图与 ECE（校准结论仅由曲线峰位与两侧陡降推得，未做 logit 分布对比）、第二 seed。配置首行注释「与 A-001 展开配置仅 `pos_weight_power` 不同」已过时——字段值实为 lr 1.5e-4 / 12 轮 / patience 2 的 C-002-b 口径，判读以字段值为准。
