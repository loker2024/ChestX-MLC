# 训练计划 25：C-005 正侧位权重四组对照

**状态：计划和四份独立配置已准备；GPU 训练和评估均未执行。**

## 目标与实验矩阵

检验 A-001 的固定正侧位聚合权重能否改善 study 级 Macro-AUC。沿用 A-001 的正位权重 0.7 为历史参考，四组训练各只改变 `model.frontal_weight`。全部使用 seed 42，从相同原始预训练权重开始全量训练，不从 A-001 checkpoint 热启动。

| 实验 / tag | 独立配置 | 正位权重 | 侧位权重 | 角色 |
| --- | --- | ---: | ---: | --- |
| A-001 | `configs/exp_A001.yaml` | 0.7 | 0.3 | 历史参考 |
| C-005-a | `configs/exp_C005a.yaml` | 0.5 | 0.5 | 候选 |
| C-005-b | `configs/exp_C005b.yaml` | 0.6 | 0.4 | 候选 |
| C-005-c | `configs/exp_C005c.yaml` | 0.8 | 0.2 | 候选 |
| C-005-d | `configs/exp_C005d.yaml` | 0.9 | 0.1 | 候选 |

四份配置完整展开、不使用 `_base`，编号与训练 tag、预测文件一一对应。按当前配置加载器展开后，各组与 A-001 仅 `model.frontal_weight` 不同；四组间也仅此字段不同。此实验不叠加 C-004 的平方根权重，固定 `train.pos_weight_power=1.0`，避免改变两个变量。

固定项为：原始 train/valid、无重划分、320×320 letterbox、ImageNet 归一化、训练增强保持代码现状、无水平翻转；ConvNeXt-Tiny `fb_in1k` ImageNet-1k 预训练、全量微调、标准分类头、dropout 0；`view_weighted` logits 聚合；BCE + pos_weight、AdamW、统一 lr=3e-4、weight_decay=0.01、batch=16 studies；8 epoch 上限、patience=2、warmup=2、cosine、lr_min=1e-5、AMP、grad_clip=5。实际停止轮数可能因早停不同。

本计划沿用 A-001 的 8 轮口径，与 C-002 / C-003 的 12 轮口径不同；不得把这些实验之间的差异单独归因于正侧位权重。C-002 选择的学习率不自动覆盖本实验。

## 聚合规则与研究边界

对于一次检查，同时存在已知正位和已知侧位，且没有未知视角时：

```text
study_logits = 正位权重 × 正位图 logits 均值 + (1 − 正位权重) × 侧位图 logits 均值
study_probability = sigmoid(study_logits)
```

只有一种视角，或存在未知视角时，现有实现回退为所有图片的 logits 均值，权重直接计算不生效。即便如此，改变训练时的聚合会改变参数更新，因此可能间接影响所有 study 的最终预测。

`frontal_weight=0.5` 是正位组和侧位组各占一半，不等于每张图等权：例如两张正位、一张侧位时，三张图分别获得 0.25、0.25、0.5。A-001-D 的 `mean` 也不能代替 C-005-a。训练、验证和推理均使用对应配置的权重，避免训练／评估口径不一致。

## 开始前核验

1. 记录代码 HEAD、`git status --short`、未提交 diff、依赖版本、设备、GPU 占用及运行环境；四组保持相同代码和环境。A-001 为历史训练，需对照历史源码、日志和配置核验数据加载、增强、聚合、优化器与调度；无法核验时仅称历史比较，不严格归因。
2. 固定原始 train/valid，核验患者、study、图像索引、标签和数据版本哈希，确认无患者泄漏；核对 `Study_id -> Subject_id` 和标签与 A-001 的 valid 一致。
3. 核验 `convnext_tiny.fb_in1k` 权重来源、许可、revision 和 SHA-256；四组使用同一文件，并尽量核对与 A-001 相同。启动日志及每个 `best.pt` 的 `metadata.pretrained_loaded` 均须确认预训练加载成功，否则该 run 无效。
4. 统计 train/valid 的 study 总数、仅正位、仅侧位、正侧位均有且无未知视角、含未知视角的数量及比例。四种类别按顺序定义为互斥：先判断是否含未知，再判断仅正位、仅侧位、同时正侧位。该统计用于解释有效作用范围，不作为修改方案或结果门槛。
5. 在本机运行下列检查。配置检查将候选权重恢复为 0.7 后，完整展开字典必须与 A-001 相等；训练环境还需核验真实运行配置、代码和数据。测试通过只表明本地验证，不表示 GPU 实验完成。

```bash
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path('src').resolve()))
from common.config import load_config
base = load_config('configs/exp_A001.yaml').to_dict()
for suffix, weight in [('a', 0.5), ('b', 0.6), ('c', 0.8), ('d', 0.9)]:
    path = Path(f'configs/exp_C005{suffix}.yaml')
    assert '_base' not in yaml.safe_load(path.read_text(encoding='utf-8'))
    candidate = load_config(path).to_dict()
    assert candidate['model']['frontal_weight'] == weight
    candidate['model']['frontal_weight'] = 0.7
    assert candidate == base, f'{path} 存在其他配置差异'
    print(f'C-005-{suffix}: {weight} / {1-weight:.1f}')
PY
.venv/bin/python -m pytest tests/test_config.py tests/test_model.py tests/test_resume.py -q
git diff --check
```

## 训练与预算

在训练环境仓库根目录串行执行；编号、配置和 tag 固定，四组训练按原定规则完成。异常中断先记录和排查；不根据先运行组的得分临时取消或更改后续组。

```bash
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C005a.yaml --seed 42 --tag C-005-a
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C005b.yaml --seed 42 --tag C-005-b
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C005c.yaml --seed 42 --tag C-005-c
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C005d.yaml --seed 42 --tag C-005-d
```

各组保留 `config.yaml`、`best.pt`、`last.pt`、`metrics.csv`、`train.log` 和可用的 `tb/`；记录真实运行目录、设备、显存峰值、开始结束时间、实际 epoch、每轮 lr 和耗时、最佳 epoch 与停止原因。OOM、数值异常或预训练未加载时记录原因，失败 run 不计入正式比较。

最多 4×8=32 个 epoch；借用历史 A-001 约 5.7 分钟/轮粗估，跑满约 **3.0 小时**，不含启动和评估；早停、硬件及负载会改变实际时间。计划不包含额外 seed 或 A-001 重训。

## 验证集评估

以下环境变量设置为四个真实训练目录，均从各自配置与 checkpoint 评估。逐 study 预测文件与编号对应，严禁覆盖历史预测：

```bash
: "${RUN_C005A:?请设置 C-005-a 的真实运行目录}"
: "${RUN_C005B:?请设置 C-005-b 的真实运行目录}"
: "${RUN_C005C:?请设置 C-005-c 的真实运行目录}"
: "${RUN_C005D:?请设置 C-005-d 的真实运行目录}"
python src/run_eval.py --config "$RUN_C005A/config.yaml" --checkpoint "$RUN_C005A/best.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-005-a.csv
python src/run_eval.py --config "$RUN_C005B/config.yaml" --checkpoint "$RUN_C005B/best.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-005-b.csv
python src/run_eval.py --config "$RUN_C005C/config.yaml" --checkpoint "$RUN_C005C/best.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-005-c.csv
python src/run_eval.py --config "$RUN_C005D/config.yaml" --checkpoint "$RUN_C005D/best.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-005-d.csv
python src/run_compare.py --runs "$RUN_C005A" "$RUN_C005B" "$RUN_C005C" "$RUN_C005D" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/comparisons/compare_C005
```

本机 `outputs/experiments/A-001` 有配置、指标和日志但没有 `best.pt`；`outputs/predictions/valid/A-001_valid_predictions.csv` 为历史逐 study 预测。核验其来源、数据版本、完整 ID、标签及 859 个 valid study 后，可直接用保存预测与四组结果同口径计算对照。**不应**把不含 checkpoint 的本机 A-001 目录传给 `run_compare.py`。若训练环境持有经核验的 A-001 完整运行目录，可额外设置 `RUN_A001` 并运行：

```bash
: "${RUN_A001:?请设置包含 config.yaml 和 best.pt 的 A-001 运行目录}"
: "${RUN_C005A:?请设置 C-005-a 的真实运行目录}"
: "${RUN_C005B:?请设置 C-005-b 的真实运行目录}"
: "${RUN_C005C:?请设置 C-005-c 的真实运行目录}"
: "${RUN_C005D:?请设置 C-005-d 的真实运行目录}"
python src/run_compare.py --runs "$RUN_A001" "$RUN_C005A" "$RUN_C005B" "$RUN_C005C" "$RUN_C005D" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/comparisons/compare_A001_C005
```

若历史预测或数据映射无法核验，暂停基于 A-001 的配对分析，只报告各组自身指标与限制。所有预测须包含 study ID、10 类真实标签及连续概率；从 valid 索引核验 `Study_id -> Subject_id`，按字符串读取 ID，不静默取交集。

## 结果判读

- 主指标为 study 级连续概率 Macro-AUC；辅助报告 F1@0.5、阈值曲线、各类 AUC / F1 / 正样本支持数、训练和验证 loss、最佳 epoch、耗时及显存。对权重直接作用的「正侧位齐全且无未知视角」子集另作描述性分析，报告样本数和各类正负支持数；类别缺少正例或负例时 AUC 标为不可计算。子集结果不替代全体主指标。
- 每组计算 ΔAUC = AUC(C-005 候选) − AUC(A-001)。预先固定筛选门：ΔAUC ≥ +0.005，且无任何类别 AUC 下降 ≥0.02，才进入优先复核；0 < ΔAUC < 0.005 为弱正向，未达门不切换默认。逐类下降应结合支持数和不确定性解释；阈值为方向性筛选，不是显著性证明。
- 对各候选与基线使用相同验证样本进行配对 bootstrap：seed=2026，目标 2,000 次有效重采样；同一患者若有多个 study，按患者簇抽样，否则按 study 抽样，每次所有模型使用相同抽样索引。任一类别仅有单一标签的抽样丢弃，最多尝试 20,000 次；报告有效次数、每组 ΔAUC 和 2.5%/97.5% 分位区间；不足 2,000 次时不报告稳定区间。现有 `run_compare.py` 不提供此 bootstrap；执行时需另写并保存分析脚本、输入和结果。
- 四组在同一验证集上比较并选最高分会产生多重比较和选择偏差；逐一的普通 bootstrap 区间不校正该偏差。报告四组全部结果，优先复核通过筛选门的候选，不把最高分直接当成独立验证结论。仅一个 seed，区间不包含训练随机性；若进入候选，需考虑独立验证或 OOF，不自动追加训练。

完成后回填 `docs/experiments.md`、`docs/training_log.md`、本计划和 `CHANGELOG.md`，保留四组预测、比较目录、bootstrap 脚本和结果；不预填成绩，不自动生成 test 提交。
