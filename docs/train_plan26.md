# 训练计划 26：C-006 亮度与对比度增强四组对照

**状态：已执行完毕（2026-11-01 a/b/c；2026-11-02 d 重跑），四组结果与判读见文末「执行记录」。**

## 目标与实验矩阵

检验训练时亮度或对比度扰动幅度对 ConvNeXt-Tiny study 级 Macro-AUC 的影响。以 A-001 的 `brightness=0.15`、`contrast=0.15` 为历史参考，四组分别弱化或加强一项增强；其余条件固定。两个参数为 torchvision `ColorJitter` 的幅度：例如 0.05 对应因子在 0.95–1.05 抽样，0.25 对应 0.75–1.25。亮度和对比度使用不同变换，不能视为同一种强度。

| 实验 / tag | 完整独立配置 | brightness | contrast | 唯一改变 |
| --- | --- | ---: | ---: | --- |
| A-001 | `configs/exp_A001.yaml` | 0.15 | 0.15 | 历史参考，旧配置缺字段时使用兼容默认值 |
| C-006-a | `configs/exp_C006a.yaml` | 0.05 | 0.15 | 减弱亮度 |
| C-006-b | `configs/exp_C006b.yaml` | 0.25 | 0.15 | 加强亮度 |
| C-006-c | `configs/exp_C006c.yaml` | 0.15 | 0.05 | 减弱对比度 |
| C-006-d | `configs/exp_C006d.yaml` | 0.15 | 0.25 | 加强对比度 |

所有 YAML 独立完整，不含 `_base`；编号与 tag、预测文件一一对应。训练图像保持原有增强顺序：letterbox → RandomAffine（旋转 ±7°、平移 5%、缩放 0.9–1.1）→ ColorJitter → 归一化；无水平翻转。每张图独立抽样，同一 study 的多张图不强制共享扰动。增强仅用于训练，valid/test 使用确定性预处理。

固定原始 train/valid，不重划分；320×320、ImageNet 归一化、ConvNeXt-Tiny `fb_in1k` 预训练、标准头、全量微调、`view_weighted` logits 聚合正侧位 0.7/0.3；BCE + pos_weight、`pos_weight_power=1.0`、AdamW、lr=0.00015、weight_decay=0.01、batch 16 studies、seed 42、12 epoch 上限、patience 2、warmup 2、cosine、lr_min=1e-5、AMP。固定的是上限与早停规则，实际轮数允许不同。C-004 的平方根权重和 C-005 的聚合权重不叠加；训练口径已于 2026-11-01 统一为 C-002-b 口径（12 epoch 上限、patience 2、lr=0.00015）。

## 开始前核验

1. 记录代码 HEAD、`git status --short`、未提交 diff、依赖版本、设备和 GPU 负载；四组保持同一代码和环境。由于 A-001 是历史运行，需核验其代码、日志、数据加载、增强、loss、优化器与调度。新代码默认亮度和对比度各为 0.15，且本机测试固定随机种子下输出与原流程一致；这不能替代历史运行代码及数据来源核验。若有不可核验差异，只能作历史参考，不能严格归因。
2. 检查原始 train/valid 的患者、study、标签、图像索引和数据版本哈希，确认无患者泄漏。核验 A-001 预测对应的 `Study_id -> Subject_id` 与标签一致。
3. 核验 `convnext_tiny.fb_in1k` 预训练文件的来源、许可、revision 与 SHA-256；四组统一使用同一文件并核对 A-001 的对应记录。启动日志与 best.pt 的 `metadata.pretrained_loaded` 均须确认加载成功；失败 run 不进入正式比较。
4. 本机核验四个配置除指定增强字段与 `train.early_stop_patience` 外与 C-002-b 展开配置一致；运行配置、增强及回归测试：

```bash
.venv/bin/python - <<'PY'
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path('src').resolve()))
from common.config import load_config
base = load_config('configs/exp_C002b.yaml').to_dict()
for suffix, field, value in [('a','brightness',0.05), ('b','brightness',0.25), ('c','contrast',0.05), ('d','contrast',0.25)]:
    path = Path(f'configs/exp_C006{suffix}.yaml')
    assert '_base' not in yaml.safe_load(path.read_text(encoding='utf-8'))
    candidate = load_config(path).to_dict()
    assert candidate['image'][field] == value
    candidate['image'][field] = base['image'][field]
    candidate['train']['early_stop_patience'] = base['train']['early_stop_patience']
    assert candidate == base, f'{path} 存在其他配置差异'
    print(f'C-006-{suffix}: image.{field}={value}')
PY
.venv/bin/python -m pytest tests/test_augmentation_config.py tests/test_config.py tests/test_dataset.py tests/test_resume.py tests/test_review_fixes.py -q
git diff --check
```

## 训练与预算

四组依次执行，均从同一原始预训练权重开始，不从 A-001 checkpoint 热启动；除故障外，不依据前面组的单次成绩提前取消或修改后续组。

```bash
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C006a.yaml --seed 42 --tag C-006-a
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C006b.yaml --seed 42 --tag C-006-b
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C006c.yaml --seed 42 --tag C-006-c
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C006d.yaml --seed 42 --tag C-006-d
```

每组保留 `config.yaml`、`best.pt`、`last.pt`、`metrics.csv`、`train.log` 和可用的 `tb/`；记录设备、实际运行目录、开始结束时间、显存峰值、每轮 lr / 耗时、最佳 epoch 与停止原因。预训练未加载、OOM 或数值异常先记录并排查，失败 run 不进入正式比较。

四组最多 48 个 epoch；按历史 A-001 约 5.7 分钟/轮粗估跑满约 4.6 小时，不含启动和评估。实际由设备、负载和早停决定。计划不要求额外 seed 或重训 A-001。

## 验证集评估

设置四个变量为真实运行目录，使用每组自身配置与 checkpoint。输出文件名与实验编号一致，不覆盖历史预测：

```bash
: "${RUN_C006A:?请设置 C-006-a 的真实运行目录}"
: "${RUN_C006B:?请设置 C-006-b 的真实运行目录}"
: "${RUN_C006C:?请设置 C-006-c 的真实运行目录}"
: "${RUN_C006D:?请设置 C-006-d 的真实运行目录}"
python src/run_eval.py --config "$RUN_C006A/config.yaml" --checkpoint "$RUN_C006A/best.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-006-a.csv
python src/run_eval.py --config "$RUN_C006B/config.yaml" --checkpoint "$RUN_C006B/best.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-006-b.csv
python src/run_eval.py --config "$RUN_C006C/config.yaml" --checkpoint "$RUN_C006C/best.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-006-c.csv
python src/run_eval.py --config "$RUN_C006D/config.yaml" --checkpoint "$RUN_C006D/best.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-006-d.csv
python src/run_compare.py --runs "$RUN_C006A" "$RUN_C006B" "$RUN_C006C" "$RUN_C006D" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/comparisons/compare_C006
```

本机 `outputs/experiments/A-001` 保留配置、指标和日志，但不含 `best.pt`；历史逐 study 预测在 `outputs/predictions/valid/A-001_valid_predictions.csv`。先核验来源、ID、标签、数据版本及 859 个 valid study，再用保存预测同口径计算基线和配对差值。不能把缺少 checkpoint 的目录传给 `run_compare.py`。如果训练环境有经核验的 A-001 完整运行目录，才执行：

```bash
: "${RUN_A001:?请设置包含 config.yaml 和 best.pt 的 A-001 目录}"
: "${RUN_C006A:?请设置 C-006-a 的真实运行目录}"
: "${RUN_C006B:?请设置 C-006-b 的真实运行目录}"
: "${RUN_C006C:?请设置 C-006-c 的真实运行目录}"
: "${RUN_C006D:?请设置 C-006-d 的真实运行目录}"
python src/run_compare.py --runs "$RUN_A001" "$RUN_C006A" "$RUN_C006B" "$RUN_C006C" "$RUN_C006D" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/comparisons/compare_A001_C006
```

所有预测须保留 Study_id、10 类真实标签及连续概率；从 valid 索引核验 `Study_id -> Subject_id`，ID 按字符串处理，不静默取交集。历史预测或数据映射无法核验时暂停配对分析，单独报告各组指标及限制。

## 判读与回填

- 主指标是 study 级连续概率 Macro-AUC；辅助报告 F1@0.5、阈值曲线、逐类 AUC / F1 / 支持数、训练和验证 loss、最佳 epoch、耗时与显存。不能把阈值 F1 改善等同于 AUC 改善。
- 每组 ΔAUC = AUC(C-006) − AUC(A-001)。预设 ΔAUC ≥ +0.005 且无任何类别 AUC 下降 ≥0.02 才进入优先复核；0 < ΔAUC < 0.005 为弱正向，其余不切换默认。门槛为方向性筛选，不代表统计显著；逐类结果应结合正负支持数解释。
- 配对 bootstrap 使用 seed 2026，目标 2,000 次有效重采样；患者有多个 study 时按患者簇抽样，否则按 study 抽样，同次所有模型共用索引。抽样后任一类只有单一标签则丢弃，最多尝试 20,000 次。报告有效次数、各组 ΔAUC 及 2.5%/97.5% 分位区间；有效次数不足则不报告稳定区间。现有 `run_compare.py` 不提供该 bootstrap，执行时单独编写并保留分析脚本、输入和结果。
- 四组使用同一 valid 选最高分会产生多重比较及选型偏差，普通逐一 bootstrap 不校正这些偏差；报告全部结果，不把最高分直接当成独立验证结论。单 seed 区间不涵盖训练随机性；候选需考虑独立验证或 OOF，不自动安排额外训练。
- 完成后回填 `docs/experiments.md`、`docs/training_log.md`、本计划与 `CHANGELOG.md`，保留预测、比较目录、bootstrap 脚本和结果。不预填成绩或自动生成 test 提交。

## 执行记录（2026-11-01 起，d 组 2026-11-02 重跑）

a/b/c 于 21:27:40–22:56:17 串行执行，autodl Tesla V100-SXM2-32G 独占，均沿用 C-002-b 训练口径（12 轮 / patience 2 / lr 1.5e-4）。d 组 11-01 22:56 首跑被实例重启打断（1 轮、无 `best.pt`），2026-11-02 12:23:14 在同一 HEAD `64d03da`、同 seed、同型号 GPU 上从头重跑，12:55:15 训练收尾、12:55:57 评测完成；中断残留归档到远端 `outputs/_aborted/20261101-225624_C-006-d`，未复用。

| 编号 | 唯一改变 | 起止（CST） | 耗时 | epochs | best | macro-AUC | F1@0.5 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| C-006-a | brightness 0.15→0.05 | 11-01 21:27:40–21:59:24 | 31m44s | 7/12 | 5 | 0.7957 | 0.4348 |
| C-006-b | brightness 0.15→0.25 | 11-01 21:59:31–22:27:41 | 28m10s | 6/12 | 4 | **0.8014** | 0.4263 |
| C-006-c | contrast 0.15→0.05 | 11-01 22:27:47–22:56:17 | 28m30s | 6/12 | 4 | 0.7927 | 0.4156 |
| C-006-d（重跑） | contrast 0.15→0.25 | 11-02 12:23:14–12:55:15 | 32m01s | 7/12 | 5 | 0.8009 | **0.4371** |

四组已按四组统一重算（`run_compare.py` → `outputs/comparisons/compare_C006/`），`valid_ids_sha256` 一致（`81e8be7f…`），均为 859 studies 的 study 级宏平均。

**判读**

1. **四组都没过 +0.005 门。** 相对 C-002-b（0.7976 / 0.4249）：a **−0.0019**、b **+0.0038**、d **+0.0033**、c **−0.0049**。b 与 d 落在本计划的「弱正向」区间（0 < ΔAUC < 0.005），但 9 类口径分别为 **−0.0011** 与 **−0.0035**，增益全部由基线 class1（0.6609，12 组最低）造成，剔除即转负。相对 A-001 分别是 −0.0081 / −0.0024 / **−0.0029** / −0.0111，无一接近。
2. **逐类降幅门已补完**（本计划第 3 节预置：任何类别 AUC 降 ≥0.02 即不进入优先复核）：a 触发（class4 **−0.0331**）、c 触发（class2 **−0.0251**）、b 通过（最低 −0.0108）、**d 通过（最低 class8 −0.0164）**。d 是唯一同时满足「组级正向」与「逐类不触门」的组合，但组级 0.0033 距门差 0.0017，按规则不作优先复核。
3. **对比度这一对构成机制证据。** c（0.05）F1 峰右移到 0.7 且到 0.7 仍上升；d（0.25）峰回到 0.6，峰值 **0.4493**、F1@0.5 **0.4371** 均为四组最高。峰位对对比度扰动呈单调响应，说明 c 的偏移是「对比信息不足 → 阳性概率整体偏低」的可复现表现，而不是单组噪声。
4. **收敛节奏**：b、c 第 4 轮达峰、第 6 轮早停；a、d 第 5 轮达峰、第 7 轮早停。四组在 12 轮上限里只用到 6–7 轮，早峰压制是共同形态；d 后段过拟合最重（valid_loss 0.9128→1.4312）。

**未完成**：配对 bootstrap（本计划第 4 节口径，患者整簇 2,000 次、seed 2026）、第二 seed、`brightness` × `contrast` 交互扫描、四组选最高分的多重比较校正。现有证据只能说明「亮度与对比度扰动在这一区间内不是有效杠杆」，仍不能给出最优增强强度。
