# 训练计划 13：A-004 分类头 dropout 与 weight decay 组合对照

**状态：已完成（2026-10-31，训练与评估均在 autodl V100-SXM2-32G 上执行；valid macro-AUC 0.7988 / F1@0.5 0.4233，未过方向性门 0.8088）。**

## 目标与配置

以 A-001 为基线，检查增强正则化后验证集排序表现及训练曲线是否改善。A-001 的记录值为 valid Macro-AUC **0.8038**、F1@0.5 **0.4255**（[训练记录](training_log.md)）；本次已在 valid 上独立重算 A-001（0.8038 / 0.4255，与历史记录一致），方向性门取重算值 + 0.005 = 0.8088。A-004 使用 [exp_A004.yaml](../configs/exp_A004.yaml)，继承完整展开的 A-001 配置，仅覆盖下表两项：

| 参数 | A-001 | A-004 |
| --- | --- | --- |
| `model.dropout` | 0.0 | 0.2 |
| `train.weight_decay` | 0.01 | 0.05 |

`model.dropout` 通过 timm 的 `drop_rate` 作用于 ConvNeXt 分类头；训练时启用，评估时关闭，不是随机深度。AdamW 使用 `weight_decay`，当前优化器对全部可训练参数使用同一个参数组。两项值是本次实验设置，并非已验证最优值。

其余保持 ConvNeXt-Tiny `fb_in1k` 预训练、全量微调、320×320 letterbox / ImageNet 归一化、关闭水平翻转、study 正侧位 7:3 加权聚合、BCE + `pos_weight`、batch 16 studies、lr 0.0003、warmup 2、cosine 至 0.00001、最多 8 轮、patience 2、seed 42。训练与验证数据划分必须与 A-001 相同；固定代码版本、timm 版本、权重文件及 SHA-256，检查展开后的配置，避免继承变化影响对照。

## 执行

1. 在训练环境运行 `python -m pytest tests/test_a004.py -q`，核验配置仅有两个预期差异，实际分类头 dropout 为 0.2；检查运行时 train/eval 状态、模型输出 10 类 logits。确认预训练权重来源、许可和实际哈希与 A-001 相同；加载失败或回退随机初始化时停止并修复。
2. 运行 `python src/check_data.py --config configs/exp_A004.yaml --full`，确认数据版本、study 与患者划分及图片完整性与基线一致。
3. 已缓存权重时，在仓库根目录执行：

   ```bash
   HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A004.yaml --seed 42 --tag A-004_convnext_tiny_dropout_wd_s42
   ```

   确认日志显示 `预训练权重: 已加载`，checkpoint 的 `metadata.pretrained_loaded` 为 true；保存实际配置、代码版本、设备、显存、耗时、停止原因、`train.log`、`metrics.csv`、`best.pt` 和 `last.pt`。
4. 将以下目录占位符替换为实际运行目录；当两个 run 都具有 `config.yaml` 和 `best.pt`，且确认使用同一 valid 数据时，统一重算：

   ```bash
   python src/run_compare.py --runs outputs/<A-001运行目录> outputs/<A-004运行目录> --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/compare_A001_A004
   python src/run_eval.py --config outputs/<A-001运行目录>/config.yaml --checkpoint outputs/<A-001运行目录>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/A-001_valid_predictions.csv
   python src/run_eval.py --config outputs/<A-004运行目录>/config.yaml --checkpoint outputs/<A-004运行目录>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/A-004_valid_predictions.csv
   ```

   保留汇总、逐类报告与逐 study ID、标签和概率；核对 ID、标签和类别顺序。若 A-001 原始产物不可用，报告记录值及其来源，不声称已完成独立配对重算。

## 判读

- 以 Macro-AUC 为主，报告相对 A-001 的差值及逐类 AUC 和支持数；报告 F1@0.5 与完整阈值曲线。历史方向性门为 AUC 增加 0.005，即参考 0.8088；若重新评估了 A-001，则使用其重算值加 0.005。该门不是统计显著性检验。
- 比较每轮训练 loss、valid loss、AUC、最佳轮次及停止原因。验证集指标改善或 loss 差距缩小可以作为正则化方向的线索，不能单独证明泛化改善；训练与验证同时下降也可能涉及优化或训练预算。
- 本组同时改变 dropout 和 weight decay，只能评价组合，不可分别归因。若结果有潜力，补充单因素实验及同 seed 2026 配对；使用逐 study 预测进行配对 bootstrap，并保留重复使用同一 valid 选型可能带来的偏差。
- 训练完成后按真实结果回填 `docs/experiments.md`、`docs/training_log.md` 和 `CHANGELOG.md`。本计划不声称训练已执行，不预设收益。

## 执行记录

2026-10-31 在 autodl V100-SXM2-32G 上执行，代码版本 `ad336f2`（`exp/A-004` 分支）：

- 验证：本机与训练环境 `python -m pytest tests/test_a004.py -q` 均 6 passed；`check_data.py --full`
  全部通过（跨 split Subject/Study/Dicom_id 无重叠）；`train.log` 确认「预训练权重: 已加载」，
  `best.pt` metadata `pretrained_loaded: true`；预训练权重缓存文件 SHA-256
  `08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276`、114,374,272 B、commit
  `b43a6303…`，与 A-001 记录一致。
- 训练：12:13:44 启动，12:46:00 结束（32m16s），第 7 轮早停（上限 8 / patience 2），best epoch 5；
  显存峰值约 14.0 GB。
- 结果：valid macro-AUC **0.7988** / F1@0.5 **0.4233**（`run_eval.py` 重算）；A-001 独立重算
  **0.8038** / **0.4255**（与历史记录一致）；差 **AUC −0.0050、F1@0.5 −0.0022**，距方向性门 0.8088
  差 **−0.0100 → 未过门**。
- 训练行为：各轮 train_loss 高于 A-001（+0.005~+0.026），valid_loss 走势未变、仍在第 4 轮见底后反弹
  （第 5–6 轮略低于 A-001）→ 正则化未推迟过拟合。
- 统计判读：配对 bootstrap（859 studies × 2,000 次）ΔAUC（A-001 − A-004）95% CI [−0.0061, +0.0167]；
  单 seed、CI 跨 0，不能判定真实退化，但没有增益证据。
- 产物：`outputs/20261031-121344_A-004_convnext_tiny_dropout_wd_s42/`、
  `outputs/eval_reports/A-004_full.txt`、`outputs/A-004_valid_predictions.csv`、`outputs/compare_A001_A004/`。
- 结论：本组同时改 dropout 与 weight decay，只能评价组合；A-001 口径保持不变，默认配置不切换。
  若继续，应拆成两个单变量组并补 seed 2026 配对。
