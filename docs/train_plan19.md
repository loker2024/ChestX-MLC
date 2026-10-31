# 训练计划 19：A-001 优化超参联合放大对照（lr 1e-3 / wd 0.03 / 10 轮 / patience 4）

**状态：已完成（2026-10-31），GPU 训练与评估已执行，结论：未过门、方向否定。**

## 目标与配置

在 A-001 的同一训练口径下，只改四项优化超参，检验「更大学习率 + 更强权重衰减 + 更长训练预算」
能否提升 study 级 Macro-AUC。候选配置 [exp_A009.yaml](../configs/exp_A009.yaml) 以
`_base: exp_A001.yaml` 继承 A-001，只覆盖四项：

| 字段 | A-001 | A-009 |
| --- | --- | --- |
| `train.lr` | 0.0003 | 0.001 |
| `train.weight_decay` | 0.01 | 0.03 |
| `train.epochs` | 8 | 10 |
| `train.early_stop_patience` | 2 | 4 |

其余口径逐字段继承 A-001：ConvNeXt-Tiny `fb_in1k` @320×320、ImageNet 归一化、无水平翻转、
正侧位 7:3 加权聚合（logits 空间）、标准分类头（dropout 0）、BCE + `pos_weight`、
`batch_studies: 16`、warmup 2 + cosine 到 1e-5、AMP、`grad_clip: 5.0`、seed 42。参数量与 A-001
相同（27,827,818），预训练权重与数据划分也与 A-001 相同，因此差值来自优化超参的组合改变。

## 与相关实验的区别

- A-004-b 只把 `train.weight_decay` 0.01 → 0.05（单变量）：0.7999 / 0.4238，差 AUC −0.0039，
  在 seed 波动带内；A-009 的 wd 只到 0.03，但同时放大 lr 约 3.3 倍并延长训练轮数。
- A-001 的 lr 峰值是 3e-4（warmup 2 轮后），A-009 为 1e-3；warmup 与 cosine 周期长度都随
  `epochs` 变化（10 轮 vs 8 轮），所以本组是「优化预算整体放大」的组合对照。
- 四项同改，差值不能拆成单项归因；若方向正面，需另开单变量实验（lr、wd 各一项）并补 seed 2026。

## 实现口径

纯配置实验，未改任何代码：

- `configs/exp_A009.yaml` 只写四项覆盖，并在文件头注释「四项同改、差值不可拆单项」；
- 训练、验证、推理均走 A-001 同一条路径（`src/run_train.py` / `src/run_eval.py`），
  聚合、损失、采样、AMP 与 grad clip 全部继承；
- 配置加载校验：训练启动时 `load_config` 正常展开，日志确认 `epochs 10`、warmup + cosine 调度、
  `pretrained_loaded: True` 与 27,827,818 可训练参数。

## 执行命令

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -u src/run_train.py --config configs/exp_A009.yaml --seed 42 \
    --tag A-009_convnext_tiny_lr1e-3_wd0.03_s42

HF_HUB_OFFLINE=1 .venv/bin/python src/run_eval.py --config configs/exp_A009.yaml \
    --checkpoint outputs/<run>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 \
    --pred-out outputs/A-009_valid_predictions.csv
```

训练前未重跑 `python -m pytest` 与 `check_data.py --full`：本组只新增配置、不改代码，数据与
A-001 / A-008 是同一份（`check_data --full` 结论沿用 A-008 的全通过记录）。

## 判读

- 对照基线 A-001（0.8038 / 0.4255，同骨干、同 seed 42、同设备），方向性门 = 0.8038 + 0.005 =
  0.8088；单 seed 差值在 ±0.0033 内视为波动。
- 若 AUC ≥ 0.8088 且 F1 曲线不整体左移，才进入 seed 2026 复核；否则方向否定，默认超参不变。
- 补配对 bootstrap（859 studies × 2,000 次，按 study 有放回重采样、seed 2026）给出 ΔAUC /
  ΔF1@0.5 的 95% CI，只覆盖 valid 内重采样、不含训练随机性。
- 结果回填 `docs/training_log.md`、`docs/experiments.md`，并更新本文件状态行。

## 当前状态

- [x] 配置 `configs/exp_A009.yaml` 就绪（`_base: exp_A001.yaml`，四项覆盖，加载校验通过）。
- [x] GPU 训练完成（2026-10-31 19:49:52 → 20:33:27，43m35s，10/10 轮跑满、无早停，best epoch 10）。
- [x] `run_eval.py` 评估（阈值 0.2–0.7）与配对 bootstrap 完成。
- [x] 实测回填 `docs/training_log.md`、`docs/experiments.md` 并更新本文件结论。

## 执行结果

- 产物：`outputs/20261031-194952_A-009_convnext_tiny_lr1e-3_wd0.03_s42/`（`best.pt` 111,385,195 B / `last.pt` 445,550,459 B / `metrics.csv` / `train.log` / `config.yaml`）、`outputs/eval_reports/A-009_full.txt`、`outputs/A-009_valid_predictions.csv`、`outputs/compare_A009/bootstrap_A009.csv`。
- 结果：10/10 轮跑满、无早停，best epoch 10，valid macro-AUC **0.7799** / F1@0.5 **0.4101**；相对 A-001（0.8038 / 0.4255）差 AUC −0.0239、F1@0.5 −0.0154，距 0.8088 门差 −0.0289 → **未过门**。
- 训练行为：AUC 从 ep1 0.7387 逐轮升到 ep10 0.7799，但 valid_loss 在 ep6 触底 0.9384 后反弹到 1.0955；A-001 则在 ep4 就见顶 0.8038。lr 放大到 1e-3 后收敛整体右移、末段过拟合。
- 配对 bootstrap（859 studies × 2,000 次，按 study 有放回重采样，seed 2026）：ΔAUC（A-001 − A-009）95% CI [+0.0121, +0.0360] P=1.000、ΔF1@0.5 95% CI [+0.0027, +0.0285] P=0.991，差距不是 valid 内采样噪声。
- 结论：优化超参联合放大方向否定（单 seed，四项同改不可单变量归因）；默认超参保持 A-001 口径，不补 seed 2026；若继续调参应改开 lr / wd 单变量实验。
