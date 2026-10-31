# 训练计划 16：轻量 ML-Decoder 分类头对照

**状态：已完成；A-006 GPU 训练、评估、配对 bootstrap 与文档回填已执行（2026-10-31）。**

## 目标与配置

在与 A-001 相同训练口径下评估轻量类别查询交叉注意力头。候选配置 [exp_ml_decoder.yaml](../configs/exp_ml_decoder.yaml) 只将 `model.head` 设为 `ml_decoder`，结构见 [实现说明](ml_decoder_implementation.md)。采用 ConvNeXt-Tiny `fb_in1k` 预训练、全量微调、320、正侧位 logits 7:3、BCE + pos_weight、batch 16、lr 0.0003、weight decay 0.01、warmup 2、cosine、最多 8 轮、patience 2、seed 42。注意 LayerNorm 位置和分类头参数同时变化，只能比较整个头。

## 执行前核对

1. 确认代码版本、未提交 diff、torch/timm、设备与数据版本；检查 train/valid ID、标签、患者不交叉和图像完整性，保持原 valid。核验实际预训练权重 SHA-256 与 A-001 记录一致；不能仅凭日志证明文件一致。
2. `python -m pytest tests/test_ml_decoder.py tests/test_csra.py -q`，再运行 `python src/check_data.py --config configs/exp_ml_decoder.yaml --full`。检查展开配置与 A-001 仅 `model.head` 不同。
3. 优先复用已验证的 A-001 `best.pt`；若数据或代码语义无法证明一致，同版本、相同数据和预算重训标准头，历史 A-001 仅作参考。CSRA 作为另一个模型独立比较，不能用它取代标准头基线。

## 训练与评估

```bash
HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_ml_decoder.yaml --seed 42 --tag A-006_ml_decoder_s42
python src/run_eval.py --config outputs/<baseline>/config.yaml --checkpoint outputs/<baseline>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/A-006_baseline_valid.csv
python src/run_eval.py --config outputs/<decoder>/config.yaml --checkpoint outputs/<decoder>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/A-006_decoder_valid.csv
python src/run_compare.py --runs outputs/<baseline> outputs/<decoder> --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/compare_A006
```

替换路径为实际 run。确认日志和 checkpoint metadata 的 `pretrained_loaded` 为 true；记录模型参数量、GPU、显存、耗时及停止原因，保留 `best.pt`、`last.pt`、`config.yaml`、`metrics.csv`、`train.log`、逐 study ID / 标签 / 概率，以及 `summary.md`、`per_class.csv`。按 Study_id 对齐两组结果并核验标签、类别顺序和验证集一致。

## 判读

以重算的配对基线 Macro-AUC + 0.005 为方向性筛选门，历史 A-001 的 0.8038 对应参考值 0.8088；门槛不是统计显著性检验。报告逐类 AUC 和正负样本数、F1@0.5 与阈值曲线、学习曲线和资源成本；配对 bootstrap 时以 study 重采样，记录 seed、次数和退化类别处理。重复使用同一 valid 选型可能高估收益。若超过门槛且逐类无明显退化，再补 seed 2026 的同版本标准头与候选配对。候选提交前按 `docs/cost.md` 在赛题指定 T4 上实测。GPU 实测结果回来后再更新实验记录。

## 执行记录（2026-10-31）

- 执行前核对：`exp/A-004` @ `97505f0` 工作区干净；V100-SXM2-32G（驱动 580.105.08）独占；`pytest
  tests/test_ml_decoder.py tests/test_csra.py -q` → 10 passed；`check_data.py --full` 全部通过
  （train 30,344 图 / 18,513 studies、valid 1,422 / 859、test 2,218 / 1,414，Subject/Study/Dicom
  三划分零重叠）；预训练 safetensors SHA-256 `08b9dc9c…` 与 A-001 记录一致；展开配置与 A-001 仅
  `model.head` 不同。
- 配对基线：`outputs/20261031-151114_A-005_baseline_s42`（`exp_A001.yaml` 标准头重训，
  15:11:14–15:40:35、29m21s、best epoch 4、**0.8022 / 0.4210**、27,827,818 参数）。按计划不用历史
  A-001（0.8038）作主对照，A-001 仅作参考；方向性门 = 0.8022 + 0.005 = **0.8072**。
- A-006：15:57:01–16:29:40（32m39s），第 7 轮早停（best epoch 5），valid macro-AUC **0.7900** /
  F1@0.5 **0.4118**（F1 峰在 0.6：0.4173），28,549,226 参数（+721,408）；
  产物 `outputs/20261031-155701_A-006_ml_decoder_s42/`。
- 评估与对比：`outputs/eval_reports/A-006_decoder_full.txt`、`outputs/A-006_decoder_valid.csv`
  （859 个 valid study）、`outputs/compare_A006/`（`summary.md` / `per_class.csv`）。
- 配对 bootstrap（`outputs/plan16_bootstrap.py`，seed 2026、2,000 次、按 study 有放回重采样，2,000
  次均 10 类有效）：ΔAUC（基线 − A-006）= **+0.0123**，95% CI **[+0.0001, +0.0245]**、P(Δ>0) =
  **0.976**；ΔF1@0.5 = +0.0092，95% CI [−0.0018, +0.0205]、P = 0.952。
- 结论：**未过方向性门**（0.7900 vs 0.8072，差 −0.0172），配对 bootstrap 显示标准头 AUC 显著更优、
  F1 方向一致 → 分类头方向否定；`model.head: standard` 保持默认，不做 seed 2026 配对，不进入候选
  提交。详细记录见 [training_log.md](training_log.md) 的 A-006 小节与 [experiments.md](experiments.md)
  的「执行与结果」。
