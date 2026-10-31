# 训练计划 20：A-009 二轮 warm start（以首轮 best.pt 为初始权重重训 10 轮）

**状态：已完成（2026-10-31），GPU 训练与评估已执行；结论：warm start 二轮无增益、未过 0.8088 门。**

## 目标

首轮 A-009（`docs/train_plan19.md`）把 lr 放大到 1e-3、wd 放大到 0.03、轮数 8→10 后，
valid macro-AUC 只到 0.7799 / F1@0.5 0.4101，未过 0.8088 方向性门；逐轮看 AUC 一直在涨
（ep1 0.7387 → ep10 0.7799），但 valid_loss 在第 6 轮触底后连续反弹，属于 lr 过大造成的
收敛右移、末段过拟合。本组回答的问题是：**在已经训到第 10 轮的 best.pt 上，以同一 A-009
配置再完整跑一轮 10 轮训练，排序指标能否抬高？** 即「warm start 第二轮」是否比首轮更好。

## 配置与口径

训练配置与首轮 A-009 完全相同（`configs/exp_A009.yaml`，`_base: exp_A001.yaml`）：
ConvNeXt-Tiny `fb_in1k` @320×320、view_weighted 7:3 logits 聚合、BCE + `pos_weight`、
AdamW `lr 1e-3` / `weight_decay 0.03`、warmup 2 + cosine 到 1e-5、`epochs 10`、
`early_stop_patience 4`、batch 16 studies、AMP、`grad_clip 5.0`、seed 42。

唯一区别是初始化：不再加载 ImageNet 预训练权重，而是加载首轮 A-009 的 `best.pt`
（epoch 10 权重，SHA-256 `6988006910b824a83c277764695d0db4a3bdeb3ab0b1b029d22dc9e3ee0f1b15`）。
训练从第 1 轮重新开始，学习率仍从 warmup 的 5e-4 升到 1e-3、再 cosine 到 1e-5，
优化器、调度器和 epoch 进度都不复用。

## 实现口径

`src/run_train.py` 新增 `--init-from <checkpoint>`（warm start）：

- 只加载 checkpoint 里的 `model` 权重，构建模型时 `pretrained=False`（避免重新联网/加载 ImageNet 权重）；
- 训练从第 1 轮开始，不恢复优化器 / 调度器 / 进度；
- `best.pt` 与 `last.pt` 的 metadata 记录 `warm_start_from`（路径）、`warm_start_sha256`、
  `warm_start_epoch` 与 `warm_start_metrics`，权重来源可事后核对；
- `--init-from` 与 `--resume` 互斥。

## 执行命令

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -u src/run_train.py --config configs/exp_A009.yaml --seed 42 \
    --init-from outputs/20261031-194952_A-009_convnext_tiny_lr1e-3_wd0.03_s42/best.pt \
    --tag A-009-r2_warmstart_convnext_tiny_lr1e-3_wd0.03_s42
```

## 判读

与首轮 A-009（0.7799 / 0.4101）直接对比；方向性门沿用 0.8088（A-001 0.8038 + 0.005）。
若低于首轮，说明在已收敛解上再走一轮同配置 cosine 只会继续过拟合；若高于首轮但未过门，
说明 warm start 有增益但不足以扭转 A-009 方向；若过门，需要补 seed 2026 复核再谈候选。

## 结果

- 训练：2026-10-31 20:49:14 → 21:13:48 CST（24m34s，V100 与另一容器共享 GPU，每轮约 6–7 分钟），
  第 5 轮结束后连续 4 个 epoch 没有提升，早停（best epoch 1）。
- 逐轮（lr / train_loss / valid_loss / macro-AUC / F1@0.5）：ep1 5.00e-04 / 0.7556 / 0.9977 / **0.7764** / 0.4117；
  ep2 1.00e-03 / 0.8260 / 0.9703 / 0.7739 / 0.4077；ep3 1.00e-03 / 0.8197 / 1.0172 / 0.7587 / 0.3980；
  ep4 9.51e-04 / 0.8100 / 0.9863 / 0.7692 / 0.3901；ep5 8.14e-04 / 0.7586 / 1.0525 / 0.7664 / 0.3985。
- 评估（`run_eval.py` 重算）：macro-AUC **0.7764** / F1@0.5 **0.4117**；阈值 0.2–0.7 曲线
  0.3684 / 0.3833 / 0.3966 / 0.4117 / **0.4137**（峰 0.6）/ 0.4022；逐类 AUC 无误（class5 0.8579 /
  class9 0.8942 / class10 0.8678，class1 0.6993 / class4 0.6764 / class8 0.6973）。
- 判读：相对首轮 A-009（0.7799 / 0.4101）差 AUC −0.0035、F1@0.5 +0.0016（seed 波动带内，无增益）；
  相对 A-001（0.8038 / 0.4255）差 AUC −0.0274、F1@0.5 −0.0138，距 0.8088 门差 −0.0324，**未过门**。
  从首轮 ep10 权重出发后，lr 重新 warmup 到 1e-3 破坏了已收敛权重（train_loss 0.6562 → 0.7556，
  valid AUC 由源 0.7799 掉到 0.7764 后连续 4 轮未回升），说明在已收敛解上再走一轮同配置 cosine
  只会继续过拟合，warm start 方向否定、默认口径不变。
- 产物：`outputs/20261031-204914_A-009-r2_warmstart_convnext_tiny_lr1e-3_wd0.03_s42/`（`best.pt`
  111,385,579 B / `last.pt` 445,550,395 B / `metrics.csv` / `train.log` / `config.yaml`）、
  `outputs/eval_reports/A-009-r2_full.txt`、`outputs/A-009-r2_valid_predictions.csv`。
