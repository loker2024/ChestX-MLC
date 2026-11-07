# D-010：train + valid 固定轮数训练设计

目标是在 D-002-a 的训练配方上，合并官方 train 与 valid 训练最终候选。四组均为待执行配置，不代表训练或评测结果。

| 配置 | 骨干 | seed |
| --- | --- | --- |
| `configs/exp_D010a.yaml` | `convnext_tiny.fb_in1k` | 42 |
| `configs/exp_D010b.yaml` | `convnext_tiny.fb_in1k` | 2026 |
| `configs/exp_D010c.yaml` | `convnext_tiny.fb_in1k` | 216 |
| `configs/exp_D010d.yaml` | `efficientnetv2_rw_s.ra2_in1k` | 42 |

四组从 ImageNet 预训练权重开始，要求实际加载成功；不允许 `--init-from`。固定 320、view_weighted 0.7、batch 16、BCE + 按合并标签重算的 pos_weight、AdamW lr 3e-4、warmup 2、EMA 0.999。训练固定 5 轮，cosine 终点仍为第 8 轮，导出第 5 轮 EMA。5 轮为预先选定的预算，不证明是最优轮数；增大训练集后每轮步数与历史 D-002-a 不同。

`data.full_train: true` 合并 train 和 valid，关闭 `resplit`、早停及训练期验证；两个来源分别使用各自图片目录。合并时复用跨来源患者、study 和 Dicom ID 重复检查。记录 `full_train_manifest.json`（各 study 来源、标签、图片 ID、CSV 哈希），断点续训时必须与 checkpoint 哈希一致。训练日志记录 pos_weight、预训练加载和最终权重哈希；`metrics.csv` 仅含训练 loss 和 lr，`last.pt` 用于续训，`final_ema.pt` 用于推理。它没有独立验证分数，原 valid 已进入训练，不能以原 valid 分数比较 D-002-a-EMA，也不能根据平台反馈反复挑模型。

在训练环境运行，四组分别使用独立 tag：

```bash
python src/run_train.py --config configs/exp_D010a.yaml --tag D-010-a
python src/run_train.py --config configs/exp_D010b.yaml --tag D-010-b
python src/run_train.py --config configs/exp_D010c.yaml --tag D-010-c
python src/run_train.py --config configs/exp_D010d.yaml --tag D-010-d
```

中断时使用 `python src/run_train.py --resume outputs/<run>`。推理使用对应 run 的 `config.yaml` 和 `final_ema.pt`：

```bash
python src/run_infer.py --config outputs/<run>/config.yaml --checkpoint outputs/<run>/final_ema.pt --out outputs/submission.csv
```

最终提交前按赛题规则核对预训练权重来源与许可、模型大小和指定设备的端到端耗时。训练与推理是否正常、实际资源和性能均需在训练环境另行确认。

## 执行记录（2026-11-07）

四组于 2026-11-07 12:25–13:23 在 sjy RTX 3090 24G 上串行执行完成，`rc=0`、日志无 warning / error / traceback：

| 组 | 起止 | 耗时 | 5 轮 train_loss | final_ema.pt SHA-256 |
| --- | --- | ---: | --- | --- |
| D-010-a | 12:25:17–12:39:39 | 14m26s | 0.9892 → 0.9364 → 0.8893 → 0.8410 → 0.7524 | `5d70af6e16e37ebb655ffc8b3c57179f89ec10a74c9caa0b1b24bd1521ebac48` |
| D-010-b | 12:39:39–12:53:44 | 14m05s | 0.9837 → 0.9303 → 0.8880 → 0.8397 → 0.7568 | `de97126925cebbe0582dab5ae5086a038bb4d7c33ebc526fc63a9cdd8f45ef69` |
| D-010-c | 12:53:44–13:08:04 | 14m20s | 0.9952 → 0.9382 → 0.8956 → 0.8460 → 0.7632 | `8ff8b95b8976bf17dc514e2844a9c9f5c620b457ff9c33ed80c410dd4eaff9c1` |
| D-010-d | 13:08:04–13:23:16 | 15m12s | 0.9881 → 0.9247 → 0.8692 → 0.8104 → 0.7037 | `3806f8c50d8521b785ec0839ff0acefc94176ce4803e54ae47c53b23fc0bb061` |

产物在 `outputs/20261107-122517_D-010-a/`、`outputs/20261107-123943_D-010-b/`、`outputs/20261107-125348_D-010-c/`、`outputs/20261107-130808_D-010-d/`。四组均确认「预训练权重: 已加载」，训练 19,372 studies、无验证集，每轮 1,210 batch；`full_train_manifest.json` 清单 SHA-256 四组一致 `0766a71e5fd5745da6cea3c0e263f60f19011cb11bc94804d2c86f36a917b45c`。没有独立验证分数；原 valid 已进入训练，不能以原 valid 分数比较 D-002-a-EMA，也不能根据平台反馈反复挑模型。详细记录见 `docs/training_log.md` 的 D-010 章节。
