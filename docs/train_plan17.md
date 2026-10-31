# 训练计划 17：ConvNeXt-Small 骨干与 ImageNet-22k 预训练权重对照

**状态：已完成（2026-10-31 17:23:05–18:00:08，8/8 轮跑满后早停，best epoch 6，macro-AUC 0.7903 / F1@0.5 0.4204 未过 0.8088 门 → 骨干/预训练权重方向否定）。**

## 目标与配置

在 A-001 的同一训练口径下评估「更大骨干 + 更强预训练权重」。候选配置
[exp_A007.yaml](../configs/exp_A007.yaml) 以 `_base: exp_A001.yaml` 继承 A-001，只覆盖
`model.backbone: convnext_tiny.fb_in1k → convnext_small.fb_in22k_ft_in1k`；timm 模型名同时决定骨干
结构与预训练权重来源（ConvNeXt-Small，ImageNet-22k 预训练后在 ImageNet-1k 微调）。

其余口径逐字段继承 A-001：320×320 letterbox、ImageNet 归一化、无水平翻转、标准分类头（dropout 0）、
`view_weighted` 聚合（正位 0.7 / 侧位 0.3）、BCE + `pos_weight`、`batch_studies: 16`、lr 0.0003、
weight decay 0.01、warmup 2 + cosine 到 1e-5、最多 8 轮、patience 2、AMP、seed 42。

模型规模：ConvNeXt-Small + 10 类头 **49,462,378** 参数（A-001 的 ConvNeXt-Tiny 为 27,827,818，
+21,634,560）。这是「骨干结构 + 预训练数据」一起变的对照，不是严格单变量，判读时不能把差异全部
归因于权重或结构之一。

## 执行前核对

1. 确认代码版本、工作区、torch/timm、设备与数据版本；保持原 valid 与 train/valid 划分不变。
2. timm `convnext_small.fb_in22k_ft_in1k` 权重首次在训练机使用：记录 HF 缓存 commit、文件大小与
   SHA-256，核对来源（`timm/convnext_small.fb_in22k_ft_in1k`）与许可（沿用 timm 登记口径，需核验）；
   日志与 checkpoint metadata 的 `pretrained_loaded` 必须为 true。
3. `python -m pytest -q`，再运行 `python src/check_data.py --config configs/exp_A007.yaml --full`；
   确认展开配置与 A-001 仅 `model.backbone` 不同。

## 训练与评估

```bash
HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A007.yaml --seed 42 --tag A-007_convnext_small_in22k_ft_s42
python src/run_eval.py --config outputs/<run>/config.yaml --checkpoint outputs/<run>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/A-007_valid.csv
python src/run_compare.py --runs outputs/<A-001> outputs/<A-007> --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/compare_A007
```

替换路径为实际 run。记录 GPU、显存峰值、耗时与停止原因，保留 `best.pt`、`last.pt`、`config.yaml`、
`metrics.csv`、`train.log`、逐 study ID / 标签 / 概率，以及 `summary.md`、`per_class.csv`。

## 判读

参考 A-001 的 valid macro-AUC **0.8038** / F1@0.5 **0.4255**（同数据、同口径；A-001 与 A-007 只差
骨干与预训练权重）。以 A-001 记录值 + 0.005 为方向性筛选门 = **0.8088**；门槛不是统计显著性检验，
且本组单 seed、单组，需报告逐类 AUC、F1@0.5 与阈值曲线、学习曲线和资源成本。超过门槛且逐类无
明显退化才补 seed 2026 与赛题 T4 端到端耗时；否则记录负向结论、默认骨干不切换。

## 执行记录

- 启动/结束：2026-10-31 17:23:05 – 18:00:08，共 **37m03s**；8/8 轮跑满后第 8 轮早停（best epoch 6）。
- 运行目录：`outputs/20261031-172305_A-007_convnext_small_in22k_ft_s42`（`best.pt` 197,987,899 B、
  `last.pt` 791,962,333 B、`metrics.csv`、`config.yaml`、`train.log`）。
- 指标（`run_eval.py`，valid 859 studies）：macro-AUC **0.7903** / F1@0.5 **0.4204**；F1 峰在 0.6
  （0.4225）；`metrics.csv` 的 best epoch 6 为 0.7903 / 0.4209（F1 差 0.0005）。相对 A-001
  （0.8038 / 0.4255）ΔAUC **−0.0135**、ΔF1@0.5 **−0.0051**，距方向性门 0.8088 差 **−0.0185 → 未过门**。
- 学习曲线：e1 0.7452 → e2 0.7672 → e3 0.7782 → e4 0.7837 → e5 0.7883 → e6 **0.7903**（best）→
  e7 0.7863 → e8 0.7776；每轮 AUC 均低于 A-001 同轮（差 −0.0133～−0.0286），峰值更晚（e6 vs e4）；
  valid_loss 第 5 轮见底 0.9267 后反弹到 1.3911 / 1.8042（A-001 为 0.8856 → 0.9333 / 1.0194）。
- 参数量 49,462,378（`best.pt` 权重统计；checkpoint metadata `total_params` / `trainable_params` 均为
  49,462,378、`pretrained_loaded: true`、`freeze_backbone: false`）；显存峰值 14,702 MiB（训练中
  `nvidia-smi` 抽样；A-001 记录约 11.7 GB）。
- 预训练权重：HF snapshot commit `fe6b358aa409414cdb522ae3d43a2c9922f7cc1b`、
  `model.safetensors` 200,928,946 B、SHA-256
  `2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9`；`HF_HUB_OFFLINE=1` 离线加载
  成功（日志「预训练权重: 已加载」）。
- 评估与对比产物：`outputs/eval_reports/A-007_full.txt`、`outputs/A-007_convnext_small_valid.csv`、
  `outputs/compare_A007/summary.md`、`outputs/compare_A007/per_class.csv`、
  `outputs/compare_A007/bootstrap_A007.csv`、`outputs/compare_A007/bootstrap_A007.log`。
- 配对 bootstrap（859 studies × 2,000 次、按 study 有放回重采样、seed 2026、退化类跳过）：
  ΔAUC（A-001 − A-007）= **+0.0136**、95% CI **[+0.0006, +0.0260]**、P(Δ>0)=**0.983**；
  ΔF1@0.5 = **+0.0051**、95% CI [−0.0065, +0.0172]、P=0.807。
- 逐类：3 升 7 降（class9 +0.0082、class8 +0.0032、class5 +0.0031；class2 −0.1079 断崖、class4
  −0.0147、class6 −0.0114），class2（support 37）是 AUC 差的主要来源（10 类平均拉低约 0.0108）。
- 判读：**骨干/预训练权重方向否定（单 seed）**——ConvNeXt-Tiny（`fb_in1k`）保持 A 组默认候选，
  不补 seed 2026、不进入候选提交；若继续更大骨干，需先解决固定 8 轮预算下的过拟合（更早停止 /
  更强正则）或改用更长训练预算，并把「结构」与「预训练数据」拆成两个单变量实验。
- 执行前核对全部完成：`check_data.py --full` 通过（图片齐全、三划分零重叠）；全量 pytest
  263 passed / 2 failed（`tests/test_xrv.py` 缺 `torchxrayvision`，环境既有问题，与本组无关）/
  4 skipped；展开配置与 A-001 仅 `model.backbone` 一处不同。
