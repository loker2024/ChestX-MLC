# 训练计划 17：ConvNeXt-Small 骨干与 ImageNet-22k 预训练权重对照

**状态：执行中（2026-10-31 17:3x 启动，GPU 实测回填后更新）。**

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

（GPU 实测回来后在此追加：启动/结束时间、早停轮次、macro-AUC / F1@0.5、参数量与显存、
评估与对比产物路径、判读。）
