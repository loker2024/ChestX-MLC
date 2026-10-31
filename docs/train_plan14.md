# 训练计划 14：A-004 单变量分解（dropout / weight decay 分别对照）

**状态：配置与本机验证已准备；GPU 训练待执行。**

## 目标与配置

A-004 同时改动了分类头 dropout（0.0 → 0.2）与 `train.weight_decay`（0.01 → 0.05），valid macro-AUC **0.7988** / F1@0.5 **0.4233**，相对 A-001（**0.8038** / **0.4255**）为 AUC −0.0050、F1@0.5 −0.0022，未过方向性门 0.8088（见 [训练计划 13](train_plan13.md) 与 [训练记录](training_log.md)）。组合结果只能评价组合，无法归因到具体一项；本计划把两项拆成两个单变量组，回答「退化来自哪一项、各占多少」。

以同一 A-001 配置为基线，两个新配置各自只覆盖一个参数：

| run | 配置 | `model.dropout` | `train.weight_decay` | 与 A-004 的关系 |
| --- | --- | --- | --- | --- |
| A-001 | [exp_A001.yaml](../configs/exp_A001.yaml) | 0.0 | 0.01 | 基线（已完成） |
| A-004-a | [exp_A004a.yaml](../configs/exp_A004a.yaml) | 0.2 | 0.01 | 只开 dropout |
| A-004-b | [exp_A004b.yaml](../configs/exp_A004b.yaml) | 0.0 | 0.05 | 只提 weight decay |
| A-004 | [exp_A004.yaml](../configs/exp_A004.yaml) | 0.2 | 0.05 | 两项同开（已完成） |

`model.dropout` 通过 timm 的 `drop_rate` 作用于 ConvNeXt 分类头（训练启用、评估关闭，不是随机深度）；`train.weight_decay` 是 AdamW 的权重衰减，当前优化器对全部可训练参数使用同一参数组。其余保持 ConvNeXt-Tiny `fb_in1k` 预训练、全量微调、320×320 letterbox / ImageNet 归一化、关闭水平翻转、study 正侧位 7:3 加权聚合、BCE + `pos_weight`、batch 16 studies、lr 0.0003、warmup 2、cosine 至 0.00001、最多 8 轮、patience 2、seed 42；数据划分、代码版本、timm 版本、权重文件及 SHA-256 必须与 A-001 相同。两个 run 串行执行（同一 GPU 不并发），各自使用独立 tag。

## 执行

1. 在训练环境运行 `python -m pytest tests/test_a004.py tests/test_a004_decomp.py -q`，核验 `exp_A004a.yaml` 相对 A-001 只差 `model.dropout`、`exp_A004b.yaml` 只差 `train.weight_decay`；确认预训练权重来源与哈希同 A-001。
2. 运行 `python src/check_data.py --config configs/exp_A004a.yaml --full`，确认数据版本、study 与患者划分及图片完整性与基线一致（两个新配置只差超参，数据检查一次即可）。
3. 依次执行（前一个结束、评估完成后再启动下一个）：

   ```bash
   HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A004a.yaml --seed 42 --tag A-004a_dropout_only_s42
   HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A004b.yaml --seed 42 --tag A-004b_weight_decay_only_s42
   ```

   确认日志显示 `预训练权重: 已加载`、checkpoint 的 `metadata.pretrained_loaded` 为 true；保存实际配置、代码版本、设备、显存、耗时、停止原因、`train.log`、`metrics.csv`、`best.pt` 和 `last.pt`。
4. 统一重算与对比（替换占位符为实际运行目录；A-001 与 A-004 的产物已在 `outputs/` 中，直接复用，不重跑）：

   ```bash
   python src/run_eval.py --config outputs/<A-004-a运行目录>/config.yaml --checkpoint outputs/<A-004-a运行目录>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/A-004a_valid_predictions.csv
   python src/run_eval.py --config outputs/<A-004-b运行目录>/config.yaml --checkpoint outputs/<A-004-b运行目录>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/A-004b_valid_predictions.csv
   python src/run_compare.py --runs outputs/<A-001运行目录> outputs/<A-004运行目录> outputs/<A-004-a运行目录> outputs/<A-004-b运行目录> --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/compare_A001_A004_decomp
   ```

   保留汇总、逐类报告与逐 study ID、标签和概率；核对 ID、标签和类别顺序。

## 判读

- 以 Macro-AUC 为主，分别报告 A-004-a、A-004-b 相对 A-001 的差值、逐类 AUC 与支持数、F1@0.5 与完整阈值曲线；方向性门同前：A-001 重算值 0.8038 + 0.005 = 0.8088。
- 归因：若 A-004-a 的负差接近 A-004 的 −0.0050、A-004-b 接近 0，则退化主要来自 dropout；反之来自 weight decay；两项都明显为负时需考虑共同作用与单 seed 噪声。用逐 study 预测做配对 bootstrap 给出区间，保留「重复使用同一 valid 选型」的偏差说明。
- 对比每轮训练 loss、valid loss、AUC 与最佳轮次及停止原因；dropout 组的 train/valid loss 差距是否缩小，是「是否推迟过拟合」的直接证据。
- 训练完成后按真实结果回填 `docs/experiments.md`、`docs/training_log.md` 和 `CHANGELOG.md`。本计划不声称训练已执行，不预设收益。
