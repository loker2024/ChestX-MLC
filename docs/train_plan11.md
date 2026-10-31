# 训练计划 11：A-003 MobileNetV3-Small 1.0× 对照

**状态：已执行（2026-10-30 23:07:25 → 2026-10-31 00:24:44 CST，第 17 轮早停、best epoch 12）。`best.pt` 为 epoch 12，valid macro-AUC **0.7808** / F1@0.5 **0.3881**，未过 0.7988 方向性筛选门（差 -0.0180）；A-002 于第 5 轮按用户要求中止、无 `best.pt`，因此 A-002 / A-003 配对比较不成立。实测回填见 [training_log.md](training_log.md) 与 [experiments.md](experiments.md)。**

## 目的与基线

以 `mobilenetv3_small_100.lamb_in1k` 的 1.0× 宽度版本检验轻量 CNN 的表现，与 [A-002 / plan10](train_plan10.md) 的 0.5× 版本对照。当前本机用 `pretrained=False` 构造 10 类模型得到 **1,528,106 个参数**，随机输入 `[1, 3, 320, 320]` 的输出形状为 `[1, 10]`；这不证明权重加载成功或训练表现。

| 项目 | A-002 | A-003 |
| --- | --- | --- |
| 模型与对应权重 | `mobilenetv3_small_050.lamb_in1k` | `mobilenetv3_small_100.lamb_in1k` |
| 10 类模型参数 | 578,474 | 1,528,106 |
| 配置 | `configs/exp_A002.yaml` | `configs/exp_A003.yaml` |
| 训练设置 | seed 42，320×320，25 轮上限、patience 5 | 与 A-002 相同 |

除模型及其对应预训练权重外，A-003 继承 A-002：letterbox、ImageNet 归一化、关闭水平翻转、全量微调、正侧位 7:3 聚合、BCE + `pos_weight`、batch 16 studies、lr 0.0003、weight decay 0.01、warmup 2、cosine 至 0.00001。`tests/test_config.py` 检查解析结果除 `model.backbone` 外一致。两种宽度的预训练权重也不同，结果不能仅归因于参数量。

历史 A-001（ConvNeXt-Tiny）的记录值为 Macro-AUC **0.8038**、F1@0.5 **0.4255**；因其为 8 轮上限、patience 2，cosine 周期也不同，A-003 与 A-001 仅作方向性比较。上述是 [训练记录](training_log.md) 中的重算值，并非本次重算。

## 准备与执行

1. 核对 A-002、A-003 的实际配置、代码、数据划分和训练环境；检查 train / valid 无患者交叉。执行 `python src/check_data.py --config configs/exp_A003.yaml --full`，核验失败则停止对照。
2. 获取 [MobileNetV3-Small 1.0× 模型卡及权重](https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k)。模型卡标注 ImageNet-1k 预训练与 Apache-2.0；固定具体 revision，保存权重链接、许可证全文、预训练数据说明和实际文件 SHA-256，并核对赛题权重与再分发要求。离线训练前先准备并核验缓存。
3. 在训练环境确认 320×320 前向、10 类输出与参数总数；启动后核对日志 `预训练权重: 已加载`，若回退随机初始化则停止，修复后新建运行。检查 checkpoint 的 `metadata.pretrained_loaded`、`total_params`、`trainable_params` 和运行时配置。
4. 在 GPU 环境执行：

   ```bash
   HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A003.yaml --seed 42 --tag A-003_mobilenetv3_small100_s42
   ```

   上述离线命令要求权重已缓存。保留 `best.pt`、`last.pt`、`config.yaml`、`metrics.csv`、`train.log`、设备、显存及时间记录。
5. A-002 和 A-003 均有实际运行产物后，替换目录占位符并重算：

   ```bash
   python src/run_compare.py --runs outputs/<A-002运行目录> outputs/<A-003运行目录> --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/compare_A002_A003
   ```

   保存 `summary.md`、`summary.csv`、`per_class.csv`；另复用现有推理接口导出每个 valid Study 的 ID、真实标签和 10 类概率，检查 ID、标签、类别顺序和数量。`run_compare.py` 的汇总表不是逐 Study 预测。A-002 若尚未运行，则不能填写其结果或声称完成配对比较。

## 判读与回填

- 以同 seed、同数据和 25+5 配方的 A-002 为主要对照，看 Study 级 Macro-AUC、逐类 AUC 与类别支持数；同时报告 F1@0.5 和 0.2–0.7 阈值曲线。AUC 差值不足以单独证明参数规模带来的效果。
- 延用 plan10 的方向性筛选门：相对 A-001 的记录值 0.8038 容忍下降 0.005，即 **0.7988**；重算基线若与记录不同，以重算结果减 0.005 为准。由于预算不同，该门不是严格骨干对照，也不是统计显著性证明。
- 对比训练 loss、valid loss、valid Macro-AUC、最佳轮次、停止原因；训练与验证均差可能是欠拟合，也可能是优化或预算问题。参数量不等于显存、文件大小或推理速度，记录实际资源并按 [cost.md](cost.md) 在指定设备实测。
- 若首轮表现接近或达到筛选门且重要类别没有明显退化，补 25+5 的 ConvNeXt-Tiny 同 seed 基线，并对相关候选模型补 seed 2026 配对；单 seed 或两 seed 均不能证明统计显著性。若 25 轮结束仍未收敛，调整预算应另开实验并保留原始结果。
- 训练后将真实配置、代码版本、权重证据、日志、逐 Study 预测、对比产物及结论回填 `docs/training_log.md`、`docs/experiments.md` 和 `CHANGELOG.md`；未执行项不能标为已验证或提交候选。
