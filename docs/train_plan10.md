# 训练计划 10：A-002 轻量 CNN 骨干对照

**状态：配置已创建（`configs/exp_A002.yaml`），尚未执行 GPU 训练。**

## 目标与对照

以约 58 万参数的 MobileNetV3-Small 0.5× 检验轻量模型在胸片多标签分类中的表现，观察验证性能、训练与验证曲线，以及实际资源消耗。参数减少不保证改善过拟合，也可能造成欠拟合。

基线为 A-001：`convnext_tiny.fb_in1k`，seed 42，320×320，8 轮上限、早停 patience 2，best epoch 4；训练记录中 valid Macro-AUC **0.8038**、F1@0.5 **0.4255**，均由 `run_eval.py` 重算。参数量 27,827,818。数据与指标记录见 [training_log.md](training_log.md) 的 A-001 小节。本计划引用记录，不代表本次重新计算。

本机以当前 timm、`pretrained=False`、`num_classes=10` 构造待测模型，统计参数为 **578,474**；该操作仅确认结构可构造和参数数量，不证明权重可用、前向兼容或模型有效。

## 实验配置

| 项目 | A-001 | A-002 |
| --- | --- | --- |
| 模型和对应预训练权重 | `convnext_tiny.fb_in1k` | `mobilenetv3_small_050.lamb_in1k` |
| 总参数，10 类头 | 27,827,818 | 578,474（本机结构统计） |
| seed | 42 | 42 |
| 输入 | 320×320，letterbox，ImageNet 归一化，关闭水平翻转 | 相同 |
| 聚合 | `view_weighted`，正位 0.7 / 侧位 0.3 | 相同 |
| 训练 | 全量微调；BCE + `pos_weight`；8 轮上限、patience 2 | 全量微调；BCE + `pos_weight`；25 轮上限、patience 5 |
| 其他参数 | batch 16 studies、lr 0.0003、weight decay 0.01、warmup 2、cosine、最低 lr 0.00001 | 相同 |

相对已完成的 A-001，实验同时改变**骨干及其预训练权重、训练上限、早停 patience**；cosine 衰减周期也随训练上限变化。因此两者仅作方向性比较，不能单独归因于骨干或参数量，也不能把轮数差异当成模型收敛快慢的证据。尽量使用相同数据、代码语义和运行环境；若代码、设备或数据有差异，记录并限定可比性。不同骨干对相同优化器设置的适配程度也可能不同。

## 准备与运行

1. 创建 `configs/exp_A002.yaml`，内容如下。A-001 是完整展开的配置，执行前比较两份解析后的配置，确认仅 `model.backbone`、`train.epochs`、`train.early_stop_patience` 三项不同，并记录配置和代码版本。

   ```yaml
   _base: exp_A001.yaml

   model:
     backbone: mobilenetv3_small_050.lamb_in1k

   train:
     epochs: 25
     early_stop_patience: 5
   ```

2. 核对实际数据划分与 A-001 一致，并确认 train / valid 无患者交叉。运行 `python src/check_data.py --config configs/exp_A002.yaml --full`；如无法取得同一份数据或核验失败，停止对照。
3. 准备 [模型卡与公开权重](https://huggingface.co/timm/mobilenetv3_small_050.lamb_in1k)：模型卡标注 ImageNet-1k 和 Apache-2.0。固定具体 revision，保存权重链接、许可证全文、训练数据说明及实际权重文件 SHA-256；核对下载文件与选用版本一致，并确认许可证及再分发符合赛题规则。服务器离线时先预下载并核验缓存。
4. 在训练环境用 `pretrained=False` 做 320×320 前向 smoke test，确认 `[2, 3, 320, 320] → [2, 10]`，再确认总参数量、可训练参数量；测试只证明接口，不证明权重已加载。
5. GPU 环境执行：

   ```bash
   HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_A002.yaml --seed 42 --tag A-002_mobilenetv3_small050_s42
   ```

   命令要求权重已在缓存；如果没有离线缓存，先按步骤 3 准备。启动后确认日志 `预训练权重: 已加载`；若回退随机初始化，停止并修复权重加载，再新建一次运行。结束后复核 `best.pt` 的 `metadata.pretrained_loaded`、`total_params` 和 `trainable_params`，以及实际展开配置。
6. 用实际目录代入下列命令，重算同一 valid 的 A-001 和 A-002；不得将示例占位符直接运行：

   ```bash
   python src/run_compare.py --runs outputs/<A-001运行目录> outputs/<A-002运行目录> --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/compare_A002
   ```

   保留 `summary.md`、`summary.csv`、`per_class.csv`；另外保存两个模型的 valid 每 Study ID、真实标签与 10 类概率，以供独立复核。现有 `run_compare.py` 不直接保存逐 Study 预测，需要复用现有推理接口导出并核对 ID、标签、列顺序与数量，不把汇总表当作逐样本数据。

## 判读与下一步

- 主指标为同 valid、同 seed 的 Study 级 Macro-AUC。以 **0.7988**（A-001 记录 0.8038 减 0.005）作为首轮方向性工程筛选门；比较前以本次重算的 A-001 值为准，若与记录有差异，记录原因并据重算基线减 0.005 判定。由于训练预算及 cosine 周期不同，这不是严格的骨干对照；该门不是统计显著性证明，也不表示线上成绩。
- 同时看逐类 AUC、每类支持数、F1@0.5 和 0.2–0.7 阈值曲线。低频类的波动需结合样本数理解；F1 与连续概率排序的 AUC 分开解释。
- 记录每轮 train loss、valid loss、valid Macro-AUC、best epoch 和实际停止轮次。验证 loss 回升且训练 loss 下降是过拟合迹象；训练和验证均较差可能是欠拟合，但也可能是学习率或训练预算不合适。不同模型的最佳 epoch、早停导致的训练时长会不同。
- 若 A-002 达到筛选门且没有重要类别明显退化，为严格比较骨干，补跑 ConvNeXt-Tiny 的 25 轮上限、patience 5 基线（其余与 A-002 一致），先在 seed 42 与 A-002 配对；再以 seed 2026 对两个骨干都按 25+5 复核，记录每个 seed 的配对差。不把不同训练预算的历史 A-001 或 RUN-008-C 当作严格配对。两 seed 仍不能证明统计显著性。
- 若低于筛选门，先检查权重加载、曲线和逐类结果；若接近门或到第 25 轮仍未收敛，不能仅凭本轮预算就判定轻量方向无效。需要调训练预算时，另开对照并保留原 A-002 结果。必要时再试约 103 万参数的 `mobilenetv3_small_075.lamb_in1k`，作为新实验，而非修改已完成的 A-002。
- 记录模型参数、checkpoint 文件大小、训练设备、显存峰值及每轮时间；跨设备或共享负载下的耗时不直接比较。候选模型还须在赛题指定设备按 [cost.md](cost.md) 实测 batch=1 端到端推理及提交一致性。参数量与实际速度不是同一指标。

训练后将真实配置、代码版本、种子、权重证据、运行日志、`best.pt`、`last.pt`、`metrics.csv`、逐 Study 预测、对照汇总、耗时及结论回填 `docs/training_log.md`、`docs/experiments.md` 和 `CHANGELOG.md`。未执行项目保持明确标注，不更改默认配置或宣布提交候选。
