# 训练计划 12：B 组医学预训练 DenseNet-121 研究对照

**状态：仅制定计划；尚未安装依赖、下载权重、修改训练代码或执行训练。**

## 目标与边界

B 组使用 TorchXRayVision 的 **DenseNet-121 + `densenet121-res224-all` 胸片预训练权重**，研究该方案在现有胸片数据上的验证表现。本计划只安排 B 组，不新增或训练 A 组。

**赛题限制：**该权重的预训练数据来自外部医学影像，不符合 [参赛规则](比赛参赛规则.md) 第 1 条。B 组仅作独立研究，不用于赛题提交、融合、候选模型选择或任何参赛链路；更换分类头不能改变骨干权重的医学来源。研究运行在独立环境、独立实验分支和独立输出目录，避免影响现有赛题默认配置及产物。

## 模型与数据

| 项目 | B 组方案 |
| --- | --- |
| 模型 | TorchXRayVision DenseNet-121；加载 `densenet121-res224-all` |
| 权重来源 | [官方模型实现与下载地址](https://github.com/mlmed/torchxrayvision/blob/main/torchxrayvision/models.py)，[模型卡](https://huggingface.co/torchxrayvision/densenet121-res224-all) |
| 预训练数据 | 公开资料标注 NIH、PadChest、CheXpert、MIMIC-CXR 等多个胸片数据集，具体以实际版本记录为准 |
| 骨干输入 | 单通道胸片；224×224；按 TorchXRayVision 归一化到 `[-1024, 1024]` |
| 输出 | 加载原始 18 类权重后，将分类头替换为 10 类；每张图输出原始 logits |
| Study 聚合 | 与现有项目相同的 `view_weighted`，正位 0.7、侧位 0.3；未知或单一视角按现有逻辑回退 |
| 训练 | 全量微调，BCEWithLogitsLoss + `pos_weight`；25 轮上限、早停 patience 5；seed 42 |
| 其他设置 | batch 16 studies、lr 0.0003、weight decay 0.01、warmup 2、cosine 至 0.00001；关闭水平翻转 |

默认采用与项目一致的 **letterbox 到 224×224**，随后做单通道及 TorchXRayVision 归一化。确认灰度读取、位深和像素范围后，按实际输入使用正确的 `maxval`，禁止无依据地按 8 位处理全部影像。模型包装内部会按权重设置调整尺寸，需记录实际进入骨干的尺寸；本轮不关闭或绕过该行为。

加载预训练模型后再替换分类头；该库不支持在构造带权重模型时直接指定不同类别数。禁用原 18 类的阈值映射及 sigmoid，验证输出为原始 `[N, 10]` logits，先聚合再计算 loss。10 类头按项目标签顺序建立，不能直接按位置把原 18 类预测当成项目标签。[模型实现](https://github.com/mlmed/torchxrayvision/blob/main/torchxrayvision/models.py)；[预处理说明](https://github.com/mlmed/torchxrayvision/blob/main/docs/source/faq.rst)。

## 执行步骤与验证门

1. **核验研究数据和权重。** 明确研究数据的使用权限，确认 train / valid 的患者划分；调查上游医学预训练集与本地 train / valid 是否存在患者或影像重叠。记录可用的原始来源、ID 和去重方法；无法判定时标记重叠风险未知，任何结果不得声称无泄漏。固定 TorchXRayVision 版本或源码 commit、模型文件链接、许可证与训练数据说明，计算实际文件 SHA-256；检查下载与使用条件。
2. **建立隔离适配。** 当前 `src/model/network.py` 通过 `timm.create_model` 建模，`requirements.txt` 未声明 TorchXRayVision；因此不能只把上述权重名称写进现有 `model.backbone`。在独立实验分支与环境中设计 B 组模型加载、单通道预处理、10 类头、配置和 checkpoint 重建方式；不改变现有赛题默认配置。明确依赖版本，记录加载来源与参数总数。
3. **测试加载和数据链路。** 用合成数据检验：加载原始权重后除分类头外参数正确保留，加载失败立即报错而不回退随机初始化；单通道、值域、224×224 及输入形状正确；新头为 10 类原始 logits，无旧阈值映射；Study 聚合可反向传播；保存并重载 checkpoint 后预测一致。测试标签顺序与缺失视角回退；真实权重检查应使用可信、完整校验的文件。
4. **执行 B 组研究训练。** GPU 环境先核对数据划分、依赖版本、权重 SHA-256、实际配置与代码版本。训练日志记录权重已加载、参数总数和可训练参数数，加载失败停止。以 seed 42、25+5 为首轮设置，记录训练 loss、valid loss、valid Macro-AUC、最佳轮次、停止原因、每轮时间和显存峰值。因适配代码与配置尚未实现，训练命令在实现完成并核对 CLI 后填入，不将现有 `run_train.py` 命令冒充可用。
5. **评估和留存。** 用 best checkpoint 在同一 valid 上重算 Study 级 Macro-AUC、逐类 AUC 与正例数、F1@0.5 和 0.2–0.7 阈值曲线；保存 `config.yaml`、`train.log`、`metrics.csv`、`best.pt`、`last.pt`、逐 Study ID / 真实标签 / 10 类概率、评估汇总与逐类报告。核对 Study 对齐、标签顺序和输出完整性。研究产物须清晰标注医学预训练，不生成赛题提交文件。

## 如何解释结果

现有 RUN-006 为 ImageNet DenseNet-121、320×320，seed 42，记录的 valid Macro-AUC **0.7880**（见 [训练记录](training_log.md)）。B 组使用医学权重、单通道与不同归一化，且实际分辨率为 224；即使数据划分相同，与 RUN-006 也只是**历史参考**，差值不能归因于医学预训练，也不是严格配对。若划分或数据版本未核实一致，则不计算直接差值。

单 seed 的指标、曲线及逐类表现只描述此次研究；不预设医学预训练一定更好。若无法排除预训练集与 valid 重叠，应醒目标注评估可能乐观。研究结束后在隔离实验记录中登记来源、方法、结果及限制，并按仓库要求同步更新 `CHANGELOG.md`；任何未执行的检查和训练保持未完成状态。
