# 冻结模型权重、训练分类头设计规格

## 目标

新增一个可复现的训练模式：加载配置指定的骨干（默认仍为 ImageNet 预训练 ResNet-18），冻结骨干参数，只更新 timm 模型的分类头。默认配置和已有实验保持全量微调行为不变。

## 约束与范围

- 通过 `model.freeze_backbone` 配置开关控制，默认值为 `false`。
- 分类头通过 timm 的 `get_classifier()` 识别，不维护按骨干名称分支。
- 开启冻结时，所有参数先设置为 `requires_grad=False`，再仅解冻分类头；分类头必须存在且至少包含一个参数，否则立即报错。
- AdamW 只接收可训练参数；梯度裁剪只统计可训练参数。
- checkpoint metadata 记录冻结开关、总参数量和可训练参数量，便于核对实验口径。
- 新增实验配置只打开冻结开关，不改变 backbone、预训练权重、数据划分、学习率、聚合方式和评估设置。
- 本机只运行合成数据和单元/集成测试，不执行真实 GPU 训练，不伪造实验指标。

## 架构与数据流

1. `load_config` 将 YAML 中的 `model.freeze_backbone` 解析为 `ModelConfig.freeze_backbone`。
2. `build_model` 创建 timm 模型并记录预训练加载状态；若冻结开关开启，调用冻结辅助逻辑设置参数状态。
3. `fit` 创建 AdamW 时过滤 `requires_grad=True` 的参数，训练前向仍保持逐图 logits、study 级聚合和原有损失/指标口径。
4. `run_training` 在保存 `last.pt` 与 `best.pt` 时写入冻结元数据；续训使用 checkpoint 中的完整配置和同样的参数状态。
5. `configs/exp_freeze_head.yaml` 作为可直接执行的实验入口，实际 GPU 环境运行后由用户保存完整产物和指标。

## 错误处理

- 当冻结模式下分类头不存在、不是 `nn.Module` 或没有参数时，`build_model` 抛出带 backbone 名称的 `ValueError`。
- 当模型没有任何可训练参数时，`fit` 在创建优化器前抛出 `ValueError`。
- 关闭冻结模式时保留当前全量参数训练逻辑，不能因为过滤逻辑改变已有训练结果的参数集合。

## 验证标准

- 配置测试确认默认值为 `false`，实验配置只覆盖冻结开关。
- 模型测试确认 ResNet-18 冻结后分类头参数可训练、其余参数冻结，且分类头可识别。
- 训练测试确认优化器参数集合只包含可训练参数，并用一个合成 batch 验证分类头权重发生变化、冻结参数保持不变。
- 完整 pytest 通过；文档、配置、测试和实现变更写入 `CHANGELOG.md`。

## 不在本次范围内

- 不修改 `configs/default.yaml` 的默认训练策略。
- 不新增逐骨干冻结层数、逐层学习率或解冻时间表。
- 不在本机执行完整数据集训练，也不记录未经实际运行的 Macro-AUC / Macro-F1。
