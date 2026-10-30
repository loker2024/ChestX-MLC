# 冻结分类头训练实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。

**目标：** 为 ChestX-MLC 增加可配置的“冻结骨干、只训练分类头”模式，并提供可复现的实验配置与验证。

**架构：** `ModelConfig.freeze_backbone` 控制模型构建时的参数冻结；timm 的 `get_classifier()` 负责识别分类头。训练引擎只把可训练参数交给 AdamW，训练入口把冻结状态和参数计数写入 checkpoint metadata。默认配置仍保持全量微调。

**技术栈：** Python 3.12、PyTorch、timm、dataclass 配置、pytest、YAML。

---

### 任务 1：配置字段与实验入口

**文件：**
- 修改：`src/common/config.py`
- 创建：`configs/exp_freeze_head.yaml`
- 测试：`tests/test_config.py`

- [ ] **步骤 1：编写失败的测试**

增加配置断言：默认 `freeze_backbone` 为 `False`；实验配置加载后为 `True`，且除该字段外与默认配置一致。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_config.py -q`
预期：实验配置文件不存在或 `ModelConfig` 没有 `freeze_backbone`，测试失败。

- [ ] **步骤 3：编写最少实现代码**

在 `ModelConfig` 增加 `freeze_backbone: bool = False`；创建：

```yaml
_base: default.yaml
model:
  freeze_backbone: true
```

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_config.py -q`
预期：配置测试全部通过。

- [ ] **步骤 5：提交**

```bash
git add src/common/config.py configs/exp_freeze_head.yaml tests/test_config.py
git commit -m "feat: 增加冻结分类头实验配置"
```

### 任务 2：模型冻结逻辑

**文件：**
- 修改：`src/model/network.py`
- 测试：`tests/test_model.py`

- [ ] **步骤 1：编写失败的测试**

新增测试构建 `freeze_backbone=True` 的 ResNet-18，断言 `get_classifier()` 的参数全部可训练、其他参数全部冻结，并断言至少存在一个可训练参数。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_model.py -q`
预期：`ModelConfig` 尚未执行冻结，骨干参数仍可训练，测试失败。

- [ ] **步骤 3：编写最少实现代码**

在 `network.py` 添加冻结辅助函数：先将所有参数设为 `False`，调用 `model.get_classifier()`，校验其为 `nn.Module` 且有参数，再将分类头参数设为 `True`。`build_model` 在模型创建后按配置调用它。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_model.py -q`
预期：模型测试全部通过。

- [ ] **步骤 5：提交**

```bash
git add src/model/network.py tests/test_model.py
git commit -m "feat: 支持冻结骨干训练分类头"
```

### 任务 3：优化器只更新分类头

**文件：**
- 修改：`src/train/engine.py`
- 测试：`tests/test_engine.py`

- [ ] **步骤 1：编写失败的测试**

新增合成训练测试：冻结模式下记录模型初始 state dict，运行一轮 `fit`，断言分类头至少一个权重变化、所有冻结参数保持不变；另加测试用可观测 optimizer 参数集合只包含 `requires_grad=True` 参数。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_engine.py -q`
预期：当前 AdamW 接收全部参数，冻结参数或测试中的优化器集合断言失败。

- [ ] **步骤 3：编写最少实现代码**

在 `fit` 中构造：

```python
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
if not trainable_parameters:
    raise ValueError("模型没有可训练参数")
optimizer = torch.optim.AdamW(trainable_parameters, ...)
```

梯度裁剪同样传入 `trainable_parameters`。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_engine.py -q`
预期：训练引擎测试全部通过，冻结参数不变。

- [ ] **步骤 5：提交**

```bash
git add src/train/engine.py tests/test_engine.py
git commit -m "fix: 优化器仅更新可训练参数"
```

### 任务 4：checkpoint 元数据与入口回归

**文件：**
- 修改：`src/run_train.py`
- 测试：`tests/test_run_train.py`

- [ ] **步骤 1：编写失败的测试**

在现有 CLI 合成训练测试中断言 `metadata.freeze_backbone`、`metadata.trainable_params` 与 `metadata.total_params` 存在且计数关系正确；增加冻结配置 CLI smoke test，确认 checkpoint metadata 记录冻结状态。

- [ ] **步骤 2：运行测试验证失败**

运行：`python -m pytest tests/test_run_train.py -q`
预期：metadata 尚未包含这些字段，测试失败。

- [ ] **步骤 3：编写最少实现代码**

在 `run_training` 中从模型参数计算总参数量和可训练参数量，新增 `_model_metadata` 或等价小函数，并在 `last.pt` / `best.pt` 的 metadata 中写入冻结状态与计数；保持原有 `pretrained_loaded` 和 `device` 字段。

- [ ] **步骤 4：运行测试验证通过**

运行：`python -m pytest tests/test_run_train.py -q`
预期：训练入口测试全部通过。

- [ ] **步骤 5：提交**

```bash
git add src/run_train.py tests/test_run_train.py
git commit -m "feat: 记录冻结训练元数据"
```

### 任务 5：文档、变更日志与全量验证

**文件：**
- 修改：`CHANGELOG.md`
- 修改：`docs/manual.md`（如现有配置表需要登记新实验入口）
- 修改：`docs/experiments.md`（记录实验待执行状态）

- [ ] **步骤 1：补文档与变更记录**

按 Asia/Shanghai 当前日期与分钟记录配置字段、冻结逻辑、测试和新增实验配置；明确本机未执行真实 GPU 训练，避免写入指标。

- [ ] **步骤 2：运行聚焦验证**

运行：`python -m pytest tests/test_config.py tests/test_model.py tests/test_engine.py tests/test_run_train.py -q`
预期：聚焦测试全部通过。

- [ ] **步骤 3：运行完整验证**

运行：`python -m pytest -q`
预期：全量测试通过，已有跳过项数量不增加。

- [ ] **步骤 4：检查差异与配置**

运行：`git diff --check`、`git status --short`、`git diff --stat`，确认没有真实数据、权重或训练产物进入变更。

- [ ] **步骤 5：提交**

```bash
git add CHANGELOG.md docs/manual.md docs/experiments.md
git commit -m "docs: 记录冻结分类头实验入口"
```
