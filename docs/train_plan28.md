# 训练计划 28：D-005 多标签 AUCM 对照

状态：**已执行（2026-11-04），两组均失败**——AUC 落到 0.52–0.58、F1@0.5 全为 0，输出塌缩。执行记录见文末。目标是比较两种 AUCM 训练方案，不预设收益。

## 实验设计

| 编号 | 起点 | 胸片训练 | 配置 |
| --- | --- | --- | --- |
| D-005-a | ConvNeXt-Tiny `fb_in1k` ImageNet 权重 + 新分类头 | 从第 1 轮起只用 AUCM | `configs/exp_D005a.yaml` |
| D-005-b | D-002-a 的 BCE checkpoint（明确使用 `best.pt` 或 `best_ema.pt`） | 新开训练，仅用 AUCM 微调 | `configs/exp_D005b.yaml` |

两组 seed 42，320、batch 16 studies、8 轮上限、warmup 2、cosine、patience 2、EMA 0.999；AUCM 使用 PESG（Adam 模式，学习率 3e-4、weight decay 0.01、epoch decay 0），这些是初筛参数，不保证最优。D-005-a 与 D-002-a 比较的是完整训练方案；D-005-b 与其 BCE 起点比较，未设置继续 BCE 训练对照，故不能排除额外训练等因素，也不能将差异单独归因于损失公式。D-005-a、b 均关闭 `pos_weight`，不同时使用 BCE 和 AUCM。

按 Study 采样，每个 batch 选择一个目标类别，各抽 8 个阳性及 8 个阴性 study（有放回）；每轮类别随机排列并循环，保证每类至少被选择一次。稀有 study 会重复采样；一个 epoch 的 batch 数为 max(类别数, ceil(训练 study 数 / 16))，不等同于每个 study 恰好看一遍。LibAUC 2.0.0 的 `MultiLabelAUCMLoss` v1、margin 1，输入为聚合后的 sigmoid 概率，仅计算所选类别；PESG 对未选类别的辅助变量和动量状态予以保护。验证时 `valid_loss` 是未加权 BCE 的监测值，与 AUCM train loss 不能比较；选点看 valid Macro-AUC。EMA 从第二阶段起点重新初始化。

## 环境与启动前检查

在独立环境安装可选依赖：`python -m pip install -r requirements-aucm.txt`，再运行 `python -m pip check`。LibAUC 完整依赖较多，必须先检查对训练环境的影响。检查所有类别训练 study 均有正负样本；任一类缺失会中止。模型的 ImageNet 权重加载失败时 D-005-a 会中止，不允许把随机初始化误当成预训练。真实 GPU、AMP、显存、权重许可及哈希须在目标环境另行核验。

D-005-b 必须指定 BCE checkpoint，例如 D-002-a 的 `best_ema.pt`；不要传目录以免 `--init-from` 自动选择 `best.pt`。记录该文件 SHA-256、来源、split、代码版本及所选 raw/EMA 权重；D-005-b 会校验来源为同 seed、同数据、图像和模型配置的 BCE checkpoint。重划分不支持此 AUCM 实现，禁止复用不同划分的 checkpoint。`--init-from` 仅加载模型，优化器和训练进度重新初始化。

## 命令

```bash
# D-005-a：全程 AUCM
python src/run_train.py --config configs/exp_D005a.yaml --seed 42 --tag D-005-a

# 指定 BCE 起点，必须使用实际文件路径
export D005_INIT='outputs/<D-002-a运行目录>/best_ema.pt'
test -f "$D005_INIT"
shasum -a 256 "$D005_INIT"

# D-005-b：BCE 起点 → AUCM
python src/run_train.py --config configs/exp_D005b.yaml --seed 42 --tag D-005-b --init-from "$D005_INIT"

```

运行后分别指定实际路径并评估 raw 与 EMA；此处示例为 EMA：

```bash
python src/run_eval.py --config configs/exp_D005a.yaml --checkpoint 'outputs/<D-005-a运行目录>/best_ema.pt' --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/D-005-a-EMA.csv
python src/run_eval.py --config configs/exp_D005b.yaml --checkpoint 'outputs/<D-005-b运行目录>/best_ema.pt' --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/D-005-b-EMA.csv
```

## 判读与保存

对照按 [冻结判读协议](experiments.md#新实验判读协议2026-11-04-起)：D-005-a 对 D-002-a 的对应 raw 或 EMA；D-005-b 的 raw 和 EMA 分别与实际加载的 BCE 起点比较，并明确各自权重类型。报告 Study 级 10 类 Macro-AUC、剔除 class1 的 9 类诊断 AUC、各类支持数与差值、F1 阈值曲线及患者整簇配对 bootstrap CI；单 seed 只作初筛。根据结果、成本和不确定性决定是否补 2026、216 对应 seed 的完整配对；两阶段比较须使用各自 seed 的 BCE 起点。

保留 `config.yaml`、`train.log`、`metrics.csv`、`best.pt`、`best_ema.pt`、`last.pt` 及逐 study 预测。AUCM 断点包含损失辅助变量和 PESG 状态；采样按 seed 与 epoch 确定，续训必须保留相同数据顺序、版本和设备。观察 raw/EMA、梯度与 train loss 是否出现非有限值；如出现则停止并排查，不当成有效结果。暂无 GPU、验证集或线上结果。

## 执行记录（2026-11-04）

**环境**：依赖没有按「独立环境」装，改成用 uv 装入主 `.venv`——`uv pip install --python .venv/bin/python --index-url https://mirrors.aliyun.com/pypi/simple libauc==2.0.0`，dry-run 确认新增 40 个包、无任何 `~`/`-` 行，torch 保持 `2.10.0+cu128`、CUDA 可用。`tests/test_aucm.py` 7 passed。这样做的代价是主环境多了 transformers/torch-geometric/medmnist 等包，但未触碰已有版本。硬件为 Tesla V100-SXM2-32G，HEAD `529f9f1`。

**执行**：两组串行，D-002-d 结束后自动接续。

| 实验 | 运行目录 | 起止（CST） | 耗时 | 实际轮数 | best | raw AUC / F1 | EMA AUC / F1 |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| D-005-a | `20261104-110534_D-005-a` | 11:05:29–11:22:44 | 17m15s | 3/8 | 1 | 0.5233 / 0.0000 | 0.5403 / 0.0000 |
| D-005-b | `20261104-112249_D-005-b` | 11:22:44–11:39:57 | 17m13s | 3/8 | 1 | 0.5411 / 0.0000 | 0.5825 / 0.0000 |

D-005-b 的 warm start 确认从 `outputs/20261102-145203_D-002-a/best_ema.pt` 加载（source epoch 5，sha256 `aeeb2106…`），`run_train.py` 的来源校验通过。

**判读**

1. **塌缩，不是收敛慢。** 9 类口径 0.5326 / 0.5538 / 0.5352 / 0.5816，与 10 类同量级；F1 在所有阈值下为 0（EMA 的峰值只在 0.3–0.4 处到 0.25–0.27）。预测分布接近常数：D-005-a raw 概率均值 0.0459、std 0.0026；D-005-b-EMA std 0.0524。
2. **D-005-b 把问题定位到优化/损失侧。** 起点是 BCE-EMA 0.8100，一个 epoch 后 0.5422、第三轮 0.5000。若是「AUCM 从零学不动」可以归因于预算不足，但起点已经被破坏，说明 PESG 的更新方向或步长有问题。
3. 根因未验证，候选按可疑度：(a) lr 3e-4 相对 1157 步/epoch 过大，配 `clip_value=1.0` 会快速把权重带离预训练解；(b) `TaskPESG` 的单类别 mask 恢复逻辑让 `alpha` 与动量状态累积偏差；(c) 8 轮预算本身不匹配 AUCM，需要更长训练与更小 lr。
4. 计划中「D-005-a/b 均关闭 pos_weight」已执行；`valid_loss` 为未加权 BCE 监测值，与 AUCM train loss 不可比，选点看 valid Macro-AUC——实际也是这么做的（第 3 轮 AUC 0.50 后早停）。

**未完成**：lr 单变量对照（3e-5 / 1e-5）、seed 2026/216 复核、患者整簇配对 bootstrap、梯度范数与 alpha 轨迹的逐步记录。按当前结果，这些都应建立在「先不塌缩」之后，故本轮不补。
