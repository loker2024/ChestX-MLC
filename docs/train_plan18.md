# 训练计划 18：A-001 的聚合空间对照（logits 空间 G0 → 概率空间 G1）

**状态：已完成（2026-10-31），GPU 训练与评估已执行，结论：未过门、方向否定。**

## 目标与配置

在 A-001 的同一训练口径下，只改「聚合在哪个空间进行」，检验概率空间平均能否提升 study 级
Macro-AUC。候选配置 [exp_A008.yaml](../configs/exp_A008.yaml) 以 `_base: exp_A001.yaml` 继承
A-001，只覆盖 `model.aggregation: view_weighted → view_weighted_probability`，并显式固定
`model.frontal_weight: 0.7`，正侧位仍是 7:3。

同一 study 内，`z_i` 为第 i 张图的 logit，`正位` / `侧位` 为两个视角分组的图片集合：

| 实验 | 聚合口径 | 定义 |
| --- | --- | --- |
| A-001（G0） | logits 空间加权 | `sigmoid(0.7·mean(z_正位) + 0.3·mean(z_侧位))` |
| A-008（G1） | 概率空间加权 | `0.7·mean(σ(z_正位)) + 0.3·mean(σ(z_侧位))` |

分组规则、单视角与含未知视角的回退规则完全沿用 `aggregate_logits`：这两种 study 回退为
全部图片等权均值，因此 A-008 与 A-001 在这些 study 上结果相同，差异只可能出现在「正侧位都
有」的多图 study。

其余口径逐字段继承 A-001：ConvNeXt-Tiny `fb_in1k` @320×320、ImageNet 归一化、无水平翻转、
标准分类头（dropout 0）、BCE + `pos_weight`、`batch_studies: 16`、lr 0.0003、weight decay
0.01、warmup 2 + cosine 到 1e-5、最多 8 轮、patience 2、AMP、seed 42。参数量与 A-001 相同
（27,827,818），训练预算相同，差值可归因于聚合空间。

## 与 AGG-01（后处理对照）的区别

[AGG-01](train_plan_aggregation.md) 固定 A-001 的权重、只在推理阶段重算 G0 / G1，回答的是
「同一个模型换后处理有没有用」。A-008 是**重新训练**：loss、验证、推理全程用 G1，回答的是
「用 G1 训练出来的模型会不会更好」。两者不能互相替代：

- AGG-01 的 G1 从未参与反向传播，模型仍是按 G0 优化的；
- A-008 的模型按 G1 优化，但训练-推理口径一致，不存在后处理失配。

若 AGG-01 显示 G1 在同权重下就更好，A-008 才值得占用 GPU 预算；若 AGG-01 明显变差，A-008
预计也难有增益。

## 实现口径

`model.aggregation: view_weighted_probability` 由 `ModelConfig.probability_space` 表达，
`aggregate_logits(..., probability_space=True)` 先逐图 sigmoid、在概率空间聚合，最后用
`logit(A(σ(z)))` 变换回 logits 空间：

- `BCEWithLogitsLoss` 对 `logit(p)` 的取值与直接用概率算 BCE 等价，训练不需要改损失；
- 推理侧 `sigmoid(logit(p)) == p`，拿到的正是概率空间加权均值；
- 两端用 `PROBABILITY_EPS = 1e-6` 夹住，避免 `logit(0)` / `logit(1)` 变成无穷；
- 概率先升 fp32 再聚合，避免 fp16 下逐图 sigmoid 下溢。

`train/engine.py`、`infer/predict.py`、`src/cost.py` 统一按 `config.model.uses_views` 决定是否
传视角、按 `config.model.probability_space` 决定聚合空间，训练、验证、提交、耗时自检共用
同一条路径。

## 执行命令

```bash
python -m pytest -q
python src/check_data.py --config configs/exp_A008.yaml --full

python src/run_train.py --config configs/exp_A008.yaml --seed 42 \
    --tag A-008_convnext_tiny_g1_s42

python src/run_eval.py --config configs/exp_A008.yaml \
    --checkpoint outputs/<run>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7
```

执行前确认展开配置与 A-001 仅 `model.aggregation` 不同（`tests/test_config.py::test_a008_probability_aggregation_config_loads`
已断言逐字段一致），并核对 checkpoint metadata 记录的是 `aggregation: view_weighted_probability`。

## 判读

- 对照基线为 A-001（0.8038 / 0.4255，同骨干、同 seed 42、同设备）。差值按项目既有门解读：
  AUC 需 ≥ +0.005 才算方向性有效，单 seed 差值在 ±0.0033 内视为波动。
- 若 A-008 优于 A-001，需在 seed 2026 复核后才可列入候选；若低于或持平，概率空间聚合训练
  方向否定，默认聚合保持 `view_weighted`。
- 同时检查 F1 曲线是否整体右移：概率空间平均会收缩极端概率，F1@0.5 可能下降而 AUC 上升，
  这类情况只作为校准差异记录，不据此调整提交阈值。
- 结果回填 `docs/training_log.md`（配置、环境、逐轮、阈值曲线、逐类对比）与
  `docs/experiments.md`（是否达门、是否复核），并更新本文件状态行。

## 当前状态

- [x] 代码支持概率空间聚合（`view_weighted_probability`），本机 `pytest` 全绿。
- [x] 配置 `configs/exp_A008.yaml` 就绪，与 A-001 仅聚合空间不同。
- [x] GPU 训练与评估（19:06:29–19:38:48，32m19s，第 7 轮早停，best epoch 5）。
- [x] 结果回填并形成结论。

## 执行结果

- 产物：`outputs/20261031-190629_A-008_convnext_tiny_g1_s42/`（`best.pt` 111,385,259 B / `last.pt` / `metrics.csv` / `train.log` / `config.yaml`）、`outputs/eval_reports/A-008_full.txt`、`outputs/A-008_valid_predictions.csv`。
- 结果：best epoch 5，valid macro-AUC **0.7974** / F1@0.5 **0.4197**；相对 A-001（0.8038 / 0.4255）差 AUC −0.0064、F1@0.5 −0.0058，距 0.8088 门差 −0.0114 → **未过门**。
- 结论：概率空间聚合训练方向否定，默认聚合保持 `view_weighted`，不做 seed 2026 复核、不进入候选提交。
