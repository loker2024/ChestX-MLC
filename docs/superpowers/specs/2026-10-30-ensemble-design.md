# 多模型融合设计规格

## 目标

新增一个独立的 `src/run_ensemble.py` 入口，加载多个训练好的 checkpoint，支持概率加权和 logits 加权两种融合方式。对 `train` / `valid` split 复用现有指标封装计算 Macro-AUC、指定阈值下的 Macro-F1、逐类报告和阈值曲线；对 `test` split 生成完整 10 类概率的 `submission.csv`。

## 约束与口径

- 每个 checkpoint 先使用自己的训练配置完成逐图前向和 `Study_id` 内聚合，然后才进行模型间融合。
- checkpoint 优先使用内部保存的完整配置；旧 checkpoint 没有配置时，必须通过等数量的 `--config` 显式补充。
- 所有模型的 study ID 必须一致，融合按 `Study_id` 对齐，不能按 DataFrame 行号盲目相加。
- `probability` 融合为 `sum(weight * sigmoid(study_logits))`。
- `logit` 融合为 `sigmoid(sum(weight * study_logits))`，权重在使用前归一化。
- 权重为非负数，至少一个权重为正；未提供权重时使用等权。
- `valid` / `train` 的标签来自第一个模型 loader，并检查其余模型的 study ID 与标签一致。
- AUC 使用融合后的连续概率；F1 使用 `eval.threshold`，可用 `--thresholds` 输出多个阈值的 Macro-F1 曲线。
- `test` 默认输出全部 10 个类别概率，使用现有 `build_submission_frame` 和 `validate_submission` 校验。

## 命令行接口

```bash
python src/run_ensemble.py \
  --checkpoint outputs/run_a/best.pt outputs/run_b/best.pt \
  --weights 0.5,0.5 \
  --fusion probability \
  --split valid \
  --thresholds 0.3,0.4,0.5,0.6,0.7 \
  --out outputs/ensemble_valid
```

参数约定：

- `--checkpoint`：一个或多个 checkpoint 路径。
- `--config`：可选的配置路径列表，数量必须与 checkpoint 相同；仅用于 checkpoint 没有内嵌配置时回退。
- `--weights`：逗号分隔的全局模型权重，数量必须与 checkpoint 相同；省略时等权。
- `--fusion`：`probability` 或 `logit`，默认 `probability`。
- `--split`：`train`、`valid` 或 `test`，默认 `test`。
- `--thresholds`：有标签 split 的 F1 阈值网格；省略时只输出配置默认阈值的指标。
- `--out`：输出目录。`test` 写 `submission.csv`；有标签 split 写 `metrics.json`、`per_class.csv`、`f1_curve.csv` 和 `predictions.csv`。

## 模块边界

- `src/infer/predict.py`：新增 study-level logits 宽表推理入口，保留现有概率入口兼容性。
- `src/infer/ensemble.py`：实现权重解析后的宽表校验、概率/logits 融合和模型配置/标签对齐辅助函数，不负责文件写出。
- `src/run_ensemble.py`：解析 CLI、按模型配置构建 loader 和模型、调用融合函数、计算指标或写 submission。
- `tests/test_ensemble.py`：覆盖两种融合公式、权重与 ID/形状校验、valid 指标产物和 test submission 入口。
- `CHANGELOG.md`：记录新增入口、融合模块和测试。

## 错误处理

- checkpoint 数量少于 1、权重数量不匹配、权重负数或权重全为 0 时立即报错。
- `--config` 只提供部分路径时立即报错，避免不同模型误用同一配置。
- 模型 study ID、标签形状或标签内容不一致时立即报错。
- 融合结果出现 NaN/无穷或概率超出 `[0, 1]` 时立即报错。
- `test` submission 继续使用现有字段、类别范围、study 覆盖和 Subject_id 校验。

## 验收标准

1. 概率加权和 logits 加权分别与手算结果一致。
2. 等权默认值、权重归一化和非法输入均有单元测试。
3. `valid` 运行复用 `common.metrics`，输出 Macro-AUC、Macro-F1、逐类报告和阈值曲线。
4. `test` 运行输出通过现有 submission 校验的完整 10 类概率文件。
5. 现有单模型推理、训练和指标测试不受影响。
