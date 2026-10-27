# cost.py：赛题第 7 条耗时自检

`cost.py` 位于仓库根目录，对应赛题「端到端单张图片推理耗时」评测：按官方指定的调用方式，在 100 张图上以 `batch=1` 串行、连续跑 3 轮，给出单张平均耗时，并顺带核对「现场预测」与最终 `submission.csv` 是否一致。规则原文见 `docs/比赛参赛规则.md`（本地副本，不入库），第 7 条与附录 A。

## 1. 快速开始

```bash
# 在仓库根目录（或提交包根目录）执行，参数顺序与规则一致
python cost.py <submission.csv> <pictures_100> <pictures_5>

# 例：本地联调时显式指定配置与权重
python cost.py outputs/submission.csv data/cost/pictures_100 data/cost/pictures_5 \
    --config configs/default.yaml \
    --checkpoint outputs/<run>/best.pt \
    --report outputs/cost_report.json
```

图片目录结构与测试集一致：`<subject_id>/<study_id>/<pic_id>.jpg`（后缀支持 `.jpg` / `.jpeg` / `.png`，见 `src/common/constants.py` 的 `IMAGE_SUFFIXES`），`pictures_100` 放 100 张计时图，`pictures_5` 放 5 张预热图。

### 1.1 参数

| 参数 | 说明 |
|---|---|
| `submission`（位置） | 最终提交的 `submission.csv`，用于预测一致性比对 |
| `pictures_100`（位置） | 100 张计时图目录 |
| `pictures_5`（位置） | 5 张预热图目录，预热不计入成绩 |
| `--config` | 配置文件；缺省按 §2 顺序自动查找 |
| `--checkpoint` | 权重路径；缺省按 §2 顺序自动查找 |
| `--device` | `auto` / `cpu` / `cuda` / `cuda:0` / `mps`，缺省取配置里的 `device` |
| `--precision {fp32,fp16}` | 推理精度，默认 `fp32`；`fp16` 须在提交说明中声明 |
| `--rounds` | 连续重复轮数，规则要求 3，默认 3 |
| `--threshold` | 后处理阈值，缺省用配置里的 `eval.threshold`（默认 0.5） |
| `--tolerance` | 与 submission 比对时允许的最大概率绝对差，默认 `1e-3` |
| `--report` | 把计时与比对明细写成 JSON 报告 |

### 1.2 退出码与日志

- 退出码 `0`：平均耗时未超 100 ms 且预测与 `submission.csv` 一致；
- 退出码 `1`：超限或存在不一致的 study；
- 退出码 `2`：参数、文件、配置或权重错误（打印 `[ERROR]`）。

日志前缀：`[INFO]` 过程信息、`[WARN]` 非致命提示（如图片数不足 100 / 不足 5 张、CPU 上跑 fp16）、`[ERROR]` 失败原因。最后一行固定输出 `[CONCLUSION]`，格式对齐规则里的日志示例，便于直接粘贴到提交材料。

## 2. 配置与权重的自动查找

真实评测环境通常只把提交包解压后直接跑 `python cost.py ...`，因此脚本不要求额外参数：

- 权重顺序：`--checkpoint` → 环境变量 `COST_CHECKPOINT` → `submission.csv` 同目录的 `best.pt` / `model/best.pt` → 仓库/提交包根目录的 `best.pt` / `model/*.pt` → `outputs/*/best.pt`（按修改时间倒序）。
- 配置顺序：`--config` → 仓库 `configs/default.yaml` → 提交包根目录 `config.yaml` → checkpoint 里存档的训练配置（经 `config_from_dict` 还原）。

找不到时报错并返回 2，提示用 `--checkpoint` / `--config` 指定。

## 3. 计时口径

单张图的计时区间（对应规则「从读取图片文件到写出最终预测结果，包含预处理与后处理」）：

1. 同步加速设备（CUDA / MPS）后取 `perf_counter`；
2. `infer.predict.predict_image`：读图（PIL）→ eval 预处理（letterbox / resize / normalize / ToTensor）→ 单图张量搬设备 → 前向 → `aggregate_logits` → `sigmoid` → 转回 CPU `float32`；
3. 后处理：`select_prediction_rows` 按阈值选出提交行（与 `infer.submission.build_submission_frame` 同一口径：概率 `>=` 阈值保留，全不过线时保留 top1 类别）；
4. 再次同步设备后取 `perf_counter`。

计时循环外层 3 轮、内层 100 张，每个列表的耗时结构为 `round_times_ms[round][image]`；`summarize_timings` 先对每张图按 3 轮取均值，再对 100 张取均值，同时给出中位数、p95、最小、最大与总耗时。100 ms 上限比较的是「100 张均值」。

预热：正式计时前把 `pictures_5` 的每张图各跑一遍（同样走 `predict_image` + 后处理），首次前向触发的 kernel 编译/缓存与初始化开销落在预热里，不计入成绩。

说明：逐张写 CSV 的磁盘 I/O 不在单张区间内——提交文件是一次性整体写出的，评测方按整体流程计；脚本在计时循环之后统一做一致性比对与（可选）报告落盘。

## 4. 与 `predict.py` 的一致性要求

规则要求 `cost.py` 必须导入并调用与 `predict.py` 相同的推理 / 加载逻辑。本实现直接复用 `src/infer/predict.py` 的公开函数，不多写一套推理：

| 用途 | 函数 |
|---|---|
| 加载权重 | `load_checkpoint_model(config, checkpoint_path, device)`（`run_infer.py` 同款） |
| 构造 eval 预处理 | `build_eval_inference_transform(config)` |
| 单图计时推理 | `predict_image(model, image_path, transform, device)` |
| study 聚合校验 | `predict_study(model, image_paths, transform, device)` |

`predict_loader`（`run_infer.py` 走批量 loader 的那条路径）与 `predict_image` / `predict_study` 共用内部的 `_forward_probabilities`，因此「loader 批量推理」和「cost.py 单图 / study 聚合」在数值上是同一套前向、聚合与 sigmoid，区别只在 batch 组织方式。

## 5. 一致性比对口径

计时结束后，`cost.py` 把 100 张图按 `<study_id>` 分组，用 `predict_study` 对每个 study 做同一次检查内的 logits 均值聚合（与 `run_infer.py` 生成提交文件时的聚合方式一致），再与 `submission.csv` 比对：

- 提交文件输出全部 10 类时：逐类比较概率绝对差，最大差 `<= --tolerance` 记为一致；
- 提交文件按阈值过滤过时（某个 study 少于 10 行）：先按 `--threshold` 对现场概率做同样的过滤，比较保留的类别集合，再比较交集类别的概率；
- 提交文件里缺少某个 study：记为不一致；
- 报告里的 `checked_images` 是参与比对的图片数（即 `pictures_100` 的图片总数）。

重要前提：study 聚合概率依赖该 study 的全部图片。如果 `pictures_100` 只包含某个 study 的一部分图片，它与提交文件（用测试集全部图片生成）的聚合概率本来就对不上。**准备计时图片时请按 study 成组提供**（一个 study 的所有图片一起放进 `pictures_100`），否则会看到「不一致」而不是脚本算错。同理，`pictures_100` 与 `pictures_5` 中的图片不会互相影响，但提交文件必须是由「同一份权重 + 同一份配置」生成的。

## 6. 精度声明

默认 `fp32`。`--precision fp16` 时脚本把模型参数转成半精度（`model.half()`），并在日志与 `[CONCLUSION]` 中标注 `推理精度 fp16`。赛题允许 FP16 / INT8，但要求**在提交说明中注明**所采用的精度；使用 fp16 后数值与 fp32 提交文件可能有细微差异，比对时按 `--tolerance` 判定（必要时可适当放宽，但需理解这是数值误差而非逻辑差异）。

## 7. 报告字段（`--report`）

JSON 顶层键：

- `checkpoint` / `device` / `precision` / `threshold`：本次实际使用的权重与推理设置；
- `config`：本次生效配置（`Config.to_dict()`）；
- `pictures_100` / `warmup_images`：计时图与预热图数量；
- `timing_ms`：`rounds`、`images`、`mean`、`median`、`p95`、`min`、`max`、`total`、`limit`、`within_limit`；
- `comparison`：`filtered_submission`、`checked_studies`、`checked_images`、`inconsistent_studies`、`inconsistent_images`、`max_abs_diff`、`tolerance`、`issues`（每个不一致 study 的 `study_id` / `reason` / `max_abs_diff`）。

报告可用于提交材料附录或赛后复盘。

## 8. 本地验证记录

- 合成数据单元测试：`tests/test_cost.py`（不在测试里读取真实影像），覆盖图片发现与身份解析、提交文件校验、阈值选择、一致性比对、计时统计算术、`predict_image` 与 `predict_loader` / `predict_study` 的数值等价，以及 `main` 端到端（含 `[CONCLUSION]`、超限返回 1、不一致返回 1、缺权重返回 2）。
- 真实结构端到端（2026-10-27，本机 Apple Silicon CPU，随机初始化 resnet18，`image.size=320`，示例数据 100 张 / 69 个 study）：3 轮本均 38.98 / 39.39 / 40.83 ms，单张均值 39.7 ms，不一致 0，最大概率差 `5.96e-08`，退出码 0。这组数字只证明链路与口径可用，**不是**提交用的实测成绩；正式成绩须在赛题指定硬件、加载训练好的最终权重后测。

## 9. 提交清单（附录 A）

提交包根目录至少包含：

```text
cost.py
config.yaml        # 或 configs/default.yaml，保证 cost.py 能自动找到
best.pt            # 或 model/best.pt，与生成 submission.csv 的权重一致
submission.csv     # 最终提交文件
```

赛题附录 A 要求提交包根目录有 `cost.py`；模型与配套文件解压后总大小不超过 500 MB（第 8 条）。`pictures_100` / `pictures_5` 由主办方提供，不需要随包提交；本地自测时请用与测试集同结构的目录。

## 10. 已知限制

- 计时受机器负载、CPU 频率与温度影响；单次 100 张 × 3 轮的样本量已能压住大部分抖动，但本机数字不能替代赛题服务器实测。
- 图片读取是同步串行的（规则要求 `batch=1` 串行），没有做 IO 并行或预取，因此测的是真实端到端墙钟。
- study 多图场景依赖 §5 的成组前提；`pictures_100` 不是 100 张时脚本会打 `[WARN]` 但继续执行，方便调试。
