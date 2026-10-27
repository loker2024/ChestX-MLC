# cost.py：赛题第 7 条耗时自检

`cost.py` 位于仓库根目录，对应赛题「端到端单张图片推理耗时」评测：按官方指定的调用方式，在 100 张图上以 `batch=1` 串行、连续跑 3 轮，给出单张平均耗时，并顺带核对「现场预测」与最终 `submission.csv` 是否一致。规则原文见 `docs/比赛参赛规则.md`，第 7 条与附录 A。

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

- 退出码 `0`：计时样本均为单图 study，平均耗时未超 100 ms 且预测与 `submission.csv` 一致；
- 退出码 `1`：超限或存在不一致的 study；
- 退出码 `2`：参数、文件、配置或权重错误，或样本包含多图 study 导致最终端到端耗时无法判定（打印 `[ERROR]`）。

完整测试集索引不可用（赛题附录 A 的提交包不要求携带测试集，评测环境常见）不算错误：脚本打印 `[WARN]` 后退化为按 `pictures_100` 自身分组比对，退出码仍按上面三条判定，报告里 `final_timing_verifiable` 为 `false`。

日志前缀：`[INFO]` 过程信息、`[WARN]` 非致命提示（如图片数不足 100 / 不足 5 张、CPU 上跑 fp16）、`[ERROR]` 失败原因。最后一行固定输出 `[CONCLUSION]`，格式对齐规则里的日志示例，便于直接粘贴到提交材料。

## 2. 配置与权重的自动查找

完整测试集索引（`data.test.csv` + `data.test.images`）可用时，脚本用它确认抽样 study 的全部图片并检查患者身份；索引可用但抽样图与索引的 Subject_id / 文件名对不上会返回 2。索引缺失时打印 `[WARN]` 后退化为按 `pictures_100` 自身分组，报告标注不可核实。配置和权重的查找顺序为：

- 权重顺序：`--checkpoint` → 环境变量 `COST_CHECKPOINT` → `submission.csv` 同目录的 `best.pt` / `model/best.pt` → 仓库/提交包根目录的 `best.pt` / `model/*.pt` → `outputs/*/best.pt`（按修改时间倒序）。
- 配置顺序：`--config` → 仓库 `configs/default.yaml` → 提交包根目录 `config.yaml` → checkpoint 里存档的训练配置（经 `config_from_dict` 还原）。

找不到时报错并返回 2，提示用 `--checkpoint` / `--config` 指定。

## 3. 计时口径

单张图的计时区间（对应规则「从读取图片文件到写出最终预测结果，包含预处理与后处理」）：

1. 同步加速设备（CUDA / MPS）后取 `perf_counter`；
2. `infer.predict.predict_image`：读图（PIL）→ eval 预处理（letterbox / resize / normalize / ToTensor）→ 单图张量搬设备 → 前向 → `aggregate_logits` → `sigmoid` → 转回 CPU `float32`；
3. 按提交文件对应 study 的模式输出全部类别或按阈值筛选；写入包含患者、检查、类别和概率的临时 CSV，文件关闭后再次同步设备并取 `perf_counter`。临时文件在计时结束后自动清理。

计时循环外层 3 轮、内层 100 张，每个列表的耗时结构为 `round_times_ms[round][image]`；`summarize_timings` 先对每张图按 3 轮取均值，再对 100 张取均值，同时给出中位数、p95、最小、最大与总耗时。100 ms 上限比较的是「100 张均值」。

预热：正式计时前把 `pictures_5` 的每张图各跑一遍（同样走 `predict_image` + 后处理），首次前向触发的 kernel 编译/缓存与初始化开销落在预热里，不计入成绩。

对于单图 study，这包含读取、预测、后处理和结果写出。对于多图 study，单图结果只是中间候选，不能代替完整 study 聚合后的提交结果；当前计时数字无法证明最终端到端耗时，脚本会明确报告“无法判定”，并返回 2。赛题对单图计时与多图聚合之间的计费口径未作明确说明，须与主办方确认后在指定硬件上重新实测。

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

计时前核对样本路径与完整测试集索引。计时结束后，`cost.py` 对抽中的每个 study 使用抽样图片和完整测试集中的其他图片，以 `predict_study` 求 logits 均值并与 `submission.csv` 比对：

- 提交文件输出全部 10 类时：逐类比较概率绝对差，最大差 `<= --tolerance` 记为一致；
- 提交文件按阈值过滤过时（某个 study 少于 10 行）：先按 `--threshold` 对现场概率做同样的过滤，比较保留的类别集合，再比较交集类别的概率；
- 提交文件里缺少某个 study：记为不一致；
- 报告里的 `checked_images` 是参与比对的图片数（即 `pictures_100` 的图片总数）。

比对同时检查样本路径、完整测试集与提交文件中的 `Subject_id`，并拒绝非有限或超出 [0, 1] 的现场概率。完整测试集缺失时退化为只按 `pictures_100` 自身分组比对，此时无法发现抽样目录之外的同 study 图片，可能出现假不一致，结果只作候选。提交文件必须由同一份权重和配置生成；额外读取的同 study 图片仅用于一致性检查，不计入上述单图计时。

## 6. 精度声明

默认 `fp32`。`--precision fp16` 时脚本把模型参数转成半精度（`model.half()`），并在日志与 `[CONCLUSION]` 中标注 `推理精度 fp16`。赛题允许 FP16 / INT8，但要求**在提交说明中注明**所采用的精度；使用 fp16 后数值与 fp32 提交文件可能有细微差异，比对时按 `--tolerance` 判定（必要时可适当放宽，但需理解这是数值误差而非逻辑差异）。

## 7. 报告字段（`--report`）

JSON 顶层键：

- `checkpoint` / `device` / `precision` / `threshold`：本次实际使用的权重与推理设置；
- `config`：本次生效配置（`Config.to_dict()`）；
- `pictures_100` / `warmup_images`：计时图与预热图数量；
- `timing_ms`：`rounds`、`images`、`mean`、`median`、`p95`、`min`、`max`、`total`、`limit`、`within_limit`、`final_timing_verifiable`（多图 study 或完整索引不可用时为 false；`within_limit` 在多图时为 null）；
- `comparison`：`filtered_submission`、`checked_studies`、`checked_images`、`inconsistent_studies`、`inconsistent_images`、`max_abs_diff`、`tolerance`、`issues`（每个不一致 study 的 `study_id` / `reason` / `max_abs_diff`）。

报告可用于提交材料附录或赛后复盘。

## 8. 本地验证记录

- 合成数据单元测试：`tests/test_cost.py`（不在测试里读取真实影像），覆盖图片发现与身份解析、提交文件校验、阈值选择、一致性比对、计时统计算术、`predict_image` 与 `predict_loader` / `predict_study` 的数值等价，以及 `main` 端到端（含 `[CONCLUSION]`、超限返回 1、不一致返回 1、缺权重返回 2、多图 study 返回 2、抽样 study 按完整索引补齐），并覆盖完整索引缺失时退化为抽样分组、报告 `final_timing_verifiable=false` 的路径。
- 真实结构端到端（2026-10-27，本机 Apple Silicon CPU，随机初始化 resnet18，`image.size=320`，示例数据 100 张 / 69 个 study）：旧版脚本曾报告单张均值 39.7 ms；该次未计入结果写出，且多图聚合不在单张计时内，不能作为现有口径的实测成绩。正式成绩须在赛题指定硬件、加载训练好的最终权重后测。

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
- 多图 study 的单图计费口径尚需主办方确认；脚本不会据候选结果耗时宣称达标。`pictures_100` 不是 100 张时脚本会打 `[WARN]` 但继续执行，方便调试。
- 评测环境通常没有完整测试集文件：脚本会退化为按 `pictures_100` 自身分组，无法补齐同 study 的未抽样图片，也不能证明「单图 study」判定；正式成绩仍以能读到完整索引的本地 / 服务器实测为准。
