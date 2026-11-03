# cost.py：赛题第 7 条单 study 耗时自检

仓库中的入口是 `src/cost.py`，对应赛题「端到端单次检查推理耗时」评测。2026-11-03 官方答疑群明确了三件改变口径的事：

1. **最小计时单元是单个 `study_id`**，不是单张图片；`batch=1` 以 study 为单位，一次检查的多张图可以同时处理；
2. **预热也以 study 为单位**，用 5 个与正式测试不相交的 study；
3. **初始化可以放 CSV 元数据加载、study 分组、路径索引**，但只许碰元数据、一次完成、打印 `init_ms` 受上限核验；解码、前向、预测不得藏进初始化。

据此，本脚本按「读该 study 全部图片 → 解码 → 预处理 → 前向 → study 聚合 → 后处理 → 逐图写出 → flush」计时，单 study 平均耗时与 100 ms 比较。规则原文见 `docs/比赛参赛规则.md`，群答疑要点见本文档第 2 节。

## 1. 快速开始

```bash
# 在仓库根目录（或提交包根目录）执行，参数顺序与规则一致
python src/cost.py <submission.csv> <pictures_100> <pictures_5>

# 本地联调：显式指定配置与权重
python src/cost.py outputs/submission.csv data/cost/pictures_100 data/cost/pictures_5 \
    --checkpoint outputs/<run>/best.pt \
    --prediction-out outputs/cost_predictions.csv \
    --report outputs/cost_report.json

# 多模型融合提交：传多个权重，或依赖清单自动发现（见第 3 节）
python src/cost.py submission.csv pictures_100 pictures_5 \
    --checkpoint a.pt b.pt c.pt d.pt --weights 1,1,1,1 --fusion probability
```

图片目录结构与测试集一致：`<subject_id>/<study_id>/<pic_id>.jpg`（后缀支持 `.jpg` / `.jpeg` / `.png`）。`pictures_100` 放 100 个计时 study 的全部图片，`pictures_5` 放 5 个预热 study 的全部图片。官方示例数据为 10 个 study / 18 张图和 5 个 study / 10 张图，脚本会打 `[WARN]` 但正常执行。

### 1.1 参数

| 参数 | 说明 |
|---|---|
| `submission`（位置） | 最终提交的 `submission.csv`，用于预测一致性比对 |
| `pictures_100`（位置） | 100 个计时 study 的图片目录 |
| `pictures_5`（位置） | 5 个预热 study 的图片目录，预热不计入成绩 |
| `--checkpoint` | 一个或多个权重路径；缺省读融合清单，再缺省按单模型顺序查找 |
| `--ensemble` | 融合清单 JSON；缺省在 `submission.csv` 同目录与仓库根目录找 `cost_ensemble.json` |
| `--weights` | 逗号分隔的模型权重，缺省等权；数量须与模型数一致 |
| `--fusion {probability,logit}` | 融合方式，缺省取清单值，再缺省 `probability` |
| `--config` | 配置文件；仅建议单模型调试时使用，多模型优先用各自 checkpoint 存档配置 |
| `--device` | `auto` / `cpu` / `cuda` / `cuda:0` / `mps`，缺省取配置里的 `device` |
| `--precision {fp32,fp16}` | 推理精度，默认 `fp32`；`fp16` 须在提交说明中声明 |
| `--rounds` | 连续重复轮数，规则要求 3，默认 3 |
| `--threshold` | 后处理阈值，缺省用配置里的 `eval.threshold`（默认 0.5） |
| `--tolerance` | 与 submission 比对时允许的最大概率绝对差，默认 `1e-3` |
| `--views-csv` | 视角元数据 CSV；缺省找 `cost_views.csv` 或配置里的 `data.test.csv` |
| `--prediction-out` | 逐图预测 CSV 的落盘路径；缺省写临时文件并在结束后清理 |
| `--preprocess-workers` | study 内并行解码/缩放的最大线程数；1 表示串行，默认 6 |
| `--init-limit-ms` | 初始化耗时自设上限，默认 2000 ms；超出只警告，不替主办方判失败 |
| `--max-model-mb` | 权重合计上限，默认 500 MB；超出只警告 |
| `--report` | 把计时、融合与比对明细写成 JSON 报告 |

### 1.2 退出码与日志

- `0`：单 study 平均耗时未超 100 ms，且预测与 `submission.csv` 一致；
- `1`：存在不一致的 study，或单 study 平均耗时超限（一致性失败优先判定）；
- `2`：参数、文件、配置、权重、融合清单或视角元数据错误。

日志前缀：`[INFO]` 过程信息、`[WARN]` 非致命提示、`[ERROR]` 失败原因。脚本输出单行 `[CONCLUSION]`，保留规则示例要求的「连续 3 轮」「平均耗时 X ms」「未超过上限」等字段，同时补充 study 数、图片数、模型数与初始化耗时。

## 2. 计时口径

### 2.0 规则原文与群答疑的口径冲突

规则第 7 条的原文是**图片口径**：

> 推理时间以端到端单张图片耗时为统计口径……对主办方固定提供的 100 张样本按 batch=1 串行执行推理；单张耗时为「从读取图片文件到写出最终预测结果」的实际耗时……
>
> 先取每张样本 3 轮均值，再对 100 张取平均。
>
> 单张平均耗时上限为 100ms。

2026-11-03 官方答疑群对「100ms 按单张图片还是单次检查计算」的回答则是**study 口径**：

> batch=1 推理的时间，batch 以 study_id 为单位，包含图片加载、预处理、后处理，不包含预热。
>
> 最小计时单元是单个 study_id（一次检查），batch=1 是以 study_id 为单位的，一个 study 下如果包含多张，可以同时处理。

两者确实不一致。本脚本按答疑的 study 口径执行，理由有三：

1. 答疑是针对该问题的后发解释，明确回答了「单张还是单次检查」；
2. 官方样例 `pictures_100` 实际是 10 个 study / 18 张图，而不是 100 张图；
3. **study 口径严于图片口径**：每个 study 至少 1 张图，按 study 达标必然按图片达标。

因此判定始终用 study 均值（更严格），但 `[CONCLUSION]` 与报告的 `per_image_mean` 同时给出「按图片折算」的数值，便于主办方按任一口径核对。若最终按规则字面的图片口径复测，数值只会更小；反过来，按图片口径通过不等于按 study 口径通过，不能拿折算值去主张达标。

### 2.1 单次计时的区间

对每个 study：

1. `torch.cuda.synchronize()` / `torch.mps.synchronize()` 后取 `perf_counter`；
2. 读该 study 的全部图片（每张只读一次）；
3. 相同预处理签名的模型共享一次确定性 eval 预处理（letterbox / resize / normalize / ToTensor）；不同分辨率（如 320 与 384）放进同一线程池并行缩放；
4. 每个模型把该 study 的全部图片拼成一个 batch，一次前向（`predict_study_multi`）；
5. 按 `model.aggregation` 做 study 聚合（均值 / 视角加权 / 概率空间 / study 注意力）与 `sigmoid`；
6. 多模型时按提交的融合公式合成最终 10 类概率；
7. 为该 study 的每一张图写一行 `Subject_id, Study_id, Image_id, p1..p10`；
8. `flush()` 后再次同步设备，取 `perf_counter`。

多图 study 不再被判定为「无法计时」：答疑明确要求同一 study 的图片在同一流程内处理并输出该检查的最终预测，脚本已按此实现。

### 2.2 文件 IO

- 每轮只打开一次预测文件，所有 study 复用同一句柄，轮末关闭；避免逐图/逐 study 反复 open/close；
- 每个 study 写完立即 `flush()`，flush 与写盘时间都在计时区间内；
- 轮总墙钟从 `open` 前开始、`close` 后结束，覆盖打开/写入/flush/关闭的全部 IO；
- 判定用的 `billed_mean_ms = 逐 study 均值 + 轮级 IO 分摊`，其中轮级分摊 = `(Σ轮总墙钟 − Σ逐 study 计时) / (轮数 × study 数)`，不豁免文件 IO；
- 不做 `fsync`：`flush()` 已把用户态缓冲交给操作系统，逐 study `fsync` 会把磁盘同步延迟计入成绩，不符合常规推理服务口径；若主办方另有要求，应统一复测。

### 2.3 轮次与统计

外层 3 轮、内层逐 study。`summarize_timings` 先对每个 study 取 3 轮均值，再对全部 study 取平均；同时给出中位数、p95、最小、最大、总耗时与每轮总墙钟。100 ms 比较的是「单 study 平均耗时」——**不是单张图片平均耗时**。多图 study 单次耗时更高是正常的，按 study 等权平均正是赛题口径。

### 2.4 初始化（不计时，但有上限）

初始化区间只包含：

- 读取并校验 `submission.csv`；
- 递归列出 `pictures_100` / `pictures_5` 的图片路径；
- 按 `<subject_id>/<study_id>/` 分组、检查同一 study 不跨患者；
- 检查预热 study 与计时 study 不相交。

脚本打印 `init_ms`，默认自设上限 2000 ms；超出只打 `[WARN]`，因为官方未公布该上限。初始化不读任何图像像素，也不在计时循环里做惰性补算。模型加载、配置解析、权重读取单独计时并打印，规则明确不计入成绩。

### 2.5 预热

`pictures_5` 的每个 study 走与正式计时完全相同的前向/融合路径（不写文件），默认要求 5 个 study；数量不符打 `[WARN]`。若预热 study 与计时 study 有交集，直接返回 2——赛题要求两者不相交。

## 3. 多模型融合

赛题第 6 条允许多模型集成；当前正式提交 `A-FUSE-3` 是 A-001 / A-004-b / RUN-009-A / C-005-c 四模型概率等权融合。**只加载单个权重的 cost.py 会把耗时低估约一半以上，并且无法复现 submission.csv**，所以脚本支持两种声明方式：

1. `--checkpoint a.pt b.pt ...` + `--weights 1,1,1,1` + `--fusion probability`；
2. 在 `submission.csv` 同目录放 `cost_ensemble.json`，评测命令不带额外参数也能自动生效：

```json
{
  "fusion": "probability",
  "weights": [1, 1, 1, 1],
  "checkpoints": ["weights/A-001.pt", "weights/A-004-b.pt", "weights/RUN-009-A.pt", "weights/C-005-c.pt"]
}
```

清单里的相对路径按清单所在目录解析；`--checkpoint` 显式给出时覆盖清单的 checkpoints，`--weights` / `--fusion` 显式给出时覆盖清单对应字段。计时循环对同一 study 只解码一次图像，再让每个模型各自预处理与前向，最后调用 `infer.ensemble.fuse_probability_vectors`，与 `run_ensemble.py` 的 `probability` / `logit` 融合公式一致。

权重合计大小会打印并按赛题第 8 条的 500 MB 上限告警；脚本按实际加载的权重文件去重后求和。

## 4. 视角元数据

`view_weighted` 聚合需要每张图的 `ViewPosition`（AP/PA/Lateral）。计时目录只有图片，没有 CSV，因此：

- 优先用 `--views-csv`；
- 否则找 `submission.csv` 同目录的 `cost_views.csv`；
- 否则读各模型配置里的 `data.test.csv`。

`cost_views.csv` 只需三列，内容与 test.csv 一致，属于答疑允许的 CSV 元数据，不含像素：

```bash
python -c "import pandas as pd; pd.read_csv('data/test.csv', dtype=str)[['Study_id','Dicom_id','ViewPosition']].to_csv('cost_views.csv', index=False)"
```

找不到视角元数据时，脚本对 `view_weighted` 模型回退为等权均值并打 `[WARN]`，此时预测可能与 submission.csv 不一致。**若正式模型使用视角加权，提交包必须带 `cost_views.csv`**。

## 5. 与 `predict.py` 的一致性

规则要求 `cost.py` 导入并调用与 `predict.py` 相同的推理 / 加载逻辑。本实现直接复用 `src/infer/predict.py` 的公开函数：

| 用途 | 函数 |
|---|---|
| 加载权重 | `load_checkpoint_model(config, checkpoint_path, device)`（`run_infer.py` / `run_ensemble.py` 同款） |
| 构造 eval 预处理 | `build_eval_inference_transform(config)` |
| 单 study 推理（已解码图像） | `predict_study_images(model, images, transform, device, views, ...)` |
| 单 study 推理（按路径） | `predict_study(...)`，内部转交 `predict_study_multi`（单模型） |
| 多模型 study 推理 | `predict_study_multi(specs, paths, device, views, max_workers)`：解码一次、同 `transform_key` 共享预处理、多分辨率并行缩放 |
| 形状预热 | `warmup_study_shapes(...)`：合成零张量预热图片数形状，只做算子初始化 |
| 多模型融合 | `infer.ensemble.fuse_probability_vectors`，与 `run_ensemble.py` 同公式 |

`predict_loader`（批量 loader 路径）、`predict_study_multi`、`predict_study_images` 共用 `_forward_probabilities`，因此「loader 批量推理」「单 study 计时」在数值上是同一套前向、聚合与 sigmoid，区别只在 batch 组织方式。

## 6. 一致性比对

计时循环用最后一轮现场算出的 study 概率，与 `submission.csv` 比对：

- 提交文件输出全部 10 类时：逐类比较概率绝对差，最大差 `<= --tolerance` 记为一致；
- 提交文件按阈值过滤过（某 study 少于 10 行）：先用 `--threshold` 做同样的过滤，再比较类别集合与交集概率；
- 提交文件缺少某个 study：记为不一致；
- `Subject_id` 与目录不一致：记为不一致；
- 计时目录是测试集的子集，submission 里有、计时目录没覆盖的 study 不参与比对。

输出文件 `--prediction-out` 的列是 `Subject_id, Study_id, Image_id, p1..p10`，一张图一行；同一 study 的多张图写同一组最终概率。一致性比对以 `Study_id` 为主键（本赛题 submission 是 study 级，没有 image_id 列），`Image_id` 仅用于核对计时覆盖的图片集合与联合主键唯一性。

## 7. 精度声明

默认 `fp32`。`--precision fp16` 时脚本把每个模型参数转成半精度（`model.half()`），并在日志与 `[CONCLUSION]` 中标注。赛题允许 FP16 / INT8，但要求在提交说明中注明；fp16 与 fp32 提交文件可能有细微数值差异，比对时按 `--tolerance` 判定。

## 8. 报告字段（`--report`）

JSON 顶层键：

- `submission` / `checkpoints` / `model_count` / `fusion` / `weights` / `ensemble_manifest` / `checkpoint_mb`：本次实际加载的模型与融合设置；
- `device` / `precision` / `threshold`：执行设置；
- `views`：视角元数据来源、是否需要、是否可用与加载耗时；
- `units`：计时 study 数、图片数、预热 study 数与图片数；
- `init_ms` / `init_limit_ms` / `init_within_limit` / `setup_ms` / `model_load_ms` / `warmup_ms`：初始化、配置解析、模型加载与预热耗时（均不计入成绩）；
- `prediction_csv` / `prediction_csv_persisted`：逐图预测文件路径，以及该文件是否保留（缺省临时文件在计时结束后清理）；
- `timing_ms`：`rounds`、`studies`、`images`、`mean`、`billed_mean`、`round_overhead_per_study`、`median`、`p95`、`min`、`max`、`total`、`round_total`、`limit`、`within_limit`；
- `comparison`：`filtered_submission`、`checked_studies`、`checked_images`、`inconsistent_studies`、`inconsistent_images`、`max_abs_diff`、`tolerance`、`issues`。

## 9. 本地验证记录

- 单元测试：`tests/test_cost.py` 57 个用例，覆盖 study 分组、视角元数据、submission 解析、逐图写出、probability/logit 融合、计时算术与轮级 IO 分摊、单模型/双模型/清单自动发现、预热不相交、初始化打印、超限与不一致退出码。全量 `python -m pytest` 为 388 passed / 5 skipped。
- 官方样例数据端到端（2026-11-03，本机 Apple Silicon CPU，随机初始化 resnet18，`image.size=128`，仅验证结构，不代表赛题成绩）：
  - 单模型：10 个 study / 18 张图 + 5 个 study / 10 张图，`init_ms` 2.2 ms，配置解析与模型加载合计约 102 ms，单 study 平均 38.2–39.3 ms（两次运行），轮级 IO 分摊约 0.02 ms，预测与 submission 最大差 0；
  - 双模型概率融合（清单自动发现）：单 study 平均 56.4–56.8 ms（两次运行），最大单 study 105.8 ms，预测与融合 submission 最大差 0。
  - 多图 study 的单次耗时明显高于单图 study（p95 65–67 ms vs 中位数约 30 ms），说明按图片平均会低估真实 study 级延迟。
- 正式成绩须在赛题指定 GPU、加载最终融合权重与视角元数据后实测；本机数字不能替代主办方复测。

### 9.1 V100 实测（2026-11-03，官方样例 10 study / 18 图 + 5 study / 10 图）

环境：Tesla V100-SXM2-32G，Xeon Platinum 8255C @2.5 GHz（96 vCPU），PyTorch 2.10.0+cu128，fp32。四组配置的现场预测与各自 submission 完全对齐（最大差 0 或 2.9e-07），说明计时链路与提交口径一致。

当天还修了 `Letterbox` 的一个实现 bug：文档写的是「灰度图先 resize 再转 RGB」，代码却先转 RGB 再 resize，3.6MP 胸片上多算了两遍通道。修复逐像素等价（`tests/test_dataset.py` 有对照），但端到端收益远大于微基准。修复前后：

| 模型组合 | 修复前 | 修复后 | 中位数 | p95 | 最大 | 判定（修复后） |
|---|---:|---:|---:|---:|---:|---|
| 1（A-001，ConvNeXt-T @320） | 129.1 ms | **83.9 ms** | 64.5 | 144.7 | 171.8 | 通过 |
| 2（A-001 + C-005-c，均 @320） | 203.0 ms | **119.9 ms** | 94.6 | 203.2 | 224.6 | 超限 |
| 2（A-001 + RUN-009-A，@320+384） | 217.1 ms | 未复测（预计约 130） | — | — | — | 超限 |
| 4（A-FUSE-3 正式提交） | 380.4 ms | **205.8 ms** | 166.7 | 335.9 | 377.9 | 超限 |

拆解与结论：

- **裸前向很快**：ConvNeXt-T 单图 6 ms、3 图 8 ms；EfficientNetV2-S@384 约 10 ms。瓶颈在 CPU 预处理：修复后单图 `Letterbox` 约 8 ms + `ToTensor` 约 3 ms；4 个模型仍对同一张图各做一次，等于把预处理重复 4 次。
- **每增加一个 320 模型，单 study 约增加 36 ms**（1.8 图 × 一次预处理 + 一次前向）；双模型 84→120 ms，四模型 206 ms。384 模型更贵。
- 本样例平均 1.8 图/study，完整 test 集为 1.57 图/study；按图片数折算，单模型约 73 ms（通过），四模型约 180 ms（仍超限约 1.8 倍）。
- **按图片口径折算**（总耗时 / 总图片次数）：单模型 **46.5 ms/图**、双模型 **66.5 ms/图**、四模型 **114.3 ms/图**。如果主办方最终按规则字面的「单张图片」判定，双模型会从「超限」翻为「通过」；但 study 口径更严格，脚本仍按 study 判定，折算值只作对照，不能拿来主张达标。
- 这是**优化前**的基线：单模型 83.9 ms 已通过，四模型 205.8 ms 超限。后续在 100 study 样本上接上共享/并行预处理与形状预热，四模型降到 **96.7 ms**（见下），所以这条不再作为最终结论。
**三模型（A-FUSE-2：A-001 + C-005-c + RUN-009-A，1:1:1）fp16 对照**：

| 精度 | 单 study | 折算单图 | 中位数 | p95 | 最大 | 一致性 |
|---|---:|---:|---:|---:|---:|---|
| fp32 | 177.4 ms | 98.4 ms/图 | 138.7 | 293.1 | 319.6 | 0 个不一致 |
| fp16 | 170.0 ms | 94.4 ms/图 | 132.1 | 279.9 | 324.2 | 2 个 study 不一致，最大差 1.62e-3 |

fp16 只快约 4%——前向不是瓶颈，预处理才是。study 口径仍超限；图片口径 94.4 ms/图勉强通过。若采用 fp16，submission 也必须用 fp16 生成（或把 `--tolerance` 放到 2e-3），并在提交说明中注明精度。

**100 study 真实分布复核（2026-11-03，按 test.csv 随机抽 100 study / 162 图，分布 52/36/11/1）**

把「同分辨率共享预处理 + study 内并行解码/缩放 + 形状预热」接上后：

| 模型组合 | 精度 | 线程 | 单 study | 折算单图 | 中位 | p95 | 最大 | 一致性 | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1（A-001） | fp32 | 6 | 58.1 ms | 35.9 | 55.9 | 75.2 | 86.0 | 提交不覆盖该样本，仅计时 | 通过 |
| 3（A-FUSE-2） | fp32 | 6 | 93.6 ms | 57.8 | 91.5 | 113.0 | 126.6 | 0 | 通过 |
| 3（A-FUSE-2） | fp16 | 6 | 94.7 ms | 58.5 | 93.8 | 110.8 | 150.7 | 29（vs fp32 提交） | 计时通过，需 fp16 重生成提交 |
| 4（A-FUSE-3） | fp32 | 6 | **96.7 / 100.5 ms**（两次） | 59.7–62.0 | 94.9 / 99.0 | 120.9 / 123.3 | 133.3 / 130.4 | 0 | 临界：一次通过、一次超限 0.5 ms |

结论：

- 优化前后，四模型从 10 study 的 205.8 ms 降到 100 study 的 **96.7–100.5 ms**；主要收益来自共享/并行预处理与形状预热，不是样本变简单。
- 三模型 **93.6 ms**，比四模型快 3.1 ms、余量 6.4 ms；valid 上 A-FUSE-2（0.8148 / 0.4420）与 A-FUSE-3（0.8148 / 0.4417）等价，**若保留融合，优先三模型 A-FUSE-2**。
- 四模型两次运行 **96.7 / 100.5 ms**，run-to-run 波动约 4 ms 就足以翻车；不作为稳健方案，若坚持使用必须再留余量。
- fp16 反而略慢（94.7 vs 93.6）且与 fp32 提交有 29 个 study 不一致，不划算；若用 fp16，必须用 fp16 重生成提交并声明精度。
- 单模型 **58.1 ms**，余量最大；若以稳为主，单模型仍是最安全的选择。

- 报告留档：`outputs/cost/20261103_v100/report_*.json`（`outputs/` 不入库），远端原始产物在 `cost_run/`。

## 10. 提交清单（附录 A）

提交包根目录至少包含：

```text
cost.py                    # 由 src/cost.py 复制
src/                       # common、data、model、infer 等源码及依赖
config.yaml                # 或 configs/default.yaml
best.pt                    # 单模型；多模型时放各自权重
submission.csv             # 最终提交文件
cost_ensemble.json         # 多模型融合时必需（权重、融合方式）
cost_views.csv             # 模型使用 view_weighted 时必需（三列元数据）
```

评测命令仍为 `python cost.py <submission.csv> <pictures_100> <pictures_5>`。模型与必要配套文件解压后合计 ≤ 500 MB（第 8 条）；`pictures_100` / `pictures_5` 由主办方提供，不随包提交。

## 11. 已知限制

- 计时受机器负载、CPU 频率、温度与 GPU 共享影响；3 轮已能压住大部分抖动，但本机数字不能替代赛题服务器实测。
- 单 study 内的读图/缩放用线程池并行（默认 6 线程，`--preprocess-workers` 可调）；在 CPU 核数较少的机器上应适当调低，避免线程争抢。
- 未使用 CUDA Graph / `torch.compile` / 算子预热；答疑允许在正式计时前用合成张量完成这些初始化，本脚本未启用，因此测的是 eager 模式。
- `flush()` 不落 `fsync`；如果主办方要求把页缓存刷盘也算进单 study 时间，需要统一改为逐 study `fsync` 并复测。
- 视角加权模型若缺少 `cost_views.csv`，会回退等权均值并导致一致性失败；这是有意暴露而不是静默通过。
- `pictures_100` 不是 100 个 study 时脚本只打 `[WARN]` 并继续，方便用官方示例数据调试。

## 附录 B 官方答疑原文要点（2026-11-03，依据群聊截图）

> 以下按主题归并四张答疑截图的可辨认文字；截图在边界处截断的，标注「原文截断」，不补写推断内容。

### B.1 计时单元与数据

- 最小计时单元是单个 `study_id`（一次检查）；`batch=1` 以 `study_id` 为单位。
- 一个 study 下如果包含多张图，可以同时处理。
- 单个 `study_id` 的全部图片在同一流程内处理，并输出该检查的最终预测；每个目录下含该检查全部图片，结构与测试集一致。
- 同 `study_id` 不会重复出现。
- 官方耗时测试数据示例（2026-11-03 记录）：
  - 链接：`https://pan.baidu.com/s/1vlIoMwAge1DVEnM5t6HxlA?pwd=5j7a`
  - 提取码：`5j7`

### B.2 延迟评测的问答

- 问：100 ms 按单张图片还是单次检查计算？是否包含读取图片、预处理、后处理与首次模型加载？batch 大小、热身和取平均/分位数的统计方式是什么？多图检查与多模型如何计时？
- 答：**batch=1 推理的时间，batch 以 `study_id` 为单位，包含图片加载、预处理、后处理，不包含预热。**

### B.3 预热、初始化与单次计时内容

- 预热图也以 Study 为单位，与正式测试的 study 不相交；允许使用指定 5 个 Study 做模型预热。
- 可放初始化（不计时）：CSV 元数据加载、Study 分组、依赖路径索引。
  - 条件：只碰元数据不碰图像像素；一次性完成，不许循环内惰性补算；打印 `init_ms` 受上限核验。
  - 禁止把解码、特征、预测藏进初始化。
- 单次计时内容：读聚合域 → 解码 → 前向 → 写出该目标图的完整一行（`study_id` + `image_id` + 10 类概率）即足够。
- 不要求包含 `submission.csv` 合并与校验。
- 按 study 写出，一致性核验以 `study_id` + `image_id` 为主键。

### B.4 IO、模型状态与预热初始化

- 正式计时开始前，可以使用合成虚拟张量完成 CUDA Graph 捕获、编译、算子初始化。
- 文件打开、`flush`、磁盘写入、关闭的全部 IO 耗时，统一计入总推理时长；不额外豁免文件 IO 开销，尽量减少频繁打开关闭文件，降低 IO 耗时。
- LN/GN 实时计算输入统计量 + BN 使用固定训练统计量 + 所有参数冻结属于合规。

### B.5 Macro-F1 口径

- Macro-F1 计算范围和主指标 AUC 保持一致，仅计算前 9 类疾病，不纳入第 10 类 No Finding。
- 「如果 No Finding 与某疾病的概率同时超过……」——原文在此截断，后续条件未记录，待补。

### B.6 与本脚本的对应关系

| 答疑要点 | 脚本落点 | 状态 |
|---|---|---|
| study 为最小计时单元、多图同批 | 第 2.1 节；`build_study_jobs` + `predict_study_images` | 已实现 |
| 预热 5 个 study 且与计时不相交 | 第 2.5 节；有交集直接返回 2 | 已实现 |
| 初始化只碰元数据、一次完成、打印 `init_ms` | 第 2.4 节；默认 2000 ms 自设上限告警 | 已实现 |
| 文件 IO 全部计入、减少开关 | 第 2.2 节；每轮一次 open、逐 study flush | 已实现 |
| 逐图写出 `study_id + image_id + 10 类概率` | 第 6 节；`COST_OUTPUT_COLUMNS` | 已实现 |
| 一致性以 `study_id + image_id` 为主键 | 本赛题 submission 是 study 级，按 `Study_id` 比对；`Image_id` 用于核对覆盖与联合主键唯一性 | 已说明 |
| 多模型如何计时 | 第 3 节；全部模型前向 + 融合均计入 | 已实现 |
| LN/GN 实时统计、BN 固定统计量、参数冻结 | `model.eval()` + `torch.inference_mode()` | 满足 |
| 允许 CUDA Graph / 编译 / 算子初始化 | 未启用，按 eager 模式计时 | 允许但未启用 |
| Macro-F1 仅前 9 类 | `cost.py` 不计算 F1，仅做耗时与提交一致性 | 不适用 |

> 附录只记录答疑原文要点，不改写规则第 7 条；后续群答疑若有补充，按日期追加，不覆盖本条。
