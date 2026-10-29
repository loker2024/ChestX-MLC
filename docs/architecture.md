# 架构与文件结构

本文件记录 ChestX-MLC 的目标代码结构、模块边界与运行方式。赛题背景、数据格式与评测口径见 [赛题说明](多目标胸部病变检测分类（ChestX-MLC）.md)，数据流示意见 [胸片多标签分类数据流](胸片多标签分类数据流.svg)。

> P1–P8 的代码与配置已按本文件落地；P7 对照实验在训练环境执行。实际改动与进度以 `CHANGELOG.md` 为准，调优结论见 [experiments.md](experiments.md)，执行命令见 [train_plan01.md](train_plan01.md)。

## 1. 设计约束

- 样本单位：study（一次检查）。训练集中一个 study 含 1–7 张图，同一 study 的图片共享同一标签。
- 标注格式：CSV 一行一张图，`Predict_class` 为分号分隔的类别 ID（如 `7;6;9`）；类别 ID 取 1–10，对应数组下标 0–9；类别 10 在数据中不与其他类别共现。
- ID 规则：`Subject_id`、`Study_id`、`Dicom_id` 均带前导零，读取与写出全程使用字符串。
- 数据划分：官方 train/valid 已按患者划分且无重叠，直接沿用；valid 只用于选型与早停。
- 部署约束：模型总分 ≤500MB、单图端到端 ≤100ms（T4 16GB），默认 ResNet-18 + 320px 输入。
- 聚合口径：逐图 forward 得到 logits，默认同一 study 内按图片取均值；可选正侧位分组均值加权，未知视角或单视角回退全图均值；聚合后 sigmoid，训练损失与推理提交使用同一口径。

## 2. 目录总览

`src/` 按 `common`、`data`、`model`、`train`、`infer` 分层；运行入口和耗时自检入口 `cost.py` 均位于 `src/`。`configs/` 存放配置，`tests/` 存放合成数据测试，`docs/` 存放说明。`data/` 和 `outputs/` 分别存放本地数据和运行产物，不入库。

## 3. 模块职责

### 3.1 common：跨模块共用

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `constants.py` | 类别数、图像后缀、ImageNet 均值方差等常量 | `NUM_CLASSES = 10`、`CLASS_IDS`、`DEFAULT_IMAGE_SIZE = 320` |
| `config.py` | YAML 读取（支持 `_base` 继承）、dataclass 校验与默认值 | `load_config(path) -> Config` |
| `utils.py` | 随机种子、设备选择、运行目录与日志 | `seed_everything(seed)`、`resolve_device("auto")`、`make_run_dir()` |
| `metrics.py` | 评估指标；训练监控、推理评估与阈值筛选共用 | `macro_auc(y_true, y_score)`、`macro_f1(y_true, y_score, threshold)`、`per_class_report(...)`、`parse_thresholds(text)`、`macro_f1_curve(y_true, y_score, thresholds)` |
| `checkpoint.py` | 权重的保存与加载，记录配置、种子与指标 | `save_checkpoint(...)`、`load_checkpoint(path, map_location)` |

### 3.2 data：数据底座

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `labels.py` | 标签字符串解析为 multi-hot；非法 token、重复类别报错 | `parse_label_string(value: str) -> np.ndarray  # float32[10]` |
| `index.py` | 递归索引图片，按 study 聚合 CSV 行并校验标签、Dicom_id 唯一性与患者一致性 | `index_images(root) -> dict[str, Path]`、`read_split(csv_path, images_root, labeled) -> list[StudyRecord]` |
| `transforms.py` | letterbox 缩放、ImageNet 归一化与训练增强（默认不做水平翻转） | `build_eval_transform(size)`、`build_train_transform(size, cfg)` |
| `dataset.py` | study 级数据集与变长批拼接 | `StudyDataset`、`collate_studies(batch)`、`build_loader(cfg, split, train, pin_memory=False)` |

`collate_studies` 输出：

```python
{
    "images": torch.Tensor,   # [N, 3, H, W]，N 为本批所有图片数
    "offsets": torch.Tensor,  # [B + 1]，第 i 个 study 的图片区间
    "labels": torch.Tensor,   # [B, 10]，test 时缺省
    "study_ids": list[str],   # 长度 B
}
```

### 3.3 model：网络结构

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `network.py` | ResNet-18 骨干 + 单层分类头；study 级 logits 聚合 | `build_model(cfg) -> nn.Module`、`aggregate_logits(logits, offsets, views=None, frontal_weight=0.7) -> Tensor[B, 10]` |
| `losses.py` | BCEWithLogitsLoss；可选按训练集频率计算 `pos_weight`，并显式指定设备 | `build_loss(cfg, train_labels=None, device=None)` |

### 3.4 train：训练逻辑

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `engine.py` | 单 epoch 训练与验证、AMP、梯度裁剪、学习率调度、早停、逐 epoch 记录、断点续训 | `fit(cfg, model, train_loader, val_loader, device, *, progress, optimizer_state, scheduler_state, scaler_state, on_epoch_end) -> dict`（`history / best_epoch / best_macro_auc / best_metrics / best_state / progress`）；`TrainProgress` 存进度与最优快照，`TrainRuntime` 暴露优化器/调度器/AMP 供回调落盘 |

### 3.5 infer：推理与提交

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `predict.py` | 加载 checkpoint 对 loader / 单图 / 单 study 推理，聚合为 study 级概率；`predict_loader`、`predict_image`、`predict_study` 共用内部前向 | `load_checkpoint_model(config, path, device)`、`build_eval_inference_transform(config)`、`predict_loader(model, loader, device, config=None) -> pd.DataFrame`（`Study_id, p1..p10`）、`predict_image(model, image_path, transform, device) -> np.ndarray`（单图，计时用）、`predict_study(model, image_paths, transform, device) -> np.ndarray`（study 聚合，一致性校验用） |
| `submission.py` | 生成提交表、格式与取值校验、UTF-8 写出；阈值必须在 `[0, 1]` | `build_submission_frame(predictions, study_subjects, threshold=None)`、`validate_submission(path, test_csv, require_all_classes=True)`、`write_submission(df, path)` |
| `profiling.py` | 耗时评测的可测口径：图片发现与身份解析、提交文件读取校验、阈值选择、study 预测比对、计时统计与结论格式化；不含 `torch` | `COST_LIMIT_MS` / `DEFAULT_ROUNDS` / `DEFAULT_WARMUP_IMAGES` / `DEFAULT_TOLERANCE`、`list_images`、`parse_image_identity`、`group_images_by_study`、`read_submission`、`select_prediction_rows`、`compare_study_predictions`、`summarize_timings`、`format_conclusion` |

### 3.6 入口脚本

| 文件 | 职责 |
|---|---|
| `check_data.py` | 数据体检：CSV 字段、ID 前导零、图片齐全、跨 split 无重叠、标签分布；`--full` 检查全量数据 |
| `run_train.py` | 训练入口：读配置、构建 loader 与模型、调用 `train.engine.fit`；每轮结束写 `last.pt`，`--resume <run 目录或 last.pt>` 可从断点接着训练 |
| `run_eval.py` | 验证入口：加载 checkpoint，在 valid（或 `--split train`）上输出 macro-AUC/F1 与逐类明细 |
| `run_infer.py` | 推理入口：test 推理并生成 `submission.csv`；默认输出全部 10 类，`--threshold`（须在 `[0, 1]`）可按概率过滤 |
| `run_compare.py` | 对比入口：按各 run 存档的配置重算 valid 指标，写出 `summary.csv` / `per_class.csv` / `summary.md`；`--thresholds` 一次算多个阈值 |
| `src/cost.py` | 耗时自检与提交一致性检查；多图 study 的最终耗时须等待官方明确口径。调用方式、索引降级行为和限制见 [cost.md](cost.md) |

入口脚本只做参数解析与流程编排，不包含业务逻辑。

## 4. 导入与运行约定

- `src/` 是导入根，本身不放 `__init__.py`；`common/`、`data/`、`model/`、`train/`、`infer/` 各放一个空的 `__init__.py` 声明常规包。
- 包内使用顶层绝对导入，例如 `from data.index import read_split`，不使用跨包的相对导入。
- `python src/run_train.py ...` 会把 `src/` 置于 `sys.path[0]`，无需安装包；测试由 `tests/conftest.py` 把 `src/` 加入 `sys.path`。
- `__init__.py` 只放模块说明，不放业务逻辑或重型导入，避免 `import data` 就加载 torch/pandas。
- 仓库根目录存在同名的 `data/` 数据目录，`src/data/__init__.py` 可避免命名空间包把两者合并。

运行命令和续训说明见 [使用手册](manual.md)；配置以 `configs/default.yaml` 及派生配置为准，测试从仓库根目录运行 `python -m pytest`，覆盖范围见 `tests/`。

## 5. 数据流与产物

1. `data.index.read_split` 读取 CSV 并递归索引图片，返回 `list[StudyRecord]`。
2. `data.dataset.StudyDataset` 按 study 取图，`collate_studies` 拼成 `images / offsets / labels`。
3. `model.network` 逐图输出 logits，`aggregate_logits` 按 `offsets` 和可选视角信息聚合每个 study 的 logits。
4. `train.engine` 以 `BCEWithLogitsLoss` 训练，按 valid macro-AUC 保存最优权重。
5. `infer.predict` 加载 checkpoint，输出 `Study_id + p1..p10`。
6. `infer.submission` 生成并校验 `submission.csv`（1,414 个 study × 10 类）。
7. `src/cost.py` 加载同一份 checkpoint 与配置，按单图串行计时，并在完整测试索引可用时补齐 study 检查提交一致性；多图 study 无法据此判定最终耗时，详见 [cost.md](cost.md)。

每次训练生成一个 `outputs/<timestamp>_<tag>/` 目录，包含 `best.pt`、`metrics.csv`、`config.yaml`、`train.log` 与断点文件 `last.pt`。`last.pt` 每轮结束写入一次（模型、优化器、学习率调度器、AMP scaler、已完成轮次、最优快照与历史记录），训练被人为中断时最多丢当前这一轮的进度，`--resume` 会复用同一目录并追加写 `train.log`。`run_compare.py` 在 `outputs/compare/` 下写 `summary.csv`、`per_class.csv` 与 `summary.md`。`data/` 与 `outputs/` 均在 `.gitignore` 中，不入库。

## 6. 待确认与已定事项

1. 全量图片根目录与训练设备（本地 GPU 或赛题 T4 镜像）。
2. 提交策略（P7 已冻结）：默认输出全部 10 类概率（保留排序信息，利于 macro-AUC）；赛题示例默认以 0.5 为阈值，需要严格对齐示例时用 `run_infer.py --threshold 0.5`。阈值必须在 `[0, 1]`，过滤后每个 study 至少保留一行。最终配置为 `configs/default.yaml`，实验配置只通过 `_base` 继承它，见 [experiments.md](experiments.md)。
3. 离线服务器能否携带 ImageNet 预训练权重；不能时回退随机初始化。
4. 类别 10 是否为“无异常”；当前按普通类别处理。

## 7. 变更约定

- 新增、修改或删除任何文件，都必须同步记录到 `CHANGELOG.md`（Asia/Shanghai，精确到分钟）。
- 仅使用主办方允许的数据；禁止提交数据集、权重、密钥与训练产物。
- 模型或训练策略调整需记录随机种子、数据划分、设备、Macro-AUC 与 Macro-F1。
