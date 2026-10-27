# 实现计划（plan0）

本文件把 [architecture.md](architecture.md) 的结构落成可执行的实现顺序。阶段之间单向依赖，每个阶段都给出涉及文件、产出与验收命令，完成一个阶段再进入下一个。

## 前置确认

1. 全量图片根目录（本地或赛题服务器路径）与训练设备。
2. 提交策略：默认输出全部 10 类概率，或按 0.5 阈值过滤。
3. 离线服务器能否携带 ImageNet 预训练权重；不能时回退随机初始化。
4. 类别 10 是否为“无异常”；当前按普通类别处理。

## 阶段总览

| 阶段 | 内容 | 依赖 | 关键路径 |
|---|---|---|---|
| P1 | 骨架与配置 | — | 是 |
| P2 | 数据底座 | P1 | 是 |
| P3 | 数据体检 | P2 | 否（可与 P4 并行） |
| P4 | 模型与指标 | P2 | 是 |
| P5 | 训练 | P2、P4 | 是 |
| P6 | 推理与提交 | P5 | 是 |
| P7 | 调优与冻结 | P6 | 否 |
| P8 | 耗时自检与提交包 | P7 | 否 |

关键路径：P1 → P2 → P4 → P5 → P6 → P7。P2 是不可跳过的关键节点，它冻结了训练与推理共用的 study 级数据契约。

## P1 骨架与配置

涉及文件：

- `configs/default.yaml`
- `src/common/__init__.py`、`constants.py`、`config.py`、`utils.py`、`checkpoint.py`
- `tests/conftest.py`

产出：

- `NUM_CLASSES = 10`、`CLASS_IDS`、`DEFAULT_IMAGE_SIZE = 320`。
- `load_config(path) -> Config`，dataclass 校验与默认值。
- `seed_everything(seed)`、`resolve_device("auto")`、`make_run_dir()`。
- `save_checkpoint(...)`、`load_checkpoint(path, map_location)`。

验收：

```bash
python -c "import sys; sys.path.insert(0, 'src'); from common.config import load_config; print(load_config('configs/default.yaml'))"
```

> P1 结束时还没有测试文件，`pytest` 从 P2 开始运行。

## P2 数据底座

涉及文件：

- `src/data/__init__.py`、`labels.py`、`transforms.py`、`index.py`、`dataset.py`
- `tests/test_labels.py`、`tests/test_index.py`、`tests/test_dataset.py`

产出：

- `parse_label_string(value) -> np.ndarray[10]`：分号解析、重复/非法 token 报错。
- `index_images(root) -> dict[str, Path]`：递归索引，图片 stem 即 `Dicom_id`。
- `read_split(csv_path, images_root, labeled) -> list[StudyRecord]`：按 study 聚合、校验同 study 标签一致、缺图报错。
- `build_eval_transform(size)`、`build_train_transform(size, cfg)`：letterbox、ImageNet 归一化、默认不做水平翻转。
- `StudyDataset`、`collate_studies(batch)`、`build_loader(cfg, split, train, pin_memory=False)`。

验收：

```bash
python -m pytest -q tests/test_labels.py tests/test_index.py tests/test_dataset.py
```

另用 `data/example` 建一次索引，确认 469 张示例图全部命中、`collate_studies` 输出 `images[N,3,320,320] / offsets[B+1] / labels[B,10]`。

## P3 数据体检

涉及文件：

- `src/check_data.py`

产出：

- CSV 字段、ID 前导零、图片齐全、跨 split 无重叠、标签分布、同 study 标签一致性检查；`--full` 时覆盖全量数据。

验收：

```bash
python src/check_data.py --config configs/default.yaml --full
```

期望：train/valid/test 分别为 30,344 / 1,422 / 2,218 张图；训练集 258 种标签串；train/valid 的 Subject 与 Study 重叠均为 0；类别 10 单独出现 12,649 个 study。

## P4 模型与指标

涉及文件：

- `src/model/__init__.py`、`network.py`、`losses.py`
- `src/common/metrics.py`
- `tests/test_metrics.py`、`tests/test_model.py`

产出：

- `build_model(cfg) -> nn.Module`：ResNet-18 骨干 + `Linear(512, 10)`。
- `aggregate_logits(logits, offsets) -> Tensor[B, 10]`：按 offsets 对同一 study 的图片 logits 求均值。
- `build_loss(cfg, train_labels=None)`：`BCEWithLogitsLoss`，可选 `pos_weight`。
- `macro_auc`、`macro_f1`、`per_class_report`：全正/全负类别剔除后再平均。

验收：

```bash
python -m pytest -q tests/test_metrics.py tests/test_model.py
python -c "import sys; sys.path.insert(0, 'src'); import torch; from common.config import load_config; from model.network import build_model, aggregate_logits; cfg = load_config('configs/default.yaml'); m = build_model(cfg); x = torch.randn(3, 3, 320, 320); print(m(x).shape); print(aggregate_logits(m(x), torch.tensor([0, 2, 3])).shape); print(sum(p.numel() for p in m.parameters()))"
```

期望：参数量 11,181,642（约 11.2M，即 1000 类版 ResNet-18 换掉分类头后的数字）；`macro_auc` 与 sklearn 结果一致；离线环境拿不到预训练权重时退回随机初始化并给出 `RuntimeWarning`。

## P5 训练

涉及文件：

- `src/train/__init__.py`、`engine.py`
- `src/run_train.py`

产出：

- `fit(cfg, model, train_loader, val_loader, device, *, progress, optimizer_state, scheduler_state, scaler_state, on_epoch_end) -> dict`：单 epoch 训练/验证、AMP、梯度裁剪、warmup、早停、逐 epoch 记录，并支持从 `TrainProgress` 接续训练。
- 训练入口：读配置、构建 loader 与模型、按 valid macro-AUC 保存 `best.pt`；每轮结束写 `last.pt`，`--resume` 可从断点继续。

验收：

1. 冒烟：临时配置指向 `data/example`，`epochs=1`、`batch_studies=2`，确认 loss 有限、参数有更新、checkpoint 可保存并重载。
2. 全量：`python src/run_train.py --config configs/default.yaml --seed 42`，记录设备、耗时与每 epoch 指标。
3. 续训：训练中途 Ctrl+C 后 `python src/run_train.py --resume outputs/<run>`，确认从下一个 epoch 接着跑，`metrics.csv` 与 `train.log` 累积而不重置。

## P6 推理与提交

涉及文件：

- `src/infer/__init__.py`、`predict.py`、`submission.py`
- `src/run_eval.py`、`src/run_infer.py`
- `tests/test_submission.py`、`tests/test_pipeline.py`

产出：

- `predict_loader(model, loader, device) -> DataFrame`：`Study_id, p1..p10`。
- `build_submission_frame(...)`、`validate_submission(path, test_csv)`、`write_submission(df, path)`。
- `run_eval` 输出 valid 的 macro-AUC/F1 与逐类明细；`run_infer` 生成提交文件。

验收：

```bash
python src/run_eval.py  --config configs/default.yaml --checkpoint outputs/<run>/best.pt
python src/run_infer.py --config configs/default.yaml --checkpoint outputs/<run>/best.pt --out outputs/submission.csv
python -m pytest -q tests/test_submission.py tests/test_pipeline.py
```

期望：1,414 个 study × 10 行，列名与顺序符合赛题，概率在 `[0,1]`，UTF-8 回读一致。

## P7 调优与冻结

涉及文件：

- `configs/exp_short.yaml`、`configs/exp_pos_weight.yaml`、`configs/exp_short_pos_weight.yaml`（用 `_base` 继承默认配置）
- `src/common/config.py`：`load_config` 支持 `_base` 继承
- `src/common/metrics.py`：`parse_thresholds`、`macro_f1_curve`
- `src/run_eval.py`：`--thresholds` 一次打印阈值网格下的 macro-F1
- `src/run_compare.py`：重算多个 run 的 valid 指标并写出 `summary.csv` / `summary.md`
- `docs/experiments.md`：实验记录、对比表与冻结结论
- 无新增核心模块

工作项：

- 阈值与 `pos_weight` 对比；必要时消融骨干或输入尺寸。
- 记录随机种子、数据划分、设备、macro-AUC 与 macro-F1，形成对比表。
- 冻结最终提交策略与配置（`configs/default.yaml`）。

验收：

```bash
python src/run_compare.py --runs outputs/<run_a> outputs/<run_b> --thresholds 0.3,0.4,0.5,0.6,0.7
python src/run_infer.py --config configs/default.yaml --checkpoint outputs/<final>/best.pt --out outputs/submission.csv
```

期望（在训练环境执行）：对比表可复现；最终 `submission.csv` 通过 `validate_submission`。

## P8 耗时自检与提交包

涉及文件：

- `cost.py`（仓库根目录，赛题附录 A 要求提交包根目录含此文件）
- `src/infer/profiling.py`：耗时评测的可测口径（图片发现与身份解析、提交文件读取校验、阈值选择、study 预测比对、计时统计、`[CONCLUSION]` 格式化）
- `src/infer/predict.py`：新增 `build_eval_inference_transform`、`predict_image`、`predict_study`，与 `predict_loader` 共用内部前向
- `src/data/dataset.py`：`load_image` 公开给推理复用
- `tests/test_cost.py`、`docs/cost.md`

产出：

- `python cost.py <submission.csv> <pictures_100> <pictures_5>`：5 张预热后对 100 张图 `batch=1` 串行跑 3 轮，单张耗时先按 3 轮取均值、再对 100 张平均，与 100 ms 上限比较。
- 复用 `infer.predict` 的 `load_checkpoint_model` / `build_eval_inference_transform` / `predict_image` / `predict_study`，保证与 `run_infer.py` 同一套推理与聚合逻辑；计时后用 `predict_study` 聚合现场预测并与 `submission.csv` 比对。
- 支持 `--precision fp16`（FP16 须在提交说明中声明）、`--report` 写出 JSON 明细；退出码 0 成功、1 超限或不一致、2 参数/文件/配置错误。

验收：

```bash
python cost.py outputs/submission.csv data/cost/pictures_100 data/cost/pictures_5 --config configs/default.yaml --checkpoint outputs/<final>/best.pt --report outputs/cost_report.json
python -m pytest -q tests/test_cost.py
```

期望（在赛题指定硬件、加载最终权重后执行）：平均单张耗时 ≤100 ms，退出码 0，`[CONCLUSION]` 里与 `submission.csv` 不一致为 0。正式成绩以赛题服务器实测为准，本地数字只用于链路验证。

## 测试映射

| 阶段 | 测试文件 | 覆盖点 |
|---|---|---|
| P1 | `tests/conftest.py` | `src/` 入 `sys.path`，合成 CSV 与小图 fixture |
| P2 | `test_labels.py` / `test_index.py` / `test_dataset.py` | 标签解析、前导零、多图聚合、缺图、重复 Dicom_id、患者一致性、变长 collate |
| P4 | `test_metrics.py` / `test_model.py` | sklearn 对齐、退化类 AUC 跳过但 F1 仍计算、阈值与空输入、参数规模、study 聚合、pos_weight 设备 |
| P5 | `test_engine.py` / `test_resume.py` | 单 epoch 参数更新与 loss 有限、warmup、早停、评估指标、全 NaN AUC 报错；断点续训的进度与学习率衔接、CLI 补跑剩余 epoch |
| P6 | `test_submission.py` / `test_pipeline.py` | 列名与顺序、study 全覆盖、概率范围、阈值范围、缺列报错、端到端闭环 |
| P7 | `test_config.py` / `test_utils.py` / `test_run_train.py` / `test_run_compare.py` | 配置校验与 `_base` 继承、种子/设备/运行目录、训练 CLI 产物与日志、run 对比表 |
| P8 | `test_cost.py` | 图片发现与 `<subject_id>/<study_id>` 解析、提交文件列名/取值/重复校验、阈值选择、study 预测比对、计时统计算术与超限、`predict_image`/`predict_study` 与 `predict_loader` 数值一致、CLI 端到端与退出码 |

## 退出标准

- `python -m pytest` 全部通过。
- `check_data.py --full` 全部检查通过。
- valid 上记录至少一组可复现的 macro-AUC / macro-F1。
- 提交文件通过 `validate_submission`，格式与赛题要求一致。
- 在赛题指定硬件上运行 `cost.py`：平均单张耗时 ≤100 ms，`[CONCLUSION]` 与 `submission.csv` 不一致 0。

## 备注

本文件是第 0 版实现计划；结构细节以 [architecture.md](architecture.md) 为准，实际改动与进度以 `CHANGELOG.md` 为准。
