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
- `StudyDataset`、`collate_studies(batch)`、`build_loader(cfg, split, train)`。

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

- `fit(cfg, model, train_loader, val_loader, device) -> dict`：单 epoch 训练/验证、AMP、梯度裁剪、warmup、早停、逐 epoch 记录。
- 训练入口：读配置、构建 loader 与模型、按 valid macro-AUC 保存 `best.pt`。

验收：

1. 冒烟：临时配置指向 `data/example`，`epochs=1`、`batch_studies=2`，确认 loss 有限、参数有更新、checkpoint 可保存并重载。
2. 全量：`python src/run_train.py --config configs/default.yaml --seed 42`，记录设备、耗时与每 epoch 指标。

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

- `configs/exp_*.yaml`（可选）
- 无新增核心模块

工作项：

- 阈值与 `pos_weight` 对比；必要时消融骨干或输入尺寸。
- 记录随机种子、数据划分、设备、macro-AUC 与 macro-F1，形成对比表。
- 冻结最终提交策略与配置。

验收：对比表可复现；最终 `submission.csv` 通过 `validate_submission`。

## 测试映射

| 阶段 | 测试文件 | 覆盖点 |
|---|---|---|
| P1 | `tests/conftest.py` | `src/` 入 `sys.path`，合成 CSV 与小图 fixture |
| P2 | `test_labels.py` / `test_index.py` / `test_dataset.py` | 标签解析、前导零、多图聚合、缺图、变长 collate |
| P4 | `test_metrics.py` / `test_model.py` | sklearn 对齐、退化类剔除、F1 阈值、参数规模与 study 聚合 |
| P5 | `test_engine.py` | 单 epoch 参数更新与 loss 有限、warmup 学习率、早停、评估指标 |
| P6 | `test_submission.py` / `test_pipeline.py` | 列名与顺序、study 全覆盖、概率范围、端到端闭环 |

## 退出标准

- `python -m pytest` 全部通过。
- `check_data.py --full` 全部检查通过。
- valid 上记录至少一组可复现的 macro-AUC / macro-F1。
- 提交文件通过 `validate_submission`，格式与赛题要求一致。

## 备注

本文件是第 0 版实现计划；结构细节以 [architecture.md](architecture.md) 为准，实际改动与进度以 `CHANGELOG.md` 为准。
