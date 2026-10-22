# 架构与文件结构

本文件记录 ChestX-MLC 的目标代码结构、模块边界与运行方式。赛题背景、数据格式与评测口径见 [赛题说明](多目标胸部病变检测分类（ChestX-MLC）.md)，数据流示意见 [胸片多标签分类数据流](胸片多标签分类数据流.svg)。

> 当前 `src/`、`tests/`、`configs/` 下的文件为规划结构，尚未全部落地；实际改动与进度以 `CHANGELOG.md` 为准。

## 1. 设计约束

- 样本单位：study（一次检查）。训练集中一个 study 含 1–7 张图，同一 study 的图片共享同一标签。
- 标注格式：CSV 一行一张图，`Predict_class` 为分号分隔的类别 ID（如 `7;6;9`）；类别 ID 取 1–10，对应数组下标 0–9；类别 10 在数据中不与其他类别共现。
- ID 规则：`Subject_id`、`Study_id`、`Dicom_id` 均带前导零，读取与写出全程使用字符串。
- 数据划分：官方 train/valid 已按患者划分且无重叠，直接沿用；valid 只用于选型与早停。
- 部署约束：模型总分 ≤500MB、单图端到端 ≤100ms（T4 16GB），默认 ResNet-18 + 320px 输入。
- 聚合口径：逐图 forward 得到 logits，同一 study 内按图片取均值，再 sigmoid；训练损失与推理提交使用同一口径。

## 2. 目录总览

```text
ChestX-MLC/
├── AGENTS.md
├── CHANGELOG.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── docs/
│   ├── 多目标胸部病变检测分类（ChestX-MLC）.md
│   ├── 胸片多标签分类数据流.svg
│   └── architecture.md
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── config.py
│   │   ├── utils.py
│   │   ├── metrics.py
│   │   └── checkpoint.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── labels.py
│   │   ├── index.py
│   │   ├── transforms.py
│   │   └── dataset.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── network.py
│   │   └── losses.py
│   ├── train/
│   │   ├── __init__.py
│   │   └── engine.py
│   ├── infer/
│   │   ├── __init__.py
│   │   ├── predict.py
│   │   └── submission.py
│   ├── check_data.py
│   ├── run_train.py
│   ├── run_eval.py
│   └── run_infer.py
├── tests/
│   ├── conftest.py
│   ├── test_labels.py
│   ├── test_index.py
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_model.py
│   ├── test_submission.py
│   └── test_pipeline.py
├── data/                # 不入库
│   ├── train.csv
│   ├── valid.csv
│   ├── test.csv
│   └── example/         # 本地联调用的小样本图片
├── checkpoints/         # 不入库，运行时生成
└── outputs/             # 不入库，运行时生成
```

`src/.gitkeep` 在 `src/` 有实际文件后删除。当前不引入 `pyproject.toml`；若后续需要在赛题服务器上安装，再加包配置并把 `src` 设为包根。

## 3. 模块职责

### 3.1 common：跨模块共用

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `constants.py` | 类别数、图像后缀、ImageNet 均值方差等常量 | `NUM_CLASSES = 10`、`CLASS_IDS`、`DEFAULT_IMAGE_SIZE = 320` |
| `config.py` | YAML 读取、dataclass 校验与默认值 | `load_config(path) -> Config` |
| `utils.py` | 随机种子、设备选择、运行目录与日志 | `seed_everything(seed)`、`resolve_device("auto")`、`make_run_dir()` |
| `metrics.py` | 评估指标；训练监控与推理评估共用 | `macro_auc(y_true, y_score)`、`macro_f1(y_true, y_score, threshold)`、`per_class_report(...)` |
| `checkpoint.py` | 权重的保存与加载，记录配置、种子与指标 | `save_checkpoint(...)`、`load_checkpoint(path, map_location)` |

### 3.2 data：数据底座

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `labels.py` | 标签字符串解析为 multi-hot；非法 token、重复类别报错 | `parse_label_string(value: str) -> np.ndarray  # float32[10]` |
| `index.py` | 递归索引图片，按 study 聚合 CSV 行并校验一致性 | `index_images(root) -> dict[str, Path]`、`read_split(csv_path, images_root, labeled) -> list[StudyRecord]` |
| `transforms.py` | letterbox 缩放、ImageNet 归一化与训练增强（默认不做水平翻转） | `build_eval_transform(size)`、`build_train_transform(size, cfg)` |
| `dataset.py` | study 级数据集与变长批拼接 | `StudyDataset`、`collate_studies(batch)`、`build_loader(cfg, split, train)` |

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
| `network.py` | ResNet-18 骨干 + 单层分类头；study 级 logits 聚合 | `build_model(cfg) -> nn.Module`、`aggregate_logits(logits, offsets) -> Tensor[B, 10]` |
| `losses.py` | BCEWithLogitsLoss；可选按训练集频率计算 `pos_weight` | `build_loss(cfg, train_labels=None)` |

### 3.4 train：训练逻辑

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `engine.py` | 单 epoch 训练与验证、AMP、梯度裁剪、学习率调度、早停、逐 epoch 记录 | `fit(cfg, model, train_loader, val_loader, device) -> dict` |

### 3.5 infer：推理与提交

| 文件 | 职责 | 关键接口 |
|---|---|---|
| `predict.py` | 加载 checkpoint 对 loader 推理，聚合为 study 级概率 | `predict_loader(model, loader, device) -> pd.DataFrame`（`Study_id, p1..p10`） |
| `submission.py` | 生成提交表、格式与取值校验、UTF-8 写出 | `build_submission_frame(...)`、`validate_submission(path, test_csv)`、`write_submission(df, path)` |

### 3.6 入口脚本

| 文件 | 职责 |
|---|---|
| `check_data.py` | 数据体检：CSV 字段、ID 前导零、图片齐全、跨 split 无重叠、标签分布；`--full` 检查全量数据 |
| `run_train.py` | 训练入口：读配置、构建 loader 与模型、调用 `train.engine.fit` |
| `run_eval.py` | 验证入口：加载 checkpoint，在 valid 上输出 macro-AUC/F1 与逐类明细 |
| `run_infer.py` | 推理入口：test 推理并生成 `submission.csv` |

入口脚本只做参数解析与流程编排，不包含业务逻辑。

## 4. 导入与运行约定

- `src/` 是导入根，本身不放 `__init__.py`；`common/`、`data/`、`model/`、`train/`、`infer/` 各放一个空的 `__init__.py` 声明常规包。
- 包内使用顶层绝对导入，例如 `from data.index import read_split`，不使用跨包的相对导入。
- `python src/run_train.py ...` 会把 `src/` 置于 `sys.path[0]`，无需安装包；测试由 `tests/conftest.py` 把 `src/` 加入 `sys.path`。
- `__init__.py` 只放模块说明，不放业务逻辑或重型导入，避免 `import data` 就加载 torch/pandas。
- 仓库根目录存在同名的 `data/` 数据目录，`src/data/__init__.py` 可避免命名空间包把两者合并。

常用命令：

```bash
python src/check_data.py --config configs/default.yaml --full
python src/run_train.py  --config configs/default.yaml --seed 42
python src/run_eval.py   --config configs/default.yaml --checkpoint outputs/<run>/best.pt
python src/run_infer.py  --config configs/default.yaml --checkpoint outputs/<run>/best.pt --out outputs/submission.csv
python -m pytest
```

## 5. 配置骨架

```yaml
data:
  train: {csv: data/train.csv, images: data/images/train}
  valid: {csv: data/valid.csv, images: data/images/valid}
  test:  {csv: data/test.csv,  images: data/images/test}
image: {size: 320, letterbox: true, normalize: imagenet}
model: {backbone: resnet18, pretrained: true, num_classes: 10}
train:
  epochs: 25
  batch_studies: 16
  lr: 3e-4
  weight_decay: 0.01
  pos_weight: false
  warmup_epochs: 2
  amp: true
  grad_clip: 5.0
  num_workers: 8
  early_stop_patience: 5
eval: {batch_studies: 32, threshold: 0.5}
output: {run_dir: outputs, checkpoint_dir: checkpoints}
seed: 42
device: auto
```

本地联调可把 `data.<split>.images` 指向 `data/example/train_example` 等示例目录；全量图片路径见第 8 节。

## 6. 测试布局

| 文件 | 覆盖点 |
|---|---|
| `conftest.py` | 把 `src/` 加入 `sys.path`；用 `tmp_path` 构造合成 CSV 与多图 study 的小图片 |
| `test_labels.py` | 分号解析、类别 10、重复/非法 token |
| `test_index.py` | 前导零保留、多图 study 聚合、缺图报错、跨 split 无重叠 |
| `test_dataset.py` | 张量形状与取值范围、变长 collate、test 无标签 |
| `test_metrics.py` | 与 sklearn 对齐、退化类剔除、F1 阈值 |
| `test_model.py` | 参数规模、study 级均值聚合与梯度、`pos_weight` 计算 |
| `test_submission.py` | 列名与顺序、study 全覆盖、概率范围、UTF-8 回读 |
| `test_pipeline.py` | 单步训练 loss 有限且参数更新、checkpoint 重载逐位一致、predict→submission 闭环 |

测试只使用合成小图，不读取真实影像。

## 7. 数据流与产物

1. `data.index.read_split` 读取 CSV 并递归索引图片，返回 `list[StudyRecord]`。
2. `data.dataset.StudyDataset` 按 study 取图，`collate_studies` 拼成 `images / offsets / labels`。
3. `model.network` 逐图输出 logits，`aggregate_logits` 按 `offsets` 求每个 study 的均值。
4. `train.engine` 以 `BCEWithLogitsLoss` 训练，按 valid macro-AUC 保存最优权重。
5. `infer.predict` 加载 checkpoint，输出 `Study_id + p1..p10`。
6. `infer.submission` 生成并校验 `submission.csv`（1,414 个 study × 10 类）。

每次训练生成一个 `outputs/<timestamp>_<tag>/` 目录，包含 `best.pt`、`metrics.csv`、`config.yaml` 与 `train.log`。`data/`、`checkpoints/`、`outputs/` 均在 `.gitignore` 中，不入库。

## 8. 待确认

1. 全量图片根目录与训练设备（本地 GPU 或赛题 T4 镜像）。
2. 提交策略：输出全部 10 类概率（默认，利于 AUC），或按 0.5 阈值过滤。
3. 离线服务器能否携带 ImageNet 预训练权重；不能时回退随机初始化。
4. 类别 10 是否为“无异常”；当前按普通类别处理。

## 9. 变更约定

- 新增、修改或删除任何文件，都必须同步记录到 `CHANGELOG.md`（Asia/Shanghai，精确到分钟）。
- 仅使用主办方允许的数据；禁止提交数据集、权重、密钥与训练产物。
- 模型或训练策略调整需记录随机种子、数据划分、设备、Macro-AUC 与 Macro-F1。
