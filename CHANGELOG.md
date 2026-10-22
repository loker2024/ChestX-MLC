# 更新日志（Changelog）

本文件记录 ChestX-MLC（多目标胸部病变检测分类）项目的全部改动。

约定：

- 每条改动标注到分钟，时区统一为 Asia/Shanghai，格式 `HH:MM`，日期作为分组标题。
- 排序规则：跨天时最新一天在最上；同一天内由早到晚，自上而下。
- 每条改动标注类型：新增 / 变更 / 修复 / 移除。

## 2026-10-22

- `13:20`（新增）`docs/spec.md`：**（该文件已于 `14:15` 删除，见本节「移除」条目）** 新增技术规格说明书，固化任务形式化（study 为样本单位、同 study 多图共享标签、logits 均值聚合）、数据索引与标注解析规则（ID 一律按字符串读入以保留前导零）、患者级划分与无泄漏校验、预处理与增强白名单（默认禁用水平翻转）、模型与损失选型约束（由 ≤500MB 与单图 ≤100ms 反推参数量与 FLOPs 上限）、训练超参与复现要求、推理与 `submission.csv` 逐字段规格及 8 项覆盖率校验、macro-AUC/F1 口径（全正或全负类别剔除而非填零）、模块接口、配置 Schema、测试清单、验收标准与 8 项风险待确认。
- `13:22`（变更）`AGENTS.md`：在项目结构一节补充 `docs/spec.md` 技术规格说明书，并说明实现前需同时阅读赛题文档与规格说明书。
- `13:36`（变更）`docs/spec.md`：**（该文件已于 `14:15` 删除）** 骨干网络定为 ResNet-18（timm `resnet18`、ImageNet-1k 预训练、11.7M 参数、512 维特征），删除原候选表；分类头改为单层 `Linear(512, 10)`，两层版本降为可选；超参随骨干调整（backbone `lr=3e-4`、weight_decay `0.01`、batch 64）；配置默认 `backbone: resnet18`；里程碑 M1/M2 与风险 Q5 处置同步改为 ResNet-18 口径。
- `13:45`（新增）`docs/architecture.md`：新增架构与文件结构说明，记录设计约束、目录树、common/data/model/train/infer 各模块职责与关键接口、导入与运行约定、YAML 配置骨架、测试布局、数据流与产物、4 项待确认问题及变更约定。
- `13:49`（变更）`docs/spec.md`：**（该文件已于 `14:15` 删除）** 精简为 119 行，只保留三部分——源码/命令行/配置与测试的文件清单（含功能与关键接口）、实现顺序七阶段（骨架→数据底座→划分管道→模型→训练→推理提交→调优，每阶段带验收命令）、以及 3 项待确认默认值；删除原 15 节中的任务形式化推导、预处理与增强细则、提交逐字段校验清单、评估口径、风险表与里程碑表。
- `13:52`（新增）`docs/plan0.md`：保存实现顺序与验收方式，按 P1 骨架与配置 → P2 数据底座 → P3 数据体检 → P4 模型与指标 → P5 训练 → P6 推理与提交 → P7 调优与冻结 推进，含每阶段涉及文件、产出、验收命令、测试映射与退出标准。
- `14:00`（新增）`configs/default.yaml`：新增默认配置，固化 train/valid/test 的 CSV 与图片路径、320px letterbox 与 ImageNet 归一化、ResNet-18 骨干、训练超参（25 epoch、batch_studies=16、lr=3e-4、warmup=2、AMP、梯度裁剪 5.0、早停 5）、评估阈值 0.5、输出目录、seed 42 与 device auto。
- `14:00`（新增）`src/common/__init__.py`：声明 common 包，只保留模块说明，不导入 torch / PyYAML 等重型依赖。
- `14:00`（新增）`src/common/constants.py`：定义 `NUM_CLASSES=10`、`CLASS_IDS`（1~10）、`DEFAULT_IMAGE_SIZE=320`、图片后缀、ImageNet 均值方差与 CSV 编码。
- `14:00`（新增）`src/common/config.py`：新增 `load_config`，把 YAML 转成 `Config` 及各节 dataclass；校验未知字段、字段类型与取值范围（epochs/lr/warmup/threshold 等），并提供 `Config.to_dict()`。
- `14:00`（新增）`src/common/utils.py`：新增 `seed_everything`（Python/NumPy/PyTorch 种子与 cuDNN 确定性）、`resolve_device("auto")`（CUDA→MPS→CPU，显式不可用设备报错）与 `make_run_dir`（`outputs/<时间戳>_<tag>/`）。
- `14:00`（新增）`src/common/checkpoint.py`：新增 `save_checkpoint` / `load_checkpoint`，权重之外记录优化器、epoch、指标、配置与 seed。
- `14:00`（移除）`src/.gitkeep`、`configs/.gitkeep`：目录已有实际文件，删除占位文件。
- `14:00`（新增）`tests/conftest.py`：把 `src/` 插入 `sys.path`，让测试与源码统一使用顶层导入（如 `from data.index import read_split`），无需安装项目。
- `14:00`（移除）`tests/.gitkeep`：目录已有实际文件，删除占位文件。
- `14:15`（移除）`docs/spec.md`：删除该技术规格说明书，规格内容改由 `docs/architecture.md` 与 `docs/plan0.md` 承载；`docs/` 下不再保留 spec 文件，本节 `13:20`／`13:36`／`13:49` 三条相关记录已同步标注“该文件已删除”。该文件此前未纳入版本控制，删除仅发生在工作区。
- `14:30`（新增）`src/data/__init__.py`：声明 data 包，避免 `src/data` 与仓库根的 `data/` 数据目录构成隐式命名空间包；只保留模块说明。
- `14:30`（新增）`src/data/labels.py`：新增 `parse_label_string`，把 `Predict_class` 的分号串解析为 float32[10] 多热向量；空值视为「无类别」，空片段、非整数、越界与重复类别直接报错并带上原始值。
- `14:30`（新增）`src/data/index.py`：新增 `index_images`（递归索引图片根，键为文件名即 `Dicom_id`，重复 stem 报错）与 `read_split`（ID 按字符串读取保留前导零，同 `Study_id` 的多行聚合为一条 `StudyRecord`，校验同 study 标签一致，一次性报出全部缺失图片），以及 frozen dataclass `StudyRecord`。
- `14:30`（新增）`src/data/transforms.py`：新增 `Letterbox`（等比缩放到最长边后居中补黑边，不把胸片拉变形）与 `build_eval_transform` / `build_train_transform`（随机仿射、亮度对比度抖动、可开关水平翻转，ImageNet 归一化）。
- `14:30`（新增）`src/data/dataset.py`：新增 `StudyDataset`（一次检查的全部图片 + 标签为一个样本）、`collate_studies`（变长图片首尾相接，输出 `images[N,3,H,W]`、`offsets[B+1]`、`labels[B,10]`、`study_ids`，test split 不含 `labels` 键）与 `build_loader`（train 用增强、打乱并按 `cfg.train.batch_studies` 取批，其余 split 确定性并按 `cfg.eval.batch_studies`）。
- `14:30`（变更）`src/common/config.py`：`ImageConfig` 新增 `hflip: bool = False`，胸片左右有方向含义，训练增强默认不水平翻转。
- `14:30`（变更）`configs/default.yaml`：`image` 节新增 `hflip: false` 并注明含义。
- `14:30`（变更）`requirements.txt`：新增测试依赖 `pytest==9.1.1`，P2 起需要运行 `python -m pytest`。
- `14:32`（变更）`.gitignore`：数据与产物目录的忽略规则由「任意层级匹配」改为「仅仓库根目录」（如 `data/` → `/data/`），否则新建的源码包 `src/data/` 会被一并忽略而无法提交。

## 2026-10-16

- `13:35`（新增）`docs/胸片多标签分类数据流.svg`：保存以 study 为样本单位的数据流图，展示逐图特征提取、同一次检查内多图特征平均聚合、study 级训练与推理流程；提交阶段筛选预测标签为 1 的类别，按 `Study_id` 和类别逐行输出类别 ID 及原始预测概率。
- `13:40`（新增）`AGENTS.md`：新增中文贡献指南，说明仓库结构、环境命令、编码与测试约定、提交及 PR 要求；明确文件增删改均须记录日志，以及仅在 `develop` 或其派生分支提交、`main` 仅允许 rebase 或 merge 的规则。
- `13:45`（新增）`src/.gitkeep`、`tests/.gitkeep`、`configs/.gitkeep`：创建源码、测试与配置目录，以空占位文件纳入版本控制。
- `13:45`（变更）`AGENTS.md`：更新目录现状，说明各目录职责及 `.gitkeep` 的用途。
- `13:47`（变更）`docs/多目标胸部病变检测分类（ChestX-MLC）.md`：压缩空行、调整提交示例表格排版，移除十类标签对照表，并清理行尾空格。
- `13:47`（修复）`CHANGELOG.md`：将数据流图文件清单修正为实际存在的 SVG，移除未保留的 PUML、PNG 及可编辑源码描述。

## 2026-10-13

- `14:00`（新增）`.gitignore`：忽略虚拟环境、数据集、模型权重与训练产物、预测结果等，避免大数据与权重入库；`.codegraph` 本机工具链符号链接不入库。
- `14:00`（新增）`docs/多目标胸部病变检测分类（ChestX-MLC）.md`：赛题说明文档，涵盖任务描述、数据集构成（训练 18513 / 验证 859 / 测试 1414 次检查）、10 个类别标签映射、数据目录结构与评估要求。
- `14:00`（新增）`requirements.txt`：依赖清单，版本对齐赛题服务器（Python 3.12 / CUDA 12.8 / torch 2.10.0 + torchvision 0.25.0），包含训练栈 `timm`、数据处理与评估 `numpy`/`pillow`/`pandas`/`scikit-learn`/`PyYAML`/`tqdm`，以及图像增强 `opencv-python-headless`/`albumentations`。
- `14:08`（变更）`requirements.txt`：补充 Windows（x64）安装说明与各平台 wheel 差异（PyPI CPU 版 vs cu128 GPU 版），并记录 macOS arm64（MPS）上的实测环境与设备选择建议。
