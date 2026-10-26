# 更新日志（Changelog）

本文件记录 ChestX-MLC（多目标胸部病变检测分类）项目的全部改动。

约定：

- 每条改动标注到分钟，时区统一为 Asia/Shanghai，格式 `HH:MM`，日期作为分组标题。
- 排序规则：跨天时最新一天在最上；同一天内由早到晚，自上而下。
- 每条改动标注类型：新增 / 变更 / 修复 / 移除。

## 2026-10-26

- `18:26`（新增）`src/infer/__init__.py`：声明 infer 包，只保留模块说明，不导入 torch / pandas 等重型依赖。
- `18:26`（新增）`src/infer/predict.py`：新增 `load_checkpoint_model`（读取 `best.pt`、强制 `pretrained=False` 以免离线下载、加载权重后切到 eval）与 `predict_loader`（`torch.inference_mode` 逐批推理，`aggregate_logits` 按 study 聚合、sigmoid 后返回 `Study_id` + `p1..p10` 宽表）。
- `18:26`（新增）`src/infer/submission.py`：新增 `SUBMISSION_COLUMNS`、`build_submission_frame`（默认每个 study 输出 10 类；给阈值时只保留过线类，全部未过线则保留 top-1 保证覆盖）、`write_submission`（UTF-8、LF、自动建父目录）与 `validate_submission`（列名与顺序、study 双向覆盖、`Subject_id` 一致、类别范围、概率有限且落在 `[0,1]`、重复行、10 类齐全）。
- `18:26`（新增）`src/run_eval.py`：验证入口，支持 `--config` / `--checkpoint` / `--split`，在带标注 split 上输出 macro-AUC、macro-F1 与逐类明细。
- `18:26`（新增）`src/run_infer.py`：推理入口，支持 `--config` / `--checkpoint` / `--out` / `--threshold`；默认输出全部 10 类概率，写完立即调用 `validate_submission` 按 test.csv 校验。
- `18:26`（变更）`configs/default.yaml`：`eval.threshold` 注释改为“计算 macro-F1 的判定阈值；提交默认输出全部 10 类概率”，避免与默认提交策略混淆。
- `18:28`（新增）`tests/test_submission.py`：覆盖全 10 类输出与列顺序、阈值过滤及 top-1 补位、UTF-8 写读闭环、允许过滤文件、前导零保留，以及列序错误、缺 study、未知 study、`Subject_id` 不匹配、类别越界、概率越界、重复行共 12 例。
- `18:28`（新增）`tests/test_pipeline.py`：端到端闭环——训练 1 epoch 确认 loss 有限且参数更新，保存并重载 checkpoint 逐位一致，test 推理列名/顺序/概率范围正确，生成并校验 submission，最后直接调用 `run_eval.main` / `run_infer.main` 冒烟。
- `18:37`（变更）`docs/architecture.md`：说明 P1–P6 已落地；补全 infer 的 `load_checkpoint_model` / `build_submission_frame` / `validate_submission` 接口与 `run_infer --threshold`，并把提交策略从待确认改为已确定（默认全部 10 类，阈值过滤可选）。
- `19:03`（修复）`src/model/losses.py`、`src/train/engine.py`：`build_loss` 增加 `device` 参数并把 `pos_weight` 放到与 logits 相同的设备，修复 P7 打开 `train.pos_weight` 后在 CUDA 上报 `Expected all tensors to be on the same device` 的问题；全正类 `pos_weight` 从 0 改为 1.0，避免该类正样本 loss 恒为 0；训练/验证空集显式报错，valid 所有类别 AUC 均为 NaN 时不再静默保存随机初始权重。
- `19:03`（变更）`src/model/network.py`、`src/common/checkpoint.py`、`src/run_train.py`：模型增加 `pretrained_loaded` 标记，checkpoint 增加 `metadata` 字段；训练日志记录预训练权重是否真正加载，并把 stderr（含 timm 回退 warning）一并写入 `train.log`；DataLoader 的 `pin_memory` 改为按设备显式传入。
- `19:03`（修复）`src/common/metrics.py`：全阴/全正类只跳过 AUC，precision/recall/F1 仍计算并参与 macro-F1，零除项按 0 处理；新增二维形状、空输入与 threshold 范围校验。
- `19:03`（修复）`src/data/index.py`：`read_split` 拒绝重复 `Dicom_id`，并拒绝同一 `Study_id` 对应多个 `Subject_id`，避免重复计图或患者映射被静默覆盖。
- `19:03`（修复）`src/infer/submission.py`、`src/run_infer.py`：`--threshold` 与 `build_submission_frame` 统一限制为 `[0,1]` 的有限数；未知 study 与 test.csv 缺 `Study_id`/`Subject_id` 时给出明确错误；推理 loader 按设备传入 `pin_memory`。
- `19:14`（修复）`src/infer/submission.py`、`tests/test_submission.py`：test.csv 中同一 study 对应多个 Subject_id 时显式报错，避免患者映射被静默覆盖。
- `19:14`（变更）`tests/test_pipeline.py`：端到端覆盖 `run_infer --threshold` 过滤分支。
- `19:20`（变更）`tests/test_engine.py`：新增 CUDA 上 `fit` + `pos_weight` 的回归测试，确保训练循环而不只是 loss 构造能跑通设备路径。
- `19:03`（修复）`src/infer/predict.py`：空 DataLoader 显式报错，避免 `torch.cat([])` 抛出难以定位的底层异常。
- `19:03`（变更）`src/common/config.py`、`configs/default.yaml`：移除未使用的 `output.checkpoint_dir`；明确 `eval.threshold` 只用于 macro-F1，不影响训练、AUC 或默认全 10 类提交。
- `19:03`（变更）`src/common/utils.py`、`src/data/dataset.py`、`src/data/transforms.py`、`src/run_eval.py`：`make_run_dir` 清洗 tag，`resolve_device` 校验 `cuda:N` 的卡号；`build_loader` 的 `pin_memory` 默认关闭、由入口显式开启；`RandomAffine` 使用双线性插值；评估入口按设备开启 `pin_memory`。
- `19:03`（变更）`tests/conftest.py`、`tests/test_engine.py`、`tests/test_index.py`、`tests/test_metrics.py`、`tests/test_model.py`、`tests/test_submission.py`，新增 `tests/test_config.py`、`tests/test_utils.py`、`tests/test_run_train.py`：覆盖 pos_weight CUDA 设备、全正类权重、全 NaN AUC、重复 Dicom_id/患者冲突、F1 退化类口径、阈值/缺列/未知 study、配置与工具函数、训练 CLI 产物与日志；全套 `103 passed, 1 skipped`。
- `19:03`（变更）`AGENTS.md`、`docs/architecture.md`、`docs/plan0.md`：更新 P1–P6 已落地状态、`src/`/`tests/`/`configs/` 的实际结构、常用入口、pytest 命令与 `checkpoint_dir` 移除；同步 `hflip`、`build_loader(pin_memory)`、阈值范围、退化类 F1 口径与新增测试映射。

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
- `14:35`（变更）`tests/conftest.py`：在原有的 `sys.path` 注入之外新增合成数据 fixture——`make_image`（在 `tmp_path` 下按目录结构生成小图）、`write_csv`（UTF-8 写出完整表头，默认含 `Predict_class`）、`labeled_split`（2 个 study / 3 张图，含同一 study 多图）、`unlabeled_split`（去掉标签列，模拟 test split）与 `write_config`（YAML 路径用 JSON 字符串包裹，`image.size=32`、`num_workers=0`）；数据全在临时目录生成，不碰真实影像。
- `14:35`（新增）`tests/test_labels.py`：覆盖单类/多类、类别 10 落在下标 9、空值与空片段、重复、越界、非整数 token 共 15 例。
- `14:35`（新增）`tests/test_index.py`：覆盖图片索引（大小写后缀、重复 stem、目录缺失）、同 study 多行聚合（图片顺序与前导零）、缺图报错、同 study 标签不一致报错、无标签列读取与缺列报错共 9 例。
- `14:35`（新增）`tests/test_dataset.py`：覆盖 letterbox 等比补边与黑边位置、评估预处理确定性、张量形状、变长 `collate_studies` 的 `images`/`offsets`/`labels` 与 test split 无标签分支、`build_loader` 的三种 split 共 9 例。
- `14:42`（新增）`src/check_data.py`：新增训练前数据体检脚本，检查 CSV 必需列、ID 空值与固定宽度前导零格式、同 split 内 Dicom_id 唯一性、图片齐全、同 study 标签与患者一致性、三份划分两两不重叠，并打印每类图像/检查计数与类别 10 单独出现次数；默认每份 CSV 只读前 1000 行，`--full` 读全量并逐行核对图片，任一检查失败以退出码 1 结束。
- `14:42`（新增）`tests/test_check_data.py`：覆盖 ID 按字符串读取、缺标签列报错、截断编号识别、缺图计数、同 study 标签顺序无关与冲突检测、跨 split 患者重叠检测共 6 例。
- `14:53`（新增）`src/common/metrics.py`：新增 `per_class_report`（逐类输出支持数、正样本比例、AUC、precision/recall/F1）、`macro_auc` 与 `macro_f1`；全正或全负的退化类别没有指标定义，记 NaN 并从平均里剔除（而不是填 0 拉低总分），标签与概率形状不一致直接报错。
- `14:53`（新增）`src/model/__init__.py`：声明 model 包，只保留模块说明，不导入 torch/timm。
- `14:53`（新增）`src/model/network.py`：新增 `build_model`（timm 构建 ResNet-18 骨干并把分类头换成 10 类；预训练权重下载失败时告警并退回随机初始化，保证离线也能跑）与 `aggregate_logits`（按 offsets 把同一次检查的多张图 logits 求均值，用 `index_add` 实现、梯度可正常回传）。
- `14:53`（新增）`src/model/losses.py`：新增 `build_loss`，默认 `BCEWithLogitsLoss`；`train.pos_weight` 打开时按训练集频率计算 `pos_weight = 负样本数 / 正样本数`，没有正样本的类别取 1，缺训练标签时直接报错。
- `14:53`（变更）`src/common/config.py`：`TrainConfig` 新增 `pos_weight: bool = False`，为 P4 的加权损失与 P7 的对比实验提供开关。
- `14:53`（变更）`configs/default.yaml`：`train` 节新增 `pos_weight: false`；模型注释中的参数量由 11.7M 更正为 11.2M（1000 类版 ResNet-18 的分类头更大）。
- `14:55`（新增）`tests/test_metrics.py`：覆盖与 sklearn 的 macro-AUC/F1 对齐、全负类与全正类剔除后与 sklearn 子集一致、全部退化返回 NaN、F1 随阈值变化、逐类明细列名与 support、形状不一致报错共 8 例。
- `14:55`（新增）`tests/test_model.py`：覆盖 `build_model` 输出 `[N,10]` 与参数量约 11.2M、`aggregate_logits` 对 2/1/3 张图的 study 求均值与梯度、默认损失无 `pos_weight`、`pos_weight` 按类别频率计算、缺训练标签报错、加权损失有限共 8 例。
- `14:57`（变更）`docs/architecture.md`：目录树与测试布局补充 `tests/test_model.py`，配置骨架的 `train` 节补充 `pos_weight: false`。
- `14:57`（变更）`docs/plan0.md`：P4 涉及文件与验收命令纳入 `tests/test_model.py`，参数量期望由约 11.7M 更正为 11,181,642，并补充离线回退预训练权重的说明。
- `15:16`（新增）`src/train/__init__.py`：声明 train 包，只保留模块说明，不导入 torch。
- `15:16`（新增）`src/train/engine.py`：新增 `train_one_epoch`（AMP、梯度反缩放后再裁剪、按 study 数加权平均 loss）、`evaluate`（按 study 聚合后输出 loss / macro-AUC / macro-F1）、`fit`（AdamW、线性 warmup、按 valid macro-AUC 在内存里保留最优权重、AUC 连续不提升即早停），返回 history / best_epoch / best_macro_auc / best_metrics / best_state；AUC 为 NaN（valid 某类全阴或全阳）时不算提升，避免 NaN 抢走最优权重。
- `15:16`（新增）`src/run_train.py`：训练入口，支持 `--config` / `--seed` / `--tag`；每次运行新建 `outputs/<时间戳>_<tag>/` 并写出 config.yaml、train.log、metrics.csv、best.pt，终端输出经 `_Tee` 同步写入日志。
- `15:17`（新增）`tests/test_engine.py`：覆盖单 epoch 参数更新与 loss 有限、warmup 学习率按 `lr/2 → lr` 爬升、评估指标键与取值范围、AUC 停滞触发早停共 4 例。
- `15:18`（变更）`docs/architecture.md`：目录树与测试布局补充 `tests/test_check_data.py` 与 `tests/test_engine.py`，train 模块接口注明 `fit` 返回字段。
- `15:18`（变更）`docs/plan0.md`：测试映射补充 P5 行（`test_engine.py`）。

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
