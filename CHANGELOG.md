# 更新日志（Changelog）

本文件记录 ChestX-MLC（多目标胸部病变检测分类）项目的全部改动。

约定：

- 每条改动标注到分钟，时区统一为 Asia/Shanghai，格式 `HH:MM`，日期作为分组标题。
- 排序规则：跨天时最新一天在最上；同一天内由早到晚，自上而下。
- 每条改动标注类型：新增 / 变更 / 修复 / 移除。

## 2026-10-22

- `09:52`（新增）`src/predict.py`、`src/submission.py`、`src/experiments.py`：统一验证/测试预测入口、检查级均值聚合、模拟提交与回读校验、独立实验快照和汇总表。
- `09:52`（新增）`tests/test_submission.py`、`tests/test_experiments.py`：验证预测ID聚合、前导零保留、坏概率/重复/缺失检测及实验防覆盖。
- `09:52`（变更）`src/continue_train.py`：增加来源、独立输出目录、追加轮数和学习率参数；默认行为兼容旧入口，本轮不执行训练。
- `09:52`（新增）`outputs/experiments/exp_001/*`、`outputs/experiments/exp_002/*`、`outputs/experiment_summary.csv`（已忽略）：注册两份历史实验快照，保留模型、配置、历史、验证预测/指标、备注及来源哈希。
- `09:52`（变更）`README.md`、`CHANGELOG.md`：记录统一预测、实验登记/比较方式及正式模板缺失的本地模拟限制。
- `09:55`（新增）`outputs/experiments/exp_002/predictions/20261022_095153_89283345/*`（已忽略）：真实验证/测试逐图和检查级 CSV、模拟 submission.csv、submission_check.json、validation_metrics.json、prediction_report.json、prediction_log.jsonl 与 csv_audit.json；验证成绩复现一致，测试2218张、1414次检查，14140行格式检查通过。
- `09:55`（变更）`README.md`、`src/continue_train.py`、`CHANGELOG.md`：记录8项测试通过、真实推理耗时、模型大小与来源哈希不变，完善通用续训说明；本轮无新训练和平台上传。
- `13:06`（修复）`src/submission.py`、`src/predict.py`：测试导出必须显式选择 --all-classes/--threshold，支持四列模板顺序，分别验证完整类别或阈值筛选及原始概率对应；复现差异默认告警，可显式严格阻断；检查点优先读取内嵌配置，支持旧 epoch_N/model.pth。
- `13:06`（新增）`src/devices.py`、`tests/test_predict.py`：统一 CUDA/MPS/CPU 选择与同步/统计，覆盖无旁路配置检查点的完整 CPU 预测、默认告警和严格失败，MPS 分支仅 mock 测试。
- `13:06`（修复）`src/full_train.py`、`src/train.py`、`src/continue_train.py`：全量/单步/续训支持 --device；非CUDA设备不调用CUDA专属统计；新逐轮检查点保存同目录配置。
- `13:06`（变更）`tests/test_submission.py`、`README.md`、`docs/PR_REVIEW_RESPONSE.md`、`CHANGELOG.md`：补齐两种提交语义、空输出/漏项/篡改概率回归测试、基线实验边界、PR描述和审查回应；历史提交未改写，官方语义仍待确认。
- `13:06`（新增）`outputs/review_checks_20261022_130325/*`、`outputs/baseline_v2/epoch_4/predictions/20261022_130319_4ed3e91b/*`（已忽略）：保存真实旧检查点全量验证复测和原真实测试概率两种导出校验结果；未重新训练或上传，未修改原始模型。
- `14:23`（变更）`develop`：撤回 PR #2（merge commit `91e9c3a`，未修复 P0-1 交付材料），删除其新增的 19 个文件（`src/`、`tests/`、`docs/PR_REVIEW_RESPONSE.md`），代码恢复到 `a8d2d47`；PR #2 在 GitHub 上仍显示 MERGED。
- `14:56`（修复）`CHANGELOG.md`：核对 develop 与 yun 分支后补回团队的14:23撤回记录，避免本地日志覆盖团队历史；本条修订尚需上传到yun分支。
- `16:07`（变更）`README.md`、`docs/PR_REVIEW_RESPONSE.md`：补充新克隆的依赖、测试、官方数据准备及默认 configs/data.json 用法；记录实际 PR #2 缺失根目录文档和配置的交付问题，明确本地文件不代表已提交到远端。
- `16:07`（变更）`CHANGELOG.md`：汇总本PR各轮新增源码、测试、配置和审查修订的路径记录；configs/data.json 的新增记录见2026-10-19 19:56，源码/测试路径覆盖原实现及2026-10-22修订；远端补传后须以 Files changed 核验。
- `18:21`（修复）`src/predict.py`：新增仅数据路径覆盖，保持检查点预处理；可选历史参考损坏默认告警/记录invalid，严格模式才阻断；报告生效路径及验证身份。
- `18:21`（修复）`src/experiments.py`：登记前验证、暂存目录原子发布、复制失败可重试、坏旧实验隔离、空汇总和原子CSV替换；新增同一模型验证复测导入和同指纹组内排名。
- `18:21`（新增）`src/fingerprints.py`、`src/training_state.py`：验证划分/标签语义和评估协议指纹；绑定模型哈希的完整epoch状态保存与恢复，不补造旧优化器状态。
- `18:21`（变更）`src/evaluate.py`、`src/full_train.py`、`src/train.py`、`src/continue_train.py`：评估记录指纹；新训练支持独立目录/配置/种子，续训支持显式完整状态恢复；合成规模开关默认关闭并记录，本轮未进行正式训练。
- `18:21`（新增）`tests/test_reliability.py`、`tests/test_training_state.py`（变更）`tests/test_predict.py`、`tests/test_experiments.py`：覆盖坏参考、路径覆盖、失败登记、分组排名、随机状态和ResNet18训练入口恢复一致性，18项测试通过。
- `18:21`（新增）`requirements-baseline.txt`、`docs/RELIABILITY_UPDATE.md`（变更）`README.md`、`docs/PR_REVIEW_RESPONSE.md`、`CHANGELOG.md`：最小直接依赖、隔离环境步骤、恢复边界和17文件上传清单；未改装全局环境或宣称完成干净安装验收。
- `18:21`（新增）`outputs/experiments/exp_002/predictions/20261022_181637_c6140eaf/*`、`outputs/experiments/exp_003/*`（变更）`outputs/experiment_summary.csv`（已忽略）：同一第4轮权重复测Macro-AUC0.730568、Macro-F10.157715并登记验证身份快照，原实验不变，无新正式训练。

## 2026-10-20

- `13:02`（新增）`src/continue_train.py`、`tests/test_continue_train.py`：从 Baseline-1 权重续训 Epoch 2–4，独立保存各轮结果并按 Macro-AUC 选优；校验原始文件哈希，异常立即停止，记录优化器状态缺失的续训限制。
- `13:02`（变更）`README.md`、`CHANGELOG.md`：补充续训命令与输出保留策略。

## 2026-10-19

- `19:16`（新增）`.agents/skills/ncccu-cxr-baseline/SKILL.md`：从 Windows 下载目录原样安装 NCCCU 胸部 X 光多标签分类项目 Skill，校验 name、description 及文件哈希一致性。
- `19:16`（变更）`CHANGELOG.md`：记录本地项目 Skill 安装与检查结果。
- `19:56`（新增）`src/data.py`、`src/check_data.py`、`configs/data.json`：严格解析官方标签、索引图片、保持比例补边、提供 Dataset/DataLoader 和真实数据检查入口。
- `19:56`（新增）`tests/test_data.py`：使用 unittest 检查标签异常、检查标签一致性、图片索引和纵横比保持。
- `19:56`（新增）`README.md`、`.gitignore`：补充数据检查用法和数据、权重、缓存排除规则。
- `19:56`（变更）`CHANGELOG.md`：记录 Round 1 数据模块开发，不修改官方原始数据。

- `19:58`（修复）`tests/test_data.py`：测试临时目录改为项目 tests 下，解决 Windows 沙箱系统临时目录权限错误；CSV 写入显式使用 UTF-8。
- `19:58`（变更）`README.md`：记录实际环境版本、3 项单元测试通过及三份官方数据随机批次加载成功；全量路径匹配成功，跨集合患者/检查/图片 ID 无重叠。

- `20:02`（新增）`src/model.py`、`src/train.py`：无预训练 ResNet18 十输出模型、真实训练集单批次 AdamW 更新、有限值检查及独立目录模型保存/恢复核验。
- `20:02`（新增）`tests/test_model.py`：验证输出与标签形状、损失、参数变化、保存加载一致性及覆盖保护；连同数据模块共 4 项测试通过。
- `20:02`（变更）`README.md`、`CHANGELOG.md`：补充 Round 2 单步测试命令、超参数、复现边界及尚缺完整验证流程的说明。

- `20:03`（新增）`outputs/round2_20261019_200329_d125e783/model.pth`、`smoke_result.json`（已忽略）：保存真实两张训练图的一次更新模型和实测记录；损失 0.7763936519622803，恢复核验通过。
- `20:03`（变更）`README.md`：记录 GPU 型号、真实测试损失、参数变化与保存加载结果，未进行完整训练或验证集评估。

- `20:07`（新增）`src/evaluate.py`、`src/full_train.py`、`tests/test_evaluate.py`：新增全量单轮训练、Study 级平均概率评估、逐类 AUC 和 F1、显存探测、结果保存与恢复验证；5 项测试通过。
- `20:07`（变更）`src/train.py`、`README.md`、`CHANGELOG.md`：保留单步入口，增加 --full-one-epoch 模式及执行说明。

- `20:38`（新增）`outputs/baseline_v1/best_model.pth`、`train_config.json`、`history.csv`、`val_study_predictions.csv`、`validation_metrics.json`（已忽略）：保存全量单轮训练与 859 次检查验证结果；Macro-AUC 0.704093，Macro-F1 0.111472，模型恢复核验通过。
- `20:38`（变更）`README.md`、`src/train.py`、`CHANGELOG.md`：记录实测耗时、显存、损失、阈值局限及入口说明；未预测测试集，未生成提交文件。

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
