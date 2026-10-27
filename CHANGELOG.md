# 更新日志（Changelog）

本文件记录 ChestX-MLC（多目标胸部病变检测分类）项目的全部改动。

约定：

- 每条改动标注到分钟，时区统一为 Asia/Shanghai，格式 `HH:MM`，日期作为分组标题。
- 排序规则：跨天时最新一天在最上；同一天内由早到晚，自上而下。
- 每条改动标注类型：新增 / 变更 / 修复 / 移除。

## 2026-10-27

- `10:14`（变更）`.gitignore`：新增忽略根目录 `/example` 示例目录，避免本地示例数据入库；同时忽略 `docs/sju.md` 算家云环境笔记，防止其中的 SSH 密码等敏感信息被提交到远端。

## 2026-10-22

- `14:23`（变更）`develop`：撤回 PR #2（merge commit `91e9c3a`，未修复 P0-1 交付材料），删除其新增的 19 个文件（`src/`、`tests/`、`docs/PR_REVIEW_RESPONSE.md`），代码恢复到 `a8d2d47`；PR #2 在 GitHub 上仍显示 MERGED。

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


