# Repository Guidelines

## 项目结构与模块组织

本仓库用于 ChestX-MLC 胸片多标签分类，目前处于初始化阶段，尚无训练、推理入口或测试用例。

- `docs/`：赛题说明与数据流图；实现前先阅读赛题文档。
- `src/`：可复用源码；当前仅有 `.gitkeep` 占位文件。
- `tests/`：测试用例；当前仅有 `.gitkeep` 占位文件。
- `configs/`：实验与运行配置；当前仅有 `.gitkeep` 占位文件。
- `requirements.txt`：Python 3.12 依赖及 macOS、Windows、Linux 安装说明。
- `.gitignore`：排除数据、虚拟环境、模型权重及训练产物。
- `CHANGELOG.md`：统一变更日志（即本仓库的 changelog）。

新增实现时，将可复用代码放入 `src/`，测试放入 `tests/`，配置放入 `configs/`。`.gitkeep` 仅用于让 Git 跟踪空目录，可在目录加入实际文件后移除。数据放在已忽略的 `data/` 或 `datasets/`，权重和结果分别放入 `checkpoints/`、`outputs/`。

## 环境搭建与开发命令

macOS/Linux 本地环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

前两条命令创建并激活隔离环境，后两条安装依赖并检查依赖兼容性。Windows 激活命令和 CUDA 安装源见 `requirements.txt`。框架与设备自检：

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

仓库尚无构建、训练或推理命令；新增入口时同步补充用法。

## 编码风格与命名

Python 使用四空格缩进，公开函数添加类型注解；模块、函数和变量使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。CSV 和标注文件显式指定 UTF-8 编码，保留赛题字段名及 ID 的前导零。当前未配置格式化或静态检查工具；引入工具时提交相应配置与依赖。

## 测试与实验验证

目前未配置测试框架或覆盖率门槛。新增代码时建议采用 `pytest`，先添加测试依赖，再使用 `python -m pytest`；文件命名为 `tests/test_<module>.py`，测试函数以 `test_` 开头。

优先验证标签映射、同次检查多图聚合、患者划分无泄漏及 CSV 字段与概率范围。使用小型合成数据测试，禁止提交真实影像。模型变更报告随机种子、数据划分、设备、Macro-AUC 与 Macro-F1；未执行的验证须明确说明。

## 提交与 Pull Request

历史提交采用 `类型: 中文说明`，例如 `docs: 新增贡献指南`、`chore: 初始化仓库`。保持单次提交主题集中。

**禁止在 `main` 上执行 commit；`main` 仅通过 rebase 或 merge 集成变更。** 提交必须位于 `develop` 或以 `develop` 为基准创建的分支，例如：

```bash
git switch -c codex/add-training develop
```

PR 写明目的、涉及文件、验证命令和结果；有关联 issue 时附链接。模型调整附实验对比，图示变更附预览。

## 变更日志与数据约束

新增、修改或删除任何代码或文件，都必须同步记录到 `CHANGELOG.md`，包含路径、改动摘要及“新增／变更／修复／移除”类型。使用 Asia/Shanghai 日期和分钟级 `HH:MM`；日期倒序，同日条目按时间正序排列。

仅使用主办方允许的训练数据，不引入外部标注数据。禁止提交数据集、权重、密钥和训练产物；提交前检查 `git diff` 与暂存区。预测文件按赛题要求输出 UTF-8 的 `submission.csv`。
