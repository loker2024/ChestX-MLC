# Repository Guidelines

## 项目结构与模块组织

本仓库用于 ChestX-MLC 胸片多标签分类。P1–P7 的代码、配置与文档已完成；P7 的对照实验和最终提交在训练环境（远程 GPU / 赛题服务器）执行，本机只做代码与测试验证。

- `docs/`：赛题说明、架构与文件结构 `docs/architecture.md`、实现计划 `docs/plan0.md`、调优记录 `docs/experiments.md` 与数据流图；实现前先阅读赛题文档与架构说明。
- `src/`：可复用源码，按 `common/`、`data/`、`model/`、`train/`、`infer/` 分层，并有 `check_data.py`、`run_train.py`、`run_eval.py`、`run_infer.py`、`run_compare.py` 五个入口。
- `tests/`：pytest 测试与合成数据 fixture；当前覆盖标签、索引、数据集、数据体检、指标、模型、训练、配置、run 对比、提交与端到端链路。
- `configs/`：运行配置；`default.yaml` 是冻结的最终配置，`exp_*.yaml` 通过 `_base` 继承它，只写实验差异。
- `requirements.txt`：Python 3.12 依赖及 macOS、Windows、Linux 安装说明。
- `.gitignore`：排除数据、虚拟环境、模型权重及训练产物。
- `CHANGELOG.md`：统一变更日志（即本仓库的 changelog）。

新增实现时，将可复用代码放入 `src/`，测试放入 `tests/`，配置放入 `configs/`。`.gitkeep` 仅用于让 Git 跟踪空目录，目录已有实际文件后应删除。数据放在已忽略的 `data/` 或 `datasets/`；训练产物（含 `best.pt`、续训用的 `last.pt`）放在每次运行新建的 `outputs/<时间戳>_<tag>/`，不再使用单独的 `checkpoints/`。

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

常用入口（在仓库根目录执行）：

```bash
python src/check_data.py --config configs/default.yaml --full
python src/run_train.py  --config configs/default.yaml --seed 42
python src/run_train.py  --resume outputs/<run>
python src/run_eval.py   --config configs/default.yaml --checkpoint outputs/<run>/best.pt --thresholds 0.3,0.4,0.5,0.6
python src/run_infer.py  --config configs/default.yaml --checkpoint outputs/<run>/best.pt --out outputs/submission.csv
python src/run_compare.py --runs outputs/<run_a> outputs/<run_b> --thresholds 0.3,0.4,0.5,0.6,0.7
```

`run_infer.py` 默认输出全部 10 类概率；需要按阈值过滤时加 `--threshold 0.5`（阈值必须在 `[0, 1]`）。

## 编码风格与命名

Python 使用四空格缩进，公开函数添加类型注解；模块、函数和变量使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。CSV 和标注文件显式指定 UTF-8 编码，保留赛题字段名及 ID 的前导零。当前未配置格式化或静态检查工具；引入工具时提交相应配置与依赖。

## 测试与实验验证

测试框架为 `pytest`（依赖已写入 `requirements.txt`），在仓库根目录运行 `python -m pytest`。新增测试文件命名为 `tests/test_<module>.py`，测试函数以 `test_` 开头；当前未配置覆盖率门槛。

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
