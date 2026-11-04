# Repository Guidelines

## 项目结构与模块组织

本仓库用于 ChestX-MLC 胸片多标签分类。P1–P8 的代码、配置与文档已完成；P7 的对照实验、P8 的耗时实测与最终提交在训练环境（远程 GPU / 赛题服务器）执行，本机只做代码与测试验证。

- `docs/`：赛题说明、架构与文件结构 `docs/architecture.md`、实现计划 `docs/plan0.md`、耗时自检说明 `docs/cost.md`、调优记录 `docs/experiments.md` 与数据流图；实现前先阅读赛题文档与架构说明。
- `src/`：可复用源码，按 `common/`、`data/`、`model/`、`train/`、`infer/` 分层（`infer/profiling.py` 放耗时评测口径），包含训练、推理、对照及赛题第 7 条耗时自检等入口。
- `tests/`：pytest 测试与合成数据 fixture；当前覆盖标签、索引、数据集、数据体检、指标、模型、训练、配置、run 对比、提交、耗时自检与端到端链路。
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
python src/cost.py outputs/submission.csv data/cost/pictures_100 data/cost/pictures_5 \
    --config configs/default.yaml --checkpoint outputs/<run>/best.pt \
    --report outputs/cost_report.json
```

`run_infer.py` 默认输出全部 10 类概率；需要按阈值过滤时加 `--threshold 0.5`（阈值必须在 `[0, 1]`）。

`cost.py` 以单个 study 为最小计时单元（`batch=1` 以 study 为单位，一次检查的全部图片同批前向），连续 3 轮，先对每个 study 取均值、再对全部 study 取平均，与 100 ms 比较；单次计时覆盖解码、预处理、前向、study 聚合、后处理与逐图写出，文件打开/写入/flush/关闭全部计入（每轮只开一次预测文件）。5 个预热 study 不计入成绩且必须与计时 study 不相交；初始化只允许 CSV 元数据、study 分组与路径索引，打印 `init_ms` 并受自设上限校验。提交为多模型融合时须随包提供 `cost_ensemble.json`（权重与融合方式），模型使用 `view_weighted` 时须提供 `cost_views.csv`，否则会低估耗时或无法复现 `submission.csv`。参数与限制见 `docs/cost.md`；`--precision fp16` 时须在提交说明中声明。

## 编码风格与命名

Python 使用四空格缩进，公开函数添加类型注解；模块、函数和变量使用 `snake_case`，类使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。CSV 和标注文件显式指定 UTF-8 编码，保留赛题字段名及 ID 的前导零。当前未配置格式化或静态检查工具；引入工具时提交相应配置与依赖。

## 测试与实验验证

测试框架为 `pytest`（依赖已写入 `requirements.txt`），在仓库根目录运行 `python -m pytest`。新增测试文件命名为 `tests/test_<module>.py`，测试函数以 `test_` 开头；当前未配置覆盖率门槛。

优先验证标签映射、同次检查多图聚合、患者划分无泄漏及 CSV 字段与概率范围。使用小型合成数据测试，禁止提交真实影像。模型变更报告随机种子、数据划分、设备、Macro-AUC 与 Macro-F1；未执行的验证须明确说明。

## 实验记录与实验表格

每次填写实验记录（`docs/training_log.md` 的章节、`docs/experiments.md` 的实验矩阵与执行小节、`docs/train_planNN.md` 的执行记录），都必须同步填写实验表格 `~/Desktop/loker_exp.xlsx`，不得只改文档。表格位于仓库外，**禁止提交进 Git**。

- sheet 归属：`Sheet1` 放 RUN 系列，`A系列实验` 放 A 系列与 A-FUSE，`B系列实验` 放 B 系列，`C系列实验` 放 C 系列。
- 列布局：A:B「实验名称」、C:D「实验性能」、E:G「实验参数」为合并单元格，H「实验分析」独立；标题在第 1 行，表头在第 3 行，数据行从第 4 行起。
- 内容口径：性能列写「验证集 Study 级 Macro-AUC / Macro-F1」并带最好轮次与耗时，参数列写配置路径与关键超参，分析列写相对基线的判读（Δ、门限、是否通过），不只堆数字。
- 修改方式：直接编辑 OOXML——重写 `xl/worksheets/sheetN.xml`，向 `xl/sharedStrings.xml` 追加字符串并同步 `count` / `uniqueCount`，其余 zip 条目原样保留；**不得用 openpyxl 整本重存**，否则会丢失 WPS 命名空间、合并单元格与行高。新增行沿用邻行样式。
- 写后校验：备份原表，`xmllint --noout` 检查 XML，比对 zip 条目确认只有目标条目变化，再用 Excel 只读打开确认无修复提示。

## 提交与 Pull Request

历史提交采用 `类型: 中文说明`，例如 `docs: 新增贡献指南`、`chore: 初始化仓库`。保持单次提交主题集中。

**禁止在 `main` 上执行 commit；`main` 仅通过 rebase 或 merge 集成变更。** 提交必须位于 `develop` 或以 `develop` 为基准创建的分支，例如：

```bash
git switch -c codex/add-training develop
```

PR 写明目的、涉及文件、验证命令和结果；有关联 issue 时附链接。模型调整附实验对比，图示变更附预览。

## 与 GitHub 同步

与 GitHub 的同步（clone、fetch、push）统一使用 SSH 密钥方式，远端地址固定为 `git@github.com:loker2024/ChestX-MLC.git`；不使用 HTTPS 地址，也不使用账号密码或 Personal Access Token 推送。

```bash
git clone git@github.com:loker2024/ChestX-MLC.git
# 已有仓库若 origin 仍为 HTTPS，改为 SSH 并确认
git remote set-url origin git@github.com:loker2024/ChestX-MLC.git
git remote -v
```

密钥为 `~/.ssh/id_ed25519`（公钥已添加到 GitHub 账号），用 `ssh -T git@github.com` 验证，输出 `Hi <username>! You've successfully authenticated` 即认证正常。私钥、口令和 token 一律不得写入仓库、提交记录或文档。本机与训练环境遵循同一约定。

## 变更日志与数据约束

新增、修改或删除任何代码或文件，都必须同步记录到 `CHANGELOG.md`，包含路径、改动摘要及“新增／变更／修复／移除”类型。使用 Asia/Shanghai 日期和分钟级 `HH:MM`；日期倒序，同日条目按时间正序排列。

仅使用主办方允许的训练数据，不引入外部标注数据。禁止提交数据集、权重、密钥和训练产物；提交前检查 `git diff` 与暂存区。预测文件按赛题要求输出 UTF-8 的 `submission.csv`。
