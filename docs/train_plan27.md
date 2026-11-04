# 训练计划 27：D-002 第三 seed、D-004 单项复核与 E-001 新骨干对照

**状态：已执行（七组，D-004-e 由用户取消）。2026-11-02 19:24:37–23:20:04 在 autodl Tesla V100-SXM2-32G（驱动 580.105.08）串行执行；训练 HEAD `7d8b076`，评测 HEAD `49b4ff9`。**

## 目标

本批八组回答三个问题，按用户指定顺序一次排入：

1. **D-002 补第三 seed。** D-002 的 EMA 只有 seed 42 / 2026 两点（0.8100 / 0.8074，均值 0.8087），均值恰好低于 +0.005 方向性门（0.8088）。补 seed 216 后给出三点均值与离散度，判断「EMA 稳定超出 A-001」是否成立。
2. **D-004 单项复核。** C 系列五项结论（正位权重 0.8 / 384 分辨率 / 对比度 0.25 / 亮度 0.25 / 正位权重 0.9）各自只在 A-001 口径、单 seed 下测过，落点集中在 0.8009–0.8029，与基线 0.7976–0.8038 的差都在 ±0.005 门附近。本组把基线整体换成**已启用 EMA 的 D-002-a 口径**，逐项单独重测——检验这些单项在更强的基线上是否仍然成立，而不是再换一次基线后重复原结论。
3. **E-001 新骨干。** ConvNeXt-Tiny 之外换 EfficientNetV2-S RW（`efficientnetv2_rw_s.ra2_in1k`，22.17M 参数），在 320 / 384 两档分辨率上各测一次，均启用 EMA。

## 实验矩阵与执行顺序

八组均以 `configs/exp_D002a.yaml` 为基线（A-001 训练口径 + `train.ema_enabled: true`、decay 0.999、8 epoch 上限、patience 2、lr 3e-4）。每组只改一项，配置完整独立、不使用 `_base`。

| 顺序 | 实验 / tag | 配置 | 唯一改变（相对 D-002-a） | seed |
| ---: | --- | --- | --- | ---: |
| 1 | D-002-c | `configs/exp_D002c.yaml` | 无（仅 seed 42→216，补第三 seed） | 216 |
| 2 | D-004-a | `configs/exp_D004a.yaml` | `model.frontal_weight` 0.7→0.8 | 42 |
| 3 | E-001 | `configs/exp_E001.yaml` | `model.backbone` convnext_tiny→efficientnetv2_rw_s | 42 |
| 4 | E-001-a | `configs/exp_E001a.yaml` | 骨干同 E-001 + `image.size` 320→384 | 42 |
| 5 | D-004-b | `configs/exp_D004b.yaml` | `image.size` 320→384 | 42 |
| 6 | D-004-c | `configs/exp_D004c.yaml` | `image.contrast` 0.15→0.25 | 42 |
| 7 | D-004-d | `configs/exp_D004d.yaml` | `image.brightness` 0.15→0.25 | 42 |
| 8 | D-004-e | `configs/exp_D004e.yaml` | `model.frontal_weight` 0.7→0.9 | 42 |

八份配置均完整展开、无 `_base`，编号与训练 tag、预测文件一一对应。固定项：原始 train/valid、不重划分、letterbox、ImageNet 归一化、`hflip: false`、标准头、dropout 0、全量微调、`view_weighted` 聚合、BCE + pos_weight（power 1.0）、AdamW、weight_decay 0.01、batch 16 studies、warmup 2、cosine、lr_min 1e-5、AMP、grad_clip 5。`model.xrv_weights` 保留 dataclass 默认值但 `arch: timm` 下不被读取。

### 与 C 系列同名实验的关系

D-004-a/b/c/d/e 与 C-005-c、C-003-a、C-006-d、C-006-b、C-005-d 改的是同一个字段、同一个取值，差别只在基线多了 EMA。**这不是重复实验**：C 系列测的是「该单项相对 A-001 是否有增益」，D-004 测的是「该单项相对启用 EMA 的 A-001 口径是否仍有增益」。两组结论可以不一致，若不一致，说明增益与 EMA 的作用重叠或相互抵消，这本身是机制信息。

## 开始前核验（已完成）

1. **配置预检**：本机把八份配置经 `common.config.load_config` 展开成字典，与 `exp_D002a.yaml` 逐字段 diff——D-002-c 仅 `seed` 42→216，其余七组各自只出现预期的那一项差异，无非预期漂移。同一份检查脚本 `check_d004_e001.py` 在远端启动前复跑，结果「全部通过」，输出写入 `logs/D004_E001_runner.log`。
2. **单元测试**：本机 `python -m pytest tests/test_config.py tests/test_dataset.py tests/test_augmentation_config.py -q` → 61 passed。
3. **预训练权重**：远端 `HF_HUB_OFFLINE=1` 下分别以 `convnext_tiny.fb_in1k`（27.83M 参数）与 `efficientnetv2_rw_s.ra2_in1k`（22.17M 参数）建模型成功，两者均在 HuggingFace 缓存中；首组启动日志确认「预训练权重: 已加载」。
4. **代码与环境**：远端 `loker` 与本地同 HEAD `7d8b076`；GPU 空闲（0 MiB 占用）；数据 `data/{train,valid,test}.csv` 与 `data/images/` 就位。

## 训练与评估

八组严格串行，脚本 `run_D004_E001.sh` 写入远端仓库根目录（不入库），日志 `logs/D004_E001_runner.log`、各组 stdout `logs/<tag>_stdout.log`、显存采样 `logs/D004_E001_vram.tsv`（30 秒一次）。

```bash
nohup setsid bash run_D004_E001.sh >> logs/D004_E001_nohup.log 2>&1 < /dev/null &
```

每组训练后由同一脚本自动评测：`best.pt` → `outputs/predictions/valid/<tag>.csv`、`best_ema.pt` → `outputs/predictions/valid/<tag>-ema.csv`，报告落 `outputs/eval_reports/`，阈值扫 0.2–0.7。全部训练结束后跑一次 `run_compare.py` 汇总八组原始权重到 `outputs/comparisons/compare_D004_E001/`（EMA 权重另行配对，见下）。

八组各 8 轮上限、patience 2，按 D-002 同口径实测 30–37 分钟/组估算，训练合计约 4–5 小时；E-001-a（effv2s@384）可能偏慢。

## 判读口径

- **主指标**：valid 859 studies 的 study 级连续概率 Macro-AUC。辅助：F1@0.5、阈值曲线、逐类 AUC/F1/支持数、best epoch、实际轮数、耗时、显存峰值。
- **主基线**：`D-002-a-EMA`（0.8100 / 0.4283）。所有八组都产出 EMA 权重，因此同类比同类，用 `-EMA` 与 `D-002-a-EMA` 配对。原始权重另与 `D-002-a`（0.8005 / 0.4222）配对，两组结果分别报告，不混用。
- **门限**：ΔAUC ≥ +0.005 且无任何类别 AUC 下降 ≥ 0.02 才算进入优先复核；0 < ΔAUC < 0.005 记为弱正向；其余不切换默认。门是方向性筛选，不代表统计显著。
- **D-002-c 单独口径**：把它与前两点（0.8100 / 0.8074）合成三点均值、极差与样本标准差；同时给出相对 A-001（0.8038）的均值差。三点均值的判读仍用 +0.005 门，但需说明 n=3 的区间不含训练随机性的完整分布。
- **不允许的推论**：八组共享同一份 valid 且反复用于选型，最高分带选择偏差；单 seed（除 D-002 三点）不能把差异归因于结构而非噪声；E-001 同时换骨干与（384 档）分辨率，不可拆项归因。

## 回填

完成后回填 `docs/experiments.md`（D-002 第三 seed 并入 D-002 小节、新增 D-004 与 E-001 小节）、`docs/training_log.md`、本文件「执行记录」、`CHANGELOG.md`，并把八组（含各自 `-EMA` 条目）写入 `~/Desktop/loker_exp.xlsx` 的 `D系列实验` / `E` 对应 sheet。

## 执行记录

### 实际执行范围

按用户指示**取消第 8 组 D-004-e**（`frontal_weight` 0.9），本批以七组收尾。原计划 4–5 小时的八组串行缩短为 3h36m 训练 + 11 分钟评测。

| 顺序 | tag | 起止（CST） | 耗时 | epochs | best（原始 / EMA） | macro-AUC（原始 / EMA） |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | D-002-c | 19:24:37–20:04:10 | 39m33s | 8/8（跑满） | 6 / 5 | 0.7961 / **0.8043** |
| 2 | D-004-a | 20:04:10–20:36:29 | 32m19s | 6/8 | 4 / 5 | 0.7971 / 0.8084 |
| 3 | E-001 | 20:36:29–21:01:21 | 24m52s | 5/8 | 3 / 5 | 0.7970 / 0.8038 |
| 4 | E-001-a | 21:01:21–21:19:00 | 17m39s | 3/8 | 1 / 3 | 0.7865 / 0.7969 |
| 5 | D-004-b | 21:19:00–21:55:53 | 36m53s | 7/8 | 5 / 5 | 0.7961 / 0.8085 |
| 6 | D-004-c | 21:55:53–22:30:50 | 34m57s | 7/8 | 5 / 5 | 0.7945 / **0.8096** |
| 7 | D-004-d | 22:30:50–23:01:00 | 约 30m10s | 6/8 | 4 / 5 | 0.7961 / 0.8058 |
| — | D-004-e | — | — | — | — | 用户取消，未执行 |

七组均确认 ImageNet 预训练加载成功（`HF_HUB_OFFLINE=1` 命中缓存），AMP 与 fp32 复评差值 ≤ 0.0001。

### 评测与对比

七组各评 `best.pt` 与 `best_ema.pt`（阈值 0.2–0.7），共 14 次 `run_eval.py`，23:09:33–23:20:04 完成；随后 `run_compare.py` 汇总七组原始权重到 `outputs/comparisons/compare_D004_E001/`，七个 run 的 `valid_ids_sha256` 均为 `81e8be7f…`、859 studies。产物：`outputs/eval_reports/<tag>{,-ema}.txt`、`outputs/predictions/valid/<tag>{,-ema}.csv`。

### 结果与判读

1. **D-002 第三 seed 把「EMA 稳定超基线」否掉了。** 三点 EMA 0.8100 / 0.8074 / 0.8043，均值 **0.8072**，相对 A-001（0.8038）**+0.0034**，未过 +0.005 门；极差 0.0057、标准差 0.0029，与效应同量级。c 是三点最低，说明 a 的 +0.0062 是上尾。EMA 的组内正向仍稳定（+0.0095 / +0.0075 / +0.0082）。
2. **D-004 四项在 EMA 基线上全部未过门，且全为负。** ΔAUC 对 `D-002-a-EMA`：frontal 0.8 **−0.0016**、384 **−0.0015**、contrast 0.25 **−0.0004**、brightness 0.25 **−0.0042**。c 的贴线由 class1 补偿（9 类 −0.0054、class2 −0.0254 触门）；d 明确负向，翻转了 C-006-b 的「弱正向」。a/b 属中性区。**结论：这四项在更强基线上不再提供可测排序收益，C 系列的「弱正向」整体不成立。**
3. **E-001 换骨干无收益。** 320 档 EMA 0.8038（vs 基线 −0.0062）、384 档 0.7969（−0.0131），9 类同向下滑，class3 分别 −0.0231 / −0.0466。384 档 best epoch 1、3 轮早停，更像优化配置不匹配而非分辨率本身无效。
4. **判读边界未变**：七组共享同一份 valid 反复选型，最高分带选择偏差；除 D-002 三点外均为单 seed，不能把差异归因于结构而非噪声。

### 执行偏差

1. **D-004-e 取消**：用户在 D-004-d 训练期间决定不跑第 8 组，`run_D004_E001.sh` 主循环被终止，避免其继续启动 D-004-e；D-004-d 的 python 训练进程保留至自然收尾，`best.pt` / `best_ema.pt` 正常落盘。
2. **评测重跑一次**：首次评测时远端另有一个 `run_eval_D004.sh` 实例并行运行，两个进程写同一批 `predictions/valid/*.csv` 与 `eval_reports/*.txt`，日志交错、结果来源不可辨。已全部终止、清理半成品，改由单一脚本 `eval_D004_E001.sh` 重跑，本节数据均取自重跑结果。
3. **正文字段核对**：本轮无任何配置改动，七组配置在开跑前已完成逐字段预检；训练与评测之间 `src/` 仅 `train/engine.py` 的 TensorBoard 写入变更，`run_eval.py` 不依赖该模块，评测口径不受影响。
