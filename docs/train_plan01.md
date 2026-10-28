# ChestX-MLC 训练计划（RUN-001～004）

> 状态：RUN-001 / 001R / 002 已记录，RUN-003 进行中（epoch 1 后暂停，可续训）；结果见 [training_log.md](training_log.md)（2026-10-27 制定，2026-10-28 更新状态）
> 执行环境：远程 GPU（RTX 3090 24G），仓库 `/root/ChestX-MLC`，数据软链 `/root/sj-tmp`
> 本机（macOS）只做 pytest 与文档验证，不训练
> 记录出口：每个 run 结束后回填 `docs/training_log.md`，本文件只放「计划 / 命令 / 判读门」

---

## 1. 用途与执行环境

本文件是 RUN-001R～004 的执行清单，所有命令从训练环境的仓库根目录运行。各入口与参数见 [使用手册](manual.md)，耗时口径见 [cost.md](cost.md)。

- **不做决策的地方**：本文件不记录实测数字与结论，那些写进 `docs/training_log.md`（每个 run 的 `metrics.csv`、阈值曲线、`run_compare/summary.md`）。
- **对照原则**：002 和 004B 分别只改 `pos_weight`；004A 同时改最大轮数和早停耐心值，结果只能评价这套延长训练策略，不能单独归因于轮数。
- **提交节奏**：每日最多 2 次提交、仅 1 次撤销。本计划里 003 / 004 都只在本地出 `submission.csv` 校验，择优后才正式提交。
- **规则约束**（`docs/比赛参赛规则.md`）：主指标 Macro-AUC（F1 仅并列时用）；主办方 T4 16G 按 batch=1 复测，单张 ≤100 ms；valid 保留用于早停与选阈值，未并入训练。第 3 条：一旦把 valid 并入训练，就不得再用 valid 做早停 / 阈值选择——本计划不触发该条。

---

## 2. 前置检查（训练前逐条过）

```bash
# 1) 在仓库根目录检查分支与工作区（禁止在 main 上 commit）
cd /root/ChestX-MLC
git -C /root/ChestX-MLC status --short
git -C /root/ChestX-MLC branch --show-current

# 2) 依赖与设备
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

# 3) 数据体检（全量，必须无报错；train 18,513 studies / valid 859 / test 1,414）
python src/check_data.py --config configs/default.yaml --full
```

- `num_workers: 8` 是 `configs/default.yaml` 的默认值，若训练机核数不同，先用 `nproc` 确认，不对就临时改配置（改完记得只影响该 run，不要污染 `default.yaml`）。
- 若 `cost.py` 相关改动（`src/infer/predict.py`、`src/data/dataset.py` 的 `load_image`）尚未同步到远端，先同步再跑 001R，否则 001R 的 checkpoint 与 `cost.py` 验证用的不是同一版代码。

---

## 3. RUN 矩阵

| run | 配置 | 相对父配置的差异 | epochs | 早停 | 预计耗时 | tag | 作用 |
|---|---|---|---|---|---|---|---|
| **RUN-001R**（重跑） | `configs/exp_short.yaml` | 无（与旧 RUN-001 同配置同 seed，只换代码版本） | 6 | 0（关） | ≈30m | `short_baseline_r2` | 回归验证 + `cost.py` 载体 + 阈值曲线 |
| **RUN-002** | `configs/exp_short_pos_weight.yaml` | `train.pos_weight: true` | 6 | 0 | ≈30m | `short_pos_weight` | pos_weight 方向判断 |
| **RUN-003** | `configs/default.yaml`（冻结） | 轮数 6→25，早停 0→5 | 25 | 5 | ≈2h00m | `full_baseline` | 正式基线 + 提交候选 |
| **RUN-004A**（默认分支） | 计划内新增 `configs/exp_full_long.yaml` | `train.epochs: 25→40`、`train.early_stop_patience: 5→8` | 40 | 8 | ≈3h10m | `full_long` | 验证「还没收敛」是否还有收益 |
| **RUN-004B**（备选分支） | `configs/exp_pos_weight.yaml` | `train.pos_weight: true`（25 轮版） | 25 | 5 | ≈2h00m | `pos_weight` | pos_weight 放大到正式训练 |

耗时依据：旧 RUN-001 六轮 28m41s（≈4m47s/轮，含验证与落盘），25 轮 ≈2h00m，40 轮 ≈3h10m，按同比例外推。

### RUN-004A 需要的配置（计划内待建，本轮未创建）

`configs/exp_full_long.yaml`，只写差异、继承冻结配置：

```yaml
# 延长训练：在冻结配置上只把轮数 25→40、早停 5→8。
# 动机：RUN-001 六轮仍单调上升、最优在最后一轮；RUN-003 满 25 轮后若最优仍在
# 末段，用这份配置评价延长训练策略；同时调整轮数和早停，不单独归因于轮数。
_base: default.yaml

train:
  epochs: 40
  early_stop_patience: 8
```

> 决策门走到 004A 时再创建该文件；若走 004B，则不需要它。

---

## 4. 每个 run 的执行口径

所有命令在训练环境仓库根目录执行，产物落在 `outputs/<时间戳>_<tag>/`（`config.yaml` / `train.log` / `metrics.csv` / `best.pt` / `last.pt`）。

### RUN-001R —— 重跑基线（先跑，拿到回归结论与 `cost.py` 载体）

```bash
python src/run_train.py --config configs/exp_short.yaml --seed 42 --tag short_baseline_r2
```

**回归门**：末轮 `valid_macro_auc` 与旧 RUN-001 的 **0.7858** 偏差 **≤0.005**（即落在 0.7808～0.7908），且曲线形状一致（6 轮单调上升、最优 epoch = 6）。

- 超门限（比如掉到 0.75 或涨到 0.82）→ **立即停下查因**，不要带着问题跑 002 / 003。查这几项：`check_data --full` 是否通过、`python -m pytest -q` 是否通过（关注失败和跳过原因）、GPU 型号与 `num_workers` 是否与首次一致、`example` 冒烟是否正常。
- 通过 → 立刻做阈值曲线，留作 001R vs 002 的对照：

```bash
python src/run_eval.py --config configs/exp_short.yaml \
    --checkpoint outputs/<001R>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7
```

### RUN-002 —— pos_weight 短跑（紧跟 001R，同一天完成）

```bash
python src/run_train.py --config configs/exp_short_pos_weight.yaml --seed 42 --tag short_pos_weight
python src/run_compare.py --runs outputs/<001R> outputs/<002> --thresholds 0.2,0.3,0.4,0.5,0.6,0.7
```

**判读门**：macro-AUC 提升 **≥ +0.005**，且 `outputs/compare/summary.md` 的逐类表里没有「少量稀有类暴涨、多数丰富类明显掉」——两条同时满足才判 `pos_weight` **有效**，作为 004B 的候选；否则维持 `pos_weight: false`，004 固定走 004A。逐类数据也保存在 `outputs/compare/per_class.csv`，F1 等指标的阈值见表中 `threshold`。

- 若出现 loss NaN / 明显不收敛：记「`pos_weight` 在 320 分辨率、当前 lr 下训练不稳」，直接关闭该方向，不重试、不占后续 GPU 时间。

### RUN-003 —— 正式基线（25 轮冻结配置）

```bash
python src/run_train.py --config configs/default.yaml --seed 42 --tag full_baseline
```

**判读门**：从 `best.pt` 的 `epoch` 字段或 `run_compare` 汇总表读取 `best_epoch`；`metrics.csv` 用于查看实际运行的各轮 `valid_macro_auc`。默认 patience=5，早停后不会有最佳轮次之后的第 6～10 轮，不能把缺失轮次当作平台期。

| `best_epoch` | 判定 | 004 走向 |
|---|---|---|
| ≥ 20 | 最优轮次靠后，可能仍有提升空间 | 走 **004A**（40 轮） |
| ≤ 15 | 若早停则仅能观察到最佳轮次后的已运行轮次，不能断言后 10 轮走势 | 002 有效则走 **004B**；否则走 004A 测试延长策略 |
| 16～19 | 边缘 | 查看实际已运行轮次与停止原因；若未满 25 轮则无法推断缺失轮次，按 002 的结论选择 004A 或 004B |

003 跑完无论走哪条，先出一次全类概率提交做本地校验（**不提交**）：

```bash
python src/run_infer.py --config configs/default.yaml \
    --checkpoint outputs/<003>/best.pt --out outputs/<003>/submission.csv
```

### RUN-004A —— 延长训练（默认分支）

```bash
python src/run_train.py --config configs/exp_full_long.yaml --seed 42 --tag full_long
```

**判读门**：比 003 的 macro-AUC **≥ +0.005** → 40 轮策略作为最终提交候选；否则维持 25 轮基线。记录两组实际训练轮数及停止原因；这个对比同时改变最大轮数和早停耐心值，不能单独归因于任何一个参数。

### RUN-004B —— pos_weight 正式版（备选分支）

```bash
python src/run_train.py --config configs/exp_pos_weight.yaml --seed 42 --tag pos_weight
```

**判读门**：与 003 单变量对比，macro-AUC **≥ +0.005** 且逐类无大退化 → 作为最终提交候选；否则回退 003。

---

## 5. RUN-001R 之后接 `cost.py`（赛题第 7 条耗时自检）

### 5.1 出提交文件

```bash
python src/run_infer.py --config outputs/<001R>/config.yaml \
    --checkpoint outputs/<001R>/best.pt --out outputs/<001R>/submission.csv
```

### 5.2 准备计时图片

- 来源：`example/test_example`（194 个 study / 283 张图，**每个 study 的图片都完整**，最多 3 张/study）。
- 抽 **100 张**放 `outputs/cost_pics/pictures_100/<subject>/<study>/<pic>.jpg`，另抽 **5 张**预热图放 `outputs/cost_pics/pictures_5/<subject>/<study>/<pic>.jpg`。
- 优先提供配置对应的完整测试集 CSV 和图片；脚本会补齐同一 study 的图片，无需按 study 成组抽取。缺少索引时会降级为按抽样图片分组，比对可能不完整，报告 `final_timing_verifiable=false`。
- 完整索引确认样本涉及多图 study，且一致性检查通过时，脚本返回 2 并报告最终耗时无法判定；若比对不一致则先返回 1。须向主办方确认多图计时口径。
- 这 100 张属于计时样本，不入库、不进 `data/`。

### 5.3 运行与验收

```bash
python cost.py outputs/<001R>/submission.csv outputs/cost_pics/pictures_100 outputs/cost_pics/pictures_5 \
    --config outputs/<001R>/config.yaml --checkpoint outputs/<001R>/best.pt \
    --report outputs/cost_report.json
```

验收：

- [ ] 完整索引确认全部样本均为单图 study、平均耗时不超限且一致性检查通过时，退出码为 **0**；多图 study 且比对一致时返回 **2**，等待官方明确口径。索引缺失时即使返回 0，也须标注结果未经完整索引核实；
- [ ] `[CONCLUSION]` 行显示与 `submission.csv` **不一致 0 张**；
- [ ] 记录实测设备与平均耗时；最终合规性以主办方指定硬件和复测结果为准。

---

## 6. 执行顺序与预计时间

按 RUN-001R → 耗时自检 → RUN-002 → RUN-003 → RUN-004A 或 004B 执行；各阶段的判读门见第 4、5 节。按旧 RUN-001 的单轮耗时估算，总 GPU 时间约 5～6.5 小时，实际耗时以训练环境记录为准。

---

## 7. 判读记录要求

每个 run 结束，把以下三项回填 `docs/training_log.md` 对应 RUN 小节与汇总表：

1. 运行命令与 tag、产物目录、起止时间与总耗时；
2. `metrics.csv`（epoch / lr / train_loss / valid_loss / valid_macro_auc / valid_macro_f1）；
3. 判读结论——**只写「是否达到第 4 节的判读门」**，例如「001R 达回归门，ΔAUC = +0.0004」，不写「效果不错」这类形容词。

对照类结论（002 vs 001R、004 vs 003）附 `outputs/compare/summary.md` 的逐类表，避免只看宏平均；覆盖输出目录前先保存该次报告。

---

## 8. 本轮约束

不修改冻结的 `configs/default.yaml`；不引入 TTA、更高分辨率、更重骨干或集成，也不将 valid 并入训练。其他优化方向见 [experiments.md](experiments.md)，须在基线完成后另行设计对照。
