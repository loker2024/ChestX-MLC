# 训练计划 23：C-003 ConvNeXt-Tiny 输入分辨率对照

**状态：暂缓执行（2026-11-01 用户决定）。两份独立配置 `configs/exp_C003a.yaml`、`configs/exp_C003b.yaml` 已存在但需先修订再执行；GPU 训练、显存测试与评估均未执行。**

**暂缓说明：** C-002 四组已于 2026-11-01 完成判读（Δ42 +0.0014 / Δ2026 −0.0002，均未达 +0.005 门槛，详见 [train_plan22.md](train_plan22.md) 第 6 节与 [训练记录](training_log.md) 的 C-002 章节），按第 2 节规则应保留 `lr=3e-4`，现有两份配置的该字段恰好与之相符。但用户决定在启动前先修订本计划的配置方案，故本轮不执行 C-003；修订完成前，第 3 节的实验矩阵、第 4 节的训练前核验与第 5 节的显存门禁均未落实。

**目标：** 在学习率与其他训练条件固定后，检验输入从 320×320 提升至 384×384，是否改善 study 级 Macro-AUC，并记录计算代价。

**执行方式：** 按以下复选框逐项执行并记录证据；本文是实验计划，不代表启动训练。使用现有 PyTorch / timm 训练与评估入口，无需更换网络模块。

## 1. 方案与范围

采用「完成 C-002 学习率对照 → 固定一个共同学习率 → 复用两个 seed 的 320 对照 → 新增两次 384 训练」。相比立即固定 3e-4 另起实验，这样能衔接当前学习率工作；相比同时扫 384/448/512，预算更小、比较次数更少。代价是需要先取得有效 C-002 结果。

本轮只检验 320 与 384。448/512 不进入当前矩阵；384 是否有效也不能直接推断更高分辨率有效。既往 EfficientNet 384 结果不能代替 ConvNeXt-Tiny 的分辨率对照。A-001 为 8 轮 / patience 2，当前为 12 轮 / patience 4，A-001 的 0.8038 只作历史参考。

参考：[C-002 学习率计划](train_plan22.md)、[实验记录](experiments.md)、[架构](architecture.md)、[赛题规则](比赛参赛规则.md)。

## 2. 固定学习率与复用条件

- [ ] 完成 C-002-a/b/c/d，使用相同 valid 重新评估 best.pt；核验预训练加载成功、seed、数据和代码来源。
- [ ] 计算低学习率相对高学习率的 Δ42 = AUC(b) − AUC(a)、Δ2026 = AUC(d) − AUC(c)。仅当两者均为正、平均增益 ≥0.005，且没有任一类别在两个 seed 均下降 ≥0.02 时，本轮选择 1.5e-4；否则保留 3e-4。0.005 与 0.02 是预先固定的筛选阈值，不是统计显著性标准；未达到条件不等于低学习率无效。
- [ ] 两个 seed 必须使用同一个学习率，不能各自选最高分；选择必须发生在查看 C-003 结果之前。将选择依据和四组真实运行目录写入执行记录。
- [ ] 复用要求：代码及未提交补丁、依赖、数据版本、预训练文件 SHA-256、GPU 型号和精度设置可核验且一致；展开配置除 image.size 外一致。运行目录包含 config.yaml、best.pt、metrics.csv、train.log，并保留 last.pt。记录是否同机独占以及运行间环境差异。
- [ ] 如果 C-002 未完成、预训练加载失败、配置不符或缺少可核验产物，暂停 C-003；先修复对照资格，不擅自用 A-001 替代。若必须新跑 320 对照，先修订本计划的编号和预算，保留原 C-002 结果身份。

重复使用 valid 选择学习率和分辨率会产生选择偏差；本轮结论只适用于选定学习率，不能据此否定其他学习率与分辨率的交互作用。

## 3. 实验矩阵

| 实验 / tag | 分辨率 | seed | 配置 | 同 seed 的 320 对照 |
| --- | --- | --- | --- | --- |
| C-003-a | 384×384 | 42 | configs/exp_C003a.yaml（已准备） | 选 3e-4 时 C-002-a；选 1.5e-4 时 C-002-b |
| C-003-b | 384×384 | 2026 | configs/exp_C003b.yaml（已准备） | 选 3e-4 时 C-002-c；选 1.5e-4 时 C-002-d |

复用的 320 run 保留 C-002 编号，不改名、不复制成新的训练结果。C-003 两组统一采用选定学习率，每对仅 image.size 不同。

固定项：

| 项目 | 固定值 |
| --- | --- |
| 骨干与初始化 | convnext_tiny.fb_in1k；同一 ImageNet-1k 预训练文件；全量微调 |
| 分类头 | standard；dropout=0；head_lr=null |
| 输入 | 三通道、letterbox=true、ImageNet 归一化、hflip=false |
| 数据增强 | 沿用现有旋转 ±7°、平移 5%、缩放 0.9–1.1、亮度/对比度 0.15 |
| 多图聚合 | view_weighted；正侧位 logits 按 0.7/0.3 聚合后 sigmoid |
| 损失 / 优化器 | BCE + 训练集 study 频率 pos_weight；AdamW；weight_decay=0.01 |
| batch | train.batch_studies=16；eval.batch_studies=32 |
| 调度 | 12 epoch 上限；warmup=2；cosine；lr_min=1e-5 |
| 早停 | patience=4；按 valid Macro-AUC 选择 best.pt |
| 其他 | AMP=true；grad_clip=5；num_workers=16；tensorboard=auto |
| 划分 | 官方原始 train/valid；resplit.enabled=false |

固定的是最大预算和早停规则；实际停止轮数可能不同。不得为 384 临时增加轮数、改变 cosine 周期、启用 TTA 或从 320 的 best.pt 热启动。两组均从原始预训练权重开始；同 seed 不保证不同分辨率的训练随机轨迹完全相同。

## 4. 配置准备与训练前核验

两份 YAML 为完整独立配置，不使用 `_base`，当前 `train.lr=0.0003`、`train.pos_weight_power=1.0`。它们尚未通过 C-002 结果确认；若第 2 节最终选择 1.5e-4，训练前将两份配置的 `train.lr` 同步改为 `0.00015` 并记录变更。不得在 C-002 判读之前启动 C-003。

- [ ] 在仓库根目录激活项目 .venv。将两个有效 320 对照的实际运行目录分别写入环境变量 RUN_320_S42、RUN_320_S2026；目录必须满足第 2 节复用条件。
- [ ] 使用 `load_config(...).to_dict()` 对比每个 C-003 YAML 与对应真实 320 运行配置；把候选的 `image.size` 临时恢复为 320 后，整个字典必须相等。逐项核验第 3 节固定值；包含新增配置项时，以实际运行时展开结果和代码版本为准。
- [ ] 记录代码 HEAD、工作区 diff、依赖、数据索引及预训练文件哈希，确认 C-003 编号未被其他运行占用。
- [ ] 执行 `.venv/bin/python -m pytest tests/test_config.py tests/test_dataset.py tests/test_model.py -q` 和 `git diff --check`。这些检查只说明本地配置及链路通过，不是 GPU 显存或收益证据。
- [ ] 若后续调整配置，连同执行记录同步写入 CHANGELOG.md；不自动提交或推送其他工作区改动。

## 5. 显存检查与失败处理

batch 的单位是 study，一批实际图片数可变，不能只用 16 张随机图片证明 batch 16 能运行。

- [ ] 在正式训练前，独立进程检查 384 的真实 train batch 16 和 valid batch 32；至少覆盖各自图片总数最多的组合。训练检查必须包含 forward、聚合、loss、backward、optimizer.step，以计入优化器状态显存。记录实际图片数、峰值 allocated/reserved、GPU 型号及空闲显存。检查结束后销毁进程，正式运行重新初始化 seed 和预训练权重。
- [ ] train batch 16 OOM：停止本方案；优先使用显存足够的同型号环境。若必须改为 batch 8，应另修订为 320/384 均 batch 8 的新配对方案并重新计入训练预算，不能复用 batch 16 的 C-002 作为纯分辨率对照。
- [ ] valid batch 32 OOM：暂停并统一规划更小评估 batch，320/384 均按同一新口径验证；不得悄悄只改一组。当前项目未提供可直接使用的梯度累积配置，不把它列为现成解决方案。
- [ ] 运行中出现 OOM、NaN 或预训练未加载，记录失败原因并排查，不把失败 run 纳入指标比较。

## 6. 正式训练与产物

以下为未来训练命令；须先完成 C-002 判读、配置确认与显存检查，再在训练环境串行执行：

```bash
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C003a.yaml --seed 42 --tag C-003-a
HF_HUB_OFFLINE=1 python -u src/run_train.py --config configs/exp_C003b.yaml --seed 2026 --tag C-003-b
```

两次训练都按原定规则完成，不因第一组单次分数偏低而临时取消第二组。技术故障除外。不要使用 --init-from 或跨分辨率 --resume。

每组保留 config.yaml、best.pt、last.pt、metrics.csv、train.log 和可用的 tb/。核验 best.pt 的 metadata.pretrained_loaded 为 true。记录真实运行目录、最佳 epoch、实际轮数、每轮耗时、总训练时间、GPU 显存峰值及模型参数量。

预算：默认新增 2 次训练，各最多 12 轮。384/320 的像素数量比为 1.44；这不是实测耗时或显存比例。若仅借用历史 320 约 5.7 分钟/轮粗算，两组跑满约 3.3 小时，不含启动、显存检查和评估；实际预算须用当前设备的 384 实测轮耗时更新。更换 batch 所需的新对照不在该估算内。

## 7. 评估命令

将两组新 run 的真实目录写入 RUN_384_S42、RUN_384_S2026，并保留前面设置的两个 320 目录。以下变量检查防止目录未设置时错误执行：

```bash
: "${RUN_320_S42:?请设置 seed42 的有效320运行目录}"
: "${RUN_320_S2026:?请设置 seed2026 的有效320运行目录}"
: "${RUN_384_S42:?请设置 C-003-a 的真实运行目录}"
: "${RUN_384_S2026:?请设置 C-003-b 的真实运行目录}"

python src/run_eval.py --config "$RUN_384_S42/config.yaml" --checkpoint "$RUN_384_S42/best.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-003-a.csv
python src/run_eval.py --config "$RUN_384_S2026/config.yaml" --checkpoint "$RUN_384_S2026/best.pt" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/predictions/valid/C-003-b.csv
python src/run_compare.py --runs "$RUN_320_S42" "$RUN_384_S42" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/comparisons/compare_C003_s42
python src/run_compare.py --runs "$RUN_320_S2026" "$RUN_384_S2026" --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/comparisons/compare_C003_s2026
```

不得用 320 的配置评估 384 的 checkpoint。对照预测沿用 C-002 原编号；若无逐 study 预测，使用 run_eval.py 从其自身 config.yaml / best.pt 重新导出到对应 C-002 文件。

## 8. 判读标准

- [ ] 核验四份预测的 Study_id、Subject_id 映射、标签和数据版本一致，ID 按字符串处理。报告 valid 的 859 个 study 是否齐全，不静默取交集。
- [ ] 主指标为 study 级连续概率 Macro-AUC；辅助报告 F1@0.5、阈值曲线、10 类 AUC / 支持数、训练及验证 loss、最佳 epoch、耗时和峰值显存。
- [ ] 计算 Δ42 = AUC(384,s42) − AUC(320,s42)，Δ2026 同理；报告各自差值及平均值，不把两次训练当成 1,718 个独立验证样本。
- [ ] 做 2,000 次配对 bootstrap（随机种子 2026），同一次重采样在四份预测中使用相同索引，报告两个 seed 差值与平均差的 95% 区间。先检查一个 Subject_id 是否有多次 study：若有，按患者整簇重采样；否则按 study 配对重采样。某次抽样任一类别只有单一标签时丢弃该次并重采样，最多尝试 20,000 次，记录有效次数；仍不足则明确不提供稳定区间。现有 run_compare.py 不提供该 bootstrap，执行时另写分析脚本并保留源码与产物。

| 观察结果 | 决策 |
| --- | --- |
| 两个 Δ 均为正，平均 Δ≥0.005，且无类别在两 seed 均下降 ≥0.02 | 384 进入优先候选，再检查部署代价与独立验证 |
| 两个 Δ 均为正，但平均 Δ<0.005 | 保留弱正向证据，不自动替换 320，不立即追加更高分辨率 |
| 两个 Δ 方向不同，或有类别在两 seed 均下降 ≥0.02 | 结论不稳定 / 存在类别代价，保留 320，不据单次最高分切换 |
| 两个 Δ 均不为正 | 本轮未观察到一致增益，停止扩大分辨率 |
| 一个 Δ 为零、另一个为正 | 证据不足，保留 320，不自动扩大分辨率 |

区间反映验证样本抽样不确定性，不覆盖训练随机性，也不修正反复使用 valid 的选型偏差。两个 seed 与 +0.005 筛选门均不能证明泛化；如进入最终候选，安排独立验证或 OOF，不默认追加 seed 或启动新的训练。

分辨率不会改变该骨干的参数数量，但会改变计算量；不得根据模型文件大小推断部署速度。进入提交候选前按 docs/cost.md 在指定设备验证耗时，不能用训练 GPU 的速度代替赛题设备结果。本计划不生成 test 提交，也不自动修改融合权重。

## 9. 完成后回填

- [ ] 在本计划记录选定学习率、四个实际 run、代码/数据/权重来源、显存检查、实际耗时、指标和决定。
- [ ] 更新 docs/experiments.md、docs/training_log.md、CHANGELOG.md，区分计划、失败、完成与未执行项。
- [ ] 保留两组比较目录的 summary.md / per_class.csv、C-002/C-003 逐 study 预测、bootstrap 脚本及结果。当前不得填写任何 C-003 分数。
