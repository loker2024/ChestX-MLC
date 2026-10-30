# 训练计划 09：RUN009-A / B 骨干 @384×384 完整口径复核

## 目的与基线

RUN-008 在 6 轮固定预算、320×320 下完成了四组骨干对照：ConvNeXt-Tiny 6 轮 macro-AUC
**0.8008**（比 RUN-007-A 高 +0.0102、超过 +0.005 判读门，且 6 轮内仍单调上升，是当前唯一
候选方向）；EfficientNetV2-S RW **0.7954**（+0.0048，未达门）；EfficientNet-B2 +0.0046、未达门。
但 RUN-008 的训练预算与 RUN-007-A（25 轮上限、第 12 轮早停）不同，单 seed，不能据此切换默认骨干。

RUN-007-B 的 384×384 探针（DenseNet-121、6 轮短跑）macro-AUC **0.7959**，比 320 下 12 轮早停的
RUN-007-A（0.7906）高 +0.0053，但轮数与 cosine 周期不同，不能单变量归因分辨率。

本计划按用户要求，在 RUN-008 的基础上对两组最有希望的骨干做完整口径复核：**训练 25 轮、
早停 patience 5、输入分辨率 384×384**，判断 6 轮短跑的优势在完整训练预算与更高分辨率下是否保留。
基线对照：RUN-007-A（DenseNet-121 @320，25 轮上限、12 轮早停，**0.7906 / F1@0.5 0.4198**）与
RUN-008-B / RUN-008-C（320、6 轮）。[实验记录](experiments.md)

## 实验矩阵

| 实验 | 配置 | timm 骨干及权重标识 | tag | epochs | 分辨率 |
| --- | --- | --- | --- | --- | --- |
| RUN-009-A | `configs/exp_run009a.yaml` | `efficientnetv2_rw_s.ra2_in1k` | `run009a_effv2s_384_s2026` | 25 | 384×384 |
| RUN-009-B | `configs/exp_run009b.yaml` | `convnext_tiny.fb_in1k` | `run009b_convnext_tiny_384_s2026` | 25 | 384×384 |

两个配置分别继承 RUN-008-B / RUN-008-C，只覆盖三处：`image.size: 320 → 384`、
`train.epochs: 6 → 25`、`train.early_stop_patience: 0 → 5`；其余完全继承 RUN-007-A 口径：
seed 2026、320 时期的正侧位 7:3 加权聚合（`model.aggregation: view_weighted`、
`frontal_weight: 0.7`）、`pos_weight: true`、`batch_studies: 16`、lr 0.0003、2 轮 warmup 后
cosine 到 0.00001、CUDA AMP、`num_workers: 16`、letterbox、ImageNet 归一化、10 类、
`pretrained: true`。本组同时改变了分辨率与训练预算两个变量，**与 RUN-008 的差值不能单变量归因**。

## 权重与数据许可

权重与 RUN-008 相同，许可已在 RUN-008 执行时核验，本计划沿用记录：
RUN-009-A 使用 `timm/efficientnetv2_rw_s.ra2_in1k`（HF commit `ee38fe1c…`，96,483,448 B，
SHA-256 `81379566…`，Apache-2.0）；RUN-009-B 使用 `timm/convnext_tiny.fb_in1k`
（HF commit `b43a6303…`，114,374,272 B，SHA-256 `08b9dc9c…`，Apache-2.0）。
本次运行在 `train.log` 中确认 `预训练权重: 已加载`，并在 `best.pt` 的
`metadata.pretrained_loaded` 复核，避免静默回退随机初始化。数据与 RUN-007-A / RUN-008
为同一份：train 30,344 图 / 18,513 studies；valid 1,422 图 / 859 studies。

## 执行安排（串行）

按要求两组串行执行，避免共享 3090 造成显存不足与耗时不可对照：先跑 RUN-009-B
（ConvNeXt-Tiny，384 下单组显存预计约 8–10 GB），结束后自动启动 RUN-009-A
（EfficientNetV2-S RW，显存预计更高，启动后复核）。执行脚本放在仓库外的
`/root/run009.sh`，顺序调用两条 `run_train.py`：

```bash
nohup /root/run009.sh > /root/run009.serial.log 2>&1 &
# 脚本内顺序：
python src/run_train.py --config configs/exp_run009b.yaml --seed 2026 --tag run009b_convnext_tiny_384_s2026 > outputs/run009b.nohup.log 2>&1
python src/run_train.py --config configs/exp_run009a.yaml --seed 2026 --tag run009a_effv2s_384_s2026 > outputs/run009a.nohup.log 2>&1
```

实际执行（2026-10-30）：原计划的串行脚本 `/root/run009.sh` 不在当前训练实例上（训练环境已换为
V100-SXM2-32G，显存 32 GB、驱动 580.105.08，torch 2.10.0+cu128、timm 1.0.30，96 核 / 375 GB
内存），在代码 `bf7e276` 上先启动 **RUN-009-A**（`HF_HUB_OFFLINE=1`，14:07:36 CST），显存峰值
约 15.9 GB、单轮约 6.7 分钟；按用户要求在第 6 轮结束后（约 14:50）停止，未跑满 25 轮、未触发
早停，**RUN-009-B 未执行**。首次启动时 HuggingFace 直连失败、预训练权重回退随机初始化，已改用
hf-mirror 预下载并以 `HF_HUB_OFFLINE=1` 重启（详见 [训练记录](training_log.md)）。因提前停止，
`best.pt` 由 `last.pt` 的 `extras.best_state` 导出后再评估。
配置解析检查确认两组相对 RUN-007-A 只差骨干、`image.size`、`train.epochs`、
`train.early_stop_patience` 四项。

## 判读门与回填

以 study 级 Macro-AUC 为主指标（`run_compare.py` 在同一份 valid 859 studies 上重算），
F1 记录 0.2–0.7 阈值曲线，不以单个阈值点为主判据。

- **RUN-009-B（ConvNeXt-Tiny）**：相对 RUN-007-A 的门槛仍是 AUC ≥ +0.005（即 ≥ 0.7956），
  这是保留候选资格的最低条件；若要宣称「384 + 完整轮数」带来增量，需相对 RUN-008-C 的
  0.8008 再高 +0.005（≥ 0.8058），并逐类无明显退化。单 seed 下只作方向性结论。
- **RUN-009-A（EfficientNetV2-S RW）**：相对 RUN-007-A 需 ≥ 0.7956 才算方向成立；与
  RUN-008-B 的 0.7954 比较判断 384 与完整轮数是否带来增量；若低于 RUN-008-B 则说明更高
  分辨率 / 完整预算在该骨干上无收益。
- **早停**：valid macro-AUC 连续 5 轮不提升即停；记录 `best_epoch`、实际轮数、停止原因。
- **风险与口径限制**：两组均单 seed；分辨率与训练轮数同时改变；384 尚未与同骨干 320 的
  完整口径配对（RUN-008 只有 6 轮 320），因此即使超过门槛也不能把差异全部归因于分辨率。
  若要切换默认骨干或分辨率，还需第二 seed 复核、赛题指定 T4 端到端耗时实测与权重许可材料。

训练完成后在训练环境回填 [训练记录](training_log.md)（配置原文、运行命令、起止时间、
产物目录、`metrics.csv`、逐类 AUC、显存与耗时）、[调优记录](experiments.md)（对照结论）
与 `CHANGELOG.md`，并按 `run_compare.py` 与 RUN-007-A / RUN-008 / RUN-007-B 对照。
