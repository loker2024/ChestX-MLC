# 实验计划：固定 checkpoint 的 logits / 概率聚合对照（A-001-G1）

## 目的与范围

固定一个已完成训练的权重，比较「平均 logits 后 sigmoid」与「逐图 sigmoid 后平均概率」，检验聚合空间是否影响 study 级 Macro-AUC。只做验证集推理，不重新训练或选择 checkpoint。

依据论文 *Ensemble of ConvNeXt V2 and MaxViT for Long-Tailed CXR Classification with View-Based Aggregation* 第 3–4 页：视图融合使用概率；当前项目使用 logits。RUN-007 / RUN-007-A 的结果覆盖训练、推理同时采用正侧位 7:3 logits 加权的方案，尚未覆盖本次固定模型的概率聚合对照。

- [论文 v2](https://arxiv.org/abs/2410.10710v2)
- [官方后处理源码（固定版本）](https://github.com/yamagishi0824/cxrlt24-multiview-pp/blob/09c68f4aa519f2892fa615b65cf3ee8b42d77d0f/code/postprocess.py)：公开代码为正侧位分组后 5:5 平均，未完整实现正文的 7:3 / 8:2。

本组只比较 G0 / G1，不加入视图分组、7:3 / 8:2、ASL、模型集成、TTA 或分辨率变化。结果不足时保留默认配置，不自动扩展候选。

## 实验设计

对每个类别分别计算，z_i 为同一 study 第 i 张图片的 logit，N 为该 study 图片数：

| 方案       | 定义                      | 唯一差异                                               |
| -------- | ----------------------- | -------------------------------------------------- |
| G0（现有基准） | `sigmoid(sum(z_i) / N)` | 在 logits 空间等权平均，即 `aggregate_logits` 的默认 `mean` 路径 |
| G1（候选）   | `sum(sigmoid(z_i)) / N` | 在概率空间等权平均                                          |

同一 checkpoint 只导出一次逐图 logits，G0 / G1 读取同一缓存。所有图片均参与等权聚合，未知视角也参与；不使用 `ViewPosition` 决定权重。单图 study 的两种结果应一致。

**聚合口径与权重。** 工具支持两种聚合模式，由被对照模型自己的配置决定；G0 / G1 在同一个聚合算子下计算，差值只反映「在哪个空间平均」：

| 模型配置                         | G0                                         | G1                                      |
| ---------------------------- | ------------------------------------------ | --------------------------------------- |
| `aggregation: mean`          | `sigmoid(mean(z))`                         | `mean(sigmoid(z))`                      |
| `aggregation: view_weighted` | `sigmoid(w·mean(z_正位) + (1-w)·mean(z_侧位))` | `w·mean(σ(z_正位)) + (1-w)·mean(σ(z_侧位))` |

`w` 为正位权重，默认取 checkpoint 存档配置的 `model.frontal_weight`（项目默认 0.7），此时 G0 就是该模型的线上口径。`export` 与 `compare` 都提供 `--frontal-weight` 覆盖它做权重扫描：覆盖后 G0 不再是线上口径，但与 G1 仍严格配对，报告里会标注「CLI 覆盖」。单视角或含未知视角（`PA LLD` 等）的 study 按 `aggregate_logits` 既有规则回退为全图等权均值。

候选权重（按本地归档名；A-001 系列为 `view_weighted`，其余为 `mean`）：

| 实验        | 骨干            | 聚合            | 本地归档目录                          | 备注                              |
| --------- | ------------- | ------------- | ------------------------------- | ------------------------------- |
| A-001     | ConvNeXt-Tiny | view_weighted | `outputs/experiments/A-001`     | 当前最佳 0.8038 / 0.4255            |
| A-001-D   | ConvNeXt-Tiny | mean          | `outputs/experiments/A-001-D`   | 同骨干同 seed 的等权对照，0.7931 / 0.4340 |
| RUN-006   | DenseNet-121  | mean          | `outputs/experiments/RUN-006`   | 原计划口径，seed 42                   |
| RUN-006-A | DenseNet-121  | mean          | `outputs/experiments/RUN-006-A` | seed 2026，达门后复核用                |
| A-006     | ConvNeXt-Tiny | view_weighted | `outputs/experiments/A-006`     | ML-Decoder 头，0.7900 / 0.4118    |

本地归档目录只保留 `config.yaml` / `metrics.csv` / `train.log` / `submission.csv`，**不含 `best.pt`**；权重只存在于训练环境，须在远程 GPU 上执行。

固定设置：320×320 letterbox、ImageNet 归一化、原验证集 859 studies / 1,422 图、10 类顺序、确定性验证预处理。模型 `eval()`，FP32 前向，关闭 autocast、TTA 和随机增强；G0 / G1 均在 FP32 中计算 sigmoid 和均值。不把训练配置 `train.amp: true` 当成此次推理开启 AMP 的依据。

## 实施文件

| 文件                                                | 责任                                                    |
| ------------------------------------------------- | ----------------------------------------------------- |
| `src/infer/aggregation_compare.py`                | 验证集逐图 logits 导出、缓存校验、G0/G1 配对聚合与报告；复用现有加载、预处理、聚合及指标函数 |
| `src/run_aggregation_compare.py`                  | CLI 参数与流程编排，提供 `export` / `compare` 两个子命令             |
| `tests/test_aggregation_compare.py`               | 聚合数值、单次前向、身份对齐、缓存校验、报告及 CLI 的合成数据测试                   |
| 结果回填 `docs/training_log.md`、`docs/experiments.md` | 记录实测及结论，不把计划写成已完成结果                                   |
| 同步 `CHANGELOG.md`                                 | 按 Asia/Shanghai 日期与分钟记录实际文件变更                         |

复用 `load_checkpoint`、`config_from_dict` / `load_config`、`build_loader`、`load_checkpoint_model`、`aggregate_logits` 和 `common.metrics`。训练入口、默认配置、现有提交与耗时推理路径不在此次工具改动范围内；不创建 `exp_aggregation.yaml`，避免仅为后处理对照复制训练配置。

## 工具行为（已实现）

- `aggregate_pair(logits, offsets, views=None, frontal_weight=0.7)` 返回 `(g0, g1)` 两个 `[studies, 10]` FP32 张量。G0 = `sigmoid(aggregate_logits(logits))`，G1 = `aggregate_logits(sigmoid(logits))`，两边共用同一个聚合算子：`views=None` 走全图等权，`views` 非空走正/侧位分组均值再加权。因此加权模型与等权模型用同一套代码，差值只来自聚合空间。offsets 必须一维整数、首项 0、严格递增、末项等于图片数；views 必须一维整数、长度为图片数、取值 ∈ {0, 1, 2}；logits 必须有限、二维且为 10 类。
- `export --run DIR --device DEVICE --out DIR [--frontal-weight W]`：优先读 `best.pt` 的配置，缺失才回退该 run 的 `config.yaml`。接受 `mean` 与 `view_weighted` 两种模型，只读 valid。按 `build_loader(config, 'valid', train=False)` 顺序逐批前向，每批仅调用一次模型；`model(images).float()` 导出为 CPU 张量。首次导出时用 `predict_loader` 独立跑一遍作基准，只允许这一次额外前向，G0 最大绝对差须 ≤ 1e-6；加权模型与 `--frontal-weight` 覆盖时，基准也会用同一权重，口径不对齐会直接报错而不是给出误导性数字。
- 缓存为 `image_logits.npz`（`logits[N,10]`、`offsets[B+1]`、`labels[B,10]`、`study_ids[B]`、`subject_ids[B]`、`dicom_ids[N]`、`views[N]`）加 `metadata.json`（格式版本、代码 commit 与工作区状态、Python/torch/timm 版本、checkpoint 路径与 SHA-256、epoch/seed、解析后配置、**聚合模式与正位权重及其来源**、valid CSV SHA-256、图片内容哈希清单、类别顺序、设备/精度、导出时间、G0 基准差值）。ID 以 Unicode 字符串保存、`allow_pickle=False`，保留前导零；目录已存在且非空时报错，不静默覆盖。
- `compare --cache DIR --thresholds GRID [--frontal-weight W]`：只读缓存与 metadata，不加载模型、不读图片；按 metadata 记录的聚合模式与权重重算，`--frontal-weight` 可在同一份缓存上做权重扫描（此时报告标注「CLI 覆盖」）。NPZ 哈希或格式版本不符、ID 重复/缺失/额外时拒绝出报告。写出 `predictions_g0.csv` / `predictions_g1.csv`（`Subject_id,Study_id,p1..p10`）、`summary.csv` / `summary.md`（全体 / 单图 / 多图 Macro-AUC、各阈值 F1、概率 max/mean 绝对差、有效 AUC 类别数、空子集标无结果）、`per_class.csv`（每类 support、两种 AUC 与差值）。

## 远程执行命令

在有目标权重和原验证集的 GPU 环境、仓库根目录执行。导出前确认验证集 859 studies / 1,422 图及 ID 清单与历史对照一致：

```bash
python src/check_data.py --config configs/exp_densenet121.yaml --full

python src/run_aggregation_compare.py export \
    --run outputs/20261029-030417_densenet121_s42 \
    --device cuda --out outputs/agg01_run006_s42

python src/run_aggregation_compare.py compare \
    --cache outputs/agg01_run006_s42 \
    --thresholds 0.2,0.3,0.4,0.5,0.6,0.7
```

`--run` 可换成任意已训练 run 目录（远程目录名尚未按本地归档重命名）：`mean` 模型跑全图等权口径，`view_weighted` 模型跑 7:3 分组加权口径，两者共用同一套报告。仅当 AGG-01 的全体验证集 G1−G0 Macro-AUC ≥ +0.005 时，以相同代码和精度执行 AGG-01-A：

```bash
python src/run_aggregation_compare.py export \
    --run outputs/20261029-051514_densenet121_s2026 \
    --device cuda --out outputs/agg01a_run006a_s2026

python src/run_aggregation_compare.py compare \
    --cache outputs/agg01a_run006a_s2026 \
    --thresholds 0.2,0.3,0.4,0.5,0.6,0.7
```

加权模型（以 A-001 为例，导出一次后即可反复扫描权重）：

```bash
python src/run_aggregation_compare.py export \
    --run outputs/<A-001 的 run 目录> \
    --device cuda --out outputs/agg01_a001_s42

# 沿用存档的 frontal_weight（0.7），G0 即线上口径
python src/run_aggregation_compare.py compare \
    --cache outputs/agg01_a001_s42 --thresholds 0.2,0.3,0.4,0.5,0.6,0.7

# 权重扫描：同一份缓存，不重跑模型；0.5 / 0.6 / 0.8 各出一份报告
for w in 0.5 0.6 0.8; do
  cp -r outputs/agg01_a001_s42 outputs/agg01_a001_s42_w${w}
  python src/run_aggregation_compare.py compare \
      --cache outputs/agg01_a001_s42_w${w} --frontal-weight ${w} \
      --thresholds 0.2,0.3,0.4,0.5,0.6,0.7
done
```

扫描时 G0 与 G1 用同一权重，差值仍只反映聚合空间；但 G0 已不等于该 checkpoint 的线上口径，报告会标注「CLI 覆盖」，不能直接当作提交口径比较。

所有缓存和报告留在已忽略的 `outputs/`。不要用 `run_compare.py --runs` 比较本次两个后处理方案：该入口只读各 run 原有的聚合口径，不会自动计算 G1。

## 判读与结果回填

- G0 一致性校验未通过、身份或数据不一致：本次实验无效，先排查，不报告候选优劣。
- AGG-01 的 ΔAUC < +0.005：未达实用增益门，停止本项，默认保持 G0；正的小幅差异可记录为趋势，不等于证明 G1 无效。
- 达门后执行 AGG-01-A，分别报告两个 seed 内的 G1−G0。第二个 seed 不提升：不采用；两个 seed 均提升但第二个未达 +0.005：证据仍不足以切换默认；两个 seed 均达 +0.005：列为候选，结合逐类退化和概率差异解释结果，不能仅凭两个 seed 宣称统计显著。
- 单图子集两种预测应在绝对误差 1e-6 内一致；多图子集变化帮助解释机制。F1 曲线只作辅助，不凭 F1@0.5 上涨认定 AUC 优化有效，不据此调整提交阈值。
- 这里的 +0.005 是预先固定的项目实用门槛，不是统计显著性检验；本轮不增加 bootstrap 或搜索更多权重。
- 回填 `docs/training_log.md`：权重/数据/代码指纹、环境、缓存目录、G0 基准差值、G0/G1/Δ、逐类结果、子集规模和命令；`docs/experiments.md` 记录是否达门、是否复核及实际限制。结果和官方提交分数分开记录。
- 本实验不生成测试集提交、不自动切换默认聚合。如果决定部署 G1，另行统一训练说明、评估、提交和 `cost.py` 共用推理路径，并在指定 T4 环境验证端到端耗时及提交一致性；不能用本次缓存聚合速度代替部署耗时。

## 当前状态

- [x] 已制定实验及工具实施计划。
- [x] 评估工具与本机合成数据测试完成。
- [x] 工具按本计划恢复（2026-10-31），全量 `python -m pytest -q` 285 passed / 3 skipped。
- [x] 支持 `view_weighted` 模型与可调正位权重（2026-10-31）：`aggregate_pair` 增加 `views` / `frontal_weight`，`export` / `compare` 均可用 `--frontal-weight` 覆盖；`tests/test_aggregation_compare.py` 28 passed，全量 294 passed / 3 skipped。
- [ ] AGG-01 远程验证完成。
- [ ] 如达门，AGG-01-A 复核完成。
- [ ] 实测结果回填并形成结论。
