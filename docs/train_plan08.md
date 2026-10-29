# 实验计划 08：RUN-008（固定 checkpoint 的 logits / 概率聚合对照）

## 目的与范围

固定 RUN-006 的 DenseNet-121 权重，比较「平均 logits 后 sigmoid」与「逐图 sigmoid 后平均概率」，检验聚合空间是否影响 study 级 Macro-AUC。只做验证集推理，不重新训练或选择 checkpoint。

依据论文 *Ensemble of ConvNeXt V2 and MaxViT for Long-Tailed CXR Classification with View-Based Aggregation* 第 3–4 页：视图融合使用概率；当前项目使用 logits。RUN-007 / RUN-007-A 的结果覆盖训练、推理同时采用正侧位 7:3 logits 加权的方案，尚未覆盖本次固定模型的概率聚合对照。

- [论文 v2](https://arxiv.org/abs/2410.10710v2)
- [官方后处理源码（固定版本）](https://github.com/yamagishi0824/cxrlt24-multiview-pp/blob/09c68f4aa519f2892fa615b65cf3ee8b42d77d0f/code/postprocess.py)：公开代码为正侧位分组后 5:5 平均，未完整实现正文的 7:3 / 8:2。

本组只比较 G0 / G1，不加入视图分组、7:3 / 8:2、ASL、模型集成、TTA 或分辨率变化。结果不足时保留默认配置，不自动扩展候选。

## 实验设计

对每个类别分别计算，z_i 为同一 study 第 i 张图片的 logit，N 为该 study 图片数：

| 方案 | 定义 | 唯一差异 |
| --- | --- | --- |
| G0（现有基准） | `sigmoid(sum(z_i) / N)` | 在 logits 空间等权平均 |
| G1（候选） | `sum(sigmoid(z_i)) / N` | 在概率空间等权平均 |

同一 checkpoint 只导出一次逐图 logits，G0 / G1 读取同一缓存。所有图片均参与等权聚合，未知视角也参与；不使用 `ViewPosition` 决定权重。单图 study 的两种结果应一致。

| 实验 | 固定权重目录 | seed | 执行条件 |
| --- | --- | --- | --- |
| RUN-008 | `outputs/20261029-030417_densenet121_s42/` 的 `best.pt` | 42 | 首先执行 |
| RUN-008-A | `outputs/20261029-051514_densenet121_s2026/` 的 `best.pt` | 2026 | RUN-008 的 G1−G0 AUC ≥ +0.005 后执行 |

两组分别固定各自已训练的权重、最佳 epoch 和配置；不重新训练所谓 seed 复核，也不跨 seed 直接相减。历史 G0 参考值为 0.7879669580 / 0.7912546151，以本次同环境重算的 G0 为配对基准。

固定设置：DenseNet-121、320×320 letterbox、ImageNet 归一化、原验证集 859 studies / 1,422 图、10 类顺序、确定性验证预处理。模型 `eval()`，FP32 前向，关闭 autocast、TTA 和随机增强；G0 / G1 均在 FP32 中计算 sigmoid 和均值。不把训练配置 `train.amp: true` 当成此次推理开启 AMP 的依据。

## 现有能力与实施文件

当前 `run_eval.py` / `run_compare.py` 经 `predict_loader` 输出已聚合的 study 概率，不能从该结果恢复逐图 logits，也没有 G1 开关。下文的新入口、参数和缓存是本计划要求实现的接口，**目前尚不存在，须完成本节工具实现后才能执行远程命令**。

| 文件 | 责任 |
| --- | --- |
| 新增 `src/infer/aggregation_compare.py` | 验证集逐图 logits 导出、缓存校验、G0/G1 配对聚合与报告；复用现有加载、预处理、聚合及指标函数 |
| 新增 `src/run_aggregation_compare.py` | CLI 参数与流程编排，提供 `export` / `compare` 两个子命令 |
| 新增 `tests/test_aggregation_compare.py` | 聚合数值、单次前向、身份对齐、缓存校验、报告及 CLI 的合成数据测试 |
| 结果回填 `docs/training_log.md`、`docs/experiments.md` | 记录实测及结论，不把计划写成已完成结果 |
| 同步 `CHANGELOG.md` | 按 Asia/Shanghai 日期与分钟记录实际文件变更 |

复用 `load_checkpoint`、`config_from_dict` / `load_config`、`build_loader`、`load_checkpoint_model`、`aggregate_logits` 和 `common.metrics`。训练入口、默认配置、现有提交与耗时推理路径不在此次工具改动范围内；不创建 `exp_run008.yaml`，避免仅为后处理对照复制训练配置。

## 实施步骤

由执行者按步骤勾选；可使用 `executing-plans` 技能在当前会话顺序执行。

### 1. 实现并验证聚合与缓存工具（本机）

- [x] 先在新测试文件加入以下核心数值用例，运行 `python -m pytest tests/test_aggregation_compare.py -q`，确认因缺少新函数失败，再实现 `aggregate_pair(logits, offsets)`，返回 `(g0, g1)` 两个 `[studies, 10]` FP32 张量。

```python
def test_aggregate_pair_changes_ranking_and_preserves_single_image():
    import torch
    from infer.aggregation_compare import aggregate_pair

    logits = torch.tensor([[8.0], [-2.0], [2.0], [2.0], [0.0]]).repeat(1, 10)
    g0, g1 = aggregate_pair(logits, torch.tensor([0, 2, 4, 5]))
    torch.testing.assert_close(g0[:, 0], torch.tensor([0.9525741, 0.8807971, 0.5]))
    torch.testing.assert_close(g1[:, 0], torch.tensor([0.5594338, 0.8807971, 0.5]))
    assert g0[0, 0] > g0[1, 0]
    assert g1[0, 0] < g1[1, 0]
    assert torch.equal(g0[2], g1[2])
```

- [x] G0 直接复用 `aggregate_logits(logits.float(), offsets)` 后 sigmoid；G1 对 `logits.float()` 先 sigmoid，再调用相同的无视图均值聚合。验证 offsets 为一维整数、首项 0、严格递增、末项等于图片数；logits 必须有限、二维且为 10 类，非法数据明确报错。
- [x] `export --run DIR --device DEVICE --out DIR`：优先读取 `best.pt` 的配置，缺失才回退该 run 的 `config.yaml`，与 `run_compare.py` 一致。仅接受 `model.aggregation == mean` 的已训练模型；读取 valid，禁止传入 test。只允许覆盖设备，不覆盖训练存档中的骨干、尺寸和预处理。
- [x] 按 `build_loader(config, 'valid', train=False)` 的顺序逐批前向，每批仅调用一次模型。将 `model(images).float()` 导出为 CPU 张量，同时保留全局 offsets；用 batch 的 `study_ids` 与 `loader.dataset.records` 对齐，再按 `record.image_paths` / `record.views` 展开图片身份，验证图片数、study 数和标签对应关系。
- [x] 首次导出时，同一批 logits 生成的 G0 要与当前 `predict_loader` 的独立验证结果按 `Study_id` 对齐比较；允许这一次额外基准前向，不为 G1 再跑模型。概率最大绝对差须 ≤ 1e-6，否则停止比较并排查。记录相对历史 AUC 的差值；若绝对差 > 1e-4，先排查配置、数据和运行环境。
- [x] 缓存使用 `image_logits.npz`：`logits[N,10]`、`offsets[B+1]`、`labels[B,10]`、`study_ids[B]`、`subject_ids[B]`、`dicom_ids[N]`、`views[N]`。ID 使用 Unicode 字符串保存，读取 `allow_pickle=False`；`Dicom_id` 为 `record.image_paths` 文件名去后缀。验证 ID 唯一性、标签为 0/1、维度与计数、offsets 和有限值，不丢前导零。
- [x] 同目录写 `metadata.json`：缓存格式版本、导出代码 commit 与工作区状态、Python/torch/timm 版本、checkpoint 路径及 SHA-256、checkpoint epoch/seed、解析后配置、valid CSV SHA-256、按加载顺序的图片身份与文件内容哈希清单、类别顺序、设备/精度、导出时间、G0 与现有路径的最大差值。缓存元数据记录 NPZ SHA-256；目录已存在且非空时报错，不静默覆盖。
- [x] `compare --cache DIR --thresholds GRID`：只读取并校验缓存及 metadata，不加载模型或图像；检查 NPZ 哈希与格式版本，离线重算 G0/G1。身份或哈希不符时报错，不继续生成报告。该子命令不能证明源图像之后没有变化；重新导出时以新指纹辨别数据版本。

### 2. 生成报告并完成本机验证

- [x] 写出 UTF-8 的 `predictions_g0.csv` / `predictions_g1.csv`（`Subject_id,Study_id,p1..p10`），一 study 一行；计算指标前严格按 Study_id 对齐标签，拒绝重复、缺失或额外 ID。
- [x] 写 `summary.csv` / `summary.md`：G0、G1 及 G1−G0 的全体 / 单图 / 多图 Macro-AUC、study 数，各阈值 F1，以及 max / mean 概率绝对差；`per_class.csv` 写每类 support、两种 AUC 与差值。分类别退化时沿用现有 NaN 口径，额外列出有效 AUC 类别数；空子集标记无结果，不填 0。主要判读始终基于完整验证集。
- [x] 增加合成数据用例：单图/相同 logits 时 G0=G1；上面的排序反转；变长 study 和未知视角；单次 export 的模型调用数等于 loader batch 数（额外基准验证调用另计）；compare 用禁止调用模型的替身验证纯缓存计算；前导零 ID 的 NPZ 往返；打乱预测行后仍正确对齐；非法 offsets、NaN/Inf、重复/缺失 ID、错误 NPZ 哈希与缺少权重时拒绝执行；CLI 小数据导出到报告的完整链路。
- [x] 运行 `python -m pytest tests/test_aggregation_compare.py -q`，通过后运行 `python -m pytest -q`；执行 `python src/run_aggregation_compare.py --help` 及两个子命令的 `--help`，确认与下文一致。只使用合成图，不下载预训练权重。
- [x] 检查 `git diff --check` 和实际 diff；记录变更日志及测试结果。提交若被安排，只能位于 develop 或其派生分支，不在 main 上 commit。

## 远程执行命令（工具实现并验证通过后）

在有 RUN-006 权重和原验证集的 GPU 环境、仓库根目录执行。导出前确认验证集 859 studies / 1,422 图及 ID 清单与历史对照一致，运行已有数据体检：

```bash
python src/check_data.py --config configs/exp_densenet121.yaml --full

python src/run_aggregation_compare.py export \
    --run outputs/20261029-030417_densenet121_s42 \
    --device cuda --out outputs/run008_aggregation_s42

python src/run_aggregation_compare.py compare \
    --cache outputs/run008_aggregation_s42 \
    --thresholds 0.2,0.3,0.4,0.5,0.6,0.7
```

仅当 RUN-008 的全体验证集 G1−G0 Macro-AUC ≥ +0.005 时，以相同代码和精度执行 RUN-008-A：

```bash
python src/run_aggregation_compare.py export \
    --run outputs/20261029-051514_densenet121_s2026 \
    --device cuda --out outputs/run008a_aggregation_s2026

python src/run_aggregation_compare.py compare \
    --cache outputs/run008a_aggregation_s2026 \
    --thresholds 0.2,0.3,0.4,0.5,0.6,0.7
```

所有缓存和报告留在已忽略的 `outputs/`，不用 `run_compare.py --runs` 比较本次两个后处理方案：该入口只读取各 run 原有的聚合口径，不会自动计算 G1。

## 判读与结果回填

- G0 一致性校验未通过、身份或数据不一致：本次实验无效，先排查，不报告候选优劣。
- RUN-008 的 ΔAUC < +0.005：未达当前项目的实用增益门，停止本项，默认保持 G0；正的小幅差异可记录为趋势，不等于证明 G1 无效。
- 达门后执行 RUN-008-A，分别报告两个 seed 内的 G1−G0。第二个 seed 不提升：不采用；两个 seed 均提升但第二个未达 +0.005：证据仍不足以切换默认；两个 seed 均达 +0.005：列为候选，结合逐类退化和概率差异解释结果，不能仅凭两个 seed 宣称统计显著。
- 单图子集两种预测应在绝对误差 1e-6 内一致；多图子集变化帮助解释机制。F1 曲线只作辅助，不凭 F1@0.5 上涨认定 AUC 优化有效，不据此调整提交阈值。
- 这里的 +0.005 是预先固定的项目实用门槛，不是统计显著性检验；本轮不增加 bootstrap 或搜索更多权重。
- 回填 `docs/training_log.md`：权重/数据/代码指纹、环境、缓存目录、G0 基准差值、G0/G1/Δ、逐类结果、子集规模和命令；`docs/experiments.md` 记录是否达门、是否复核及实际限制。结果和官方提交分数分开记录。
- 本实验不生成测试集提交、不自动切换默认聚合。如果决定部署 G1，另行统一训练说明、评估、提交和 `cost.py` 共用推理路径，并在指定 T4 环境验证端到端耗时及提交一致性；不能用本次缓存聚合速度代替部署耗时。

## 当前状态

- [x] 已制定实验及工具实施计划。
- [x] 新评估工具和本机测试完成。
- [ ] RUN-008 远程验证完成。
- [ ] 如达门，RUN-008-A 复核完成。
- [ ] 实测结果回填并形成结论。
