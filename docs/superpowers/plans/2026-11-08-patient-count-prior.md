# 患者检查次数辅助后处理实施计划

> 面向执行者：逐任务执行并勾选进度。最初交付为计划，后续用户已授权实现代码；当前落地范围见下方状态及使用说明。文中拟议实验产物不表示已运行。执行期间遵守仓库 AGENTS.md：默认不运行测试、训练、评估或耗时脚本；需要执行时另行取得用户明确授权。

**目标：** 判断患者检查次数能否在影像预测之外改善 Study 级 10 类 Macro-AUC，并提供默认关闭、参数冻结、可撤销的后处理模块。

**架构：** 由独立元数据模块按患者去重统计 Study_id；在单模型 study 聚合或多模型融合结束后，统一对概率施加次数相关的 logit 偏置。离线研究、参数选型和正式提交使用明确不同的入口权限；正式使用受赛规解释与独立评估结果约束。

**技术栈：** Python 3.12、NumPy、pandas、scikit-learn、PyYAML，复用现有 PyTorch 模型、指标与推理链路，不新增依赖。

**状态：** 已编写自动计数、数值核心、研究 CLI、OOF 编排及 train/valid 评估接入代码和合成测试；2026-11-08 新增模块40项测试通过，D-002-a-EMA历史valid CPU探索已执行；真实OOF/GPU训练与耗时未运行。正式 test/cost 接入条件未满足，入口明确拒绝。无新增训练结果、无冻结参数、无正式提交资格结论；历史valid探索不构成独立收益证据。实际使用命令见 [使用说明](../../patient_count_prior.md)；下列复选框保留为含运行验收的原计划，不表示已全部完成。

## 1. 当前证据与设计边界

- `docs/classes.md`：类别 1–9 为异常，类别 10 为“无异常发现”，不能全部同向提高概率。
- `src/data/index.py` 的 `StudyRecord` 已含 `study_id`、`subject_id`；不需要把患者 ID 输入神经网络。
- `src/run_eval.py` 可导出 Study_id、label_1..label_10、p1..p10；未含患者 ID，需与来源 CSV 显式关联。
- `src/run_infer.py` 产生单模型概率宽表；`src/run_ensemble.py` 调用 `fuse_prediction_frames` 后产生融合概率宽表。
- `src/cost.py::_fuse_study` 复用 `fuse_probability_vectors`；后处理应接在融合之后，并处于单 study 计时范围内。
- `src/data/resplit.py` 会合并官方 train/valid 后随机划分，不能用作“只在官方 train 内生成 OOF”的实现。
- 当前文件清单未发现可直接使用的 OOF/fold 预测清单；执行时必须重新枚举并核验，不能假设 OOF 已存在。
- D-009 是训练重采样实验，不是推理先验实验；其负向结果不能直接代替本方案的验证，也不能被本方案解释为已经改善。
- 先前 valid 上的观察属于提出假设的依据，尚未证明检查次数在控制影像预测后还有增量信息。

### 1.1 赛规判断与正式接入条件

仓库 `docs/比赛参赛规则.md` 第 5 条禁止用测试集拟合参数、估计类别先验；第 6 条还明确禁止“跨多张测试图汇总统计量”。从完整 test.csv 按 Subject_id 汇总检查数，属于跨检查信息，不能仅凭参数提前冻结便认定允许。

因此，默认交付是 **train/valid 离线研究能力**。正式测试入口在没有书面许可时必须拒绝启用本特征。向主办方提出一个具体问题：

> 能否在测试推理时，仅通过官方元数据按 Subject_id 去重统计该患者在完整测试集中的 Study_id 数，将它输入训练阶段已经固定的后处理函数？不使用其他检查的像素、标签或预测，不在测试集拟合任何参数。如果允许，耗时复测是否提供或允许随包携带同一范围的计数元数据？

只有明确许可同时覆盖计数来源、统计范围和复测可复现性，才进入任务 6 的正式接入。当前规则不能自动推导出许可。不能通过预生成计数表、写入配置或把患者数量藏进模型来绕过这一限制。

若主办方不允许：保留离线结论，正式提交保持关闭；单 study 的图片数是另一种特征，不自动替换本方案，替换须另立假设。

### 1.2 本次文档交付范围

只新增本计划与 CHANGELOG 条目，不改已有实验记录，不登记虚构结果；因此本次不修改 `~/Desktop/loker_exp.xlsx`。真正执行并填写实验记录时，必须同步台账。

## 2. 方案选择

| 方案 | 参数量 | 优点 | 主要问题 | 本期决策 |
| --- | ---: | --- | --- | --- |
| 截断 log 次数 + 共用强度的 logit 偏置 | 1 个强度 | 容易解释、快速离线扫描、可原样关闭 | 不同病种可能不共享方向和幅度 | 采用 |
| 各类别独立系数，或次数分桶查表 | 至少 10 个 | 可拟合类别差异 | 稀有类过拟合、选择次数增加 | 本期不做 |
| 将次数拼入模型分类头联合训练 | 额外模型参数 | 可学习影像与次数的交互 | 要重训、难归因，计数可用性问题仍在 | 本期不做 |

第一期只评估一套符号约定、一种次数变换和四个强度候选，不同时搜索封顶数、分桶边界、温度、类别权重或阈值。

## 3. 固定的数学定义

对患者 j，在当前允许的元数据范围内定义：

\[
n_j=|\{\mathrm{Study\_id}:\mathrm{Subject\_id}=j\}|,
\qquad
g(n_j)=\log(1+\min(n_j,10))-\log 2.
\]

封顶值 10 是本期事前固定的设计常量，不由测试分布决定；n=1 时 g=0，n≥10 后不再加强。它描述“该数据范围内可见的检查数”，不称为患者一生的检查总数，也不称为当前检查之前的历史次数。

对最终基线概率 p_c：

\[
z_c=\operatorname{logit}(\operatorname{clip}(p_c,10^{-6},1-10^{-6})),
\qquad
p'_c=\sigma(z_c+s_c\beta g(n_j)),
\]

其中 s_1..s_9=+1，s_10=−1；候选强度固定为 **β∈{0, 0.05, 0.10, 0.20}**。β=0 是原始基线。β=0 或功能关闭时必须直接返回原概率，不经过 clip/logit；即使 β>0，g=0 的行也直接保留原值，避免端点概率被数值裁剪改变。

例：β=0.20、n≥10 时，异常类 logit 增加约 0.341，赔率乘约 1.41；0.30 的异常概率变为约 0.376。该例仅解释函数强度，不代表推荐参数或预期收益。

约束：

- 概率仍按 10 个独立 sigmoid 输出，不做 softmax，不强制类别 10 与其他类别概率互斥。
- 这是候选得分调整，未经额外验证不能称为可靠校准的疾病概率。
- 同一精确 n 的患者得到同一类的相同偏置，组内排序原则上不变；整体 AUC 变化来自不同 n 之间的重排。数值裁剪可能引入极值平局，应统计被裁剪数。
- 所有样本统一平移 logit 不改善 AUC；本方案的效果必须通过跨 n 排序变化验证。
- 后处理仅执行一次，顺序为“逐图→study 聚合→融合→次数偏置→指标/提交”。不能先给每个成员加偏置再融合。
- 保留原预测，输出为新文件；支持同时保存 raw 与 adjusted，方便逐 study 对比。

## 4. 数据契约与产物

### 4.1 计数范围

| 阶段 | 计数来源 | 禁止混入 |
| --- | --- | --- |
| train 离线相关性分析 | 官方 train.csv | valid/test |
| train 外折/内折评估 | 当前折所属患者在官方 train 中的全部 study；患者不拆折，因此与该患者完整折内计数一致 | valid/test、图片行数 |
| valid 复核 | 官方 valid.csv | train/test |
| 正式 test（仅许可后） | 许可指定的完整 test 元数据范围 | 耗时样本子目录局部计数、其他划分 |

不读取其他 study 的预测或标签作为本 study 的特征；不使用 Subject_id 的数字大小、字符串编码、排序位置；元数据全程字符串读取，保留前导零。标签只进入诊断和评分，绝不进入计数函数。

### 4.2 输入输出 schema

- 元数据输入：UTF-8 CSV，至少 `Subject_id,Study_id`；允许同 study 的多图行，去重后计数。空 ID 或同一 Study_id 对应不同 Subject_id 立即报错。
- 计数输出 `patient_counts.csv`：`Subject_id,Study_id,patient_study_count`，每 study 一行。
- OOF 输入：`Subject_id,Study_id,fold,label_1..label_10,p1..p10`，每个官方 train study 恰好一次；另有 manifest 证明该预测来自未见该患者的模型。
- 调整输出：`adjusted_predictions.csv` 使用原 ID 顺序和 p1..p10，另存 `raw_predictions.csv`、`audit.csv`（计数、g、β、每类偏置、是否裁剪）。
- 任何预测表与计数表按 Study_id 一对一关联，禁止按行号拼接；重复、缺失、额外 ID、患者映射冲突均报错。

### 4.3 冻结参数文件

`patient_count_prior.json` 至少包含：

```json
{
  "schema_version": 1,
  "method": "signed_capped_log_count",
  "count_scope": "split_distinct_studies",
  "cap": 10,
  "beta": 0.0,
  "class_signs": [1, 1, 1, 1, 1, 1, 1, 1, 1, -1],
  "probability_eps": 0.000001,
  "selection_source": "official_train_oof",
  "deployment_status": "research_only"
}
```

实际导出时还必须写入：训练源 CSV、fold manifest、OOF 预测和基础模型配方的 SHA-256；选型脚本版本；候选网格；选定 β；seed；类别顺序；冻结时间。部署资格作为独立记录，附许可原文文件及哈希；手工改 `deployment_status` 不构成许可。

输出目录统一为 `outputs/<时间戳>_patient_count_prior_<stage>/`，研究结果不放入 Git。参数文件绑定基础模型/融合配方；不能把单模型上选出的 β 直接套到另一套融合上声称已经验证。

## 5. 统计分析、选型和停止条件

### 5.1 先确认观察的含义

分别输出 `<5`、`5–9`、`≥10`、`≥5`，以及精确 n 的样本量：患者数、study 数、每类阳性/阴性数、单 study 至少一种异常比例、单 study 平均异常标签数（只数 1–9）、类别 10 阳性率。另列患者累计异常种类，但不将其增长解释为单次异常风险增长。

同时报告 study 等权与患者等权描述统计，避免少数高次数患者支配解释；比赛主指标仍按 study 等权。所有趋势仅称相关性，不作因果解释。

### 5.2 两层证据流程

**低成本初筛：** 基础配方固定为 D-002-a 对应的 ConvNeXt-Tiny、seed 42、5 个固定 epoch、末轮 EMA，其他优化与增强参数沿用 `configs/exp_D002a.yaml`，保留原有 scheduler 参数并记录完整配置。五折患者划分 seed=2026；官方 valid 不进入模型训练、早停或 β 选择。5 epoch 是本计划冻结的预算选择，不是该后处理的最优轮数结论。

在已核验的五折 OOF 上扫描四个 β，报告逐外折和合并结果。若合并 10 类 ΔAUC≤0，或少于 4/5 折为正，停止，不增加网格或调封顶数。若有候选通过，仅进入严格复核，不能直接晋级正式提交。

**严格复核：** 五个外折分别完成下面的过程：

1. 外折患者留作最终评估；该折标签不参与 β 选择。
2. 在剩余 4/5 train 患者中再分三折，生成内层 OOF（同样固定 5 epoch、末轮 EMA）。
3. 仅按内层 OOF 的 10 类 Macro-AUC 从四个 β 中选择；差值≤1e-6 时取较小 β。所有候选均不优于 β=0 时选 0。
4. 外层影像模型在全部外层训练患者上按固定配方训练，使用内层选择的 β 预测外折；与同一个外层模型的 β=0 配对比较。

初筛的五个外层影像模型满足清单、配方和固定末轮条件时可复用；严格复核新增 15 个内层模型，总计 20 个基础模型训练，不是只训练 5 次。若模型产物已存在，先验真再决定复用。不得因现有分数好而重新选择外折划分或 epoch。

不能把“全局 OOF 的其他四折用于拟合、留一折评分”描述成严格无泄漏的二级验证：其他折的基础模型可能训练过留出折患者。本文用嵌套流程避免这一依赖。初筛的同批选 β 与评分亦明确属于探索结果。

### 5.3 指标与晋级规则

- 主结果：五个外折的 10 类 Macro-AUC 差值、均值与方向；合并外折 AUC 另列，不替代折内配对结果，因为外折可能使用不同 β。
- 同时报告逐类 AUC、class2–10 的 9 类诊断 AUC、class1 和 class10、Macro-F1@0.5、Brier score；Brier 只监测得分可靠性，不用于另外选 β。
- 分层报告 `<5`、`5–9`、`≥10`、`≥5` 的 AUC 和有效类别；与历史 ≥10 子集指标口径保持一致。单标签类别标记不可计算，禁止把有效类别均值冒充固定 10 类均值。
- 配对 bootstrap 2000 次、seed 2026，外折内按患者有放回抽样并携带全部 studies，计算折差后取均值；固定十类，任一所需类别缺正/负则该次无效。报告无效率；若超过 5%，不作可靠 CI 晋级结论。
- 严格复核晋级要求：平均 ΔAUC≥0.002、至少 4/5 外折为正、95% CI 下界>0；0.002 是事前规定的实际收益门槛，不是已有结果。条件不全则保持关闭，不通过增加搜索次数追分。
- bootstrap CI 只覆盖固定流程结果的患者抽样不确定性，不覆盖训练随机性、数据集选择或所有选型偏差。单 seed 结论不得称“跨 seed 稳定”。
- 通过严格复核后，只在全部 train OOF 上按同一网格确定最终 β，再在官方 valid 对既有 train-only 模型作一次固定配方复核；不因 valid 结果微调 β。valid 此前已参与过探索，称“患者不重叠的复核”，不称完全未触及的盲测。
- valid 10 类 ΔAUC≤0 时不部署；仅作否决，不改 β 重试。若全 train 固定 5 epoch 模型不存在，须按同配方训练，不能混用历史早停 best_ema 当成严格配对。
- 已有 D-010/A-FUSE-7 含 valid 训练数据，不能用其 valid 预测评估本后处理。任何新的融合应用必须为相同成员和融合方式重新生成 OOF 并执行上述参数流程。
- 不按测试集构成加权分数选参数，不以平台分数选择 β，不扫描 test 的输出后修改后处理。

## 6. 文件与模块职责

| 文件 | 类型 | 职责 |
| --- | --- | --- |
| `src/data/patient_counts.py` | 新增 | 仅用 ID 元数据生成去重计数、校验关联 |
| `src/infer/patient_count_prior.py` | 新增 | 参数解析、校验、向量/宽表共用的数学变换 |
| `src/run_patient_count_prior.py` | 新增 | 离线统计、OOF scan/select、固定参数 apply、报告与 manifest |
| `src/data/patient_folds.py` | 新增 | 官方 train 内外折 manifest、分折 CSV 生成与来源核验 |
| `src/run_patient_oof.py` | 新增 | 五外折及三内折编排、固定末轮 EMA 导出、OOF 对齐 |
| `configs/patient_count_prior.yaml` | 新增 | 独立研究配置，固定网格和评估规则，不继承模型 YAML |
| `src/run_eval.py` | 条件修改 | 接受显式参数文件，保留原始与调整后评估 |
| `src/run_infer.py`、`src/run_ensemble.py` | 许可后修改 | 最终概率后统一接入，test 执行资格校验 |
| `src/cost.py` | 许可后修改 | 共享变换、完整计数上下文、清单与计时 |
| `tests/test_patient_counts.py`、`tests/test_patient_count_prior.py` | 新增 | 计数、变换和污染输入的合成用例 |
| `tests/test_patient_oof.py`、`tests/test_patient_count_prior_cli.py` | 新增 | OOF 来源、选型隔离、各入口行为 |
| `docs/architecture.md`、`docs/cost.md`、`CHANGELOG.md` | 修改 | 实现落地时同步边界、接口与改动状态 |

不修改 `StudyRecord`、模型 forward、训练损失和训练采样默认值。第一期使用显式 CLI 侧车参数文件，不增加 `Config` 顶层字段，从而避免旧 checkpoint 与 resume 配置序列化的兼容改动。

## 7. 分任务实施

### 任务 1：计数与数值核心（CPU，默认关闭）

- [ ] 新建计数模块，实现以下接口；读取 CSV 用 `dtype=str, keep_default_na=False, encoding="utf-8"`，先验证 study→patient 唯一性，再去重并 groupby 患者计数。

| 接口 | 返回与实现约定 |
| --- | --- |
| `build_patient_counts(metadata: pd.DataFrame) -> pd.DataFrame` | 按第 4.2 节返回一 study 一行的三列宽表；`drop_duplicates(["Subject_id", "Study_id"])` 后按 Subject_id 做 `transform("size")` |
| `align_patient_counts(predictions: pd.DataFrame, counts: pd.DataFrame) -> np.ndarray` | 先比较 ID 集合，再使用 `set_index("Study_id").loc[predictions["Study_id"]]` 按原顺序返回整数计数；不依赖 pandas 默认排序 |

- [ ] 新建后处理模块，定义 `PatientCountPrior`（冻结 dataclass），字段对应第 4.3 节；纯函数接口如下。向量入口接收单 study 十类概率，宽表入口按 ID 关联后调用同一核心，不各写一份公式。

| 接口 | 返回与实现约定 |
| --- | --- |
| `load_prior(path: str | Path) -> PatientCountPrior` | JSON 读取后严格校验 schema、字段和哈希，实例化冻结 dataclass |
| `adjust_probabilities(probabilities: np.ndarray, counts: np.ndarray, prior: PatientCountPrior) -> np.ndarray` | 支持 `[10]` + 标量 count，以及 `[N,10]` + `[N]` counts，内部统一为二维后处理并恢复形状 |
| `apply_prior_frame(predictions: pd.DataFrame, counts: pd.DataFrame, prior: PatientCountPrior) -> pd.DataFrame` | 调用 align，抽取固定 p1..p10 顺序，执行数值核心后写入副本 |

核心变换在输入校验后的二维实现为：

```python
adjusted = probabilities.copy()
if prior.beta == 0:
    return adjusted
g = np.log1p(np.minimum(counts, prior.cap)) - np.log(2.0)
active = g != 0
if not active.any():
    return adjusted
p = np.clip(probabilities[active], prior.probability_eps,
            1.0 - prior.probability_eps)
z = np.log(p) - np.log1p(-p)
signs = np.asarray(prior.class_signs, dtype=np.float64)
shifted = z + prior.beta * g[active, None] * signs[None, :]
adjusted[active] = 1.0 / (1.0 + np.exp(-shifted))
return adjusted
```

- [ ] 拒绝 count<1、非整数 count、NaN/Inf、概率越界、类别顺序错误、beta 不在候选集合、cap/sign/eps 与 schema 不符；不使用静默默认值掩盖损坏参数。
- [ ] 编写合成用例：同 study 三图片只计一次；跨 patient 冲突失败；前导零保留；打乱表行仍按 ID 对齐；beta=0 包含 p=0/1 时逐值相等；n=1 不变；cap 生效；异常类升、class10 降；输入不被原地改写。
- [ ] 静态验收：所有异常条件均有明确分支；关闭路径不读计数文件；CLI 未提供参数文件时不改变当前输出路径。

### 任务 2：研究入口、报告与审计（不依赖 GPU）

- [ ] 新建独立 YAML，固定 cap=10、beta_grid=[0,0.05,0.10,0.20]、eps=1e-6、seed=2026、bootstrap=2000、min_delta=0.002、min_positive_folds=4、invalid_bootstrap_limit=0.05。
- [ ] 新建 CLI 子命令 `describe`、`scan`、`select`、`apply`。`describe` 只读带标签 train/valid；`scan` 读 train OOF 并输出探索表；`select` 读取内层 OOF/manifest，导出固定参数；`apply` 不接受标签输入和拟合选项。
- [ ] `--split test` + research_only 参数立即失败；study/subject/fold/hash 清单不符立即失败。输出目录存在非空结果时拒绝覆盖。
- [ ] 输出第 5 节统计表、每折/逐类指标、配对差值、无效重采样数、候选选择记录、raw/adjusted 概率、manifest。
- [ ] 覆盖合成用例：训练源是 test 时不能 select；标签缺失不能评分但允许纯 apply；置换 test 以外的无关 CSV 不影响训练计数；增删图片重复行不改变 study 计数。

### 任务 3：生成和核验患者级 OOF（GPU 运行另行授权）

- [ ] 创建 folds 模块：官方 train ID 排序后，以 GroupKFold 的患者分组和固定 shuffle/random_state 建五折；每个外层训练池另建三内折，seed=2026+outer_fold。存全部 study、patient、fold、label、CSV hash，禁止官方 valid/test 出现在训练来源。
- [ ] 分折生成原格式 train/valid CSV，图像路径仍使用官方 train images 根目录；配置 `resplit.enabled=false`、`full_train=false`、epochs=5、early_stop_patience=0、save_last=true、EMA 开启。分折 YAML 和 CSV 都置于 outputs；不更改官方源文件。
- [ ] 编排器调用既有训练入口；训练可计算折验证日志，但不得使用其 best.pt/best_ema.pt 做 OOF。每个 run 结束核验 `last.pt` 已到第 5 epoch，并从 checkpoint 的 `extras["ema_state"]["model"]` 导出专用 `fold_final_ema.pt`，用 `common.checkpoint.save_checkpoint` 保存完整配置与来源哈希。该路径已按当前 `src/run_train.py::_save_last_checkpoint` 的 extras 保存逻辑核对；实现时继续确认 epoch 字段的起算方式。
- [ ] 只允许从通用 ImageNet 权重初始化，不能从已见外折患者的 D-002/D-010 权重 warm start；验证导出的 EMA updates、epoch、fold 与训练患者集合。
- [ ] 使用 `run_eval.py --pred-out` 的导出格式生成外折预测，编排器补充 Subject_id/fold；内折流程相同。训练预测不能冒充 OOF。
- [ ] 支持续跑时按 manifest/配置/checkpoint hash 复用已完成折；不完整 run 不进入拼接。每 study 恰好一次，患者不跨折，fold 的训练患者与预测患者交集必须为空。
- [ ] 合成用例检查：患者跨折、重复 OOF、缺 study、旧 manifest、使用 best checkpoint 替代末轮、warm start 已见患者均失败。
- [ ] 先授权执行一个外折，记录实际耗时、显存与磁盘，然后据此估算剩余四外折和十五内折；不套用历史单次训练耗时直接承诺总预算。

### 任务 4：按规则完成离线初筛和严格复核

- [ ] 执行任务 2 的 describe，区分累计异常种类与单 study 阳性率；图表若需要另行制作，核心交付为机器可读 CSV/JSON。
- [ ] 在五外折 OOF 上扫描四个 β，满足初筛条件才执行三内折复核；不满足则输出停止原因并保持 beta=0。
- [ ] 严格复核逐外折选 β 并在外折评分，按第 5.3 节统计 bootstrap 与晋级条件。
- [ ] 通过后在全部 train OOF 上冻结 β，给同配方 train-only 模型做一次官方 valid 复核；未通过则结束，不追加参数搜索。
- [ ] 实验记录注明模型配方、seed、折划分、末轮 EMA、所有候选、选型与评分范围、成本和限制。填写 `docs/training_log.md`/`docs/experiments.md` 时同步 `~/Desktop/loker_exp.xlsx`，按既有 D 系列研究记录方式归档；正式实验编号在执行前核对当前矩阵的下一个空位，不占用其他任务已有编号。

### 任务 5：评估入口接入与冻结配方

- [ ] `run_eval.py` 新增 `--patient-count-prior`，只支持显式指定匹配的固定参数；从当前 loader.records 构建计数表，与其预测来源一致，不额外加载其他划分。
- [ ] 同一次评估分别打印 raw/adjusted 指标；`--pred-out` 保留原语义，新增 `--adjusted-pred-out` 输出调整结果。仅指定调整输出而未提供参数时失败。
- [ ] 记录基础 checkpoint/融合配方指纹，拒绝配方不匹配；冻结 β 后不随 raw/EMA、seed 或融合成员自动迁移。
- [ ] 若基础模型最终合并 train+valid 重训，须先冻结后处理和训练配方，再固定轮数重训一次；合并后不利用 valid 重选 β、轮数或 checkpoint。记录 OOF 与最终全量模型分数尺度可能变化的迁移限制。

### 任务 6：许可与收益条件满足后的正式推理接入

- [ ] 核验许可原文、允许的 count_scope 与最终 β/配方清单；条件不满足时正式入口拒绝启用，任务状态记“条件未满足”，离线模块仍可完整交付。
- [ ] `run_infer.py` 和 `run_ensemble.py` 新增显式 `--patient-count-prior`；test 数据只做冻结函数 apply，禁止 scan/select。融合先完成后再统一调整一次。
- [ ] 复用参数解析与计数核心，提交保持完整十类概率与 UTF-8；不对 AUC 提交追加阈值过滤。
- [ ] `cost.py` 支持同一参数文件，并在 submission 同目录写 `cost_patient_count_prior.json`，保存 prior hash、许可记录 hash、完整计数文件 hash、count_scope、基础融合清单 hash。必要配套文件计入 ≤500MB 清单。
- [ ] 初始化阶段只加载主办方许可的完整计数上下文，计时 study 每次调用共享核心做 lookup+偏置，计入端到端时间；禁止从 pictures_100/pictures_5 重新估计次数，也不能用提交概率反推偏置。
- [ ] 计数文件缺失、scope/hash 不符立即失败，禁止退回 n=1 或仅统计子目录。初始化时间与额外配套文件大小写入报告。
- [ ] 在相同输入精度下确认批量预测、单 study 路径和 cost 输出一致；现场 GPU 浮点容差沿用现有规则，同时记录实测最大差值。最终耗时报告引用 `docs/cost.md` 当前答疑的 study 口径，不通过除以图片数声称另一种口径达标。
- [ ] 同步 `docs/architecture.md`、`docs/cost.md`、提交清单与 CHANGELOG；提供不带参数文件的回退命令，并保留原始 submission。

## 8. 拟议命令与验证方式

以下新命令均在实现完成后才可用；本次不执行。

```bash
# 元数据趋势，必须显式给出来源，不自动读取 test。
python src/run_patient_count_prior.py describe --metadata data/train.csv --split train --out outputs/count_prior_train_description

# 生成官方 train 的五折计划；执行 GPU 作业须另有授权。
python src/run_patient_oof.py prepare --config configs/exp_D002a.yaml --folds 5 --inner-folds 3 --seed 2026 --epochs 5 --out outputs/count_prior_oof
python src/run_patient_oof.py run --manifest outputs/count_prior_oof/manifest.json --stage outer --fold 0

# 先初筛，后按通过条件安排 inner 作业。
python src/run_patient_count_prior.py scan --config configs/patient_count_prior.yaml --oof outputs/count_prior_oof/outer_oof.csv --manifest outputs/count_prior_oof/manifest.json --out outputs/count_prior_screen
python src/run_patient_oof.py run --manifest outputs/count_prior_oof/manifest.json --stage inner --outer-fold 0

# 第一个外折只使用它自己的 inner OOF 选择参数。
python src/run_patient_count_prior.py select --config configs/patient_count_prior.yaml --oof outputs/count_prior_oof/outer_0/inner_oof.csv --manifest outputs/count_prior_oof/outer_0/inner_manifest.json --out outputs/count_prior_outer_0_selection
```

CLI 的输出路径规则在实现时固定为以上约定；其余外折通过 `--fold 1..4` 顺序执行。最终离线研究报告由 CLI `scan` 的 nested 模式读取五个外折的冻结参数记录和外折概率生成，显式命名选项为 `--nested-selections-dir`，不得默认使用全 OOF 最优 β 代替逐外折 β。

代码实现后的默认检查是阅读 diff、检查数据流、数值边界、来源约束及文档同步。以下属于需要用户明确授权后才执行的验证：

```bash
python -m pytest tests/test_patient_counts.py tests/test_patient_count_prior.py tests/test_patient_oof.py tests/test_patient_count_prior_cli.py
```

测试场景除任务中列明内容外，还包括：单 study 与宽表变换一致；关闭功能时原输出相同；同一模型融合后只调整一次；正式 test 未许可时失败；完整计数上下文与计时子目录解耦；修改外折标签不会改变内层选择的 β。真实 OOF、valid 评估、端到端耗时分别记录，不能用单元测试替代实际结果。

## 9. 交付与回退标准

1. **工程交付：** 元数据、变换、离线 CLI、OOF 清单接口和文档可审查；默认关闭，原始概率可复现。未执行验证的项逐一标注。
2. **研究交付：** 初筛/嵌套复核的原始预测、参数选择、差值、CI、成本与失败原因全部落盘；允许结论为“无收益”，此时结束该方案。
3. **部署交付：** 仅在许可、收益门、配方一致性及耗时复测均满足时完成；否则保持研究状态，不声称“已可参赛”。
4. **回退：** 去掉 `--patient-count-prior` 即回到原推理；不覆盖原始权重、预测或已有提交。不改变训练采样，避免与 D-009 混淆。

执行前检查当前 Git 分支与未提交文件；代码如需提交，切至 develop 或以 develop 为基准的分支并妥善保留既有改动，不在 main 或当前不符合提交约定的分支直接 commit。本次制定计划不自动 commit、push 或运行实验。
