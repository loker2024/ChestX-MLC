# 患者检查次数后处理

## 当前状态

已提供自动计数、固定 logit 偏置、历史预测探索、离线描述/OOF 扫描/选参/应用、患者级嵌套 OOF 编排，以及单模型和融合的 train/valid 评估接入。2026-11-08 已运行新增模块测试（40项通过）及 D-002-a-EMA 的一次历史 valid CPU 探索（0.87s）；最高点 ΔAUC仅+0.000200，未证明稳定收益。没有运行真实 OOF 训练或 GPU 耗时测试，也未冻结 beta 参数。

正式 test 的跨检查计数尚未取得主办方书面许可。`run_infer.py` 和 `run_ensemble.py --split test` 显式指定 `--patient-count-prior` 时会在加载数据或模型前报错；普通 test 推理不自动启用本模块。`cost.py` 尚未接入该后处理，不能用于对调整后的提交作达标声明。

完整实验设计与晋级条件见 [实施计划](superpowers/plans/2026-11-08-patient-count-prior.md)。以下命令为使用说明；本次仅执行历史预测 explore 示例，其余训练、OOF 与正式评估命令未执行。

## 自动计数和变换

推理程序由当前完整 split 的 `StudyRecord` 自动统计每个患者的不同 Study_id 数，在影像前向前完成。离线工具直接扫描 `--metadata` 指定的 CSV；不读取影像或标签来计算次数，不要求用户事先生成计数文件。程序只在内存中保存映射，输出的计数 CSV 用于审计。

一次检查的多张图片只计一次。计算不依赖 batch 大小或输入顺序，不与其他 split 合并，也不使用患者 ID 的数值作为特征。

```text
读取当前完整元数据 → 自动计数
影像前向 → 同次检查聚合 → 可选多模型融合
→ 对最终概率调整一次 → 指标/研究输出
```

固定函数为：`g(n) = log(1 + min(n, 10)) - log(2)`，异常类 1–9 加 `beta*g(n)` 的 logit 偏置，类别 10 减同等偏置。候选 beta 为 `0 / 0.05 / 0.10 / 0.20`。beta=0 或 n=1 时保留原概率；其余行的 logit 转换使用 1e-6 的概率裁剪。只改变得分，不强制标签互斥、不改变阈值、不保证概率已校准。

## 只分析元数据

```bash
python src/run_patient_count_prior.py describe \
  --metadata data/train.csv --split train \
  --out outputs/count_prior_train_description
```

输出包括：按次数分层/精确次数的 study 与患者等权统计、逐类正负例和阳性率、单 study 的异常比例/标签数、患者累计异常种类。累计异常种类单独输出，避免将观察次数增加解释为单次风险提高。所有研究输出目录要求为空，不覆盖已有结果。

## 无卡探索历史 valid 预测

已有检查级概率时，可以直接运行 `explore`。只读取 CSV 并进行 CPU 数值计算，不加载权重、不解码图片，不要求 OOF 清单、训练配置或事先生成的参数 JSON。

```bash
python src/run_patient_count_prior.py explore \
  --predictions outputs/predictions/valid/D-002-a-EMA.csv \
  --metadata data/valid.csv \
  --out outputs/count_prior_explore_D002a_EMA
```

默认 `--split valid`，也支持显式 `--split train` 作训练集探索；test 拒绝。预测表至少含 `Study_id,p1..p10`。可以带 Subject_id 和完整十类标签：存在时必须与元数据一致；标签省略时从元数据按 Study_id 对齐补充，部分标签列会报错。预测须完整覆盖这份元数据的 study 集合，重复/缺失/额外 study、患者归属冲突及非法概率均报错。计数自动从完整元数据生成，不依赖预测行顺序。

四个固定 beta 分别应用到同一份原始概率，不串行叠加偏置。输出包括：

- `summary.csv`：每个 beta 的原始/调整后 Macro-AUC、F1@0.5、差值、9 类诊断 AUC 和 Brier。
- `report.json`：原始与逐候选的整体、逐类及次数分层指标、样本量、输入文件哈希和探索标记。某类缺正/负时，其 AUC 及固定十类均值记为 null，不静默换类别。
- `raw_predictions.csv`、`adjusted_beta_0.csv`、`adjusted_beta_0.05.csv`、`adjusted_beta_0.1.csv`、`adjusted_beta_0.2.csv`：保留原顺序的原始及调整后概率。
- `patient_counts.csv`、`per_class_beta_*.csv`、`beta_*/audit.csv`：自动计数、逐类 AUC 差值及偏置/裁剪记录。

这是同一 valid 上的候选比较，不生成 `patient_count_prior.json`，不自动认定最优参数，不证明独立泛化收益。严格参数选择仍走下面的 OOF 流程。`explore` 不默认执行 bootstrap。

## 准备与生成 OOF

```bash
python src/run_patient_oof.py prepare \
  --config configs/exp_D002a.yaml \
  --out outputs/count_prior_oof
```

`prepare` 只读官方 train 元数据，生成五外折、每个外层训练池的三内折 CSV/YAML 和来源 manifest，不启动训练。患者不拆折；官方 valid/test 不进入折训练。配置固定 seed 42、5 epoch、末轮 EMA、关闭早停、保存 last.pt，并要求成功加载 ImageNet 权重。输出及配置使用绝对路径，跨机器移动需重新准备并记录新清单。

以下 `run` 命令会启动训练与评估，执行须有用户明确授权：

```bash
# 先只训练第一个外折，用实测成本评估后续预算。
python src/run_patient_oof.py run \
  --manifest outputs/count_prior_oof/manifest.json --stage outer --fold 0

# 其余外折在预算允许后执行；不指定 --fold 表示执行/复用该层全部折。
python src/run_patient_oof.py run \
  --manifest outputs/count_prior_oof/manifest.json --stage outer
```

每折导出 `fixed_last_ema.pt`，从第 5 轮 `last.pt` 的 EMA 状态得到；不使用外折指标挑选的 best checkpoint。完成全部外折后生成 `outer_oof.csv`。折级 `provenance.json` 绑定训练/评估患者集合、来源清单、配置、原始预测、last 与 EMA 权重哈希。已有可核验产物可以复用；中断续训使用原 run，不猜测多个 run 目录的归属。

## 初筛、严格复核和冻结参数

```bash
python src/run_patient_count_prior.py scan \
  --config configs/patient_count_prior.yaml \
  --oof outputs/count_prior_oof/outer_oof.csv \
  --manifest outputs/count_prior_oof/manifest.json \
  --out outputs/count_prior_screen
```

扫描与同批 OOF 评分明确标为探索。仅当整体 AUC 增加且至少 4/5 折增加时才进入嵌套复核；否则保持关闭，不扩大网格。

```bash
# 以下示例针对外折0的训练池。外折1..4分别使用对应 --outer-fold。
python src/run_patient_oof.py run \
  --manifest outputs/count_prior_oof/manifest.json --stage inner --outer-fold 0

python src/run_patient_count_prior.py select \
  --config configs/patient_count_prior.yaml \
  --oof outputs/count_prior_oof/outer_0/inner_oof.csv \
  --manifest outputs/count_prior_oof/outer_0/inner_manifest.json \
  --out outputs/count_prior_selections/outer_0
```

完成五个内层参数选择后：

```bash
python src/run_patient_count_prior.py scan \
  --config configs/patient_count_prior.yaml \
  --oof outputs/count_prior_oof/outer_oof.csv \
  --manifest outputs/count_prior_oof/manifest.json \
  --nested-selections-dir outputs/count_prior_selections \
  --out outputs/count_prior_nested
```

严格报告只将外折训练池的内层 OOF 参数应用到该外折；核验父清单、训练患者和内层 OOF 哈希。患者整簇 bootstrap 固定 2000 次、seed 2026；固定十类，不可计算的重采样记无效。晋级门为外折平均 ΔAUC≥0.002、至少 4/5 折正向、95% CI 下界>0，且无效重采样比例≤5%。CI 不覆盖训练随机性或所有选型偏差。五外折加十五内折共 20 个影像模型训练，另有全 train 固定配方模型的训练成本。

通过严格复核后才能在全部 train OOF 上执行 `select`，导出用于 valid 复核的冻结参数：

```bash
python src/run_patient_count_prior.py select \
  --config configs/patient_count_prior.yaml \
  --oof outputs/count_prior_oof/outer_oof.csv \
  --manifest outputs/count_prior_oof/manifest.json \
  --out outputs/count_prior_final
```

`select` 本身不证明已晋级，其结果永远标为 research_only。报告、原始概率、每个候选概率及计数/偏置/裁剪审计均保留。写正式实验结果时另行同步实验文档与仓库外台账。

## 与评估入口接入

`prepare` 另外生成 `train_only.yaml` 与 `train_only_eval.yaml`。前者只在官方 train 上训练，训练期 valid 槽指向相同 train，用于兼容现有循环；其中打印的指标属于训练内诊断，不作泛化结果、不作选轮依据。后者用于最后一次官方 valid 复核，模型和训练配方相同，仅评估数据不同。

```bash
# 显式授权训练后执行；固定5轮，不取 best_ema。
python src/run_train.py --config outputs/count_prior_oof/train_only.yaml --tag count_prior_train_only

# 将下面路径替换为此次训练实际产生的时间戳目录。
python src/run_patient_oof.py export-last-ema \
  --checkpoint outputs/count_prior_oof/train_only_runs/<run>/last.pt \
  --out outputs/count_prior_oof/train_only_final_ema.pt

python src/run_eval.py \
  --config outputs/count_prior_oof/train_only_eval.yaml \
  --checkpoint outputs/count_prior_oof/train_only_final_ema.pt \
  --patient-count-prior outputs/count_prior_final/patient_count_prior.json \
  --pred-out outputs/count_prior_valid_raw.csv \
  --adjusted-pred-out outputs/count_prior_valid_adjusted.csv
```

参数与实际 checkpoint 的模型/图像/训练配方、seed 和 EMA 类型必须匹配。历史早停模型或已并入 valid 的 D-010 不可替代上述配方。原始 `--pred-out` 语义保留；融合入口在 `train/valid` 下也支持该参数，并把调整结果写入输出目录的 `adjusted/`，只在融合后调整一次。融合配方需要它自己的 OOF 参数，单模型参数不能直接迁移。

纯离线 `apply` 支持无标签概率宽表，使用 `--metadata` 自动计数，并要求 `--recipe-manifest` 或 `--recipe-sha256` 与固定参数的配方匹配。它不接受标签列，不进行拟合，仍拒绝 test。原始预测和调整后预测分别落盘。

不提供参数时保持原行为。正式 test 和耗时路径的接入属于取得书面许可后才能执行的条件任务。
