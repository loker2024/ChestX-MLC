# 训练计划 29：A-FUSE-5 离线蒸馏初筛

状态：**蒸馏代码已实现，实验尚未训练**。本文件保留 D-006 的初始设计；首次实验现按 [Distill-1 计划](train_plan30.md) 的编号和配置执行，不启动 D-006。目标是在固定教师和训练预算下，检验蒸馏是否改善 ConvNeXt-Tiny 单模型，并比较其与 A-FUSE-5 的精度和 V100 耗时。本计划不预设收益，不改变当前提交方案。

## 一、实验设计

| 编号 | 配置基准 | 唯一实验变量 | 目的 |
| --- | --- | --- | --- |
| D-006-a | `configs/exp_D002a.yaml` 完整展开的独立配置 | 关闭蒸馏 | 同代码版本、同预算的学生对照 |
| D-006-b | 与 D-006-a 相同的独立配置 | 开启固定教师蒸馏，λ=0.5、T=1 | 测试蒸馏的增量 |

两组都从同一 ImageNet 预训练权重及新分类头开始，seed 42，320，batch 16 studies，lr 3e-4，warmup 2、cosine 8 轮、patience 2，BCE + 原 `pos_weight`，EMA 0.999，亮度/对比度 0.15，聚合采用 view_weighted 0.7。分别启用相同的 EMA 选模及验证口径，保留 raw 和 EMA 结果；比较主口径预先固定为各自 `best_ema.pt`。两个独立 YAML 除实验标识、蒸馏开关及教师缓存路径外必须逐字段一致。D-002-a 的历史 EMA 结果作为参考，不替代新代码版本下的配对基线。验证指标、早停和 scheduler 沿用既有 D-002-a 规则；不因蒸馏临时调整预算。

教师固定为 A-FUSE-5：D-002-a `best_ema.pt`、C-005-c `best.pt`、E-001 `best.pt`，每个模型按自身 checkpoint 配置与视角权重分别聚合，输出 study 级概率后，复用现有 `src/infer/ensemble.py` 的 **logit 4:2:3** 融合（含 `1e-6` 裁剪），得到每类教师概率。不可将三个模型的视角权重混同于融合权重。教师参数不更新；生成时 `eval()` 和 `inference_mode()`。

## 二、软标签与损失

只对原始训练集使用确定性的验证预处理生成教师软标签：每个 `Study_id` 一行、`p1`–`p10` 十列。valid 保留作早停和评估，test 不用于生成训练目标；不使用历史 valid/test 预测充当训练软标签。学生继续使用原训练随机亮度/对比度增强，因此学习的是**确定性教师预测对增强学生输入的监督**，不是两者相同增强视图；这属于本实验的明确设计，可能影响效果。

对 batch 中每个 study、每个类别，设学生聚合后的 logit 为 `z`、教师概率为 `q`、真实标签为 `y`：

```text
L_hard = BCEWithLogits(z, y, pos_weight=训练集原有类别权重).mean()
L_soft = BCEWithLogits(z, q, pos_weight=None).mean()
L_total = L_hard + 0.5 * L_soft
```

`q` 为 T=1 的 sigmoid 概率；这是逐类别 Bernoulli 软目标 BCE，不使用跨类别 softmax，不对软目标使用 `pos_weight`。软目标 BCE 与 `KL(Bernoulli(q) || Bernoulli(sigmoid(z)))` 对学生的梯度相同，但数值差教师熵，不能直接比较两种 loss 数值。两组 `L_hard`、总损失及蒸馏组 `L_soft` 分别记录；验证 loss 与选模规则保持原样。此处 λ=0.5 仅为初筛，不表示最优。

教师本身曾用训练集标签训练，训练集软标签是 **in-sample** 预测，可能包含过拟合；不能称作 OOF 教师。教师和融合方案也经过现有 valid 选型，学生最终 valid 结果仍有间接选型偏差。若初筛有价值，再考虑患者级 OOF 教师及独立验证，不在本次扩展。

## 三、执行前置检查与缓存契约

1. 在 GPU 训练环境确认三份实际 checkpoint 存在；本机 `outputs/experiments/` 中相应目录未发现权重，不能仅凭目录名启动。逐个记录绝对路径、SHA-256、raw/EMA 身份、嵌入配置、类别顺序、预处理、聚合、训练划分及来源；与 [cost.md §9.3](cost.md) 的导出包身份核对。缺失或不匹配则停止，不使用替代 checkpoint。
2. 固定 train/valid CSV 与图片索引、study 到患者映射，检查划分无患者交叉；计算 CSV 与 split manifest（如有）的 SHA-256，记录代码版本、依赖、设备和种子。生成教师缓存前确认数据与教师 checkpoint 的划分兼容，不能复用不同划分产物。
3. 对 train 使用无随机增强的 eval transform、每个 study 的全部图片及对应 ViewPosition；严格复用各教师预测和融合实现。记录每个成员的 checkpoint hash、展开配置、融合方法/权重、类别顺序、数据和视角索引 hash、代码版本和输出文件 SHA-256，写到缓存旁 `manifest.json`。缓存只存训练 study，不存 valid/test 输入及标签。
4. 校验 ID 为字符串并保留前导零；缓存与实际训练记录 study ID 集合完全相等，唯一且无缺失、重复或额外项；列序准确、概率有限且处于 [0,1]。校验缓存与当前 split、配置、checkpoint 和 manifest 一致；任何不一致直接报错，不能静默跳过或回退成普通训练。抽样逐 study 核对三个成员预测和融合值，并在 valid 上复算 A-FUSE-5，与现存预测及约 0.8171 的记录核对（容许数值误差，记录精确差异）；valid 复算仅作教师身份核验，不并入训练缓存。
5. 完整缓存原子写入并校验后再开始训练，避免读取半成品。训练按 `study_ids` 查找，而非依赖 DataLoader 顺序；续训须保存和核验缓存 SHA-256 与蒸馏配置，缺失或变化时拒绝续训。

## 四、拟实施工作与验证

以下是实现范围，不代表现有 CLI 已支持；实际命令应以实现后的 `--help` 为准，不提前虚构参数。

- 新增独立的教师缓存导出入口和缓存读取/校验模块，复用现有模型加载、study 预测和融合路径，不重写融合数学；支持 train split 和确定性预处理，写入 manifest。
- 在 `src/common/config.py` 增加默认关闭的蒸馏设置，在 `src/train/engine.py` 对聚合后的 logits 按 study ID 计算软损失；在 `src/run_train.py` 接通缓存、记录与续训校验。普通训练和旧 checkpoint 默认行为必须保持兼容；蒸馏初版只允许 BCE、原始 split，不支持 AUCM 或 resplit，遇到不支持组合明确报错。
- 新增 `configs/exp_D006a.yaml`、`configs/exp_D006b.yaml` 两份完整独立配置；实现后核对展开差异、训练与评估命令、配置和归档命名。
- 先写合成数据测试：多图 study 聚合和成员不同视角权重、logit 加权及极值裁剪；ID 前导零、乱序对齐、缺失/重复/额外 ID、列序、非有限/越界值；manifest/hash 不匹配和续训拒绝；软目标 BCE 手算值及学生梯度、教师不更新；禁用蒸馏时旧训练路径不变，以及缓存中断不产生可误用文件。覆盖配置校验及 CPU 小数据端到端训练/续训。
- 运行对应 focused pytest、相关训练/续训/推理测试及全量 `python -m pytest`，检查 YAML 展开差异和 `git diff --check`；随后在 GPU 上先做小批 smoke test（检查预训练加载、有限 loss、显存及缓存匹配），再串行训练两组。代码测试不代表 GPU 性能。

## 五、评估与决策

每组保存 `config.yaml`、`train.log`、`metrics.csv`、`best.pt`、`best_ema.pt`、`last.pt`、教师 manifest 和 train 缓存哈希；导出 valid 的 study ID、标签、概率与 `per_class.csv`。校验两组及教师在同一 valid 上 ID、标签、类别顺序一致。

1. 主要比较 D-006-b EMA − D-006-a EMA 的 valid study 级 10 类 Macro-AUC；报告 F1@0.5、阈值曲线、class2–class10 的诊断性 9 类 AUC、逐类 support 与 AUC，及 raw 结果。F1 的变化可能受校准影响。
2. 对主比较按 `Subject_id` 整簇配对 bootstrap，2,000 次、seed 2026；同一重采样用于双方，记录缺少正负样本的退化类别及有效次数。CI 跨零不等于两者等价；区间不覆盖训练随机性、教师选择或 valid 反复选型偏差。+0.010 作为优先复核参考线，不作为显著性门槛；即使未达到，也可结合置信区间及逐类结果讨论。
3. 与 A-FUSE-5 同一 valid 预测比较 AUC 和逐类差距，作为压缩取舍的描述，不宣称非劣。先比较两组，再与教师比较；若有潜力，再预先固定 seed 2026、其他配置不变复核，后续独立验证或 OOF 评估泛化。
4. 将选定学生生成匹配 checkpoint、精度与预处理的 submission；在现有 V100 上按 [cost.md](cost.md) 同一 100-study / 5-study、3 轮、fp32、workers=6 口径测 5 次并验证 submission 一致性，与 A-FUSE-5 的 78.69 ms 中位数比较。记录样本、GPU/CPU、代码版本与实际差异；环境或负载不一致时只作参考。无 T4 卡，**不报告 T4 达标**，最终以主办方环境复测为准。
5. 完成后按仓库规则同步 `docs/training_log.md`、`docs/experiments.md`、本计划执行记录及仓库外实验表格 C/D 系列归属：D-006 属 D 系列，现有表格没有 D 系列 sheet，先确认实际表格现有归属，不能擅自写入 C 系列；记录所有产物与失败项。未训练前不填写实验结果。

## 六、已实现的本地入口（历史 D-006 示例，不作为本次实验命令）

以下命令保留初始设计的 D-006 示例；实际运行请使用 [train_plan30.md](train_plan30.md) 中 Distill-1 命令。`src/run_distill_teacher.py` 固定接收三个 checkpoint，顺序为 D-002-a EMA、C-005-c、E-001；入口核对 [cost.md §9.3](cost.md) 记录的 MD5 前缀，使用各自内嵌配置和 train 的确定性预处理，输出完整缓存目录及清单。训练前仍须人工核对完整 SHA-256、历史训练划分、valid 融合复算与患者无泄漏；当前缓存校验的图片索引摘要覆盖 study/患者、文件名和视角，不对图片内容逐文件计算哈希。缓存不会覆盖已有目录。

```bash
# 三个变量填写实际 checkpoint 路径；禁止用其他模型替代。
python src/run_distill_teacher.py --checkpoint "$D002_EMA" "$C005_CHECKPOINT" "$E001_CHECKPOINT" --out outputs/distill/afuse5_train
python src/run_train.py --config configs/exp_D006a.yaml --seed 42 --tag D-006-a
python src/run_train.py --config configs/exp_D006b.yaml --seed 42 --tag D-006-b
```

蒸馏训练的 `metrics.csv` 额外保存 `train_hard_loss`、`train_soft_loss`，`train_loss` 是总损失；`last.pt` 元数据记录缓存与清单共同的摘要，续训时重新校验。两组命令应在缓存与权重身份、数据划分和预训练加载验证通过后执行。未获得 checkpoint、GPU 指标或目标设备耗时之前，不能宣称教师复算或蒸馏实验已完成。
