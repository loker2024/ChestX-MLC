# 训练计划 15：ConvNeXt-Tiny 单头 CSRA 分类头对照

**状态：计划；本机功能测试已完成，GPU 训练与评估尚未执行。**

## 目标与实验设计

检验在 A-001 训练预算内，把原 timm 分类头换成 CSRA 分类头，是否改善 study 级 valid Macro-AUC。本次 CSRA 实验命名为 **A-005**；首轮直接训练一个 seed 42 的候选，以现有 A-001 结果和预测作为参考基线；不重跑标准头，也不同时尝试 ML-Decoder、多头 CSRA 或调整其他超参数。若参考基线的数据或代码口径无法核验，仍可报告候选结果，但差值只能作为历史参考，不能称为严格配对对照。

| 项目 | 基线 A-001 | 候选 CSRA |
| --- | --- | --- |
| 配置 | `configs/exp_A001.yaml` | `configs/exp_csra.yaml` |
| 分类头 | timm 原分类头 | 单头 CSRA，残差系数 0.4，温度 1.0，无 bias |
| 骨干 | 两组均为 `convnext_tiny.fb_in1k` 预训练，全量微调 | 相同 |
| 图像 | 320 × 320、letterbox、ImageNet 归一化、关闭水平翻转 | 相同 |
| 聚合 / 损失 | 正侧位组均值 0.7 / 0.3 聚合 logits；BCE + `pos_weight` | 相同 |
| 训练 | batch 16 studies，AdamW lr 0.0003、weight decay 0.01，warmup 2，cosine 至 0.00001 | 相同 |
| 预算 / seed | 最多 8 epoch，patience 2，seed 42 | 相同 |

CSRA 的空间分数使用类别权重的 L2 归一化：每类输出为该类空间均值加 0.4 倍空间注意力加权分数。复用 ConvNeXt 预训练 head 的 LayerNorm 参数，但原头在平均池化之后归一化，新头在池化前按位置归一化；同时类别投影形式和 bias 也改变。因此这是**分类头整体方案**的对照，不是只改变注意力的消融。两个头随机初始化方式也不同，固定 seed 不意味着同一初始分类头权重。

## 1. 训练前核对

1. 确认运行环境已包含 CSRA 代码及配置；记录实际 `git rev-parse HEAD`、`git status --short`、相关未提交 diff 和依赖版本。未提交代码不能仅用 HEAD 标识：需先固定并保存完整 diff 或版本快照。禁止把其他工作区改动混入实验而不记录。
2. 在仓库根目录运行：

   ```bash
   python -m pytest tests/test_csra.py -q
   python src/check_data.py --config configs/exp_csra.yaml --full
   ```

   检查测试通过、数据版本及 train/valid 的 Study_id、Subject_id、标签与图片来源和 A-001 一致，无患者泄漏；不要更改 `valid.csv`。实际检查结果和 CSV 哈希记录到实验日志。
3. 核对配置差异：

   ```bash
   PYTHONPATH=src python - <<'PY'
   from common.config import load_config
   baseline = load_config('configs/exp_A001.yaml').to_dict()
   candidate = load_config('configs/exp_csra.yaml').to_dict()
   assert baseline['model']['head'] == 'standard'
   assert candidate['model']['head'] == 'csra'
   candidate['model']['head'] = 'standard'
   assert candidate == baseline, '存在额外配置差异，请先核对'
   print('配置核验通过：仅分类头选择不同')
   PY
   ```

   对照 A-001 的 checkpoint 存档配置、数据指纹和预处理代码；只有配置相同不足以证明代码或数据相同。确认 CUDA 设备、torch / timm 版本、缓存的预训练权重及完整 SHA-256；A-001 记录为 `08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276`（见 [训练记录](training_log.md)），需对实际加载文件独立核验。若基线产物、数据或权重证据缺失，标记为不可严格配对，不能宣称严格单变量。
4. A-001 原记录目录为 `outputs/20261030-150749_A-001_convnext_tiny_s42/`；优先复用其 `best.pt` 和存档配置，以及本机已有逐 study 预测 `outputs/predictions/valid/A-001_valid_predictions.csv`。检查代码和数据语义是否可比；仅有同名路径或相同 study 数不足以证明数据一致。若 checkpoint 缺失，则使用已有评估记录及预测，在能核验 ID、标签与评估口径时做参考比较；证据不足时明确限制，不补跑标准头。

## 2. 训练

在已核验权重缓存的 GPU 训练环境中，独占 GPU 执行：

```bash
HF_HUB_OFFLINE=1 python src/run_train.py --config configs/exp_csra.yaml --seed 42 --tag A-005
```

检查启动日志 `预训练权重: 已加载` 和 `best.pt` 的 `metadata.pretrained_loaded == True`；CSRA 路径加载预训练失败会直接报错，应修复原因后重新运行。记录实际起止时间、设备、训练进程是否独占、代码及环境版本、权重 SHA-256、参数量、峰值显存、每轮指标、best epoch、实际停止原因。各实验不并行运行。

## 3. 同口径评估与产物

以下路径需替换为实际 run 目录。基线使用已有 A-001；若其 checkpoint 可用且数据、代码口径可核验，先重算 A-001 并与 CSRA 比较：

```bash
python src/run_eval.py --config outputs/<baseline>/config.yaml --checkpoint outputs/<baseline>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/standard_head_valid.csv
python src/run_eval.py --config outputs/<csra>/config.yaml --checkpoint outputs/<csra>/best.pt --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --pred-out outputs/A-005_valid.csv
python src/run_compare.py --runs outputs/<baseline> outputs/<csra> --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out outputs/compare_A001_A005
```

对两份逐 study 预测按 `Study_id` 对齐，确认 ID 唯一且集合一致、10 类标签逐项一致、类别顺序一致；数据版本不一致或无法对齐时停止配对判读，代码语义未能核验时仅作参考。保留已有 A-001 产物和 CSRA run 的 `config.yaml`、`train.log`、`metrics.csv`、`best.pt`、`last.pt`，以及可获得的两份预测 CSV、比较汇总和 `per_class.csv`。`run_compare.py` 优先采用 checkpoint 存档配置，但不会替代数据版本核验。

若 A-001 的 `best.pt` 不可用，跳过基线重算及依赖两个 checkpoint 的 `run_compare.py`；评估 CSRA 后，使用已有 A-001 预测文件核验并对齐 Study_id 与标签，再按相同指标口径比较。若预测或数据版本无法核验，只报告候选指标和 A-001 文档记录的参考值，不计算配对 bootstrap，也不声称严格对照。

## 4. 判读与下一步

- 主要指标：valid Macro-AUC；以已有 A-001 同口径结果为参考，方向性门设为**基线 Macro-AUC + 0.005**。历史记录值 0.8038 对应参考门 0.8088；若基线无法独立核验，门槛和差值仅供方向性参考。此门仅供筛选，不代表统计显著性。
- 同时报告每类 AUC、正负样本数、AUC 差、F1@0.5、0.2–0.7 阈值曲线、best epoch、训练 / 验证损失及耗时；阈值变化影响 F1，不改变连续分数的 AUC。检查整体提升是否伴随重要类别明显下降，不根据单 seed 单类别波动断言可靠改善。
- 如计算配对 bootstrap，以 study 为单位对齐后共同重采样，预先记录随机种子、次数及单类重采样缺少正负样本时的处理，并报告有效重复次数和区间；区间仅描述该验证集不确定性，不消除重复使用 valid 选型偏差。
- 首轮未过门或数据/代码证据不足：保留结果，不切换默认头；若有方向性改善但小于 0.005，结合逐类结果和不确定性再决定是否继续。过门且没有明显类别退化时，记录 seed 2026 复核建议，是否另行训练由后续实验决策确定。现有 RUN008-C 与 A-001 预算、早停不同，不能直接充当这组基线；两 seed 仍不能证明泛化能力。
- 只有成为提交候选后，按 [耗时自检说明](cost.md) 在赛题指定 T4 环境实测端到端耗时和提交一致性；本机测试不代表 GPU 指标或性能。
- 实验执行后如实回填 `docs/experiments.md`、`docs/training_log.md`、本计划和 `CHANGELOG.md`，记录所有已执行与未执行的验证。
