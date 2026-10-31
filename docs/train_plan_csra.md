# CSRA 分类头对照实验

**状态：代码和本机合成数据验证已准备；尚未执行 GPU 训练或报告实验结果。**

## 目的与范围

以 `configs/exp_A001.yaml` 为基线，仅把分类头由 timm 原分类头替换为单头 CSRA；其余数据划分、ConvNeXt-Tiny `fb_in1k` 预训练、全量微调、320 × 320、正侧位 7:3 logits 聚合、BCE + pos_weight、训练预算与 seed 沿用 A-001。`configs/exp_csra.yaml` 仅显式覆盖 `model.head`。代码版本、timm 版本、预训练权重文件及 SHA-256 应在训练环境核对，保证与基线可比。

CSRA 用归一化类别权重在特征图上计算每类空间分数，输出每类的空间均值加 0.4 倍注意力加权分数；注意力是空间分数乘温度 1.0 后的 softmax。该版本为单头、无 bias；不是多头 CSRA。复用预训练 ConvNeXt head 的 LayerNorm 参数，但将其从全局池化之后移到空间特征图上逐位置应用，因而归一化位置和计算次序也随分类头一同变化；这属于整个分类头方案对照，不能仅把结果归因于注意力。新 CSRA 头随机初始化，骨干全量微调。训练和推理仍先按 study 聚合逐图 logits，再计算损失或 sigmoid。

## 运行及验证

```bash
python -m pytest tests/test_csra.py -q
python src/run_train.py --config configs/exp_csra.yaml --seed 42 --tag csra_convnext_tiny_s42
python src/run_eval.py --config configs/exp_csra.yaml --checkpoint outputs/<run>/best.pt --thresholds 0.3,0.4,0.5,0.6 --pred-out outputs/csra_valid_predictions.csv
python src/run_compare.py --runs outputs/<A-001_run> outputs/<csra_run> --thresholds 0.3,0.4,0.5,0.6,0.7
```

训练前检查相同 train/valid、配置展开、预训练来源和哈希；检查启动日志 `预训练权重: 已加载`，CSRA 路径权重加载失败时直接报错，不允许随机骨干回退。保留 `best.pt`、`last.pt`、`config.yaml`、`metrics.csv`、`train.log`、比较汇总、`per_class.csv`、逐 study ID、标签和概率；检查 ID、标签和类别顺序一致。评估前确认基线及候选使用相同验证数据。记录模型参数量、训练设备、峰值显存、耗时；目标 T4 端到端耗时需另行实测。

以 A-001 独立重算的 Macro-AUC 为参考，方向性门为 +0.005（若参考值为 0.8038 即 0.8088）；该门不是统计显著性检验。报告逐类 AUC、正负样本数、F1@0.5 和阈值曲线，必要时按 study 配对 bootstrap。若结果有希望，在相同预算下补 seed 2026 的基线与 CSRA 配对，不把 RUN008-C 当成同预算基线。单次结果不证明泛化或提交收益。
