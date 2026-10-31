# ML-Decoder 思路分类头：实现计划与实际实现

**状态：代码已实现；GPU 训练未执行。**

## 目标和范围

在现有 `build_model` 增加可选 `ml_decoder`，旧配置默认标准头；限定 ConvNeXt-Tiny `fb_in1k`、全量微调、dropout 0。逐图输出 `[N, 10]` 原始 logits，训练及推理继续使用现有 study 级聚合。单层轻量变体用于实验，不声称复现论文或作者完整实现。

## 实现内容

1. `src/model/ml_decoder.py`：先复用 ConvNeXt 预训练分类头 LayerNorm 并逐位置归一化，768→256 投影，10 个可训练类别查询、4 个注意力头、单层 cross-attention、残差 LayerNorm、256→512→256 前馈层、每类独立线性投影与 bias；无 query 自注意力、无 group decoding，dropout 0。新头随机初始化。
2. `src/common/config.py`、`src/model/network.py`：添加 `head: ml_decoder`，加载完整预训练骨干后替换 head；加载失败停止，拒绝冻结骨干与非零 dropout。原标准头和 CSRA 路径保留。
3. `src/infer/predict.py` 已校验 `head`，checkpoint 保存的结构与推理配置不符时拒绝加载。训练和续训使用既有配置序列化与 checkpoint 路径。
4. `configs/exp_ml_decoder.yaml` 继承 A-001，只覆盖分类头；`tests/test_ml_decoder.py` 验证配置、输出、梯度、checkpoint 与错误配置，回归测试验证原路径。

## 设计边界

作者的 ML-Decoder 使用 query 与空间特征交叉注意力，并支持分组解码；本实现是借鉴思路的 10 类全解码变体，不包含分组解码或零样本机制，也不等同作者配置。网络层归一化位置由原 head 的池化后改为池化前，模型比较是整个头的对照。细节见 [原论文](https://arxiv.org/abs/2111.12933) 与 [作者代码](https://github.com/Alibaba-MIIL/ML_Decoder)。
