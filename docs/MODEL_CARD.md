# ENS-02来源与边界
R1/R2都是TorchVision DenseNet121，384 RGB letterbox /255 ImageNet Normalize，10 logits→Sigmoid→各自G0→0.5/0.5概率平均。无TTA、无元数据输入。
原始预训练：DenseNet121_Weights.IMAGENET1K_V1；https://download.pytorch.org/models/densenet121-a639ec97.pth
SHA256 a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395。
项目按用户明确确认及归档公开证据启用此权重；不冒称独立核验主办方书面答复。此包用于团队私有交接，不替代最终赛事完整许可材料。
两模型都仅使用固定v2内部训练9396患者/15734 Study/25790图；v2验证1659患者/2779 Study/4554图未参与训练。最佳Epoch均5，实际8轮。
v2：R1 AUC .802965，R2 .805944，ENS02 .809031。
官方valid：859 Study/228患者，R1 .786109，R2 .784789，ENS02 .790110；ENS02−R2 .005321，患者配对Bootstrap1000次95%CI [.000955,.010424]。
不能直接比较不同验证集分数；尚未验证与朋友A-FUSE-2集成收益。官方T4端到端100ms未验收，包内未提供正式cost.py；不是最终比赛提交包。
