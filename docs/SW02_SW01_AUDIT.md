# SW01本地资产审查

已完成记录：`outputs/X_SWINV2_BF16_run_20261103_094042`。
实际validation_mode：fixed_v2_internal；最大轮数12，实际12。
训练Study=15734，验证Study=2779。

## 已核实事实
- 该已完成实验使用v2内部划分，不是全量官方train/valid；数据指纹可与冻结v2核对。
- 普通/EMA独立选优，两者均patience=3才停；12轮达到预算，二者should_stop=false。
- 仅成功optimizer step更新EMA；BF16无Scaler；FP32 Loss/Study归约，尾组按实际Study数归一化。
- 调度按Epoch更新，先设LR，再optimizer，成功后EMA，轮末验证选优/早停。
- 本地未找到官方valid20轮已完成的SW01日志；官方20轮配置和交付包不等于训练结果。
- 新SW02按官方20轮配方准备，仅Loss不同；服务器必须交入真实SW01结果通过门禁才能开始。
- 不把v2的0.808079当作SW02官方valid对照，不对服务器未知结果宣称无泄漏。

## 最佳权重
- normal: Epoch 11, AUC 0.808056906848, SHA256 `4d58f0b711ccc3e6182ad7b4c8030f2c0641f07b39a01c70792a37d64c03545c`。
- ema: Epoch 12, AUC 0.808079066253, SHA256 `2640baa432a77ba34f8e0d1e6133c2b552de59e041749ebee6de34315a85775b`。

## 最后4轮（真实history）

epoch,variant,macro_auc,backbone_lr,head_lr
9,normal,0.8076298303096021,4.916040103221506e-06,4.201536739689893e-05
9,ema,0.8072749186281742,4.916040103221506e-06,4.201536739689893e-05
10,normal,0.8077306734196854,2.8143385534380006e-06,2.0002809059692745e-05
10,ema,0.8078778505927044,2.8143385534380006e-06,2.0002809059692745e-05
11,normal,0.8080569068476903,1.4649630951960416e-06,5.869876628632224e-06
11,ema,0.8080228018786043,1.4649630951960416e-06,5.869876628632224e-06
12,normal,0.8077827394892987,1e-06,1e-06
12,ema,0.8080790662526345,1e-06,1e-06


## 实现位置
- `src/run_swinv2.py`: train/precheck/smoke，普通与EMA独立保存。
- `src/swinv2_experiment.py`: accumulated_update、EMA、amp_components、rates。
- `src/sw02.py`: 逐标签ASL→类别均值→Study内图片均值→更新组Study均值；SW01实际配置/指纹/CheckpointSHA门禁。
- validation_image_bce继续FP32 BCE；train_study_asl不与BCE数值直接对比。