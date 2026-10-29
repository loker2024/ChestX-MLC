# A1开发与最小测试记录（2026-10-28）

## 代码范围

新增：src/augmentation.py、src/augmentation_preview.py、tests/test_light_aug.py、configs/experiments/A1_DenseNet121_320_Study_LightAug.json、docs/A1_A0_REFERENCE.json、docs/A1_LIGHTAUG_README.md及本报告。

修改：src/data.py、src/full_train.py、src/continue_train.py、src/training_state.py、src/pt1.py、src/run_experiment.py、src/experiments.py、src/study_smoke.py、README.md、CHANGELOG.md。model.py/evaluate.py/predict.py及旧配置没有改动。

## 实测

- 初次单模块调用被环境中第三方同名tests包抢占；已改用项目现有unittest discover及同目录测试导入，不修改系统环境。临时目录权限错误已改为项目内隔离目录后复测。
- 开发回归70项通过（106.095秒）；最终增强定向5项通过（47.645秒）；最终独立交付副本全套71项通过（102.792秒），含原ResNet18、TV/RA DenseNet、Study均衡数学和完整覆盖、早停/调度、Checkpoint绑定、增强开关、合成首轮及分段续训。
- 多worker合成图片：两个独立2-worker loader同seed有序图像张量一致；恢复整轮边界状态后下一轮增强张量一致，增强参数变更被拒绝；标签/Study不变。
- train/valid各4张真实图片与原B1交付data.py预处理逐位相同，原灰度复制RGB三个通道一致（Normalize之前）；无标签测试只使用合成fixture，没有读取官方test。
- 旧配置、E00归档及原始TV权重共24文件SHA检查未变。原始官方train/valid CSV与权重在真实冒烟前后哈希一致。
- 官方train/valid预检查通过：train30344图/18513 Study、valid1422图/859 Study，划分与标签指纹一致、无ID泄漏。只遍历完整采样索引验证覆盖，不运行完整官方Epoch；首轮预览2328步，最多16图，无Study截断或重复。

## 真实GPU冒烟

RTX4060 Laptop GPU，seed42，原始TV DenseNet121 ImageNet SHA绑定，604个骨干张量逐位核对、121个旧BN零计数器兼容，分类头新建10类。使用warmup第1轮LR（backbone1e-5/head1e-4），不是旧B1训练后权重。

- workers4：两个Batch13/11张图片，Loss .720045686/.705611289；另测最大16图；全部前后向/更新/保存重载通过。
- 最终配置workers8：1个完整8 Study Batch（13图）与额外最大16图Batch，Loss .720045686和.738119364；输出[13,10]/[16,10]；验证DataLoader输出[16,3,320,320]。
- backbone/head最大参数变化约.0000201464/.0002001952；重载推理逐位相同。
- 峰值allocated=4412358144 bytes、reserved=6031409152 bytes；参数6964106；小批Checkpoint28464145 bytes。
- workers8首次启动/读图86.20秒，不代表稳定每步耗时。workers4第二Batch等待约.00194秒，传输/训练.459秒。不能以这两个不同Batch的数字断言8比4快。
- batch1初测前向22.72ms、同一缓存train图片读取到JSON均值70.50ms；5次预热/10次测量，没有Study聚合，未在官方T4实测。

只检查少量完整valid Study的评价接口，部分疾病AUC未定义，未提供完整A1验证成绩。不能把合成集测试指标或小样本验证当作A1性能。

## 私有材料及公平比较

最终私有预览：outputs/A1_preview_final_20261028/preview.png，6张训练图片，含单图/多图Study和AP/PA/侧位，每张A0+4个A1采样。已目视检查，未见翻转或明显大面积裁剪，仅有预期小幅几何与强度变化；抽样检查不能保证所有未来图像的视觉质量。预览和患者明细不随ZIP分发。

完整日志：outputs/A1_all_tests_20261028.log、outputs/A1_final_target_tests_20261028.log、outputs/A1_delivery_check_20261028/unpacked_tests.log、outputs/A1_real_smoke_20261028.log、outputs/A1_workers8_smoke_20261028.log。完整小批结果见outputs/A1_LightAug_smoke_20261028_170447_75fd3437/smoke_result.json。

真实A0对照已核对：0.7954665375350549（Epoch6），workers8，其余实际实验参数与A1完全一致，只有编号和训练增强不同。权重许可沿用原指定TV文件的项目确认记录，不扩大到未知权重。建议队友复用A0原Python/Torch/CUDA环境；本地与服务器不同，不声称跨设备逐位复现。正式命令及应交回文件见A1_LIGHTAUG_README.md。
