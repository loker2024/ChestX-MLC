# 更新日志（Changelog）

本文件记录 ChestX-MLC（多目标胸部病变检测分类）项目的全部改动。

约定：

- 每条改动标注到分钟，时区统一为 Asia/Shanghai，格式 `HH:MM`，日期作为分组标题。
- 排序规则：跨天时最新一天在最上；同一天内由早到晚，自上而下。
- 每条改动标注类型：新增 / 变更 / 修复 / 移除。

## 2026-11-03

- `09:31`（修复／变更）`src/swinv2_experiment.py`、`src/run_swinv2.py`、`tests/test_swinv2.py`、`configs/experiments/X_SWINV2_01.json`、`docs/X_SWINV2_01_README.md`：修复Inf梯度范数导致半截JSON及冒烟未强制成功更新；init_scale显式1024，冒烟最多8次backoff、训练连续8次跳步停止，EMA仅成功step更新。18项CPU测试通过；新增`outputs\X_SWINV2_AMP_fix_20261103_093153`报告及`outputs/X_SWINV2_AMP_fix_tests_20261103.log`。未GPU/训练或历史覆盖。

- `09:39`（修复／变更）`src/swinv2_experiment.py`、`src/run_swinv2.py`、`tests/test_swinv2.py`、`configs/experiments/X_SWINV2_01.json`、`docs/X_SWINV2_01_README.md`：指定BF16 AMP替代反复溢出的FP16，BF16禁用Scaler，FP32 BCE移出autocast，原生BF16支持门禁及非有限参数诊断；训练/冒烟同源。21项CPU测试及真实预训练单张train图三精度CPU对照完成，BF16/FP32全部梯度有限；首次诊断日志NaN导致写入失败已显式归档修复。新增`outputs\X_SWINV2_BF16_fix_20261103_093626`、`outputs/X_SWINV2_latest_BF16_fix.txt`及测试日志；旧配置归档，不自动GPU/训练/推理或改Batch/LR。

- `13:10`（新增）`src/continue_swinv2.py`、`tests/test_continue_swinv2.py`、`docs/X_SWINV2_EXTRA3_README.md`及`outputs/X_SWINV2_extra3_precheck_20261103/`/日志：用户接受AdamW重置，从原EMA Epoch12权重独立追加13—15轮，两组固定LR1e-6，保留BF16/v2/Study目标；原最佳参与选优，无改善则继续选原EMA。新增训练状态保存，明确非无缝resume；3项CPU测试及真实源Checkpoint/划分预检通过，未启动GPU训练/推理，原实验未改。


- `16:26`（新增／修复）`configs/experiments/X_SWINV2_01_server20.json`、`tests/test_swinv2_server20.py`、`docs/X_SWINV2_SERVER20_README.md`、`scripts/package_swinv2_server20.py`、`scripts/swinv2_server_paths.py`、`scripts/run_swinv2_server20.sh`：创建独立20轮、早停3、worker8服务器重训交接；`src/run_swinv2.py`训练上限改为读取配置、早停读取配置、权重路径来自五路径配置；`src/swinv2_experiment.py`保留12轮并支持20轮及worker校验。交付`deliveries/X_SWINV2_01_server20_v1/`及同名ZIP，含许可证据及哈希，不含数据/划分明细/模型；独立解压检查记录在`outputs/X_SWINV2_server20_delivery_check/`。不启动训练/GPU推理，不覆盖历史模型。

- `16:41`（新增／变更）`src/swinv2_official_data.py`、`configs/experiments/X_SWINV2_01_official20.json`、`tests/test_swinv2_official.py`、`scripts/run_swinv2_official20.sh`、`docs/X_SWINV2_OFFICIAL20_README.md`：新增完整官方train训练、官方valid选优的20轮独立配置（patience3/worker8/BF16/EMA）；`src/run_swinv2.py`按validation_mode路由数据及记录选优来源，官方valid导出核验同一训练边界；`src/swinv2_experiment.py`校验模式；`scripts/swinv2_server_paths.py`和`scripts/package_swinv2_server20.py`支持官方五路径及独立打包。新增`deliveries/X_SWINV2_01_official20_v1/`及ZIP、`outputs/X_SWINV2_official20_development_check/`验收记录，32项CPU测试及真实官方数据核验，不读取test、不训练/GPU或覆盖旧v2交付包。

- `19:42`（新增／变更）`src/sw02.py`、`configs/experiments/SW02_SwinV2_ASL_EMA.json`、`tests/test_sw02.py`、`scripts/run_sw02.sh`、`docs/SW02_README.md`、`docs/SW02_SW01_AUDIT.md`：SW02复用逐标签ASL及Study均衡，参数1/4/0.05，官方train/valid20轮对照；新增服务器SW01实际配置/划分/历史/最佳SHA门禁，缺结果不允许训练。`src/swinv2_experiment.py`、`src/run_swinv2.py`接入可选Study损失，分别标记train_study_asl/validation_image_bce；`scripts/package_swinv2_server20.py`支持独立SW02包。40项CPU测试通过，官方数据/原始权重CPU核验通过但SW01服务器对照缺失、GPU待本人冒烟。新增`deliveries/SW02_SwinV2_ASL_EMA_v1/`及ZIP、`outputs/SW02_development_check/`，无训练/GPU推理/test/Git或历史覆盖。

## 2026-11-02

- `10:22`（新增）`outputs/R2_official_valid_predictions/`：核验R2最佳Epoch5权重及384预处理，对官方valid1422图/859 Study执行FP32固定推理，G0平均生成8590行四列预测；保存单独真实标签、逐图/Study缓存、配置、SHA和报告。CSV回读/类别覆盖/源哈希保护通过，AUC0.7847891294，F1 0.3938648488。无训练/test/TTA或旧文件覆盖；官方valid分数不与v2直接比较。

- `10:40`（新增）`outputs/ENS02_official_valid/`：准备R1官方valid推理和固定集成脚本；R2缓存哈希清单检查发现缺失/不符，按用户要求在GPU初始化前停止，保存失败日志与资产检查报告。未完成推理/集成/Bootstrap，无训练/test/Git，未修改原缓存。

- `10:45`（新增／变更）`outputs/ENS02_official_valid/`：用户交回R2四列原文件且SHA与历史清单一致，保留原R2目录；继续R1 Epoch5官方valid1422图FP32推理，固定G0及50:50集成。保存8590行R1/ENS02预测、三方案十类AUC/F1、1000次患者配对Bootstrap、配置报告与源哈希；ENS02 AUC0.7901100716，较R2 +0.0053209422，95%CI[0.0009554383,0.0104243641]。旧中止报告/日志保留，未训练/test/TTA/权重搜索/Git或Checkpoint修改。

- `10:50`（新增）`deliveries/ENS02_R1_R2_384_handoff_v1/`及独立code/models ZIP：按configs/docs/src/tests组织最小固定推理交接，附中文说明、SHA及来源；两份最佳Checkpoint私下传递，不含原始影像/CSV或患者预测。3项单测、CPU严格权重加载、ZIP完整性/解压SHA检查通过；不训练、不GPU推理、不Git。另新增`outputs/ENS02_handoff_unpack_check_v1/`模拟解压副本。

- `19:45`（新增）`configs/experiments/X_SWINV2_01.draft.json`、`outputs/X_SWINV2_01_preflight_20261102/`及`checkpoints/pretrained/swin_v2_t-b137f0e2.pth`：归档官方枚举/recipe/许可/完整下载SHA，随机初始化384 CPU前后向通过；具体权重许可范围pending、Batch32单位待用户确认，按要求暂停正式工程推进，未冒称EMA/AMP/训练入口已完成。无训练/GPU推理/Git或历史覆盖。

- `20:02`（新增）`src/swinv2_experiment.py`、`src/run_swinv2.py`、`tests/test_swinv2.py`、`configs/experiments/X_SWINV2_01.json`、`configs/internal_ab/paths.swinv2.local.json`、`configs/swinv2_weight_review.json`、`docs/X_SWINV2_01_README.md`及`outputs/X_SWINV2_01_engineering_check_20261102/`/测试日志：实现约32图完整Study累积、AMP/EMA/裁剪、normal/EMA独立选优、官方valid及单图计时筛选入口。12项CPU测试及真实v2预检通过，原始权重严格CPU张量身份通过；许可仍pending、GPU AMP/显存/T4未实测，正式训练门禁保留。未训练、GPU推理、Git或历史覆盖；测速不冒充正式cost.py。

- `20:12`（新增／变更）`configs/swinv2_weight_review.json`、`configs/experiments/X_SWINV2_01.json`、`tests/test_swinv2.py`、`rules_compliance.md`、`docs/X_SWINV2_01_README.md`及`docs/pretrained/X_SWINV2_user_confirmation_20261102_201128/`：依据用户对指定Swin权重的明确许可确认登记approved，归档原pending记录，不冒称主办方书面核验；原门禁/权重SHA/证据SHA保留。3项许可门禁测试通过（approved接受、pending拒绝、证据篡改拒绝），仅CPU核验权重SHA，未GPU/训练。新增`outputs\X_SWINV2_license_confirm_20261102_201128`验收记录及`outputs/X_SWINV2_latest_license_check_path.txt`指针。


## 2026-11-01

- `23:12`（新增）`outputs/ENS02_R384_offline_20261101/`：R1/R2最佳Checkpoint只读CPU核验、逐图G0复算及固定等权ENS-02离线评价；保存四方案十类指标、患者配对Bootstrap各1000次、分组/互补性、源哈希与报告。ENS02 AUC=0.8090314204，无训练/GPU推理/test/Git或旧结果覆盖。

- `23:47`（新增）`outputs/ENS02_R1_R2_384_v2/`：按用户指定目录重新执行R1/R2最佳Epoch5缓存的50:50概率集成，逐图G0复核、2779 Study/1659患者对齐、1000次患者配对Bootstrap及CSV回读验收通过；保存指标、配置、预测、报告、日志与哈希清单。复现AUC0.8090314204/F1 0.3688653229；无训练、模型前向、GPU、test或原结果覆盖。


## 2026-10-31

- `12:52`（新增）`outputs/internal_subject_split_v1/`：以官方train/valid及seed42、15%患者留出生成完整8文件；9396/1659名患者、14461/4052次Study、23846/6498张图，三层ID无交叉且十类双侧正负覆盖。真实数据触发标签比例告警：第10类内部valid48.94%，原train36.10%；保留首次结果、不重选seed，不启动A/B训练，待结构性划分问题核验，不将该候选宣称为已认可的最终划分。原始CSV及P0—P8结果未修改，无Git或test访问。

- `13:05`（新增）`src/split_quality.py`、`configs/analysis/patient_holdout_v2.json`、`tests/test_split_v2.py`：固定结构分层配额+同层有界患者交换，兼顾十类正负Study/独立患者与图量、单/多图；生成前固定seed42、目标及验收阈值，只使用train标签结构，不搜模型AUC或多seed。
- `13:05`（变更）`src/patient_holdout.py`、`src/internal_split.py`：追加v2导出/质量报告与训练前质量复算门禁，保留v1接口和旧文件；不更改训练参数或其他训练入口。
- `13:05`（新增）`outputs/internal_subject_split_v2/`、`outputs/split_quality_v1_v2/`、`outputs/split_v2_*`及`outputs/internal_ab_v2_preflight_20261031.json`、`configs/internal_ab/paths.v2.local.json`、`docs/INTERNAL_SPLIT_V2.md`：真实v2全部预设质量检查通过，第10类valid36.0561%对源train36.0990%；同seed重放名单一致，原12文件SHA不变，5新增+8回归测试通过；A/B共用v2仅骨干LR不同，workers共同4，实际只预检无模型加载/训练/Git/test或ZIP。

- `16:54`（新增）`src/tta_hflip.py`、`tests/test_tta_hflip.py`、`docs/TTA_A_HFLIP.md`及`outputs/tta_hflip_*`、`outputs/TTA_A_hflip_v1/`：暂停前无TTA代码/结果，本轮按用户确认十类映射完成A固定内部v2的原图/水平翻转概率均值对照、G0、两级CSV和配对患者Bootstrap；4项合成测试通过，实际运行及结论由输出记录。无训练、官方valid/test选型、尺度或其他增强；不改旧模型、数据、训练/评价模块。

- `16:57`（新增）`docs/TTA_A_HFLIP_RESULTS.md`及TTA私有输出验收：真实4554图/2779 Study/1659患者，T0复现.798441，T1=.799489，配对差值95%CI[-.002197,.004392]跨0，不推广T1；11项模拟/聚合计分测试通过，原输入/权重/BN状态不变。本地batch1热缓存32图62.907/91.288ms，仅初测，不冒充T4；无训练或正式test访问。

- `17:14`（新增）`src/asl.py`、`src/run_asl.py`、`src/asl_comparison.py`、`tests/test_asl.py`、`configs/experiments/C1_ASL_DenseNet121_320_v2.json`、`docs/C1_ASL_EXPERIMENT.md`：固定ASL(1,4,.05,1e-8，标准detach focusing)，逐标签损失共享Study均衡归约；按A真实配置绑定同一v2与原始ImageNet初始化，默认只计划、显式执行单次C1。完成8项ASL及19项原采样/训练/保存/评价回归，不搜索参数、不生成ZIP。
- `17:14`（变更）`src/train.py`、`src/study_training.py`、`src/internal_ab.py`：默认BCE路径保持数值归约兼容，追加可选逐标签Loss与C1配置/初始张量门禁，ASL训练Loss和原验证BCE明确分名；最佳轮按内部Study Macro-AUC同步保存，原A/B及历史结果不修改。
- `17:14`（新增）`outputs/C1_asl_*`测试/计划、`outputs/C1_reference_preservation_before.json`及`outputs/C1_ASL_training_20261031.log`、`outputs/C1_ASL_DenseNet121_320_v2/`：按用户明确授权在本地CUDA启动一次完整C1候选，首批ASL有限，实际训练结果等待运行完成；保留原始输入/历史权重保护摘要，异常停止、不自动改超参数；不使用官方valid/test成绩调参。

- `17:29`（变更）C1运行状态：用户明确要求停止并仅交付开发/命令，已向原执行会话发送中断并确认退出（exit1），GPU总占用由约7.2GB降至828MiB；已保存首轮及第二轮部分日志，保留所有部分结果，不宣称完整训练或ASL胜负。后续只能由用户显式运行新目录命令，不自动恢复。

- `19:30`（新增）`docs/DATA_PREPROCESSING_AUDIT_20261031.md`及`outputs/preprocessing_audit_20261031/`只读审查脚本/统计/私有预览/日志：核验冻结v2三层隔离、全量train/valid图片头与路径、51图解码及32图0/4worker逐项一致、完整Study采样索引、A/C1缓存G0回算和Checkpoint元数据；10项目录发现测试通过，首次同名tests包导入失败如实保留。108个受保护文件哈希未变，无模型前向/训练/Git；test仅格式路径与ID边界，不统计分布、不读取预测。不改预处理、数据、历史结果；报告区分无现行必修错误与患者结构/填充/元数据缺失等风险，未将差异归因为榜单分数。

- `19:38`（新增）`outputs/E1_BCE_ASL_equal_v2/`：固定A/C1最佳Epoch5缓存的0.5/0.5 Study概率平均、三方案整体与十类指标、患者配对Bootstrap1000次、预测CSV、离线脚本/真实日志/来源哈希和简洁报告。2779Study/1659患者严格对齐，同冻结v2与类别顺序、根预测与最佳轮副本一致；行列打乱不变与CSV回读通过。集成AUC=.803762，相对ASL+.004173、95%CI[.001894,.006454]，7类升3类降，保留候选但不自动推广或搜权重；仅离线读CSV/JSON，未读Checkpoint或图片、未训练/模型推理/Git/officialvalid/test访问或正式提交，源文件未改。

- `19:52`（新增）`src/view_position.py`、`src/compare_view_position.py`、`tests/test_view_position.py`、`docs/VP01_README.md`及`outputs/VP01_unit_tests*_20261031.log`：独立VP-01正位AP/PA、侧位LATERAL/LL组内均值后固定7:3；单图/任一缺失或未知/缺任一组则整Study G0回退，保留全部图片正权重并审计。默认A单模型、冻结v2及最佳轮缓存门禁，离线两方案十类指标/患者配对Bootstrap/CSV回读与来源记录；8项合成+7项原聚合评价共15项通过2.215秒，CLI帮助通过。只开发与模拟测试，未读真实预测进行全量评价、未训练/完整推理/Checkpoint加载/Git/test访问或自动提交，不改旧聚合、模型、训练配置及历史实验；真实命令由用户手动执行。

- `20:07`（新增）`outputs/ENS01_VP01_v2_20261031/`：依用户授权完成A/C1最佳轮图片缓存的固定VP-01及0.5/0.5集成离线评价；先复现A=.798441、C1=.799589、ENS-G0=.803762，再复用原7:3组规则。4554图/2779Study/1659患者完整保留，1283Study加权、1496回退；ENS-VP01=.803785，差值+.00002248、患者配对1000次95%CI[-.001137,.001202]跨0，7类升3类降，保留ENS-G0、不搜索权重。六套预测CSV回读、先后线性等价、回退不变、输入哈希核验通过；新增私有报告/指标/覆盖/来源清单和一次性脚本日志，无Checkpoint读取、GPU启动、训练/推理/Git/officialvalid/test访问或提交，不改原规则与历史实验。

- `20:31`（新增）`src/metadata_fusion.py`、`src/metadata_training.py`、`src/run_metadata.py`、`src/evaluate_metadata.py`：M1独立两字段训练词表/MISSING与UNK、按Dicom_id连接、DenseNet1024+元数据32→16的Concat十维logits、头/骨干正确LR分组、原Study采样/BCE/G0及早停选优、映射/配置/预处理绑定Checkpoint、两级缓存和训练后患者配对/缺失与视角分组。默认只显示计划，正式训练必须用户显式--execute；不改旧训练/模型/预处理模块。
- `20:31`（新增）`configs/experiments/M1_metadata_fusion_v1.json`、`tests/test_metadata_fusion.py`、`docs/M1_METADATA_README.md`、`outputs/M1_development_20261031/`及`outputs/M1_*20261031.log`：从A真实配置保持workers4及全部既有参数，只变编号/融合结构；26项CPU合成与回归通过15.464秒，M1专项12项复测通过12.650秒。CPU核准原始ImageNet604张量加载、320随机张量前向与保存重载通过；CPU合成反向/三分支更新与4worker对应关系通过，参数6,965,178（较A+1072），临时最小Checkpoint约28.46MB。元数据真实唯一值/缺失及固定v2隔离/全ID对应已核验，test只读表头；108旧文件哈希不变，未启动GPU/正式训练/真实全量推理/Git，正式M1输出目录未创建，未声称M1成绩或显存实测。

- `22:45`（新增）`src/resolution_experiments.py`、`src/run_resolution.py`、`tests/test_resolution.py`、`configs/experiments/R1_DenseNet121_384_BCE_v2.json`、`configs/experiments/R2_DenseNet121_384_ASL_v2.json`、`docs/R384_PREPARATION.md`及`outputs/R384_preparation_20261031/`：仅尺寸/编号不同的固定384准备，绑定原A/C1及v2、同原图目录和原始ImageNet/随机头初态；保持workers4、8Study/最多16图、FP32/原BN行为。27项不训练测试通过11.341秒，CPU真实首批训练/验证输入[11,3,384,384]/[16,3,384,384]；显存仅约5.87GiB allocated算术预估、未保证8GB可运行。stdin多worker失败日志保留，文件入口复测通过；未生成448配置或启动训练/GPU前向。
- `22:45`（变更）`src/internal_ab.py`：新增严格R1/R2配置与原参考检查、原图目录绑定及同初态门禁，Dataset/Checkpoint尺寸读取input_size；旧A/B/C1仍保持320及原参数，不静默改物理Batch或ASL。130旧文件保护审计中仅该授权源码变化，历史权重/配置/原数据/v2未变，无Git。
- `22:45`（新增）`outputs/frequency_initialization_audit_20261031/`：当前v2 A/C1/M1/ENS缓存按单多图、1/2—3/≥4患者频次及交叉分组，ENS高频740Study/132患者AUC=.745109；澄清旧.743695来自官方valid本地.782766重跑而非当前ENS/原最佳P5。三个弱类总体单图更弱、高频类别4仍弱，按阳性患者数标注不确定性；CPU独立重建原始ImageNet完整初态/共同features Hash与三份真实Checkpoint元数据核验一致。只读来源哈希未变，无训练/模型前向/GPU或test访问，不作分布差异因果推断。

## 2026-10-29

- `13:45`（新增）`src/analysis_common.py`、`src/diagnose_p5.py`、`src/patient_holdout.py`：缓存优先P5验证病例诊断、十类AUC/计数、单图/多图与患者检查数子组、患者聚类Bootstrap及未定义状态；可选服务器P5身份门禁/valid-only前向；一次固定患者级标签计数平衡划分、严格原train/valid指纹/边界、内部CSV与清单，排他目录与SHA溯源。无本地真实模型/GPU训练或test访问。
- `13:45`（新增）`configs/analysis/p5_diagnosis.json`、`configs/analysis/patient_holdout.json`、`tests/test_analysis_tools.py`、`requirements-analysis.txt`、`docs/P5_DIAGNOSTICS_HOLDOUT_README.md`、`docs/P5_DIAGNOSTICS_CODE_REVIEW.md`、`docs/P5_DIAGNOSTICS_TEST_RESULTS.md`：默认P5 SHA/1-4-8诊断与15%患者留出、模拟/回归18项通过；服务器三种输入及划分命令、私有回传/输出规范，明确旧全train权重不能独立评估内部集，旧训练入口本轮不接入内部模式。
- `13:45`（变更）`README.md`：新增P8诊断与内部验证准备文档入口，保留旧训练/推理/模型/评价及P0—P7结果。
- `13:45`（新增）`outputs/build_analysis_delivery_20261029.py`、`outputs/analysis_*`模拟日志/交付验收及`deliveries/P8_P5_Diagnostics_SubjectHoldout_v1/`、ZIP和check.json：独立纯代码交付，排除原始CSV/图片/权重/真实患者预测，不运行GPU、训练或正式submission。

- `13:51`（变更）`docs/P5_DIAGNOSTICS_TEST_RESULTS.md`及P8交付验收记录：实际独立解压副本18项通过6.588秒，40个旧核心源码/实验配置哈希未变；最终ZIP逐文件回读/排除数据及权重，保留内部划分尚未接入旧训练入口的明确限制。

- `19:35`（新增）`src/internal_split.py`、`src/internal_ab.py`、`src/compare_internal_ab.py`：冻结患者清单/原始CSV/标签和三层ID边界接入，复用原DenseNet/Study均衡BCE/调度/早停/G0；显式执行、同ImageNet与初始头、最佳轮模型/两级预测/配置绑定；配对患者Bootstrap、纯骨干LR差异及环境对照，不改变旧训练入口。
- `19:35`（新增）`configs/experiments/Internal_A_DenseNet121_320_Study.json`、`configs/experiments/Internal_B_DenseNet121_320_Study.json`、`configs/internal_ab/paths.example.json`、`tests/test_internal_ab.py`、`docs/INTERNAL_AB_README.md`、`docs/INTERNAL_AB_TEST_RESULTS.md`：A骨干1e-4、B3e-5，Head1e-3及其余一致、workers8；五路径待服务器填写；8项新增模拟测试及独立交付30项通过，明确无本地真实模型/GPU训练。
- `19:35`（新增）`outputs/internal_ab_*`测试日志/独立解压/旧文件SHA记录、`deliveries/Internal_AB_DenseNet121_320_Study_v1/`及ZIP/验收记录：独立交付保留原许可门禁和证据，不包含官方CSV/图片/模型/患者预测；40个旧核心源码及实验配置SHA不变，不执行Git或触碰P0—P8结果。

## 2026-10-28

- `17:06`（新增）`src/augmentation.py`、`src/augmentation_preview.py`、`configs/experiments/A1_DenseNet121_320_Study_LightAug.json`、`docs/A1_A0_REFERENCE.json`、`docs/A1_LIGHTAUG_README.md`、`tests/test_light_aug.py`：训练专用联合轻仿射和灰度强度、Normalize前在线增强、私有6图预览；核对交回A0真实配置（TV非RA，workers8）与指标，A1只变实验编号和增强；独立配置/交接与数学、种子、两进程恢复及合成全流程测试。
- `17:06`（变更）`src/data.py`、`src/full_train.py`、`src/continue_train.py`、`src/study_smoke.py`、`src/training_state.py`：训练Dataset显式增强，验证/推理保持旧确定性变换；保存/校验增强状态及有序非持久worker恢复，保留旧Checkpoint兼容。
- `17:06`（变更）`src/pt1.py`、`src/run_experiment.py`、`src/experiments.py`、`README.md`：A1入口/完整运行根目录及结果快照、同Checkpoint指标、增强配置/预览归档及summary参数；配置支持none关闭，源权重门禁不变。
- `17:06`（新增）`outputs/A1_*`测试日志/哈希/私有预览/小批模型及交付验收记录、`outputs/build_a1_delivery.py`、`deliveries/A1_DenseNet121_320_Study_LightAug_v1/`及同名ZIP：运行产物不含在源码提交中；ZIP排除官方图片/原CSV/权重/患者预测，独立解压验收；不执行Git或完整官方Epoch。最终实测结果写交付检查记录。

- `17:12`（新增）`docs/A1_TEST_RESULTS.md`及独立ZIP验收记录：最终交付副本71项通过102.792秒；实际A0参数与A1仅差编号/增强，workers8真实13图+16图测试通过，保存重载一致；私有预览目视检查、旧24文件哈希及实际train/valid旧预处理逐位核对通过，未进行完整训练或test读取。

- `17:34`（新增）`configs/experiments/B2_EfficientNetB2_320_Study.json`、`tests/test_b2.py`、`docs/B2_README.md`、`docs/B2_PRETRAIN_COMPLIANCE.md`、`docs/B2_TEST_RESULTS.md`及`docs/pretrained/B2_evidence_20261028_172420/`：TV EfficientNet-B2 ImageNet/320/Study均衡独立候选，固定A0实际workers8/FP32，权重完整SHA及原作者Apache许可/TV移植同源核查，四项骨干/参数组/加载/续训测试。
- `17:34`（变更）`src/model.py`、`src/pretrained.py`、`src/pt1.py`、`src/run_experiment.py`、`src/experiments.py`、`src/study_smoke.py`、`configs/pretrained_registry.json`、`pretrained_registry.csv`、`rules_compliance.md`、`README.md`：追加B2工厂、保留原生Dropout而只替换10类Linear、正确Head LR与严格检查点加载；独立入口/根目录结果、原始权重门禁与来源登记，复用Study/BCE/G0，无新增图像增强或AMP。
- `17:34`（新增）`outputs/prepare_b2_sources.py`、`outputs/build_b2_delivery.py`、`outputs/B2_*`本地证据及`deliveries/B2_EfficientNetB2_320_Study_v1/`与ZIP/验收：原始公开候选仅下载核查及小批测试；75项回归通过107.800秒，真实13/11/16图更新及保存重载通过；27旧文件SHA、原登记行不变。私有产物与两份原始权重不进入代码包；无Git、test读取或官方整轮训练。

- `17:38`（变更）`docs/B2_TEST_RESULTS.md`及B2交付包验收：独立目录75项通过106.574秒；ZIP回读/解压及缺权重门禁验收，公开权重单独提供、代码包不含数据/患者预测/模型。未开始正式训练。

## 2026-10-27

- `13:13`（变更）`src/data.py`、`configs/data.json`、`src/full_train.py`、`src/continue_train.py`、`src/pt1.py`：按用户要求默认num_workers=4，可显式设0；新训练/验证记录真实进程数，续训沿用检查点，旧实验不静默切换。仅并行读取，模型/损失/学习率/处理/Study聚合不变。
- `13:13`（修复）`src/training_state.py`：原生确定性ChestXrayDataset、非持久有序worker支持完整epoch边界状态恢复；保存与核对训练/验证worker数及恢复模式，旧单进程状态兼容，其他多进程Dataset/持久worker和改变进程数明确拒绝。不声称支持中途断点或未来随机增强。
- `13:13`（变更）`src/predict.py`、`src/audit_training_result.py`、`tests/test_data.py`、`tests/test_p0_framework.py`、`tests/test_pt1.py`：预测/历史审计及非并行目标测试显式0，避免默认值更改影响推理口径或回归开销。（新增）`tests/test_loader_workers.py`：真实4进程合成图片顺序/张量与0对照、参数更新和下一轮恢复逐位一致、worker数改变拒绝；全套58项通过51.054秒。
- `13:13`（新增）`docs/WORKERS4.md`（变更）`README.md`、`CHANGELOG.md`：记录4进程设置、0兼容、完整恢复边界、Windows启动与稳定吞吐量区别、已运行任务不自动切换。（新增，已忽略）`outputs/check_real_workers4_20261027.py`、`outputs/workers4_real_check_20261027.json`及同名log、`outputs/workers4_tests_20261027.log`：真实train16图0/4同序、图片和标签逐位一致、CSV不变，无完整训练或test读取。
- `13:13`（新增，已忽略）`outputs/build_workers4_delivery_20261027.py`、`outputs/finalize_workers4_delivery_20261027.py`、`outputs/workers4_delivery_check_20261027/*`、`deliveries/B1_DenseNet121_ImageNet_v2_workers4/`及同名ZIP和check.json：保留v1，创建新版独立代码/配置/测试/说明交付；独立解压运行测试，具体结果及逐文件SHA/排除权重数据检查见交付验收JSON。无Git、旧模型覆盖、原始数据修改或完整训练。

- `16:49`（新增）`src/study_training.py`（变更）`src/train.py`、`src/data.py`：训练Study完整组批，最多8Study/16图可配置、先图片10类BCE平均再Study内平均再Batch平均；单Study超上限拒绝截断，原图片级Loss默认保留。采样器独立RNG、只读预览长度、逐轮Study/图片覆盖与类别有效数量/Step报告；添加pin_memory透传，无新增强或类别权重。
- `16:49`（新增）`configs/experiments/B1_DenseNet121_320_Study.json`（变更）`src/pt1.py`、`src/full_train.py`、`src/continue_train.py`、`src/training_state.py`、`src/run_experiment.py`、`src/experiments.py`：独立320/Study实验配置和入口，保留旧配置；从同一原始ImageNet初始化，首轮/续训接同一均衡策略，验证仍逐图BCE和全量G0。冻结size320供预测，保留全局Epoch级2轮Warmup/Cosine和最高AUC选优；保存/核对Study采样RNG及Loss/上限签名，归档study_sampling_report并在汇总记录Study数/图片上限/总更新步数，避免冒称预算相同。
- `16:49`（新增）`tests/test_study_training.py`、`src/study_smoke.py`、`scripts/check_study320.py`：1图/3图Study手工Loss及梯度权重、320等比例补边、索引不重不漏/上限不拆Study、微型合成数据完整首轮/续训/恢复/最佳指标绑定/实验登记测试；真实官方数据/许可/指纹/无泄漏前检查和最多4个批次加上限批的GPU冒烟，禁止把小子集AUC当完整验证成绩。原模型工厂、G0评价及预训练许可检查复用，未修改原始CSV/图片或权重。
- `16:49`（新增）`docs/B1_320_STUDY_README.md`（变更）`README.md`、`CHANGELOG.md`：320+Study组合实验定义、规范配置字段、命令、包放置/安装、输出/回传及OOM处理；2328首轮Step预览相对旧1897约+22.7%，不单因素归因；旧B1-224 AUC0.785445为用户报告，未在本轮重测。许可继续按已有项目用户确认及来源材料，不虚构新主办方原件。
- `16:49`（新增，已忽略）`outputs/study320_plan_20261027.json`、`outputs/study320_unit_20261027.log`、`outputs/study320_all_tests_20261027.log`、`outputs/study320_real_smoke_20261027.log`、`outputs/study320_preflight_20261027.json`、`outputs/B1_320_Study_smoke_20261027_163802_cf37fa8e/*`：4项新测通过9.581秒、全套62项通过66.340秒；真实ImageNet604源张量一致，3批13/11/15图加16图上限，Loss0.722819/0.710506/0.684390及0.711641，更新和重载一致；验证实际[16,3,320,320]，仅9Study子集接口，1—4类AUC未定义。RTX4060峰值分配4412833280/保留6622806016字节、检查点28463633字节、batch1前向17.98ms/读图至JSON64.00ms（非T4），首批等待含Windows启动与并行CPU测试，不冒称稳定吞吐量。
- `16:49`（新增，已忽略）`outputs/build_study320_delivery_20261027.py`、`outputs/finalize_study320_delivery_20261027.py`、`outputs/study320_delivery_check_20261027/*`、`deliveries/B1_DenseNet121_320_Study_v1/`及同名ZIP/check.json：独立交付代码/配置/测试/中文说明，候选解压62项通过56.709秒；包内无权重/官方数据/患者预测，最终文件清单、原实验/旧配置完整性、路径与ZIP逐文件SHA验收见check.json。无Git、完整官方训练或测试集调参。

- `19:21`（新增）`configs/experiments/B1_DenseNet121_RA_320_Study.json`、`requirements-ra.txt`（变更）`src/pretrained.py`、`src/model.py`、`src/train.py`、`src/pt1.py`、`src/full_train.py`、`src/training_state.py`、`src/run_experiment.py`、`src/experiments.py`、`src/study_smoke.py`：独立原生timm densenet121.ra_in1k后端/固定RA身份/安全本地safetensors加载，先1000类严格匹配和725骨干张量核对，再替换10类头；冻结后端、版本、配方与源SHA，检查点/恢复后端冲突拒绝，不回退TV/随机/A1。保留320、Study均衡BCE、LR/调度/早停/G0，默认worker8，其余旧配置不变；采样/评价复用原模块，冒烟增加启动与逐步日志。
- `19:21`（新增）`docs/B1_RA_COMPLIANCE.md`、`docs/B1_RA_README.md`、`docs/pretrained/B1_RA_evidence_20261027_185950/`、`docs/pretrained/B1_RA_evidence_20261027_190011/`（变更）`configs/pretrained_registry.json`、`pretrained_registry.csv`、`rules_compliance.md`、`README.md`、`CHANGELOG.md`：保留失败/成功来源记录，具体作者模型卡Apache-2.0及许可全文、ImageNet-1K/RA来源、非gated与固定revision、实际文件SHA匹配HF LFS，独立登记RA approved，旧两行登记不改。不冒称主办方新增签章；记录再分发/修改说明义务、官方预处理默认与本次沿用letterbox/bilinear区别。
- `19:21`（新增）`tests/test_ra.py`：4项合成fixture专项，验证RA725张量严格匹配/320输出、无TV/随机静默回退、A1替换拒绝/许可前阻断、后端保存重载和篡改拒绝、微型Study训练/续训/最佳指标绑定/跨后端恢复拒绝；专项4项通过13.797秒，全套66项通过71.157秒，未把fixture当真实许可或模型成绩。
- `19:21`（新增，已忽略）`.venv_ra/`（变更）`.gitignore`：项目内隔离安装timm1.0.30、safetensors0.8.0等依赖，复用原Torch，不改全局环境；首次ensurepip失败后修复项目venv系统包可见设置，真实加载和测试通过。pip check仍发现继承基础环境未使用的d2l旧版本固定依赖冲突，日志如实保留，不为消除告警升级/降级原框架。
- `19:21`（新增，已忽略）`checkpoints/pretrained/densenet121.ra_in1k-92007b62.safetensors`、`outputs/review_ra_sources_20261027.py`、`outputs/download_ra_weight_20261027.py`、`outputs/register_ra_weight_20261027.py`、`outputs/B1_RA_current_paths.json`、`outputs/B1_RA_review_20261027_*/*`、`outputs/B1_RA_dependencies_install_20261027.log`、`outputs/B1_RA_pip_check_20261027.log`、`outputs/B1_RA_unit_tests_20261027.log`、`outputs/B1_RA_all_tests_20261027.log`、`outputs/B1_RA_real_smoke_20261027.log`、`outputs/B1_RA_320_Study_smoke_20261027_191133_d8c0c7a2/*`：实际公开下载32334434字节、完整SHA eae900d8ae5b3716efa7f443fcc692aeb5a7f01f72581835c03c8c495f9eda10；真实3批13/11/15图Loss0.736085/0.715067/0.705781和16图上限0.722383、更新/重载通过。RTX4060分配/保留峰值4412833280/6622806016字节、检查点28472657字节、前向19.68ms/读图至JSON72.28ms（非T4），首次8worker启动约130.55秒，不冒称稳定I/O延迟或8更快。未完整训练/完整验证/使用test。
- `19:21`（新增，已忽略）`outputs/build_ra_delivery_20261027.py`、`outputs/finalize_ra_delivery_20261027.py`、`outputs/B1_RA_delivery_check_20261027/*`、`deliveries/B1_DenseNet121_RA_320_Study_v1/`及同名ZIP/check.json：包含必要代码/配置/测试/附加依赖/中文README和许可材料，排除环境/权重/官方数据/患者预测；独立解压及实际路径前检查、旧登记/配置/E00与原始权重完整性、逐文件ZIP SHA检查见交付验收JSON。不执行Git、不覆盖旧实验或直接运行完整训练，不声称跨设备复现。

## 2026-10-26

- `10:31`（新增）`src/audit_training_result.py`：只读核对history最优轮次、模型结构/内嵌配置、保存预测全量Study指标及类别AUC、图片/Study概率对应、源目录哈希；可选8图前向，超过容差明确告警，不伪称全量或逐位复现，不改写历史交付。
- `10:31`（新增）`src/lr_schedule.py`（变更）`src/full_train.py`、`src/continue_train.py`、`src/training_state.py`：全局epoch级Warmup+Cosine，保留两组LR比例，记录每轮LR、调度配置和位置，完整恢复时拒绝修改预算或伪装权重恢复；新指标绑定checkpoint_sha256，按实际最高Macro-AUC保存最佳及对应预测，旧固定LR/状态保持兼容。
- `10:31`（新增）`configs/experiments/PT2_ES.json`、`configs/experiments/PT2_WC_ES.json`（变更）`src/pt1.py`、`src/run_experiment.py`：独立12轮上限候选、patience3/min_delta0.0001，分别只早停及2轮Warmup+Cosine+早停；复用原首轮/续训/登记，保持224/BCE/G0/seed42/数据划分/无增强，PT1原配置不变，默认仅检查，未启动正式训练。
- `10:31`（新增）`tests/test_training_strategy.py`：3项策略测试，含合成4图ResNet真实更新、分段恢复参数逐位一致、LR端点/比例、入口早停传递、保存指标与模型SHA/篡改拒绝；修复审计调用遗漏labeled参数后全套46项通过，最终19.853秒。
- `10:31`（新增）`docs/PT1_ROUND2_ANALYSIS.md`、`docs/PT2_TRAINING_STRATEGY.md`（变更）`README.md`、`CHANGELOG.md`：读取用户指定Desktop结果，确认内嵌最佳epoch4、AUC0.7713936918/F10.2734281983/Loss0.3276828578和10类AUC；记录RTX3090原环境、无增强/固定分组LR、许可登记依据及跨环境抽样差异，不将训练成功等同官方许可核准；提供正式命令与交接清单。
- `10:31`（新增）`outputs/PT1_round2_audit_20261026.json`、`outputs/PT1_round2_audit_cpu_20261026.json`、`outputs/PT2_additional_checks_20261026.json`、`outputs/PT2_ES_plan_20261026.json`、`outputs/PT2_WC_ES_plan_20261026.json`、`outputs/PT2_strategy_tests_20261026*.log`、`outputs/PT2_final_integrity_20261026.json`（已忽略）：真实1422图/859Study保存CSV复算一致，BCE概率重建差约3.6e-9；8图GPU/CPU前向最大概率差0.000761/0.000562，超过1e-4容差，保留原环境待复核告警；历史早停回放第6轮停止且保留第4轮。原交付全文件、PT1配置/登记及E00归档27文件哈希不变，无Git、官方完整训练或test调参。

- `11:06`（新增）`deliveries/PT2_training_v1/`：完整73文件独立交付，保留`src/*.py`24模块、`tests/test_*.py`14测试模块、`configs/`8份配置及证据相对结构；源码和原配置57项源复制文件逐字节核对，完整路径及摘要见该目录`FILE_LIST.txt`、`SOURCE_COPY_MANIFEST.json`和`MANIFEST.json`，不含官方图片/原始CSV/患者预测/权重/环境/无关历史实验结果。
- `11:06`（新增）`deliveries/PT2_training_v1/README.md`、`requirements.txt`、`requirements-local-tested.txt`、`scripts/check_pt2_delivery.py`、`configs/pt2_reference.json`、`docs/PT2_IMPLEMENTATION.md`、交付版`rules_compliance.md`、`pretrain_compliance_report.md`、`CHANGELOG.md`、`DELIVERY_CHECK.md`、`DELIVERY_CHECK.json`：中文安装/路径/两套训练命令/选优产物说明；增加包内只读前检查，固定seed42/224/BCE/G0及相同划分，比较固定LR+早停和Warmup+Cosine+相同早停；明确用户确认依据、不冒充主办方新授权或跨设备完全复现。
- `11:06`（新增）`deliveries/PT2_training_v1.zip`、`deliveries/PT2_training_v1_check.json`、`outputs/PT2_delivery_check_20261026/*`（测试输出已忽略）：候选ZIP独立解压后46项测试通过（20.876秒），缺输入正确拒绝，修改测试副本路径后真实train/valid只读前检查与两套权重门禁通过；最终73文件ZIP 644326字节、CRC和全部载荷SHA核对、最终再解压静态检查通过。初次venv ensurepip失败后使用独立venv共享已安装库，直接依赖及21包依赖闭包通过；全局pip check仍有旧d2l的7项约束冲突，未修改全局环境，不将其宣称纯净联网安装成功。无Git、模型/官方数据修改、正式训练或test选型。

- `20:13`（变更）`src/model.py`、`src/train.py`、`src/pretrained.py`：模型工厂加入DenseNet121/原生classifier10类头，默认ResNet18与旧权重兼容，分类头按参数身份分组、保存真实architecture；本地1000类核准权重严格匹配后替头，保留官方旧Dense键名迁移及冲突检查；逐骨干限定枚举/URL，补SHA的Python3.10兼容分支，不下载或绕过候选许可。
- `20:13`（变更）`src/pt1.py`、`src/full_train.py`、`src/run_experiment.py`、`src/training_state.py`、`src/audit_training_result.py`：复用既有Warmup/Cosine、早停、G0及训练/恢复/预测链路，接入B1骨干和权重身份、保存/审计头形状与跨骨干恢复拒绝；单批接口支持显式测试批量，旧默认不变。
- `20:13`（修复）`src/experiments.py`：集成测试发现旧登记/汇总仅支持PT1等旧编号，增加已定义PT2/B1白名单与汇总发现；保留旧exp_*损坏目录告警隔离及路径安全校验，兼容新配置loss字段，不重建或覆盖本地历史汇总。
- `20:13`（新增）`src/b1_smoke.py`、`tests/test_b1.py`、`configs/experiments/B1_DenseNet121.json`：224/BCE/AdamW、两组基础LR1e-4/1e-3、warmup2+cosine、12轮上限、patience3/min_delta0.0001、seed42、batch16、无新增强；8项B1测试含合成权重严格加载/迁移、梯度/分组、共享预测/Study、恢复/审计/归档、pending阻断/编号隔离/SHA兼容，最终全套54项通过（30.524秒），测试目录隔离调整追加目标测试通过。
- `20:13`（变更）`configs/pretrained_registry.json`、`pretrained_registry.csv`（新增）`docs/pretrained/B1_evidence_20261026_194313_b791c02c/*`、`docs/B1_PRETRAIN_COMPLIANCE.md`：官方Dense候选实际无鉴权HTTP200下载32342954字节，完整SHA a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395；保留8份原始响应、官方模型页web核对笔记/urllib403事实、来源和下载记录共11文件；具体使用/再分发许可范围待核实，Dense条目pending，原ResNet条目不变，未反序列化真实候选。
- `20:13`（新增）`docs/B1_README.md`（变更）`README.md`、`rules_compliance.md`、`CHANGELOG.md`：记录实际单批/参数量/初测口径、与ResNet区别、完整文件清单及许可通过后的条件训练命令；当前不具备真实预训练正式训练条件，不宣称官方T4/完整500MB材料验收。
- `20:13`（新增）`outputs/B1_review_20261026_194313_b791c02c/*`、`outputs/B1_current_review_path.txt`、`outputs/B1_DenseNet121_interface_smoke_20261026_195201_b873872f/*`、`outputs/B1_tests_20261026*.log`、`outputs/B1_isolated_registration_test_20261026.log`、`outputs/B1_interface_profile_20261026.log`、`outputs/B1_plan_20261026.json`、`outputs/B1_final_integrity_20261026.json`、`outputs/B1_development_manifest_20261026.json`（已忽略）：真实train随机初始化16图更新一步，Loss0.7292451262、输出[16,10]、重载一致；6964106参数、检查点28462417字节、RTX4060峰值分配2159319040/保留2583691264字节；batch1前向均值27.20ms/读图至JSON均值55.28ms（5预热/30次/同训练图缓存，不含Study聚合，非T4验收）。保留迭代失败日志；实际pending候选加载/正式训练入口阻断通过，旧E00模型重载及27文件SHA不变、旧实验配置不变，无Git/官方完整训练/test选型。

- `20:35`（变更）`src/model.py`、`src/pt1.py`、`src/full_train.py`：增加首次训练前逐张量骨干来源核对（分类头排除、旧BN零计数兼容），将核对结果与完整源SHA写入单步及正式检查点元数据；实际待核实候选仍在torch.load前阻断，无许可绕过。
- `20:35`（新增）`configs/experiments/B1_DenseNet121_ImageNet.json`（变更）`src/run_experiment.py`、`src/experiments.py`、`tests/test_b1.py`：独立ImageNet实验编号/登记/配置检查，沿用224、batch16、seed42、两组LR、2轮Warmup+Cosine、12轮上限、早停3/0.0001、BCE、G0和无增强；增加合成权重骨干一致/随机差异/篡改拒绝与真实门禁隔离测试。全套56项通过（33.044秒），未执行官方完整训练。
- `20:35`（新增）`docs/B1_IMAGENET_RECHECK.md`、`docs/B1_IMAGENET_README.md`、`docs/B1_WEIGHT_PERMISSION_INQUIRY.md`、`docs/pretrained/B1_ImageNet_evidence_20261026_202532_cfcc101c/`（9文件：6份来源原文、失败/重试清单、文件身份）（变更）`README.md`、`docs/B1_PRETRAIN_COMPLIANCE.md`、`rules_compliance.md`、`CHANGELOG.md`：保存官方/原作者来源链和许可证据，明确尚缺具体转换权重使用/再分发范围，保持pending_license；提供未发送询问稿、条件单步及正式训练命令，不将证据缺失认定为违规或复用ResNet18授权。
- `20:35`（新增）`checkpoints/pretrained/densenet121-a639ec97.pth`、`outputs/B1_ImageNet_current_paths.json`、`outputs/B1_ImageNet_recheck_20261026_202532_cfcc101c/*`、`outputs/B1_ImageNet_tests_20261026.log`（已忽略）：原始候选复制至配置路径，源文件保留、完整SHA一致、32342954字节；验收脚本/日志记录实际加载与正式入口在反序列化前阻断，旧配置和登记不变，E00归档27文件不变、旧ResNet18真实重载前向[1,10]。本轮真实ImageNet加载/16图更新/候选测速均未执行，无结果冒充，无Git、测试集选型或历史覆盖。

- `21:50`（新增）`docs/pretrained/B1_user_confirmation_20261026.md`（变更）`configs/pretrained_registry.json`、`pretrained_registry.csv`：按用户明确确认启用指定DenseNet121 ImageNet V1，保留证据/完整SHA门禁；记录为项目用户确认，不冒称已核验主办方原件或新增发布方授权，原始答复未入库和独立许可范围保留意见继续披露。
- `21:50`（新增）`docs/B1_HANDOFF.md`、`scripts/check_b1_delivery.py`（变更）`README.md`、`docs/B1_README.md`、`docs/B1_IMAGENET_README.md`、`docs/B1_IMAGENET_RECHECK.md`、`rules_compliance.md`、`CHANGELOG.md`：交接步骤、真实16图验收及历史状态区分；前检查核对train/valid完整数量/指纹、实际图像标签形状、登记证据和权重SHA，不读test。修复初次前检查导入不存在函数的问题，失败日志保留，重跑通过。
- `21:50`（新增）`outputs/enable_b1_user_confirmed_20261026.py`、`outputs/build_b1_delivery_20261026.py`、`outputs/finalize_b1_delivery_20261026.py`、`outputs/B1_user_confirmed_20261026_214309/*`、`outputs/B1_pretrained_smoke_20261026.log`、`outputs/B1_user_confirmed_tests_20261026.log`、`outputs/B1_handoff_preflight*_20261026.log`、`outputs/B1_DenseNet121_ImageNet_pretrained_smoke_20261026_214347_3eaeddc2/*`（已忽略）：真实指定ImageNet加载核对604骨干张量/121旧BN零计数兼容，官方train16图Loss0.7206231356、骨干及头更新、重载逐位一致；6964106参数/28463633字节检查点，本地RTX4060前向均值27.28ms/读图至JSON57.93ms（同训练图缓存、不含Study聚合、非T4验收）。本地56测试通过32.944秒，无完整Epoch、Git、test训练/调参或历史覆盖。
- `21:50`（新增）`deliveries/B1_DenseNet121_ImageNet_v1/`、`deliveries/B1_DenseNet121_ImageNet_v1.zip`、`deliveries/B1_DenseNet121_ImageNet_v1_check.json`、`outputs/B1_delivery_unpack_20261026/*`（产物/测试副本已忽略）：完整相对目录代码/配置/测试/来源规则材料、中文安装/路径/单步/正式命令/输出说明；原样复制清单及包内SHA清单供校验，不携带官方影像、原始CSV、患者预测、环境、模型权重或历史实验。独立解压56测试通过32.974秒，最终文件清单、缺输入阻断及路径调整后检查结果见交付验收JSON；不声称跨设备完全复现。

## 2026-10-25

- `09:43`（新增）`src/early_stopping.py`（变更）`src/continue_train.py`：验证Study级Macro-AUC早停，patience/min_delta显式配置、默认关闭；完整历史重建等待计数，保留真实最高AUC模型及最后轮完整状态，记录实际轮数/请求上限/停止原因，不根据单次Loss或F1下降误判过拟合。
- `09:43`（新增）`scripts/train_to15.sh`：Linux nohup预检查脚本，核对第8轮模型绑定状态，最多追加7轮至累计15轮，patience4/min_delta0.001，输出与日志使用唯一目录；未在本机执行正式训练或服务器脚本。
- `09:43`（新增）`tests/test_early_stopping.py`：5项早停测试，含CPU微型合成数据真实续训循环、mock验证分数、最佳模型/最后状态保存与停止后恢复；全套29项测试通过（13.285秒）。
- `09:43`（新增）`docs/EARLY_STOPPING.md`（变更）`README.md`、`CHANGELOG.md`：累计15轮的训练同学命令、预检查路径、早停规则、限制与7文件交付清单；第8轮0.750488来自同学报告，本机未重新验证该模型。
- `09:43`（新增）`outputs/early_stopping_tests_20261025.log`（已忽略）：保存29项测试输出；本轮无正式训练、Git操作、数据或旧模型修改。
- `11:26`（新增）`src/aggregation.py`、`src/compare_aggregation.py`：G0概率均值/G1概率最大/G2平均logits后sigmoid，优先原始logits、概率裁剪反变换回退、单图等价，完整验证CSV对照入口与独立目录原子发布，无模型加载或训练。
- `11:26`（变更）`src/submission.py`、`src/evaluate.py`、`src/fingerprints.py`：复用共享Study聚合、分离已聚合表的标签对齐计分、分开G0/G1/G2评估协议；旧调用默认G0，训练程序/结构/权重不变。
- `11:26`（新增）`tests/test_aggregation.py`：6项聚合/协议/边界/ID/真值来源/CSV门禁测试；全套35项通过（13.477秒），不伪造未定义类别AUC。
- `11:26`（新增）`docs/AGGREGATION_README.md`（变更）`README.md`、`CHANGELOG.md`：Windows命令、新模型逐图CSV交接、9文件清单及完整对照结果；不打包官方数据或真实ID预测，不创建压缩包。
- `11:26`（新增）`outputs/experiments/G_aggregation_v1/*`、`outputs/aggregation_tests_20261025.log`（已忽略）：1422图/859Study同源比较，G0 AUC0.730568169109816、G1 AUC0.7249389704628、G2 AUC0.7312862338995675，多图468Study；源E00为4轮模型，G2为概率反变换路径，E00归档27文件哈希保持不变，无正式测试集选型或G3搜索。
- `16:08`（新增）`rules_compliance.md`、`docs/rules/competition_rules_20261025.pdf`、`docs/rules/competition_rules_20261025.txt`：完整读取用户提供6页正式规则，归档同字节PDF与逐页原文、SHA256，记录预训练许可/再分发、验证边界、test冻结、TTA/集成、100+5图计时、500MB全量口径与材料清单；CSV语义/平台日次数/指定环境等未覆盖事项保留待核对。
- `16:08`（变更）`AGENTS.md`：加入后续任务必读规则记录与原文的入口，保留既有协作约束，不将旧摘要假设冒充正式规定；本轮只记录规则，无模型、训练、官方数据、Git或平台操作。
- `16:08`（新增）`outputs/rules_review_20261025/*`（已忽略）：保存6页渲染与联系图用于规则阅读核对；未修改用户原PDF。
- `19:42`（新增）`src/pretrained.py`、`configs/pretrained_registry.json`、`pretrained_registry.csv`：候选逐文件登记与本地加载门禁，记录官方V1/ImageNet-1K来源、本机缓存完整SHA256，许可证/再分发仍为pending，不自动下载、不以代码BSD代替权重许可。
- `19:42`（新增）`src/pt1.py`、`configs/experiments/PT1.json`（变更）`src/run_experiment.py`：PT1配置、默认只检查、随机初始化接口冒烟、需核准的真实预训练冒烟及条件完整训练入口，复用原训练/续训/实验登记，未执行官方完整Epoch。
- `19:42`（变更）`src/model.py`、`src/data.py`、`src/train.py`、`src/predict.py`：保留原随机初始化，核准本地1000类权重严格匹配后替换10类头，增加确定性ImageNet归一化与两组AdamW学习率，预测读取检查点冻结预处理，旧模式保持兼容。
- `19:42`（变更）`src/full_train.py`、`src/continue_train.py`、`src/training_state.py`、`src/experiments.py`：接入条件PT1、分组学习率与归一化的一致训练/恢复、核对head_lr、防覆盖PT1登记；原随机初始化恢复一致性测试继续通过。
- `19:42`（新增）`tests/test_pt1.py`：8项许可门禁、合成权重严格加载、灰度补边归一化、分组更新与恢复、共享预测一致性、PT1启动阻断测试；全套43项通过（16.066秒）。真实候选未用于训练，合成许可材料仅限单测。
- `19:42`（新增）`docs/pretrained/PT1_evidence/torchvision_CODE_LICENSE.txt`、`docs/pretrained/PT1_evidence/web_review.md`、`docs/pretrained/PT1_evidence/sources.json`、`docs/PT1_README.md`（变更）`README.md`、`rules_compliance.md`、`CHANGELOG.md`：保存代码许可全文/来源链接/审查事实、尚缺权重许可材料、22文件清单与队友待执行命令；如实记录网页socket失败与web核查方式，不伪造授权。
- `19:42`（新增）`outputs/PT1_interface_smoke_20261025_193132_234eeedd/*`、`outputs/PT1_tests_20261025*.log`、`outputs/PT1_plan_20261025.json`、`outputs/PT1_integrity_20261025.json`（已忽略）：RTX4060真实训练图片2张、随机初始化接口一步更新，Loss0.7750405073、输出[2,10]、重载一致；E00归档27文件与缓存哈希不变，旧baseline_v2源目录当前缺失，未删除/重建；无Git、正式test调参或官方完整训练。

- `20:17`（新增）`pretrain_compliance_report.md`、`docs/PT1_organizer_inquiry.md`：按正式规则重新区分已确认条款、证据不足与不符合规则；记录附录ResNet50示例、MIT/BSD材料差异、同一ResNet18文件的timm BSD标注，准备主办方询问稿，未代发送。
- `20:17`（新增）`docs/pretrained/PT1_recheck_evidence_20261025/torchvision_v0.26.0_LICENSE.txt`、`torchvision_v0.26.0_weight_notice.txt`、`torchvision_v0.26.0_resnet18_enum.txt`、`torchvision_v0.26.0_training_recipe.txt`、`timm_resnet18_model_card_excerpt.txt`、`timm_resnet18_source_excerpt.txt`、`retrieval_manifest.json`、`download_verification.json`（均在同一证据目录）：实际保存官方资料、同文件权重级许可支持、获取时间与响应/摘录摘要，以及无鉴权官方权重HTTP200和完整SHA记录。
- `20:17`（变更）`configs/pretrained_registry.json`、`pretrained_registry.csv`、`README.md`、`docs/PT1_README.md`、`rules_compliance.md`、`CHANGELOG.md`：更新实际下载事实及证据链接，保持pending_license，明确证据不足不等于禁止ImageNet，不伪造已核准或赛事/再分发授权；复用既有PT1实现和12轮配置，本轮无源码重复开发。
- `20:17`（新增）`outputs/pretrain_recheck_20261025_195736_b4e7f5ff/*`、`outputs/pretrain_recheck_download_20261025_195808_7b238f87/*`、`outputs/PT1_recheck_tests_20261025.log`、`outputs/PT1_interface_smoke_20261025_200205_a0b4b80f/*`、`outputs/PT1_recheck_plan_20261025.json`、`outputs/PT1_recheck_integrity_20261025.json`（已忽略）：分别保留初次网络失败及成功下载记录，46,830,571字节完整SHA与旧缓存一致；43项测试通过（17.092秒），真实训练图片随机初始化单批Loss0.7750405073、重载一致；真实候选未加载训练，E00归档哈希不变，无Git、正式训练或test调参。

- `20:33`（新增）`docs/pretrained/PT1_user_confirmation_20261025.md`、`docs/PT1_HANDOFF.md`（变更）`configs/pretrained_registry.json`、`pretrained_registry.csv`：按用户明确确认及同文件timm BSD公开证据启用指定ImageNet候选，记录用户确认依据，不冒充主办方书面授权；登记证据使用跨平台相对路径，交接说明含手动下载/SHA/单批/12轮训练命令、未启用早停、结果回传清单。
- `20:33`（变更）`src/pretrained.py`、`tests/test_pt1.py`：将用户确认依据写入模型溯源；pending门禁测试改用独立fixture，避免真实项目启用后测试错误依赖生产登记状态，保留权限/证据/完整SHA检查，不增加绕过开关。
- `20:33`（变更）`README.md`、`docs/PT1_README.md`、`pretrain_compliance_report.md`、`rules_compliance.md`、`CHANGELOG.md`：更新当前交接状态并保留前次待核实报告的历史意见；当前操作统一指向PT1_HANDOFF，不宣称新增官方许可答复。
- `20:33`（新增）`checkpoints/pretrained/resnet18-f37072fd.pth`、`outputs/PT1_pretrained_smoke_20261025_202909_c2c729f9/*`、`outputs/PT1_user_confirmed_tests_20261025.log`、`outputs/PT1_user_confirmed_verification_20261025.json`、`outputs/handoff/PT1_resnet18_imagenet_handoff_20261025.zip`及同目录交接清单（均已忽略）：同SHA候选复制到配置路径，真实预训练官方train两图单步Loss0.8107124567、骨干/头更新、重载一致；43测试通过（17.849秒）；逐张量检查骨干原权重一致、E00历史文件哈希不变，交接包排除数据/权重/患者预测/环境，未开始完整训练或Git操作。

## 2026-10-24

- `10:47`（变更）`src/data.py`：统一返回ViewPosition、PatientOrientation、Rows、Columns原始字符串及缺失标记；模型仍只使用图片，标签仅作目标。
- `10:47`（新增）`src/experiment_config.py`、`src/run_experiment.py`、`configs/experiments/E00_baseline.json`、`configs/experiments/E01_resnet18.json`：JSON配置校验与默认只检查的实验入口；显式--execute复用单轮/完整状态续训，未实现优化选项明确拒绝。
- `10:47`（新增）`src/prediction_outputs.py`（变更）`src/predict.py`、`src/evaluate.py`、`src/submission.py`：图片级预测回连元数据、CSV回读及ID归属校验；评估/聚合兼容丰富预测表，仍只按官方标签计分与平均概率聚合，按ID对齐避免行序误报。
- `10:47`（变更）`src/full_train.py`、`src/continue_train.py`：新配置精确控制尺寸、批量、学习率、衰减、种子，保留旧入口；验证可选保存图片级预测并随最优权重保存。本轮无正式训练。
- `10:47`（新增）`src/archive_baseline.py`（变更）`src/experiments.py`：E编号归档、附加文件哈希/原子快照、历史与当前环境区分、扩充汇总训练参数与主要改动，同协议组内并列AUC同排名。
- `10:47`（新增）`tests/test_p0_framework.py`（变更）`tests/test_predict.py`：配置/未实现选项、元数据、丰富CSV对齐、反向ID排序、mock执行调度、E00防覆盖与排名测试；原18项加新增6项，共24项测试通过（最终7.354秒）。
- `10:47`（新增）`docs/P0_EXPERIMENTS.md`（变更）`README.md`、`CHANGELOG.md`：P0交接、配置/输出/运行边界、18文件清单、历史第1轮后重置优化器的事实、真实验收与下一阶段聚合修改点。
- `10:47`（新增）`outputs/experiments/E00_baseline/*`、`outputs/experiments/exp_003/predictions/20261024_104015_4bc40f78/*`、`outputs/experiments/exp_003/predictions/20261024_104206_bba3e362/*`、`outputs/p0_checks/20261024_104300_495e0364/*`、`outputs/p0_unit_tests*.log`、`outputs/p0_config_*_check.json`（变更）`outputs/experiment_summary.csv`（已忽略）：保留首次排序校验失败日志及修复后成功结果；RTX4060对1422张/859次验证复现AUC0.730568169109816/F10.15771485793395476，各split真实抽样8张检查元数据，E00归档27文件哈希与来源/旧实验核验通过；未进行正式训练、Git操作或平台上传。

## 2026-10-22

- `09:52`（新增）`src/predict.py`、`src/submission.py`、`src/experiments.py`：统一验证/测试预测入口、检查级均值聚合、模拟提交与回读校验、独立实验快照和汇总表。
- `09:52`（新增）`tests/test_submission.py`、`tests/test_experiments.py`：验证预测ID聚合、前导零保留、坏概率/重复/缺失检测及实验防覆盖。
- `09:52`（变更）`src/continue_train.py`：增加来源、独立输出目录、追加轮数和学习率参数；默认行为兼容旧入口，本轮不执行训练。
- `09:52`（新增）`outputs/experiments/exp_001/*`、`outputs/experiments/exp_002/*`、`outputs/experiment_summary.csv`（已忽略）：注册两份历史实验快照，保留模型、配置、历史、验证预测/指标、备注及来源哈希。
- `09:52`（变更）`README.md`、`CHANGELOG.md`：记录统一预测、实验登记/比较方式及正式模板缺失的本地模拟限制。
- `09:55`（新增）`outputs/experiments/exp_002/predictions/20261022_095153_89283345/*`（已忽略）：真实验证/测试逐图和检查级 CSV、模拟 submission.csv、submission_check.json、validation_metrics.json、prediction_report.json、prediction_log.jsonl 与 csv_audit.json；验证成绩复现一致，测试2218张、1414次检查，14140行格式检查通过。
- `09:55`（变更）`README.md`、`src/continue_train.py`、`CHANGELOG.md`：记录8项测试通过、真实推理耗时、模型大小与来源哈希不变，完善通用续训说明；本轮无新训练和平台上传。
- `13:06`（修复）`src/submission.py`、`src/predict.py`：测试导出必须显式选择 --all-classes/--threshold，支持四列模板顺序，分别验证完整类别或阈值筛选及原始概率对应；复现差异默认告警，可显式严格阻断；检查点优先读取内嵌配置，支持旧 epoch_N/model.pth。
- `13:06`（新增）`src/devices.py`、`tests/test_predict.py`：统一 CUDA/MPS/CPU 选择与同步/统计，覆盖无旁路配置检查点的完整 CPU 预测、默认告警和严格失败，MPS 分支仅 mock 测试。
- `13:06`（修复）`src/full_train.py`、`src/train.py`、`src/continue_train.py`：全量/单步/续训支持 --device；非CUDA设备不调用CUDA专属统计；新逐轮检查点保存同目录配置。
- `13:06`（变更）`tests/test_submission.py`、`README.md`、`docs/PR_REVIEW_RESPONSE.md`、`CHANGELOG.md`：补齐两种提交语义、空输出/漏项/篡改概率回归测试、基线实验边界、PR描述和审查回应；历史提交未改写，官方语义仍待确认。
- `13:06`（新增）`outputs/review_checks_20261022_130325/*`、`outputs/baseline_v2/epoch_4/predictions/20261022_130319_4ed3e91b/*`（已忽略）：保存真实旧检查点全量验证复测和原真实测试概率两种导出校验结果；未重新训练或上传，未修改原始模型。
- `14:23`（变更）`develop`：撤回 PR #2（merge commit `91e9c3a`，未修复 P0-1 交付材料），删除其新增的 19 个文件（`src/`、`tests/`、`docs/PR_REVIEW_RESPONSE.md`），代码恢复到 `a8d2d47`；PR #2 在 GitHub 上仍显示 MERGED。
- `14:56`（修复）`CHANGELOG.md`：核对 develop 与 yun 分支后补回团队的14:23撤回记录，避免本地日志覆盖团队历史；本条修订尚需上传到yun分支。
- `16:07`（变更）`README.md`、`docs/PR_REVIEW_RESPONSE.md`：补充新克隆的依赖、测试、官方数据准备及默认 configs/data.json 用法；记录实际 PR #2 缺失根目录文档和配置的交付问题，明确本地文件不代表已提交到远端。
- `16:07`（变更）`CHANGELOG.md`：汇总本PR各轮新增源码、测试、配置和审查修订的路径记录；configs/data.json 的新增记录见2026-10-19 19:56，源码/测试路径覆盖原实现及2026-10-22修订；远端补传后须以 Files changed 核验。
- `18:21`（修复）`src/predict.py`：新增仅数据路径覆盖，保持检查点预处理；可选历史参考损坏默认告警/记录invalid，严格模式才阻断；报告生效路径及验证身份。
- `18:21`（修复）`src/experiments.py`：登记前验证、暂存目录原子发布、复制失败可重试、坏旧实验隔离、空汇总和原子CSV替换；新增同一模型验证复测导入和同指纹组内排名。
- `18:21`（新增）`src/fingerprints.py`、`src/training_state.py`：验证划分/标签语义和评估协议指纹；绑定模型哈希的完整epoch状态保存与恢复，不补造旧优化器状态。
- `18:21`（变更）`src/evaluate.py`、`src/full_train.py`、`src/train.py`、`src/continue_train.py`：评估记录指纹；新训练支持独立目录/配置/种子，续训支持显式完整状态恢复；合成规模开关默认关闭并记录，本轮未进行正式训练。
- `18:21`（新增）`tests/test_reliability.py`、`tests/test_training_state.py`（变更）`tests/test_predict.py`、`tests/test_experiments.py`：覆盖坏参考、路径覆盖、失败登记、分组排名、随机状态和ResNet18训练入口恢复一致性，18项测试通过。
- `18:21`（新增）`requirements-baseline.txt`、`docs/RELIABILITY_UPDATE.md`（变更）`README.md`、`docs/PR_REVIEW_RESPONSE.md`、`CHANGELOG.md`：最小直接依赖、隔离环境步骤、恢复边界和17文件上传清单；未改装全局环境或宣称完成干净安装验收。
- `18:21`（新增）`outputs/experiments/exp_002/predictions/20261022_181637_c6140eaf/*`、`outputs/experiments/exp_003/*`（变更）`outputs/experiment_summary.csv`（已忽略）：同一第4轮权重复测Macro-AUC0.730568、Macro-F10.157715并登记验证身份快照，原实验不变，无新正式训练。

## 2026-10-20

- `13:02`（新增）`src/continue_train.py`、`tests/test_continue_train.py`：从 Baseline-1 权重续训 Epoch 2–4，独立保存各轮结果并按 Macro-AUC 选优；校验原始文件哈希，异常立即停止，记录优化器状态缺失的续训限制。
- `13:02`（变更）`README.md`、`CHANGELOG.md`：补充续训命令与输出保留策略。

## 2026-10-19

- `19:16`（新增）`.agents/skills/ncccu-cxr-baseline/SKILL.md`：从 Windows 下载目录原样安装 NCCCU 胸部 X 光多标签分类项目 Skill，校验 name、description 及文件哈希一致性。
- `19:16`（变更）`CHANGELOG.md`：记录本地项目 Skill 安装与检查结果。
- `19:56`（新增）`src/data.py`、`src/check_data.py`、`configs/data.json`：严格解析官方标签、索引图片、保持比例补边、提供 Dataset/DataLoader 和真实数据检查入口。
- `19:56`（新增）`tests/test_data.py`：使用 unittest 检查标签异常、检查标签一致性、图片索引和纵横比保持。
- `19:56`（新增）`README.md`、`.gitignore`：补充数据检查用法和数据、权重、缓存排除规则。
- `19:56`（变更）`CHANGELOG.md`：记录 Round 1 数据模块开发，不修改官方原始数据。

- `19:58`（修复）`tests/test_data.py`：测试临时目录改为项目 tests 下，解决 Windows 沙箱系统临时目录权限错误；CSV 写入显式使用 UTF-8。
- `19:58`（变更）`README.md`：记录实际环境版本、3 项单元测试通过及三份官方数据随机批次加载成功；全量路径匹配成功，跨集合患者/检查/图片 ID 无重叠。

- `20:02`（新增）`src/model.py`、`src/train.py`：无预训练 ResNet18 十输出模型、真实训练集单批次 AdamW 更新、有限值检查及独立目录模型保存/恢复核验。
- `20:02`（新增）`tests/test_model.py`：验证输出与标签形状、损失、参数变化、保存加载一致性及覆盖保护；连同数据模块共 4 项测试通过。
- `20:02`（变更）`README.md`、`CHANGELOG.md`：补充 Round 2 单步测试命令、超参数、复现边界及尚缺完整验证流程的说明。

- `20:03`（新增）`outputs/round2_20261019_200329_d125e783/model.pth`、`smoke_result.json`（已忽略）：保存真实两张训练图的一次更新模型和实测记录；损失 0.7763936519622803，恢复核验通过。
- `20:03`（变更）`README.md`：记录 GPU 型号、真实测试损失、参数变化与保存加载结果，未进行完整训练或验证集评估。

- `20:07`（新增）`src/evaluate.py`、`src/full_train.py`、`tests/test_evaluate.py`：新增全量单轮训练、Study 级平均概率评估、逐类 AUC 和 F1、显存探测、结果保存与恢复验证；5 项测试通过。
- `20:07`（变更）`src/train.py`、`README.md`、`CHANGELOG.md`：保留单步入口，增加 --full-one-epoch 模式及执行说明。

- `20:38`（新增）`outputs/baseline_v1/best_model.pth`、`train_config.json`、`history.csv`、`val_study_predictions.csv`、`validation_metrics.json`（已忽略）：保存全量单轮训练与 859 次检查验证结果；Macro-AUC 0.704093，Macro-F1 0.111472，模型恢复核验通过。
- `20:38`（变更）`README.md`、`src/train.py`、`CHANGELOG.md`：记录实测耗时、显存、损失、阈值局限及入口说明；未预测测试集，未生成提交文件。

## 2026-10-16

- `13:35`（新增）`docs/胸片多标签分类数据流.svg`：保存以 study 为样本单位的数据流图，展示逐图特征提取、同一次检查内多图特征平均聚合、study 级训练与推理流程；提交阶段筛选预测标签为 1 的类别，按 `Study_id` 和类别逐行输出类别 ID 及原始预测概率。
- `13:40`（新增）`AGENTS.md`：新增中文贡献指南，说明仓库结构、环境命令、编码与测试约定、提交及 PR 要求；明确文件增删改均须记录日志，以及仅在 `develop` 或其派生分支提交、`main` 仅允许 rebase 或 merge 的规则。
- `13:45`（新增）`src/.gitkeep`、`tests/.gitkeep`、`configs/.gitkeep`：创建源码、测试与配置目录，以空占位文件纳入版本控制。
- `13:45`（变更）`AGENTS.md`：更新目录现状，说明各目录职责及 `.gitkeep` 的用途。
- `13:47`（变更）`docs/多目标胸部病变检测分类（ChestX-MLC）.md`：压缩空行、调整提交示例表格排版，移除十类标签对照表，并清理行尾空格。
- `13:47`（修复）`CHANGELOG.md`：将数据流图文件清单修正为实际存在的 SVG，移除未保留的 PUML、PNG 及可编辑源码描述。

## 2026-10-13

- `14:00`（新增）`.gitignore`：忽略虚拟环境、数据集、模型权重与训练产物、预测结果等，避免大数据与权重入库；`.codegraph` 本机工具链符号链接不入库。
- `14:00`（新增）`docs/多目标胸部病变检测分类（ChestX-MLC）.md`：赛题说明文档，涵盖任务描述、数据集构成（训练 18513 / 验证 859 / 测试 1414 次检查）、10 个类别标签映射、数据目录结构与评估要求。
- `14:00`（新增）`requirements.txt`：依赖清单，版本对齐赛题服务器（Python 3.12 / CUDA 12.8 / torch 2.10.0 + torchvision 0.25.0），包含训练栈 `timm`、数据处理与评估 `numpy`/`pillow`/`pandas`/`scikit-learn`/`PyYAML`/`tqdm`，以及图像增强 `opencv-python-headless`/`albumentations`。
- `14:08`（变更）`requirements.txt`：补充 Windows（x64）安装说明与各平台 wheel 差异（PyPI CPU 版 vs cu128 GPU 版），并记录 macOS arm64（MPS）上的实测环境与设备选择建议。
