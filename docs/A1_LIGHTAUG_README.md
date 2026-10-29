# A1：DenseNet121-320-Study-LightAug

本包用于在已有B1流程上单独比较训练轻增强。代码验收仅用少量官方train Batch，不包含A1完整训练成绩。正式训练须由运行同学显式启动。

## 对照与设置

已读取交回的A0实际train_config.json及validation_metrics.json：TV DenseNet121，最佳Epoch 6，Study Macro-AUC=0.7954665375350549，Macro-F1=0.2939811180708684。外层文件夹名称带RA，但内部权重为TorchVision IMAGENET1K_V1，不能按文件夹名判断权重。无患者明细的配置对照及来源哈希见docs/A1_A0_REFERENCE.json。

A1相对上述实际experiment_config仅改变experiment_id和augmentation。旧仓库B1配置workers=4，而交回的A0实际workers=8；A1因此明确使用8，旧配置不改。其余：320 letterbox/BILINEAR/黑色补边、ImageNet mean/std、每Batch 8 Study/最多16图、验证Batch16、seed42、Study均衡逐图BCE、AdamW/weight_decay=.01、backbone/head LR=1e-4/1e-3、2轮Warmup+Cosine、最多12轮、早停3/.0001、官方train训练及完整valid859 Study评估、G0选优。

每张训练图片独立采样：p=.5联合一次RandomAffine（±5度、横纵平移≤2%、scale .98–1.02、无shear、黑色填充）；p=.3联合亮度.95–1.05/对比度.90–1.10。增强在letterbox后、Normalize前进行，不翻转。灰度强度处理不引入色调变化；RGB复制后Normalize使用三通道各自mean/std，因此归一化后的通道数值不要求相等。

配置的augmentation可设为字符串"none"关闭；只能在新实验首次启动前修改，既有目录拒绝覆盖。训练Dataset显式training=True；valid和推理默认none，且禁止在非训练Dataset传入随机增强。所有随机增强只使用PyTorch RNG；DataLoader generator设seed、worker由PyTorch设置base_seed+worker_id，有序返回且非持久worker。支持同设备/同版本/同worker数整轮边界恢复，不保证跨设备逐位复现，不支持中途Batch恢复。

## 放置及环境

建议解压到独立新目录，例如/root/exp-A1，避免覆盖旧工程与模型。复用已跑通A0的Python/CUDA环境；不要为了本实验升级Torch。缺少直接依赖时使用包内requirements.txt建立独立环境。requirements-ra.txt仅供原RA回归测试，A1不依赖timm。

修改configs/data.json的train/valid.csv及图片目录为本机路径；无需读取test。把原始TV权重放到checkpoints/pretrained/densenet121-a639ec97.pth，或修改A1配置中的pretrained_weight_path。不要放入旧B1 best_model.pth。

原始权重SHA-256必须为：
`a639ec97d7c33b07ae66f0b5fb7d0192f95a3b11b7576c66c0126c2a727c4395`

沿用项目已确认的指定TV权重登记和许可证据，程序仍核查SHA及证据文件。许可记录表达的是项目用户已确认采用，不伪造主办方原始回复。不自动下载其他权重。

## Linux命令

进入解压后的项目根目录，以下假设已激活A0的Python环境：

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/check_study320.py --config configs/experiments/A1_DenseNet121_320_Study_LightAug.json
python -m src.augmentation_preview --config configs/experiments/A1_DenseNet121_320_Study_LightAug.json --output outputs/A1_preview_check
python -m src.study_smoke --config configs/experiments/A1_DenseNet121_320_Study_LightAug.json --device cuda --batches 2
```

预览及冒烟用独立目录，正式训练入口不会读取其更新后模型。首次运行Windows多worker较慢，不能用首次启动时间代表稳定吞吐；显存不足须显式报告，不能静默改Batch。

确认预览和冒烟通过后，才执行：

```bash
mkdir -p outputs/logs
train_log="outputs/logs/A1_LightAug_$(date +%Y%m%d_%H%M%S).log"
nohup python -u -m src.run_experiment --config configs/experiments/A1_DenseNet121_320_Study_LightAug.json --execute --device cuda > "$train_log" 2>&1 < /dev/null &
echo "PID: $!  日志: $train_log"
```

如使用/root/my-baseline/.venv环境而未激活，将python替换为/root/my-baseline/.venv/bin/python。

## 输出与交回

正式输出根目录outputs/A1_DenseNet121_320_Study_LightAug/，包含initial（首轮）、result（后续轮）、私有增强预览、run_result。完成后根目录还保存：best_model.pth、train_config.json、history.csv、validation_metrics.json（同一最佳Checkpoint的十类AUC、Macro-AUC和Macro-F1）、valid_image_predictions.csv、val_study_predictions.csv、study_sampling_report.json、augmentation.json、augmentation_preview.png。

同时归档outputs/experiments/A1_DenseNet121_320_Study_LightAug/并更新outputs/experiment_summary.csv。最高原始Macro-AUC决定最佳模型；min_delta仅影响早停计数。因此最佳保存规则与早停改善幅度判定有意区分。

交回模型、配置、history、metrics、采样报告和两级验证预测，均通过私有渠道；真实预览只留内部。代码ZIP不含图片、官方CSV、带真实ID的预测、模型权重或虚拟环境，预览PNG不进入ZIP。

## 公平比较

以A0=0.795467为主对照。确认同一权重SHA、train/valid指纹、Study采样、Step预算、seed、软件版本、归一化、G0和最佳模型选择规则。训练Loss包含随机增强而A0没有，不能仅比较训练Loss判胜负；主要比较同一859 Study上的最佳Macro-AUC及十类AUC，同时记录最佳Epoch、实际训练轮数、耗时和F1。单次种子结果不代表统计显著提升。不要用test选择增强或阈值。

本地小批测试不提供完整A1 AUC，不代表官方T4的100ms验收，也不证明跨设备完全复现。
