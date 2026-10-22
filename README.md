# ChestX-MLC：Round 1 数据读取

## 从新克隆的仓库开始

Python 依赖见 requirements.txt。安装到隔离环境后检查依赖与测试：

```powershell
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
```

单元测试使用临时合成样本，不需要官方数据或训练好的模型。真实数据检查需要先准备：

```text
项目根目录/
  configs/data.json
  cxr_first/cxr_first/
    train.csv
    valid.csv
    test.csv
    train/  # 内含真实图片，可有患者/检查子目录
    valid/
    test/
```

这些数据不随仓库上传。若数据放在其他位置，修改已提交的 configs/data.json，
其中 csv/images 路径相对于命令执行目录，也可使用绝对路径。不要修改官方数据。
然后在项目根目录执行：

```powershell
python -m src.check_data --config configs/data.json
```

只有读取通过后才运行训练。预测章节里的 outputs/... 是历史本机实验路径，
新克隆中不存在；需要先训练产生检查点，或另行取得允许共享的模型及内嵌配置。
本仓库不提供数据、模型权重和预测CSV，不会自动下载预训练权重。

## 审查修订与基线边界

预测/单步/全量/续训入口均支持 --device auto|cuda|mps|cpu，自动顺序 CUDA→MPS→CPU。
非 CUDA 全量首轮采用 batch=2，显存统计写 null，不调用 CUDA 专属 API。
CPU 预测已实测；本机无 MPS，MPS 仅做分支选择测试，未声称通过苹果设备训练实测。
详细审查回应和可复制 PR 描述见 docs/PR_REVIEW_RESPONSE.md。

本项目已完成四轮（不是仅一轮）可复现的随机初始化基线：
Macro-AUC 0.704093→0.711101→0.721307→0.730568，验证 Loss 连续下降。
这证明流程可比较，不代表已经达到竞赛所需性能。保留 weights=None、无增强、输入 /255
以兼容既有模型；未经用户或官方确认不下载预训练权重。timm 是原依赖清单的一部分，
ResNet18 使用 torchvision，不为满足依赖存在而引入另一套模型实现。
输入归一化/增强应作为独立新实验实施并重训、保存相应预处理元数据，不能套用到旧权重。

## Round 4：统一预测与实验比较

所有命令均在项目根目录运行。只预测，不更新参数：

```powershell
python -m src.predict --model outputs/experiments/exp_002/best_model.pth --split both --all-classes
```

以上是本地全部类别模式示例，不代表已确认平台提交语义。也可选择 `--split valid` 或 `--split test`。
配置优先从检查点 metadata 加载，兼容旧模型的同目录 train_config.json，
因此 epoch_N/model.pth 无需额外复制配置也可预测。历史验证指标和预测文件可选。
自动读取图片尺寸、批量、数据路径和预处理配置，支持本项目 ResNet18 十输出检查点。
每次新建模型目录下 predictions/时间戳_随机编号/，包含逐图/逐检查预测、复测指标、
模拟 submission.csv、submission_check.json、prediction_report.json 和实际运行日志。
验证复测比较 AUC/F1 和逐项概率；差异默认告警并写入指标，不阻断后续预测。
需要严格复现时使用 --strict-reproduce；容差可通过 --metric-atol 和 --probability-atol 配置，
默认均为 1e-4。图片/ID错位、非法概率等数据错误仍会立即停止。
测试集没有标签，程序不计算测试 AUC。批量推理的平均耗时不是官方 T4 单图延迟证明。

本地材料未找到正式 sample_submission.csv。模拟文件采用
Subject_id,Study_id,Predict_class,Probability 四列，不添加 Gate。
测试集输出没有默认模式，必须显式选 --all-classes 或 --threshold 0.5；两者互斥。
全部模式每检查输出10类；阈值模式只输出 Probability >= threshold 的类别，
无阳性的检查可无行，文件也可能仅有表头。阈值模式不再强制14140行，
而是用完整检查概率核验筛选结果，漏掉应输出的阳性记录也会报错。
--template 可核对四列名称并遵循模板顺序；若包含 Gate 等未知字段会拒绝生成。
模板列校验不能代替主办方对提交语义的确认。
本地文档有按阈值输出的描述，因此每检查是否必须输出全部类别、Gate 和疾病中文名称映射，
仍需在正式上传前对照平台最终要求确认。本轮不上传任何文件。

已有实验：exp_001 保存 Baseline-1，exp_002 保存四轮训练最佳模型；均为原结果的哈希校验快照，
没有重新训练。outputs/experiment_summary.csv 汇总参数、最佳 epoch、验证 AUC/F1 和备注，
按 Macro-AUC 排序。原始实验独立保留，汇总表为可重建索引。

未来新实验使用独立训练输出目录，然后登记（以下训练命令本轮没有执行）：

```powershell
python -m src.continue_train --source outputs/experiments/exp_002 --output outputs/runs/run_003 --additional-epochs 3 --learning-rate 0.0003
python -m src.experiments --source outputs/runs/run_003 --id exp_003 --notes "仅降低学习率，与exp_002比较"
python -m src.predict --model outputs/experiments/exp_003/best_model.pth --split valid
```

续训入口复用原模型和预处理，按最佳模型的 epoch 开始计数；恢复模型权重，
AdamW 和数据加载器随机状态重新初始化，属于新实验，配置会明确记录。
保持验证集和聚合方法不变，通过验证 Macro-AUC 选择最佳模型；F1 使用固定0.5阈值。
实验登记拒绝重复编号，不修改旧实验。每个快照含权重、训练参数、历史、验证预测、
验证指标、experiment_notes.md 和来源哈希；不凭测试概率或训练损失评判性能。

2026-10-22 实测结果：验证 1,422 张 / 859 次检查，Macro-AUC=0.730568169109816、
Macro-F1=0.15771485793395476，与第4轮历史成绩一致。最大概率差异 1.11e-16。
测试 2,218 张 / 1,414 次检查，无读取失败；模拟提交 14,140 行，692,501 字节，
四列顺序、ID、1..10 类、重复/缺失、概率范围及 UTF-8 回读均通过。
模型 44,806,795 字节，小于 500 MB。RTX 4060 Laptop GPU、batch=16，
测试图像解码/预处理/传输/前向共 109.27 秒，批量摊销约 49.26 ms/张，
未在官方 T4 实测，不作为单图100ms达标证明。8 项单元测试通过。
结果目录：outputs/experiments/exp_002/predictions/20261022_095153_89283345/。
csv_audit.json 记录全部新预测 CSV 的实际回读结果和来源/快照哈希核验。
本轮未训练或上传；未来续训参数入口只完成帮助命令及防覆盖验证，未执行新训练。

## Round 3-B：从第一轮权重继续三轮

```powershell
python -m src.continue_train
```

读取 outputs/baseline_v1 权重，固定续训 Epoch 2–4，保留 lr=0.001、
batch=16、seed=42、预处理和 Study 级评估。旧检查点没有优化器状态，
因此仅恢复模型权重，AdamW 动量重新初始化，数据加载器随机状态也重新初始化。
输出 outputs/baseline_v2，目录已存在时拒绝覆盖；不修改 baseline_v1，
并记录、复查全部原文件 SHA-256。每轮保存模型、优化器/随机状态、验证预测和指标。
以 Epoch 1 为初始候选，按 Macro-AUC 严格提高选优，平分保留较早模型。
发生 NaN、OOM 或其他异常时停止并记录 failure.json，不自动重试或跳过批次。

## Round 3：完整训练一轮

```powershell
python -m src.train --full-one-epoch
```

固定 seed=42、AdamW lr=0.001、weight_decay=0.01、224×224，无外部权重。
先用独立模型探测 batch=16，显存不足时尝试 8；正式模型重新初始化。
只遍历训练集一次，使用全部验证图片，先 sigmoid 再按 Study_id 平均概率。
Loss 是按图片数加权的 BCE 均值；AUC、F1 是 Study 级指标，F1 阈值 0.5。
任何类别 AUC 无定义时单独列出，Macro-AUC 留空，不替换为 0.5。
输出到 outputs/baseline_v1；目录存在时拒绝覆盖。不会读取测试集或生成提交文件。

Round 3 实测（2026-10-19，seed=42）：完整训练 30,344 张，验证 1,422 张 / 859 次检查。
RTX 4060 Laptop GPU 8 GiB，batch=16，PyTorch 峰值已分配显存 524.65 MiB、
峰值保留显存 714 MiB（不包含其他进程和驱动占用）。训练 1676.21 秒，验证 75.13 秒。
训练 Loss=0.344937，验证 Loss=0.337641；Study Macro-AUC=0.704093，Macro-F1=0.111472（0.5 阈值）。
10 类 AUC 均有定义，模型重新加载的参数与输出一致；5 项测试通过。
0.5 阈值下类别 1、2、3、4、5、7、8 没有预测阳性，F1 较低；单轮结果不足以判断过拟合。
输出包含 best_model.pth、train_config.json、history.csv、val_study_predictions.csv、validation_metrics.json。
这里只有一轮，best_model.pth 即这一轮模型，没有进行多轮模型比较。

## Round 2：一次参数更新测试

```powershell
python -m unittest discover -s tests -v
python -m src.train --data-config configs/data.json --seed 42
```

此入口仅取训练集两张真实图片、执行一次 AdamW 更新（学习率 0.001，权重衰减 0.01）。
ResNet18 使用 weights=None，输出十维 logits，损失为 BCEWithLogitsLoss。
自动优先 CUDA，其次 MPS，最后 CPU；固定随机种子与 cuDNN 设置，但不保证跨硬件完全一致。
每次运行创建独立 outputs/round2_* 目录，保存 model.pth 和 smoke_result.json，避免覆盖。
保存后检查完整状态字典及同批次 eval 输出完全相等。此权重仅用于流程验证，不能视为训练好的模型。
Round 2 入口不包含完整验证；Round 3 入口已补充单轮训练和 Study 级评估。

Round 2 实测：RTX 4060 Laptop GPU / CUDA，种子 42，真实训练集 batch=2，
输出与标签均为 [2,10]，单步损失 0.7763936519622803，分类层最大参数变化
0.0010004416108131409；完整参数和重新加载后的 eval 输出一致。4 项单元测试通过。
结果位于 outputs/round2_20261019_200329_d125e783/，未计算验证集指标。

在项目根目录运行（不会启动训练，也不会下载权重）：

```powershell
python -m unittest discover -s tests -v
python -m src.check_data --config configs/data.json
```

`configs/data.json` 配置三个 CSV 和图片根目录，路径相对于运行目录。
读取器递归扫描 JPG/PNG，以文件名不含扩展名的部分匹配 Dicom_id；重复和缺失均报错。
CSV 按字符串读取，保留前导零。类别 1..10 对应标签向量第 1..10 位；空值、不确定值、非法类别和检查内标签冲突均报错。

图片转为 RGB，按比例缩放并黑色补边为 224×224，张量为 float32、范围 [0,1]，不做翻转。
Dataset 返回 image、Subject_id、Study_id、Dicom_id，训练/验证另含 label；测试集不生成标签。
DataLoader 默认 num_workers=0，适合 Windows；随机读取默认种子 42。
检查入口匹配全部图片路径，每份数据随机解码 4 张，并检查患者、检查和图片 ID 的跨集合重叠。
它不代表所有图片均已完成解码验证。

本轮使用 Python 标准库 unittest，不增加测试框架依赖。当前环境版本以实际检查结果为准；
requirements.txt 中原有目标版本保留，不在本轮更改或重装环境。

## 2026-10-19 实测

- Python 3.13.11、torch 2.11.0+cu128、torchvision 0.26.0+cu128、pandas 3.0.3、Pillow 12.0.0 均导入成功。
- unittest：3 项通过。首次因沙箱系统临时目录权限失败，改用项目测试目录后通过。
- train：30,344 张 / 18,513 次检查；valid：1,422 张 / 859 次检查；test：2,218 张 / 1,414 次检查。
- 三份数据全部图片路径匹配成功；各随机读取 4 张真实 JPG，图像形状均为 [4,3,224,224]，训练/验证标签为 [4,10]，测试无标签。
- 三个集合之间 Subject_id、Study_id、Dicom_id 均无交集；训练/验证同次检查标签一致。
- 尚未训练模型、验证 CUDA 可用性或逐张解码全部影像；当前环境与仓库目标版本不同。
