# ChestX-MLC 使用手册

> 本手册介绍配置、训练产物与运行命令；以当前检出的代码和配置为准。
> 赛题背景见 [赛题说明](多目标胸部病变检测分类（ChestX-MLC）.md)，模块划分见
> [架构与文件结构](architecture.md)，调优结论见 [P7 调优记录](experiments.md)。

## 目录

1. [环境与数据准备](#1-环境与数据准备)
2. [配置说明](#2-配置说明)
3. [output 目录结构与产物](#3-output-目录结构与产物)
4. [运行指令](#4-运行指令)
5. [典型工作流](#5-典型工作流)
6. [排错速查](#6-排错速查)
7. [相关文档](#7-相关文档)

---

## 1. 环境与数据准备

### 1.1 安装环境

**所有命令都在仓库根目录执行**。配置里的相对路径按“当前工作目录”解析，不在根目录运行会把 `data/`、`outputs/` 解析到错误位置。

macOS / Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Windows 的虚拟环境激活方式、CUDA 版 PyTorch 安装源见 `requirements.txt` 顶部注释。安装完成后自检框架与设备：

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

### 1.2 准备数据

默认配置 `configs/default.yaml` 期望如下目录结构：

```text
data/
├── train.csv                 # 训练集标注（含 Predict_class）
├── valid.csv                 # 验证集标注（含 Predict_class）
├── test.csv                  # 测试集标注（无 Predict_class）
└── images/
    ├── train/
    │   └── subject_00000/
    │       └── study_00000/
    │           ├── pic_000000.jpg
    │           └── pic_000001.jpg
    ├── valid/
    └── test/
```

数据约定：

| 项目 | 说明 |
| --- | --- |
| 一行一图 | CSV 每行对应一张图，`Dicom_id` 是图片唯一标识，也是文件名（不含后缀）。 |
| study 聚合 | `Study_id` 相同的多行属于同一次检查，共享同一组标签；一次检查含 1–7 张图。 |
| 标签格式 | `Predict_class` 为分号分隔的类别 ID，如 `7;6;9`；类别 ID 取 1–10，模型按下标 0–9 处理。 |
| ID 格式 | 全程按字符串读取以保留前导零。`check_data.py` 校验 `subject_\d{5}`、`study_\d{5}`、`pic_\d{6}` 三种格式。 |
| 图片索引 | 递归扫描图片目录，按文件名（stem）匹配 `Dicom_id`；支持 `.jpg` / `.jpeg` / `.png`，大小写不敏感。 |
| 数据划分 | train / valid / test 的 `Subject_id`、`Study_id`、`Dicom_id` 都不应重叠，训练前用 `check_data.py` 确认。 |

`data/` 在 `.gitignore` 中，不会被提交。若数据放在其他位置，改配置里的 `data.<split>.csv` 与 `data.<split>.images` 即可，也可以写绝对路径。

> 示例数据：`configs/exp_example.yaml` 期望数据位于 `data/example/`。如果仓库根已有 `example/`（`train.csv`、`valid.csv`、`test.csv` 与 `*_example/` 图片目录），可以建立软链接后直接使用：
>
> ```bash
> mkdir -p data
> ln -s ../example data/example
> ```

### 1.3 第一次运行前

```bash
python -m pytest
python src/check_data.py --config configs/default.yaml --full
```

前者确认代码与依赖正常（测试只用合成小图），后者确认 CSV、图片、标签和划分无问题。

---

## 2. 配置说明

### 2.1 配置文件如何被读取

入口脚本通过 `--config` 指定 YAML（默认 `configs/default.yaml`），由 `src/common/config.py` 读取、合并并校验，最终转成 `Config` 对象。

几个关键规则：

1. **严格校验，不静默忽略**。字段名拼错、类型不对、取值越界都会在加载时报错，并指出具体是哪一节、哪个字段。例如把 `epochs` 写成 `epoch`，会报“`配置根节点.train` 含未知字段”。
2. **相对路径按当前工作目录解析**。`data.train.csv: data/train.csv` 是相对“你执行命令时所在的目录”，不是相对配置文件所在目录。因此请始终在仓库根目录运行。
3. **YAML 1.1 的数字坑**。`3e-4` 会被 PyYAML 读成字符串，导致类型校验失败；请写成小数 `0.0003`。
4. **支持 `_base` 继承**。实验配置用顶层 `_base` 指向基础配置，只写要覆盖的字段。`_base` 的相对路径按**当前配置文件所在目录**解析。
5. **继承的合并方式**。嵌套的 `data` / `train` 等小节逐字段合并；标量、列表整体替换；`_base` 形成循环继承会直接报错。

一个最小实验配置：

```yaml
_base: default.yaml

train:
  pos_weight: false     # 覆盖当前默认值，其余沿用 default.yaml
```

### 2.2 现有配置文件

| 文件 | 用途 | 继承关系 | 与默认配置的主要差异 |
| --- | --- | --- | --- |
| `configs/default.yaml` | P7 冻结的最终配置，正式训练与提交使用 | 无 | 25 epoch、`pos_weight: true` |
| `configs/exp_short.yaml` | 短周期方向筛查 | `_base: default.yaml` | `epochs: 6`、关闭早停和 `pos_weight` |
| `configs/exp_view_weighted.yaml` | 正侧位加权聚合对照，待训练 | `_base: default.yaml` | `aggregation: view_weighted`，正位权重 0.7 |
| `configs/exp_freeze_head.yaml` | 冻结骨干、仅训练分类头的对照，待训练 | `_base: default.yaml` | `model.freeze_backbone: true` |
| `configs/exp_cosine.yaml` | RUN-003-A 历史余弦调度对照 | `_base: default.yaml` | 固定 `cosine`、关闭 `pos_weight` |
| `configs/exp_pos_weight.yaml` | 完整周期的 `pos_weight` 实验配置 | `_base: default.yaml` | 显式设置 `pos_weight: true`，与当前默认一致 |
| `configs/exp_run003b.yaml` | RUN-003-B：当前默认口径的 ResNet-18 完整复跑（cosine + pos_weight，seed 42），已完成 | `_base: default.yaml` | 显式固定 `backbone: resnet18`、`pos_weight: true`、`lr_scheduler: cosine`、`lr_min`、`seed: 42`（取值与当前默认一致） |
| `configs/exp_short_pos_weight.yaml` | 短周期 `pos_weight` 对照 | `_base: exp_short.yaml` | 在短周期基础上打开 `pos_weight` |
| `configs/exp_densenet121.yaml` | DenseNet-121 骨干对照，已完成（RUN-006） | `_base: default.yaml` | `backbone: densenet121.ra_in1k` |
| `configs/exp_run006a.yaml` | RUN-006-A：RUN-006 的 seed 复核（DenseNet-121，seed 2026），已完成 | `_base: exp_densenet121.yaml` | 只覆盖顶层 `seed: 2026` |
| `configs/exp_run007.yaml` | RUN-007：DenseNet-121 的正侧位 7:3 加权聚合对照，已完成 | `_base: exp_densenet121.yaml` | `aggregation: view_weighted`、`frontal_weight: 0.7` |
| `configs/exp_run007a.yaml` | RUN-007-A：RUN-007 的 seed 复核（DenseNet-121 加权聚合，seed 2026），已完成 | `_base: exp_run007.yaml` | 只覆盖顶层 `seed: 2026` |
| `configs/exp_run007b.yaml` | RUN-007-B：RUN-007-A 的 384×384 + 6 轮短周期探针，已完成 | `_base: exp_run007a.yaml` | `image.size: 384`、`train.epochs: 6` |
| `configs/exp_run008a.yaml` | RUN008-A：EfficientNet-B2 骨干对照（6 轮无早停，与 RUN008-B 并行），已完成 | `_base: exp_run007a.yaml` | `model.backbone: efficientnet_b2.ra_in1k`、`train.epochs: 6`、`train.early_stop_patience: 0` |
| `configs/exp_run008b.yaml` | RUN008-B：EfficientNetV2-S（RW）骨干对照（6 轮无早停，与 RUN008-A 并行），已完成 | `_base: exp_run007a.yaml` | `model.backbone: efficientnetv2_rw_s.ra2_in1k`、`train.epochs: 6`、`train.early_stop_patience: 0` |
| `configs/exp_run008c.yaml` | RUN008-C：ConvNeXt-Tiny 骨干对照（6 轮无早停，与 RUN008-D 并行），已完成 | `_base: exp_run007a.yaml` | `model.backbone: convnext_tiny.fb_in1k`、`train.epochs: 6`、`train.early_stop_patience: 0` |
| `configs/exp_run008d.yaml` | RUN008-D：ConvNeXtV2-Nano 骨干对照（研究用，CC BY-NC 4.0 不能提交；训练异常），已完成 | `_base: exp_run007a.yaml` | `model.backbone: convnextv2_nano.fcmae_ft_in1k`、`train.epochs: 6`、`train.early_stop_patience: 0` |
| `configs/exp_A001.yaml` | A-001：RUN-008-C 的重跑（ConvNeXt-Tiny @320×320，epoch=8 / 早停 2 / seed 42），已完成 | 无（完整展开，不继承 `default.yaml`） | 与 `exp_run008c.yaml` 展开结果仅差 `train.epochs: 8`、`train.early_stop_patience: 2`、`seed: 42` |
| `configs/exp_A001b.yaml` | A-001-B：A-001 的 ASL 损失对照（ASL 替代 BCE + `pos_weight`，其余口径不变），已完成 | `_base: exp_A001.yaml` | 只覆盖 `train.loss: asl`、`train.pos_weight: false`、`train.asl_gamma_pos: 0.0`、`train.asl_gamma_neg: 4.0`、`train.asl_clip: 0.05` 五项 |
| `configs/exp_resnet50.yaml` | ResNet-50 骨干对照，已完成（RUN-005） | `_base: default.yaml` | `backbone: resnet50` |
| `configs/exp_example.yaml` | 示例数据冒烟 | `_base: default.yaml` | `data` 三个 split 指向 `data/example` |

继承关系：

```text
default.yaml
├── exp_short.yaml
│   └── exp_short_pos_weight.yaml
├── exp_view_weighted.yaml
├── exp_freeze_head.yaml
├── exp_cosine.yaml
├── exp_pos_weight.yaml
├── exp_run003b.yaml
├── exp_densenet121.yaml
│   ├── exp_run006a.yaml
│   └── exp_run007.yaml
│       └── exp_run007a.yaml
│           ├── exp_run007b.yaml
│           ├── exp_run008a.yaml
│           ├── exp_run008b.yaml
│           ├── exp_run008c.yaml
│           └── exp_run008d.yaml
├── exp_resnet50.yaml
└── exp_example.yaml
```

> `configs/exp_A001.yaml` 为 A-001 的完整展开配置，不使用 `_base` 继承，故未画入上面的继承树。
> `configs/exp_A001b.yaml` 以 `_base: exp_A001.yaml` 继承 A-001，只覆盖损失相关五项，未单独画入继承树。
>
> 做实验不要直接改 `default.yaml`。默认配置是最终提交口径，实验差异写进 `_base` 派生的新文件，结论和命令记录到 `docs/experiments.md`。

骨干实验的实测结果和选择依据见 [调优记录](experiments.md)；各 run 的运行配置与指标见 [训练记录](training_log.md)。

### 2.3 字段总览

| 节 | 作用 | 是否必填 |
| --- | --- | --- |
| `data` | 三个 split 的标注 CSV 与图片根目录 | 必填，三个 split 都要写 |
| `image` | 输入尺寸、letterbox、归一化、训练水平翻转 | 有默认值 |
| `model` | 骨干网络、是否加载预训练、类别数 | 有默认值 |
| `train` | 训练轮数、batch、学习率、AMP、早停等 | 有默认值 |
| `eval` | 验证/推理 batch 与 F1 判定阈值 | 有默认值 |
| `output` | 训练产物根目录 | 有默认值 |
| `seed` | 随机种子 | 有默认值 |
| `device` | 运行设备 | 有默认值 |

### 2.4 `data`：数据划分

```yaml
data:
  train:
    csv: data/train.csv
    images: data/images/train
  valid:
    csv: data/valid.csv
    images: data/images/valid
  test:
    csv: data/test.csv
    images: data/images/test
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `data.<split>.csv` | 非空路径字符串 | 该 split 的标注 CSV。`train` / `valid` 必须含 `Dicom_id`、`Subject_id`、`Study_id`、`Predict_class`；`test` 只要求前三个。 |
| `data.<split>.images` | 非空路径字符串 | 该 split 的图片根目录，递归索引；目录不存在会报错。 |

`<split>` 只能是 `train`、`valid`、`test`。三个 split 缺一不可。

### 2.5 `image`：预处理

| 字段 | 类型 | 默认 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `image.size` | 整数 | `320` | `> 0` | letterbox 之后的方形边长。建议取 32 的倍数以对齐 ResNet 下采样；改动后必须重新训练，推理也要用同一尺寸。 |
| `image.letterbox` | 布尔 | `true` | `true` / `false` | `true`：等比缩放到最长边等于 `size`，再居中补黑边，保持胸片比例；`false`：直接拉伸成 `size × size`，可能使器官轮廓变形。 |
| `image.normalize` | 字符串 | `imagenet` | `imagenet` / `none` | `imagenet`：减 ImageNet 均值、除标准差，配合预训练权重；`none`：只把像素缩到 0–1。 |
| `image.hflip` | 布尔 | `false` | `true` / `false` | 仅训练增强是否做水平翻转。胸片左右有方向含义，默认关闭。 |

训练时的其他增强（±7° 旋转、0.9–1.1 缩放、5% 平移、亮度/对比度 ±15%）由代码固定，不受配置控制；验证和推理只做确定性的缩放与归一化。

### 2.6 `model`：模型

| 字段 | 类型 | 默认 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `model.backbone` | 非空字符串 | `resnet18` | timm 支持的模型名 | 骨干网络。换骨干等于换模型，必须重新训练；推理配置要与训练一致。 |
| `model.pretrained` | 布尔 | `true` | `true` / `false` | 是否加载 ImageNet 预训练权重。离线环境拿不到权重时会告警并回退随机初始化，训练日志与 checkpoint 的 `metadata.pretrained_loaded` 会记录真实情况。 |
| `model.freeze_backbone` | 布尔 | `false` | `true` / `false` | 是否冻结骨干、仅训练 timm 识别的分类头。冻结实验应使用单独配置；默认关闭以保持全量微调。 |
| `model.num_classes` | 整数 | `10` | 必须等于 `10` | 赛题固定 10 类，写其他值会直接报错。 |
| `model.aggregation` | 字符串 | `mean` | `mean` / `view_weighted` | 图片等权均值或按正侧位均值加权。 |
| `model.frontal_weight` | 数字 | `0.7` | `0 < 值 < 1` | 视角加权时正位权重；侧位权重为 `1 - frontal_weight`。 |

默认结构为 ResNet-18 骨干 + 单层 `Linear(512, 10)` 分类头，参数量约 11.2M。

### 2.7 `train`：训练超参数

| 字段 | 类型 | 默认 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `train.epochs` | 整数 | `25` | `>= 1` | 总训练轮数。续训时可以在新配置里调大以延长训练。 |
| `train.batch_studies` | 整数 | `16` | `>= 1` | 每个 batch 的**检查数**，不是图片数；实际图片数是一次检查的 1–7 张之和。显存不足时优先调小。 |
| `train.lr` | 浮点 | `0.0003` | `> 0` | AdamW 基础学习率。必须用小数写法，不要写 `3e-4`。 |
| `train.weight_decay` | 浮点 | `0.01` | `>= 0` | AdamW 权重衰减。 |
| `train.pos_weight` | 布尔 | `true` | `true` / `false` | 默认配置按训练集类别频率给 BCE 正样本加权（`负样本数 / 正样本数`）；未指定字段时兼容旧配置，使用 `false`。 |
| `train.warmup_epochs` | 整数 | `2` | `0 <= warmup < epochs` | 线性 warmup 轮数，按 epoch 逐步把学习率升到 `lr`。默认下第 1 轮为 `lr/2`、第 2 轮为 `lr`。 |
| `train.lr_scheduler` | 字符串 | `cosine` | `warmup` / `cosine` | 默认配置在预热后余弦衰减；`warmup` 在预热后保持学习率。未指定字段时兼容旧配置，使用 `warmup`。 |
| `train.lr_min` | 浮点 | `0.00001` | `0 <= lr_min <= lr` | 余弦调度的最低学习率；提前停止时可能尚未到达。 |
| `train.amp` | 布尔 | `true` | `true` / `false` | 混合精度。仅在 CUDA 上生效，CPU / MPS 上自动等价于关闭。 |
| `train.grad_clip` | 浮点 | `5.0` | `>= 0` | 梯度范数裁剪上限；`0` 表示不裁剪。 |
| `train.num_workers` | 整数 | `8` | `>= 0` | DataLoader 工作进程数，train / valid / test 共用。Windows 或调试时可设为 `0`。 |
| `train.early_stop_patience` | 整数 | `5` | `>= 0` | valid macro-AUC 连续多少个 epoch 不提升就停止；`0` 关闭早停。AUC 为 NaN 的轮次不算提升。 |
| `train.loss` | 字符串 | `bce` | `bce` / `asl` | 训练损失。`bce` 为普通二分类交叉熵；`asl` 为非对称损失（Ben-Baruch et al., ICCV 2021），自带正负样本非对称重加权，因此不能与 `train.pos_weight` 同时开启。 |
| `train.asl_gamma_pos` | 浮点 | `0.0` | `>= 0` | ASL 正样本聚焦指数；论文默认 `0`，即不衰减正样本。 |
| `train.asl_gamma_neg` | 浮点 | `4.0` | `>= 0` | ASL 负样本聚焦指数；越大，简单负样本被压制得越狠。 |
| `train.asl_clip` | 浮点 | `0.05` | `[0, 1)` | ASL 概率移位量 `m`：负样本概率先加 `m` 再取对数，等价于把负样本概率推低 `m`；`0` 表示关闭移位。 |

### 2.8 `eval`：验证与推理

| 字段 | 类型 | 默认 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `eval.batch_studies` | 整数 | `32` | `>= 1` | 验证与推理的 batch 大小，同样按检查数计。 |
| `eval.threshold` | 浮点 | `0.5` | `[0, 1]` | 计算 macro-F1 时的判定阈值，也用于 `run_eval` 的单阈值结果。它不参与训练、不影响 AUC，也**不会**默认过滤提交文件。 |

### 2.9 `output`、`seed`、`device`

| 字段 | 类型 | 默认 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| `output.run_dir` | 非空路径字符串 | `outputs` | 目录可创建 | 训练产物根目录。每次训练在其下新建 `<时间戳>_<tag>/`，不会覆盖已有 run。 |
| `seed` | 整数 | `42` | `>= 0` | Python / NumPy / PyTorch 随机种子。命令行 `--seed` 会覆盖它。 |
| `device` | 非空字符串 | `auto` | `auto` / `cpu` / `cuda` / `cuda:N` / `mps` | `auto` 按 CUDA → MPS → CPU 选择。显式指定不可用设备会直接报错，不会静默退回 CPU。 |

### 2.10 常见改法

**换数据路径**（只改 `data`，不改其他字段）：

```yaml
data:
  train:
    csv: /path/to/train.csv
    images: /path/to/train_images
  valid:
    csv: /path/to/valid.csv
    images: /path/to/valid_images
  test:
    csv: /path/to/test.csv
    images: /path/to/test_images
```

**显存不足**：先把 `train.batch_studies` 从 16 降到 8 或 4，再考虑把 `train.num_workers` 降到 4；仍不足时再降 `image.size`（但这会改变输入分辨率，需要重新训练）。

**离线服务器**：把 `model.pretrained` 设为 `false`；或提前把 timm 权重缓存到本地，让 `pretrained: true` 能命中缓存。训练日志会打印“预训练权重: 已加载/未加载”。

**新增对照实验**：在 `configs/` 下新建文件，不要复制整份默认配置：

```yaml
_base: default.yaml

train:
  epochs: 10
  early_stop_patience: 0
```

然后 `python src/run_train.py --config configs/exp_xxx.yaml --tag xxx`。

### 2.11 配置校验常见报错

| 报错关键词 | 原因 | 处理 |
| --- | --- | --- |
| `含未知字段` | 字段名拼错或不属于该节 | 对照第 2.3–2.9 节的字段名 |
| `应该是整数` / `应该是 true/false` / `应该是非空字符串` | YAML 类型不对 | 按字段类型修改；布尔值不要加引号 |
| `train.lr 应该是数字，实际是 '3e-4'` | PyYAML 把科学计数法读成字符串 | 改成 `0.0003` |
| `train.warmup_epochs 必须满足 ...` | warmup 大于等于 epochs | 调小 warmup 或调大 epochs |
| `model.num_classes 必须等于 10` | 类别数被改动 | 改回 10 |
| `配置文件通过 _base 形成了循环继承` | 继承链成环 | 检查 `_base` 指向 |
| `配置文件内容为空` | YAML 为空 | 至少写入 `data` 或 `_base` |

---

## 3. output 目录结构与产物

### 3.1 总体结构

`output.run_dir` 默认是仓库根目录下的 `outputs/`（已在 `.gitignore` 中）。一次完整的训练、对比和提交大致如下：

```text
outputs/
├── 20261027-083159_resnet18_s320/     # 一次训练 run
│   ├── config.yaml                   # 本次训练实际使用的完整配置
│   ├── train.log                     # 终端输出（含 stderr）
│   ├── metrics.csv                   # 每个 epoch 的指标
│   ├── best.pt                       # valid macro-AUC 最优权重
│   └── last.pt                       # 每轮保存的完整训练状态，用于续训
├── 20261027-101500_example/
│   ├── config.yaml
│   ├── config_resume.yaml            # 仅续训时出现：本次续训生效的配置
│   ├── train.log                     # 续训时追加写入
│   ├── metrics.csv
│   ├── best.pt
│   └── last.pt
├── compare/                          # run_compare.py 的输出
│   ├── summary.csv
│   ├── per_class.csv
│   └── summary.md
├── submission.csv                    # run_infer.py 的默认提交文件
└── example_submission.csv            # 示例数据冒烟时的提交文件（示例）
```

### 3.2 run 目录命名

训练入口调用 `make_run_dir` 创建目录，格式为：

```text
<output.run_dir>/<YYYYMMDD-HHMMSS>_<tag>/
```

| 组成 | 规则 |
| --- | --- |
| 时间戳 | 本机当前时间，精确到秒。 |
| `tag` | 由 `--tag` 指定；省略时默认 `{backbone}_s{size}`，例如 `resnet18_s320`。 |
| 清洗 | 非 `[0-9A-Za-z._-]` 的字符替换为 `_`，首尾的 `._-` 去掉；清洗后为空则用 `run`。 |
| 冲突 | 同一秒内重复创建同一目录会报错（`exist_ok=False`），避免两次训练写进同一个 run。 |

建议给每个实验一个可读的 ASCII tag，例如 `short_baseline`、`short_pos_weight`、`full_baseline`，方便 `run_compare.py` 和人工辨认。

### 3.3 各文件说明

| 文件 | 何时产生 | 内容与用途 |
| --- | --- | --- |
| `config.yaml` | 训练开始时 | 合并 `_base` 并经校验后的初始完整配置。续训后还需核对 `config_resume.yaml` 和 checkpoint 中存档的配置。 |
| `config_resume.yaml` | 使用 `--resume` 时 | 本次续训实际生效的配置（省略 `--config` 时就是 checkpoint 里存档的配置）。原来的 `config.yaml` 保留不动，便于对照“最初配置”和“续训配置”。 |
| `train.log` | 训练开始时创建，续训时追加 | 终端 stdout 与 stderr 的完整副本。包含配置来源、run 目录、设备、seed、预训练是否加载、每个 epoch 一行指标、早停/中断提示。 |
| `metrics.csv` | 每个 epoch 结束时重写 | 每个 epoch 一行的训练/验证指标，见 3.4。中断时保留最后一个完整 epoch。 |
| `best.pt` | 训练**正常结束后** | valid macro-AUC 最高的权重。推理、验证、对比都使用它。训练被中断时不会生成。 |
| `last.pt` | 每个 epoch 结束时 | 可续训的完整状态：权重、优化器、学习率调度器、AMP scaler、已完成轮次、最优快照与历史。文件明显大于 `best.pt`。 |
| `summary.csv` / `per_class.csv` / `summary.md` | `run_compare.py` 运行后 | 多个 run 的指标对比表与逐类指标，见 3.6。 |
| `submission.csv` | `run_infer.py` 运行后 | 赛题提交文件，长表格式，见 3.7。 |

> `best.pt` 只在训练结束时落盘，是为了保证它始终是一个完整文件，不会出现“写到一半”的坏权重。若训练中断，请用 `last.pt` 续训，训练完成后自然会生成 `best.pt`。

### 3.4 `metrics.csv` 列含义

| 列 | 含义 |
| --- | --- |
| `epoch` | 轮次，从 1 开始。 |
| `lr` | 该轮训练实际使用的学习率（warmup 期间小于配置的 `train.lr`）。 |
| `train_loss` | 训练集按 study 数加权的平均 BCE loss。 |
| `valid_loss` | 验证集按 study 数加权的平均 loss。 |
| `valid_macro_auc` | 验证集 macro-AUC，是挑选最优权重和早停的依据；退化类别（全阴/全阳）不参与平均，全部退化时为 NaN。 |
| `valid_macro_f1` | 验证集 macro-F1，按 `eval.threshold` 计算。 |

每轮重写整个文件，所以即使训练中断，已完成的轮次也不会丢；续训会沿用并继续追加历史。

### 3.5 checkpoint 内部字段

`best.pt` 与 `last.pt` 都是 `torch.save` 的字典：

| 键 | `best.pt` | `last.pt` | 说明 |
| --- | --- | --- | --- |
| `model` | 有 | 有 | 模型 `state_dict`。 |
| `optimizer` | `None` | 有 | AdamW 状态，续训用。 |
| `scheduler` | `None` | 有 | 学习率调度器状态。 |
| `scaler` | `None` | 有 | AMP GradScaler 状态。 |
| `epoch` | 最优 epoch | 已完成 epoch | 轮次编号从 1 开始。 |
| `metrics` | 最优轮的验证指标 | 当前最优轮的验证指标 | 含 `loss` / `macro_auc` / `macro_f1`。 |
| `config` | 完整配置 | 完整配置 | 续训省略 `--config` 时从这里恢复。 |
| `seed` | 训练 seed | 训练 seed | 无命令行覆盖时等于配置值。 |
| `metadata` | 有 | 有 | `pretrained_loaded`、`device`、`freeze_backbone`，以及 `total_params` / `trainable_params` 参数计数。 |
| `extras` | `None` | 有 | 训练进度、最优权重快照、历史记录、早停计数；续训只认带 `extras` 的文件。 |

因此：**推理用 `best.pt`，续训用 `last.pt`**。拿 `best.pt` 续训会因缺少 `extras` 被明确拒绝。

### 3.6 `compare/` 产物

`run_compare.py` 默认在 `outputs/compare/` 写出：

| 文件 | 内容 |
| --- | --- |
| `summary.csv` | 每个 run 一行的机器可读表格。 |
| `per_class.csv` | 各 run 的逐类 AUC、precision、recall 和 F1。 |
| `summary.md` | 汇总表和逐类表的 Markdown 版本，附生成时间、评估设备和阈值网格。 |

`summary.csv` 的列包括：

```text
run, backbone, image_size, pos_weight, seed, epochs, best_epoch,
pretrained, train_device, valid_studies, macro_auc, f1@0.3, f1@0.4, ...
```

其中 `pos_weight`、`pretrained` 显示为“是/否/未知”，`train_device` 来自 checkpoint 的 `metadata.device`。对比优先使用 `best.pt` 中的配置重建 valid；旧权重没有存档配置时回退到 `config.yaml`，表中标注配置来源与验证集路径。比较前还需确认验证集内容一致。

### 3.7 `submission.csv` 格式

提交文件是长表，一行 = 一次检查 × 一个类别：

```csv
Subject_id,Study_id,Predict_class,Probability
subject_00000,study_00000,1,0.920000
subject_00000,study_00000,2,0.030000
...
```

| 列 | 说明 |
| --- | --- |
| `Subject_id` | 患者 ID，从 `test.csv` 的 `Study_id → Subject_id` 映射补齐。 |
| `Study_id` | 检查 ID，必须覆盖 `test.csv` 中全部 study。 |
| `Predict_class` | 类别 ID，1–10。 |
| `Probability` | `[0, 1]` 内的概率。 |
| `Gate`（可选） | 类别阈值，默认 0.5；需要自定义阈值时在最后一列填入，否则该列不填。 |

> 赛题最新版提交格式在四列之后增加了可选的 `Gate` 列，用于表达自定义类别阈值（默认 0.5，不填即用默认值）。当前 `run_infer` 仍按四列输出，`validate_submission` / `read_submission` 也严格校验四列；若提交时要启用 `Gate`，需同步放开校验（尚未实现）。

两种输出模式：

- **默认（不加 `--threshold`）**：每个 study 输出全部 10 类概率，行数 = test study 数 × 10，保留完整排序信息，适合 macro-AUC。
- **加 `--threshold 0.5`**：只输出概率 ≥ 阈值的类别；若某 study 一个都没过线，保留概率最高的那一类，保证每个 study 至少一行。

写完后会自动调用校验：列名与顺序、study 双向覆盖、`Subject_id` 一致、类别范围、概率范围、重复行，以及默认模式下的 10 类齐全。校验不通过会直接报错。

### 3.8 磁盘与清理建议

- `outputs/` 已被 `.gitignore` 忽略，不会误提交。
- `last.pt` 含模型权重、最优权重快照和 AdamW 两个动量，体积约为 `best.pt` 的数倍；确认不再续训后可以删除。
- 建议至少保留每个 run 的 `config.yaml`、`metrics.csv`、`train.log` 与 `best.pt`；`last.pt` 按需保留。
- 不要手动修改 run 目录里的 `config.yaml`，否则后续 `run_eval` / `run_compare` 会按错误配置重建数据。

---

## 4. 运行指令

### 4.1 数据体检：`check_data.py`

```bash
# 快速模式：每份 CSV 只读前 1000 行
python src/check_data.py --config configs/default.yaml

# 全量模式：正式训练前至少跑一次
python src/check_data.py --config configs/default.yaml --full
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--config` | `configs/default.yaml` | 读取哪份配置里的数据路径。 |
| `--full` | 关闭 | 读取全量 CSV 并逐行核对图片；不加时每份 CSV 只查前 1000 行。 |

检查内容：必需列、ID 空值与固定格式、同 split 内 `Dicom_id` 唯一、图片齐全、同一 study 的标签与患者一致、三份 split 之间无重叠、标签分布。任一项不通过会打印原因并以退出码 1 结束，可直接串进训练前流程。

### 4.2 训练：`run_train.py`

```bash
# 按默认配置训练，固定 seed
python src/run_train.py --config configs/default.yaml --seed 42

# 给 run 一个可读的 tag
python src/run_train.py --config configs/default.yaml --seed 42 --tag full_baseline

# 用短周期配置做方向筛查
python src/run_train.py --config configs/exp_short.yaml --seed 42 --tag short_baseline
```

参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--config` | `configs/default.yaml` | YAML 配置路径。续训时省略则使用 checkpoint 里存档的配置。 |
| `--seed` | 配置里的 `seed` | 覆盖随机种子。 |
| `--tag` | `{backbone}_s{size}` | 输出目录后缀，会被清洗为安全字符。 |
| `--resume` | 无 | 断点续训，给 run 目录或其中的 `last.pt`。 |

训练过程：

1. 读配置并校验，创建 `outputs/<时间戳>_<tag>/`；
2. 写入 `config.yaml`，随后开始逐 epoch 训练；
3. 每个 epoch 结束写 `metrics.csv` 与 `last.pt`；
4. valid macro-AUC 提升时在内存中保存最优权重；连续 `early_stop_patience` 轮不提升则早停；
5. 训练结束后把最优权重写成 `best.pt`。

终端与 `train.log` 的典型输出：

```text
配置文件: configs/default.yaml
运行目录: outputs/20261027-101500_full_baseline
设备: cuda | 随机种子: 42
预训练权重: 已加载
训练 study 数: 18513 | 验证 study 数: 859
epoch  1/25 | lr 1.50e-04 | train_loss 0.4123 | valid_loss 0.3987 | macro_auc 0.7123 | macro_f1 0.3012
...
训练结束：最优 epoch 18，valid macro-AUC 0.8321
```

注意：

- `train_loss` / `valid_loss` 是按 study 加权的平均 loss，不是逐图平均。
- 默认 `warmup_epochs: 2`，所以第 1 轮学习率是 `lr/2`。
- 预训练权重加载失败时会打印警告并改用随机初始化；`metadata.pretrained_loaded` 会如实记录。

#### 断点续训

```bash
# 给 run 目录
python src/run_train.py --resume outputs/20261027-101500_full_baseline

# 也可以直接给 last.pt
python src/run_train.py --resume outputs/20261027-101500_full_baseline/last.pt
```

行为：

- 复用原 run 目录，不新建目录；`train.log` **追加**写入，并插入 `===== 续训 ... =====` 分隔行。
- 从 `last.pt` 恢复模型、优化器、学习率调度器、AMP scaler、已完成轮次、最优权重、历史记录与早停计数，接着跑剩余 epoch。
- 省略 `--config` 时，沿用 `last.pt` 里存档的配置；显式传 `--config` 时，只允许修改 `train.epochs`、`train.num_workers`、`eval.batch_studies`、`device` 和 `output`，其余差异会报错。续训配置另存为 `config_resume.yaml`，原 `config.yaml` 不动。
- 如果该 run 已跑满 `train.epochs`，不会重复训练，只重新导出 `best.pt`。
- 训练被 Ctrl+C 中断时，最多丢失当前这一轮；用上面的 `--resume` 命令继续。
- 续训保存的是模型、优化器、调度器、AMP 与训练进度，不含随机数生成器状态；恢复后数据打乱与增强会从 `seed` 重新开始，指标可能与“不中断一次跑完”有细微差异。正式对比仍应以完整 run 为准。

延长训练轮数的写法：

```bash
python src/run_train.py --resume outputs/<run> --config configs/exp_long.yaml
```

`exp_long.yaml` 只需在原训练配置的基础上把 `train.epochs` 调大。学习率、权重衰减、数据与预处理等训练口径不可通过续训配置更改；需要改变这些参数时应新建 run。

> 只有训练过程中生成的 `last.pt` 才能续训。`best.pt` 没有优化器、调度器和 `extras`，拿它续训会报错。

### 4.3 验证：`run_eval.py`

```bash
# 在 valid 上评估
python src/run_eval.py \
    --config outputs/<run>/config.yaml \
    --checkpoint outputs/<run>/best.pt

# 同时看多个阈值下的 macro-F1
python src/run_eval.py \
    --config outputs/<run>/config.yaml \
    --checkpoint outputs/<run>/best.pt \
    --thresholds 0.3,0.4,0.5,0.6,0.7

# 在 train split 上评估（排查过拟合）
python src/run_eval.py \
    --config outputs/<run>/config.yaml \
    --checkpoint outputs/<run>/best.pt \
    --split train
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--config` | `configs/default.yaml` | 建议传 run 目录里的 `config.yaml`，保证图像尺寸、归一化与训练一致。 |
| `--checkpoint` | 必填 | 训练产出的 `best.pt`。 |
| `--split` | `valid` | 只能是 `train` 或 `valid`；test 没有标签，不能用于评估。 |
| `--thresholds` | 无 | 逗号分隔的阈值网格，例如 `0.3,0.4,0.5`；每个阈值须在 `[0, 1]`。给了就额外打印每个阈值下的 macro-F1。 |

输出包括：macro-AUC、按 `eval.threshold` 计算的 macro-F1、可选的多阈值 F1 表、逐类明细（`class_id`、`support`、`pos_rate`、`auc`、`precision`、`recall`、`f1`）。AUC 不受阈值影响；退化类别的 AUC 为 NaN，不参与 macro 平均。

### 4.4 推理：`run_infer.py`

```bash
# 默认：输出全部 10 类概率
python src/run_infer.py \
    --config outputs/<run>/config.yaml \
    --checkpoint outputs/<run>/best.pt \
    --out outputs/submission.csv

# 可选：只输出概率 >= 0.5 的类别
python src/run_infer.py \
    --config outputs/<run>/config.yaml \
    --checkpoint outputs/<run>/best.pt \
    --out outputs/submission.csv \
    --threshold 0.5
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--config` | `configs/default.yaml` | 建议用 run 目录里的 `config.yaml`，确保模型结构与预处理一致。 |
| `--checkpoint` | 必填 | 训练产出的 `best.pt`。 |
| `--out` | `outputs/submission.csv` | 提交文件路径，父目录会自动创建。 |
| `--threshold` | 无 | 只输出概率 ≥ 该值的类别，必须在 `[0, 1]`。省略时输出全部 10 类概率。 |

推理流程：按配置构建 test loader → 加载权重 → 逐图前向并按配置聚合 study logits → sigmoid → 摊平成提交长表 → 按 `test.csv` 校验。终端会打印 checkpoint、设备、test study 数、提交文件路径、输出策略与行数。

建议始终使用 run 目录里的配置：

```bash
python src/run_infer.py \
    --config outputs/20261027-101500_full_baseline/config.yaml \
    --checkpoint outputs/20261027-101500_full_baseline/best.pt \
    --out outputs/submission.csv
```

因为 `config.yaml` 记录的是训练时的 `backbone`、聚合方式与权重、`image.size` 和数据路径，能避免“用新默认配置推理旧权重”导致的结构或预处理不一致。

### 4.5 多 run 对比：`run_compare.py`

```bash
python src/run_compare.py \
    --runs outputs/<ts>_short_baseline outputs/<ts>_short_pos_weight \
    --thresholds 0.3,0.4,0.5,0.6,0.7 \
    --out outputs/compare
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--runs` | 必填 | 一个或多个 run 目录，每个目录必须同时有 `config.yaml` 与 `best.pt`。 |
| `--thresholds` | `0.3,0.4,0.5,0.6,0.7` | 计算各阈值下 macro-F1 的网格，每个阈值须在 `[0, 1]`。 |
| `--out` | `outputs/compare` | `summary.csv` / `per_class.csv` / `summary.md` 的输出目录。 |
| `--device` | `auto` | 评估设备；与 run 里的配置无关。 |

脚本对每个 run：优先用 `best.pt` 的配置重建 valid loader → 重算 macro-AUC、多阈值 F1 和逐类指标 → 汇总配置要点。参与对比的 run 必须使用同一份验证数据，否则指标不是同一口径。

### 4.6 测试

```bash
python -m pytest
```

测试使用 `tests/` 下临时生成的合成小图，不读取真实影像；覆盖标签解析、索引、数据集、数据体检、指标、模型、训练、配置、续训、提交和端到端链路。

### 4.7 入口与参数速查

| 入口 | 必填参数 | 常用可选参数 | 主要产物 |
| --- | --- | --- | --- |
| `src/check_data.py` | 无 | `--config`、`--full` | 终端检查报告，失败退出码 1 |
| `src/run_train.py` | 无 | `--config`、`--seed`、`--tag`、`--resume` | run 目录：`config.yaml`、`train.log`、`metrics.csv`、`best.pt`、`last.pt` |
| `src/run_eval.py` | `--checkpoint` | `--config`、`--split`、`--thresholds` | 终端指标与逐类明细 |
| `src/run_infer.py` | `--checkpoint` | `--config`、`--out`、`--threshold` | `submission.csv` |
| `src/run_compare.py` | `--runs` | `--thresholds`、`--out`、`--device` | `summary.csv`、`per_class.csv`、`summary.md` |

---

## 5. 典型工作流

按数据体检 → 对照训练 → 同一验证集评估 → 最终训练 → 推理提交的顺序执行。各入口命令见第 4 节；实验矩阵与判读见 [experiments.md](experiments.md)，具体执行步骤见 [训练计划](train_plan01.md)，实测结果见 [training_log.md](training_log.md)。

---

## 6. 排错速查

| 现象 | 常见原因 | 处理 |
| --- | --- | --- |
| `FileNotFoundError: data/train.csv` 或图片根目录不存在 | 不在仓库根目录执行，或数据未放到约定位置 | `cd` 到仓库根目录；或把配置里的路径改成绝对路径 |
| 报错“缺少必需的列” | CSV 缺 `Dicom_id` / `Subject_id` / `Study_id`，或 train/valid 缺 `Predict_class` | 核对 CSV 表头；test 不应有标签列要求 |
| 报错“找不到图片”或图片数少于 CSV 行数 | 文件名 stem 与 `Dicom_id` 不一致、后缀不在 `.jpg/.jpeg/.png`、图片目录指错 | 用 `check_data.py --full` 定位；确保 `Dicom_id` 等于文件名去后缀 |
| 报错“图片名重复” | 不同子目录下有同名图片 | 保证 `Dicom_id` 全局唯一，整理目录 |
| 报错“同一 study 的标签不一致” / “对应多个 Subject_id” | 标注导出错误 | 修正 CSV；同一 study 必须共享标签且属于同一患者 |
| 配置报“含未知字段” | 字段名拼写错误 | 对照第 2 节字段表；注意 `train.epoch` 应为 `train.epochs` |
| `train.lr 应该是数字，实际是 '3e-4'` | YAML 1.1 把科学计数法读成字符串 | 写成 `0.0003` |
| `CUDA out of memory` | batch 太大或图片太大 | 降 `train.batch_studies`（16 → 8 → 4）、降 `train.num_workers`；必要时降 `image.size` 并重新训练 |
| `配置要求使用 cuda，但当前环境没有可用的 CUDA` | 显式指定了不可用设备 | 改 `device: auto`、`cpu` 或 `mps` |
| 日志出现“预训练权重加载失败” | 离线或权重缓存缺失 | 忽略并继续（随机初始化），或设置 `model.pretrained: false`，或提前准备缓存 |
| `验证集所有类别的 AUC 都是 NaN` | valid 太小，或某些类别全阴/全阳 | 增加验证样本；检查 valid 划分；不要用抽样后的小 valid 做正式训练 |
| 早停过早 | `early_stop_patience` 太小，或 valid AUC 波动 | 调大 patience；短周期实验可设 `0` 关闭早停 |
| `找不到 .../last.pt` | run 目录里没有 `last.pt`，或路径写错 | 只有训练过程保存过 `last.pt` 的 run 才能续训；确认路径 |
| `不是训练断点（缺少 extras）` | 用 `best.pt` 续训 | 改用 `last.pt` 或 run 目录 |
| 续训后指标对不上 | 改了 backbone / 图像尺寸 / 归一化 / 类别数 | 续训只改 `epochs`、`num_workers` 等非结构字段；结构变更请从头训练 |
| `run_compare` 报缺 `config.yaml` 或 `best.pt` | 传入了非 `run_train.py` 产物目录 | 指向包含这两个文件的 run 目录 |
| 提交校验报“缺少 test study” | 推理用的 `test.csv` 与提交校验用的不是同一份，或推理中途失败 | 用与训练一致的配置和 test.csv 重新推理 |
| 提交校验报“只有 N 个类别” | 加了 `--threshold`，但用了默认全类校验逻辑 | 这是预期行为：阈值过滤模式下不要求 10 类齐全；默认不加阈值时应为 10 类 |
| `Predict_class 必须在 1~10` | 标签用了 0–9 下标 | 标注与提交统一用赛题类别 ID 1–10 |
| 同一秒训练两次报目录已存在 | `make_run_dir` 使用 `exist_ok=False` | 换一个 `--tag` 或稍后重试 |
| Windows 上 DataLoader 报多进程错误 | `num_workers > 0` 的启动方式问题 | 临时设 `train.num_workers: 0` |
| MPS 上遇到不支持的算子 | Apple MPS 后端限制 | 改 `device: cpu`，或改用 CUDA 环境 |

---

## 7. 相关文档

| 文档 | 内容 |
| --- | --- |
| [赛题说明](多目标胸部病变检测分类（ChestX-MLC）.md) | 任务、数据格式、评价指标与提交要求 |
| [架构与文件结构](architecture.md) | 模块职责、接口与数据流 |
| [P7 调优记录](experiments.md) | 实验矩阵、判读与冻结结论 |
| [训练计划](train_plan01.md) | 训练步骤与判读门 |
| [训练记录](training_log.md) | 各 run 的配置与实测结果 |
| [耗时自检](cost.md) | 计时口径与限制 |
| [实现计划](plan0.md) | P1–P8 的阶段划分与验收方式 |
| `CHANGELOG.md` | 全部代码与文档改动记录 |
