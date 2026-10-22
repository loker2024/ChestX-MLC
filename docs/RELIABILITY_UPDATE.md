# 可靠性修订与上传清单（2026-10-22）

本文件描述本地修订，不代表 GitHub 分支已经更新。上传到团队仓库
`loker2024/ChestX-MLC` 的 `yun` 分支，PR 方向为 `yun → develop`，由维护者审核合并。

## 本次完成

1. 预测 `--data-config` 可覆盖 CSV/图片路径；尺寸、预处理、批量保持检查点设置，实际路径写入报告。
2. 历史参考 JSON 损坏、重复 ID、缺列、NaN 指标在默认模式下记录 invalid 并告警；严格模式失败。
   当前数据/标签/预测本身的错误仍必须停止，不被参考文件容错掩盖。
3. 登记前完整验证输入，在同盘隐藏暂存目录内复制并校验哈希，原子发布到新实验编号。
   JSON/复制失败不会占用最终编号；汇总跳过坏旧目录、支持空表、使用原子文件替换。
4. 评估记录 ID归属/标签语义指纹和评估协议，实验只在同组内按 Macro-AUC 排名。
   旧指标没有指纹时保留但不排名；可通过同一模型的新验证运行建立独立新快照，不能补造历史身份。
5. 新检查点保存完整 epoch 边界状态，恢复模型绑定、AdamW、Python/NumPy/PyTorch随机状态、
   设备随机状态、训练及验证 DataLoader随机状态。完整恢复不能同时调参或换数据/设备类型，要求num_workers=0。
   旧状态缺少所需字段时明确拒绝完整恢复；仍可使用仅权重的新实验。不能恢复任意batch中断点。
6. 从零训练入口支持独立 --output、--data-config 和 --seed，正式数据规模校验默认保留。
   --allow-nonofficial-counts 仅用于合成小样本测试，并写入配置和报告。

## 已实际执行

```powershell
python -m unittest discover -s tests -v
python -m src.predict --model outputs/experiments/exp_002/best_model.pth --split valid --data-config configs/data.json
```

- 18项测试通过，最后一轮测试耗时7.177秒。
- 换机路径的完整CPU验证/测试预测通过（合成样本12次检查/24张图）。
- 训练入口的完整恢复集成测试使用极小合成数据：训练和验证各4张图，CPU ResNet18，
  连续运行与断开后从完整epoch状态恢复，到相同轮次的所有参数逐项完全一致。
  合成测试成绩不是比赛成绩，测试临时模型不保留为正式实验。
- 真实第4轮权重验证：1422张图、859次检查；Macro-AUC=0.730568169109816、
  Macro-F1=0.15771485793395476，与历史结果一致，原权重不变。
- 实际验证目录：outputs/experiments/exp_002/predictions/20261022_181637_c6140eaf/。
  注册为独立exp_003验证身份快照，模型哈希与原第4轮相同，不是新增训练轮次。

## 依赖与尚未验证的范围

新增 requirements-baseline.txt，固定本机使用的六个直接依赖；原requirements.txt保留。
本机实际最小基线依赖闭包共22个包，约束检查未发现冲突；全局pip check仍有其他项目d2l的7项冲突。
没有卸载或改装全局环境，也没有完成新venv的下载安装验收。最小清单不等于完整传递依赖锁文件，
不保证官方Python3.12/PyTorch2.10镜像兼容。隔离安装步骤见README。

MPS、官方T4和正式平台提交语义仍待外部验收。数据指纹不包含图片字节，要求官方图片保持不变。
完整状态恢复已在同机CPU上实测，不保证跨硬件/库版本逐位相同。没有正式数据训练、平台上传或Git操作。

## 本次需要上传的17个文件

以下是GitHub仓库内的路径，电脑上的文件均在 `C:\计算机能力挑战赛` 下。

| GitHub路径 | 类型 |
| --- | --- |
| src/predict.py | 修改 |
| src/experiments.py | 修改 |
| src/evaluate.py | 修改 |
| src/full_train.py | 修改 |
| src/train.py | 修改 |
| src/continue_train.py | 修改 |
| src/fingerprints.py | 新增，不能漏传 |
| src/training_state.py | 新增，不能漏传 |
| tests/test_experiments.py | 修改 |
| tests/test_predict.py | 修改 |
| tests/test_reliability.py | 新增 |
| tests/test_training_state.py | 新增 |
| README.md | 修改，根目录 |
| CHANGELOG.md | 修改，根目录；保留团队14:23撤回记录 |
| requirements-baseline.txt | 新增，根目录 |
| docs/PR_REVIEW_RESPONSE.md | 修改 |
| docs/RELIABILITY_UPDATE.md | 新增，本说明 |

不要上传 outputs、数据、权重、.venv 或 __pycache__。configs/data.json 和原 requirements.txt 本轮没有修改。
上传同名文件会更新内容；如果yun分支上已有同学的独立修改，先对比再更新，避免整文件覆盖他人的改动。
提交说明建议：`fix: 完善预测路径、实验登记与状态恢复`；测试和文档可分别用test:/docs:类型提交。

## 可用于PR更新的说明

修复预测数据路径无法换机覆盖、可选历史参考损坏阻断预测、失败登记污染后续实验的问题。
增加验证集及协议指纹分组排名，增加经过集成测试的完整epoch状态恢复；原权重和历史实验保留。
18项测试通过，真实第4轮模型全量验证Macro-AUC仍为0.730568，Macro-F1为0.157715。
新增最小依赖清单与隔离安装说明，未宣称全局环境、MPS或官方T4已通过验收。
