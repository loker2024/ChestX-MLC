# A-FUSE-5 推理耗时优化计划

状态：计划，尚未执行。目标是在保持最终提交预测一致和完整计时口径的前提下，降低单 study 平均耗时，并为 100 ms 上限留出余量。

## 1. 对照对象和计时口径

- 固定 A-FUSE-5：D-002-a-EMA、C-005-c、E-001，logit 融合权重 4:2:3；核对实际 checkpoint、顺序、内嵌配置、融合清单、视角元数据与 submission。现有 93.6 ms 是 A-FUSE-2 的结果，不能作为 A-FUSE-5 的基线。
- 记录 checkpoint、清单、submission、视角文件及抽样清单的 SHA-256，代码 HEAD 和完整未提交 diff，Python、PyTorch、CUDA、cuDNN、GPU、CPU、存储位置及运行负载。
- 初始基线为 FP32、`--preprocess-workers 6`；不改变模型、预处理、聚合、融合权重和提交内容。
- 使用相同的 100 个完整 study，保留每个 study 的全部图片；另用 5 个不相交的预热 study。记录抽样方式和每 study 图片数分布。
- 每次运行连续 3 轮，以 `billed_mean` 判定；读取、解码、预处理、全部模型前向、study 聚合、融合、逐图写出、flush 与轮级文件 IO 均计入。不得预先缓存图像像素或预测。具体口径参见 [cost.md](cost.md)。

## 2. 建立基线

在 GPU 空闲的同一环境独立运行 5 次，保存各次报告和预测文件。先设置下列变量为实际路径；检查清单里的 checkpoint 路径、顺序及权重：

```bash
: "${SUBMISSION:?设置 A-FUSE-5 submission.csv 路径}"
: "${PICTURES_100:?设置 100 study 图片目录}"
: "${PICTURES_5:?设置 5 study 预热目录}"
: "${ENSEMBLE:?设置 A-FUSE-5 cost_ensemble.json 路径}"
: "${VIEWS_CSV:?设置包含所需视角的 CSV 路径}"
: "${REPORT_DIR:?设置本次报告目录}"
mkdir -p "$REPORT_DIR"
for run in 1 2 3 4 5; do
    .venv/bin/python src/cost.py \
        "$SUBMISSION" "$PICTURES_100" "$PICTURES_5" \
        --ensemble "$ENSEMBLE" \
        --views-csv "$VIEWS_CSV" \
        --device cuda --precision fp32 --preprocess-workers 6 \
        --prediction-out "$REPORT_DIR/baseline_predictions_${run}.csv" \
        --report "$REPORT_DIR/baseline_${run}.json"
done
```

逐次检查退出码与报告，不把失败的运行计作成功；如果样本不满足完整 study 或预热不相交等要求，先修正样本再测。记录每次 `billed_mean`、study 中位数、p95、最大值、一致性最大差、超限状态和 GPU 负载，汇总 5 次的中位数与最慢值。5 次预测均通过 submission 比对后再进入优化。

工程目标：5 次独立运行的 `billed_mean` 均不超过 90 ms/study。这是给 100 ms 上限留余量的内部目标，不是官方门槛；达不到时如实报告。

## 3. 分段诊断

建立独立诊断模式或临时脚本，在同批 study 上测量：图片读取与解码、缩放与归一化、CPU→GPU 传输、各模型前向与 study 聚合、GPU→CPU 回传、融合与写出。分别报告单图和多图 study。

GPU 分段测量可能需要同步或 CUDA event，因此分段数值只用于定位瓶颈；正式成绩仍以不加诊断同步的原始端到端路径为准。

## 4. 逐项优化与对照

每项单独与固定基线比较，保留可追溯的代码快照，不把多个变量一起改后归因于某一项。每项 5 次基线与 5 次候选交替运行，记录每对差值、5 次中位数与最慢值；发现机器负载干扰时标记并重测。

### O1：复用预处理线程池

- 将 [predict.py](../src/infer/predict.py) 的线程池生命周期扩大到一次 `cost.py` 运行，预热与正式计时共用，异常和退出时关闭。任务仍限于当前 study，不跨 study 预读或预处理；`max_workers=1` 保持串行。
- 测试单图、多图、多预处理签名、线程数为 1 和异常清理；逐 study 比较原实现与候选概率。

### O2：复用当前 study 的设备输入

- 在单个 study 内按预处理签名、目标设备和 dtype 缓存设备输入；不同签名或 dtype、未知签名不能错误共享。offsets 与视角信息可按需在同 study 复用；缓存不得跨 study。
- 第一版保持各模型 logits 单独回 CPU、CPU 上 sigmoid、NumPy 融合的现有顺序，隔离输入复用的影响。
- 测试相同及不同签名、不同 dtype、缺失或不可哈希 key、单图与多图、正侧位及未知视角、不同聚合模式；对比逐模型 study logits、最终融合概率与 submission，并确认传输次数减少。

### O3：减少逐模型回传与同步

仅在 O2 完成且诊断显示回传或同步有可观成本时进行。将各模型 study logits 在设备上保留后集中回传，再按原顺序在 CPU 上做 sigmoid 与融合；保留计时边界同步，确保设备工作全部计入。

比较各模型 logits、概率融合和 logit 融合结果，重点覆盖 A-FUSE-5 的 4:2:3；同时检查显存及不同 study 图片数量下的行为。

### O4：线程数与预处理热点

在 O1–O3 的有效实现基础上，单独对比 `--preprocess-workers 1 / 2 / 4 / 6 / 8`；固定机器、代码、样本、模型和精度，每组至少独立运行 5 次。若解码或缩放仍占主要时间，再研究相应实现。更换解码库、插值或归一化可能改变数值，先比较像素、张量及最终预测，无法保持可接受一致性则不采用。

CUDA Graph、`torch.compile`、FP16 和量化列为第二阶段：仅当上述方案不能达到目标且诊断证实 GPU 前向占主要时间时另行评估。既有 FP16 测试未显示稳定收益。

## 5. 验收门槛

1. **正确性**：模型顺序、视角聚合、融合公式不变；完整测试通过；100 study 与既有 submission 比对不一致数为 0，记录最大概率绝对差。默认 `1e-3` 仅为 submission 比对上限；另记录实现前后更精细的数值差异，不靠放宽阈值掩盖变化。
2. **性能**：同机同样本交替运行 5 对，候选中位改善至少 2 ms/study 且方向稳定，才作为默认优化；不足 2 ms 且明显增加复杂度的方案不纳入默认路径。
3. **计时完整性**：不跨 study 预读或缓存预测，读图到写出全部计入；不得通过省略同步或文件 IO 获得虚假的加速。
4. **回归**：运行 `tests/test_cost.py`、相关数据集及推理测试、全量 `pytest` 和 `git diff --check`。
5. **最终测量**：固定最终组合后完整复测；最终合规以主办方指定 T4 环境和复测为准。

## 6. 交付与记录

交付优化代码及缓存边界、数值一致性、资源释放测试；保存基线与各候选报告、环境、模型和数据 hash、逐次耗时表。更新 [cost.md](cost.md) 与 `CHANGELOG.md`。

如将结果登记到 `docs/training_log.md`、`docs/experiments.md` 或训练计划的实验执行记录，依仓库规定同步更新仓库外实验表格。单独在 `cost.md` 记录工程耗时测试不预设为训练实验。最终说明实际采用的优化、各项实测收益、预测差异及尚未完成的 T4 验证。
