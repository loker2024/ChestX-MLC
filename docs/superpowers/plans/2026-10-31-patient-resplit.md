# 患者级重划分实现计划

**目标：** 可选合并官方训练、验证数据并按患者随机划出新验证集；移除当前分支中的 clean-mask 实现（如存在）。

**架构：** 配置默认关闭；按患者生成并保存含数据来源、标签和图片 ID 的划分清单。训练、续训和带标注评估核验清单及 checkpoint 摘要，提交 test 不改变。验证集退化类别提示，跨划分对比拒绝。

1. 在 `tests/test_resplit.py` 覆盖患者隔离、确定性、源数据变化、重叠与清单绑定。
2. 在 `src/common/config.py`、`src/data/resplit.py` 和 `src/data/dataset.py` 实现配置、清单及 loader。
3. 接入 `src/run_train.py`、`src/run_eval.py`、`src/run_compare.py`、`src/run_ensemble.py`；续训保持原划分。
4. 检查并删除当前分支的 clean-mask 代码和文件；更新配置、手册、架构和 changelog。
5. 执行合成数据测试、完整 pytest、diff 检查；不启动 GPU 训练。
