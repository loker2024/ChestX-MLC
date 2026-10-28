历史实验：在 RUN-003 的基础上，只改变学习率调度策略。当前 `configs/exp_cosine.yaml` 显式关闭 `pos_weight`，以保留当时设置；其他字段继承当前 `default.yaml`。

| 配置项 | RUN-003-A |
|---|---|
| 随机种子 | 42 |
| 学习率 | 初始峰值 0.0003 |
| 调度器 | 前 2 轮线性 warmup，之后按余弦衰减至 0.00001 |
| 最大 epoch | 25 |
| early-stop patience | 5 |
| pos_weight | 关闭 |
| 其他配置 | 与 RUN-003 保持一致（含 `num_workers: 16`） |

```bash
python src/run_train.py --config configs/exp_cosine.yaml --seed 42 --tag run003a_cosine
```

训练后，与 RUN-003 比较最佳 epoch、Macro-AUC、逐类 AUC 和训练曲线。两次若提前早停，比较各自实际选出的最佳 checkpoint。