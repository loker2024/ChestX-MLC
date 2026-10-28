
使用现有配置 configs/exp_pos_weight.yaml，分别训练两个随机种子：

| 实验 | seed | 最大 epoch | early-stop patience | pos_weight |
|---|---:|---:|---:|---|
| RUN-002-A | 42 | 25 | 5 | 开启 |
| RUN-002-B | 2026 | 25 | 5 | 开启 |

两次均使用现有完整训练配置，除 seed 外保持一致。设置的是最多 25 轮，可能提前早停。

python src/run_train.py --config configs/exp_pos_weight.yaml \
    --seed 42 --tag pos_weight_s42

python src/run_train.py --config configs/exp_pos_weight.yaml \
    --seed 2026 --tag pos_weight_s2026