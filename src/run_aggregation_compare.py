"""聚合空间对照入口：export 逐图 logits 缓存，compare 离线重算 G0 / G1。

用法（在仓库根目录下执行）：

    python src/run_aggregation_compare.py export \
        --run outputs/<run> \
        --device cuda \
        --out outputs/run008_aggregation_s42

    python src/run_aggregation_compare.py compare \
        --cache outputs/run008_aggregation_s42 \
        --thresholds 0.2,0.3,0.4,0.5,0.6,0.7

具体实现与口径见 `infer.aggregation_compare`：G0 = sigmoid(mean(logits))，
G1 = mean(sigmoid(logits))；export 只做一次逐图前向，compare 不加载模型。
"""

from infer.aggregation_compare import main

if __name__ == "__main__":
    main()
