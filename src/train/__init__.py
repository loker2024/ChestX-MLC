"""训练包：逐 epoch 的训练 / 验证循环、学习率 warmup、早停与最优权重挑选。

这里只放模块说明，不导入 torch：`import train` 不应该顺带把深度学习框架
加载进来。需要训练循环时从具体模块导入，例如 `from train.engine import fit`。
"""
