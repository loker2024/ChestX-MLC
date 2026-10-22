"""模型包：网络结构（逐图 forward + study 级均值聚合）与训练损失。

这里只放模块说明，不导入 timm / torch：`import model` 不应该顺带把整个
深度学习框架加载进来。需要什么就从具体模块导入，例如
`from model.network import build_model, aggregate_logits`。
"""
