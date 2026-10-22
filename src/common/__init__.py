"""跨模块共用的基础设施：常量、配置、工具函数与 checkpoint 读写。

这里只放模块说明，刻意不导入任何子模块：
`import common` 时不应该连带加载 torch / PyYAML 这些较重的依赖。
需要什么就从具体模块里单独导入，例如 `from common.config import load_config`。
"""
