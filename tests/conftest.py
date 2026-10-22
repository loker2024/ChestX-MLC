"""pytest 全局配置：把 src/ 加入模块搜索路径。

仓库约定的导入根是 src/ 本身，而不是把 src 当成包：
源码里写 `from data.index import read_split`，测试里也就用同样的顶层导入，
不需要安装项目。pytest 启动时会先执行 tests/conftest.py，因此在这里插入路径
对所有测试文件都生效。
"""

import sys
from pathlib import Path

# __file__ 是 tests/conftest.py，parents[1] 就是仓库根目录，再拼上 src/。
SRC_DIR = Path(__file__).resolve().parents[1] / "src"

# 放在 sys.path 最前面，保证测试导入的是仓库里的源码，而不是环境里同名的包。
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
