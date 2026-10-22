"""把 CSV 里的标签字符串解析成 multi-hot 向量。

赛题的 Predict_class 列长这样：`7;6;9`，分号分隔若干类别 ID，ID 取值 1~10。
模型需要的是长度 10 的 0/1 向量，约定「数组下标 = 类别 ID - 1」：

    "7;6;9"  ->  [0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
                  ↑下标 0 是类别 1              ↑下标 8 是类别 9

这个换算只写在这一处，其他地方一律用「下标」说话，避免 0 基和 1 基混用时
出现差一位的隐蔽 bug。
"""

import math
from typing import Any

import numpy as np

from common.constants import NUM_CLASSES

# 标签字符串里使用的分隔符，与 CSV 中一致。
LABEL_SEPARATOR = ";"


def parse_label_string(value: Any) -> np.ndarray:
    """把 `"7;6;9"` 这样的标签串转成 float32 的 10 维 multi-hot 向量。

    返回 float32 而不是 int：下游直接拼进张量喂给 BCEWithLogitsLoss，
    省掉一次类型转换，也避免和 logits 做运算时的隐式提升。

    空值（None、NaN、空串、纯空白）表示「这次检查没命中任何类别」，返回全 0。

    重复类别（如 `"7;7"`）和超出 1~10 的 ID 一律报错：
    这类问题几乎都是标注导出环节出了错，静默去重或截断会让脏数据一路带进训练，
    等到指标异常时已经很难追查来源。
    """
    vector = np.zeros(NUM_CLASSES, dtype=np.float32)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return vector

    text = str(value).strip()
    if not text:
        return vector

    for token in text.split(LABEL_SEPARATOR):
        token = token.strip()
        if not token:
            # 例如 "7;;9" 或末尾多了分号，说明这一行明显有问题。
            raise ValueError(f"标签串 {value!r} 含空片段，正确写法形如 '7;6;9'")

        try:
            class_id = int(token)
        except ValueError:
            raise ValueError(
                f"标签串 {value!r} 里的 {token!r} 不是整数类别 ID"
            ) from None

        if not 1 <= class_id <= NUM_CLASSES:
            raise ValueError(
                f"标签串 {value!r} 里的类别 ID {class_id} 超出 1~{NUM_CLASSES} 的范围"
            )

        index = class_id - 1
        if vector[index] != 0:
            raise ValueError(f"标签串 {value!r} 里类别 {class_id} 重复出现")
        vector[index] = 1.0

    return vector
