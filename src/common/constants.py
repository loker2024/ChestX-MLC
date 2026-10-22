"""全局常量：类别数、类别 ID、图像尺寸与归一化参数。

赛题把 10 个类别编号为 1~10（CSV 的 Predict_class 列里写的就是这些数字），
而模型输出是长度 10 的向量，所以代码里统一约定：

    类别 ID（1~10）  <-->  数组下标（0~9），即 下标 = 类别 ID - 1

所有需要“类别 ID”的地方都从 CLASS_IDS 取，避免手写 1~10 时漏项或写错。
"""

# 类别总数，赛题固定为 10 类。
NUM_CLASSES = 10

# 标注 CSV 里使用的类别 ID：1, 2, ..., 10。
# 下标 i 对应类别 ID 为 CLASS_IDS[i]，也就是 i + 1。
CLASS_IDS = tuple(range(1, NUM_CLASSES + 1))

# 默认输入边长（像素）。320 = 32 × 10，是 32 的整数倍，
# 与 ResNet 的 32 倍下采样步长对齐，也是 ImageNet 预训练权重的常用输入尺寸。
DEFAULT_IMAGE_SIZE = 320

# 递归索引图片时认可的后缀。比较前统一转小写，所以 .JPG 也能命中。
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# ImageNet 预训练模型的 RGB 三通道均值与标准差（0~1 量纲）。
# 使用预训练权重时按这套参数归一化，才能和权重训练时的输入分布对上。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 读写 CSV 时显式使用的编码。赛题要求 UTF-8；Windows 默认是 GBK，
# 不显式指定容易产生中文乱码或写出失败。
CSV_ENCODING = "utf-8"
