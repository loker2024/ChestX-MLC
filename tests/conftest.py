"""pytest 全局配置：把 src/ 加入模块搜索路径，并提供合成数据 fixture。

仓库约定的导入根是 src/ 本身，而不是把 src 当成包：源码里写
`from data.index import read_split`，测试里也就用同样的顶层导入，不需要
安装项目。pytest 启动时会先执行 tests/conftest.py，因此在这里插入路径
对所有测试文件都生效。

下面的 fixture 都在 tmp_path（每个测试独立的临时目录）里造数据：CSV 用
csv 模块写，图片用 PIL 生成纯色小图。全程不碰真实影像——既跑得快，
也避免把赛题数据带进仓库。
"""

import csv
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

# __file__ 是 tests/conftest.py，parents[1] 就是仓库根目录，再拼上 src/。
SRC_DIR = Path(__file__).resolve().parents[1] / "src"

# 放在 sys.path 最前面，保证测试导入的是仓库里的源码，而不是环境里同名的包。
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 赛题标注 CSV 的完整表头。图片相关的列目前不参与训练，但要照原样写出来，
# 这样测试用的 CSV 和真实文件长得一样。
LABELED_HEADER = [
    "Dicom_id",
    "Subject_id",
    "Study_id",
    "ViewPosition",
    "PatientOrientation",
    "Rows",
    "Columns",
    "Predict_class",
]


@pytest.fixture
def make_image(tmp_path: Path):
    """返回一个建图函数：在 tmp_path 下按相对路径生成一张合成小图。

    用法：`make_image("subject_00000/study_00000/pic_000000.png")`。
    默认尺寸 64×48（非方形），正好用来验证 letterbox 没有把图拉变形。
    """

    def _make(relative: str, size: tuple[int, int] = (64, 48)) -> Path:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (120, 120, 120)).save(path)
        return path

    return _make


@pytest.fixture
def write_csv(tmp_path: Path):
    """返回一个写 CSV 的函数：给定文件名和数据行，写出 UTF-8 文件。

    表头默认用带标注的完整表头；需要「没有标签列」的场景（test split）时，
    显式传 `header=LABELED_HEADER[:-1]`。

    显式 encoding="utf-8"：Windows 默认是 GBK，不指定会写出乱码。
    """

    def _write(
        name: str, rows: list[list[str]], header: list[str] = LABELED_HEADER
    ) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    return _write


@pytest.fixture
def labeled_split(tmp_path: Path, make_image, write_csv):
    """造一个带标注的迷你 split：2 个 study，共 3 张图。

    目录结构与赛题一致（`<subject_id>/<study_id>/<dicom_id>.jpg`），并且故意
    让 study_00000 有两张图（AP + LATERAL），用来验证「同一检查多图」的聚合；
    study_00001 只有一张图，覆盖单图的情况。

    返回 dict：`csv` 标注文件、`images` 图片根目录、`rows` 写进 CSV 的原始行。
    """
    image_entries = [
        "subject_00000/study_00000/pic_000010.png",
        "subject_00000/study_00000/pic_000011.png",
        "subject_00001/study_00001/pic_000012.png",
    ]
    for relative in image_entries:
        make_image(f"images/{relative}")

    rows = [
        ["pic_000010", "subject_00000", "study_00000", "AP", "Erect", "48", "64", "7;6"],
        ["pic_000011", "subject_00000", "study_00000", "LATERAL", "Erect", "48", "64", "7;6"],
        ["pic_000012", "subject_00001", "study_00001", "AP", "Erect", "48", "64", "10"],
    ]
    return {
        "csv": write_csv("train.csv", rows),
        "images": tmp_path / "images",
        "rows": rows,
    }


@pytest.fixture
def unlabeled_split(labeled_split, write_csv):
    """同样的图片，但 CSV 去掉 Predict_class 列，模拟赛题的 test split。

    图片是复用的，所以这里的行数、目录都和 labeled_split 完全一致，只是
    test split 确实没有标签这一列。
    """
    rows = [row[:-1] for row in labeled_split["rows"]]
    return {
        "csv": write_csv("test.csv", rows, header=LABELED_HEADER[:-1]),
        "images": labeled_split["images"],
        "rows": rows,
    }


@pytest.fixture
def write_config(tmp_path: Path):
    """返回一个写 YAML 配置的函数，把各个 split 指向测试造的临时数据。

    用法：`write_config({"train": (csv, images), "valid": (csv, images),
    "test": (csv, images)})`。

    路径统一过 json.dumps：JSON 字符串本身就是合法的 YAML 字符串，既能包住
    含空格的路径，也能避免 Windows 路径里的反斜杠被 YAML 当成转义符。

    测试里把 num_workers 固定为 0、image.size 压到 32：不起子进程、图也小，
    单个测试能在毫秒级跑完。
    """

    def _write(splits: dict[str, tuple[Path, Path]]) -> Path:
        lines = ["data:"]
        for name, (csv_path, images) in splits.items():
            lines.append(f"  {name}:")
            lines.append(f"    csv: {json.dumps(str(csv_path))}")
            lines.append(f"    images: {json.dumps(str(images))}")
        lines.extend(
            [
                "image:",
                "  size: 32",
                "train:",
                "  batch_studies: 2",
                "  num_workers: 0",
                "eval:",
                "  batch_studies: 2",
            ]
        )
        path = tmp_path / "config.yaml"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    return _write
