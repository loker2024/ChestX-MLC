"""标签解析：分号分隔、类别 10、空值、重复与非法 token。"""

import numpy as np
import pytest

from data.labels import parse_label_string


def test_single_class_returns_multi_hot():
    """类别 ID 10 落在下标 9 上，这是最容易差一位的地方。"""
    vector = parse_label_string("10")

    assert vector.shape == (10,)
    assert vector.dtype == np.float32
    np.testing.assert_array_equal(vector, [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


def test_multiple_classes_are_set_independently():
    vector = parse_label_string("7;6;9")

    np.testing.assert_array_equal(vector, [0, 0, 0, 0, 0, 1, 1, 0, 1, 0])
    # 标签串里类别的书写顺序不影响结果
    np.testing.assert_array_equal(parse_label_string("9;6;7"), vector)


def test_whitespace_around_tokens_is_tolerated():
    np.testing.assert_array_equal(
        parse_label_string(" 1 ; 10 "), [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    )


@pytest.mark.parametrize("value", [None, "", "   ", float("nan")])
def test_empty_value_means_no_class(value):
    """空值按「没有命中任何类别」处理，返回全 0 而不是报错。"""
    assert parse_label_string(value).tolist() == [0.0] * 10


def test_duplicate_class_raises():
    with pytest.raises(ValueError, match="重复"):
        parse_label_string("7;7")


@pytest.mark.parametrize("value", ["0", "11", "1;0"])
def test_class_id_out_of_range_raises(value):
    with pytest.raises(ValueError, match="超出"):
        parse_label_string(value)


@pytest.mark.parametrize("value", ["abc", "7;a", "7.0"])
def test_non_integer_token_raises(value):
    with pytest.raises(ValueError, match="不是整数"):
        parse_label_string(value)


def test_empty_fragment_raises():
    """「7;;9」这种多写了一个分号的情况要当场报错，不能悄悄跳过。"""
    with pytest.raises(ValueError, match="空片段"):
        parse_label_string("7;;9")
