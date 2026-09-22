"""Tests of the dataset reader and of the split."""

from pathlib import Path

import numpy as np
import pytest

from mlp.data import (
    N_FEATURES,
    read_dataset,
    split_dataset,
    to_arrays,
)
from mlp.errors import ConfigurationError, DatasetError


def row(identifier: int, label: str, value: str = "1.5") -> str:
    """Return one CSV line with every feature set to `value`."""
    return ",".join([str(identifier), label] + [value] * N_FEATURES)


def write_csv(tmp_path: Path, lines: list[str]) -> Path:
    """Write `lines` to a CSV file and return its path."""
    path = tmp_path / "data.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_read_and_convert(tmp_path: Path) -> None:
    """Labels become 0 for B and 1 for M, features become floats."""
    path = write_csv(tmp_path, [row(1, "M"), row(2, "B"), row(3, "M")])
    x, y = to_arrays(read_dataset(path))
    assert x.shape == (3, N_FEATURES)
    assert x.dtype == np.float64
    np.testing.assert_array_equal(y, [1, 0, 1])
    assert y.dtype == np.int64


def test_missing_file(tmp_path: Path) -> None:
    """A missing file is a dataset error, with the cause kept."""
    with pytest.raises(DatasetError, match="cannot read") as info:
        read_dataset(tmp_path / "absent.csv")
    assert isinstance(info.value.__cause__, OSError)


def test_empty_file(tmp_path: Path) -> None:
    """An empty file cannot be parsed."""
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DatasetError):
        read_dataset(path)


def test_wrong_number_of_columns(tmp_path: Path) -> None:
    """Every row needs an id, a label and all the features."""
    path = write_csv(tmp_path, ["1,M,2.0", "2,B,3.0"])
    with pytest.raises(DatasetError, match="columns"):
        read_dataset(path)


@pytest.mark.parametrize(
    ("bad_row", "match"),
    [
        (row(2, "X"), "line 2: expected a label"),
        (row(2, "B", ""), "line 2: missing value"),
        (row(2, "B", "abc"), "line 2: expected numeric"),
    ],
)
def test_faulty_line_is_reported(
    tmp_path: Path,
    bad_row: str,
    match: str,
) -> None:
    """The message points at the faulty line of the file."""
    path = write_csv(tmp_path, [row(1, "M"), bad_row, row(3, "B")])
    with pytest.raises(DatasetError, match=match):
        read_dataset(path)


def test_split_is_a_partition() -> None:
    """Every row lands in exactly one of the two sets."""
    train, valid = split_dataset(10, 0.8, np.random.default_rng(0))
    assert len(train) == 8 and len(valid) == 2
    assert sorted(np.concatenate([train, valid])) == list(range(10))


def test_split_is_reproducible() -> None:
    """Same seed, same split."""
    first = split_dataset(50, 0.7, np.random.default_rng(3))
    second = split_dataset(50, 0.7, np.random.default_rng(3))
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("ratio", [0.0, 1.0, 1.5, -0.2])
def test_split_rejects_ratio_out_of_range(ratio: float) -> None:
    """The ratio must be in ]0, 1[."""
    with pytest.raises(ConfigurationError, match="ratio"):
        split_dataset(10, ratio, np.random.default_rng(0))


def test_split_rejects_empty_set() -> None:
    """A ratio too close to 0 or 1 for the size leaves a set empty."""
    with pytest.raises(ConfigurationError, match="empty"):
        split_dataset(3, 0.9, np.random.default_rng(0))
