"""Reading and splitting the dataset: the only module importing pandas.

The CSV has no header. Each row is an id, a diagnosis (``"B"`` or
``"M"``) and :data:`N_FEATURES` numeric features. :func:`read_dataset`
checks all of it and points at the faulty line; :func:`to_arrays` is the
frontier past which nobody sees a ``DataFrame`` any more.
"""

from typing import Final

import numpy as np
import pandas as pd

from mlp.errors import ConfigurationError, DatasetError
from mlp.types import FloatArray, IntArray, StrPath

LABEL_COLUMN: Final[int] = 1
FIRST_FEATURE: Final[int] = 2
N_FEATURES: Final[int] = 30
N_COLUMNS: Final[int] = FIRST_FEATURE + N_FEATURES

LABELS: Final[tuple[str, ...]] = ("B", "M")
"""Class names, in the order of their integer label."""


def _first_line(mask: "pd.Series[bool]") -> int:
    """Return the 1-based file line of the first True row of `mask`."""
    return int(np.flatnonzero(mask.to_numpy())[0]) + 1


def read_dataset(path: StrPath) -> pd.DataFrame:
    """Read and check the CSV file at `path`.

    The returned frame has :data:`N_COLUMNS` columns, labels among
    :data:`LABELS` and float64 features.

    Raise DatasetError when the file cannot be read or parsed, has the
    wrong number of columns, or holds a missing value, an unknown label
    or a non numeric feature; the message gives the faulty line.
    """
    try:
        df = pd.read_csv(path, header=None)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError,
            pd.errors.EmptyDataError) as e:
        raise DatasetError(f"cannot read {path!r}: {e}") from e

    if df.shape[1] != N_COLUMNS:
        raise DatasetError(
            f"{path}: expected {N_COLUMNS} columns (id, label and "
            f"{N_FEATURES} features), received {df.shape[1]}"
        )
    missing = df.isna().any(axis=1)
    if missing.any():
        raise DatasetError(
            f"{path}: line {_first_line(missing)}: missing value")
    unknown = ~df[LABEL_COLUMN].isin(LABELS)
    if unknown.any():
        line = _first_line(unknown)
        raise DatasetError(
            f"{path}: line {line}: expected a label among {LABELS}, "
            f"received {df[LABEL_COLUMN].iloc[line - 1]!r}"
        )

    features: pd.DataFrame = df.iloc[:, FIRST_FEATURE:].apply(
        pd.to_numeric, errors="coerce")
    not_numeric = features.isna().any(axis=1)
    if not_numeric.any():
        raise DatasetError(
            f"{path}: line {_first_line(not_numeric)}: expected numeric "
            "features"
        )
    checked: pd.DataFrame = pd.concat(
        [df.iloc[:, :FIRST_FEATURE], features.astype(np.float64)], axis=1)
    return checked


def to_arrays(df: pd.DataFrame) -> tuple[FloatArray, IntArray]:
    """Return the features ``(m, N_FEATURES)`` and the labels ``(m,)``.

    A label is the index of its class name in :data:`LABELS`. `df` is
    expected to come from :func:`read_dataset`.
    """
    x = df.iloc[:, FIRST_FEATURE:].to_numpy(dtype=np.float64)
    codes = pd.Categorical(df[LABEL_COLUMN], categories=LABELS).codes
    y = np.asarray(codes, dtype=np.int64)
    return x, y


def split_dataset(
    n_rows: int,
    ratio: float,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray]:
    """Return shuffled row indices ``(train, valid)``.

    ``round(ratio * n_rows)`` rows go to the training set, the others to
    the validation set. Only indices are returned, so the function does
    not care what is split.

    Raise ConfigurationError when `ratio` is not in ]0, 1[ or when one
    of the two sets would be empty.
    """
    if not 0.0 < ratio < 1.0:
        raise ConfigurationError(
            f"expected a ratio in ]0, 1[, received {ratio!r}")
    cut = round(ratio * n_rows)
    if cut == 0 or cut == n_rows:
        raise ConfigurationError(
            f"a ratio of {ratio} over {n_rows} rows leaves an empty set")
    indices = rng.permutation(n_rows).astype(np.int64)
    return indices[:cut], indices[cut:]


__all__ = [
    "LABEL_COLUMN",
    "FIRST_FEATURE",
    "N_FEATURES",
    "N_COLUMNS",
    "LABELS",
    "read_dataset",
    "to_arrays",
    "split_dataset",
]
