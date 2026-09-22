"""Known cases, edge cases and shape validation of the metrics."""

import numpy as np
import pytest

from mlp.errors import ShapeError
from mlp.metrics import (
    METRICS,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from mlp.types import IntArray

# 2 true positives, 1 false positive, 1 false negative, 2 true negatives.
Y_TRUE: IntArray = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
Y_PRED: IntArray = np.array([1, 1, 0, 1, 0, 0], dtype=np.int64)

EMPTY: IntArray = np.array([], dtype=np.int64)


def test_accuracy_on_a_known_case() -> None:
    """Four of the six samples are classified correctly."""
    assert accuracy_score(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)


def test_precision_recall_f1_on_a_known_case() -> None:
    """Precision 2/3, recall 2/3, so F1 is 2/3 as well."""
    assert precision_score(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    assert recall_score(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)
    assert f1_score(Y_TRUE, Y_PRED) == pytest.approx(2 / 3)


def test_pos_label_reports_the_other_class() -> None:
    """With pos_label=0 the roles of the two classes are swapped."""
    assert precision_score(Y_TRUE, Y_PRED, pos_label=0) == pytest.approx(2 / 3)
    assert recall_score(Y_TRUE, Y_PRED, pos_label=0) == pytest.approx(2 / 3)


def test_f1_forwards_pos_label() -> None:
    """F1 must be computed on the requested class, not always on 1."""
    y_true: IntArray = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred: IntArray = np.array([0, 1, 1, 1], dtype=np.int64)

    precision = precision_score(y_true, y_pred, pos_label=0)
    recall = recall_score(y_true, y_pred, pos_label=0)
    expected = 2 * precision * recall / (precision + recall)

    assert f1_score(y_true, y_pred, pos_label=0) == pytest.approx(expected)
    assert f1_score(y_true, y_pred, pos_label=0) != pytest.approx(
        f1_score(y_true, y_pred, pos_label=1))


def test_perfect_and_worst_predictions() -> None:
    """All correct gives 1.0, all wrong gives 0.0."""
    assert accuracy_score(Y_TRUE, Y_TRUE) == pytest.approx(1.0)
    assert f1_score(Y_TRUE, Y_TRUE) == pytest.approx(1.0)
    assert accuracy_score(Y_TRUE, 1 - Y_TRUE) == pytest.approx(0.0)
    assert f1_score(Y_TRUE, 1 - Y_TRUE) == pytest.approx(0.0)


def test_empty_arrays_return_zero() -> None:
    """No sample means no score, and certainly no NaN."""
    for metric in METRICS.values():
        assert metric(EMPTY, EMPTY) == 0.0


def test_no_predicted_positive_returns_zero() -> None:
    """A zero denominator is a legitimate state, not an error."""
    never: IntArray = np.zeros(4, dtype=np.int64)
    always: IntArray = np.ones(4, dtype=np.int64)

    assert precision_score(always, never) == 0.0
    assert recall_score(never, always) == 0.0
    assert f1_score(always, never) == 0.0


@pytest.mark.parametrize("name", sorted(METRICS))
def test_mismatched_shapes_raise_shape_error(name: str) -> None:
    """Comparing arrays of different lengths is a ShapeError."""
    metric = METRICS[name]

    with pytest.raises(ShapeError, match="same shape"):
        metric(Y_TRUE, Y_PRED[:3])


@pytest.mark.parametrize("name", sorted(METRICS))
def test_two_dimensional_input_raises_shape_error(name: str) -> None:
    """Metrics take class labels, never the raw output matrix."""
    metric = METRICS[name]
    column: IntArray = Y_TRUE.reshape(-1, 1)

    with pytest.raises(ShapeError, match="one-dimensional"):
        metric(column, column)
