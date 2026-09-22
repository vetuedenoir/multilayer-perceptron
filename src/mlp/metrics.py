"""Classification metrics, as pure functions gathered in a registry.

Every metric works on one-dimensional arrays of integer class labels,
never on the raw output matrix of the network: turning probabilities
into labels is the job of the caller. Both arrays must have the same
shape, otherwise a :class:`~mlp.errors.ShapeError` is raised.

A metric whose denominator is zero (no sample, no positive prediction,
no positive label) returns ``0.0`` rather than raising: an epoch with no
predicted positive is a legitimate state of the training, not an error.
"""

from typing import Callable, Final, Mapping, TypeAlias

import numpy as np

from mlp.errors import ShapeError
from mlp.types import IntArray

MetricFn: TypeAlias = Callable[[IntArray, IntArray], float]


def _check_labels(y: IntArray, y_pred: IntArray) -> None:
    """Raise ShapeError unless `y` and `y_pred` are 1-D and aligned."""
    if y.ndim != 1 or y_pred.ndim != 1:
        raise ShapeError(
            "expected one-dimensional arrays of class labels, received "
            f"shapes {y.shape} and {y_pred.shape}"
        )
    if y.shape != y_pred.shape:
        raise ShapeError(
            "expected y and y_pred to have the same shape, received "
            f"{y.shape} and {y_pred.shape}"
        )


def accuracy_score(y: IntArray, y_pred: IntArray) -> float:
    """Return the fraction of samples predicted correctly."""
    _check_labels(y, y_pred)
    if y.size == 0:
        return 0.0
    return float(np.mean(y == y_pred))


def precision_score(
    y: IntArray,
    y_pred: IntArray,
    pos_label: int = 1,
) -> float:
    """Return the precision of class `pos_label`.

    That is, the fraction of samples predicted as `pos_label` that
    really belong to it.
    """
    _check_labels(y, y_pred)
    predicted_positive = y_pred == pos_label
    n_predicted = int(np.count_nonzero(predicted_positive))
    if n_predicted == 0:
        return 0.0
    true_positive = int(np.count_nonzero(predicted_positive & (y == y_pred)))
    return true_positive / n_predicted


def recall_score(
    y: IntArray,
    y_pred: IntArray,
    pos_label: int = 1,
) -> float:
    """Return the recall of class `pos_label`.

    That is, the fraction of the samples truly labelled `pos_label` that
    were predicted as such.
    """
    _check_labels(y, y_pred)
    actual_positive = y == pos_label
    n_actual = int(np.count_nonzero(actual_positive))
    if n_actual == 0:
        return 0.0
    true_positive = int(np.count_nonzero(actual_positive & (y == y_pred)))
    return true_positive / n_actual


def f1_score(y: IntArray, y_pred: IntArray, pos_label: int = 1) -> float:
    """Return the harmonic mean of precision and recall for `pos_label`."""
    precision = precision_score(y, y_pred, pos_label)
    recall = recall_score(y, y_pred, pos_label)
    if precision + recall == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


METRICS: Final[Mapping[str, MetricFn]] = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
}

__all__ = [
    "MetricFn",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "METRICS",
]
