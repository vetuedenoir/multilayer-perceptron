"""Feature scaling and label encoding, as pure numpy functions.

A :class:`Scaler` is a plain value: its kind and two vectors, applied as
``(x - a) / b``. That single form covers both scalers of the registry,
``(mean, std)`` for ``"standard"`` and ``(min, max - min)`` for
``"minmax"``, and fits in ``model.json`` as it is. The prediction
program therefore replays exactly the scaling fitted on the training
set, without refitting anything.
"""

from dataclasses import dataclass
from typing import Any, Callable, Final, Mapping, TypeAlias

import numpy as np

from mlp.errors import ConfigurationError, ModelFileError, ShapeError
from mlp.registry import get_from_registry
from mlp.types import FloatArray, IntArray


@dataclass(frozen=True, eq=False)
class Scaler:
    """A fitted affine scaling ``x -> (x - a) / b``, column by column.

    `a` and `b` are 1-D, with one value per feature, and `b` never holds
    a zero. Compared with `==` fields holding arrays would give an
    array, so equality is left to identity (``eq=False``).
    """

    kind: str
    a: FloatArray
    b: FloatArray

    def __post_init__(self) -> None:
        """Check the kind and the shapes of `a` and `b`.

        Raise ConfigurationError on an unknown kind, ShapeError when `a`
        and `b` are not 1-D vectors of the same length, and
        ConfigurationError on a non finite value or a zero in `b`.
        """
        get_from_registry(SCALERS, self.kind, "scaler")
        if self.a.ndim != 1 or self.a.shape != self.b.shape:
            raise ShapeError(
                "expected two 1-D vectors of the same length, received "
                f"shapes {self.a.shape} and {self.b.shape}"
            )
        if not (np.all(np.isfinite(self.a)) and np.all(np.isfinite(self.b))):
            raise ConfigurationError(
                "expected finite scaler parameters, received NaN or inf")
        if np.any(self.b == 0.0):
            raise ConfigurationError(
                "expected a scaler denominator without zeros")

    @property
    def n_features(self) -> int:
        """Return the number of columns the scaler was fitted on."""
        return int(self.a.shape[0])

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serializable copy of the scaler."""
        return {"kind": self.kind, "a": self.a.tolist(),
                "b": self.b.tolist()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scaler":
        """Rebuild a scaler from the output of to_dict().

        Raise ModelFileError when `data` does not have that layout.
        """
        try:
            return cls(
                kind=data["kind"],
                a=np.asarray(data["a"], dtype=np.float64),
                b=np.asarray(data["b"], dtype=np.float64),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ModelFileError(
                "expected a scaler of the form "
                "{'kind': str, 'a': [float], 'b': [float]}, "
                f"received: {e!r}"
            ) from e


def _check_features(x: FloatArray) -> None:
    """Raise ShapeError unless `x` is a non-empty ``(m, n)`` array."""
    if x.ndim != 2 or x.shape[0] == 0:
        raise ShapeError(
            f"expected a non-empty array of shape (m, n), received {x.shape}")


def _nonzero(denominator: FloatArray) -> FloatArray:
    """Replace the zeros of `denominator` by 1.

    A constant column has a zero spread; dividing it by 1 maps it to 0
    instead of NaN.
    """
    return np.where(denominator == 0.0, 1.0, denominator)


def fit_standard(x: FloatArray) -> Scaler:
    """Return the scaler centring and reducing every column of `x`."""
    _check_features(x)
    return Scaler("standard", x.mean(axis=0), _nonzero(x.std(axis=0)))


def fit_minmax(x: FloatArray) -> Scaler:
    """Return the scaler mapping every column of `x` onto [0, 1]."""
    _check_features(x)
    low = x.min(axis=0)
    return Scaler("minmax", low, _nonzero(x.max(axis=0) - low))


def transform(scaler: Scaler, x: FloatArray) -> FloatArray:
    """Return `x` scaled by `scaler`.

    Raise ShapeError when the number of columns of `x` differs from the
    one the scaler was fitted on.
    """
    if x.ndim != 2 or x.shape[1] != scaler.n_features:
        raise ShapeError(
            f"expected an input of shape (m, {scaler.n_features}), "
            f"received {x.shape}"
        )
    scaled: FloatArray = (x - scaler.a) / scaler.b
    return scaled


ScalerFit: TypeAlias = Callable[[FloatArray], Scaler]
"""Fit a scaler on a training set."""

SCALERS: Final[Mapping[str, ScalerFit]] = {
    "standard": fit_standard,
    "minmax": fit_minmax,
}


def fit_scaler(kind: str, x: FloatArray) -> Scaler:
    """Fit the scaler of the registry named `kind` on `x`."""
    return get_from_registry(SCALERS, kind, "scaler")(x)


def one_hot(labels: IntArray, n_classes: int) -> FloatArray:
    """Return the one-hot encoding ``(m, n_classes)`` of `labels`.

    Raise ConfigurationError when `n_classes` is lower than 2, and
    ShapeError when `labels` is not 1-D or holds a value outside
    ``[0, n_classes)``.
    """
    if isinstance(n_classes, bool) or not isinstance(n_classes, int) \
            or n_classes < 2:
        raise ConfigurationError(
            f"expected at least 2 classes, received {n_classes!r}")
    if labels.ndim != 1:
        raise ShapeError(
            f"expected 1-D labels, received shape {labels.shape}")
    if labels.size and (labels.min() < 0 or labels.max() >= n_classes):
        raise ShapeError(
            f"expected labels in [0, {n_classes}), received values in "
            f"[{labels.min()}, {labels.max()}]"
        )
    encoded: FloatArray = np.eye(n_classes)[labels]
    return encoded


__all__ = [
    "Scaler",
    "ScalerFit",
    "SCALERS",
    "fit_standard",
    "fit_minmax",
    "fit_scaler",
    "transform",
    "one_hot",
]
