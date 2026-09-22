"""Shared helpers for the test suite: finite-difference gradient checks."""

from typing import Callable

import numpy as np

from mlp.types import FloatArray

STEP = 1e-6


def numerical_gradient(
    f: Callable[[FloatArray], float],
    x: FloatArray,
    step: float = STEP,
) -> FloatArray:
    """Return the gradient of `f` at `x` by central differences.

    `x` is restored after every perturbation, so the caller keeps its
    array untouched.
    """
    grad = np.zeros_like(x)
    for index in np.ndindex(*x.shape):
        original = x[index]
        x[index] = original + step
        plus = f(x)
        x[index] = original - step
        minus = f(x)
        x[index] = original
        grad[index] = (plus - minus) / (2.0 * step)
    return grad


def relative_error(a: FloatArray, b: FloatArray) -> float:
    """Return the gap between `a` and `b`, relative to their scale.

    The largest absolute difference is divided by the largest magnitude
    of the two gradients, rather than element by element. A central
    difference loses precision by cancellation, and the loss is driven
    by the scale of the whole expression: judging a nearly-flat
    component against itself would measure that cancellation, not the
    correctness of the formula.
    """
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-12)
    return float(np.max(np.abs(a - b))) / scale


def away_from_zero(x: FloatArray, margin: float = 0.1) -> FloatArray:
    """Push every element of `x` at least `margin` away from zero.

    ReLU and leaky ReLU have no derivative at zero: a finite-difference
    check straddling it would compare the analytic gradient of one
    branch with the average slope of both.
    """
    return np.sign(x) * (np.abs(x) + margin)
