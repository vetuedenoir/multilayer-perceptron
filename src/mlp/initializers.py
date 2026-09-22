"""Weight initializers, as pure functions gathered in a registry.

Every initializer takes the shape of the weight matrix and the random
generator to draw from, and returns a ``(units, input_size)`` array. The
generator is a parameter and never a module-level global: two runs given
the same seed must build the same network, and a function drawing from
``np.random`` directly could not promise that.
"""

from typing import Callable, Final, Mapping, TypeAlias

import numpy as np

from mlp.types import FloatArray

Initializer: TypeAlias = Callable[[int, int, np.random.Generator], FloatArray]

DEFAULT_INITIALIZER: Final[str] = "random_normal"


def zeros(
    units: int,
    input_size: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Return a zero-filled weight matrix."""
    return np.zeros((units, input_size))


def random_normal(
    units: int,
    input_size: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw the weights from the standard normal distribution."""
    return rng.normal(loc=0.0, scale=1.0, size=(units, input_size))


def random_uniform(
    units: int,
    input_size: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw the weights uniformly in [-1, 1]."""
    return rng.uniform(-1.0, 1.0, size=(units, input_size))


def he_uniform(
    units: int,
    input_size: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw the weights uniformly in [-limit, limit], He scaling."""
    limit = np.sqrt(6.0 / input_size)
    return rng.uniform(-limit, limit, size=(units, input_size))


def he_normal(
    units: int,
    input_size: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw the weights from a centred normal, He scaling."""
    sigma = np.sqrt(2.0 / input_size)
    return rng.normal(loc=0.0, scale=sigma, size=(units, input_size))


def glorot_uniform(
    units: int,
    input_size: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw the weights uniformly in [-limit, limit], Glorot scaling."""
    limit = np.sqrt(6.0 / (input_size + units))
    return rng.uniform(-limit, limit, size=(units, input_size))


def glorot_normal(
    units: int,
    input_size: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Draw the weights from a centred normal, Glorot scaling."""
    sigma = np.sqrt(2.0 / (input_size + units))
    return rng.normal(loc=0.0, scale=sigma, size=(units, input_size))


INITIALIZERS: Final[Mapping[str, Initializer]] = {
    "zeros": zeros,
    "random_normal": random_normal,
    "random_uniform": random_uniform,
    "he_uniform": he_uniform,
    "he_normal": he_normal,
    "glorot_uniform": glorot_uniform,
    "glorot_normal": glorot_normal,
    # Aliases: the camelCase names used by the subject and so far here.
    "zero": zeros,
    "randomNormal": random_normal,
    "randomUniform": random_uniform,
    "heUniform": he_uniform,
    "heNormal": he_normal,
    "glorotUniform": glorot_uniform,
    "glorotNormal": glorot_normal,
}

__all__ = [
    "Initializer",
    "DEFAULT_INITIALIZER",
    "zeros",
    "random_normal",
    "random_uniform",
    "he_uniform",
    "he_normal",
    "glorot_uniform",
    "glorot_normal",
    "INITIALIZERS",
]
