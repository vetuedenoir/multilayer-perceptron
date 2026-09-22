"""Parameter update rules, behind a single ``Optimizer`` protocol.

A layer owns its parameters and their gradients; it does not own the
rule that turns one into the other. ``step(params, grads)`` returns the
new parameters, so the descent lives here and a layer only has to expose
``params()`` / ``grads()`` / ``set_params()``.

Adding Adam therefore means adding a class in this module and a key in
:data:`OPTIMIZERS`, without touching a single layer.
"""

from dataclasses import dataclass
from typing import Final, Mapping, Protocol, Sequence

from mlp.errors import ConfigurationError
from mlp.types import FloatArray


def sgd_step(
    params: Sequence[FloatArray],
    grads: Sequence[FloatArray],
    learning_rate: float,
) -> list[FloatArray]:
    """Return `params` moved one step down the gradient.

    Pure: the returned arrays are new ones, the inputs are untouched.
    """
    return [p - learning_rate * g for p, g in zip(params, grads)]


class Optimizer(Protocol):
    """Turn parameters and their gradients into updated parameters."""

    name: str

    def step(
        self,
        params: Sequence[FloatArray],
        grads: Sequence[FloatArray],
    ) -> list[FloatArray]:
        """Return the updated parameters."""
        ...


@dataclass
class SGD:
    """Plain stochastic gradient descent."""

    learning_rate: float = 0.001
    name: str = "sgd"

    def __post_init__(self) -> None:
        """Validate the learning rate."""
        if self.learning_rate <= 0.0:
            raise ConfigurationError(
                "expected a strictly positive learning rate, received "
                f"{self.learning_rate!r}"
            )

    def step(
        self,
        params: Sequence[FloatArray],
        grads: Sequence[FloatArray],
    ) -> list[FloatArray]:
        """Return the updated parameters."""
        return sgd_step(params, grads, self.learning_rate)


OPTIMIZERS: Final[Mapping[str, type[Optimizer]]] = {
    "sgd": SGD,
    "SGD": SGD,
}

__all__ = ["sgd_step", "Optimizer", "SGD", "OPTIMIZERS"]
