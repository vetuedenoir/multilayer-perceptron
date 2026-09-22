"""Layers of the network, behind a single ``Layer`` protocol.

The arithmetic of a dense layer lives in two pure functions,
:func:`dense_forward` and :func:`dense_backward`; :class:`DenseLayer`
only keeps the parameters, caches what the backward pass needs and
delegates the computation to them.

:class:`~mlp.network.Model` knows nothing but the :class:`Layer`
protocol: adding a new kind of layer (dropout, batch normalization...)
does not touch the model.
"""

from typing import (
    Callable,
    Final,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    TypedDict,
    runtime_checkable,
)

import numpy as np

from mlp.activations import ACTIVATIONS
from mlp.errors import ConfigurationError, NotBuiltError, ShapeError
from mlp.initializers import DEFAULT_INITIALIZER, INITIALIZERS
from mlp.registry import get_from_registry
from mlp.types import FloatArray


class LayerConfig(TypedDict):
    """Everything needed to rebuild a layer, weights aside."""

    type: str
    units: int
    activation: str
    initializer: str


def dense_forward(W: FloatArray, b: FloatArray, x: FloatArray) -> FloatArray:
    """Return the pre-activation ``z = x @ W.T + b``.

    `W` has shape ``(units, input_size)``, `b` ``(1, units)`` and `x`
    ``(m, input_size)``; `z` has shape ``(m, units)``.
    """
    z: FloatArray = x @ W.T + b
    return z


def dense_backward(
    W: FloatArray,
    x: FloatArray,
    dZ: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(dW, db, dA_prev)`` given the gradient `dZ` w.r.t. z.

    The losses average over the ``m`` samples of the batch but their
    gradients do not: the division by ``m`` happens here, once.
    """
    m = x.shape[0]
    dW: FloatArray = (dZ.T @ x) / m
    db: FloatArray = np.sum(dZ, axis=0, keepdims=True) / m
    dA_prev: FloatArray = dZ @ W
    return dW, db, dA_prev


@runtime_checkable
class Layer(Protocol):
    """What the model needs from a layer, and nothing more."""

    @property
    def built(self) -> bool:
        """Tell whether the parameters have been allocated."""
        ...

    def build(self, input_size: int, rng: np.random.Generator) -> None:
        """Allocate and initialize the parameters."""
        ...

    def forward(self, x: FloatArray) -> FloatArray:
        """Return the output of the layer, caching what backward needs."""
        ...

    def backward(self, dA: FloatArray, from_loss: bool = False) -> FloatArray:
        """Store the parameter gradients and return dA for the previous layer.

        `from_loss` tells that `dA` is already the gradient w.r.t. the
        pre-activation (fused loss and output activation).
        """
        ...

    def params(self) -> list[FloatArray]:
        """Return the trainable parameters."""
        ...

    def grads(self) -> list[FloatArray]:
        """Return the gradients of the last backward, in params() order."""
        ...

    def set_params(self, params: Sequence[FloatArray]) -> None:
        """Replace the trainable parameters, in params() order."""
        ...

    def get_config(self) -> LayerConfig:
        """Return the configuration needed to rebuild the layer."""
        ...


class DenseLayer:
    """A fully connected layer followed by an activation."""

    def __init__(
        self,
        units: int,
        activation: str,
        weights_initializer: str = DEFAULT_INITIALIZER,
    ) -> None:
        """Resolve the activation and the initializer from the registries.

        Raise ConfigurationError on a non positive `units` or on an
        unknown activation or initializer name.
        """
        if isinstance(units, bool) or not isinstance(units, int) \
                or units <= 0:
            raise ConfigurationError(
                "expected a positive integer for 'units', "
                f"received {units!r}"
            )
        self.units = units
        # Resolved once, here: the layer holds an immutable Activation,
        # so nothing downstream ever has to instantiate anything.
        self.activation = get_from_registry(
            ACTIVATIONS, activation, "activation")
        self.initializer_name = weights_initializer
        self.initializer = get_from_registry(
            INITIALIZERS, weights_initializer, "weights initializer")

        self.weights: FloatArray = np.empty((0, 0))
        self.bias: FloatArray = np.empty((0, 0))
        self._built = False
        self._input: FloatArray | None = None
        self._z: FloatArray | None = None
        self._dW: FloatArray | None = None
        self._db: FloatArray | None = None

    @property
    def built(self) -> bool:
        """Tell whether the weights have been allocated."""
        return self._built

    def build(self, input_size: int, rng: np.random.Generator) -> None:
        """Draw the weights with the initializer, set the bias to zero.

        Raise ConfigurationError on a non positive `input_size`.
        """
        if isinstance(input_size, bool) or not isinstance(input_size, int) \
                or input_size <= 0:
            raise ConfigurationError(
                "expected a positive integer for 'input_size', "
                f"received {input_size!r}"
            )
        self.weights = self.initializer(self.units, input_size, rng)
        self.bias = np.zeros((1, self.units))
        self._built = True

    def forward(self, x: FloatArray) -> FloatArray:
        """Return the activated output of the layer for the batch `x`.

        Raise NotBuiltError before build(), ShapeError when the number
        of columns of `x` differs from the input size of the layer.
        """
        if not self._built:
            raise NotBuiltError(
                "the layer must be built before calling forward()")
        if x.ndim != 2 or x.shape[1] != self.weights.shape[1]:
            raise ShapeError(
                f"expected an input of shape (m, {self.weights.shape[1]}), "
                f"received {x.shape}"
            )
        self._input = x
        self._z = dense_forward(self.weights, self.bias, x)
        return self.activation.forward(self._z)

    def backward(self, dA: FloatArray, from_loss: bool = False) -> FloatArray:
        """Store dW and db, and return dA for the previous layer.

        Raise NotBuiltError when called before forward().
        """
        if self._input is None or self._z is None:
            raise NotBuiltError(
                "forward() must be called before backward()")
        dZ = dA if from_loss else self.activation.backward(self._z, dA)
        self._dW, self._db, dA_prev = dense_backward(
            self.weights, self._input, dZ)
        return dA_prev

    def params(self) -> list[FloatArray]:
        """Return ``[W, b]``."""
        return [self.weights, self.bias]

    def grads(self) -> list[FloatArray]:
        """Return ``[dW, db]`` from the last backward pass.

        Raise NotBuiltError when no backward pass has run yet.
        """
        if self._dW is None or self._db is None:
            raise NotBuiltError(
                "backward() must be called before reading the gradients")
        return [self._dW, self._db]

    def set_params(self, params: Sequence[FloatArray]) -> None:
        """Replace W and b, which builds the layer if it was not.

        Raise ShapeError when `params` is not ``[W, b]`` with
        ``W.shape == (units, input_size)`` and ``b.shape == (1, units)``,
        ``input_size`` being the current one once the layer is built.
        """
        if len(params) != 2:
            raise ShapeError(
                f"expected 2 parameters [W, b], received {len(params)}")
        weights, bias = (np.asarray(p, dtype=np.float64) for p in params)
        input_size = self.weights.shape[1] if self._built else None
        if weights.ndim != 2 or weights.shape[0] != self.units \
                or weights.shape[1] == 0 \
                or input_size not in (None, weights.shape[1]):
            expected = f"({self.units}, {input_size or 'input_size'})"
            raise ShapeError(
                f"expected weights of shape {expected}, "
                f"received {weights.shape}"
            )
        if bias.shape != (1, self.units):
            raise ShapeError(
                f"expected a bias of shape (1, {self.units}), "
                f"received {bias.shape}"
            )
        self.weights = weights
        self.bias = bias
        self._built = True

    def get_config(self) -> LayerConfig:
        """Return the type, units, activation and initializer names."""
        return {
            "type": "dense",
            "units": self.units,
            "activation": self.activation.name,
            "initializer": self.initializer_name,
        }

    @classmethod
    def from_config(cls, config: LayerConfig) -> "DenseLayer":
        """Build an unbuilt layer from the output of get_config()."""
        return cls(config["units"], config["activation"],
                   config["initializer"])

    def __repr__(self) -> str:
        """Return a constructor-like representation."""
        return (
            f"DenseLayer(units={self.units}, "
            f"activation={self.activation.name!r}, "
            f"weights_initializer={self.initializer_name!r})"
        )


LayerFactory: TypeAlias = Callable[[LayerConfig], Layer]
"""Build an unbuilt layer from its configuration."""

LAYERS: Final[Mapping[str, LayerFactory]] = {
    "dense": DenseLayer.from_config,
}


__all__ = [
    "LayerConfig",
    "Layer",
    "LayerFactory",
    "LAYERS",
    "DenseLayer",
    "dense_forward",
    "dense_backward",
]
