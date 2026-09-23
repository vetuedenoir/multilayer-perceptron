"""Parameter update rules, behind a single ``Optimizer`` protocol.

A layer owns its parameters and their gradients; it does not own the
rule that turns one into the other. ``step(params, grads)`` returns the
new parameters, so the descent lives here and a layer only has to expose
``params()`` / ``grads()`` / ``set_params()``.

The model calls ``step`` once per batch with the flat list of all its
parameters, always in the same order. An optimizer with a state (a
velocity, running averages) therefore indexes it by position in that
list; the state is allocated at the first step, and ``reset()`` empties
it.

Every update rule is a pure function returning the new parameters and
the new state; the classes only keep the state between two calls.
"""

import inspect
from dataclasses import dataclass, field
from functools import partial
from typing import (
    Callable,
    Final,
    Mapping,
    Protocol,
    Sequence,
    TypeAlias,
    TypedDict,
)

import numpy as np

from mlp.errors import ConfigurationError, ShapeError
from mlp.registry import get_from_registry
from mlp.types import FloatArray

Hyperparameters: TypeAlias = dict[str, float | bool]
"""Hyperparameters of an optimizer, besides its learning rate."""


def _check_lengths(
    params: Sequence[FloatArray],
    grads: Sequence[FloatArray],
) -> None:
    """Raise ShapeError unless there is one gradient per parameter."""
    if len(params) != len(grads):
        raise ShapeError(
            f"expected one gradient per parameter ({len(params)}), "
            f"received {len(grads)}"
        )


def sgd_step(
    params: Sequence[FloatArray],
    grads: Sequence[FloatArray],
    learning_rate: float,
) -> list[FloatArray]:
    """Return `params` moved one step down the gradient.

    Pure: the returned arrays are new ones, the inputs are untouched.
    """
    _check_lengths(params, grads)
    return [p - learning_rate * g for p, g in zip(params, grads)]


def momentum_step(
    params: Sequence[FloatArray],
    grads: Sequence[FloatArray],
    velocity: Sequence[FloatArray],
    learning_rate: float,
    momentum: float,
    nesterov: bool,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Return the parameters and the velocity after one momentum step.

    Classical momentum accumulates a velocity and moves along it::

        v = momentum * v - learning_rate * g
        p = p + v

    Nesterov momentum (Sutskever et al., 2013) takes the gradient at the
    look-ahead point ``theta + momentum * v`` instead of at ``theta``::

        v' = momentum * v - learning_rate * grad(theta + momentum * v)
        theta' = theta + v'

    Evaluating a gradient elsewhere than at the current parameters would
    need a second forward pass. The trick is to store the look-ahead
    point ``p = theta + momentum * v`` as the parameters: the gradient
    ``g`` computed at ``p`` is then exactly the one needed, and
    substituting ``theta = p - momentum * v`` gives::

        v' = momentum * v - learning_rate * g
        p' = theta' + momentum * v'
           = p - momentum * v + v' + momentum * v'
           = p + momentum * v' - learning_rate * g

    since ``v' - momentum * v = -learning_rate * g``. This is the form
    used by Keras, and the one computed here.

    Pure: the returned arrays are new ones, the inputs are untouched.
    """
    _check_lengths(params, grads)
    new_params: list[FloatArray] = []
    new_velocity: list[FloatArray] = []
    for p, g, v in zip(params, grads, velocity):
        v = momentum * v - learning_rate * g
        if nesterov:
            new_params.append(p + momentum * v - learning_rate * g)
        else:
            new_params.append(p + v)
        new_velocity.append(v)
    return new_params, new_velocity


def rmsprop_step(
    params: Sequence[FloatArray],
    grads: Sequence[FloatArray],
    sq_avg: Sequence[FloatArray],
    learning_rate: float,
    rho: float,
    epsilon: float,
) -> tuple[list[FloatArray], list[FloatArray]]:
    """Return the parameters and the squared average after one step.

    Each component is divided by a running root mean square of its own
    gradients, so that steep and flat directions move at similar
    speeds::

        s = rho * s + (1 - rho) * g**2
        p = p - learning_rate * g / (sqrt(s) + epsilon)

    Pure: the returned arrays are new ones, the inputs are untouched.
    """
    _check_lengths(params, grads)
    new_params: list[FloatArray] = []
    new_sq_avg: list[FloatArray] = []
    for p, g, s in zip(params, grads, sq_avg):
        s = rho * s + (1.0 - rho) * g * g
        new_params.append(p - learning_rate * g / (np.sqrt(s) + epsilon))
        new_sq_avg.append(s)
    return new_params, new_sq_avg


def adam_step(
    params: Sequence[FloatArray],
    grads: Sequence[FloatArray],
    m: Sequence[FloatArray],
    v: Sequence[FloatArray],
    t: int,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
) -> tuple[list[FloatArray], list[FloatArray], list[FloatArray]]:
    """Return the parameters and both moments after Adam step `t`.

    Momentum on the gradient, RMSprop on its square, both corrected for
    their bias towards zero (they start at zero)::

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        p = p - learning_rate * m_hat / (sqrt(v_hat) + epsilon)

    `t` counts the steps from 1. Pure: the returned arrays are new ones,
    the inputs are untouched. Raise ConfigurationError when `t` < 1.
    """
    if isinstance(t, bool) or not isinstance(t, int) or t < 1:
        raise ConfigurationError(
            f"expected a step number t >= 1, received {t!r}")
    _check_lengths(params, grads)
    correction1 = 1.0 - beta1 ** t
    correction2 = 1.0 - beta2 ** t
    new_params: list[FloatArray] = []
    new_m: list[FloatArray] = []
    new_v: list[FloatArray] = []
    for p, g, m_i, v_i in zip(params, grads, m, v):
        m_i = beta1 * m_i + (1.0 - beta1) * g
        v_i = beta2 * v_i + (1.0 - beta2) * g * g
        m_hat = m_i / correction1
        v_hat = v_i / correction2
        new_params.append(
            p - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon))
        new_m.append(m_i)
        new_v.append(v_i)
    return new_params, new_m, new_v


class OptimizerConfig(TypedDict):
    """Everything needed to rebuild an optimizer from its registry."""

    name: str
    learning_rate: float
    hyperparameters: Hyperparameters


class Optimizer(Protocol):
    """Turn parameters and their gradients into updated parameters."""

    @property
    def name(self) -> str:
        """Return the registry name of the optimizer."""
        ...

    def step(
        self,
        params: Sequence[FloatArray],
        grads: Sequence[FloatArray],
    ) -> list[FloatArray]:
        """Return the updated parameters."""
        ...

    def reset(self) -> None:
        """Forget the state accumulated by the previous steps."""
        ...

    def get_config(self) -> OptimizerConfig:
        """Return the configuration needed to rebuild the optimizer."""
        ...


def _check_real(
    value: object,
    name: str,
    low: float,
    high: float | None = None,
    low_inclusive: bool = True,
) -> float:
    """Return `value` as a float, checked against its bounds.

    The accepted interval is ``[low, high[`` (``]low, high[`` when
    `low_inclusive` is False), unbounded above when `high` is None.
    Raise ConfigurationError otherwise.
    """
    left = "[" if low_inclusive else "]"
    interval = f"{left}{low}, {high if high is not None else 'inf'}["
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"expected a number in {interval} for {name!r}, "
            f"received {value!r}"
        )
    number = float(value)
    too_low = number < low if low_inclusive else number <= low
    too_high = high is not None and number >= high
    if too_low or too_high or not np.isfinite(number):
        raise ConfigurationError(
            f"expected {name!r} in {interval}, received {value!r}")
    return number


def _check_learning_rate(value: object) -> float:
    """Return `value` if it is a strictly positive learning rate."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not value > 0.0 or not np.isfinite(value):
        raise ConfigurationError(
            "expected a strictly positive learning rate, received "
            f"{value!r}"
        )
    return float(value)


def _check_state(
    state: list[FloatArray],
    params: Sequence[FloatArray],
) -> list[FloatArray]:
    """Return `state`, allocated with zeros if empty.

    Raise ShapeError when a state already built does not match the
    number or the shapes of `params`.
    """
    if not state:
        return [np.zeros_like(p) for p in params]
    if len(state) != len(params):
        raise ShapeError(
            f"optimizer state built for {len(state)} parameters, "
            f"received {len(params)}; call reset()"
        )
    for i, (s, p) in enumerate(zip(state, params)):
        if s.shape != p.shape:
            raise ShapeError(
                f"optimizer state built for a parameter {i} of shape "
                f"{s.shape}, received {p.shape}; call reset()"
            )
    return state


@dataclass
class SGD:
    """Plain stochastic gradient descent, without any state."""

    learning_rate: float = 0.001

    def __post_init__(self) -> None:
        """Validate the learning rate."""
        self.learning_rate = _check_learning_rate(self.learning_rate)

    @property
    def name(self) -> str:
        """Return the registry name of the optimizer."""
        return "sgd"

    def step(
        self,
        params: Sequence[FloatArray],
        grads: Sequence[FloatArray],
    ) -> list[FloatArray]:
        """Return the updated parameters."""
        return sgd_step(params, grads, self.learning_rate)

    def reset(self) -> None:
        """Do nothing: plain SGD has no state."""

    def get_config(self) -> OptimizerConfig:
        """Return the name and the learning rate."""
        return {"name": self.name, "learning_rate": self.learning_rate,
                "hyperparameters": {}}


@dataclass
class Momentum:
    """SGD with momentum, classical or Nesterov (see momentum_step)."""

    learning_rate: float = 0.01
    momentum: float = 0.9
    nesterov: bool = False
    _velocity: list[FloatArray] = field(
        default_factory=list, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the hyperparameters: 0 <= momentum < 1."""
        self.learning_rate = _check_learning_rate(self.learning_rate)
        self.momentum = _check_real(self.momentum, "momentum", 0.0, 1.0)
        if not isinstance(self.nesterov, bool):
            raise ConfigurationError(
                "expected a boolean for 'nesterov', received "
                f"{self.nesterov!r}"
            )

    @property
    def name(self) -> str:
        """Return ``"nesterov"`` or ``"momentum"``."""
        return "nesterov" if self.nesterov else "momentum"

    def step(
        self,
        params: Sequence[FloatArray],
        grads: Sequence[FloatArray],
    ) -> list[FloatArray]:
        """Return the updated parameters, and keep the new velocity."""
        velocity = _check_state(self._velocity, params)
        new_params, self._velocity = momentum_step(
            params, grads, velocity, self.learning_rate, self.momentum,
            self.nesterov)
        return new_params

    def reset(self) -> None:
        """Forget the velocity."""
        self._velocity = []

    def get_config(self) -> OptimizerConfig:
        """Return the name, the learning rate and the momentum."""
        return {"name": self.name, "learning_rate": self.learning_rate,
                "hyperparameters": {"momentum": self.momentum,
                                    "nesterov": self.nesterov}}


@dataclass
class RMSprop:
    """Gradient scaled by a running RMS of itself (see rmsprop_step)."""

    learning_rate: float = 0.001
    rho: float = 0.9
    epsilon: float = 1e-8
    _sq_avg: list[FloatArray] = field(
        default_factory=list, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the hyperparameters: 0 < rho < 1, epsilon > 0."""
        self.learning_rate = _check_learning_rate(self.learning_rate)
        self.rho = _check_real(self.rho, "rho", 0.0, 1.0,
                               low_inclusive=False)
        self.epsilon = _check_real(self.epsilon, "epsilon", 0.0,
                                   low_inclusive=False)

    @property
    def name(self) -> str:
        """Return the registry name of the optimizer."""
        return "rmsprop"

    def step(
        self,
        params: Sequence[FloatArray],
        grads: Sequence[FloatArray],
    ) -> list[FloatArray]:
        """Return the updated parameters, and keep the new average."""
        sq_avg = _check_state(self._sq_avg, params)
        new_params, self._sq_avg = rmsprop_step(
            params, grads, sq_avg, self.learning_rate, self.rho,
            self.epsilon)
        return new_params

    def reset(self) -> None:
        """Forget the running average."""
        self._sq_avg = []

    def get_config(self) -> OptimizerConfig:
        """Return the name, the learning rate, rho and epsilon."""
        return {"name": self.name, "learning_rate": self.learning_rate,
                "hyperparameters": {"rho": self.rho,
                                    "epsilon": self.epsilon}}


@dataclass
class Adam:
    """Adaptive moment estimation (see adam_step)."""

    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    _m: list[FloatArray] = field(
        default_factory=list, init=False, repr=False, compare=False)
    _v: list[FloatArray] = field(
        default_factory=list, init=False, repr=False, compare=False)
    _t: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate: 0 <= beta1, beta2 < 1 and epsilon > 0."""
        self.learning_rate = _check_learning_rate(self.learning_rate)
        self.beta1 = _check_real(self.beta1, "beta1", 0.0, 1.0)
        self.beta2 = _check_real(self.beta2, "beta2", 0.0, 1.0)
        self.epsilon = _check_real(self.epsilon, "epsilon", 0.0,
                                   low_inclusive=False)

    @property
    def name(self) -> str:
        """Return the registry name of the optimizer."""
        return "adam"

    def step(
        self,
        params: Sequence[FloatArray],
        grads: Sequence[FloatArray],
    ) -> list[FloatArray]:
        """Return the updated parameters, and keep the new moments."""
        m = _check_state(self._m, params)
        v = _check_state(self._v, params)
        new_params, self._m, self._v = adam_step(
            params, grads, m, v, self._t + 1, self.learning_rate,
            self.beta1, self.beta2, self.epsilon)
        self._t += 1
        return new_params

    def reset(self) -> None:
        """Forget both moments and the step count."""
        self._m = []
        self._v = []
        self._t = 0

    def get_config(self) -> OptimizerConfig:
        """Return the name, the learning rate, the betas and epsilon."""
        return {"name": self.name, "learning_rate": self.learning_rate,
                "hyperparameters": {"beta1": self.beta1,
                                    "beta2": self.beta2,
                                    "epsilon": self.epsilon}}


OptimizerFactory: TypeAlias = Callable[..., Optimizer]
"""Build an optimizer from its learning rate and hyperparameters."""

OPTIMIZERS: Final[Mapping[str, OptimizerFactory]] = {
    "sgd": SGD,
    "momentum": Momentum,
    "nesterov": partial(Momentum, nesterov=True),
    "rmsprop": RMSprop,
    "adam": Adam,
    "SGD": SGD,
    "RMSProp": RMSprop,
    "Adam": Adam,
}


def accepted_hyperparameters(factory: OptimizerFactory) -> list[str]:
    """Return the hyperparameters `factory` accepts, learning rate aside.

    Read from its signature, so that a new optimizer needs no list of
    its own.
    """
    return [name for name in inspect.signature(factory).parameters
            if name != "learning_rate"]


def make_optimizer(
    name: str,
    learning_rate: float,
    **hyperparameters: float | bool,
) -> Optimizer:
    """Return the optimizer registered as `name`, with a fresh state.

    Raise ConfigurationError on an unknown name, on a hyperparameter the
    optimizer does not accept (listing the accepted ones), or on a value
    out of its bounds.
    """
    factory = get_from_registry(OPTIMIZERS, name, "optimizer")
    accepted = accepted_hyperparameters(factory)
    unknown = sorted(set(hyperparameters) - set(accepted))
    if unknown:
        raise ConfigurationError(
            f"unknown hyperparameter(s) for optimizer {name!r}: "
            f"{', '.join(map(repr, unknown))}. Accepted: "
            f"{', '.join(map(repr, accepted)) or 'none'}"
        )
    return factory(learning_rate, **hyperparameters)


__all__ = [
    "Hyperparameters",
    "sgd_step",
    "momentum_step",
    "rmsprop_step",
    "adam_step",
    "Optimizer",
    "OptimizerConfig",
    "OptimizerFactory",
    "SGD",
    "Momentum",
    "RMSprop",
    "Adam",
    "OPTIMIZERS",
    "accepted_hyperparameters",
    "make_optimizer",
]
