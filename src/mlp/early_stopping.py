"""Early stopping: end the training when a monitored value stops improving.

:class:`EarlyStopping` is an immutable, validated configuration. The
decision itself is taken by pure functions over an immutable
:class:`EarlyStoppingState`, with no reference to the model: the patience,
``min_delta`` and min/max logic can be tested on hand-made sequences of
values. Keeping a copy of the best weights and restoring them is the job
of :meth:`mlp.network.Model.fit`.
"""

import math
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence, TypedDict

from mlp.errors import ConfigurationError

MODES: Final[tuple[str, ...]] = ("min", "max", "auto")
SPLITS: Final[tuple[str, ...]] = ("train", "valid")


class EarlyStoppingConfig(TypedDict):
    """The fields of :class:`EarlyStopping`, in a serializable form."""

    monitor: str
    patience: int
    min_delta: float
    mode: str
    restore_best_weights: bool


@dataclass(frozen=True)
class EarlyStopping:
    """When to stop a training, and whether to restore the best weights.

    `monitor` is ``"<train|valid>_<loss|metric>"``. The training stops
    once it has not improved by more than `min_delta` for `patience`
    consecutive epochs. With `restore_best_weights`, the model ends up
    with the weights of its best epoch rather than of its last one.
    """

    monitor: str = "valid_loss"
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "auto"
    restore_best_weights: bool = True

    def __post_init__(self) -> None:
        """Raise ConfigurationError on any invalid field."""
        if (isinstance(self.patience, bool)
                or not isinstance(self.patience, int) or self.patience < 1):
            raise ConfigurationError(
                "expected an integer >= 1 for the patience, "
                f"received {self.patience!r}"
            )
        if (isinstance(self.min_delta, bool)
                or not isinstance(self.min_delta, (int, float))
                or not math.isfinite(self.min_delta) or self.min_delta < 0):
            raise ConfigurationError(
                "expected a finite number >= 0 for min_delta, "
                f"received {self.min_delta!r}"
            )
        if self.mode not in MODES:
            raise ConfigurationError(
                f"expected a mode in {', '.join(MODES)}, "
                f"received {self.mode!r}"
            )
        if not isinstance(self.restore_best_weights, bool):
            raise ConfigurationError(
                "expected a boolean for restore_best_weights, "
                f"received {self.restore_best_weights!r}"
            )
        split_monitor(self.monitor)

    @property
    def split(self) -> str:
        """Return the monitored set, ``"train"`` or ``"valid"``."""
        return split_monitor(self.monitor)[0]

    @property
    def name(self) -> str:
        """Return the monitored quantity, ``"loss"`` or a metric name."""
        return split_monitor(self.monitor)[1]

    def resolved_mode(self) -> str:
        """Return ``"min"`` or ``"max"``.

        ``"auto"`` minimises a loss and maximises a metric (accuracy,
        f1...), which are all scores where higher is better.
        """
        if self.mode != "auto":
            return self.mode
        return "min" if self.name == "loss" else "max"

    def to_config(self) -> EarlyStoppingConfig:
        """Return the fields as a JSON serializable dict."""
        return {
            "monitor": self.monitor,
            "patience": self.patience,
            "min_delta": float(self.min_delta),
            "mode": self.mode,
            "restore_best_weights": self.restore_best_weights,
        }

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> "EarlyStopping":
        """Rebuild the configuration from the output of to_config().

        Raise ConfigurationError on a missing, unknown or invalid field.
        """
        expected = set(EarlyStoppingConfig.__annotations__)
        if set(data) != expected:
            raise ConfigurationError(
                f"expected the early stopping fields {sorted(expected)}, "
                f"received {sorted(data)}"
            )
        return cls(**data)


def split_monitor(monitor: str) -> tuple[str, str]:
    """Split ``"valid_loss"`` into ``("valid", "loss")``.

    Raise ConfigurationError unless `monitor` is a set name, an
    underscore and a non empty quantity name.
    """
    split, sep, name = (monitor.partition("_") if isinstance(monitor, str)
                        else ("", "", ""))
    if split not in SPLITS or not sep or not name:
        raise ConfigurationError(
            "expected a monitored value of the form "
            f"'<{'|'.join(SPLITS)}>_<loss|metric>', received {monitor!r}"
        )
    return split, name


def check_monitor(
    config: EarlyStopping,
    metrics: Sequence[str],
    has_validation: bool,
) -> None:
    """Check that the training will record the monitored value.

    Raise ConfigurationError when a ``valid_*`` value is monitored
    without validation data, or when the quantity is neither the loss
    nor one of the compiled `metrics`.
    """
    if config.split == "valid" and not has_validation:
        raise ConfigurationError(
            f"cannot monitor {config.monitor!r} without validation data")
    if config.name != "loss" and config.name not in metrics:
        splits = SPLITS if has_validation else ("train",)
        available = [f"{split}_{name}" for split in splits
                     for name in ("loss", *metrics)]
        raise ConfigurationError(
            f"cannot monitor {config.monitor!r}, expected one of: "
            f"{', '.join(available)}"
        )


@dataclass(frozen=True)
class EarlyStoppingState:
    """Progress of the early stopping after some epochs.

    `best` starts at +inf in min mode and -inf in max mode, and
    `best_epoch` (1-based) at 0 while no epoch has been seen.
    """

    best: float
    best_epoch: int
    wait: int


def is_improvement(
    value: float,
    best: float,
    min_delta: float,
    mode: str,
) -> bool:
    """Tell whether `value` beats `best` by more than `min_delta`."""
    if mode == "min":
        return value < best - min_delta
    if mode == "max":
        return value > best + min_delta
    raise ConfigurationError(
        f"expected a mode in min, max, received {mode!r}")


def initial_state(config: EarlyStopping) -> EarlyStoppingState:
    """Return the state before the first epoch."""
    worst = math.inf if config.resolved_mode() == "min" else -math.inf
    return EarlyStoppingState(best=worst, best_epoch=0, wait=0)


def update_state(
    state: EarlyStoppingState,
    value: float,
    epoch: int,
    config: EarlyStopping,
) -> tuple[EarlyStoppingState, bool]:
    """Take the monitored `value` of the 1-based `epoch` into account.

    Return the new state and whether the training must stop, which
    happens at epoch ``best_epoch + patience`` without improvement.
    """
    if is_improvement(value, state.best, config.min_delta,
                      config.resolved_mode()):
        return EarlyStoppingState(best=value, best_epoch=epoch, wait=0), False
    wait = state.wait + 1
    return (EarlyStoppingState(state.best, state.best_epoch, wait),
            wait >= config.patience)


__all__ = [
    "EarlyStopping",
    "EarlyStoppingConfig",
    "EarlyStoppingState",
    "split_monitor",
    "check_monitor",
    "is_improvement",
    "initial_state",
    "update_state",
]
