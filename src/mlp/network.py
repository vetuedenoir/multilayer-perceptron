"""The sequential model: build, compile, fit, evaluate and predict.

:class:`Model` only talks to its layers through the
:class:`~mlp.layers.Layer` protocol, and to the parameter update rule
through the :class:`~mlp.optimizers.Optimizer` protocol. Everything that
can be computed without state lives in module level pure functions.
"""

from typing import Iterable, Iterator, Mapping, Sequence, TypedDict

import numpy as np

from mlp.errors import (
    ConfigurationError,
    NotBuiltError,
    ShapeError,
    TrainingDivergedError,
)
from mlp.history import History
from mlp.layers import Layer
from mlp.losses import LOSSES, GradFn, Loss, resolve_output_grad
from mlp.metrics import METRICS, MetricFn
from mlp.optimizers import OPTIMIZERS, Optimizer, OptimizerConfig
from mlp.registry import get_from_registry, get_many_from_registry
from mlp.types import FloatArray, IntArray


class CompileConfig(TypedDict):
    """The arguments of compile(), in a serializable form."""

    loss: str
    optimizer: OptimizerConfig
    metrics: list[str]


def _check_positive_int(value: int, name: str) -> None:
    """Raise ConfigurationError unless `value` is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(
            f"expected a positive integer for {name!r}, received {value!r}")


def iter_batches(
    x: FloatArray,
    y: FloatArray,
    batch_size: int,
    rng: np.random.Generator | None = None,
) -> Iterator[tuple[FloatArray, FloatArray]]:
    """Yield ``(x_batch, y_batch)`` pairs covering the dataset once.

    The samples are shuffled with `rng`, or kept in order when it is
    None. The last batch is smaller when the number of samples is not a
    multiple of `batch_size`.
    """
    _check_positive_int(batch_size, "batch_size")
    m = x.shape[0]
    indices = rng.permutation(m) if rng is not None else np.arange(m)
    for start in range(0, m, batch_size):
        batch = indices[start:start + batch_size]
        yield x[batch], y[batch]


def to_class_labels(y: FloatArray) -> IntArray:
    """Turn network outputs or targets into 1-D class labels.

    One column is a binary probability, thresholded at 0.5. Several
    columns are a distribution (or a one-hot encoding), so the class is
    the argmax.
    """
    if y.ndim != 2 or y.shape[1] == 0:
        raise ShapeError(
            f"expected an array of shape (m, n) with n >= 1, "
            f"received {y.shape}"
        )
    if y.shape[1] == 1:
        return (y[:, 0] >= 0.5).astype(np.int64)
    return np.argmax(y, axis=1).astype(np.int64)


def check_targets(loss: Loss, y: FloatArray, output_units: int) -> None:
    """Check that `y` and the output layer fit the loss.

    Raise ShapeError when the binary cross-entropy does not get a
    ``(m, 1)`` target and a single output unit, or when the categorical
    cross-entropy does not get a ``(m, n)`` one-hot target with ``n > 1``
    matching the number of output units.
    """
    if y.ndim != 2:
        raise ShapeError(
            f"expected a two-dimensional target, received shape {y.shape}")
    n = y.shape[1]
    if loss.name == "binaryCrossentropy":
        if n != 1:
            raise ShapeError(
                "expected a target of shape (m, 1) for the binary "
                f"cross-entropy, received {y.shape}"
            )
        if output_units != 1:
            raise ShapeError(
                "expected 1 unit in the output layer for the binary "
                f"cross-entropy, received {output_units}"
            )
    elif loss.name == "categoricalCrossentropy":
        if n <= 1:
            raise ShapeError(
                "expected a one-hot target of shape (m, n) with n > 1 for "
                f"the categorical cross-entropy, received {y.shape}"
            )
        if output_units != n:
            raise ShapeError(
                f"expected {n} units in the output layer to match the "
                f"target of shape {y.shape}, received {output_units}"
            )


def format_epoch_line(
    epoch: int,
    epochs: int,
    train: Mapping[str, float],
    valid: Mapping[str, float] | None = None,
) -> str:
    """Return the line displayed at the end of an epoch.

    `epoch` is 1-based. Example::

        epoch  3/60 - train_loss: 0.2104 | valid_loss: 0.1987
    """
    width = len(str(epochs))
    line = f"epoch {epoch:>{width}}/{epochs} - " + ", ".join(
        f"train_{name}: {value:.4f}" for name, value in train.items())
    if valid:
        line += " | " + ", ".join(
            f"valid_{name}: {value:.4f}" for name, value in valid.items())
    return line


class Model:
    """A stack of layers trained by gradient descent."""

    def __init__(self, layers: Iterable[Layer] | None = None) -> None:
        """Create a model from `layers`, copied into a new list."""
        # Copied: a caller keeping a reference to its list must not be
        # able to mutate the model through it.
        self.layers: list[Layer] = []
        for layer in layers or ():
            self.add(layer)
        self._loss: Loss | None = None
        self._output_grad: GradFn | None = None
        self._fused = False
        self._optimizer: Optimizer | None = None
        self._metrics: dict[str, MetricFn] = {}
        self.history = History()

    def add(self, layer: Layer) -> None:
        """Append `layer` at the end of the model."""
        if not isinstance(layer, Layer):
            raise TypeError(
                "expected an object implementing the Layer protocol, "
                f"received {type(layer).__name__}"
            )
        self.layers.append(layer)

    def __len__(self) -> int:
        """Return the number of layers."""
        return len(self.layers)

    @property
    def built(self) -> bool:
        """Tell whether every layer has its parameters."""
        return bool(self.layers) and all(lay.built for lay in self.layers)

    @property
    def compiled(self) -> bool:
        """Tell whether compile() has been called."""
        return self._loss is not None

    @property
    def metrics(self) -> list[str]:
        """Return the names of the metrics given to compile()."""
        return list(self._metrics)

    @property
    def input_size(self) -> int:
        """Return the number of features the model expects.

        Raise NotBuiltError when the model is not built yet.
        """
        if not self.built:
            raise NotBuiltError(
                "the model must be built before its input size is known")
        return int(self.layers[0].params()[0].shape[1])

    def get_compile_config(self) -> CompileConfig:
        """Return the loss, optimizer and metrics given to compile().

        Raise NotBuiltError when the model is not compiled.
        """
        loss, optimizer = self._require_compiled()
        return {
            "loss": loss.name,
            "optimizer": optimizer.get_config(),
            "metrics": self.metrics,
        }

    def build(self, input_size: int, seed: int | None = None) -> None:
        """Allocate the parameters of the layers that do not have any.

        Idempotent: layers already built (by a previous build, a fit or
        a loaded model) keep their parameters.
        """
        self._build(input_size, np.random.default_rng(seed))

    def _build(self, input_size: int, rng: np.random.Generator) -> None:
        """Build the missing layers, drawing from `rng`."""
        if not self.layers:
            raise ConfigurationError("cannot build a model without layers")
        _check_positive_int(input_size, "input_size")
        size = input_size
        for layer in self.layers:
            if not layer.built:
                layer.build(size, rng)
            size = layer.get_config()["units"]

    def summary(self) -> str:
        """Return a table of the layers and of the parameter counts.

        Raise NotBuiltError when the model is not built yet: before
        that, the shapes of the weights are unknown.
        """
        if not self.built:
            raise NotBuiltError(
                "the model must be built (build() or fit()) before "
                "calling summary()"
            )
        lines = []
        total = 0
        for i, layer in enumerate(self.layers):
            config = layer.get_config()
            params = layer.params()
            total += sum(p.size for p in params)
            shapes = ", ".join(str(p.shape) for p in params)
            lines.append(
                f"layer {i}: {config['type']}, units={config['units']}, "
                f"activation={config['activation']}, params=[{shapes}]"
            )
        lines.append(f"total parameters: {total}")
        return "\n".join(lines)

    def compile(
        self,
        loss: str,
        optimizer: str | Optimizer = "sgd",
        metrics: Sequence[str] = ("accuracy",),
        learning_rate: float = 0.001,
    ) -> None:
        """Choose the loss, the optimizer and the metrics.

        `optimizer` is either a registry name, instantiated with
        `learning_rate`, or an already configured optimizer, in which
        case `learning_rate` is ignored. The layers are never modified,
        so compile() can be called again.

        Raise ConfigurationError on an unknown name, on a softmax output
        with the binary cross-entropy, or on a softmax over one unit.
        """
        if not self.layers:
            raise ConfigurationError(
                "cannot compile a model without layers")
        resolved_loss = get_from_registry(LOSSES, loss, "loss")
        last = self.layers[-1].get_config()
        if last["activation"] == "softmax":
            if resolved_loss.name == "binaryCrossentropy":
                raise ConfigurationError(
                    "a softmax output layer is not compatible with the "
                    "binary cross-entropy, use the categorical one"
                )
            if last["units"] == 1:
                raise ConfigurationError(
                    "expected at least 2 units in a softmax output layer "
                    "(it outputs a distribution over its units), "
                    "received 1"
                )
        if isinstance(optimizer, str):
            factory = get_from_registry(OPTIMIZERS, optimizer, "optimizer")
            resolved_optimizer = factory(learning_rate)
        else:
            resolved_optimizer = optimizer
        metric_fns = get_many_from_registry(METRICS, metrics, "metric")

        self._loss = resolved_loss
        self._output_grad, self._fused = resolve_output_grad(
            resolved_loss, last["activation"])
        self._optimizer = resolved_optimizer
        self._metrics = dict(zip(metrics, metric_fns))

    def forward(self, x: FloatArray) -> FloatArray:
        """Return the output of the network for the batch `x`."""
        if not self.built:
            raise NotBuiltError(
                "the model must be built (build() or fit()) before "
                "calling forward()"
            )
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: FloatArray, from_loss: bool = False) -> None:
        """Backpropagate `grad` through every layer, last one first.

        `from_loss` tells that `grad` is already the gradient w.r.t. the
        pre-activation of the output layer.
        """
        grad = self.layers[-1].backward(grad, from_loss=from_loss)
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)

    def update(self) -> None:
        """Apply the optimizer to the parameters of every layer."""
        optimizer = self._require_compiled()[1]
        for layer in self.layers:
            layer.set_params(optimizer.step(layer.params(), layer.grads()))

    def predict_proba(self, x: FloatArray) -> FloatArray:
        """Return the raw output of the network (probabilities)."""
        return self.forward(x)

    def predict_classes(self, x: FloatArray) -> IntArray:
        """Return the predicted class label of every sample of `x`."""
        return to_class_labels(self.forward(x))

    def evaluate(
        self,
        x: FloatArray,
        y: FloatArray,
    ) -> tuple[float, dict[str, float]]:
        """Return the loss and the metrics of the model on ``(x, y)``."""
        loss = self._require_compiled()[0]
        y_hat = self.forward(x)
        y_true = to_class_labels(y)
        y_pred = to_class_labels(y_hat)
        metrics = {
            name: fn(y_true, y_pred) for name, fn in self._metrics.items()
        }
        return loss.forward(y, y_hat), metrics

    def fit(
        self,
        x: FloatArray,
        y: FloatArray,
        *,
        epochs: int,
        batch_size: int,
        validation_data: tuple[FloatArray, FloatArray] | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> History:
        """Train the model and return the history of this training.

        The layers not built yet are built first, with the same seeded
        generator that shuffles the batches: a given seed gives the same
        training. Layers already built keep their weights, so calling
        fit() again resumes the training.

        Raise NotBuiltError before compile(), ShapeError on targets that
        do not fit the loss, and TrainingDivergedError when the loss
        stops being finite.
        """
        loss, _ = self._require_compiled()
        _check_positive_int(epochs, "epochs")
        _check_positive_int(batch_size, "batch_size")
        self._check_samples(x, y)
        if validation_data is not None:
            self._check_samples(*validation_data)

        rng = np.random.default_rng(seed)
        self._build(x.shape[1], rng)
        output_units = self.layers[-1].get_config()["units"]
        check_targets(loss, y, output_units)
        if validation_data is not None:
            check_targets(loss, validation_data[1], output_units)

        history = History()
        for epoch in range(1, epochs + 1):
            for x_batch, y_batch in iter_batches(x, y, batch_size, rng):
                self._train_step(x_batch, y_batch)

            train_loss, train_metrics = self.evaluate(x, y)
            train = {"loss": train_loss, **train_metrics}
            valid: dict[str, float] | None = None
            if validation_data is not None:
                valid_loss, valid_metrics = self.evaluate(*validation_data)
                valid = {"loss": valid_loss, **valid_metrics}
            for values in (train, valid):
                if values is not None and not np.isfinite(values["loss"]):
                    raise TrainingDivergedError(
                        f"the loss became {values['loss']} at epoch "
                        f"{epoch}, try a lower learning rate"
                    )
            history.append(train, valid)
            if verbose:
                print(format_epoch_line(epoch, epochs, train, valid))
        self.history = history
        return history

    def _train_step(self, x: FloatArray, y: FloatArray) -> None:
        """Run forward, backward and update on one batch."""
        if self._output_grad is None:
            raise NotBuiltError("the model must be compiled")
        y_hat = self.forward(x)
        self.backward(self._output_grad(y, y_hat), from_loss=self._fused)
        self.update()

    def _require_compiled(self) -> tuple[Loss, Optimizer]:
        """Return the loss and optimizer, or raise NotBuiltError."""
        if self._loss is None or self._optimizer is None:
            raise NotBuiltError(
                "the model must be compiled (compile()) first")
        return self._loss, self._optimizer

    @staticmethod
    def _check_samples(x: FloatArray, y: FloatArray) -> None:
        """Raise ShapeError unless x and y are 2-D with as many rows."""
        if x.ndim != 2 or y.ndim != 2:
            raise ShapeError(
                "expected two-dimensional x and y, received shapes "
                f"{x.shape} and {y.shape}"
            )
        if x.shape[0] != y.shape[0]:
            raise ShapeError(
                "expected as many samples in x as in y, received "
                f"{x.shape[0]} and {y.shape[0]}"
            )


__all__ = [
    "Model",
    "CompileConfig",
    "iter_batches",
    "to_class_labels",
    "check_targets",
    "format_epoch_line",
]
