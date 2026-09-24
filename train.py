#!/usr/bin/env python3
"""Train a multilayer perceptron and save it to a model file.

Second of the three programs of the subject. The scaler is fitted on the
training set only, then applied to both sets: the validation set stays
unseen data. Everything the prediction program needs (architecture,
weights, scaler, class names, history) ends up in the model file.

The network is described either by the command line options (their
defaults give the default network), or entirely by a JSON file given
with --arch-file, which then cannot be combined with those options.
"""

import argparse
import json
import sys
from typing import Any, Final, Sequence

import numpy as np

from mlp.data import LABELS, read_dataset, to_arrays
from mlp.early_stopping import EarlyStopping, check_monitor
from mlp.errors import ConfigurationError, MLPError
from mlp.layers import DenseLayer
from mlp.losses import LOSSES
from mlp.network import Model
from mlp.optimizers import make_optimizer
from mlp.plotting import plot_history
from mlp.preprocessing import fit_scaler, one_hot, transform
from mlp.registry import get_from_registry
from mlp.serialization import save_model
from mlp.types import FloatArray, IntArray

PROG: Final[str] = "train.py"
SCALER: Final[str] = "standard"
METRICS: Final[tuple[str, ...]] = ("accuracy", "precision", "recall", "f1")

ARCH_OPTIONS: Final[tuple[str, ...]] = (
    "layers", "activation", "initializer", "output_activation", "loss",
    "optimizer", "learning_rate",
)
"""Options describing the network, replaced as a whole by --arch-file."""

DEFAULT_LOSSES: Final[dict[str, str]] = {
    "softmax": "categoricalCrossentropy",
    "sigmoid": "binaryCrossentropy",
}
"""Loss used when --loss is not given, by output activation."""

EARLY_STOPPING: Final[EarlyStopping] = EarlyStopping()
"""Source of the defaults of the early stopping options."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line, `argv` defaulting to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Train a multilayer perceptron on the training set "
                    "and save it.",
    )
    parser.add_argument("--train", default="data_train.csv",
                        help="the training set (default: %(default)s)")
    parser.add_argument("--valid", default="data_valid.csv",
                        help="the validation set (default: %(default)s)")
    parser.add_argument("--layers", type=int, nargs="+", default=[64, 32],
                        metavar="UNITS",
                        help="units of each hidden layer "
                             "(default: %(default)s)")
    parser.add_argument("--activation", default="sigmoid",
                        help="activation of the hidden layers "
                             "(default: %(default)s)")
    parser.add_argument("--initializer", default="glorot_uniform",
                        help="weights initializer of every layer "
                             "(default: %(default)s)")
    parser.add_argument("--output-activation", default="softmax",
                        help="activation of the output layer "
                             "(default: %(default)s)")
    parser.add_argument("--loss", default=None,
                        help="loss function (default: categoricalCrossentropy "
                             "for a softmax output, binaryCrossentropy for "
                             "a sigmoid one)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="samples per gradient step "
                             "(default: %(default)s)")
    parser.add_argument("--optimizer", default="rmsprop",
                        help="update rule, one of sgd, momentum, nesterov, "
                             "rmsprop, adam, with the default "
                             "hyperparameters of the literature "
                             "(default: %(default)s)")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="step of the gradient descent "
                             "(default: %(default)s)")
    parser.add_argument("--arch-file", metavar="PATH",
                        help="read the whole network (layers, loss, "
                             "optimizer and its hyperparameters) from a "
                             "JSON file instead of the options above")
    parser.add_argument("--seed", type=int, default=14,
                        help="seed of the weights and of the shuffle "
                             "(default: %(default)s)")
    parser.add_argument("--model", default="model.json",
                        help="where the model is saved "
                             "(default: %(default)s)")
    parser.add_argument("--plot", action="store_true",
                        help="plot the learning curves at the end")

    stopping = parser.add_argument_group(
        "early stopping",
        "ignored without --early-stopping",
    )
    stopping.add_argument("--early-stopping", action="store_true",
                          help="stop when the monitored value stops "
                               "improving")
    stopping.add_argument("--patience", type=int,
                          default=EARLY_STOPPING.patience,
                          help="epochs without improvement before stopping "
                               "(default: %(default)s)")
    stopping.add_argument("--min-delta", type=float,
                          default=EARLY_STOPPING.min_delta,
                          help="smallest change counted as an improvement "
                               "(default: %(default)s)")
    stopping.add_argument("--monitor", default=EARLY_STOPPING.monitor,
                          help="value watched, <train|valid>_<loss|metric> "
                               "(default: %(default)s)")
    stopping.add_argument("--no-restore-best", action="store_true",
                          help="keep the weights of the last epoch instead "
                               "of the best one")
    args = parser.parse_args(argv)
    if args.arch_file is not None:
        given = [name for name in ARCH_OPTIONS
                 if getattr(args, name) != parser.get_default(name)]
        if given:
            parser.error("--arch-file cannot be combined with "
                         + ", ".join("--" + name.replace("_", "-")
                                     for name in given))
    return args


def resolve_loss(loss: str | None, output_activation: str) -> str:
    """Return the canonical name of the loss to train with.

    Without an explicit `loss`, the one fused with `output_activation`
    is chosen. Raise ConfigurationError when there is none, or when
    `loss` is not in the registry.
    """
    if loss is None:
        loss = get_from_registry(
            DEFAULT_LOSSES, output_activation.lower(),
            "output activation without an explicit --loss")
    return get_from_registry(LOSSES, loss, "loss").name


def output_units(loss: str) -> int:
    """Return the size of the output layer the loss expects.

    One probability for the binary cross-entropy, one per class for the
    categorical one.
    """
    return 1 if loss == "binaryCrossentropy" else len(LABELS)


def encode_targets(labels: IntArray, loss: str) -> FloatArray:
    """Return `labels` in the target form the loss expects.

    A ``(m, 1)`` column of 0/1 for the binary cross-entropy, a one-hot
    ``(m, n_classes)`` matrix for the categorical one.
    """
    if loss == "binaryCrossentropy":
        return labels.reshape(-1, 1).astype(np.float64)
    return one_hot(labels, len(LABELS))


def cli_architecture(args: argparse.Namespace) -> dict[str, Any]:
    """Return the network of the command line options, as --arch-file."""
    return {
        "hidden": [{"units": units, "activation": args.activation,
                    "initializer": args.initializer}
                   for units in args.layers],
        "output": {"activation": args.output_activation,
                   "initializer": args.initializer},
        "loss": args.loss,
        "optimizer": {"name": args.optimizer,
                      "learning_rate": args.learning_rate},
    }


def read_architecture(path: str) -> Any:
    """Return the content of the JSON file `path`.

    Raise ConfigurationError when it cannot be read or parsed.
    """
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ConfigurationError(f"cannot read {path!r}: {e}") from e


def build_model(arch: Any) -> tuple[Model, str]:
    """Return the compiled network `arch` describes, and its loss.

    `arch` has the layout of :func:`cli_architecture`; the loss may be
    absent or null, and the optimizer hyperparameters absent. The output
    units follow from the loss. The names and values are checked by the
    layers, the optimizer and compile(); a missing key or a wrong type
    is a ConfigurationError too.
    """
    try:
        output = arch["output"]
        loss = resolve_loss(arch.get("loss"), output["activation"])
        layers = [DenseLayer(layer["units"], activation=layer["activation"],
                             weights_initializer=layer["initializer"])
                  for layer in arch["hidden"]]
        layers.append(DenseLayer(output_units(loss),
                                 activation=output["activation"],
                                 weights_initializer=output["initializer"]))
        optimizer = arch["optimizer"]
        model = Model(layers)
        model.compile(loss, metrics=METRICS, optimizer=make_optimizer(
            optimizer["name"], optimizer["learning_rate"],
            **optimizer.get("hyperparameters", {})))
    except KeyError as e:
        raise ConfigurationError(f"missing key {e}") from e
    except (TypeError, AttributeError) as e:
        raise ConfigurationError(f"invalid architecture: {e}") from e
    return model, loss


def build_early_stopping(args: argparse.Namespace) -> EarlyStopping | None:
    """Return the early stopping of the command line, None when off.

    The monitored value is checked against the compiled metrics right
    away, so that a typo fails before the datasets are read.
    """
    if not args.early_stopping:
        return None
    early_stopping = EarlyStopping(
        monitor=args.monitor,
        patience=args.patience,
        min_delta=args.min_delta,
        restore_best_weights=not args.no_restore_best,
    )
    check_monitor(early_stopping, METRICS, has_validation=True)
    return early_stopping


def main(argv: Sequence[str] | None = None) -> int:
    """Run the program and return its exit status."""
    args = parse_args(argv)
    try:
        if args.arch_file is None:
            model, loss = build_model(cli_architecture(args))
        else:
            try:
                model, loss = build_model(read_architecture(args.arch_file))
            except ConfigurationError as e:
                raise ConfigurationError(f"{args.arch_file}: {e}") from e
        early_stopping = build_early_stopping(args)

        x_train, labels_train = to_arrays(read_dataset(args.train))
        x_valid, labels_valid = to_arrays(read_dataset(args.valid))
        # Fitted on the training set only: the statistics of the
        # validation set must not leak into the training.
        scaler = fit_scaler(SCALER, x_train)
        x_train = transform(scaler, x_train)
        x_valid = transform(scaler, x_valid)
        y_train = encode_targets(labels_train, loss)
        y_valid = encode_targets(labels_valid, loss)
        print(f"x_train shape : {x_train.shape}")
        print(f"x_valid shape : {x_valid.shape}")

        model.build(x_train.shape[1], seed=args.seed)
        print(model.summary())
        history = model.fit(
            x_train, y_train,
            validation_data=(x_valid, y_valid),
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
            early_stopping=early_stopping,
        )
        save_model(args.model, model, scaler, LABELS,
                   {"epochs": args.epochs, "batch_size": args.batch_size,
                    "seed": args.seed,
                    "early_stopping": (early_stopping.to_config()
                                       if early_stopping else None)})
        print(f"> saving model '{args.model}' to disk...")
    except (MLPError, OSError) as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 1

    if args.plot:
        plot_history(history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
