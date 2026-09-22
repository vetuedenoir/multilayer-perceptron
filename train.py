#!/usr/bin/env python3
"""Train a multilayer perceptron and save it to a model file.

Second of the three programs of the subject. The scaler is fitted on the
training set only, then applied to both sets: the validation set stays
unseen data. Everything the prediction program needs (architecture,
weights, scaler, class names, history) ends up in the model file.
"""

import argparse
import sys
from typing import Final, Sequence

import numpy as np

from mlp.data import LABELS, read_dataset, to_arrays
from mlp.errors import MLPError
from mlp.layers import DenseLayer
from mlp.losses import LOSSES
from mlp.network import Model
from mlp.plotting import plot_history
from mlp.preprocessing import fit_scaler, one_hot, transform
from mlp.registry import get_from_registry
from mlp.serialization import save_model
from mlp.types import FloatArray, IntArray

PROG: Final[str] = "train.py"
SCALER: Final[str] = "standard"
METRICS: Final[tuple[str, ...]] = ("accuracy", "precision", "recall", "f1")

DEFAULT_LOSSES: Final[dict[str, str]] = {
    "softmax": "categoricalCrossentropy",
    "sigmoid": "binaryCrossentropy",
}
"""Loss used when --loss is not given, by output activation."""


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
    parser.add_argument("--layers", type=int, nargs="+", default=[24, 24],
                        metavar="UNITS",
                        help="units of each hidden layer "
                             "(default: %(default)s)")
    parser.add_argument("--activation", default="relu",
                        help="activation of the hidden layers "
                             "(default: %(default)s)")
    parser.add_argument("--initializer", default="heUniform",
                        help="weights initializer of every layer "
                             "(default: %(default)s)")
    parser.add_argument("--output-activation", default="softmax",
                        help="activation of the output layer "
                             "(default: %(default)s)")
    parser.add_argument("--loss", default=None,
                        help="loss function (default: categoricalCrossentropy "
                             "for a softmax output, binaryCrossentropy for "
                             "a sigmoid one)")
    parser.add_argument("--epochs", type=int, default=84,
                        help="number of epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="samples per gradient step "
                             "(default: %(default)s)")
    parser.add_argument("--learning-rate", type=float, default=0.0314,
                        help="step of the gradient descent "
                             "(default: %(default)s)")
    parser.add_argument("--seed", type=int, default=14,
                        help="seed of the weights and of the shuffle "
                             "(default: %(default)s)")
    parser.add_argument("--model", default="model.json",
                        help="where the model is saved "
                             "(default: %(default)s)")
    parser.add_argument("--plot", action="store_true",
                        help="plot the learning curves at the end")
    return parser.parse_args(argv)


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


def build_model(args: argparse.Namespace, loss: str) -> Model:
    """Return the network described by the command line, compiled."""
    layers = [DenseLayer(units, activation=args.activation,
                         weights_initializer=args.initializer)
              for units in args.layers]
    layers.append(DenseLayer(output_units(loss),
                             activation=args.output_activation,
                             weights_initializer=args.initializer))
    model = Model(layers)
    model.compile(loss, optimizer="sgd", metrics=METRICS,
                  learning_rate=args.learning_rate)
    return model


def main(argv: Sequence[str] | None = None) -> int:
    """Run the program and return its exit status."""
    args = parse_args(argv)
    try:
        loss = resolve_loss(args.loss, args.output_activation)
        model = build_model(args, loss)

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
        )
        save_model(args.model, model, scaler, LABELS,
                   {"epochs": args.epochs, "batch_size": args.batch_size,
                    "seed": args.seed})
        print(f"> saving model '{args.model}' to disk...")
    except (MLPError, OSError) as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 1

    if args.plot:
        plot_history(history)
    return 0


if __name__ == "__main__":
    sys.exit(main())
