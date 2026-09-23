"""Saving and loading a self-contained ``model.json``.

The file holds everything the prediction program needs, so that it never
redeclares anything: the architecture, the weights, the compile
arguments, the scaler fitted on the training set, the class names, and
the hyperparameters and history of the training.

Version 2 of the format adds the early stopping configuration of the
training and the best and stopped epochs of its history. A version 1
file is still read, with no early stopping.

The optimizer is saved as its name, learning rate and hyperparameters
(``compile.optimizer.hyperparameters``, read with default values when
absent), never with its internal state (velocities, moment estimates):
the file is meant to predict, not to resume a training where it
stopped. A loaded model that is trained again starts with an empty
optimizer state.

:func:`model_to_dict` and :func:`model_from_dict` are pure; only
:func:`save_model` and :func:`load_model` touch the file system.
"""

import json
from typing import Any, Final, Mapping, NamedTuple, Sequence, TypedDict

import numpy as np

from mlp.early_stopping import EarlyStopping, EarlyStoppingConfig
from mlp.errors import ModelFileError, ShapeError
from mlp.history import History
from mlp.layers import LAYERS, LayerConfig
from mlp.network import Model
from mlp.optimizers import Optimizer, make_optimizer
from mlp.preprocessing import Scaler
from mlp.registry import get_from_registry
from mlp.types import StrPath

FORMAT_VERSION: Final[int] = 2
"""Version written by :func:`save_model`."""

SUPPORTED_VERSIONS: Final[tuple[int, ...]] = (1, 2)
"""Versions :func:`load_model` can read."""

PARAM_NAMES: Final[tuple[str, str]] = ("W", "b")
"""Keys of the parameters of a dense layer, in params() order."""


class TrainingConfig(TypedDict):
    """Hyperparameters of the training, kept for the record."""

    epochs: int
    batch_size: int
    seed: int | None
    early_stopping: EarlyStoppingConfig | None


class LoadedModel(NamedTuple):
    """What :func:`load_model` rebuilds from a model file."""

    model: Model
    scaler: Scaler
    labels: list[str]
    history: History
    training: TrainingConfig


def _expected_labels(output_units: int) -> int:
    """Return the number of classes an output layer stands for.

    A single unit is a binary probability, hence two classes.
    """
    return max(2, output_units)


def model_to_dict(
    model: Model,
    scaler: Scaler,
    labels: Sequence[str],
    training: TrainingConfig,
) -> dict[str, Any]:
    """Return the JSON serializable description of a trained model.

    Raise NotBuiltError when the model is not built and compiled, and
    ShapeError when the scaler or the labels do not match the input and
    output sizes of the model.
    """
    compile_config = model.get_compile_config()
    input_size = model.input_size
    if scaler.n_features != input_size:
        raise ShapeError(
            f"expected a scaler fitted on {input_size} features, "
            f"received one fitted on {scaler.n_features}"
        )
    architecture = [layer.get_config() for layer in model.layers]
    n_classes = _expected_labels(architecture[-1]["units"])
    if len(labels) != n_classes:
        raise ShapeError(
            f"expected {n_classes} labels for the output layer, "
            f"received {len(labels)}"
        )
    return {
        "format_version": FORMAT_VERSION,
        "architecture": architecture,
        "input_size": input_size,
        "compile": compile_config,
        "preprocessing": {"scaler": scaler.to_dict(),
                          "labels": list(labels)},
        "weights": [
            {name: p.tolist() for name, p in zip(PARAM_NAMES, layer.params())}
            for layer in model.layers
        ],
        "training": {**training, "history": model.history.to_dict()},
    }


def _layer_config(data: Any) -> LayerConfig:
    """Return `data` as a layer configuration, checking its fields."""
    if not isinstance(data, dict):
        raise TypeError(f"expected a layer object, received {data!r}")
    for key, kind in (("type", str), ("units", int), ("activation", str),
                      ("initializer", str)):
        if not isinstance(data[key], kind):
            raise TypeError(
                f"expected {kind.__name__} for {key!r}, "
                f"received {data[key]!r}"
            )
    return {"type": data["type"], "units": data["units"],
            "activation": data["activation"],
            "initializer": data["initializer"]}


def _positive_int(data: Mapping[str, Any], key: str) -> int:
    """Return ``data[key]``, which must be a positive integer."""
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"expected a positive integer for {key!r}, received {value!r}")
    return value


def _build_model(data: Mapping[str, Any]) -> Model:
    """Rebuild the layers, load their weights and compile the model."""
    input_size = _positive_int(data, "input_size")
    configs = [_layer_config(c) for c in data["architecture"]]
    weights = data["weights"]
    if not configs or len(weights) != len(configs):
        raise ValueError(
            f"expected as many weight entries as layers ({len(configs)}), "
            f"received {len(weights)}"
        )
    model = Model()
    size = input_size
    for i, (config, params) in enumerate(zip(configs, weights)):
        layer = get_from_registry(LAYERS, config["type"], "layer type")(
            config)
        W, b = (np.asarray(params[name], dtype=np.float64)
                for name in PARAM_NAMES)
        # set_params checks the rows and the bias; the columns chain one
        # layer to the previous one and are checked here.
        if W.ndim != 2 or W.shape[1] != size:
            raise ShapeError(
                f"layer {i}: expected weights with {size} columns, "
                f"received shape {W.shape}"
            )
        if not (np.all(np.isfinite(W)) and np.all(np.isfinite(b))):
            raise ValueError(f"layer {i}: non finite weights")
        try:
            layer.set_params([W, b])
        except ShapeError as e:
            raise ShapeError(f"layer {i}: {e}") from e
        model.add(layer)
        size = config["units"]

    compile_config = data["compile"]
    model.compile(
        compile_config["loss"],
        optimizer=_optimizer(compile_config["optimizer"]),
        metrics=list(compile_config["metrics"]),
    )
    return model


def _optimizer(data: Any) -> Optimizer:
    """Return the optimizer described by `data`, with an empty state.

    The hyperparameters are optional (files written before they were
    saved): the optimizer then takes its default values.
    """
    if not isinstance(data, dict):
        raise TypeError(f"expected an optimizer object, received {data!r}")
    hyperparameters = data.get("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        raise TypeError(
            "expected an object for the optimizer hyperparameters, "
            f"received {hyperparameters!r}"
        )
    return make_optimizer(data["name"], data["learning_rate"],
                          **hyperparameters)


def _training_config(data: Mapping[str, Any], version: int) -> TrainingConfig:
    """Return the training hyperparameters stored in `data`.

    A version 1 file predates early stopping, which is then None.
    """
    seed = data["seed"]
    if seed is not None and (isinstance(seed, bool)
                             or not isinstance(seed, int)):
        raise TypeError(f"expected an integer or null seed, received {seed!r}")
    early_stopping = data["early_stopping"] if version >= 2 else None
    if early_stopping is not None:
        if not isinstance(early_stopping, dict):
            raise TypeError(
                "expected an object or null for early_stopping, "
                f"received {early_stopping!r}"
            )
        # Round trip through the class: validated and normalised.
        early_stopping = EarlyStopping.from_config(early_stopping).to_config()
    return {"epochs": _positive_int(data, "epochs"),
            "batch_size": _positive_int(data, "batch_size"),
            "seed": seed,
            "early_stopping": early_stopping}


def model_from_dict(data: Any) -> LoadedModel:
    """Rebuild the model and its companions from :func:`model_to_dict`.

    Raise ModelFileError on an unknown format version, a missing key, a
    wrong type, an unknown name, or inconsistent shapes.
    """
    if not isinstance(data, dict):
        raise ModelFileError(
            f"expected a JSON object, received {type(data).__name__}")
    version = data.get("format_version")
    if isinstance(version, bool) or version not in SUPPORTED_VERSIONS:
        raise ModelFileError(
            "expected a format_version in "
            f"{', '.join(map(str, SUPPORTED_VERSIONS))}, "
            f"received {version!r}"
        )
    try:
        model = _build_model(data)
        preprocessing = data["preprocessing"]
        scaler = Scaler.from_dict(preprocessing["scaler"])
        if scaler.n_features != model.input_size:
            raise ShapeError(
                f"expected a scaler over {model.input_size} features, "
                f"received one over {scaler.n_features}"
            )
        labels = preprocessing["labels"]
        n_classes = _expected_labels(model.layers[-1].get_config()["units"])
        if not isinstance(labels, list) or len(labels) != n_classes \
                or not all(isinstance(label, str) for label in labels):
            raise ValueError(
                f"expected {n_classes} class names, received {labels!r}")
        training = _training_config(data["training"], version)
        history = History.from_dict(data["training"]["history"])
    except KeyError as e:
        raise ModelFileError(f"invalid model: missing key {e}") from e
    except (TypeError, ValueError, AttributeError) as e:
        # ConfigurationError and ShapeError are ValueErrors too.
        raise ModelFileError(f"invalid model: {e}") from e
    model.history = history
    return LoadedModel(model, scaler, list(labels), history, training)


def save_model(
    path: StrPath,
    model: Model,
    scaler: Scaler,
    labels: Sequence[str],
    training: TrainingConfig,
) -> None:
    """Write the model description of :func:`model_to_dict` to `path`.

    Raise ModelFileError when the file cannot be written, plus the
    errors of :func:`model_to_dict`.
    """
    content = model_to_dict(model, scaler, labels, training)
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(content, file, allow_nan=False)
    except (OSError, ValueError) as e:
        raise ModelFileError(f"cannot write {str(path)!r}: {e}") from e


def load_model(path: StrPath) -> LoadedModel:
    """Read `path` and rebuild the model it describes.

    Raise ModelFileError when the file cannot be read, is not valid
    JSON, or does not describe a valid model.
    """
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ModelFileError(f"cannot read {str(path)!r}: {e}") from e
    try:
        return model_from_dict(data)
    except ModelFileError as e:
        raise ModelFileError(f"{str(path)!r}: {e}") from e


__all__ = [
    "FORMAT_VERSION",
    "SUPPORTED_VERSIONS",
    "TrainingConfig",
    "LoadedModel",
    "model_to_dict",
    "model_from_dict",
    "save_model",
    "load_model",
]
