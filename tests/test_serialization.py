"""Tests of model.json: round trip and rejection of broken files."""

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from mlp.errors import ModelFileError, NotBuiltError, ShapeError
from mlp.layers import DenseLayer
from mlp.network import Model
from mlp.preprocessing import Scaler, fit_standard, one_hot, transform
from mlp.serialization import (
    FORMAT_VERSION,
    TrainingConfig,
    load_model,
    model_from_dict,
    model_to_dict,
    save_model,
)
from mlp.types import FloatArray, IntArray

FEATURES = 4
LABELS = ["B", "M"]
TRAINING: TrainingConfig = {"epochs": 5, "batch_size": 8, "seed": 0,
                            "early_stopping": None}


def raw_data(
    samples: int = 80,
    seed: int = 0,
) -> tuple[FloatArray, IntArray]:
    """Return unscaled features and 1-D integer labels."""
    rng = np.random.default_rng(seed)
    x = rng.normal(5.0, 3.0, size=(samples, FEATURES))
    y = (x[:, 0] + x[:, 1] > 10.0).astype(np.int64)
    return x, y


def trained_softmax() -> tuple[Model, Scaler]:
    """Return a softmax model trained on scaled data, and its scaler."""
    x, y = raw_data()
    scaler = fit_standard(x)
    model = Model([
        DenseLayer(6, "relu", "heUniform"),
        DenseLayer(5, "relu", "heUniform"),
        DenseLayer(2, "softmax", "glorotUniform"),
    ])
    model.compile("categoricalCrossentropy", metrics=["accuracy", "f1"],
                  learning_rate=0.05)
    model.fit(transform(scaler, x), one_hot(y, 2), epochs=5, batch_size=8,
              seed=0, verbose=False)
    return model, scaler


def trained_sigmoid() -> tuple[Model, Scaler]:
    """Return a sigmoid model trained on scaled data, and its scaler."""
    x, y = raw_data()
    scaler = fit_standard(x)
    model = Model([DenseLayer(3, "leaky_relu"), DenseLayer(1, "sigmoid")])
    model.compile("binaryCrossentropy", learning_rate=0.05)
    model.fit(transform(scaler, x), y.reshape(-1, 1).astype(np.float64),
              epochs=3, batch_size=8, seed=0, verbose=False)
    return model, scaler


@pytest.mark.parametrize("trained", [trained_softmax, trained_sigmoid])
def test_round_trip_gives_identical_evaluation(
    tmp_path: Path,
    trained: Callable[[], tuple[Model, Scaler]],
) -> None:
    """Fit, save, load and evaluate: exactly the same numbers.

    The raw validation data goes through the *stored* scaler, as the
    prediction program will do it.
    """
    model, scaler = trained()
    path = tmp_path / "model.json"
    save_model(path, model, scaler, LABELS, TRAINING)
    loaded = load_model(path)

    x_valid, y_valid = raw_data(30, seed=1)
    output_units = model.layers[-1].get_config()["units"]
    y = (one_hot(y_valid, 2) if output_units == 2
         else y_valid.reshape(-1, 1).astype(np.float64))
    expected = model.evaluate(transform(scaler, x_valid), y)
    actual = loaded.model.evaluate(transform(loaded.scaler, x_valid), y)
    assert actual == expected

    assert loaded.labels == LABELS
    assert loaded.training == TRAINING
    assert loaded.history == model.history
    assert loaded.model.get_compile_config() == model.get_compile_config()
    assert [lay.get_config() for lay in loaded.model.layers] == \
        [lay.get_config() for lay in model.layers]


def test_loaded_model_can_resume_training(tmp_path: Path) -> None:
    """The loaded model is built and compiled: fit works right away."""
    model, scaler = trained_softmax()
    path = tmp_path / "model.json"
    save_model(path, model, scaler, LABELS, TRAINING)
    loaded = load_model(path).model
    x, y = raw_data()
    loaded.fit(transform(scaler, x), one_hot(y, 2), epochs=1, batch_size=8,
               verbose=False)


def test_file_is_plain_json_with_expected_sections(tmp_path: Path) -> None:
    """The layout documented in PLAN.md is what lands on disk."""
    model, scaler = trained_softmax()
    path = tmp_path / "model.json"
    save_model(path, model, scaler, LABELS, TRAINING)
    content = json.loads(path.read_text())
    assert content["format_version"] == FORMAT_VERSION
    assert content["input_size"] == FEATURES
    assert content["architecture"][-1] == {
        "type": "dense", "units": 2, "activation": "softmax",
        "initializer": "glorotUniform"}
    assert content["compile"]["optimizer"] == {"name": "sgd",
                                               "learning_rate": 0.05}
    assert content["preprocessing"]["labels"] == LABELS
    assert content["preprocessing"]["scaler"]["kind"] == "standard"
    assert set(content["weights"][0]) == {"W", "b"}
    assert content["training"]["history"]["train"]["loss"]


def test_save_requires_a_trained_model(tmp_path: Path) -> None:
    """An unbuilt model has no weights to save."""
    model = Model([DenseLayer(2, "softmax")])
    model.compile("categoricalCrossentropy")
    with pytest.raises(NotBuiltError):
        save_model(tmp_path / "m.json", model, fit_standard(np.ones((2, 4))),
                   LABELS, TRAINING)


def test_save_rejects_mismatched_scaler_or_labels(tmp_path: Path) -> None:
    """The scaler width and the labels must match the model."""
    model, scaler = trained_softmax()
    with pytest.raises(ShapeError, match="features"):
        model_to_dict(model, fit_standard(np.ones((2, 3))), LABELS, TRAINING)
    with pytest.raises(ShapeError, match="labels"):
        model_to_dict(model, scaler, ["B", "M", "X"], TRAINING)


def test_save_to_unwritable_path(tmp_path: Path) -> None:
    """An OSError on write becomes a ModelFileError."""
    model, scaler = trained_softmax()
    with pytest.raises(ModelFileError, match="cannot write"):
        save_model(tmp_path / "absent" / "m.json", model, scaler, LABELS,
                   TRAINING)


# Broken files --------------------------------------------------------------


def valid_dict() -> dict[str, Any]:
    """Return the dict of a trained model, through JSON like on disk."""
    model, scaler = trained_softmax()
    content: dict[str, Any] = json.loads(
        json.dumps(model_to_dict(model, scaler, LABELS, TRAINING)))
    return content


def test_missing_file(tmp_path: Path) -> None:
    """A missing file is a model file error, with the cause kept."""
    with pytest.raises(ModelFileError, match="cannot read") as info:
        load_model(tmp_path / "absent.json")
    assert isinstance(info.value.__cause__, OSError)


def test_truncated_file(tmp_path: Path) -> None:
    """A truncated model.json is invalid JSON."""
    model, scaler = trained_softmax()
    path = tmp_path / "model.json"
    save_model(path, model, scaler, LABELS, TRAINING)
    text = path.read_text()
    path.write_text(text[:len(text) // 2])
    with pytest.raises(ModelFileError, match="cannot read") as info:
        load_model(path)
    assert isinstance(info.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize("version", [None, 0, 3, "1", "2", True])
def test_unknown_format_version(version: object) -> None:
    """Only the supported format versions are understood."""
    data = valid_dict()
    data["format_version"] = version
    with pytest.raises(ModelFileError, match="format_version"):
        model_from_dict(data)


def test_not_an_object() -> None:
    """The top level of the file must be a JSON object."""
    with pytest.raises(ModelFileError, match="JSON object"):
        model_from_dict([1, 2, 3])


def break_weight_columns(data: dict[str, Any]) -> None:
    """Drop a column of the weights of the second layer."""
    data["weights"][1]["W"] = [r[:-1] for r in data["weights"][1]["W"]]


def break_weight_rows(data: dict[str, Any]) -> None:
    """Drop a row of the weights of the last layer."""
    data["weights"][-1]["W"] = data["weights"][-1]["W"][:-1]


def break_bias(data: dict[str, Any]) -> None:
    """Give the first layer a bias of the wrong length."""
    data["weights"][0]["b"] = [[0.0]]


def break_input_size(data: dict[str, Any]) -> None:
    """Declare more features than the first layer takes."""
    data["input_size"] += 1


def drop_a_layer_of_weights(data: dict[str, Any]) -> None:
    """Keep the architecture but lose the weights of one layer."""
    data["weights"].pop()


def break_scaler_width(data: dict[str, Any]) -> None:
    """Store a scaler over fewer features than the model takes."""
    scaler = data["preprocessing"]["scaler"]
    scaler["a"], scaler["b"] = scaler["a"][:-1], scaler["b"][:-1]


@pytest.mark.parametrize(
    ("breaker", "match"),
    [
        (break_weight_columns, "layer 1: expected weights with 6 columns"),
        (break_weight_rows, "layer 2"),
        (break_bias, "layer 0.*bias"),
        (break_input_size, "layer 0: expected weights with 5 columns"),
        (drop_a_layer_of_weights, "weight entries"),
        (break_scaler_width, "scaler"),
    ],
)
def test_inconsistent_shapes(
    breaker: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    """Every inconsistent shape is refused with a precise message."""
    data = valid_dict()
    breaker(data)
    with pytest.raises(ModelFileError, match=match):
        model_from_dict(data)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("architecture", 0, "activation"), "gelu", "activation"),
        (("architecture", 0, "type"), "conv", "layer type"),
        (("architecture", 0, "units"), "6", "'units'"),
        (("compile", "loss"), "mse", "loss"),
        (("compile", "optimizer", "learning_rate"), -1.0, "learning rate"),
        (("preprocessing", "labels"), ["B"], "class names"),
        (("training", "epochs"), 0, "epochs"),
        (("weights", 0, "b"), [[float("nan")] * 6], "non finite"),
    ],
)
def test_invalid_values(
    path: tuple[str | int, ...],
    value: object,
    match: str,
) -> None:
    """Unknown names and invalid values are model file errors."""
    data = valid_dict()
    target: Any = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ModelFileError, match=match):
        model_from_dict(data)


@pytest.mark.parametrize(
    "path",
    [("weights",), ("compile",), ("preprocessing", "scaler"),
     ("training", "history"), ("architecture", 1, "units")],
)
def test_missing_keys(path: tuple[str | int, ...]) -> None:
    """A missing key is named in the message."""
    data = valid_dict()
    target: Any = data
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    with pytest.raises(ModelFileError):
        model_from_dict(data)
