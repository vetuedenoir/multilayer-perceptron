"""End-to-end tests of the programs: split, train, predict and explore.

Every ``main()`` is called with a hard-coded argv on a small synthetic
dataset. A success returns 0; a failure returns 1 with a one-line
message on stderr and no traceback.
"""

from pathlib import Path

import numpy as np
import pytest

import explore
import predict
import split
import train
from mlp.data import N_FEATURES
from mlp.errors import ShapeError
from mlp.serialization import load_model

N_ROWS = 60
FAST = ["--epochs", "3", "--layers", "8", "8"]


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """Write a separable dataset in the layout of data.csv."""
    rng = np.random.default_rng(0)
    lines = []
    for i in range(N_ROWS):
        label = "M" if i % 2 else "B"
        shift = 2.0 if label == "M" else 0.0
        features = rng.normal(shift, 1.0, N_FEATURES)
        lines.append(",".join([str(1000 + i), label]
                              + [f"{v:.6f}" for v in features]))
    path = tmp_path / "data.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def split_dir(tmp_path: Path, dataset: Path) -> Path:
    """Split `dataset` into tmp_path and return that directory."""
    assert split.main(["--dataset", str(dataset),
                       "--out-dir", str(tmp_path)]) == 0
    return tmp_path


def train_args(directory: Path, *extra: str) -> list[str]:
    """Return the argv of a quick training on the split of `directory`."""
    return ["--train", str(directory / "data_train.csv"),
            "--valid", str(directory / "data_valid.csv"),
            "--model", str(directory / "model.json"), *FAST, *extra]


def assert_clean_failure(
    status: int,
    capsys: pytest.CaptureFixture[str],
    program: str,
    match: str,
) -> None:
    """Check a failure: status 1 and a single error line on stderr."""
    err = capsys.readouterr().err
    assert status == 1
    assert err.startswith(f"{program}: error: ")
    assert match in err
    assert "Traceback" not in err


def read_ids(path: Path) -> list[str]:
    """Return the first column of a CSV file."""
    return [line.split(",")[0] for line in path.read_text().splitlines()]


# split.py


def test_split_writes_both_sets(split_dir: Path, dataset: Path) -> None:
    """The rows are shared out 80/20, none lost and none duplicated."""
    train_ids = read_ids(split_dir / "data_train.csv")
    valid_ids = read_ids(split_dir / "data_valid.csv")
    assert (len(train_ids), len(valid_ids)) == (48, 12)
    assert sorted(train_ids + valid_ids) == sorted(read_ids(dataset))


def test_split_is_reproducible(tmp_path: Path, dataset: Path) -> None:
    """The same seed gives the same files."""
    outputs = []
    for name in ("a", "b"):
        assert split.main(["--dataset", str(dataset), "--seed", "7",
                           "--out-dir", str(tmp_path / name)]) == 0
        outputs.append((tmp_path / name / "data_train.csv").read_text())
    assert outputs[0] == outputs[1]


def test_split_missing_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing file is reported, not raised."""
    status = split.main(["--dataset", str(tmp_path / "absent.csv")])
    assert_clean_failure(status, capsys, "split.py", "cannot read")


def test_split_invalid_ratio(
    tmp_path: Path,
    dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A ratio outside ]0, 1[ is a configuration error."""
    status = split.main(["--dataset", str(dataset), "--ratio", "1.5",
                         "--out-dir", str(tmp_path)])
    assert_clean_failure(status, capsys, "split.py", "ratio")


# train.py and predict.py


@pytest.mark.parametrize("extra", [[], ["--output-activation", "sigmoid"]])
def test_train_then_predict(
    split_dir: Path,
    extra: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The BCE of predict.py equals the last validation loss of train.py.

    For a sigmoid output it is the same loss. For a two-unit softmax,
    the categorical cross-entropy over (1 - p, p) is exactly the binary
    cross-entropy over p.
    """
    model_path = split_dir / "model.json"
    assert train.main(train_args(split_dir, *extra)) == 0
    history = load_model(model_path).history
    capsys.readouterr()

    assert predict.main(["--model", str(model_path),
                         "--dataset", str(split_dir / "data_valid.csv")]) == 0
    out = capsys.readouterr().out
    bce_line = next(line for line in out.splitlines()
                    if line.startswith("binary cross-entropy:"))
    bce = float(bce_line.split(":")[1])
    assert bce == pytest.approx(history.valid["loss"][-1], abs=1e-4)
    for name in ("accuracy", "precision", "recall", "f1"):
        assert f"{name}: " in out


def test_train_saves_the_architecture(split_dir: Path) -> None:
    """Hidden layers from --layers, softmax over 2 units by default."""
    assert train.main(train_args(split_dir)) == 0
    model = load_model(split_dir / "model.json").model
    configs = [layer.get_config() for layer in model.layers]
    assert [c["units"] for c in configs] == [8, 8, 2]
    assert [c["activation"] for c in configs] == ["relu", "relu", "softmax"]


def test_train_with_another_optimizer(split_dir: Path) -> None:
    """--optimizer is saved with its default hyperparameters."""
    assert train.main(train_args(split_dir, "--optimizer", "adam",
                                 "--learning-rate", "0.001")) == 0
    config = load_model(split_dir / "model.json").model.get_compile_config()
    assert config["optimizer"] == {
        "name": "adam", "learning_rate": 0.001,
        "hyperparameters": {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}}


def test_train_unknown_optimizer(
    split_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unknown optimizer lists the valid ones."""
    status = train.main(train_args(split_dir, "--optimizer", "adagrad"))
    assert_clean_failure(status, capsys, "train.py", "Available values")


def test_train_zero_units(
    split_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A layer of 0 units is refused."""
    status = train.main(train_args(split_dir, "--layers", "0"))
    assert_clean_failure(status, capsys, "train.py", "units")


def test_train_unknown_activation(
    split_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unknown activation lists the valid ones."""
    status = train.main(train_args(split_dir, "--activation", "gelu"))
    assert_clean_failure(status, capsys, "train.py", "Available values")


def test_train_missing_training_set(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing training set is reported, not raised."""
    status = train.main(train_args(tmp_path))
    assert_clean_failure(status, capsys, "train.py", "cannot read")


def test_train_early_stopping_then_predict(
    split_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The saved weights are those of the best epoch, not the last one."""
    model_path = split_dir / "model.json"
    assert train.main(train_args(split_dir, "--epochs", "40",
                                 "--early-stopping", "--patience", "3")) == 0
    out = capsys.readouterr().out
    assert "early stopping" in out.splitlines()[-2]
    loaded = load_model(model_path)
    best_epoch = loaded.history.best_epoch
    assert best_epoch is not None
    assert loaded.training["early_stopping"] is not None
    assert loaded.training["early_stopping"]["patience"] == 3

    assert predict.main(["--model", str(model_path),
                         "--dataset", str(split_dir / "data_valid.csv")]) == 0
    bce_line = next(line for line in capsys.readouterr().out.splitlines()
                    if line.startswith("binary cross-entropy:"))
    assert float(bce_line.split(":")[1]) == pytest.approx(
        loaded.history.valid["loss"][best_epoch - 1], abs=1e-4)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--patience", "0"], "patience"),
        (["--min-delta", "-1"], "min_delta"),
        (["--monitor", "valid_auc"], "'valid_auc', expected one of"),
        (["--monitor", "loss"], "monitored value"),
    ],
)
def test_train_invalid_early_stopping(
    split_dir: Path,
    capsys: pytest.CaptureFixture[str],
    extra: list[str],
    match: str,
) -> None:
    """An invalid early stopping option is a clean failure."""
    status = train.main(train_args(split_dir, "--early-stopping", *extra))
    assert_clean_failure(status, capsys, "train.py", match)


def test_train_ignores_early_stopping_options_when_off(
    split_dir: Path,
) -> None:
    """Without --early-stopping, the other options are not even read."""
    assert train.main(train_args(split_dir, "--patience", "0")) == 0
    loaded = load_model(split_dir / "model.json")
    assert loaded.training["early_stopping"] is None
    assert loaded.history.best_epoch is None


def test_predict_truncated_model(
    split_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A truncated model file is an invalid JSON."""
    assert train.main(train_args(split_dir)) == 0
    model_path = split_dir / "model.json"
    model_path.write_text(model_path.read_text()[:100])
    capsys.readouterr()
    status = predict.main(["--model", str(model_path),
                           "--dataset", str(split_dir / "data_valid.csv")])
    assert_clean_failure(status, capsys, "predict.py", "cannot read")


def test_predict_missing_model(
    tmp_path: Path,
    dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing model file is reported, not raised."""
    status = predict.main(["--model", str(tmp_path / "absent.json"),
                           "--dataset", str(dataset)])
    assert_clean_failure(status, capsys, "predict.py", "cannot read")


def test_positive_probability() -> None:
    """The last column is kept; more than two classes are refused."""
    proba = np.array([[0.2, 0.8], [0.9, 0.1]])
    np.testing.assert_array_equal(predict.positive_probability(proba),
                                  [[0.8], [0.1]])
    with pytest.raises(ShapeError):
        predict.positive_probability(np.full((2, 3), 1.0 / 3.0))


# explore.py


def test_explore_report(
    dataset: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The report covers the classes and the correlations."""
    assert explore.main(["--dataset", str(dataset), "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "classes:" in out
    assert "top 3 features correlated with the diagnosis" in out


def test_explore_missing_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing file is reported, not raised."""
    status = explore.main(["--dataset", str(tmp_path / "absent.csv")])
    assert_clean_failure(status, capsys, "explore.py", "cannot read")
