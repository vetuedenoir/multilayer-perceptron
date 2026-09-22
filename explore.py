#!/usr/bin/env python3
"""Explore the dataset before training anything.

Not one of the three required programs: it serves the data understanding
phase the subject asks for. It prints the summary statistics, the rate
of missing values, the balance of the classes and the features the most
correlated with the diagnosis, and with ``--plot`` draws the histogram
of every feature for each class.

Pure pandas and matplotlib, on purpose: nothing here is imported from
the ``mlp`` library, which only ever sees numpy arrays.
"""

import argparse
import sys
from typing import Final, Sequence

import matplotlib.pyplot as plt
import pandas as pd

PROG: Final[str] = "explore.py"
LABEL: Final[str] = "diagnosis"
MEASURES: Final[tuple[str, ...]] = (
    "radius", "texture", "perimeter", "area", "smoothness", "compactness",
    "concavity", "concave_points", "symmetry", "fractal_dimension",
)
STATISTICS: Final[tuple[str, ...]] = ("mean", "se", "worst")
FEATURES: Final[list[str]] = [
    f"{measure}_{stat}" for stat in STATISTICS for measure in MEASURES]
COLUMNS: Final[list[str]] = ["id", LABEL, *FEATURES]
CLASS_COLORS: Final[dict[str, str]] = {"B": "tab:blue", "M": "tab:red"}
GRID_COLUMNS: Final[int] = len(MEASURES)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line, `argv` defaulting to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(
        prog=PROG, description="Describe the dataset.")
    parser.add_argument("--dataset", default="data.csv",
                        help="the CSV file to explore (default: %(default)s)")
    parser.add_argument("--top", type=int, default=10,
                        help="number of most correlated features shown "
                             "(default: %(default)s)")
    parser.add_argument("--plot", action="store_true",
                        help="draw the histograms of the features by class")
    return parser.parse_args(argv)


def load(path: str) -> pd.DataFrame:
    """Read the header-less CSV and name its columns.

    Raise ValueError when the number of columns is not the expected one.
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"expected {len(COLUMNS)} columns, received {df.shape[1]}")
    df.columns = pd.Index(COLUMNS)
    return df


def correlations(df: pd.DataFrame) -> "pd.Series[float]":
    """Return the correlation of every feature with the diagnosis.

    The diagnosis is encoded as 1 for malignant, 0 for benign; features
    are sorted by decreasing absolute correlation.
    """
    target = (df[LABEL] == "M").astype(float)
    corr = df[FEATURES].corrwith(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def report(df: pd.DataFrame, top: int) -> str:
    """Return the textual description of `df`."""
    missing = df.isna().mean()
    sections = [
        f"rows: {len(df)}, features: {len(FEATURES)}",
        "classes:\n" + df[LABEL].value_counts().to_string(),
        "missing values rate (non zero only):\n"
        + (missing[missing > 0].to_string() if missing.any() else "none"),
        "summary statistics:\n"
        + df[FEATURES].describe().T.to_string(float_format="{:.4g}".format),
        f"top {top} features correlated with the diagnosis:\n"
        + correlations(df).head(top).to_string(float_format="{:+.3f}".format),
    ]
    return "\n\n".join(sections)


def plot_histograms(df: pd.DataFrame) -> None:
    """Draw the histogram of every feature, one colour per class."""
    rows = len(FEATURES) // GRID_COLUMNS
    fig, axes = plt.subplots(rows, GRID_COLUMNS, figsize=(24, 8))
    for ax, feature in zip(axes.flat, FEATURES):
        for label, color in CLASS_COLORS.items():
            ax.hist(df.loc[df[LABEL] == label, feature], bins=20,
                    alpha=0.5, color=color, label=label)
        ax.set_title(feature, fontsize=8)
        ax.tick_params(labelsize=6)
    axes.flat[0].legend()
    fig.suptitle("Feature distributions by diagnosis")
    fig.tight_layout()
    plt.show()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the program and return its exit status."""
    args = parse_args(argv)
    try:
        df = load(args.dataset)
    # pandas' ParserError and EmptyDataError are ValueErrors too.
    except (OSError, ValueError) as e:
        print(f"{PROG}: error: cannot read {args.dataset!r}: {e}",
              file=sys.stderr)
        return 1
    print(report(df, args.top))
    if args.plot:
        plot_histograms(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
