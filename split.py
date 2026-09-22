#!/usr/bin/env python3
"""Split the dataset into a training and a validation file.

First of the three programs of the subject. The rows are shuffled with a
seeded generator, so a given seed always gives the same split, and are
written back unchanged: ``data_train.csv`` and ``data_valid.csv`` have
the same layout as the original file.
"""

import argparse
import sys
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from mlp.data import read_dataset, split_dataset, write_dataset
from mlp.errors import MLPError

PROG: Final[str] = "split.py"
TRAIN_FILE: Final[str] = "data_train.csv"
VALID_FILE: Final[str] = "data_valid.csv"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line, `argv` defaulting to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Split a dataset into a training and a validation set.",
    )
    parser.add_argument("--dataset", default="data.csv",
                        help="the CSV file to split (default: %(default)s)")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="fraction of the rows kept for training "
                             "(default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed of the shuffle (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=Path("."),
                        help=f"where {TRAIN_FILE} and {VALID_FILE} are "
                             "written (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the program and return its exit status."""
    args = parse_args(argv)
    try:
        df = read_dataset(args.dataset)
        train, valid = split_dataset(
            len(df), args.ratio, np.random.default_rng(args.seed))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, indices in ((TRAIN_FILE, train), (VALID_FILE, valid)):
            path = args.out_dir / name
            write_dataset(df.iloc[indices], path)
            print(f"{path}: {len(indices)} rows")
    except (MLPError, OSError) as e:
        print(f"{PROG}: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
