#!/usr/bin/env python3


import argparse
from mlp.serialization import load_model


def parse():
    parser = argparse.ArgumentParser(prog="predict.py")
    parser.add_argument("-m", "--model", default="model.json",
                        help="The model file written by train.py")
    return parser.parse_args()


def	main():
    args = parse()
    loaded = load_model(args.model)
    print(loaded.model.summary())


if __name__ == "__main__":
    main()
