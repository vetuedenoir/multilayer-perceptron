#!/usr/bin/env python3


import argparse
from mlp.network import Model
from mlp.layers import DenseLayer


def parse():
    parser = argparse.ArgumentParser(prog="predict.py")
    parser.add_argument("-w", "--weights", help="The Weights of the model")
    return parser.parse_args()


def	main():
    args = parse()
    model = Model(([
        DenseLayer(40, activation='ReLU', weights_initializer="glorotUniform"),
        DenseLayer(120, activation='ReLU', weights_initializer="glorotUniform"),
        DenseLayer(80, activation='ReLU', weights_initializer="glorotUniform"),
        DenseLayer(1, activation='Sigmoid')
    ]))

    model.load_weights_bias(args.weights)
    model.printWeights()


if __name__ == "__main__":
    main()