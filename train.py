#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from model.mlp import Model
from model.layers import DenseLayer




def parse():
    parser = argparse.ArgumentParser(prog="training.py")
    parser.add_argument("dataset", help="the path to the dataset to train the model on")
    parser.add_argument("-p", "--plot", action = 'store_true', help= "plot the training metrics")
    return parser.parse_args()

def  load_dataset(dataset: str):
    data = pd.read_csv(dataset, header=None)
    if data is None:
        raise RuntimeError(f"Cannot open the file {dataset}")
    # print(data.head())
    # print(pd.DataFrame.describe(data))
    Y = data.iloc[:, 1].values

    X = data.iloc[:, 2:].values
    # print(X)

    # normalization min max

    maxs = np.max(X, axis=0)
    mins = np.min(X, axis=0)

    X_normal = (X - mins) / (maxs - mins)
    eps = 0.00001
    X_normal = np.clip(X_normal, 0 + eps, 1 - eps).astype(np.float32)

    # print(X_normal)


    for i in range(len(Y)):
        if Y[i] == 'M':
            Y[i] = 1.0
        else:
            Y[i] = 0.0
    # print(Y)
    Y = Y.reshape(-1, 1).astype(np.float32)
    print(Y.shape)
    

    return X_normal, Y






def normalize_dataset(dataset: str):
    print("norm")    

def train_model(X, Y):
    np.random.seed(14)
    indices = np.random.permutation(X.shape[0])

    X = X[indices]
    Y = Y[indices]

    split = int(0.9 * X.shape[0])

    X_train = X[:split]
    X_validation = X[split:]

    Y_train = Y[:split]
    Y_validation = Y[split:]
 
    print(f"X_train.shape = {X_train.shape}")
    print(f"y_train.shape = {Y_train.shape}")
    print(f"X_validation.shape = {X_validation.shape}")
    print(f"y_validation.shape = {Y_validation.shape}")



    model = Model([
        DenseLayer(30, activation='ReLU'),
        DenseLayer(40, activation='ReLU'),
        DenseLayer(80, activation='ReLU'),
        DenseLayer(20, activation='ReLU'),
        DenseLayer(1, activation='Sigmoid')
    ])

    model.createWeigts(X_train)
    model.compile("BinaryCrossentropy")

    model.summary()

    model.fit_(X_train, Y_train, validation_X=X_validation, validation_Y=Y_validation)



def	main():
    args = parse()
    X, Y = load_dataset(args.dataset)
    train_model(X, Y)


if __name__ == "__main__":
    main()