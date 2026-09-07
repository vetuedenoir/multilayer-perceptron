import numpy as np
from mlp.activations import Sigmoid, ReLU, LeakyReLU, Softmax, ACTIVATIONS
from mlp.weights_initializer import WEIGHTS_INITIALIZERS
from mlp.get_from_registry import get_from_registry



class DenseLayer:
    def	__init__(self, units: int, activation: str, weights_initializer=''):
        if not isinstance(units, int) or units <= 0:
            raise ValueError("Received an Invalid value for 'units', "
            "expected a positive integer.")
        self.weights = np.empty((0)) # tableau numpy
        self.bias = np.empty((0)) # tableau numpy
        self.units = units

        self.activation = get_from_registry(ACTIVATIONS, activation, "activation")
        self.weights_initializer = get_from_registry(WEIGHTS_INITIALIZERS, weights_initializer, "weights_initializer")


    def init_weightBias(self, input_size: int):
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("Received an Invalid value for 'input_size', "
                    "expected a positive integer.")

        self.weights = self.weights_initializer(self.units, input_size)
        self.bias = np.zeros((1, self.units))
        

    def	forward(self, x):
        self.input = x
        self.z = self.input @ self.weights.T + self.bias
        self.a = self.activation.forward(self.z)
        return self.a
    
    def backward(self, dA, from_loss=False):
        if from_loss is True:
            dZ = dA
        else:    
            dZ = dA * self.activation.backward(self.z)
        m = self.input.shape[0]
        self.dW = (dZ.T @ self.input) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        dA_prev = dZ @ self.weights
        return dA_prev
    
    def update(self, alpha):
        # alpha c'est le learning rate.

        self.weights -= alpha * self.dW
        self.bias -= alpha * self.db
    
    def __str__(self):
        return f"units={self.units}, activation={self.activation}"