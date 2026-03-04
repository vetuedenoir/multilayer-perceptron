import numpy as np
from model.activations import Sigmoid, ReLU, LeakyReLU, Softmax


class DenseLayer:
    def	__init__(self, units: int, activation: str, input_size=None):
        if not isinstance(units, int) or units <= 0:
            raise ValueError("Received an invalide value for 'units', "
            "expected a positive integer.")
        self.weights = np.empty((0)) # tableau numpy
        self.bias = np.empty((0)) # tableau numpy
        self.units = units
        if isinstance(input_size, int):
            self.weights = np.random.randn(self.units, input_size)
            # self.bias = np.random.randn(self.units, 1)
            self.bias = np.zeros((1, units))

        if activation == 'Sigmoid':
            self.activation = Sigmoid
        elif activation == 'ReLU':
            self.activation = ReLU
        elif activation == 'LeakyReLU':
            self.activation = LeakyReLU
        elif activation == 'Softmax':
            self.activation = Softmax
        else:
            raise NameError("Received an invalide name for 'activation', "
                    "expected an str equal to 'Sigmoid', 'ReLU', 'LeakyReLU'"
                    " or 'Softmax'.")


    def init_weightBias(self, input_size: int):
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("Received an invalide value for 'input_size', "
                    "expected a positive integer.")
        
        self.weights = np.random.randn(self.units, input_size).astype(np.float32)
        # self.weights = np.random.randn(input_size, self.units)

        self.bias = np.zeros((1, self.units))
        

    def	forward(self, x):
        self.input = x
        # self.z = self.weights * self.input  + self.bias
        self.z = self.input @ self.weights.T + self.bias
        self.a = self.activation.forward(self.z)
        return self.a
    
    def backward(self, dA, from_loss=False):
        # print("dA shape = ", dA.shape)
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
        # print("weights dtype:", self.weights.dtype)
        # print("dW dtype:", self.dW.dtype)

        # print(f"weights shape {self.weights.shape}, et self.dW shape = {self.dW.shape}")

        self.weights -= alpha * self.dW
        self.bias -= alpha * self.db
    
    def __str__(self):
        return f"units={self.units}, activation={self.activation}"