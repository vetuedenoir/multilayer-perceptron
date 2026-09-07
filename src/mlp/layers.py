import numpy as np
from mlp.activations import Sigmoid, ReLU, LeakyReLU, Softmax


class DenseLayer:
    def	__init__(self, units: int, activation: str, weights_initializer=''):
        if not isinstance(units, int) or units <= 0:
            raise ValueError("Received an Invalid value for 'units', "
            "expected a positive integer.")
        self.weights = np.empty((0)) # tableau numpy
        self.bias = np.empty((0)) # tableau numpy
        self.units = units

        if activation == 'Sigmoid':
            self.activation = Sigmoid
        elif activation == 'ReLU':
            self.activation = ReLU
        elif activation == 'LeakyReLU':
            self.activation = LeakyReLU
        elif activation == 'Softmax':
            self.activation = Softmax
        else:
            raise NameError("Received an Invalid name for 'activation', "
                    "expected an str equal to 'Sigmoid', 'ReLU', 'LeakyReLU'"
                    " or 'Softmax'.")
        
        if weights_initializer != '' and weights_initializer != 'zero' \
            and weights_initializer != 'randomNormal' \
            and weights_initializer != 'randomUniform' \
            and weights_initializer != 'heUniform' \
            and weights_initializer != 'heNormal' \
            and weights_initializer != 'glorotUniform' \
            and weights_initializer != 'glorotNormal':
            raise NameError("Received an Invalid name for 'weights_initializer', "
                    "expected an str equal to 'zero', 'randomNormal',"
                    "'randomUniform', 'heUniform', 'heNormal', "
                    "'glorotUniform' or 'glorotNormal ")
        self.weights_initializer = weights_initializer
        if weights_initializer == '':
            self.weights_initializer = 'randomNormal'
            


    def init_weightBias(self, input_size: int):
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("Received an Invalid value for 'input_size', "
                    "expected a positive integer.")

        match self.weights_initializer:
            case "zero":
                self.weights = np.zeros((self.units, input_size))
                # problem
            case "randomNormal":
                self.weights = np.random.normal(loc=0.0, scale=1.0, size=(self.units, input_size))
            case "randomUniform":
                self.weights = np.random.uniform(-1, 1, size=(self.units, input_size))
            case "heUniform":
                limit = np.sqrt(6 / input_size)
                self.weights = np.random.uniform(-limit, limit, size=(self.units, input_size))
            case "heNormal":
                sigma =  np.sqrt(2 / input_size)
                self.weights = np.random.normal(loc=0, scale=sigma, size=(self.units, input_size))
            case "glorotUniform":
                limit = np.sqrt(6 / (input_size + self.units))
                self.weights = np.random.uniform(-limit, limit, size=(self.units, input_size))
            case "glorotNormal":
                sigma = np.sqrt(2 / (input_size + self.units))
                self.weights = np.random.normal(loc=0, scale=sigma, size=(self.units, input_size))
            case _:
                self.weights = np.random.randn(self.units, input_size)

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