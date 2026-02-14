import numpy as np
from activation import Sigmoid, ReLU, LeakyReLU


class DenseLayer:
    def	__init__(self, units: int, activation: str, input_size=None):
        if not isinstance(units, int) or units <= 0:
            raise ValueError("Received an invalide value for 'units', "
            "expected a positive integer.")
        self.weight = np.empty((0)) # tableau numpy
        self.bias = np.empty((0)) # tableau numpy
        self.units = units
        if isinstance(input_size, int):
            self.weight = np.random.randn(self.units, input_size)
            self.bias = np.random.randn(self.units, 1)

        if activation == 'Sigmoid':
            self.activation = Sigmoid
        elif activation == 'ReLU':
            self.activation = ReLU
        elif activation == 'LeakyReLU':
            self.activation = LeakyReLU
        else:
            raise NameError("Received an invalide name for 'activation', "
                    "expected an str equal to 'Sigmoid', 'ReLU' or 'LeakyReLU'.")


    def init_weightBias(self, input_size: int):
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("Received an invalide value for 'input_size', "
                    "expected a positive integer.")
        
        self.weight = np.random.randn(self.units, input_size)
        self.bias = np.random.randn(self.units, 1)

        

    def	forward(self, x):
        self.input = x
        # self.z = "self.weight * self.input  + self.bias"
        self.z = self.weight * self.input + self.bias
        self.a = self.activation.forward(self.z)
        return self.a
    
    def backward(self, dA):
        # dA la derive de l'acitivation  des neuron du layer suivant
        dZ = self.activation.backward(self.z) # la derive de z
        self.dW = 3 # calculer la deriver de self.weight a l'aide de dz
        self.db = 3 # calculer la deriver de self.bias a l'aide de dz
        dA_prev = 3 # la deriver de notre layer pour le layers precedent
        return dA_prev
    
    def update(self, alpha):
        # alpha c'est le learning rate.

        self.weight -= alpha * self.dW
        self.bias -= alpha * self.db
    
    def __str__(self):
        return f"units={self.units}, activation={self.activation}"