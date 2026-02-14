import numpy as np


class Sigmoid:
    def forward(self, z):
        return 1 / (1 + np.exp(-z))
    
    def backward(self, z):
        s = self.forward(z)
        return s * (1 - s)


class ReLU:
    def forward(self, z):
        return np.maximum(0, z)
    
    def backward(self, z):
        return np.where(z <= 0, 0, 1)


class LeakyReLU:
    def forward(self, z):
        return np.where(z <= 0, 0.01 * z, z)
    
    def backward(self, z):
        return np.where(z <= 0, 0.01, z)
