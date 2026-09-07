import numpy as np


class Softmax:
    
    def forward(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

# Softmax n'est pas une activation independante par neurone.
# Softmax depend de tout les neurones de la couche
# le calcule de la deriver requiere une matrice jacobienne et necessiterais
# le gradient entrant( devivee de la fonction de cout par rapport
# a la sortie de softmax)

# Heureusement nous utilisons la fonction de cout CrossEntropy
# donc la derive de z sera juste y_pred - y_true

# Pour rappel z c'est la sortie du neurone avant la fonction d'activation.

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
        return np.where(z <= 0, 0.01, 1)


ACTIVATIONS = {
    "Sigmoid": Sigmoid,
    "ReLU": ReLU,
    "LeakyReLU": LeakyReLU,
    "Softmax": Softmax
}