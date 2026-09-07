import numpy as np

# An activation exposes:
#   forward(z)      -> a, the activated output
#   backward(z, dA) -> dZ, the gradient of the loss with respect to z
#
# backward takes the incoming gradient dA instead of only returning the local
# derivative, because Softmax cannot be differentiated element-wise: each of its
# outputs depends on every neuron of the layer, so the chain rule goes through a
# Jacobian matrix. Passing dA in lets every activation return dZ directly.


class Softmax:

    def forward(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, z, dA):
        # Jacobian-vector product, computed without ever building the Jacobian:
        # dZ_i = a_i * (dA_i - sum_k(dA_k * a_k))
        a = self.forward(z)
        return a * (dA - np.sum(dA * a, axis=1, keepdims=True))


# When Softmax is the output layer and the loss is CategoricalCrossentropy, the
# derivative with respect to z simplifies to y_pred - y_true. That shortcut is
# handled by CategoricalCrossentropy.backward_X_Softmax and avoids the product
# above entirely. The backward here is what makes Softmax usable outside of that
# pairing (as a hidden layer, or with another loss).
#
# Reminder: z is the neuron output before the activation function.


class Sigmoid:
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, z, dA):
        s = self.forward(z)
        return dA * s * (1 - s)


class ReLU:
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z, dA):
        return dA * np.where(z <= 0, 0, 1)


class LeakyReLU:
    def forward(self, z):
        return np.where(z <= 0, 0.01 * z, z)

    def backward(self, z, dA):
        return dA * np.where(z <= 0, 0.01, 1)


ACTIVATIONS = {
    "Sigmoid": Sigmoid,
    "ReLU": ReLU,
    "LeakyReLU": LeakyReLU,
    "Softmax": Softmax
}
