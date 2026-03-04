from model.layers import DenseLayer
from model.losses import BinaryCrossentropy, CategoricalCrossentropy
from model.activations import Sigmoid, Softmax
from model.metrics import accuracy_score_ , precision_score_, recall_score_, f1_score_
import numpy as np
import json


class Model:
    def __init__(self, layers=[]):
            if not isinstance(layers, list):
                raise TypeError("Invalide type, layers must be a list")
            if not all(isinstance(layer, DenseLayer) for layer in layers):
                raise TypeError("Invalide type, elements in layers must be a DenseLayer")
            self.layers = layers
            self.loss = None
            self.backward_loss = None
            self.opti = False

    def createWeigts(self, X):

        input_size = X.shape[1]

        if len(self.layers) == 0:
            raise RuntimeError("Cannot create the weigths if the model is empty")
        if (self.layers[0].weights.size == 0):
            self.layers[0].init_weightBias(input_size)
        # Nombre de poids=(Nombre de neurones dans la couche d’entreˊe+1)
        # ×
        # Nombre de neurones dans le premier layer cache
        for i in range(1, len(self.layers)):
            self.layers[i].init_weightBias(self.layers[i - 1].units)

    def printWeight(self):
        for layer in self.layers:
            print(layer)
            print("Weights")
            print(layer.weight)
            print("Bias")
            print(layer.bias)
            print()

    def summary(self):
        total_weight = 0
        total_bias = 0
        for layer, i in zip(self.layers, range(len(self.layers))):
            shapeW = layer.weights.shape
            total_weight += layer.weights.size
            total_bias += layer.bias.size
            print(f"layers {i}, units={layer.units}, weights_shape={shapeW}")
        
        print(f"the number of weights in the model: {total_weight}")
        print(f"the number of bias in the model: {total_bias}")


    def add(self, layer):
        if not isinstance(layer, DenseLayer):
                raise TypeError("Invalide type, layer must be a DenseLayer")
        self.layers.append(layer)

    def pop(self, pos=None):
        if pos is None:
            return self.layers.pop()
        return self.layers.pop(pos)

    def len(self):
        return len(self.layers)

    def foward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, from_loss=False):
        loss_grad = self.layers[-1].backward(loss_grad, from_loss=from_loss)
        for i in range(len(self.layers) - 2, -1, -1):
            loss_grad = self.layers[i].backward(loss_grad)

    def update(self, alpha):
        for layer in self.layers:
            layer.update(alpha)

    def evaluate(self, validation_X, validation_Y):
        y_pred = self.foward(validation_X)
        loss = self.loss.forward(validation_Y, y_pred)


    def fit_(self, x, y, epoch=600, learning_rate=0.001,  batch_size=128, validation_X=None, validation_Y=None):
        
        if self.loss is None:
            raise RuntimeError("Cannot fit the model if the loss function is not defined, "
            "please use the compile method to define the loss function before fitting the model.")
        
        if self.loss == BinaryCrossentropy and y.shape[1] != 1:
            raise ValueError("Invalide shape for y, expected a shape of (m, 1) "
            "for BinaryCrossentropy loss.")
        if self.loss == BinaryCrossentropy and self.layers[-1].units != 1:
            raise ValueError("Invalide number of units in the last layer, " \
            "expected 1 for BinaryCrossentropy loss.")
        
        if self.loss == CategoricalCrossentropy and y.shape[1] <= 1:
            raise ValueError("Invalide shape for y, expected a shape of (m, n) with n > 1 "
            "for CategoricalCrossentropy loss.")
        if self.loss == CategoricalCrossentropy and self.layers[-1].units != y.shape[1]:
            raise ValueError("Invalide shape for y, expected a shape of (m, n) " \
            "with n equal to the number of units in the last layer for CategoricalCrossentropy loss.")

        for ep in range(epoch):
            y_pred = self.foward(x)

            loss = self.loss.forward(y, y_pred)
            
            dA = self.backward_loss(y, y_pred)
            # print("dA shape in mlp = ", dA.shape)
            self.backward(dA, from_loss=self.opti)
            
            self.update(learning_rate)


            accuracy = accuracy_score_(y,(y_pred >= 0.5).astype(int))
            if validation_X is not None:
                y_pred_valide = self.foward(validation_X)
                loss_valide = self.loss.forward(validation_Y, y_pred_valide)
                accuracy_valide = accuracy_score_(validation_Y, (y_pred_valide >= 0.5).astype(int))
                print(f"epoch:{ep} , loss: {loss:5f}, accuracy: {accuracy:5f} | "
                      f"loss_valid: {loss_valide:5f}, accuracy_valid: {accuracy_valide:5f}")
            else:
                print(f"epoch:{ep} , loss: {loss:5f}, accuracy: {accuracy:5f}")

            # accuracy = accuracy_score_(y, np.where(y_pred <= 0.5, 0, 1))
            # accuracy_valide = accuracy_score_(validation_Y, np.where(y_pred_valide <= 0.5, 0, 1))




    def compile(self, loss):
        if loss == "BinaryCrossentropy":
            self.loss = BinaryCrossentropy()
            if self.layers[-1].activation == Sigmoid:
                self.opti = True
                self.backward_loss = self.loss.backward_X_Sigmoid
            else:
                self.backward_loss = self.loss.backward
        elif loss == "CategoricalCrossentropy":
            self.loss = CategoricalCrossentropy()
            if self.layers[-1].activation == Softmax:  
                self.opti = True
                self.backward_loss = self.loss.backward_X_Softmax
            else:
                self.backward_loss = self.loss.backward
        else:
            raise NameError("Received an invalide name for 'loss', "
                    "expected an str equal to 'BinaryCrossentropy' or 'CategoricalCrossentropy'.")
        
        for i in range(len(self.layers)):
            self.layers[i].activation = self.layers[i].activation()



    def save_weigts_bias(self):
        models_WeightsBias = {}
        for layer, i in zip(self.layers, range(len(self.layers))):
            layerWB = {"wheights": layer.weight.tolist(),
                       "bias": layer.bias.tolist()
                       }
            models_WeightsBias["layers" + str(i)] = layerWB
        try:
            file = "weight.json"
            with open(file, "w") as file:
                json.dump(models_WeightsBias, file, indent=2)
        except IOError as e:
            print(f"Error: cannot write in file: {e}")

    def load_weight_bias(self, path: str):
        try:
            with open(path, 'r') as file:
                dic_wb = {layer: wb for layer, wb in json.load(file).items()}
            for layer, i in zip(self.layers, range(len(self.layers))):
                layer.weight = np.array(dic_wb["layers" + str(i)]["wheights"])
                layer.bias = np.array(dic_wb["layers" + str(i)]["bias"])

                x, y = self.layers[i].weight.shape

                if self.layers[i].units != x:
                    raise ValueError(f"The number of weights dont feat the number" \
                    f" of neurone in the {i} layers:" \
                    f" there is {self.layers[i].units} neurones"
                    f" and weights shape is (^{x}^, {y})"
                    )
                if i > 0 and self.layers[i - 1].units != y:
                    raise ValueError(f"The number of weights dont feat the number" \
                    f" of input in the {i} layers:" \
                    f" the input size is {self.layers[i - 1].units}"
                    f" and weights shape is ({x}, ^{y}^)"
                    )

        except IOError as e:
            print(f"Error: cannot read in file {e}")



# ml = Model([
    # DenseLayer(10, 'ReLU'),
    # DenseLayer(10, 'ReLU', input_size=30),
    # DenseLayer(5, 'ReLU'),
    # DenseLayer(5, 'ReLU'),

    # DenseLayer(5, 'Sigmoid'),
# ])

# ml.add(DenseLayer(2, 'ReLU'))

# ml.createWeigts()
# ml.printWeight()
# ml.summary()
# ml.save_weigts_bias()
# ml.load_weight_bias("weight.json")
# ml.summary()
# ml.printWeight()

# print(ml.len())
# last = ml.pop()
# print(last)
