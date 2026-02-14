from layers import DenseLayer
import numpy as np
import json


class Model:
    def __init__(self, layers=[]):
            if not isinstance(layers, list):
                raise TypeError("Invalide type, layers must be a list")
            if not all(isinstance(layer, DenseLayer) for layer in layers):
                raise TypeError("Invalide type, elements in layers must be a DenseLayer")
            self.layers = layers
    
    def createWeigts(self):
        if len(self.layers) == 0:
            raise RuntimeError("Cannot create the weigths if the model is empty")
        if (self.layers[0].weight.size == 0):
            self.layers[0].init_weightBias(1)
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
            shapeW = layer.weight.shape
            total_weight += layer.weight.size
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

    def backward(self, loss_grad):
        for layer in self.layers:
            loss_grad = layer.backward(loss_grad)
    
    def update(self, alpha):
        for layer in self.layers:
            layer.update(alpha)

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



ml = Model([
    # DenseLayer(10, 'ReLU'),
    DenseLayer(10, 'ReLU'),
    # DenseLayer(5, 'ReLU'),
    DenseLayer(50, 'ReLU'),

    DenseLayer(5, 'Sigmoid'),
])

ml.add(DenseLayer(2, 'ReLU'))

# ml.createWeigts()
# ml.printWeight()
# ml.summary()
# ml.save_weigts_bias()
ml.load_weight_bias("weight.json")
ml.summary()
ml.printWeight()

# print(ml.len())
# last = ml.pop()
# print(last)
