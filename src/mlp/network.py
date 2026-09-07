from mlp.layers import DenseLayer
from mlp.losses import BinaryCrossentropy, CategoricalCrossentropy
from mlp.activations import Sigmoid, Softmax
from mlp.metrics import accuracy_score_ , precision_score_, recall_score_, f1_score_
import numpy as np
import json
import matplotlib.pyplot as plt


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
            self.metric_functions = {
                'accuracy': accuracy_score_,
                'precision': precision_score_,
                'recall': recall_score_,
                'f1': f1_score_
            }
            self.train_historique = {'loss': []}
            self.valid_historique = {'loss': []}

    def createWeigts(self, X):

        input_size = X.shape[1]

        if len(self.layers) == 0:
            raise RuntimeError("Cannot create the weigths if the model is empty")
        if (self.layers[0].weights.size == 0):
            self.layers[0].init_weightBias(input_size)
        # Nombre de poids=(Nombre de neurones dans la couche d'entre`e+1)
        # x
        # Nombre de neurones dans le premier layer cache
        for i in range(1, len(self.layers)):
            self.layers[i].init_weightBias(self.layers[i - 1].units)

    def printWeights(self):
        for layer in self.layers:
            print(layer)
            print("Weights")
            print(layer.weights)
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


    def fit_(self, x, y, epochs=100, learning_rate=0.001,  batch_size=128, validation_X=None, validation_Y=None):

        self.epochs = epochs
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

        for ep in range(epochs):
            # --- Forward Pass (Training) ---
            y_pred = self.foward(x)
            loss = self.loss.forward(y, y_pred)

            # --- Backward Pass ---
            dA = self.backward_loss(y, y_pred)
            self.backward(dA, from_loss=self.opti)     
            self.update(learning_rate)


            # --- Calcul des métriques de training ---
            train_metrics = {}
            classified_pred_train = (y_pred >= 0.5).astype(int)

            self.train_historique['loss'].append(loss)

            for metric_name, metric_func in self.metric_functions.items():
                if metric_name in self.metrics:
                    metric_value = metric_func(y, classified_pred_train)
                    train_metrics[metric_name] = metric_value
                    self.train_historique[metric_name].append(metric_value)


             # --- Calcul des métriques de validation ---
            valide_metrics = {}
            if validation_X is not None:
                y_pred_valide = self.foward(validation_X)
                classified_pred_valide = (y_pred_valide >= 0.5).astype(int)

                loss_valide = self.loss.forward(validation_Y, y_pred_valide)
                self.valid_historique['loss'].append(loss_valide)
                
                for metric_name, metric_func in self.metric_functions.items():
                    if metric_name in self.metrics:
                        metric_value = metric_func(validation_Y, classified_pred_valide)
                        valide_metrics[metric_name] = metric_value
                        self.valid_historique[metric_name].append(metric_value)

            self._print_metrics(ep, train_metrics, valide_metrics)


    def _print_metrics(self, epoch, metrics_train, metrics_validation=None):

        string_MT = [f"train_{key}: {value:.4f}, " for key, value in metrics_train.items()]
        if string_MT:
            string_MT[-1] = string_MT[-1][:-2]

        final_string = f"epochs:{epoch} >> " + "".join(string_MT)
        if metrics_validation is not None:
            string_MV = [f"valid_{key}: {value:.4f}, " for key, value in metrics_validation.items()]
            if string_MV:
                string_MV[-1] = string_MV[-1][:-2]
            final_string = final_string + " | " + "".join(string_MV)
        print(final_string)


    def compile(self, loss, metrics=['accuracy']):
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

        valid_metrics = set(self.metric_functions.keys())
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Metric '{metric}' is not supported. "
                    f"Valid metrics are: {valid_metrics}."
                )
            self.train_historique[metric] = []
            self.valid_historique[metric] = []
        self.metrics = metrics


    def save_weigts_bias(self):
        models_WeightsBias = {}
        for layer, i in zip(self.layers, range(len(self.layers))):
            layerWB = {"wheights": layer.weights.tolist(),
                       "bias": layer.bias.tolist()
                       }
            models_WeightsBias["layers" + str(i)] = layerWB
        try:
            file = "weights.json"
            with open(file, "w") as file:
                json.dump(models_WeightsBias, file, indent=2)
        except IOError as e:
            print(f"Error: cannot write in file: {e}")

    def load_weights_bias(self, path: str):
        try:
            with open(path, 'r') as file:
                dic_wb = {layer: wb for layer, wb in json.load(file).items()}
            for layer, i in zip(self.layers, range(len(self.layers))):
                layer.weights = np.array(dic_wb["layers" + str(i)]["wheights"])
                layer.bias = np.array(dic_wb["layers" + str(i)]["bias"])

                x, y = self.layers[i].weights.shape

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

    def plot_loss(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

        metric_colors = {
            'loss': 'blue',
            'accuracy': 'green',
            'precision': 'red',
            'recall': 'purple',
            'f1': 'orange'
        }

        train_style = '-'
        val_style = '--'

        epochs = range(1, 1 + len((self.train_historique['loss'])))
        # subplot 1: Loss
        ax1.plot(epochs, self.train_historique['loss'], label='Train Loss',
                 color='royalblue', linestyle=train_style)
        ax1.set_title('Training Loss')
        if self.valid_historique['loss']:
            ax1.plot(epochs, self.valid_historique['loss'], label='Validation Loss',
                     color=metric_colors['loss'], linestyle=val_style)
            ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # subplot 2: metrics
        for metric in self.metrics:
            if metric in self.train_historique and metric in self.valid_historique:
                ax2.plot(
                    epochs,
                    self.train_historique[metric],
                    label=f"Train {metric.capitalize()}",
                    color=metric_colors[metric],
                    linestyle=train_style
                )

                if len(self.valid_historique[metric]) >= 1:
                    ax2.plot(
                        epochs,
                        self.valid_historique[metric],
                        label=f"Validation {metric.capitalize()}",
                        color=metric_colors[metric],
                        linestyle=val_style
                    )

        ax2.set_title('Training metrics')
        if self.valid_historique['loss']:
            ax2.set_title('Training and Validation metrics')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    

# ml = Model(
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
# ml.load_weight_bias("weights.json")
# ml.summary()
# ml.printWeight()

# print(ml.len())
# last = ml.pop()
# print(last)
