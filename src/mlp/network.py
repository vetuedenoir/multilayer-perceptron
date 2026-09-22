
from mlp.layers import DenseLayer
from mlp.losses import LOSSES, resolve_output_grad
from mlp.metrics import METRICS
from mlp.registry import get_from_registry
import numpy as np
import json
import matplotlib.pyplot as plt


def iter_batches(X, y, batch_size, shuffle=True):
    """Yield (X_batch, y_batch) pairs covering the whole dataset once.

    The last batch is smaller when the dataset size is not a multiple of
    batch_size. Shuffling is done per call, so each epoch sees a different
    partition of the samples.
    """
    m = X.shape[0]
    indices = np.random.permutation(m) if shuffle else np.arange(m)

    for start in range(0, m, batch_size):
        batch = indices[start:start + batch_size]
        yield X[batch], y[batch]


class Model:
    def __init__(self, layers=None):
            if layers is None:
                layers = []
            if not isinstance(layers, list) and not isinstance(layers, tuple):
                raise TypeError("Invalid type, layers must be a list")
            if not all(isinstance(layer, DenseLayer) for layer in layers):
                raise TypeError("Invalid type, elements in layers must be a DenseLayer")
            # copied: a caller keeping a reference to its list must not be able
            # to mutate the model through it
            self.layers = list(layers)
            self.loss = None
            self.backward_loss = None
            self.opti = False
            self.metric_functions = METRICS
            self.train_history = {'loss': []}
            self.valid_history = {'loss': []}

    def createWeigts(self, X):

        input_size = X.shape[1]

        if len(self.layers) == 0:
            raise RuntimeError("Cannot create the weigths if the model is empty")
        if self.layers[0].weights.size == 0:
            self.layers[0].init_weightBias(input_size)
        # Nombre de poids=(Nombre de neurones dans la couche d'entre`e+1)
        # x
        # Nombre de neurones dans le premier layer cache
        for i in range(1, len(self.layers)):
            if self.layers[i].weights.size == 0:
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
                raise TypeError("Invalid type, layer must be a DenseLayer")
        self.layers.append(layer)

    def pop(self, pos=None):
        if pos is None:
            return self.layers.pop()
        return self.layers.pop(pos)

    def len(self):
        return len(self.layers)

    def forward(self, x):
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

    def predict_classes(self, y_pred):
        """Turn network outputs into 1D class labels.

        One output column is a binary probability, thresholded at 0.5. Several
        columns are a probability distribution, so the class is the argmax.
        The metrics work on these labels, never on the raw output matrix.
        """
        if y_pred.shape[1] == 1:
            return (y_pred >= 0.5).astype(int).ravel()
        return np.argmax(y_pred, axis=1)

    def evaluate(self, X, y):
        """Run a forward pass and return (loss, {metric_name: value})."""
        y_pred = self.forward(X)
        loss = self.loss.forward(y, y_pred)

        classified_pred = self.predict_classes(y_pred)
        classified_true = self.predict_classes(y)
        metrics = {}
        for metric_name in self.metrics:
            metrics[metric_name] = self.metric_functions[metric_name](classified_true,
                                                                      classified_pred)

        return loss, metrics


    def fit_(self, x, y, epochs=100, learning_rate=0.001,  batch_size=128, validation_X=None, validation_Y=None):

        self.createWeigts(x)
        self.epochs = epochs
        if self.loss is None:
            raise RuntimeError("Cannot fit the model if the loss function is not defined, "
            "please use the compile method to define the loss function before fitting the model.")

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Received an Invalid value for 'batch_size', "
            "expected a positive integer.")

        is_binary = self.loss.name == "binaryCrossentropy"
        is_categorical = self.loss.name == "categoricalCrossentropy"

        if is_binary and y.shape[1] != 1:
            raise ValueError("Invalid shape for y, expected a shape of (m, 1) "
            "for BinaryCrossentropy loss.")
        if is_binary and self.layers[-1].units != 1:
            raise ValueError("Invalid number of units in the last layer, " \
            "expected 1 for BinaryCrossentropy loss.")

        if is_categorical and y.shape[1] <= 1:
            raise ValueError("Invalid shape for y, expected a shape of (m, n) with n > 1 "
            "for CategoricalCrossentropy loss.")
        if is_categorical and self.layers[-1].units != y.shape[1]:
            raise ValueError("Invalid shape for y, expected a shape of (m, n) " \
            "with n equal to the number of units in the last layer for CategoricalCrossentropy loss.")


        for ep in range(epochs):
            for x_batch, y_batch in iter_batches(x, y, batch_size):
                y_pred = self.forward(x_batch)

                dA = self.backward_loss(y_batch, y_pred)
                self.backward(dA, from_loss=self.opti)
                self.update(learning_rate)


            loss, train_metrics = self.evaluate(x, y)
            self.train_history['loss'].append(loss)
            for metric_name, metric_value in train_metrics.items():
                self.train_history[metric_name].append(metric_value)

            valide_metrics = None
            if validation_X is not None:
                loss_valide, valide_metrics = self.evaluate(validation_X, validation_Y)
                self.valid_history['loss'].append(loss_valide)
                for metric_name, metric_value in valide_metrics.items():
                    self.valid_history[metric_name].append(metric_value)
                valide_metrics = {'loss': loss_valide, **valide_metrics}

            self._print_metrics(ep, {'loss': loss, **train_metrics}, valide_metrics)


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

        last_activation = self.layers[-1].activation.name

        self.loss = get_from_registry(LOSSES, loss, "loss")

        if last_activation == "softmax" and self.loss.name == "binaryCrossentropy":
            raise ValueError("Invalid combination of loss function and activation function in the last layer, "
            "Softmax activation is not compatible with BinaryCrossentropy loss.")

        if last_activation == "softmax" and self.layers[-1].units == 1:
            raise ValueError("Invalid number of units in the last layer, "
            "Softmax needs at least 2 units (it outputs a probability distribution "
            "over the units of the layer).")

        self.backward_loss, self.opti = resolve_output_grad(self.loss,
                                                            last_activation)

        valid_metrics = set(self.metric_functions.keys())
        for metric in metrics:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Metric '{metric}' is not supported. "
                    f"Valid metrics are: {valid_metrics}."
                )
            self.train_history[metric] = []
            self.valid_history[metric] = []
        self.metrics = metrics

        #   self.metric_functions = get_from_registry(METRICS, metrics, "metrics")
    
        #     for metric in metrics:
        #         self.train_history[metric] = []
        #         self.valid_history[metric] = []
        #     self.metrics = metrics
    


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

        epochs = range(1, 1 + len((self.train_history['loss'])))
        # subplot 1: Loss
        ax1.plot(epochs, self.train_history['loss'], label='Train Loss',
                 color='royalblue', linestyle=train_style)
        ax1.set_title('Training Loss')
        if self.valid_history['loss']:
            ax1.plot(epochs, self.valid_history['loss'], label='Validation Loss',
                     color=metric_colors['loss'], linestyle=val_style)
            ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # subplot 2: metrics
        for metric in self.metrics:
            if metric in self.train_history and metric in self.valid_history:
                ax2.plot(
                    epochs,
                    self.train_history[metric],
                    label=f"Train {metric.capitalize()}",
                    color=metric_colors[metric],
                    linestyle=train_style
                )

                if len(self.valid_history[metric]) >= 1:
                    ax2.plot(
                        epochs,
                        self.valid_history[metric],
                        label=f"Validation {metric.capitalize()}",
                        color=metric_colors[metric],
                        linestyle=val_style
                    )

        ax2.set_title('Training metrics')
        if self.valid_history['loss']:
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
