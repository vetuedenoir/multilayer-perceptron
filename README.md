# Multilayer Perceptron

Bibliothèque de réseaux de neurones multicouches écrite from scratch en NumPy : couches
denses, activations, losses, initialisations, optimiseurs, early stopping, sauvegarde et
rechargement de modèles. Elle s'utilise en Python (package `mlp`) ou via trois programmes
en ligne de commande qui couvrent le cycle complet : découper un dataset, entraîner,
prédire.

## Installation

```bash
uv sync                 # crée .venv et installe le package mlp en éditable
uv sync --group dev     # + mypy, flake8, pytest
```

## Ce que propose la bibliothèque

| Brique | Choix disponibles |
|---|---|
| Couche | `DenseLayer(units, activation, weights_initializer)` |
| Activations | `sigmoid`, `relu`, `leaky_relu`, `softmax` |
| Initialisations | `zeros`, `random_normal`, `random_uniform`, `he_uniform`, `he_normal`, `glorot_uniform`, `glorot_normal` |
| Losses | `binaryCrossentropy`, `categoricalCrossentropy` |
| Optimiseurs | `sgd`, `momentum`, `nesterov`, `rmsprop`, `adam` |
| Métriques | `accuracy`, `precision`, `recall`, `f1` |
| Prétraitement | scalers `standard` et `minmax`, `one_hot` |
| Entraînement | mini-batch, validation à chaque epoch, `EarlyStopping` avec restauration des meilleurs poids |
| Persistance | `save_model` / `load_model` : architecture, poids, scaler, labels et historique dans un JSON |

Tous les composants se choisissent par leur nom ; les variantes CamelCase (`ReLU`,
`heUniform`, `RMSProp`, `CategoricalCrossentropy`…) sont acceptées. Un nom inconnu lève
une `ConfigurationError` qui liste les valeurs valides.

## Utilisation en Python

```python
from mlp import Adam, DenseLayer, EarlyStopping, Model, one_hot, transform
from mlp import load_model, save_model
from mlp.data import LABELS, read_dataset, to_arrays
from mlp.preprocessing import fit_scaler

# Données : features (m, n) et labels entiers (m,)
x_train, labels_train = to_arrays(read_dataset("data_train.csv"))
x_valid, labels_valid = to_arrays(read_dataset("data_valid.csv"))

# Normalisation ajustée sur le train seul
scaler = fit_scaler("standard", x_train)
x_train, x_valid = transform(scaler, x_train), transform(scaler, x_valid)
y_train, y_valid = one_hot(labels_train, 2), one_hot(labels_valid, 2)

# Réseau
model = Model()
model.add(DenseLayer(32, "relu", "he_uniform"))
model.add(DenseLayer(16, "leaky_relu", "he_uniform"))
model.add(DenseLayer(2, "softmax", "glorot_uniform"))
model.compile(loss="categoricalCrossentropy",
              optimizer=Adam(learning_rate=0.001),   # ou "adam", learning_rate=0.001
              metrics=("accuracy", "f1"))
model.build(x_train.shape[1], seed=14)
print(model.summary())

# Entraînement
history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                    epochs=200, batch_size=8, seed=14,
                    early_stopping=EarlyStopping(monitor="valid_loss", patience=15))

loss, metrics = model.evaluate(x_valid, y_valid)
probas = model.predict_proba(x_valid)      # sortie brute du réseau
classes = model.predict_classes(x_valid)   # labels entiers

# Sauvegarde puis rechargement : le réseau est reconstruit depuis le fichier
save_model("model.json", model, scaler, LABELS,
           {"epochs": 200, "batch_size": 8, "seed": 14, "early_stopping": None})
loaded = load_model("model.json")
x_new, _ = to_arrays(read_dataset("data_valid.csv"))
predictions = loaded.model.predict_classes(transform(loaded.scaler, x_new))
print([loaded.labels[i] for i in predictions[:5]])   # ['B', 'B', 'B', 'B', 'M']
```

Points utiles :

- Une sortie `sigmoid` à 1 unité s'utilise avec `binaryCrossentropy` et des cibles
  `(m, 1)` ; une sortie `softmax` avec `categoricalCrossentropy` et des cibles one-hot.
- `compile` accepte un nom d'optimiseur (état neuf) ou une instance déjà configurée
  (`Momentum(0.01, momentum=0.9, nesterov=True)`, `RMSprop(0.001, rho=0.9)`,
  `Adam(0.001, beta1=0.9, beta2=0.999)`…).
- À seed égale, l'entraînement est reproductible (poids initiaux et mélange des batchs).
- `fit` renvoie un `History` (courbes train/valid, meilleure epoch) que
  `mlp.plotting.plot_history` trace.
- Les erreurs héritent toutes de `MLPError` (`ShapeError`, `NotBuiltError`,
  `TrainingDivergedError`, `ModelFileError`…).

## Utilisation en ligne de commande

```bash
uv run python split.py   --dataset data.csv --ratio 0.8 --seed 42
uv run python train.py   --train data_train.csv --valid data_valid.csv --plot
uv run python predict.py --model model.json --dataset data_valid.csv
```

- `split.py` mélange le dataset et l'écrit en `data_train.csv` / `data_valid.csv`.
- `train.py` construit le réseau, affiche la loss et les métriques (train et validation)
  à chaque epoch, écrit `model.json` et, avec `--plot`, trace les courbes
  d'apprentissage.
- `predict.py` recharge `model.json` (aucune couche à redéclarer), applique le scaler
  sauvegardé et affiche la binary cross-entropy et les métriques.

Le réseau se décrit directement en options :

```bash
uv run python train.py --layers 32 16 --activation leaky_relu --initializer he_normal \
                       --optimizer adam --learning-rate 0.001 \
                       --epochs 300 --early-stopping --patience 15
```

| Option | Défaut |
|---|---|
| `--layers UNITS...` | `64 32` |
| `--activation` / `--initializer` | `sigmoid` / `glorot_uniform` |
| `--output-activation` / `--loss` | `softmax` / déduite de la sortie |
| `--optimizer` / `--learning-rate` | `rmsprop` / `0.0001` |
| `--epochs` / `--batch-size` / `--seed` | `100` / `8` / `14` |
| `--early-stopping` `--patience` `--min-delta` `--monitor` `--no-restore-best` | désactivé, `10`, `0.0`, `valid_loss` |

`uv run python train.py --help` détaille chaque option.

## Décrire un réseau dans un fichier JSON

`--arch-file PATH` lit tout le réseau depuis un fichier : couches cachées, sortie, loss,
optimiseur et ses hyperparamètres (`momentum`, `rho`, `beta1`, `beta2`, `epsilon`…). Il
remplace les options d'architecture ci-dessus.

```bash
uv run python train.py --arch-file architectures/small_nesterov.json --epochs 300 --early-stopping
```

```json
{
  "hidden": [
    {"units": 8, "activation": "sigmoid", "initializer": "heUniform"},
    {"units": 4, "activation": "sigmoid", "initializer": "heUniform"}
  ],
  "output": {"activation": "softmax", "initializer": "heUniform"},
  "loss": "categoricalCrossentropy",
  "optimizer": {"name": "nesterov", "learning_rate": 0.005,
                "hyperparameters": {"momentum": 0.9}}
}
```

Le nombre d'unités de sortie découle de la loss. `loss` peut être omise (déduite de
l'activation de sortie), `hyperparameters` aussi (valeurs par défaut).

### Exemples d'architectures

Le dossier `architectures/` rassemble des réseaux prêts à l'emploi, qui montrent chacun
une façon de combiner les briques :

| Fichier | Ce qu'il montre |
|---|---|
| `default.json` | le réseau par défaut : 64-32 `sigmoid`, `glorot_uniform`, sortie `softmax`, `rmsprop` |
| `subject.json` | trois couches `sigmoid` de 24 unités |
| `sigmoid_output.json` | sortie `sigmoid` à 1 unité avec `binaryCrossentropy` |
| `deep_adam.json` | réseau plus profond (32-32-16) avec `adam` et ses hyperparamètres explicites |
| `deep_leaky_rmsprop_binary.json` | quatre couches `leaky_relu` (64-32-16-8), `rmsprop`, sortie binaire |
| `small_nesterov.json` | petit réseau 8-4 `sigmoid` avec `nesterov` |
| `single_layer_random_normal_sgd.json` | une seule couche cachée, loss omise, `hyperparameters` vide |
| `sigmoid_glorot_momentum.json` | couches `sigmoid` initialisées en `glorot_uniform`, `momentum` |
| `random_uniform_momentum_nesterov_flag.json` | `momentum` avec `"nesterov": true`, initialisation `random_uniform` |
| `relu_he_normal_rmsprop.json` | `he_normal`, `rmsprop` avec `rho` et `epsilon`, `"loss": null` |
| `leaky_glorot_normal_adam_binary.json` | `leaky_relu` + `glorot_normal`, `adam`, sortie binaire |
| `mixed_layers_nesterov.json` | une activation et une initialisation différentes par couche |
| `camelcase_aliases.json` | noms en CamelCase : `ReLU`, `LeakyReLU`, `Softmax`, `RMSProp` |

## Dataset de démonstration

Le dépôt fournit `data.csv`, le dataset Wisconsin Breast Cancer, pour essayer la
bibliothèque sur un vrai problème : classer des tumeurs en maligne (`M`) ou bénigne
(`B`) à partir de 30 mesures extraites d'images de cellules. 569 lignes, sans en-tête :
un identifiant, le diagnostic, puis les 30 features.

Avec le split par défaut (455 lignes d'entraînement, 114 de validation), le réseau par
défaut classe correctement **les 114 tumeurs de validation** :

```bash
uv run python split.py
uv run python train.py --plot
uv run python predict.py
```

```
samples: 114
binary cross-entropy: 0.0314
accuracy: 1.0000
precision: 1.0000
recall: 1.0000
f1: 1.0000
```

Le résultat ne dépend pas d'une initialisation chanceuse : sur ce split, les seeds 0 à 19
atteignent toutes 100 % d'accuracy à la dernière epoch.

`explore.py` donne un premier aperçu du dataset : statistiques, équilibre des classes,
features les plus corrélées au diagnostic et, avec `--plot`, l'histogramme de chaque
feature par classe.

## Vérifications

```bash
uv sync --group dev
uv run mypy
uv run flake8 src tests split.py train.py predict.py explore.py
uv run pytest
```
