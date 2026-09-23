# Multilayer Perceptron

Réseau de neurones multicouche implémenté from scratch en NumPy, entraîné à classer les
tumeurs du dataset Wisconsin Breast Cancer (`M` maligne / `B` bénigne).
Architecture du code : voir `STRUCTURE.md`.

## Utilisation

```bash
uv sync
uv run python split.py --dataset data.csv --ratio 0.8 --seed 42
uv run python train.py --train data_train.csv --valid data_valid.csv --plot
uv run python predict.py --model model.json --dataset data_valid.csv
```

`train.py` affiche la loss et les métriques (train et validation) à chaque epoch, puis
écrit `model.json` : architecture, poids, scaler, labels et historique. `predict.py`
reconstruit le réseau depuis ce fichier et affiche la binary cross-entropy et les
métriques. `uv run python train.py --help` liste les hyperparamètres.

## Résultats : softmax/CCE contre sigmoid/BCE

Même split (`split.py --seed 42`, 455 lignes de train / 114 de validation), même
architecture et mêmes hyperparamètres : deux couches cachées ReLU de 24 unités,
`heUniform`, SGD, 84 epochs, `batch_size=8`, `learning_rate=0.0314`, seed 14.
Seule la sortie change.

| Sortie | Loss d'entraînement | Epoch | train loss | valid loss | valid accuracy | precision | recall | f1 |
|---|---|---|---|---|---|---|---|---|
| softmax (2 unités) | categorical CE | 1  | 0.1506 | 0.1457 | 0.9474 | 0.9286 | 0.9286 | 0.9286 |
| softmax (2 unités) | categorical CE | 84 | 0.0031 | 0.1067 | **0.9737** | 0.9535 | 0.9762 | 0.9647 |
| sigmoid (1 unité)  | binary CE      | 1  | 0.1841 | 0.1694 | 0.9561 | 1.0000 | 0.8810 | 0.9367 |
| sigmoid (1 unité)  | binary CE      | 84 | 0.0033 | 0.1277 | **0.9737** | 0.9535 | 0.9762 | 0.9647 |

`predict.py` sur `data_valid.csv` redonne exactement la dernière `valid loss` de chaque
modèle comme binary cross-entropy (0.1067 et 0.1277) : sur deux classes, la BCE calculée
sur la colonne `M` de la softmax est égale à sa categorical CE.

Lecture :

- Les deux sorties convergent vers les **mêmes prédictions** sur la validation (mêmes
  accuracy, precision, recall et f1) et des courbes de même allure. C'est attendu : une
  softmax sur 2 unités est une sigmoid appliquée à la différence des deux logits.
- La train loss tend vers 0 alors que la valid loss remonte après son minimum
  (0.0619 à l'epoch 11 en softmax, 0.0410 à l'epoch 15 en sigmoid) : le modèle
  sur-apprend. L'accuracy de validation a culminé plus tôt (0.9912 et 1.0000) avant de
  redescendre à 0.9737 ; un early stopping (bonus) en tirerait parti.

Le chemin softmax + CCE (celui du sujet) et cette équivalence sont vérifiés par
`tests/test_softmax_path.py`.

## Résultats : comparaison des optimiseurs (bonus)

Même split, même réseau (24-24 ReLU, softmax(2) + CCE, `heUniform`, `batch_size=8`,
seed 14), hyperparamètres par défaut de chaque optimiseur (`momentum=0.9` ; `rho=0.9` ;
`beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`), avec
`--epochs 300 --early-stopping --patience 15` : les chiffres sont ceux de la **meilleure
epoch** (poids restaurés), l'arrêt tombant 15 epochs plus tard.

```bash
uv run python train.py --optimizer adam --learning-rate 0.001 --epochs 300 --early-stopping --patience 15
```

| Optimiseur | learning rate | Meilleure epoch | Arrêt | train loss | valid loss | valid accuracy | precision | recall | f1 |
|---|---|---|---|---|---|---|---|---|---|
| sgd      | 0.0314 | 11 | 26 | 0.0463 | 0.0619 | **0.9825** | 0.9762 | 0.9762 | 0.9762 |
| momentum | 0.01   | 5  | 20 | 0.0373 | **0.0598** | **0.9825** | 0.9762 | 0.9762 | 0.9762 |
| nesterov | 0.01   | 5  | 20 | 0.0376 | **0.0598** | **0.9825** | 0.9762 | 0.9762 | 0.9762 |
| rmsprop  | 0.001  | 6  | 21 | 0.0585 | 0.0611 | 0.9737 | 0.9535 | 0.9762 | 0.9647 |
| adam     | 0.001  | 7  | 22 | 0.0529 | 0.0746 | 0.9737 | 0.9535 | 0.9762 | 0.9647 |

Lecture :

- Tous atteignent au moins 0.9737 d'accuracy de validation. Momentum et Nesterov trouvent
  leur meilleure epoch deux fois plus tôt que SGD (5 contre 11), avec la plus basse valid
  loss : la vitesse accumulée multiplie le pas effectif par ~`1 / (1 - 0.9) = 10`.
- Nesterov et momentum classique sont quasi confondus : sur un problème aussi simple, le
  « coup d'œil en avant » ne change presque rien.
- RMSprop et Adam, à `lr=0.001`, convergent aussi vite (meilleure epoch 6 et 7) mais
  s'arrêtent sur une valid loss un peu plus haute et une accuracy de 0.9737 : un exemple
  de validation de plus est mal classé.
- Le dataset (455 lignes, 30 features) est trop petit pour que les optimiseurs adaptatifs
  fassent la différence ; l'intérêt est surtout de pouvoir les brancher sans toucher aux
  couches (`optimizers.py`, `Model.update()`).

Les hyperparamètres propres à chaque optimiseur ne sont pas des options CLI : ils prennent
les valeurs de la littérature, et se règlent dans un fichier d'architecture (clé
`optimizer.hyperparameters`, voir ci-dessous).

## Fichier d'architecture (bonus)

Sans option, `train.py` entraîne le réseau par défaut ; `--layers`, `--activation`,
`--initializer`, `--output-activation`, `--loss`, `--optimizer` et `--learning-rate` le
modifient. `--arch-file PATH` décrit **tout** le réseau dans un fichier JSON, y compris les
hyperparamètres de l'optimiseur ; il ne se combine pas avec ces options (erreur d'usage).

```bash
uv run python train.py --arch-file architectures/deep_adam.json --epochs 300 --early-stopping
```

```json
{
  "hidden": [
    {"units": 16, "activation": "leaky_relu", "initializer": "heUniform"},
    {"units": 8, "activation": "leaky_relu", "initializer": "heUniform"}
  ],
  "output": {"activation": "softmax", "initializer": "heUniform"},
  "loss": "categoricalCrossentropy",
  "optimizer": {"name": "nesterov", "learning_rate": 0.01,
                "hyperparameters": {"momentum": 0.9}}
}
```

Le nombre d'unités de sortie découle de la loss ; `loss` peut être omise (même règle que
`--loss`) et `hyperparameters` aussi (valeurs par défaut). Les noms et les valeurs sont
vérifiés par les couches, l'optimiseur et `compile()`. Exemples dans `architectures/` :

| Fichier | Couches cachées | Sortie | Optimiseur |
|---|---|---|---|
| `default.json` | 24, 24 relu | softmax(2) + CCE | sgd 0.0314 |
| `subject.json` | 24, 24, 24 sigmoid (exemple du sujet) | softmax(2) + CCE | sgd 0.0314 |
| `sigmoid_output.json` | 24, 24 relu | sigmoid(1) + BCE | sgd 0.0314 |
| `deep_adam.json` | 32, 32, 16 relu | softmax(2) + CCE | adam 0.001 |
| `small_nesterov.json` | 16, 8 leaky_relu | softmax(2) + CCE | nesterov 0.01, momentum 0.9 |

## Vérifications

```bash
uv sync --group dev
uv run mypy
uv run flake8 src tests split.py train.py predict.py explore.py
uv run pytest
```
