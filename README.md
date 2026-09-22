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

## Vérifications

```bash
uv sync --group dev
uv run mypy
uv run flake8 src tests split.py train.py predict.py explore.py
uv run pytest
```
