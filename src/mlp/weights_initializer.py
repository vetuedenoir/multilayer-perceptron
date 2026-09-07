import numpy as np


def zero_initializer(units: int, input_size: int):
    return np.zeros((units, input_size))


def randomNormal_initializer(units: int, input_size: int):
    return np.random.normal(loc=0.0, scale=1.0, size=(units, input_size))


def randomUniform_initializer(units: int, input_size: int):
    return np.random.uniform(-1, 1, size=(units, input_size))

def heUniform_initializer(units: int, input_size: int):
    limit = np.sqrt(6 / input_size)
    return np.random.uniform(-limit, limit, size=(units, input_size))


def heNormal_initializer(units: int, input_size: int):
    sigma =  np.sqrt(2 / input_size)
    return np.random.normal(loc=0, scale=sigma, size=(units, input_size))


def glorotUniform_initializer(units: int, input_size: int):
    limit = np.sqrt(6 / (input_size + units))
    return np.random.uniform(-limit, limit, size=(units, input_size))


def glorotNormal_initializer(units: int, input_size: int):
    sigma = np.sqrt(2 / (input_size + units))
    return np.random.normal(loc=0, scale=sigma, size=(units, input_size))


WEIGHTS_INITIALIZERS = {
    "": randomNormal_initializer,
    "zero": zero_initializer,
    "randomNormal": randomNormal_initializer,
    "randomUniform": randomUniform_initializer,
    "heUniform": heUniform_initializer,
    "heNormal": heNormal_initializer,
    "glorotUniform": glorotUniform_initializer,
    "glorotNormal": glorotNormal_initializer
}