import torch.nn as nn


def get_activation(name: str):
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "softsign": nn.Softsign,
        "selu": nn.SELU,
        "celu": nn.CELU,
    }

    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")

    return activations[name]()
