from typing import Literal, Union, cast

import torch
import torch.nn as nn
import math

ActivationType = Union[
    nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.Softplus, nn.Softsign, nn.SELU, nn.CELU
]
ActivationName = Literal[
    "relu", "gelu", "tanh", "sigmoid", "softplus", "softsign", "selu", "celu"
]


def get_activation(name: str) -> ActivationType:
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

    return cast(ActivationType, activations[name]())


def log_gaussian(
    z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    log_p = (
        -0.5 * torch.log(2 * math.pi) - logvar / 2 - (z - mu) ** 2 / (2 * logvar.exp())
    )
    return log_p.sum(dim=-1)
