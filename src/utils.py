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
    
def ids_to_text_list(ids: torch.Tensor, pad_id=None, bos_id=None, eos_id=None) -> list[str]:
    """Convert batch of token ids to whitespace-joined strings.
    
    Args:
        ids: Tensor of token IDs with shape [batch_size, seq_len]
        pad_id: ID of PAD token to skip (optional)
        bos_id: ID of BOS token to skip (optional) 
        eos_id: ID of EOS token to stop at (optional)
        
    Returns:
        List of text strings, one per batch item
    """
    ids = ids.detach().cpu()
    texts: list[str] = []
    
    for row in ids:
        tokens: list[str] = []
        for tid in row.tolist():
            if pad_id is not None and tid == pad_id:
                continue
            if bos_id is not None and tid == bos_id:
                continue
            if eos_id is not None and tid == eos_id:
                break
            tokens.append(str(tid))
        texts.append(' '.join(tokens))
    
    return texts