from typing import Any, cast

import lightning as L
import torch
import torch.nn as nn

from src.utils import get_activation


class PermuteLayer(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class EncoderModel(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        hidden_layers: list = [128, 128],
        nhead: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.nhead = nhead
        self.dropout = dropout
        self.activation = activation
        self.hidden_layers = hidden_layers

        # Build layers
        layers: list[Any] = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):

            # Input projection
            if i == 0:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(get_activation(activation))
                prev_dim = hidden_dim

            # Transformer encoder layers
            else:
                layers.append(
                    nn.Sequential(
                        PermuteLayer((1, 0, 2)),
                        nn.TransformerEncoderLayer(
                            d_model=prev_dim,
                            nhead=self.nhead,
                            dim_feedforward=hidden_dim,
                            dropout=self.dropout,
                            activation=self.activation,
                        ),
                        PermuteLayer((1, 0, 2)),
                    )
                )
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(get_activation(activation))
                prev_dim = hidden_dim

        # Output projection to hidden state
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

        # Q(z | x) networks
        self.mu = nn.Linear(output_dim, latent_dim)
        self.logvar = nn.Linear(output_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_st = cast(torch.Tensor, self.model(x))
        z_mu = self.mu(hidden_st)
        z_logvar = self.logvar(hidden_st)
        return z_mu, z_logvar, hidden_st






