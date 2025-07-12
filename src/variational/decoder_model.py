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


class DecoderModel(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list = [128, 128],
        nhead: int = 4,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nhead = nhead
        self.dropout = dropout
        self.activation = activation
        self.hidden_layers = hidden_layers

        # Build decoder layers (reverse of encoder)
        layers: list[Any] = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_layers):
            if i == 0:
                # Input projection from latent space
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(get_activation(activation))
                prev_dim = hidden_dim
            else:
                # Transformer decoder layer
                layers.append(
                    nn.Sequential(
                        PermuteLayer((1, 0, 2)),  # (B, S, D) -> (S, B, D)
                        nn.TransformerDecoderLayer(
                            d_model=prev_dim,
                            nhead=self.nhead,
                            dim_feedforward=hidden_dim,
                            dropout=self.dropout,
                            activation=self.activation,
                        ),
                        PermuteLayer((1, 0, 2)),  # (S, B, D) -> (B, S, D)
                    )
                )
                # Project to next dimension
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(get_activation(activation))
                prev_dim = hidden_dim

        # Final projection to output space
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations back to data space.

        Args:
            x: Latent representations (batch, seq_len, input_dim)

        Returns:
            Reconstructed data (batch, seq_len, output_dim)
        """
        return cast(torch.Tensor, self.model(x))
