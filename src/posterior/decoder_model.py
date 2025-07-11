from typing import Any, cast

import lightning as L
import torch
import torch.nn as nn

from src.utils import get_activation


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

        # Input projection
        layers: list[Any] = [nn.Linear(self.input_dim, self.hidden_layers[0])]
        layers.append(get_activation(self.activation))

        for hidden_dim in self.hidden_layers[1:]:
            layers.append(
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=self.nhead,
                    dim_feedforward=hidden_dim,
                    dropout=self.dropout,
                    activation=self.activation,
                )
            )

        # Output projection
        layers.append(nn.Linear(self.hidden_layers[-1], self.output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        output: (batch, seq_len, output_dim)
        """
        return cast(torch.Tensor, self.model(x))
