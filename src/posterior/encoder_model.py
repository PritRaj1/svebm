import torch
import torch.nn as nn
import lightning as L

from src.utils import get_activation

class encoder_model(L.LightningModule):
    def __init__(self, input_dim: int, latent_dim: int, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.nhead = kwargs.get("nhead", 4)
        self.dropout = kwargs.get("dropout", 0.1)
        self.activation = kwargs.get("activation", "relu")
        self.hidden_layers = kwargs.get("hidden_layers", [128, 128])

        # Input projection
        layers = [nn.Linear(self.input_dim, self.hidden_layers[0])]
        layers.append(get_activation(self.activation))

        for hidden_dim in self.hidden_layers[1:]:
            layers.append(nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=self.nhead,
                dim_feedforward=hidden_dim,
                dropout=self.dropout,
                activation=self.activation
            ))

        # Output projection 
        layers.append(nn.Linear(self.hidden_layers[-1], self.latent_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        output: (batch, seq_len, latent_dim)
        """
        x = self.model(x)
        return x
