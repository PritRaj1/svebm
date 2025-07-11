import lightning as L
import torch
import torch.nn as nn

from src.utils import get_activation


class EBMModel(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list = [128, 128],
        activation: str = "relu",
        eta: float = 1.0,
        N: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eta = eta  # ULA step size
        self.N = N  # Number of ULA steps

        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(get_activation(activation))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
