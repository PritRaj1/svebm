from typing import Any, cast

import lightning as L
import torch
import torch.nn as nn

from src.utils import get_activation


class EBM_fcn(L.LightningModule):
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_layers: list = [128, 128],
        activation: str = "relu",
        eta: float = 1.0,
        N: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.eta = eta  # ULA step size
        self.N = N  # Number of ULA steps

        layers: list[Any] = []
        prev_dim = latent_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(get_activation(activation))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x))

    def ebm_prior(self, x: torch.Tensor, cls_output: bool = False, temperature: float = 1.0) -> torch.Tensor:
        assert len(x.size()) == 2, f"Expected 2D input, got shape {x.size()}"
        logits = self.forward(x)
        if cls_output:
            return logits
        return temperature * (logits / temperature).logsumexp(dim=1)
