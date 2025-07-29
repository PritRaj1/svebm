from typing import Any, cast

import lightning as L
import torch
import torch.nn as nn

from src.utils import get_activation

class LatentDistribution(L.LightningModule):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar
        
        
    

    
        