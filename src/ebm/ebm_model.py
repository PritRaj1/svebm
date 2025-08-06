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
        hidden_layers: list = [64, 32],
        activation: str = "relu",
        num_latent_samples: int = 20,
        num_gmm_components: int = 5,
        eta: float = 1.0,
        N: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_latent_samples = num_latent_samples
        self.num_gmm_components = num_gmm_components
        self.eta = eta
        self.N = N
        
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)
        
        # Gaussian mixture model
        mus = torch.randn(num_latent_samples, num_gmm_components, latent_dim)
        logvar = torch.randn(num_latent_samples, num_gmm_components, latent_dim)
        self.register_parameter('mix_mus', nn.Parameter(mus, requires_grad=True))
        self.register_parameter('mix_logvars', nn.Parameter(logvar, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.model(x))

    def ebm_prior(self, x: torch.Tensor, cls_output: bool = False, temperature: float = 1.0) -> torch.Tensor:
        assert len(x.size()) == 2, f"Expected 2D input, got shape {x.size()}"
        logits = self.forward(x)
        if cls_output:
            return logits
        return temperature * (logits / temperature).logsumexp(dim=1)
