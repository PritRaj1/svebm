import torch
import torch.nn as nn
import lightning as L

from src.utils import get_activation

class ebm_model(L.LightningModule):
    def __init__(self, latent_dim: int, decoder_dim: int, **kwargs):
        super().__init__()
        self.input_dim = latent_dim
        self.output_dim = decoder_dim
        self.activation = kwargs.get("activation", "relu")
        self.hidden_layers = kwargs.get("hidden_layers", [128, 128])

        layers = []
        input_dim = self.input_dim  

        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(get_activation(self.activation))
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, self.output_dim))  
        layers.append(get_activation(self.activation))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
