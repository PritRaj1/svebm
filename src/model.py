import torch
import torch.nn as nn
import lightning as L

from src.prior.ebm_model import ebm_model
from src.posterior.encoder_model import encoder_model

class SVEBM(L.LightningModule):
    def __init__(
            self, 
            ebm_model,
            gen_model,
            **kwargs
    ):
        super().__init__()
        self.ebm = ebm_model
        self.enc = encoder_model
        self.dec = decoder_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)