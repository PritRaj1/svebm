import torch
from lightning.pytorch.cli import LightningCLI

from src.data import TextDataModule
from src.model import SVEBM


class LightningCLI_JIT(LightningCLI):
    def before_fit(self):
        self.model = torch.compile(SVEBM)
        super().before_fit()

if __name__ == "__main__":
    cli = LightningCLI_JIT(SVEBM, TextDataModule)
    cli.run()