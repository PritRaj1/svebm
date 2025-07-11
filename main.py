import torch
from lightning.pytorch.cli import LightningCLI

from src.data import TextDataModule
from src.model import SVEBM


class LightningCLI_JIT(LightningCLI):
    def before_fit(self):
        compile_options = {
            "mode": "max-autotune",
            "backend": "inductor",
            "fullgraph": True,
            "dynamic": True,
        }
        
        try:
            self.model = torch.compile(self.model, **compile_options)
            print(f"✓ Model JIT-compiled successfully")
            print(f"   Mode: {compile_options['mode']}")
            print(f"   Backend: {compile_options['backend']}")
        except Exception as e:
            print(f"✗ Compilation failed, falling back to eager: {e}")
        
        super().before_fit()

if __name__ == "__main__":
    cli = LightningCLI_JIT(SVEBM, TextDataModule)
    cli.run()