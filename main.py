import torch
from lightning.pytorch.cli import LightningCLI

from src.data import TextDataModule
from src.model import SVEBM


class LightningCLI_JIT(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile", action="store_true", help="Compile the model with torch.compile")
        parser.add_argument("--compile_mode", type=str, default="max-autotune", help="torch.compile mode")
        parser.add_argument("--compile_backend", type=str, default="inductor", help="torch.compile backend")

    def before_fit(self):
        if self.config.get("compile", False):
            compile_options = {
                "mode": self.config.get("compile_mode", "max-autotune"),
                "backend": self.config.get("compile_backend", "inductor"),
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

if __name__ == "__main__":
    cli = LightningCLI_JIT(
        model_class=None,
        datamodule_class=None,
        save_config_callback=None,
        run=True
    )