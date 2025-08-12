import os
import pytest
import torch
from torch.utils.data import Dataset

from lightning.pytorch.cli import LightningCLI


class MockTextDataset(Dataset):
    """Like IMDB."""

    def __init__(self, size=50):
        self.size = size
        self.texts = [
            f"This is sample text number {i}. "
            f"It contains some words for testing the model pipeline."
            for i in range(size)
        ]
        self.labels = [i % 2 for i in range(size)]  # One hot binary

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}


class TestModelCLI:

    @pytest.mark.parametrize("config_path", ["config/test_conf.yml"])
    def test_model_builds_from_config(self, config_path):
        assert os.path.exists(config_path), "Missing test config file"

        cli = LightningCLI(
            model_class=None,
            datamodule_class=None,
            save_config_callback=None,
            run=False,
            args=[f"--config={config_path}"],
        )

        assert cli.model is not None
        assert hasattr(cli.model, "enc") and cli.model.enc is not None
        assert hasattr(cli.model, "dec") and cli.model.dec is not None
        assert hasattr(cli.model, "ebm") and cli.model.ebm is not None

        assert hasattr(cli.model, "kl_annealer")
        assert cli.model.kl_annealer is not None

        assert cli.datamodule is not None

    def test_cli_with_mock_data(self):
        config_args = [
            "--config=config/test_conf.yml",
            "--data.dataset_cls=tests.test_model_cli.MockTextDataset",
            "--data.dataset_kwargs={size: 20}",
            "--data.batch_size=4",
            "--trainer.max_epochs=1",
            "--trainer.limit_train_batches=2",
            "--trainer.limit_val_batches=1",
        ]

        cli = LightningCLI(
            model_class=None,
            datamodule_class=None,
            save_config_callback=None,
            run=False,
            args=config_args,
        )

        cli.datamodule.setup(stage="fit")
        train_loader = cli.datamodule.train_dataloader()

        batch = next(iter(train_loader))

        assert isinstance(batch, dict)
        assert "encoder_inputs" in batch
        assert "inputs" in batch
        assert "targets" in batch
        assert "tgt_probs" in batch

        cli.model.train()
        outputs = cli.model(batch, mode="train")

        assert isinstance(outputs, dict)
        assert "logits" in outputs
        assert "z_mu" in outputs
        assert "z_logvar" in outputs
        assert "cd" in outputs
        assert "mi" in outputs

        loss = cli.model.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss).all()

        val_loss = cli.model.validation_step(batch, batch_idx=0)
        assert torch.isfinite(val_loss).all()

    def test_cli_generation_mode(self):
        cli = LightningCLI(
            model_class=None,
            datamodule_class=None,
            save_config_callback=None,
            run=False,
            args=[
                "--config=config/test_conf.yml",
                "--data.dataset_cls=tests.test_model_cli.MockTextDataset",
                "--data.dataset_kwargs={size: 8}",
                "--data.batch_size=2",
            ],
        )

        cli.datamodule.setup(stage="fit")
        train_loader = cli.datamodule.train_dataloader()
        batch = next(iter(train_loader))

        cli.model.train()
        gen_outputs = cli.model(batch, mode="test", gen_type="greedy")
        assert "logits" in gen_outputs

        beam_outputs = cli.model(batch, mode="test", gen_type="beam")
        assert "logits" in beam_outputs

        test_result = cli.model.test_step(batch, batch_idx=0)
        assert isinstance(test_result, dict)
