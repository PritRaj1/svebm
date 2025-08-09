import os
import pytest

from lightning.pytorch.cli import LightningCLI


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
