from typing import Optional

import lightning as L
import torch
from torchmetrics.text import BLEUScore

from src.criterion import LogProb
from src.variational.kl_annealing import KLAnnealer
from src.ebm.unadjusted_langevin import ula_prior
from src.utils import ids_to_text_list


class SVEBM(L.LightningModule):
    def __init__(
        self,
        ebm_model: L.LightningModule,
        encoder_model: L.LightningModule,
        decoder_model: L.LightningModule,
        loss_struct: LogProb,
        learning_rate: float = 1e-3,
        data_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        ebm_out_dim: Optional[int] = None,
        kl_annealer: KLAnnealer | None = None,
        ebm_learning_rate: float = 1e-4,
    ):
        super().__init__()

        self.ebm = ebm_model
        self.enc = encoder_model
        self.dec = decoder_model
        self.loss_struct = loss_struct
        self.learning_rate = learning_rate
        self.ebm_learning_rate = ebm_learning_rate
        self.kl_annealer = kl_annealer
        # Metrics
        self.bleu = BLEUScore(n_gram=4, smooth=True)
        # Default generation type for test/inference
        self.gen_type: str = "greedy"

        if latent_dim is None:
            if hasattr(self.enc, "latent_dim"):
                attr_value = getattr(self.enc, "latent_dim")
                if isinstance(attr_value, int):
                    latent_dim = attr_value
                else:
                    raise ValueError("latent_dim attribute must be an integer")
            else:
                raise ValueError(
                    "latent_dim must be provided or encoder must have "
                    "latent_dim attribute"
                )

        if ebm_out_dim is None:
            ebm_out_dim = latent_dim

        if data_dim is None:
            if hasattr(self.enc, "input_dim"):
                attr_value = getattr(self.enc, "input_dim")
                if isinstance(attr_value, int):
                    data_dim = attr_value
                else:
                    raise ValueError("input_dim attribute must be an integer")
            else:
                raise ValueError(
                    "data_dim must be provided or encoder must have "
                    "input_dim attribute"
                )

        assert isinstance(data_dim, int)
        assert isinstance(latent_dim, int)
        assert isinstance(ebm_out_dim, int)

        self._validate_dimensions(data_dim, latent_dim, ebm_out_dim)

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.ebm_out_dim = ebm_out_dim

        self.save_hyperparameters(
            ignore=["ebm_model", "encoder_model", "decoder_model"]
        )

    def _validate_dimensions(
        self,
        data_dim: int,
        latent_dim: int,
        ebm_out_dim: int,
    ) -> None:
        if hasattr(self.enc, "input_dim") and self.enc.input_dim != data_dim:
            raise ValueError(
                f"Encoder input_dim ({self.enc.input_dim}) != " f"data_dim ({data_dim})"
            )

        if hasattr(self.enc, "latent_dim") and self.enc.latent_dim != latent_dim:
            raise ValueError(
                f"Encoder latent_dim ({self.enc.latent_dim}) != "
                f"latent_dim ({latent_dim})"
            )

        if hasattr(self.ebm, "input_dim") and self.ebm.input_dim != latent_dim:
            raise ValueError(
                f"EBM input_dim ({self.ebm.input_dim}) != " f"latent_dim ({latent_dim})"
            )

        if hasattr(self.ebm, "output_dim") and self.ebm.output_dim != ebm_out_dim:
            raise ValueError(
                f"EBM output_dim ({self.ebm.output_dim}) != "
                f"ebm_out_dim ({ebm_out_dim})"
            )

        if hasattr(self.dec, "input_dim") and self.dec.input_dim != ebm_out_dim:
            raise ValueError(
                f"Decoder input_dim ({self.dec.input_dim}) != "
                f"ebm_out_dim ({ebm_out_dim})"
            )

        if hasattr(self.dec, "output_dim") and self.dec.output_dim != data_dim:
            raise ValueError(
                f"Decoder output_dim ({self.dec.output_dim}) != "
                f"data_dim ({data_dim})"
            )

    def get_kl_weight(self, step: int) -> float:
        if self.kl_annealer is not None:
            return self.kl_annealer.get_weight(step)
        return 1.0

    def forward(self, batch, mode="train", gen_type="greedy"):
        x = batch["encoder_inputs"] if isinstance(batch, dict) else batch
        z_mu, z_logvar, hidden = self.enc(x)
        z = self.loss_struct.reparameterize(z_mu, z_logvar)

        z_prior = ula_prior(self.ebm, torch.randn_like(z))

        cd = self.loss_struct.contrastive_loss(self.ebm, z_prior, z)
        mi = self.loss_struct.mutual_information(self.ebm, z)

        if mode in ("train", "val"):
            logits = self.dec(batch["inputs"], hidden, mode="TEACH_FORCE")
        else:
            logits = self.dec(None, hidden, mode="GENERATE", gen_type=gen_type)

        return {
            "logits": logits,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "z": z,
            "cd": cd,
            "mi": mi,
            "hidden": hidden,
        }

    def training_step(self, batch, batch_idx):
        kl_weight = self.get_kl_weight(self.global_step)
        outputs = self(batch, mode="train")

        nll = self.loss_struct.nll_entropy(outputs["logits"], batch["targets"])
        z_mu_in = outputs["z_mu"]
        z_logvar_in = outputs["z_logvar"]

        if z_mu_in.dim() == 2 and getattr(self.ebm, "num_latent_samples", 1) > 1:
            z_mu_in = (
                z_mu_in.unsqueeze(1)
                .expand(-1, self.ebm.num_latent_samples, -1)
                .contiguous()
                .view(z_mu_in.size(0), -1)
            )
            z_logvar_in = (
                z_logvar_in.unsqueeze(1)
                .expand(-1, self.ebm.num_latent_samples, -1)
                .contiguous()
                .view(z_logvar_in.size(0), -1)
            )

        kl = self.loss_struct.kl_div(
            self.ebm, batch["tgt_probs"], z_mu_in, z_logvar_in
        ).mean()

        total_loss = (
            nll
            + kl_weight * (kl - outputs["cd"])
            - self.loss_struct.mi_weight * outputs["mi"]
        )

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_nll", nll, on_step=True, on_epoch=True)
        self.log("train_kl", kl, on_step=True, on_epoch=True)
        self.log("train_cd", outputs["cd"], on_step=True, on_epoch=True)
        self.log("train_mi", outputs["mi"], on_step=True, on_epoch=True)
        self.log("kl_weight", kl_weight, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch, mode="val")

        nll = self.loss_struct.nll_entropy(outputs["logits"], batch["targets"])
        z_mu_in = outputs["z_mu"]
        z_logvar_in = outputs["z_logvar"]
        if z_mu_in.dim() == 2 and getattr(self.ebm, "num_latent_samples", 1) > 1:
            z_mu_in = (
                z_mu_in.unsqueeze(1)
                .expand(-1, self.ebm.num_latent_samples, -1)
                .contiguous()
                .view(z_mu_in.size(0), -1)
            )
            z_logvar_in = (
                z_logvar_in.unsqueeze(1)
                .expand(-1, self.ebm.num_latent_samples, -1)
                .contiguous()
                .view(z_logvar_in.size(0), -1)
            )
        kl = self.loss_struct.kl_div(
            self.ebm, batch["tgt_probs"], z_mu_in, z_logvar_in
        ).mean()

        # Use full weight (no anneal) for kl and cd during test_step
        total_loss = (
            nll + (kl - outputs["cd"]) - self.loss_struct.mi_weight * outputs["mi"]
        )
        perplexity = torch.exp(nll)

        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_nll", nll, on_step=False, on_epoch=True)
        self.log("val_kl", kl, on_step=False, on_epoch=True)
        self.log("val_cd", outputs["cd"], on_step=False, on_epoch=True)
        self.log("val_mi", outputs["mi"], on_step=False, on_epoch=True)
        self.log(
            "val_perplexity", perplexity, on_step=False, on_epoch=True, prog_bar=True
        )

        return total_loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch, mode="test", gen_type=self.gen_type)
        pred_ids = outputs["logits"]

        pad_id = bos_id = eos_id = None
        if hasattr(self.dec, "token_manager"):
            tm = getattr(self.dec, "token_manager")
            pad_id = getattr(tm, "pad_id", None)
            bos_id = getattr(tm, "bos_id", None)
            eos_id = getattr(tm, "eos_id", None)

        preds_text = ids_to_text_list(pred_ids, pad_id, bos_id, eos_id)
        refs_text = [
            [t] for t in ids_to_text_list(batch["targets"], pad_id, bos_id, eos_id)
        ]
        self.bleu.update(preds_text, refs_text)
        return {}

    def on_test_epoch_end(self):
        bleu = self.bleu.compute()
        self.log("test_bleu", bleu, prog_bar=True)
        self.bleu.reset()

    def generate(self, batch_size, max_len=None):
        if max_len is None:
            max_len = self.max_dec_len

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        z = ula_prior(self.ebm, z)
        return self.dec.generate(z, max_len)

    def configure_optimizers(self):
        non_ebm_params = [
            p for n, p in self.named_parameters() if not n.startswith("ebm.")
        ]
        ebm_params = [p for n, p in self.named_parameters() if n.startswith("ebm.")]
        return torch.optim.Adam(
            [
                {"params": non_ebm_params, "lr": self.learning_rate},
                {"params": ebm_params, "lr": self.ebm_learning_rate},
            ]
        )
