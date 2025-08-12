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
        gen_type: str = "greedy",
        posterior_sample_n: int = 1,
        word_dropout_rate: float = 0.0,
        beta: float = 0.2,
    ):
        super().__init__()

        self.ebm = ebm_model
        self.enc = encoder_model
        self.dec = decoder_model
        self.loss_struct = loss_struct
        self.learning_rate = learning_rate
        self.ebm_learning_rate = ebm_learning_rate
        self.kl_annealer = kl_annealer
        self.posterior_sample_n = posterior_sample_n
        self.word_dropout_rate = word_dropout_rate
        self.beta = beta
        self.bleu = BLEUScore(n_gram=4, smooth=True)
        self.gen_type: str = gen_type

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

        if hasattr(self.dec, "max_dec_len"):
            self.max_dec_len = self.dec.max_dec_len
        else:
            self.max_dec_len = 50

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

    def _prep_kl_shape(
        self, z_mu: torch.Tensor, z_logvar: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_mu_in = z_mu
        z_logvar_in = z_logvar

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

        return z_mu_in, z_logvar_in

    def forward(self, batch, mode="train", gen_type="greedy"):
        x = batch["encoder_inputs"] if isinstance(batch, dict) else batch
        posterior_sample_n = (
            self.posterior_sample_n if self.training else 1
        )  # Num reparameterized samples
        z_mu, z_logvar, hidden = self.enc(x)

        # Posterior samples
        if posterior_sample_n > 1:
            z_mu_sampled = z_mu.repeat(posterior_sample_n, 1)
            z_logvar_sampled = z_logvar.repeat(posterior_sample_n, 1)
        else:
            z_mu_sampled, z_logvar_sampled = z_mu, z_logvar

        z = self.loss_struct.reparameterize(z_mu_sampled, z_logvar_sampled)

        # Prior samples
        prior_initialization = torch.randn(
            z_mu.size(0), z_mu.size(1), device=z_mu.device
        )
        z_prior = ula_prior(self.ebm, prior_initialization.requires_grad_(True))

        contrastive_divergence = self.loss_struct.contrastive_loss(self.ebm, z_prior, z)
        mutual_information = self.loss_struct.mutual_information(self.ebm, z)

        if mode in ("train", "val"):
            inputs = batch["inputs"]
            targets = batch["targets"]

            if self.word_dropout_rate > 0.0 and self.training:
                prob = torch.rand(inputs.size(), device=inputs.device)

                # Don't dropout special tokens
                special_mask = (
                    (inputs == 1) | (inputs == 0) | (inputs == 2)  # BOS  # PAD  # EOS
                )
                prob[special_mask] = 1.0

                inputs_copy = inputs.clone()
                inputs_copy[prob < self.word_dropout_rate] = 3  # UNK
                inputs = inputs_copy

            if posterior_sample_n > 1:
                inputs = inputs.repeat(posterior_sample_n, 1)
                targets = targets.repeat(posterior_sample_n, 1)

            logits = self.dec.teacher_force_forward(inputs, hidden)
        else:
            logits = self.dec.generate(hidden, gen_type)

        return {
            "logits": logits,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "z": z,
            "z_prior": z_prior,
            "contrastive_divergence": contrastive_divergence,
            "mutual_information": mutual_information,
            "hidden": hidden,
            "targets": targets if mode in ("train", "val") else None,
        }

    def training_step(self, batch, batch_idx):
        kl_weight = self.get_kl_weight(self.global_step)
        outputs = self(batch, mode="train")

        reconstruction_loss = self.loss_struct.nll_entropy(
            outputs["logits"], outputs["targets"]
        )

        kl_divergence = self.loss_struct.simple_kl_div(
            outputs["z_mu"], outputs["z_logvar"]
        )

        posterior_energy = self.ebm.ebm_prior(outputs["z"]).mean()
        prior_energy = self.ebm.ebm_prior(outputs["z_prior"]).mean()

        total_loss = (
            reconstruction_loss
            + kl_weight * (kl_divergence - posterior_energy + prior_energy)
            - self.loss_struct.mi_weight * outputs["mutual_information"]
        )

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_reconstruction_loss",
            reconstruction_loss,
            on_step=True,
            on_epoch=True,
        )
        self.log("train_kl_divergence", kl_divergence, on_step=True, on_epoch=True)
        self.log(
            "train_posterior_energy", posterior_energy, on_step=True, on_epoch=True
        )
        self.log("train_prior_energy", prior_energy, on_step=True, on_epoch=True)
        self.log(
            "train_mutual_information",
            outputs["mutual_information"],
            on_step=True,
            on_epoch=True,
        )
        self.log("kl_weight", kl_weight, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch, mode="val")

        reconstruction_loss = self.loss_struct.nll_entropy(
            outputs["logits"], outputs["targets"]
        )
        kl_divergence = self.loss_struct.simple_kl_div(
            outputs["z_mu"], outputs["z_logvar"]
        )

        posterior_energy = self.ebm.ebm_prior(outputs["z"]).mean()
        prior_energy = self.ebm.ebm_prior(outputs["z_prior"]).mean()

        # Use full weight (no anneal) for validation
        total_loss = (
            reconstruction_loss
            + (kl_divergence - posterior_energy + prior_energy)
            - self.loss_struct.mi_weight * outputs["mutual_information"]
        )
        perplexity = torch.exp(reconstruction_loss)

        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val_reconstruction_loss", reconstruction_loss, on_step=False, on_epoch=True
        )
        self.log("val_kl_divergence", kl_divergence, on_step=False, on_epoch=True)
        self.log("val_posterior_energy", posterior_energy, on_step=False, on_epoch=True)
        self.log("val_prior_energy", prior_energy, on_step=False, on_epoch=True)
        self.log(
            "val_mutual_information",
            outputs["mutual_information"],
            on_step=False,
            on_epoch=True,
        )
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
