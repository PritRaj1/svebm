import lightning as L
import torch


class SVEBM(L.LightningModule):
    def __init__(
        self,
        ebm_model: L.LightningModule,
        encoder_model: L.LightningModule,
        decoder_model: L.LightningModule,
        learning_rate: float = 1e-3,
        data_dim: int = None,
        latent_dim: int = None,
        ebm_out_dim: int = None,
    ):
        super().__init__()

        self.ebm = ebm_model
        self.enc = encoder_model
        self.dec = decoder_model
        self.learning_rate = learning_rate

        if latent_dim is None:
            if hasattr(self.enc, "latent_dim"):
                latent_dim = self.enc.latent_dim
            else:
                raise ValueError(
                    "latent_dim must be provided or encoder must have "
                    "latent_dim attribute"
                )

        if ebm_out_dim is None:
            ebm_out_dim = latent_dim

        if data_dim is None:
            if hasattr(self.enc, "input_dim"):
                data_dim = self.enc.input_dim
            else:
                raise ValueError(
                    "data_dim must be provided or encoder must have "
                    "input_dim attribute"
                )

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
    ):
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

        print("Dimension validation passed:")
        print(
            f"   Data: {data_dim} → Encoder → Latent: {latent_dim} → "
            f"EBM → EBM_out: {ebm_out_dim} → Decoder → Data: {data_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        ebm_out = self.ebm(z)
        x_hat = self.dec(ebm_out)
        return x_hat
