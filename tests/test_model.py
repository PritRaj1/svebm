import torch
import pytest

from src.model import SVEBM
from src.criterion import LogProb
from src.ebm.ebm_model import EBM_fcn
from src.variational.encoder_model import EncoderModel
from src.variational.decoder_model import DecoderModel
from src.variational.kl_annealing import KLAnnealer


class TestModelSteps:

    @pytest.fixture
    def model_fixture(self):
        latent_dim = 8
        memory_dim = 16
        vocab_size = 50

        enc = EncoderModel(
            input_dim=32,
            memory_dim=memory_dim,
            latent_dim=latent_dim,
            hidden_layers=[32, 16],
            nhead=4,
            dropout=0.1,
            activation="relu",
            pad_id=0,
        )
        dec = DecoderModel(
            vocab_size=vocab_size,
            embed_size=32,
            latent_dim=latent_dim,
            memory_dim=memory_dim,
            hidden_layers=[16, 16],
            nhead=4,
            dropout=0.1,
            activation="relu",
            max_dec_len=10,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            concat_latent=False,
        )
        ebm = EBM_fcn(
            latent_dim=latent_dim,
            num_classes=5,
            hidden_layers=[16, 8],
            num_latent_samples=2,
            num_gmm_components=3,
        )
        loss_struct = LogProb(ignore_index=0, cls_id=0)
        kl_annealer = KLAnnealer(
            total_steps=100,
            n_cycle=2,
            ratio_increase=0.25,
            ratio_zero=0.5,
            max_kl_weight=0.5,
        )

        model = SVEBM(
            ebm_model=ebm,
            encoder_model=enc,
            decoder_model=dec,
            loss_struct=loss_struct,
            learning_rate=1e-3,
            data_dim=32,
            latent_dim=latent_dim,
            ebm_out_dim=latent_dim,
            kl_annealer=kl_annealer,
            ebm_learning_rate=1e-4,
        )
        return model

    def _synthetic_batch(
        self, model, batch_size=4, seq_len=12, input_dim=32, vocab_size=50
    ):
        x = torch.randn(batch_size, seq_len, input_dim)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len - 1))
        tgt_probs = torch.softmax(
            torch.randn(
                batch_size, model.ebm.num_latent_samples, model.ebm.num_gmm_components
            ),
            dim=-1,
        )
        return {
            "encoder_inputs": x,
            "inputs": targets,
            "targets": targets,
            "tgt_probs": tgt_probs,
        }

    def test_training_validation_testing_steps(self, model_fixture):
        model = model_fixture
        batch = self._synthetic_batch(model)

        loss = model.training_step(batch, batch_idx=0)
        assert torch.isfinite(loss).all().item()

        val_loss = model.validation_step(batch, batch_idx=0)
        assert torch.isfinite(val_loss).all().item()

        out = model.test_step(batch, batch_idx=0)
        assert isinstance(out, dict)
        model.on_test_epoch_end()
