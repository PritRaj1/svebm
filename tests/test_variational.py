import pytest
import torch

from src.ebm.ebm_model import EBM_fcn
from src.variational.encoder_model import EncoderModel
from src.variational.criterions import mutual_information


class TestVariationalComponents:

    @pytest.fixture
    def test_encoder(self):
        return EncoderModel(
            input_dim=768,
            output_dim=128,
            hidden_layers=[256, 128],
            nhead=8,
            dropout=0.1,
            activation="relu",
        )

    @pytest.fixture
    def test_ebm(self):
        return EBM_fcn(
            input_dim=128, output_dim=10, hidden_layers=[64, 32], activation="relu"
        )

    def test_encoder_instantiation(self, test_encoder):
        assert test_encoder.input_dim == 768
        assert test_encoder.output_dim == 128
        assert test_encoder.nhead == 8
        assert test_encoder.dropout == 0.1
        assert test_encoder.activation == "relu"
        assert len(test_encoder.hidden_layers) == 2
        assert test_encoder.hidden_layers == [256, 128]

    def test_encoder_forward(self, test_encoder):
        batch_size = 32
        seq_len = 16
        input_dim = 768

        x = torch.randn(batch_size, seq_len, input_dim)
        z = test_encoder(x)

        assert z.shape == (batch_size, seq_len, 128)
        assert not torch.isnan(z).any()
        assert not torch.isinf(z).any()

    def test_mi_shapes(self, test_ebm):
        batch_size = 32
        seq_len = 10
        latent_dim = 128

        z = torch.randn(batch_size, seq_len, latent_dim)
        mi = mutual_information(test_ebm, z)

        assert isinstance(mi, torch.Tensor)
        assert mi.shape == ()  # Scalar
        assert not torch.isnan(mi)
        assert not torch.isinf(mi)
        assert mi >= 0

    def test_mi_output(self, test_ebm):
        batch_size = 64
        seq_len = 10
        latent_dim = 128

        z1 = torch.randn(batch_size, seq_len, latent_dim)
        z2 = torch.randn(batch_size, seq_len, latent_dim)

        mi1 = mutual_information(test_ebm, z1)
        mi2 = mutual_information(test_ebm, z2)

        assert not torch.isnan(mi1)
        assert not torch.isnan(mi2)
        assert not torch.isinf(mi1)
        assert not torch.isinf(mi2)
        assert mi1 >= 0
        assert mi2 >= 0
