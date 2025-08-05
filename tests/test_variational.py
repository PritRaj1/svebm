import pytest
import torch

from src.ebm.ebm_model import EBM_fcn
from src.variational.encoder_model import EncoderModel
from src.variational.decoder_model import DecoderModel
from src.criterion import LogProb


class TestVariationalComponents:

    @pytest.fixture
    def test_encoder(self):
        return EncoderModel(
            input_dim=768,
            output_dim=128,
            latent_dim=128,
            hidden_layers=[256, 128],
            nhead=8,
            dropout=0.1,
            activation="relu",
            pad_id=0,
        )

    @pytest.fixture
    def test_decoder(self):
        return DecoderModel(
            vocab_size=30522,
            embed_size=768,
            latent_dim=128,
            hidden_layers=[128, 256],
            nhead=8,
            dropout=0.1,
            activation="relu",
            max_dec_len=50,
            pad_id=0,
            bos_id=1,
            eos_id=2,
            unk_id=3,
            concat_latent=False,
        )

    @pytest.fixture
    def test_ebm(self):
        return EBM_fcn(
            latent_dim=128,
            num_classes=10,
            hidden_layers=[64, 32],
            activation="relu",
        )

    @pytest.fixture
    def test_logprob(self):
        return LogProb(
            ignore_index=0,
            cls_id=0,
            kl_weight=1.0,
            nll_weight=1.0,
        )

    def test_encoder_instantiation(self, test_encoder):
        assert test_encoder.input_dim == 768
        assert test_encoder.output_dim == 128
        assert test_encoder.latent_dim == 128
        assert test_encoder.nhead == 8
        assert test_encoder.dropout == 0.1
        assert test_encoder.activation == "relu"
        assert test_encoder.pad_id == 0
        assert len(test_encoder.hidden_layers) == 2
        assert test_encoder.hidden_layers == [256, 128]

    def test_encoder_forward_no_mask(self, test_encoder):
        batch_size = 32
        seq_len = 16
        input_dim = 768

        x = torch.randn(batch_size, seq_len, input_dim)
        z_mu, z_logvar, hidden_st = test_encoder(x)

        # Global latent variables (one per sequence)
        assert z_mu.shape == (batch_size, 128)
        assert z_logvar.shape == (batch_size, 128)
        assert hidden_st.shape == (batch_size, seq_len, 128)
        
        assert not torch.isnan(z_mu).any()
        assert not torch.isinf(z_mu).any()
        assert not torch.isnan(z_logvar).any()
        assert not torch.isinf(z_logvar).any()
        assert not torch.isnan(hidden_st).any()
        assert not torch.isinf(hidden_st).any()

    def test_encoder_forward_with_mask(self, test_encoder):
        batch_size = 32
        seq_len = 16
        input_dim = 768

        x = torch.randn(batch_size, seq_len, input_dim)
        # Create padding mask (last 4 tokens are padding)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -4:] = True  # Last 4 tokens are padding
        
        z_mu, z_logvar, hidden_st = test_encoder(x, mask=mask)

        # Global latent variables (one per sequence)
        assert z_mu.shape == (batch_size, 128)
        assert z_logvar.shape == (batch_size, 128)
        assert hidden_st.shape == (batch_size, seq_len, 128)
        
        assert not torch.isnan(z_mu).any()
        assert not torch.isinf(z_mu).any()
        assert not torch.isnan(z_logvar).any()
        assert not torch.isinf(z_logvar).any()
        assert not torch.isnan(hidden_st).any()
        assert not torch.isinf(hidden_st).any()

    def test_decoder_instantiation(self, test_decoder):
        assert test_decoder.vocab_size == 30522
        assert test_decoder.embed_size == 768
        assert test_decoder.latent_dim == 128
        assert test_decoder.nhead == 8
        assert test_decoder.dropout == 0.1
        assert test_decoder.activation == "relu"
        assert test_decoder.max_dec_len == 50
        assert test_decoder.pad_id == 0
        assert test_decoder.bos_id == 1
        assert test_decoder.eos_id == 2
        assert test_decoder.unk_id == 3
        assert test_decoder.concat_latent == False

    def test_decoder_teacher_force(self, test_decoder):
        batch_size = 32
        seq_len = 16
        embed_size = 768

        # Input tokens
        inputs = torch.randint(0, 1000, (batch_size, seq_len))
        # Encoder memory (hidden states)
        memory = torch.randn(batch_size, seq_len, embed_size)
        
        logits = test_decoder(inputs, memory, mode="TEACH_FORCE")
        
        assert logits.shape == (batch_size, seq_len, 30522)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_decoder_greedy_generation(self, test_decoder):
        batch_size = 8  # Smaller batch for generation
        seq_len = 16
        embed_size = 768

        # Encoder memory (hidden states)
        memory = torch.randn(batch_size, seq_len, embed_size)
        
        generated = test_decoder(None, memory, mode="GENERATE", gen_type="greedy")
        
        # Should return generated sequences
        assert generated.shape[0] == batch_size
        assert generated.dtype == torch.long
        assert not torch.isnan(generated).any()

    def test_decoder_beam_search(self, test_decoder):
        batch_size = 4  # Smaller batch for beam search
        seq_len = 16
        embed_size = 768

        # Encoder memory (hidden states)
        memory = torch.randn(batch_size, seq_len, embed_size)
        
        generated = test_decoder(None, memory, mode="GENERATE", gen_type="beam")
        
        # Should return generated sequences
        assert generated.shape[0] == batch_size
        assert generated.dtype == torch.long
        assert not torch.isnan(generated).any()

    def test_mi_shapes_global_latent(self, test_ebm, test_logprob):
        batch_size = 32
        latent_dim = 128

        # Global latent variables (one per sequence)
        z = torch.randn(batch_size, latent_dim)
        mi = test_logprob.mutual_information(test_ebm, z)

        assert isinstance(mi, torch.Tensor)
        assert mi.shape == ()  # Scalar
        assert not torch.isnan(mi)
        assert not torch.isinf(mi)
        assert mi >= 0

    def test_mi_output_global_latent(self, test_ebm, test_logprob):
        batch_size = 64
        latent_dim = 128

        # Global latent variables (one per sequence)
        z1 = torch.randn(batch_size, latent_dim)
        z2 = torch.randn(batch_size, latent_dim)

        mi1 = test_logprob.mutual_information(test_ebm, z1)
        mi2 = test_logprob.mutual_information(test_ebm, z2)

        assert not torch.isnan(mi1)
        assert not torch.isnan(mi2)
        assert not torch.isinf(mi1)
        assert not torch.isinf(mi2)
        assert mi1 >= 0
        assert mi2 >= 0

    def test_encoder_decoder_integration(self, test_encoder, test_decoder):
        batch_size = 16
        seq_len = 12
        input_dim = 768

        x = torch.randn(batch_size, seq_len, input_dim)
        z_mu, z_logvar, hidden_st = test_encoder(x)
        
        # Project encoder output to decoder's expected embed_size (placeholder for ebm flow)
        projection = torch.nn.Linear(128, 768)
        memory = projection(hidden_st)
        
        inputs = torch.randint(0, 1000, (batch_size, seq_len))
        logits = test_decoder(inputs, memory, mode="TEACH_FORCE")
        
        assert z_mu.shape == (batch_size, 128)  # Global latent
        assert z_logvar.shape == (batch_size, 128)  # Global latent
        assert hidden_st.shape == (batch_size, seq_len, 128)  # Hidden states
        assert memory.shape == (batch_size, seq_len, 768)  # Projected memory
        assert logits.shape == (batch_size, seq_len, 30522)  # Decoder output
