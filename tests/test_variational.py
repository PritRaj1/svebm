import pytest
import torch

from src.ebm.ebm_model import EBM_fcn
from src.variational.encoder_model import EncoderModel
from src.variational.decoder_model import DecoderModel
from src.criterion import LogProb
from src.ebm.unadjusted_langevin import ula_prior


class TestVariationalComponents:

    @pytest.fixture
    def test_encoder(self):
        return EncoderModel(
            input_dim=768,
            memory_dim=128,
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
            memory_dim=128,
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
            num_latent_samples=20,
            num_gmm_components=5,
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
        assert test_encoder.memory_dim == 128
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
        assert z_mu.shape == (batch_size, test_encoder.latent_dim)
        assert z_logvar.shape == (batch_size, test_encoder.latent_dim)
        assert hidden_st.shape == (batch_size, seq_len, test_encoder.memory_dim)

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
        assert z_mu.shape == (batch_size, test_encoder.latent_dim)
        assert z_logvar.shape == (batch_size, test_encoder.latent_dim)
        assert hidden_st.shape == (batch_size, seq_len, test_encoder.memory_dim)

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
        assert test_decoder.token_manager.pad_id == 0
        assert test_decoder.token_manager.bos_id == 1
        assert test_decoder.token_manager.eos_id == 2
        assert test_decoder.token_manager.unk_id == 3
        assert not test_decoder.concat_latent

    def test_decoder_teacher_force(self, test_decoder):
        batch_size = 32
        seq_len = 16
        memory_dim = test_decoder.memory_dim

        inputs = torch.randint(0, 1000, (batch_size, seq_len))
        memory = torch.randn(batch_size, seq_len, memory_dim)
        logits = test_decoder(inputs, memory, mode="TEACH_FORCE")

        assert logits.shape == (batch_size, seq_len, test_decoder.vocab_size)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_decoder_greedy_generation(self, test_decoder):
        batch_size = 8
        seq_len = 16
        memory_dim = test_decoder.memory_dim

        memory = torch.randn(batch_size, seq_len, memory_dim)
        generated = test_decoder(None, memory, mode="GENERATE", gen_type="greedy")
        assert generated.shape[0] == batch_size
        assert generated.dtype == torch.long
        assert not torch.isnan(generated).any()

    def test_decoder_beam_search(self, test_decoder):
        batch_size = 4
        seq_len = 16
        memory_dim = test_decoder.memory_dim

        memory = torch.randn(batch_size, seq_len, memory_dim)
        generated = test_decoder(None, memory, mode="GENERATE", gen_type="beam")

        assert generated.shape[0] == batch_size
        assert generated.dtype == torch.long
        assert not torch.isnan(generated).any()

    def test_mi_shapes_global_latent(self, test_ebm, test_logprob):
        batch_size = 32
        latent_dim = 128

        z = torch.randn(batch_size, latent_dim)
        mi = test_logprob.mutual_information(test_ebm, z)

        assert isinstance(mi, torch.Tensor)
        assert mi.shape == ()
        assert not torch.isnan(mi)
        assert not torch.isinf(mi)
        assert mi >= 0

    def test_mi_output_global_latent(self, test_ebm, test_logprob):
        batch_size = 64
        latent_dim = 128

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

    def test_encoder_decoder_integration(
        self, test_encoder, test_decoder, test_ebm, test_logprob
    ):
        batch_size = 16
        seq_len = 12
        input_dim = 768
        latent_dim = 128
        vocab_size = 30522

        x = torch.randint(
            4, 1000, (batch_size, seq_len)
        )  # Avoid special tokens (0, 1, 2, 3)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -2:] = True  # Last two tokens are padding

        embedding = torch.nn.Embedding(vocab_size, input_dim, padding_idx=0)
        embedded = embedding(x)

        z_mu, z_logvar, hidden_st = test_encoder(embedded, mask=mask)
        sample_z = test_logprob.reparameterize(z_mu, z_logvar, sample=True)

        inputs = x[:, :-1]  # Ground-truth input tokens (shifted)
        logits = test_decoder(inputs, hidden_st, mode="TEACH_FORCE")

        labels = x[:, 1:].contiguous()  # Target tokens (shifted)
        nll = test_logprob.nll_entropy(logits, labels)
        mi = test_logprob.mutual_information(test_ebm, sample_z, cls=False)

        # Dummy probs (matches GMM structure)
        tgt_probs = torch.softmax(
            torch.randn(
                batch_size, test_ebm.num_latent_samples, test_ebm.num_gmm_components
            ),
            dim=-1,
        )
        z_mu = z_mu.unsqueeze(1).expand(-1, test_ebm.num_latent_samples, -1)
        z_logvar = z_logvar.unsqueeze(1).expand(-1, test_ebm.num_latent_samples, -1)
        zkl = test_logprob.kl_div(
            test_ebm, tgt_probs=tgt_probs, mean=z_mu, logvar=z_logvar
        )
        prob_pos = test_ebm.ebm_prior(sample_z).mean()

        z_e_0 = torch.randn(batch_size, latent_dim, device=sample_z.device)
        prior_z = ula_prior(test_ebm, z_e_0)
        prob_neg = test_ebm.ebm_prior(prior_z.detach()).mean()

        assert z_mu.shape == (batch_size, latent_dim)
        assert z_logvar.shape == (batch_size, latent_dim)
        assert hidden_st.shape == (batch_size, seq_len, test_encoder.memory_dim)
        assert logits.shape == (batch_size, seq_len - 1, vocab_size)
        assert nll.shape == ()
        assert mi.shape == ()
        assert zkl.shape == (batch_size, test_ebm.num_latent_samples)
        assert prob_pos.shape == ()
        assert prob_neg.shape == ()
        assert not torch.isnan(z_mu).any()
        assert not torch.isinf(z_mu).any()
        assert not torch.isnan(z_logvar).any()
        assert not torch.isinf(z_logvar).any()
        assert not torch.isnan(hidden_st).any()
        assert not torch.isinf(hidden_st).any()
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        assert not torch.isnan(nll)
        assert not torch.isinf(nll)
        assert not torch.isnan(mi)
        assert not torch.isinf(mi)
        assert not torch.isnan(zkl).any()
        assert not torch.isinf(zkl).any()
        assert not torch.isnan(prob_pos)
        assert not torch.isinf(prob_pos)
        assert not torch.isnan(prob_neg)
        assert not torch.isinf(prob_neg)
