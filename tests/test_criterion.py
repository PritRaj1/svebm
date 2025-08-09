import pytest
import torch

from src.criterion import LogProb
from src.ebm.ebm_model import EBM_fcn


class TestLogProb:

    @pytest.fixture
    def logprob(self):
        return LogProb(
            ignore_index=0, cls_id=0, kl_weight=1.0, nll_weight=1.0, mi_weight=1.0
        )

    @pytest.fixture
    def ebm(self):
        return EBM_fcn(
            latent_dim=8,
            num_classes=5,
            hidden_layers=[16, 8],
            num_latent_samples=2,
            num_gmm_components=3,
        )

    def test_nll_entropy_and_reparameterize(self, logprob):
        batch_size, seq_len, vocab = 4, 6, 10
        logits = torch.randn(batch_size, seq_len, vocab)
        targets = torch.randint(0, vocab, (batch_size, seq_len))

        nll = logprob.nll_entropy(logits, targets)
        assert isinstance(nll, torch.Tensor)
        assert nll.dim() == 0
        assert torch.isfinite(nll)

        mu = torch.randn(batch_size, 8)
        logvar = torch.randn(batch_size, 8)
        z_sampled = logprob.reparameterize(mu, logvar, sample=True)
        z_deterministic = logprob.reparameterize(mu, logvar, sample=False)

        assert z_sampled.shape == mu.shape
        assert torch.allclose(z_deterministic, mu)

    def test_mutual_info_kl_and_contrastive(self, logprob, ebm):
        batch_size = 5
        latent_dim = ebm.latent_dim
        num_latent_samples = ebm.num_latent_samples
        num_gmm = ebm.num_gmm_components

        z = torch.randn(batch_size, latent_dim)
        mi = logprob.mutual_information(ebm, z)
        assert isinstance(mi, torch.Tensor)
        assert mi.dim() == 0
        assert torch.isfinite(mi)

        mean = torch.randn(batch_size, num_latent_samples * latent_dim)
        logvar = torch.randn(batch_size, num_latent_samples * latent_dim)
        tgt_probs = torch.full((batch_size, num_latent_samples, num_gmm), 1.0 / num_gmm)

        kl_mean_prior = logprob.kl_div(ebm, tgt_probs, mean, logvar, mean_prior=True)
        assert isinstance(kl_mean_prior, torch.Tensor)
        assert kl_mean_prior.shape[0] == batch_size
        assert torch.isfinite(kl_mean_prior).all()

        z_prior = torch.randn(batch_size, latent_dim)
        z_post = torch.randn(batch_size, latent_dim)
        cd = logprob.contrastive_loss(ebm, z_prior, z_post)
        assert isinstance(cd, torch.Tensor)
        assert cd.dim() == 0
        assert torch.isfinite(cd)
