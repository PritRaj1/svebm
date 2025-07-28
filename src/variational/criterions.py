import torch
import torch.nn.functional as F

from src.ebm.ebm_model import EBM_fcn

# Optional criterion for information bottleneck
def mutual_information(ebm: EBM_fcn, z: torch.Tensor, cls: bool = True) -> torch.Tensor:
    """Mutual information between z and class labels."""
    batch_size = z.shape[0]

    if cls:
        z = z[:, 0, :]  # CLS token
    else:
        z = z.sum(dim=1) # Sum pooled (placeholder for learned pooling)
    
    # P(y|z)
    log_conditional = F.log_softmax(ebm(z), dim=-1)
    conditional = torch.exp(log_conditional)

    # P(y) with Monte Carlo estimate
    log_marginal = torch.log(torch.mean(conditional, dim=0))

    # Entropies
    H_y = -torch.sum(torch.exp(log_marginal) * log_marginal)
    H_z = -torch.sum(torch.exp(log_conditional) * log_conditional) / batch_size

    return H_y - H_z

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
    """Reparameterize a Gaussian distribution."""

    if sample:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    else:
        return mu

# KL(q∥p) = 1/2 [ Tr(Σ_q^{-1} Σ_p) + (μ_q − μ_p)^T Σ_q^{-1} (μ_q − μ_p) − k + log(det Σ_q / det Σ_p) ]
def kl_div(
    ebm: EBM_fcn,
    tgt_probs: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    mean_prior: bool = True,
) -> torch.Tensor:
    """KL divergence between Gaussian q(z|x) and p(z|y)."""
    batch, seq_len, latent_dim = mean.shape
    k = tgt_probs.shape[-1]

    # Per-token Gaussians assuming independent latent dim
    mean = mean.view(batch, seq_len, latent_dim)
    logvar = logvar.view(batch, seq_len, latent_dim)

    if mean_prior:
        tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, k, latent_dim)
        eta1 = ebm.mus / torch.exp(ebm.logvars)
        eta2 = -0.5 * torch.exp(-ebm.logvars)

        Eeta1 = torch.sum(tgt_probs_ * eta1, dim=2)
        Eeta2 = torch.sum(tgt_probs_ * eta2, dim=2)

        Emu = -0.5 * Eeta1 / Eeta2
        Evar = -0.5 / Eeta2

        kl = 0.5 * (
            torch.sum(torch.exp(logvar) / Evar, dim=-1)
            + torch.sum((Emu - mean) ** 2 / Evar, dim=-1)
            - latent_dim
            + torch.sum(torch.log(Evar) - logvar, dim=-1)
        )
        return kl

    else:
        mu_repeat = mean.unsqueeze(2).expand(-1, -1, k, -1)
        logvar_repeat = logvar.unsqueeze(2).expand(-1, -1, k, -1)

        kl = 0.5 * (
            torch.sum(torch.exp(logvar_repeat) / torch.exp(ebm.logvars), dim=-1)
            + torch.sum((ebm.mus - mu_repeat) ** 2 / torch.exp(ebm.logvars), dim=-1)
            - latent_dim
            + torch.sum(ebm.logvars - logvar_repeat, dim=-1)
        )

        kl = torch.sum(kl * tgt_probs, dim=-1)
        return kl
