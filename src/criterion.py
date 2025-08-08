import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ebm.ebm_model import EBM_fcn


class LogProb:
    def __init__(
        self,
        ignore_index: int,
        cls_id: int,
        kl_weight: float = 1.0,
        nll_weight: float = 1.0,
        mi_weight: float = 1.0,
    ):
        self.ignore_index = ignore_index  # PAD token index
        self.cls_id = cls_id
        self.kl_weight = kl_weight
        self.nll_weight = nll_weight
        self.mi_weight = mi_weight
        self.nll_criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction="mean"
        )

    def contrastive_loss(
        self,
        ebm: EBM_fcn,
        z_prior: torch.Tensor,
        z_posterior: torch.Tensor,
    ) -> torch.Tensor:
        """Contrastive loss between prior and posterior latent samples."""
        energy_prior = ebm.ebm_prior(z_prior)
        energy_posterior = ebm.ebm_prior(z_posterior)
        return energy_posterior.mean() - energy_prior.mean()

    def nll_entropy(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss for sequence reconstruction."""
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
        return self.nll_criterion(logits, targets)

    def mutual_information(
        self, ebm: EBM_fcn, z: torch.Tensor, cls: bool = True
    ) -> torch.Tensor:
        """Mutual information between z and class labels."""
        batch_size = z.shape[0]

        if z.dim() == 3:
            if cls and self.cls_id is not None:
                z = z[:, 0, :]
            else:
                # Mean pooling with padding mask, (placeholder for learned pooling)
                mask = (z != 0).float().sum(dim=-1, keepdim=True)
                z = z.sum(dim=1) / mask.clamp(min=1)

        # P(y|z)
        log_conditional = F.log_softmax(ebm(z), dim=-1)
        conditional = torch.exp(log_conditional)

        # P(y) with Monte Carlo estimate
        log_marginal = torch.log(torch.mean(conditional, dim=0) + 1e-15)

        # Entropies
        H_y = -torch.sum(torch.exp(log_marginal) * log_marginal)
        H_z = -torch.sum(torch.exp(log_conditional) * log_conditional) / batch_size

        return H_y - H_z

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """Reparameterize a Gaussian distribution."""
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def kl_div(
        self,
        ebm: EBM_fcn,
        tgt_probs: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        mean_prior: bool = True,
    ) -> torch.Tensor:
        """KL divergence between Gaussian q(z|x) and GMM prior p(z|y)."""
        mean = mean.view(-1, ebm.num_latent_samples, ebm.latent_dim)
        logvar = logvar.view(-1, ebm.num_latent_samples, ebm.latent_dim)

        # KL against mean prior
        if mean_prior:
            tgt_probs_ = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, ebm.latent_dim)
            eta1 = ebm.mix_mus / torch.exp(ebm.mix_logvars)  # eta1 = \Sigma^-1 * mu
            eta2 = -0.5 * torch.pow(torch.exp(ebm.mix_logvars), -1)

            Eeta1 = torch.sum(tgt_probs_ * eta1, dim=-2)
            Eeta2 = torch.sum(tgt_probs_ * eta2, dim=-2)

            Emu = -0.5 * Eeta1 / Eeta2
            Evar = -0.5 / Eeta2

            kl = 0.5 * (
                torch.sum(torch.exp(logvar) / Evar, dim=-1)
                + torch.sum((Emu - mean) ** 2 / Evar, dim=-1)
                - mean.size(-1)
                + torch.sum(torch.log(Evar) - logvar, dim=-1)
            )
            return kl

        # Direct KL against each GMM component
        else:
            mu_repeat = mean.unsqueeze(-2).expand(-1, -1, ebm.num_gmm_components, -1)
            logvar_repeat = logvar.unsqueeze(-2).expand(
                -1, -1, ebm.num_gmm_components, -1
            )

            kl = 0.5 * (
                torch.sum(torch.exp(logvar_repeat) / torch.exp(ebm.mix_logvars), dim=-1)
                + torch.sum(
                    (ebm.mix_mus - mu_repeat) ** 2 / torch.exp(ebm.mix_logvars), dim=-1
                )
                - mean.size(-1)
                + torch.sum(ebm.mix_logvars - logvar_repeat, dim=-1)
            )

            return torch.sum(kl * tgt_probs, dim=-1)
