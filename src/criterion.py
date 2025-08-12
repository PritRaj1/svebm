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
        dim_target_kl: float = 1.0,
    ):
        self.ignore_index = ignore_index  # PAD token index
        self.cls_id = cls_id
        self.kl_weight = kl_weight
        self.nll_weight = nll_weight
        self.mi_weight = mi_weight
        self.dim_target_kl = dim_target_kl
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
                # Mean pooling with padding mask
                padding_mask = (z != 0).float().sum(dim=-1, keepdim=True)
                z = z.sum(dim=1) / padding_mask.clamp(min=1)

        # P(y|z)
        log_cond = F.log_softmax(ebm(z), dim=-1)
        cond = torch.exp(log_cond)

        # P(y) (Monte Carlo estimate)
        log_marg = torch.log(torch.mean(cond, dim=0) + 1e-15)

        # Entropies
        H_y = -torch.sum(torch.exp(log_marg) * log_marg)
        H_z = -torch.sum(torch.exp(log_cond) * log_cond) / batch_size

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

    def simple_kl_div(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence with target masking."""
        kl_per_dim = -0.5 * (1 + logvar - mean**2 - logvar.exp())
        kl_mask = (kl_per_dim > self.dim_target_kl).float()
        kl = (kl_mask * kl_per_dim).sum(dim=1).mean()
        return kl

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
            tgt_probs_exp = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, ebm.latent_dim)
            eta1 = ebm.mix_mus / torch.exp(ebm.mix_logvars)  # \Sigma^-1 * \mu
            eta2 = -0.5 * torch.pow(torch.exp(ebm.mix_logvars), -1)  # -0.5 * \Sigma^-1

            E_eta1 = torch.sum(tgt_probs_exp * eta1, dim=-2)
            E_eta2 = torch.sum(tgt_probs_exp * eta2, dim=-2)

            E_mu = -0.5 * E_eta1 / E_eta2
            E_var = -0.5 / E_eta2

            kl = 0.5 * (
                torch.sum(torch.exp(logvar) / E_var, dim=-1)
                + torch.sum((E_mu - mean) ** 2 / E_var, dim=-1)
                - mean.size(-1)
                + torch.sum(torch.log(E_var) - logvar, dim=-1)
            )
            return kl

        # Direct KL against each GMM component
        else:
            mean_exp = mean.unsqueeze(-2).expand(-1, -1, ebm.num_gmm_components, -1)
            logvar_exp = logvar.unsqueeze(-2).expand(-1, -1, ebm.num_gmm_components, -1)

            kl = 0.5 * (
                torch.sum(torch.exp(logvar_exp) / torch.exp(ebm.mix_logvars), dim=-1)
                + torch.sum(
                    (ebm.mix_mus - mean_exp) ** 2 / torch.exp(ebm.mix_logvars), dim=-1
                )
                - mean.size(-1)
                + torch.sum(ebm.mix_logvars - logvar_exp, dim=-1)
            )

            return torch.sum(kl * tgt_probs, dim=-1)

    def dispersion(self, ebm: EBM_fcn, tgt_probs: torch.Tensor) -> torch.Tensor:
        """Dispersion loss for GMM (from original IB-EBM)."""
        tgt_probs_exp = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, ebm.latent_dim)
        eta1 = ebm.mix_mus / torch.exp(ebm.mix_logvars)  # \Sigma^-1 * \mu
        eta2 = -0.5 * torch.pow(torch.exp(ebm.mix_logvars), -1)  # -0.5 * \Sigma^-1

        E_eta1 = torch.sum(tgt_probs_exp * eta1, dim=-2)
        E_eta2 = torch.sum(tgt_probs_exp * eta2, dim=-2)

        # Expected log-likelihood under expected parameters
        E_log_lik = -0.25 * E_eta1 * E_eta1 / E_eta2 - 0.5 * torch.log(-2 * E_eta2)
        E_log_lik = torch.mean(torch.sum(E_log_lik, dim=(-1, -2)))

        # Expectation of log-likelihood under individual parameters
        log_lik = torch.sum(
            -0.25 * eta1 * eta1 / eta2 - 0.5 * torch.log(-2 * eta2), dim=-1
        )
        log_lik = torch.mean(torch.sum(tgt_probs * log_lik, dim=(-1, -2)))

        return log_lik - E_log_lik

    def param_var(self, ebm: EBM_fcn, tgt_probs: torch.Tensor) -> torch.Tensor:
        """Parameter variance loss (from original IB-EBM)."""
        tgt_probs_exp = tgt_probs.unsqueeze(-1).expand(-1, -1, -1, ebm.latent_dim)
        eta1 = ebm.mix_mus / torch.exp(ebm.mix_logvars)  # \Sigma^-1 * \mu
        eta2 = -0.5 * torch.pow(torch.exp(ebm.mix_logvars), -1)  # -0.5 * \Sigma^-1

        var_eta1 = torch.sum(tgt_probs_exp * (eta1 * eta1), dim=-2) - torch.sum(
            tgt_probs_exp * eta1, dim=-2
        ).pow(2)
        var_eta2 = torch.sum(tgt_probs_exp * (eta2 * eta2), dim=-2) - torch.sum(
            tgt_probs_exp * eta2, dim=-2
        ).pow(2)

        return torch.sum(var_eta1 + var_eta2) / tgt_probs.size(0)
