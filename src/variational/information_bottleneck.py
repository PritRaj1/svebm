import torch
import torch.nn.functional as F

from src.ebm.ebm_model import EBM_fcn


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
