import torch
import torch.nn.functional as F

from src.ebm.ebm_model import EBM_fcn

def mutual_information(ebm: EBM_fcn, z: torch.Tensor) -> torch.Tensor:
    """Mutual information between z and class labels."""

    z = z[:, 0, :] # CLS token
    batch_size = z.shape[0]
    
    # P(y|z)
    log_conditional = F.log_softmax(ebm(z), dim=-1)
    conditional = torch.exp(log_conditional)

    # P(y) with Monte Carlo estimate
    log_marginal = torch.log(torch.mean(conditional, dim=0))

    # Entropies
    H_y = -torch.sum(torch.exp(log_marginal) * log_marginal)
    H_z = -torch.sum(torch.exp(log_conditional) * log_conditional) / batch_size

    return H_y - H_z

