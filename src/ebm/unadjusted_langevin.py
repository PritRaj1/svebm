import torch

from src.ebm.ebm_model import EBM_fcn


def ula_prior(ebm: EBM_fcn, z: torch.Tensor) -> torch.Tensor:
    """
    Unadjusted Langevin Algorithm (ULA) to evolve init z.

    Args:
        ebm (EBM_fcn): The energy-based model.
        z (torch.Tensor): Init sample (batch, latent_dim).

    Returns:
        torch.Tensor: Evolved samples (batch, latent_dim).
    """
    eta = ebm.eta
    N = ebm.N
    sqrt_eta = eta**0.5

    z = z.clone().detach().requires_grad_(True)
    
    for _ in range(N):
        energy = ebm(z).sum()
    
        grad = torch.autograd.grad(energy, z, only_inputs=True, retain_graph=False)[0]        
        noise = torch.randn_like(z, requires_grad=True)
        z = z - 0.5 * eta * grad + sqrt_eta * noise        

    return z.detach()
