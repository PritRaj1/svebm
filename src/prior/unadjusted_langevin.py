import torch

from src.prior.ebm_model import EBMModel


def ula(ebm: EBMModel, z: torch.Tensor) -> torch.Tensor:
    """
    Unadjusted Langevin Algorithm (ULA) to evolve init z.

    Args:
        ebm (EBMModel): The energy-based model.
        z (torch.Tensor): Init sample (batch, latent_dim).

    Returns:
        torch.Tensor: Evolved samples (batch, latent_dim).
    """
    eta = ebm.eta
    N = ebm.N
    sqrt_eta = eta ** 0.5

    z = z.detach() 
    for _ in range(N):
        z.requires_grad_(True)
        energy = ebm(z).sum()
        grad = torch.autograd.grad(energy, z, only_inputs=True, retain_graph=False)[0]
        noise = torch.randn_like(z)
        z = (z - 0.5 * eta * grad + sqrt_eta * noise).detach()

    return z

