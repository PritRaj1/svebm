import pytest
import torch
import torch.nn as nn
import numpy as np
from src.prior.ebm_model import EBMModel
from src.prior.unadjusted_langevin import ula


class GaussianEBM(EBMModel):
    def __init__(self, mean=2.0, std=1.0):
        super().__init__(input_dim=1, output_dim=1, hidden_layers=[64, 32])
        self.mean = mean
        self.std = std
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        return 0.5 * ((x - self.mean) / self.std) ** 2


class TestULAConvergence:
    
    @pytest.fixture
    def test_gaussian(self):
        return GaussianEBM(mean=2.0, std=1.0)
    
    @pytest.fixture
    def test_ebm(self):
        return EBMModel(input_dim=50, output_dim=1, hidden_layers=[64, 32])
    
    def test_ula_basics(self, test_ebm):
        batch_size = 32
        latent_dim = 50
        
        z_init = torch.randn(batch_size, latent_dim)        
        z_final = ula(test_ebm, z_init)
        
        assert z_final.shape == z_init.shape
        assert not z_final.requires_grad  # Should be detached
        
    def test_gaussian_convergence(self, test_gaussian):
        batch_size = 1000
        test_gaussian.eta = 0.1  
        test_gaussian.N = 50     
        
        z_init = torch.rand(batch_size, 1) * 6 - 3 # Uniform on [-3, 3]
        z_final = ula(test_gaussian, z_init)
        
        mean_final = z_final.mean().item()
        std_final = z_final.std().item()
        
        assert abs(mean_final - test_gaussian.mean) < 0.3
        assert abs(std_final - test_gaussian.std) < 0.3
        
    def test_energy_decrease(self, test_ebm):
        batch_size = 32
        latent_dim = 50
        test_ebm.eta = 0.1 
        test_ebm.N = 20     
        
        energy_changes = []
        for _ in range(5):
            z_init = torch.randn(batch_size, latent_dim)
            initial_energy = test_ebm(z_init).mean().item()
            z_final = ula(test_ebm, z_init)
            final_energy = test_ebm(z_final).mean().item()
            
            assert np.isfinite(final_energy)
            assert np.isfinite(initial_energy)
            
            energy_changes.append(final_energy - initial_energy)
        
        mean_change = np.mean(energy_changes)
        assert mean_change <= 1.0  # Average change should be reasonable
        assert np.std(energy_changes) > 0  # Should have some variance due to noise 
        
    def test_ula_med_dimensional(self, test_ebm):
        batch_size = 32
        latent_dim = 50
        test_ebm.eta = 0.1
        test_ebm.N = 15
        
        z_init = torch.randn(batch_size, latent_dim)
        z_final = ula(test_ebm, z_init)
        final_energy = test_ebm(z_final).mean().item()
        
        assert z_final.shape == (batch_size, latent_dim)
        assert not z_final.requires_grad
        assert np.isfinite(final_energy)
        
    def test_ula_fixed_seed(self, test_gaussian):
        batch_size = 32
        z_init = torch.randn(batch_size, 1)
        test_gaussian.eta = 0.1
        test_gaussian.N = 20
        
        torch.manual_seed(42)
        z_final_1 = ula(test_gaussian, z_init.clone())
        
        torch.manual_seed(42)
        z_final_2 = ula(test_gaussian, z_init.clone())
        
        assert torch.allclose(z_final_1, z_final_2, atol=1e-6) 