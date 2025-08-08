import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

class KLAnnealer:
    """KL weight annealing scheduler from beta-VAE."""
    
    def __init__(
        self,
        total_steps: int,
        n_cycle: int = 10,
        ratio_increase: float = 0.01,
        ratio_zero: float = 0.01,
        max_kl_weight: float = 1.0,
        min_kl_weight: float = 0.0,
    ):
        """
        Initialize KL annealer with cyclic scheduling.
        
        Args:
            total_steps: Total number of training steps
            n_cycle: Number of annealing cycles
            ratio_increase: Ratio of cycle for increasing KL weight
            ratio_zero: Ratio of cycle for zero KL weight
            max_kl_weight: Maximum KL weight
            min_kl_weight: Minimum KL weight
        """
        self.total_steps = total_steps
        self.n_cycle = n_cycle
        self.ratio_increase = ratio_increase
        self.ratio_zero = ratio_zero
        self.max_kl_weight = max_kl_weight
        self.min_kl_weight = min_kl_weight
        
        self._schedule = self._compute_schedule()
    
    def _compute_schedule(self) -> np.ndarray:
        """KL weight schedule using frange_cycle_zero_linear."""
        schedule = np.ones(self.total_steps) * self.max_kl_weight
        period = self.total_steps / self.n_cycle
        step_size = (self.max_kl_weight - self.min_kl_weight) / (period * self.ratio_increase)
        
        for cycle in range(self.n_cycle):
            weight, step = self.min_kl_weight, 0
            while weight <= self.max_kl_weight and (int(step + cycle * period) < self.total_steps):
                idx = int(step + cycle * period)
                
                if step < period * self.ratio_zero:
                    schedule[idx] = self.min_kl_weight
                else:
                    schedule[idx] = weight
                    weight += step_size
                
                step += 1
        
        return schedule
    
    def get_weight(self, step: int) -> float:
        if step < len(self._schedule):
            return float(self._schedule[step])
        else:
            return self.max_kl_weight
    
    def get_cycle_progress(self, step: int) -> float:
        period = self.total_steps / self.n_cycle
        return (step % period) / period
    
    def get_phase(self, step: int) -> str:
        """Get the current schedule phase: 'zero', 'increasing', or 'max'."""
        cycle_progress = self.get_cycle_progress(step)
        
        if cycle_progress < self.ratio_zero:
            return "zero"
        elif cycle_progress < (self.ratio_zero + self.ratio_increase):
            return "increasing"
        else:
            return "max"
    
    def plot_schedule(self, save_path: Optional[str] = None):
        steps = np.arange(self.total_steps)
        weights = self._schedule
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, weights, 'b-', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('KL Weight')
        plt.title(f'Annealing Schedule ({self.n_cycle} cycles)')
        plt.grid(True, alpha=0.3)
        plt.ylim(self.min_kl_weight - 0.05, self.max_kl_weight + 0.05)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def __repr__(self) -> str:
        return (f"KLAnnealer(total_steps={self.total_steps}, n_cycle={self.n_cycle}, "
                f"ratio_increase={self.ratio_increase}, ratio_zero={self.ratio_zero}, "
                f"max_kl_weight={self.max_kl_weight})")
