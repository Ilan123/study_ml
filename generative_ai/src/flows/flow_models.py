import torch
from torch import nn, Tensor
from typing import Tuple
from torch.distributions import MultivariateNormal


class FlowSequential(nn.Module):
    """
        Sequential container for normalizing flow layers.
        
        Chains multiple invertible transformations and manages the base distribution
        for sampling and likelihood computation.
        
        Args:
            data_dim: Dimensionality of the data space
            layers: List of invertible transformation modules
    """
    def __init__(self, data_dim: int, layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

        # Validate that all layers implement the required interface
        for i, module in enumerate(self.layers):
            if not hasattr(module, 'forward') or not hasattr(module, 'inverse'):
                raise TypeError(f"Module at index {i} must implement both 'forward' and 'inverse' methods.")
            
        # Base distribution (standard multivariate Gaussian)
        self.register_buffer('base_mean', torch.zeros(data_dim))
        self.register_buffer('base_cov', torch.eye(data_dim))

    def forward(
        self,
        x: torch.Tensor,
        log_det: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Transform data to latent space through the flow.
            
            Args:
                x: Input data tensor
                log_det: Running log determinant (initialized to zero if None)
                
            Returns:
                z: Latent representation
                log_det: Total log determinant of the transformation
        """
        if log_det is None:
            log_det = torch.zeros(len(x), device=x.device)
        for module in self.layers:
            x, log_det = module(x, log_det)
        return x, log_det

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Transform from latent space back to data space."""
        for module in reversed(self.layers):
            z = module.inverse(z)
        return z
    
    def __call__(
        self,
        x: torch.Tensor,
        log_det: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Optional: put extra logic here before/after forward
        return self.forward(x, log_det)
    
    def sample(self, num_samples: int, device:str=device) -> Tensor:
        """
            Generate samples by transforming base distribution samples.
            
            Args:
                num_samples: Number of samples to generate
                device: Device to generate samples on
                
            Returns:
                Generated samples in data space
        """
        base_dist = MultivariateNormal(self.base_mean, self.base_cov)
        z = base_dist.sample((num_samples,)).to(device)
        
        # Transform through inverse flow
        with torch.no_grad():
            x = self.inverse(z)
        
        return x
    
    def log_prob(self, x: Tensor):
        z, log_det = self.forward(x)
        
        base_dist = MultivariateNormal(self.base_mean, self.base_cov)
        log_prob_base = base_dist.log_prob(z)
        
        log_prob = log_prob_base + log_det
        return log_prob