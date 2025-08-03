import torch
from torch import nn, Tensor
from typing import Callable

class LambdaLayer(nn.Module):
    """
        Wrapper for arbitrary invertible functions in the flow.
        
        Useful for data preprocessing steps like reshaping, normalization, etc.
        
        Args:
            func: Forward transformation function
            inv_func: Inverse transformation function
    """
    def __init__(
        self,
        func: Callable[[Tensor], Tensor],
        inv_func: Callable[[Tensor], Tensor],
    ) -> None:
        super().__init__()
        self.func = func
        self.inv_func = inv_func

    def forward(self, x: Tensor, *args, **kwargs):
        """Apply forward transformation, preserving additional arguments."""
        y = self.func(x)
        if args or kwargs:
            return (y, *args) if not kwargs else (y, *args, kwargs)
        return y
    
    def inverse(self, x: Tensor) -> Tensor:
        """Apply inverse transformation."""
        return self.inv_func(x)

# Utility functions for data preprocessing
def vec_to_img(x: Tensor) -> Tensor:
    """Reshape flattened vector back to 28x28 image."""
    return x.view(-1, 28, 28)

def img_to_vec(x: Tensor) -> Tensor:
    """Flatten 28x28 image to vector."""
    return x.flatten(1)


def sigmoid_inverse(x: Tensor):
    """Logit function: inverse of sigmoid for unbounded latent space."""
    x = torch.clamp(x, 1e-6, 1 - 1e-6)  # Numerical stability
    return torch.log(x) - torch.log(1 - x)

def add_noise(x: Tensor):
    """Add uniform dequantization noise to discrete pixel values."""
    return x + torch.rand_like(x) / 256.0
