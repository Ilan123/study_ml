import torch
from torch import nn, Tensor
from typing import Tuple


class CouplingLayer(nn.Module):
    """
        Affine coupling layer for normalizing flows.
        
        Splits input into two parts [x_A, x_B], keeps x_A unchanged as z_A,
        and transforms x_B using learned scale s(z_A) and shift b(z_A) functions:
        
        Forward:  z_B = exp(-s(z_A)) ⊙ (x_B - b(z_A))
        Inverse:  x_B = exp(s(z_A)) ⊙ z_B + b(z_A)
        
        Args:
            split_at: Dimension at which to split the input tensor
            scale_net: Neural network computing log-scale parameters s(z_A)  
            shift_net: Neural network computing shift parameters b(z_A)
            alternate_parts: If True, transform first half instead of second half
    """
    def __init__(
        self,
        split_at: int,
        scale_net: nn.Module, # s
        shift_net: nn.Module, # b
        alternate_parts: bool = False
    ) -> None:
        super().__init__()
        self.split_at = split_at
        self.scale_net = scale_net
        self.shift_net = shift_net
        self.alternate_parts = alternate_parts
    

    def _split(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Split input tensor into two parts."""
        if self.alternate_parts:
            return x[:, self.split_at:], x[:, :self.split_at]
        else:
            return x[:, :self.split_at], x[:, self.split_at:]


    def _merge(self, xA: Tensor, xB: Tensor) -> Tensor:
        """Merge two tensor parts back together."""
        if self.alternate_parts:
            return torch.cat((xB, xA), dim=1)
        else:
            return torch.cat((xA, xB), dim=1)


    def _get_scale_and_shift(self, zA: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute scale and shift parameters from unchanged part."""
        log_scale = self.scale_net(zA)
        log_scale = torch.clamp(log_scale, min=-5, max=3)  # Stability
        shift = self.shift_net(zA)
        return log_scale, shift


    def forward(self, x: Tensor, log_det_total: Tensor) -> Tuple[Tensor, Tensor]:
        """
            Forward transformation: data → latent space.
            
            Returns:
                z: Transformed tensor
                log_det_total: Updated log determinant for probability computation
        """
        xA, xB = self._split(x)
        zA = xA
        log_scale, shift = self._get_scale_and_shift(zA)

        zB = torch.exp(-log_scale) * (xB - shift)
        z = self._merge(zA, zB)

        log_det_current = -torch.sum(log_scale, dim=1)
        log_det_total = log_det_total + log_det_current
        return z, log_det_total


    def inverse(self, z: Tensor) -> Tensor:
        """Inverse transformation: latent space → data space."""
        zA, zB = self._split(z)
        xA = zA
        log_scale, shift = self._get_scale_and_shift(zA)

        xB = torch.exp(log_scale) * zB + shift
        x = self._merge(xA, xB)
        return x