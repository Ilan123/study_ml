import torch
from torch import nn, Tensor
from Buffer import AbstractBuffer
from typing import Tuple, Union
import math

class LangevinSampler:
    """
    Langevin MCMC sampler for energy-based models.
    
    Implements Langevin dynamics to sample from the distribution defined by an energy function.
    Uses gradient descent with noise to perform MCMC sampling.
    
    Args:
        model: Energy function (lower energy = higher probability)
        default_n_steps: Default number of Langevin steps
        default_step_size: Default step size for Langevin updates
        buffer: Optional replay buffer for persistent sampling
        clip_grad: Whether to clip gradients to [-0.03, 0.03]
        clip_sample: Whether to clip samples to [-1, 1]
    """
    def __init__(
        self,
        model: nn.Module,
        default_n_steps: int,
        default_step_size: float,
        default_noise_scale: float,
        buffer: AbstractBuffer = None,
        clip_grad: bool = True,
        clip_sample: bool = True,
    ):
        self._model = model
        self._default_n_steps = default_n_steps
        self._default_step_size = default_step_size
        self._default_noise_scale = default_noise_scale
        self._buffer = buffer
        self._clip_grad = clip_grad
        self._clip_sample = clip_sample
        self._was_param_freazed = {}


    def _freeze_model(self):
        """
        Freeze model parameters to prevent updates during sampling,
        while keeping original requires_grad states.
        """
        for p in self._model.parameters():
            self._was_param_freazed[p] = p.requires_grad
            p.requires_grad = False


    def _unfreeze_model(self):
        """Restore original requires_grad state of model parameters."""
        for p in self._model.parameters():
            p.requires_grad = self._was_param_freazed[p]


    def _compute_grads(self, x: Tensor):
        """
        Compute gradients of energy w.r.t. input x.      
        Returns:
            Gradients of energy function w.r.t. x
        """
        x.requires_grad_(True)
        e = self._model(x).sum()
        grad = torch.autograd.grad(e, x, grad_outputs=torch.ones_like(e))[0]
        if self._clip_grad:
            grad.clamp_(-0.03, 0.03)
        x.requires_grad_(False)
        return grad
    

    def _get_grad_and_noise(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        noise = torch.randn_like(x, device=x.get_device())
        grad = self._compute_grads(x).detach()
        return grad, noise
    

    def _get_n_steps_and_step_size(self, n_steps, step_size):
        """Get n_steps and step_size, using defaults if None."""
        if n_steps is None:
            n_steps = self._default_n_steps
        if step_size is None:
            step_size = self._default_step_size
        return n_steps, step_size
    
    def _get_noise_scale(self, noise_scale):
        if noise_scale is None:
            noise_scale = self._default_noise_scale
        return noise_scale
    
    def _get_device(self, device):
        if device is None:
            device = self._buffer._device
        return device

    def _perform_langevin_chain(
        self,
        x: Tensor,
        n_steps: Union[int, None] = None,
        step_size: Union[float, None] = None,
        noise_scale: Union[float, None] = None,
    ) -> Tensor:
        """
        Perform Langevin MCMC chain starting from x.
        
        Updates: x_{t+1} = x_t - step_size * ∇E(x_t) + sqrt(2*step_size) * noise
        
        Args:
            x: Initial sample
            n_steps: Number of Langevin steps
            step_size: Step size for updates
            
        Returns:
            Final sample after n_steps
        """
        n_steps, step_size = self._get_n_steps_and_step_size(n_steps, step_size)
        noise_scale = self._get_noise_scale(noise_scale)
        self._freeze_model()
        for _ in range(n_steps):
            grad, noise = self._get_grad_and_noise(x)
            x.data.add_(grad.data, alpha=-1 * step_size)
            x.data.add_(noise.data , alpha=math.sqrt(2 * noise_scale))
            if self._clip_sample:
                x.data.clamp_(-1, 1)
        self._unfreeze_model()
        return x


    def persist_sample(
        self,
        n: int,
        n_steps: Union[int, None] = None,
        step_size: Union[float, None] = None,
        noise_scale: Union[float, None] = None,
        device: Union[str, None] = None,
    ) -> Tensor:
        """Sample using persistent chains from replay buffer."""
        idx, sample = self._buffer.get_random_sample(n)
        sample = self._perform_langevin_chain(sample, n_steps, step_size, noise_scale).to(device)
        self._buffer.update_buffer(idx, sample)
        return sample


    def sample(
        self,
        n: int,
        n_steps: Union[int, None] = None,
        step_size: Union[float, None] = None,
        noise_scale: Union[float, None] = None,
        device: Union[str, None] = None,
    ) -> Tensor:
        """Generate samples from random initialization."""
        device = self._get_device(device)
        sample = torch.randn([n, *self._buffer.shape], device=device)
        sample = self._perform_langevin_chain(sample, n_steps, step_size, noise_scale).to(device)
        return sample

    
