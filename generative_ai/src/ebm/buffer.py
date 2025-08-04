import torch
from torch import nn
from torch import Tensor
from abc import ABC, abstractmethod

def choice(a, size, device='cpu'):
    return torch.randperm(a, device=device)[:size]


class AbstractBuffer(ABC):
    def __init__(
            self,
            size: int,
            shape: torch.Size,
            reinit_prob: float,
            device: str
        ):
        self._size = size
        self.shape = shape
        self._reinit_prob = reinit_prob
        self._device = device
        self._buffer = torch.empty([size, *shape], device=device)
    
    def _reinit(self, sample: Tensor) -> Tensor:
        mask = torch.rand(sample.size(0), device=self._device) < self._reinit_prob
        sample[mask] = torch.randn_like(sample[mask], device=self._device)
        return sample

    @abstractmethod
    def get_random_sample(self, n: int, random_init: bool = True) -> tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def update_buffer(self, idx: Tensor, sample: Tensor):
        pass


class Buffer(AbstractBuffer):
    """
    A replay buffer for storing and sampling tensor data with optional reinitialization.
    
    Args:
        size: Maximum number of samples to store in the buffer.
        shape: Shape of individual samples (excluding batch dimension).
        reinit_prob: Probability of reinitializing samples during retrieval.
        device: Device to store tensors on ('cpu' or 'cuda').
    """
    def __init__(self, size: int, shape: torch.Size, reinit_prob: float, device: str):
        super().__init__(size, shape, reinit_prob, device)
        # Initialize the buffer with random samples eagerly
        self._buffer = torch.randn([size, *shape], device=device)

    
    def get_random_sample(self, n: int, random_init: bool = True) -> Tensor:
        """
        Retrieve random samples from the buffer.
        
        Args:
            n: Number of samples to retrieve.
            random_init: Whether to apply random reinitialization to samples.
            
        Returns:
            Tuple of (indices, samples) where indices are the buffer positions
            and samples are the retrieved tensors.
        """
        idx = choice(self._size, n, self._device)
        sample = self._buffer[idx].detach().clone()

        if random_init:
            sample = self._reinit(sample)
        return idx, sample

    
    def update_buffer(self, idx: Tensor, sample: Tensor):
        """Update buffer at specified indices with new samples.
        
        Args:
            idx: Indices in the buffer to update.
            sample: New sample values to store at the given indices.
        """
        self._buffer[idx] = sample.detach().clone()



class BufferLazy(AbstractBuffer):
    """
    A memory-efficient buffer with lazy initialization and stochastic refresh.
    
    Maintains a fixed-size tensor buffer where slots are initialized on-demand
    when first accessed or updated. Supports probabilistic sample refreshing.
    
    Args:
        size: Maximum number of samples to store in the buffer.
        shape: Shape of individual samples (excluding batch dimension).
        refresh_prob: Probability of reinitializing samples during retrieval.
        device: Device to store tensors on ('cpu' or 'cuda').
    """
    def __init__(self, size: int, shape: torch.Size, reinit_prob: float, device: str):
        super().__init__(size, shape, reinit_prob, device)
        self._buffer = torch.empty([size, *shape], device=device)
        self._non_init_flags = torch.ones(size, dtype=torch.bool, device=device)

    
    def get_random_sample(self, n: int, random_init: bool = True) -> Tensor:
        """
        Retrieve random samples from the buffer, initializing unset slots on-demand.
        
        Args:
            n: Number of samples to retrieve.
            random_init: Whether to apply random reinitialization to samples.
            
        Returns:
            Tuple of (indices, samples) where indices are the buffer positions
            and samples are the retrieved tensors.
        """
        idx = choice(self._size, n, self._device)
        sample = self._buffer[idx].detach().clone()

        non_init_mask = self._non_init_flags[idx]
        if non_init_mask.any():
            sample[non_init_mask] = torch.randn_like(sample[non_init_mask])

        if random_init:
            sample = self._reinit(sample)
        return idx, sample
    
    def update_buffer(self, idx: Tensor, sample: Tensor):
        """
        Update buffer at specified indices and mark slots as initialized.
        
        Args:
            idx: Indices in the buffer to update.
            sample: New sample values to store at the given indices.
        """
        self._buffer[idx] = sample.detach().clone()
        self._non_init_flags[idx] = False

