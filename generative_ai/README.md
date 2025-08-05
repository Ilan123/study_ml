# Generative AI Models

Implementation of fundamental generative modeling techniques from first principles, focusing on Energy-Based Models and Normalizing Flows for image generation.

## Overview

This module contains implementations of two core generative modeling approaches:

- **Energy-Based Models (EBMs)** with Langevin MCMC sampling
- **Normalizing Flows** using affine coupling layers

The models were trained on the Fashion-MNIST and MNIST datasets, respectively, for clothing and digit generation tasks.

## Implementation Details

### Energy-Based Models (`src/ebm/`)

EBMs learn an energy function where lower energy corresponds to higher data probability. The implementation includes:

- **Buffer System** (`buffer.py`): Replay buffer for persistent MCMC chains
- **Langevin Sampler** (`langevin_sampler.py`): MCMC sampling using Langevin dynamics
- **Training Framework** (`trainer.py`): Contrastive divergence training with early stopping

The model learns to assign low energy to real data samples and higher energy to generated/random samples through contrastive divergence loss.

### Normalizing Flows (`src/flows/`)

Normalizing flows learn invertible transformations between data and a simple base distribution:

- **Coupling Layers** (`coupling_layers.py`): Affine coupling transformations that split input and apply learned scale/shift functions
- **Flow Models** (`flow_models.py`): Sequential composition of invertible layers with base distribution management

The flow transforms Fashion-MNIST images through a series of coupling layers to match a standard Gaussian distribution.

## Training Results

### EBM Training
- **Dataset**: MNIST (digit 0 only)
- **Architecture**: 3-layer MLP (784 → 512 → 512 → 1)
- **Sampling**: 60-step Langevin chains with persistent buffer
- **Loss**: Contrastive divergence with L2 regularization

### Flow Training  
- **Dataset**: Fashion-MNIST (T-shirts/tops only)
- **Architecture**: 6 coupling layers with MLP scale/shift networks
- **Preprocessing**: Dequantization and logit transformation
- **Loss**: Negative log-likelihood

## Usage

### Energy-Based Model
```python
from ebm.buffer import Buffer
from ebm.langevin_sampler import LangevinSampler
from ebm.trainer import EnergyModelTrainer
from models.mlp import MLP

# Create model and components
model = MLP([784, 512, 512, 1], device=device, flat=True)
buffer = Buffer(256, [1, 28, 28], 0.05, device)
sampler = LangevinSampler(model, 60, 10, 5e-3, buffer)

# Train
trainer = EnergyModelTrainer(model, 1e-1, 5e-3, sampler, device)
train_losses, val_losses = trainer.train(train_dataloader, val_dataloader)
```

### Normalizing Flow
```python
from flows.coupling_layers import CouplingLayer
from flows.flow_models import FlowSequential
from utils.training import train_flow

# Build flow model
flow_net = FlowSequential(784, [
    LambdaLayer(img_to_vec, vec_to_img),
    CouplingLayer(392, scale_net, shift_net),
    # ... additional layers
])

# Train
train_losses, val_losses = train_flow(
    flow_net, train_dataloader, val_dataloader, device
)
```

## Key Features

- **From-scratch implementations** of core generative modeling algorithms
- **Modular architecture** with reusable components
- **Comprehensive training** with validation, early stopping, and checkpointing
- **Visualization tools** for monitoring training progress and sample quality

## Requirements

```bash
# Local installation
pip install torch torchvision matplotlib tqdm

# Docker (GPU support)
../create_GPU_pytorch_container.cmd
```

See the Jupyter notebooks (`energy_based_model.ipynb`, `normalized_flow.ipynb`) for complete training examples and visualizations.


## References

1. Dinh, L., Krueger, D., & Bengio, Y. (2014). NICE: Non-linear independent components estimation. *arXiv preprint arXiv:1410.8516*.

2. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using real NVP. *arXiv preprint arXiv:1605.08803*.

3. Du, Y., & Mordatch, I. (2019). Implicit generation and modeling with energy based models. *Advances in Neural Information Processing Systems*, 32.

4. Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *Proceedings of the 28th International Conference on Machine Learning*.