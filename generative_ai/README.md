# Generative AI Models

This module contains implementations of two core generative modeling approaches:

- **Energy-Based Models (EBMs)** with Langevin MCMC sampling
- **Normalizing Flows** using affine coupling layers

Full training pipelines and visualizations are available in the Jupyter notebooks (`energy_based_model.ipynb`, `normalized_flow.ipynb`).

## Model Overview

### Energy-Based Models (`src/ebm/`)

EBMs learn an energy function where lower energy corresponds to higher data probability.

- **Dataset**: MNIST (digit 0 only)
- **Architecture**: 3-layer MLP (784 → 512 → 512 → 1)
- **Sampling**: 60-step Langevin chains with a persistent buffer
- **Loss**: Contrastive divergence with L2 regularization
- **Implementation**:
    - `buffer.py`: Replay buffer for persistent MCMC chains
    - `langevin_sampler.py`: MCMC sampling using Langevin dynamics
    - `trainer.py`: Contrastive divergence training with early stopping

### Normalizing Flows (`src/flows/`)

Normalizing flows learn invertible transformations that map data to a simple base distribution (e.g., Gaussian).

- **Dataset**: Fashion-MNIST (T-shirts/tops only)
- **Architecture**: 6 coupling layers with MLP-based scale/shift networks
- **Preprocessing**: Dequantization and logit transformation
- **Loss**: Negative log-likelihood
- **Implementation**:
    - `coupling_layers.py`: Affine transformations that split input and apply learned scale/shift functions
    - `flow_models.py`: Sequential composition of invertible layers

## Requirements

```bash
# Local installation
pip install torch torchvision matplotlib tqdm

# Docker (GPU support)
../create_GPU_pytorch_container.cmd
```


 <!--
## References

1. Dinh, L., Krueger, D., & Bengio, Y. (2014). NICE: Non-linear independent components estimation. *arXiv preprint arXiv:1410.8516*.

2. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2016). Density estimation using real NVP. *arXiv preprint arXiv:1605.08803*.

3. Du, Y., & Mordatch, I. (2019). Implicit generation and modeling with energy based models. *Advances in Neural Information Processing Systems*, 32.

4. Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *Proceedings of the 28th International Conference on Machine Learning*.

->
