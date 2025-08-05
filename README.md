# Machine Learning Research & Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A study-oriented repository for implementing machine learning and statistical methods.

## Repository Structure

```
study_ml/
├── generative_ai/          # Generative modeling implementations
├── sampling_methods/       # Statistical sampling algorithms
└── create_*_container.cmd  # Docker environment scripts
```

## Implemented Techniques

### Generative Models (`generative_ai/`)
- Energy-Based Models (EBMs) with Langevin MCMC sampling
- Normalizing Flows with affine coupling layers

### Statistical Sampling (`sampling_methods/`)
- Box-Muller transformation
- Rejection Sampling
- Markov Chain Monte Carlo methods
- Stochastic gradient Langevin dynamics

## Module Documentation

For detailed implementation guides, usage examples, and specific setup instructions:

- **[`generative_ai/README.md`](generative_ai/README.md)** - Detailed generative models documentation
- **[`sampling_methods/README.md`](sampling_methods/README.md)** - Statistical sampling algorithms documentation

## Quick Start

### Environment Setup

Installation requirements vary by module:

**For `generative_ai`:**
```bash
# Local installation
pip install torch torchvision matplotlib tqdm

# Docker (GPU support)
./create_GPU_pytorch_container.cmd
```

**For `sampling_methods`:**
```bash
# Local installation  
pip install numpy scipy matplotlib

# Docker (scientific computing)
./create_scipy_container.cmd
```
