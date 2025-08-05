# Machine Learning Research & Implementation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A research-focused repository implementing fundamental machine learning algorithms and models from first principles, with emphasis on generative modeling and statistical sampling methods.

## Overview

This repository contains clean, well-documented implementations of core ML techniques designed for both research and educational purposes. All algorithms are implemented from scratch with mathematical rigor and modular architecture.

## Repository Structure

```
study_ml/
├── generative_ai/          # Generative modeling implementations
├── sampling_methods/       # Statistical sampling algorithms
└── create_*_container.cmd  # Docker environment scripts
```

## Implemented Techniques

### Generative Models (`generative_ai/`)
- **Energy-Based Models (EBMs)** with Langevin MCMC sampling
- **Normalizing Flows** with affine coupling layers
- Comprehensive training frameworks with early stopping and checkpointing

### Statistical Sampling (`sampling_methods/`)
- Core sampling algorithms and Monte Carlo methods
- Implementations of fundamental statistical techniques

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

See individual module READMEs for detailed setup instructions. Docker containers provide pre-configured environments - `create_GPU_pytorch_container.cmd` includes PyTorch with CUDA support, while `create_scipy_container.cmd` provides Jupyter with scientific computing stack.

### Exploring the Code

Each module contains self-contained implementations that can be studied and experimented with:

- **`generative_ai/`** - Complete implementations of EBMs and Normalizing Flows with training scripts
- **`sampling_methods/`** - Statistical sampling algorithms and Monte Carlo methods

See individual module READMEs for detailed documentation and usage examples.

## Key Features

- **Mathematical Rigor**: Implementations follow theoretical foundations closely
- **Modular Design**: Reusable components for rapid experimentation
- **Containerized Environments**: Reproducible setups with Docker support

## Module Documentation

- [`generative_ai/README.md`](generative_ai/README.md) - Detailed generative models documentation
- [`sampling_methods/README.md`](sampling_methods/README.md) - Statistical sampling methods


## License

MIT License - see [LICENSE](LICENSE) file for details.