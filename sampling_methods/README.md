# Sampling Methods

Implementation of fundamental statistical sampling algorithms including Box-Muller transformation, rejection sampling, Markov Chain Monte Carlo methods, and Langevin dynamics.

## Overview

This module contains from-scratch implementations of core Monte Carlo methods and sampling techniques. Each algorithm is implemented with mathematical derivations and visualizations for learning purposes.

## Implemented Algorithms

### Box-Muller Transform (`box_muller.ipynb`)

Generates normally distributed samples from uniform random variables using polar coordinate transformation:

- **Polar method**: Unit circle rejection sampling with transformation $Z = V\sqrt{\frac{-2 \ln(R)}{R}}$
- **Validation**: Comparison against scipy's normal distribution
- **Visualization**: Histogram overlay with theoretical normal PDF

### Rejection Sampling (`rejection_sampling.ipynb`)

General-purpose sampling for arbitrary probability distributions using acceptance-rejection:

- **Target distribution**: f(x) = sin(x)/2 on [0,π]
- **Proposal**: Uniform distribution with envelope function
- **Validation**: Comparison with analytical inverse CDF method
- **Efficiency**: Demonstrates acceptance rate mechanics

### Markov Chain Monte Carlo (`markov_chain_monte_carlo.ipynb`)

MCMC implementations with Metropolis and Metropolis-Hastings algorithms:

**Target Distributions:**
- Rhombus distribution: uniform on {(x,y): |x| + |y| ≤ 1}
- Gaussian mixture models with analytical gradients

**Proposal Mechanisms:**
- Symmetric proposals: Normal distributions with various scales (σ=0.5, 1.0, 1.5)
- Symmetric proposals: Uniform distributions with various ranges (±0.5, ±1.0, ±1.5)
- Asymmetric proposal: Mixed local-global uniform distribution using Metropolis-Hastings

**Analysis:**
- Acceptance rate comparison across proposal types
- Visual validation against target distributions

### Langevin Dynamics (`langevin_sampling.ipynb`)

Gradient-based MCMC sampling using score function information:

- **Implementation**: Discrete SDE with analytical gradient computation
- **Update rule**: $x_{t+1} = x_t + \frac{\varepsilon}{2} \cdot \nabla \log p(x_t) + \sqrt{\varepsilon} \cdot \eta$
- **Target**: Gaussian mixture model with components N(3, 1.5²) and N(-1, 1²)
- **Visualization**: Interactive animation showing sample evolution over iterations

## Usage

Navigate to the project root and launch the scientific computing container:

```bash
# Using Docker (recommended)
./create_scipy_container.cmd

# Or install locally
pip install numpy scipy matplotlib jupyter tqdm ipympl
```

Then access the notebooks:

```bash
cd sampling_methods/src
jupyter notebook
```

Each notebook is self-contained with complete implementations.


## Key Features

- **From-scratch implementations** with mathematical derivations and documented equations
- **Visualization** of sample distributions against theoretical PDFs
- **Educational focus** with clear mathematical foundations


## Requirements

```bash
pip install numpy scipy matplotlib jupyter tqdm ipympl
```

The `ipympl` package enables interactive matplotlib animations within Jupyter notebooks.

## References

1. Box, G. E. P., & Muller, M. E. (1958). A note on the generation of random normal deviates. *The Annals of Mathematical Statistics*, 29(2), 610-611.

2. Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of state calculations by fast computing machines. *The Journal of Chemical Physics*, 21(6), 1087-1092.

3. Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1), 97-109.

4. Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341-363.