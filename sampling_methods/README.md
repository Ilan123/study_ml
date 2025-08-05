# Sampling Methods

Implementation of fundamental statistical sampling algorithms in self-contained Jupyter notebooks.


## Implemented Algorithms

### Box-Muller Transform

Generates normally distributed samples from uniform random variables using a polar coordinate transformation.

### Rejection Sampling

General-purpose method for sampling from a target distribution  $p(x)$ using a proposal distribution $q(x)$ and an acceptance-rejection criterion.  
Demonstrated with $f(x) = \frac{\sin(x)}{2}$ over the interval $x \in [0, \pi]$

### Markov Chain Monte Carlo

MCMC implementations with Metropolis and Metropolis-Hastings algorithms on rhombus domain $\(|x| + |y| \leq 1\)$ and Gaussian mixture distributions.

### Langevin Dynamics

Gradient-based MCMC sampling using a discretized stochastic differential equation (SDE).

- **Update rule**: $x_{t+1} = x_t + \frac{\varepsilon}{2} \cdot \nabla \log p(x_t) + \sqrt{\varepsilon} \cdot \eta$

## Usage

Navigate to the project root and launch the scientific computing container:

```bash
# Using Docker (recommended)
./create_scipy_container.cmd

# Or install locally
pip install numpy scipy matplotlib jupyter tqdm ipympl
```

<!--
## References

1. Box, G. E. P., & Muller, M. E. (1958). A note on the generation of random normal deviates. *The Annals of Mathematical Statistics*, 29(2), 610-611.

2. Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of state calculations by fast computing machines. *The Journal of Chemical Physics*, 21(6), 1087-1092.

3. Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. *Biometrika*, 57(1), 97-109.

4. Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341-363.

-->
