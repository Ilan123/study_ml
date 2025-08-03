# Normalizing Flows Mathematical Documentation

## Coupling Layer

The affine coupling layer implements the following transformations:

### Forward transformation
$$z_B = \exp\left(-s(z_A)\right) \odot \left(x_B - b(z_A)\right)$$

### Inverse transformation  
$$x_B = \exp\big(s(z_A, w)\big) \odot z_B + b(z_A, w)$$

### Jacobian matrix
$$J = \begin{bmatrix}
I_d & 0 \\
\frac{\partial z_B}{\partial x_A} & \mathrm{diag}\big(\exp(-s)\big)
\end{bmatrix}$$

See implementation in `coupling_layers.py::CouplingLayer`