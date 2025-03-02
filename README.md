# Trajectory optimization for simple rocket dynamics using direct collocation
Author: Marvin Ahlborn</br>
Date: 2025-03-02

<img src="https://github.com/MarvinAhl/rocket-traj-opt/blob/main/rocket.gif" alt="An animated gif showing three transitions between the equilibrium states." width="350"/>

Trapezoidal collocation is used to solve the optimal control problem of a
simple 2d rocket (see dynamics.mlx), only powered by one throttleable
thrust-vector engine. The resulting NLP was solved with fmincon and the
gradient of the objective, the jacobian of the constraints, and the
hessian of the lagrangian were supplied analytically.
