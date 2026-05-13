# Principle of Least Action — Least Action Network (LAN)

A PyTorch implementation of a neural network architecture derived from the 
variational principle of least action, treating deep learning as a physical 
optimisation problem over a smooth energy manifold.

---

## Motivation

Standard neural networks minimise empirical loss without any constraint on the 
*path* taken through the solution space. This project applies Hamilton's 
principle — that a physical system evolves along the trajectory which minimises 
the action integral — to enforce smooth, energy-efficient state transitions 
between network layers.

The action functional is defined as:

$$S[\mathbf{x}] = \int_{t_0}^{t_1} \mathcal{L}(\mathbf{x}(t), \dot{\mathbf{x}}(t))\, dt$$

where $\mathcal{L} = T - V$ is the Lagrangian, $T$ is the kinetic energy of the 
state transition, and $V$ is the potential energy of the representation.

---

## Architecture

Each residual block enforces the Euler-Lagrange condition on the hidden state 
trajectory:

$$\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{\mathbf{x}}} - \frac{\partial \mathcal{L}}{\partial \mathbf{x}} = 0$$

In discrete form across layers $l$, this becomes the residual constraint:

$$\mathbf{x}_{l+1} = \mathbf{x}_l + \Delta t \cdot f_\theta(\mathbf{x}_l)$$

where $f_\theta$ is learned to satisfy the stationarity condition 
$\delta S = 0$.

### Loss Function

The total loss balances classification fidelity against action cost:

$$\mathcal{L}_{\text{total}} = \underbrace{H(y, \hat{y})}_{\text{cross entropy}} + \lambda \underbrace{\sum_{l=1}^{L} \|\mathbf{x}_{l+1} - \mathbf{x}_l\|^2}_{\text{action cost}}$$

The action cost term penalises large inter-layer state jumps, enforcing the 
least-action path through the network's latent space. $\lambda$ controls the 
trade-off between predictive accuracy and dynamical smoothness.

---

## Key Properties

- **Smooth state transitions** — the action penalty suppresses discontinuous 
  jumps in latent representation across depth
- **Energy-aware optimisation** — treats each forward pass as a physical 
  trajectory, not an arbitrary composition of functions
- **Variational regularisation** — the Euler-Lagrange constraint acts as a 
  principled alternative to dropout or weight decay

---

## Implementation

Built in PyTorch using a custom `nn.Module` with:

- Residual blocks enforcing the discrete Euler-Lagrange condition
- A composite loss function combining cross-entropy with the action integral 
  approximation
- Modular design allowing the action cost weight $\lambda$ to be tuned 
  independently

---

## Background

The principle of least action states that the true trajectory of a physical 
system between two states is the one for which the action $S$ is stationary 
($\delta S = 0$). Applied to deep networks, this reframes each layer transition 
as a timestep in a Hamiltonian system, connecting neural architecture design to 
classical mechanics and variational calculus.

---

*Original research and implementation by Catherine Earl, 2025. MIT License.*
