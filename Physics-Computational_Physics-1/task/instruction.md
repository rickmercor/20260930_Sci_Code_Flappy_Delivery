# Physics-Computational_Physics-1

## Background

The neutron transport equation describes how neutrons are distributed in position, direction, and energy as they stream, scatter, and are absorbed inside a medium. Solving it accurately matters for reactor design and radiation shielding, but it is numerically difficult because neutron behavior spans two very different physical regimes. In optically thin, weakly-interacting regions neutrons travel long distances between collisions and the transport is dominated by free streaming. In optically thick, strongly-scattering regions neutrons collide so frequently that their collective behavior is well approximated by a diffusion equation. A single problem can contain both regimes at once, in different spatial regions.

The two classical families of numerical methods each handle only one of these regimes well. Deterministic methods that discretize the angular variable directly are accurate and efficient in the diffusion-dominated regime but become expensive as the angular resolution needed for accuracy increases, especially in more than one dimension. Monte Carlo methods handle free-streaming transport naturally, since they simulate individual particle trajectories, but they converge slowly and produce noisy results in optically thick, diffusion-dominated regions, where an enormous number of collision events must be sampled to resolve the solution.

This tension has motivated hybrid numerical schemes that couple a deterministic treatment of the smooth, near-equilibrium part of the solution with a stochastic, particle-based treatment of the non-equilibrium part, adaptively shifting the balance between the two depending on the local degree of collisionality. Such multiscale schemes were originally developed for rarefied gas dynamics, where the same free-streaming-to-diffusion tension arises, and have since been extended to related kinetic transport problems, including radiative transfer and plasma physics. Extending this style of method to neutron transport, so that a single numerical scheme remains efficient and accurate across both optically thin and optically thick regions without switching methods or refining the angular grid, is an active area of computational reactor physics research.

## Problem

The neutron transport equation governs neutron phase-space distributions across regimes ranging from free streaming to diffusion. Classical deterministic solvers require angular discretization that is expensive in multiple dimensions, while Monte Carlo methods suffer from slow convergence in diffusion-dominated regions. Recent literature extends multiscale wave-particle schemes, previously developed for rarefied gas dynamics and radiative transfer, to the neutron transport equation; the scheme decomposes the neutron angular flux into deterministic and stochastic components within a finite-volume framework. The method constructs numerical fluxes from an integral solution of the kinetic equation and adaptively controls the number of stochastic particles based on the local optical thickness.

Consider the 1D single-group neutron transport equation with isotropic scattering on $x \in [0, 1]$ with periodic boundary conditions. After non-dimensionalization, the neutron velocity is $v = 1$ and the angular variable is $\mu \in [-1, 1]$. A spatially varying isotropic external source drives the system from zero initial conditions to steady state:

$$\phi(0, x, \mu) = 0, \quad x \in [0, 1], \quad \mu \in [-1, 1].$$

The external source is nonzero only in the central half of the domain:

$$q(x) = \begin{cases} 0.5 & \text{if } 0.25 \leq x < 0.75 \\ 0 & \text{otherwise} \end{cases}$$

Here $q(x)$ is an angular source density per unit $\mu$, independent of $\mu$: it enters the transport equation as $+q(x)$ in every direction, so its angular integral is $\int_{-1}^{1} q(x) \, \mathrm{d}\mu = 2q(x)$.

The macroscopic scalar flux is defined as

$$\psi(x) = \int_{-1}^{1} \phi(x, \mu) \, \mathrm{d}\mu.$$

### Cross-Section Parameters

| Quantity | Value |
|----------|-------|
| Scattering cross-section | $\Sigma_s = 10 \; \text{cm}^{-1}$ |
| Absorption cross-section | $\Sigma_a = 1 \; \text{cm}^{-1}$ |
| Total cross-section | $\Sigma = \Sigma_s + \Sigma_a = 11 \; \text{cm}^{-1}$ |
| Fission production cross-section | $\nu \Sigma_f = 0$ |

### Simulation Parameters

| Quantity | Value |
|----------|-------|
| Number of spatial cells | $N_x = 40$ |
| CFL number | $0.2$ |
| Initial target particles per cell (at $\psi = 1$) | $200$ |
| Initial number of time steps | $20{,}000$ |
| Averaging burn-in (iterations before averaging) | $5{,}000$ |

The characteristic collision time is $\tau = 1 / (v \Sigma)$. The spatial mesh is uniform with cell width $\Delta x = 1 / N_x$.

Using the wave-particle method described in the source paper, implement the solver for the 1D single-group model with periodic boundaries and time-march from zero initial conditions. Average the macroscopic scalar flux $\psi(x)$ over the post-burn-in iterations to obtain the steady-state solution.

Define the midpoint estimate at $x=0.5$ by linear interpolation between the two neighboring cell centers, which on this mesh is the arithmetic mean of the time-averaged fluxes in zero-based cells 19 and 20. The requested quantity is the sampling-converged value of this midpoint estimate, that is, its expectation over independent realizations of the scheme, with the particle count and averaging duration listed above as minimum settings.

Output as your final answer this converged midpoint estimate, rounded to two significant figures.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

reconstruct_interfaces

Goal
----
Implement reconstruct_interfaces to compute slope-limited left and right
values of a cell-averaged quantity at all periodic cell interfaces.

```python
import numpy as np

def reconstruct_interfaces(
    psi: np.ndarray, dx: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    psi : numpy.ndarray
        Cell-averaged values, shape (N_x,).
    dx : float
        Uniform cell width.

    Returns
    -------
    result : tuple[numpy.ndarray, numpy.ndarray]
        (psi_L, psi_R) each of shape (N_x,).
        psi_L[j] is the value reconstructed from the left cell at interface j.
        psi_R[j] is the value reconstructed from the right cell at interface j.

    Raises
    ------
    ValueError
        If dx is not positive.
    """
    return psi_L, psi_R
```

### Step 2

compute_macroscopic_flux

Goal
----
Implement compute_macroscopic_flux to evaluate the deterministic macroscopic
numerical flux H^{ma} at every periodic cell interface for the one-dimensional
single-group neutron transport equation with isotropic scattering.

```python
import numpy as np

def compute_macroscopic_flux(
    psi: np.ndarray,
    sigma_s: float,
    sigma: float,
    dx: float,
    dt: float,
    tau: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    psi : numpy.ndarray
        Cell-averaged macroscopic scalar flux, shape (N_x,).
    sigma_s : float
        Macroscopic scattering cross-section.
    sigma : float
        Total macroscopic cross-section.
    dx : float
        Uniform cell width.
    dt : float
        Time step.
    tau : float
        Characteristic collision time, 1 / (v * Sigma).

    Returns
    -------
    H_ma : numpy.ndarray
        Macroscopic numerical flux at each periodic interface, shape (N_x,).

    Raises
    ------
    ValueError
        If dx, dt or tau is not positive.
    """
    return H_ma
```

### Step 3

classify_particles

Goal
----
Implement classify_particles to sample the free-transport time of each
particle and classify it as collisional or collisionless.

```python
import numpy as np

def classify_particles(
    tau: float, dt: float, rng: np.random.Generator, n_particles: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    tau : float
        Characteristic collision time, tau = 1 / (v * Sigma).
    dt : float
        Time step size.
    rng : numpy.random.Generator
        Random number generator. Draw one uniform survival probability per
        particle from rng.random in particle order.
    n_particles : int
        Number of particles to classify.

    Returns
    -------
    result : tuple[numpy.ndarray, numpy.ndarray]
        (t_f, is_collisionless) where t_f has shape (n_particles,) giving the
        free transport time for each particle (capped at dt), and
        is_collisionless is a boolean array (True if t_f == dt).

    Raises
    ------
    ValueError
        If n_particles < 0, or tau or dt is not positive.
    """
    return t_f, is_collisionless
```

### Step 4

free_transport_step

Goal
----
Implement free_transport_step to stream particles along their characteristic
lines and compute their microscopic numerical flux at periodic cell interfaces.

```python
import numpy as np

def free_transport_step(
    x_p: np.ndarray,
    xi_p: np.ndarray,
    w_p: np.ndarray,
    dx: float,
    N_x: int,
    dt: float,
    t_f: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    x_p : numpy.ndarray
        Particle positions, shape (n_particles,).
    xi_p : numpy.ndarray
        Particle velocities xi = v * mu, shape (n_particles,). Range [-1, 1]
        for v = 1.
    w_p : numpy.ndarray
        Per-particle mass, shape (n_particles,).
    dx : float
        Cell width.
    N_x : int
        Number of cells.
    dt : float
        Time step size.
    t_f : numpy.ndarray
        Free transport time for each particle, shape (n_particles,).

    Returns
    -------
    result : tuple[numpy.ndarray, numpy.ndarray]
        (x_new, H_mi_free) where x_new is the updated particle positions
        (wrapped to [0, 1)), and H_mi_free is the microscopic flux at each
        periodic interface, shape (N_x,).

    Raises
    ------
    ValueError
        If N_x < 1, or dx or dt is not positive.
    """
    return x_new, H_mi_free
```

### Step 5

macroscopic_update

Goal
----
Implement macroscopic_update to advance the macroscopic scalar flux by one
UGKWP time step, incorporating macroscopic, microscopic, and collisional flux
contributions along with the spatially varying external source.

```python
import numpy as np

def macroscopic_update(
    psi: np.ndarray,
    H_ma: np.ndarray,
    H_mi_free: np.ndarray,
    psi_ma: np.ndarray,
    sigma: float,
    sigma_s: float,
    q_arr: np.ndarray,
    dx: float,
    dt: float,
    tau: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    psi : numpy.ndarray
        Current cell-averaged scalar flux psi^n, shape (N_x,).
    H_ma : numpy.ndarray
        Macroscopic flux from Step 2, shape (N_x,).
    H_mi_free : numpy.ndarray
        Free-transport microscopic flux from Step 4, shape (N_x,).
    psi_ma : numpy.ndarray
        Macroscopic remainder psi - psi_mi formed at the end of the previous
        step (the full remainder, before the collisionless fraction
        exp(-dt / tau) is taken for resampling), shape (N_x,).
    sigma : float
        Total macroscopic cross-section.
    sigma_s : float
        Scattering cross-section.
    q_arr : numpy.ndarray
        Spatially varying external source, shape (N_x,).
    dx : float
        Cell width.
    dt : float
        Time step.
    tau : float
        Characteristic collision time.

    Returns
    -------
    psi_new : numpy.ndarray
        Updated scalar flux psi^{n+1}, shape (N_x,).

    Raises
    ------
    ValueError
        If dx, dt or tau is not positive.
    """
    return psi_new
```

### Step 6

resample_particles

Goal
----
Implement resample_particles to create new particles from the updated
macroscopic distribution after a UGKWP time step.

```python
import numpy as np

def resample_particles(
    psi_ma: np.ndarray,
    dt: float,
    tau: float,
    dx: float,
    m_e: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    psi_ma : numpy.ndarray
        Macroscopic part of the scalar flux, shape (N_x,).
    dt : float
        Nonnegative time step.
    tau : float
        Positive characteristic collision time.
    dx : float
        Positive cell width.
    m_e : float
        Positive target mass per particle.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    result : tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        (x_resamp, xi_resamp, w_resamp) where x_resamp and xi_resamp are the
        positions and velocities of resampled particles, and w_resamp is the
        per-particle mass for each resampled particle.

    Raises
    ------
    ValueError
        If dt is negative, or tau, m_e or dx is not positive.

    Notes
    -----
    Random draws must follow this exact order for reproducibility: cells are
    processed in ascending order, and for each particle one uniform position
    draw is taken first, then one uniform velocity draw on [-1, 1].
    """
    return x_resamp, xi_resamp, w_resamp
```

### Step 7

solve_ugkwp_steady

Goal
----
Implement solve_ugkwp_steady to compute the steady-state macroscopic scalar
flux for the one-dimensional neutron transport equation with isotropic
scattering and periodic boundary conditions using the UGKWP method.

```python
import numpy as np

def solve_ugkwp_steady(
    sigma_s: float,
    sigma_a: float,
    q_arr: np.ndarray,
    N_x: int,
    CFL: float,
    n_ppc: int,
    max_iter: int,
    avg_start: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Parameters
    ----------
    sigma_s : float
        Scattering cross-section.
    sigma_a : float
        Absorption cross-section.
    q_arr : numpy.ndarray
        Spatially varying external source, shape (N_x,).
    N_x : int
        Number of spatial cells.
    CFL : float
        CFL number in (0, 1).
    n_ppc : int
        Target number of particles per cell.
    max_iter : int
        Maximum number of time steps.
    avg_start : int
        Iteration to begin Welford averaging.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    psi_avg : numpy.ndarray
        Time-averaged macroscopic scalar flux at cell centers, shape (N_x,).

    Raises
    ------
    ValueError
        If N_x < 1, CFL is not positive, n_ppc < 1, or avg_start is not in [0, max_iter).

    Notes
    -----
    Preserve particle ordering for reproducibility. Existing surviving particles retain their relative order, followed by newly resampled particles in generation order. Whenever classification is performed, draw one uniform survival probability per classified particle using rng.random in particle order. Classification draws precede resampling draws in each iteration. Resampling draws are taken cell by cell in ascending order, with one position draw followed by one velocity draw per generated particle, as specified in step 06.
    """
    return psi_avg
```
