# Mathematics-Numerical_Linear_Algebra-37

## Background

Dynamical low-rank approximation evolves a fixed-rank factorisation of the solution in place of the full phase-space array. In the error bounds of such integrators, eps_r denotes a bound on the component of the vector field normal to the manifold of rank-r matrices.

## Problem

The Wigner equation gives a phase-space formulation of quantum mechanics and is the quantum analogue of the Boltzmann equation in classical kinetic theory. Its pseudo-differential operator is nonlocal in the wave vector and couples the spatial and momentum variables, so evaluating its tangent-space projection would ordinarily require reconstructing the full phase-space tensor at every step, which defeats the purpose of a dynamical low-rank approximation and leaves the method exposed to the curse of dimensionality.

Your task is to solve one concrete deterministic example of a dynamical low-rank pipeline for this equation. To solve this, describe the pipeline in the abstract. Use the following configuration:
- one spatial and one wave-vector dimension, hbar = m = omega = 1, and V(x) = m*omega*x^2/2
- Omega_x = Omega_k = [-10, 10), with Nx = Nk = 128 uniform points per direction, periodic, left endpoint included and right endpoint excluded
- all derivatives in x and k evaluated spectrally
- f0(x, k) = (1/pi) * exp(-(u - 1)^2 / 2 - 2*v^2), where u = cos(pi/5)*x - sin(pi/5)*k and v = sin(pi/5)*x + cos(pi/5)*k
- initial low-rank factors from the rank-12 truncated SVD of f0 on the grid
- rank r = 12, held fixed throughout
- final time T = 2, advanced in Nt = 600 uniform steps
- time integration by a fixed-rank low-rank splitting integrator that is robust to arbitrarily small retained singular values, in which each stage forms an explicit increment matrix and supplies it to the practical splitting calculation rather than integrating the factor differential equations
- stage construction: the two-stage explicit Runge-Kutta realization of that integrator whose local error is bounded by C(h^3 + h*eps_r), with constants independent of the smallest retained singular value
- tableau: among the admissible two-stage tableaux for that construction, the one whose first-stage field value enters the final increment with weight b1 = -1

The reasoning should make clear how the structure of this particular potential, the choice of truncation, and the stage construction determine the result. Support the reported value with quantitative evidence of its sensitivity to the time discretisation and to the stage construction. Your final answer must be a single number: the value of the reconstructed low-rank Wigner function at x = -1.25, k = 0.78125 at time T, to nine significant figures.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_difference_potential_matrix

Goal
----
Construct the central difference potential by sampling it on the tensor product of a spatial grid and a grid in the dual variable. The potential is supplied as a callable and must be evaluated directly at the shifted arguments. Raise ValueError for a non-callable potential, for grids that are not one-dimensional, for empty grids, for grids containing non-finite entries, and for a potential that does not broadcast elementwise over its argument.

```python
def difference_potential_matrix(V: "Callable[[np.ndarray], np.ndarray]",
                                x_grid: "np.ndarray",
                                y_grid: "np.ndarray") -> "np.ndarray":
    '''Sample the central difference potential on a tensor grid.

    Parameters
    ----------
    V : Callable[[np.ndarray], np.ndarray]
        Potential energy function, evaluated elementwise on a numpy array of
        arbitrary shape and returning an array of that same shape.
    x_grid : np.ndarray
        (Nx,) one-dimensional array of spatial nodes.
    y_grid : np.ndarray
        (Ny,) one-dimensional array of nodes in the dual variable.

    Returns
    -------
    D_V : np.ndarray
        (Nx, Ny) array of the central difference potential, float64.
    '''
    return D_V  # placeholder
```

### Step 2

02_separation_factors

Goal
----
Compute the separable expansion of a sampled central difference potential and return its factors packed as a single array. Determine the number of retained terms as the count of singular values strictly exceeding rtol times the largest one, distribute each retained singular value symmetrically as its square root between the two factors, and fix the residual sign freedom by requiring that, in each spatial factor column, the first entry whose magnitude exceeds 1e-12 times that column's largest magnitude be positive, applying the same flip to the matching dual factor column. Return the spatial factors stacked above the dual factors. Raise ValueError if the input is not a non-empty two-dimensional array, if it contains non-finite entries, or if rtol does not satisfy 0 < rtol < 1.

```python
def separation_factors(D_V: "np.ndarray", rtol: float = 1e-12) -> "np.ndarray":
    '''Separate a sampled central difference potential into paired factors.

    Parameters
    ----------
    D_V : np.ndarray
        (Nx, Ny) array of the central difference potential on a tensor grid.
    rtol : float
        Relative threshold on the singular values, 0 < rtol < 1.

    Returns
    -------
    factors : np.ndarray
        (Nx + Ny, R) array whose first Nx rows hold the spatial factors and
        whose remaining Ny rows hold the dual factors, float64.
    '''
    return factors  # placeholder
```

### Step 3

03_spectral_derivative_matrix

Goal
----
Build the differentiation matrix that acts on values sampled at n_points equispaced nodes covering one period, with the left endpoint included and the right endpoint excluded, so that the matrix returned differentiates the periodic interpolant of a real-valued function exactly. Raise ValueError if n_points is not an integer, if it is smaller than two, or if the period is not a finite positive real number.

```python
def spectral_derivative_matrix(n_points: int, period: float) -> "np.ndarray":
    '''Differentiation matrix for a periodic equispaced grid.

    Parameters
    ----------
    n_points : int
        Number of equispaced nodes over one period, at least 2.
    period : float
        Length of the periodic interval, finite and positive.

    Returns
    -------
    D : np.ndarray
        (n_points, n_points) differentiation matrix, float64.
    '''
    return D  # placeholder
```

### Step 4

04_initial_low_rank_state

Goal
----
Project sampled initial data onto the fixed-rank manifold and return the resulting factors packed as a single array. Retain the leading rank terms of the truncated singular value decomposition, keep the coefficient matrix as the diagonal matrix of retained singular values, and fix the sign freedom by requiring that, in each left factor column, the earliest entry whose magnitude is within a relative 1e-9 of that column's largest magnitude be positive, applying the same flip to the matching right factor column. Return the left factors stacked above the coefficient matrix, stacked above the right factors. Raise ValueError if f_grid is not a non-empty two-dimensional array, if it contains non-finite entries, if rank is not an integer lying between one and the smaller grid dimension, or if the retained singular value at the truncation is no larger than 1e-14 times the leading one.

```python
def initial_low_rank_state(f_grid: "np.ndarray", rank: int) -> "np.ndarray":
    '''Project sampled initial data onto the fixed-rank manifold.

    Parameters
    ----------
    f_grid : np.ndarray
        (Nx, Nk) array of the initial data sampled on the phase-space grid.
    rank : int
        Retained rank, satisfying 1 <= rank <= min(Nx, Nk).

    Returns
    -------
    state : np.ndarray
        (Nx + rank + Nk, rank) array holding the left factors in its first Nx
        rows, the coefficient matrix in the next rank rows, and the right
        factors in the remaining Nk rows, float64.
    '''
    return state  # placeholder
```

### Step 5

05_wigner_rhs

Goal
----
Evaluate the right-hand side of the Wigner equation (hbar = m = 1) at a packed low-rank state, for a potential supplied only through the separation factors of its central difference potential, and return it as a dense array on the phase-space grid. The state arrives packed as left factors, coefficient matrix and right factors stacked in that order. The factors arrive packed as returned by the separation step: the spatial factors sampled at the spatial nodes, stacked above the dual factors sampled at the nodes of the grid dual to the wave-vector grid. For Nk wave-vector nodes with spacing dk, the dual nodes are y_m = 2*pi*m/(Nk*dk) for the integers m from -floor(Nk/2) to Nk - floor(Nk/2) - 1, in increasing order. For even Nk the unpaired dual node m = -Nk/2 contributes nothing, and the returned array is the real part of the result. Take the spatial derivative with the supplied differentiation operator. Raise ValueError if the state is not two-dimensional, if the grids are not non-empty one-dimensional arrays, if the state's row count does not equal the spatial grid size plus the rank plus the wave-vector grid size, if the differentiation operator is not square and matched to the spatial grid, if the factors are not a two-dimensional array whose row count equals the spatial grid size plus the wave-vector grid size, or if any input contains non-finite entries.

```python
def wigner_rhs(state: "np.ndarray",
               d_x: "np.ndarray",
               x_grid: "np.ndarray",
               k_grid: "np.ndarray",
               factors: "np.ndarray") -> "np.ndarray":
    '''Evaluate the Wigner right-hand side at a low-rank state.

    Parameters
    ----------
    state : np.ndarray
        (Nx + r + Nk, r) packed state: left factors, coefficient matrix,
        right factors, stacked in that order.
    d_x : np.ndarray
        (Nx, Nx) differentiation operator for the spatial coordinate.
    x_grid : np.ndarray
        (Nx,) spatial nodes.
    k_grid : np.ndarray
        (Nk,) uniformly spaced wave-vector nodes.
    factors : np.ndarray
        (Nx + Nk, R) separation factors of the central difference potential:
        R spatial factors at the spatial nodes stacked above R dual factors
        at the dual nodes, in increasing order. R may be zero.

    Returns
    -------
    rhs : np.ndarray
        (Nx, Nk) dense right-hand side evaluated at the state, float64.
    '''
    return rhs  # placeholder
```

### Step 6

06_projector_splitting_step

Goal
----
Apply one practical projector-splitting calculation to a packed low-rank state given a prescribed increment array, and return the updated packed state in the same layout. Both orthonormal factorisations must be resolved to a unique convention by requiring the diagonal of the triangular factor to be non-negative, transferring any sign change to the orthonormal factor. Raise ValueError if the state or increment is not two-dimensional, if the state's row count does not equal the increment's row count plus the rank plus the increment's column count, if any input contains non-finite entries, or if either matrix being factorised has collapsed to rounding error, meaning its largest singular value is at most 1e-13 times the sum of the Frobenius norms of the two terms forming it, or has lost full column rank, meaning its smallest singular value is at most 1e-15 times its largest.

```python
def projector_splitting_step(state: "np.ndarray", increment: "np.ndarray") -> "np.ndarray":
    '''Apply one practical projector-splitting calculation.

    Parameters
    ----------
    state : np.ndarray
        (Nx + r + Nk, r) packed state: left factors, coefficient matrix,
        right factors, stacked in that order.
    increment : np.ndarray
        (Nx, Nk) prescribed increment of the ambient solution.

    Returns
    -------
    new_state : np.ndarray
        (Nx + r + Nk, r) updated packed state in the same layout, float64.
    '''
    return new_state  # placeholder
```

### Step 7

07_two_stage_tableau

Goal
----
Given the final weight assigned to the first stage, recover the tableau with that first weight among the admissible tableaux of the robust two-stage construction, and return it as the stage abscissa followed by the two final weights. A tableau is admissible only if its internal stage is evaluated inside the step it belongs to, strictly after the start of the step and no later than its end. Raise ValueError if b1 is not given as an integer or floating-point number (a string such as "0" must be rejected, not converted), if it is not finite, or if no admissible tableau has that first weight.

```python
def two_stage_tableau(b1: float) -> "np.ndarray":
    '''Recover the admissible two-stage tableau from its first final weight.

    Parameters
    ----------
    b1 : float
        Final weight assigned to the first stage.

    Returns
    -------
    tableau : np.ndarray
        (3,) array holding the stage abscissa, the first final weight and the
        second final weight, in that order, float64.
    '''
    return tableau  # placeholder
```

### Step 8

08_robust_psrk_step

Goal
----
Advance a packed low-rank state by one time step of the robust two-stage construction, using the tableau recovered from the supplied first-stage weight and the right-hand side of the preceding step for the potential given by the separation factors. Raise ValueError if the state is not two-dimensional, if the grids are not non-empty one-dimensional arrays, if the state's row count does not equal the spatial grid size plus the rank plus the wave-vector grid size, if the differentiation operator is not square and matched to the spatial grid, if h is not a finite positive real number, if any input contains non-finite entries, if the supplied weight admits no tableau, or if an evaluation of the right-hand side or an application of the practical splitting calculation raises.

```python
def robust_psrk_step(state: "np.ndarray",
                     d_x: "np.ndarray",
                     x_grid: "np.ndarray",
                     k_grid: "np.ndarray",
                     factors: "np.ndarray",
                     h: float,
                     b1: float) -> "np.ndarray":
    '''Advance a packed low-rank state by one two-stage step.

    Parameters
    ----------
    state : np.ndarray
        (Nx + r + Nk, r) packed state: left factors, coefficient matrix,
        right factors, stacked in that order.
    d_x : np.ndarray
        (Nx, Nx) differentiation operator for the spatial coordinate.
    x_grid : np.ndarray
        (Nx,) spatial nodes.
    k_grid : np.ndarray
        (Nk,) uniformly spaced wave-vector nodes.
    factors : np.ndarray
        (Nx + Nk, R) separation factors of the central difference potential,
        in the layout the right-hand-side step takes.
    h : float
        Time step, finite and positive.
    b1 : float
        Final weight assigned to the first stage.

    Returns
    -------
    new_state : np.ndarray
        (Nx + r + Nk, r) advanced packed state in the same layout, float64.
    '''
    return new_state  # placeholder
```

### Step 9

09_wigner_dlra_target

Goal
----
Run the complete pipeline end to end and return the single scalar it targets. Build the periodic phase-space grid from the node count and half-width, sample the harmonic difference potential on the spatial grid and on the grid dual to the wave-vector grid, as the right-hand-side step defines it, verify through the earlier steps that it has the separation rank the potential implies and that the recovered factors reproduce it, use those factors for the potential term throughout, confirm that the supplied first-stage weight admits a tableau, form the initial low-rank state from the rotated squeezed Gaussian, advance it by the requested number of uniform steps, and evaluate the reconstruction at the requested phase-space point. Raise ValueError if the node count, rank or step count is not a positive integer, if any real parameter is not finite, if the half-width or final time is not positive, if the supplied first-stage weight admits no tableau, if the difference potential does not have that separation rank, or if the evaluation point does not lie on the grid.

```python
def wigner_dlra_target(n_points: int,
                       half_width: float,
                       theta: float,
                       rank: int,
                       t_final: float,
                       n_steps: int,
                       b1: float,
                       x_eval: float,
                       k_eval: float) -> float:
    '''Run the full pipeline and return the reconstructed value at one point.

    Parameters
    ----------
    n_points : int
        Number of grid nodes per direction.
    half_width : float
        Half-width of the periodic interval, used for both coordinates.
    theta : float
        Rotation angle of the initial squeezed Gaussian.
    rank : int
        Retained rank, held fixed throughout.
    t_final : float
        Final time.
    n_steps : int
        Number of uniform time steps.
    b1 : float
        Final weight assigned to the first stage.
    x_eval : float
        Spatial coordinate of the evaluation point, lying on the grid.
    k_eval : float
        Wave-vector coordinate of the evaluation point, lying on the grid.

    Returns
    -------
    value : float
        Reconstructed quasi-distribution at the evaluation point and final
        time, as a native Python float.
    '''
    return value  # placeholder
```
