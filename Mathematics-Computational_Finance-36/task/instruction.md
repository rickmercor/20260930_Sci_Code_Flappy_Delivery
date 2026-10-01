# Deterministic numerical valuation benchmark

## Background

Mathematical models for derivative valuation often lead to parabolic equations whose numerical treatment must preserve important qualitative properties of the underlying continuous problem. European call valuation provides a useful setting because the terminal payoff is continuous but has a loss of smoothness at the strike, which can affect numerical differentiation and convergence. Structure-preserving discretization techniques can incorporate information from reduced differential problems when constructing numerical approximations of the full equation. Implicit formulations are particularly useful for problems in which stability over relatively large discretization steps is important. The numerical solution can be assessed against an analytical valuation when one is available, providing a deterministic measure of discretization error. Such comparisons also make it possible to study how faithfully a numerical construction reproduces the behavior of the underlying continuous model.

## Problem

European call valuation under constant coefficients leads to a parabolic pricing problem whose numerical treatment is sensitive to the nonsmooth payoff at the strike. Implement the deterministic numerical construction specified by the supplied reference work for the benchmark instance below, preserving the reference work’s treatment of the payoff, transformed computational formulation, and implicit structure-preserving discretization. The numerical calculation is performed on a uniform finite space-time mesh over the prescribed truncated asset-price domain and must use the reference construction rather than substituting a different option-pricing or finite-difference method. The benchmark parameters are `r = 0.035`, `sigma = 0.35`, `T = 1.1`, `K = 1.2`, `S_max = 8.4`, `epsilon = 0.07`, `M = 21`, and `N = 25`.

The required result is the maximum absolute discrepancy between the complete numerical valuation surface and the corresponding analytical reference valuation evaluated on the same mesh. Implement the supplied sub-problem interfaces as independent numerical stages and make the final stage compose those stages to obtain the requested scalar. The spatial coefficient rows use the benchmark convention that the first returned row corresponds to the first interior spatial node and subsequent rows follow the same left-to-right ordering.

Your response should explain the scientific interpretation of the reference construction and the main numerical decisions needed to obtain the deterministic result. Do not substitute a conventional finite-difference scheme, Monte Carlo valuation, or another numerical pricing method for the specified reference construction.

## Output format

Return the scientific reasoning inside `<reasoning>...</reasoning>` and the single deterministic numerical result inside `<final_answer>...</final_answer>`.

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

build_smoothed_payoff

Goal
----
Construct the spatial discretization and the corresponding regularized initial data required by the numerical formulation specified in the reference work. The construction must faithfully reproduce the reference treatment of the nonsmooth terminal payoff while remaining consistent with the supplied option parameters and computational domain.






The implementation should return the complete spatial grid together with the initial data evaluated on that grid. Preserve the ordering, numerical precision, and dimensionality required by the downstream numerical stages.

```python
def build_smoothed_payoff(K: float, S_max: float, epsilon: float, M: int) -> "np.ndarray":
    """Return the uniform asset grid and the reference regularized call payoff.

    Parameters
    ----------
    K : float
        Positive strike price.
    S_max : float
        Positive upper endpoint of the truncated asset domain.
    epsilon : float
        Positive half-width of the local payoff regularization.
    M : int
        Number of spatial intervals; must be at least 2.

    Returns
    -------
    "np.ndarray"
        A float64 array of shape `(M+1, 2)`. Column 0 is the uniform spatial grid in increasing order; column 1 is the regularized initial value at the corresponding node.

    Raises
    ------
    ValueError
        If an input violates the stated positivity or interval-count requirements.
    
    Notes
    -----
    The returned arrays are newly constructed and are not required to share storage with inputs.
    """
    return None
```

### Step 2

compute_nsfd_coefficients

Goal
----
Construct all discretization quantities required by the reference work for the interior spatial rows and the temporal evolution of the implicit numerical formulation.






The implementation must recover the paper-specific construction from the reference method rather than replacing it with a conventional finite-difference approximation. The returned quantities must be organized so that each spatial row corresponds consistently to the associated interior grid point and can be consumed directly by the subsequent linear-system construction.

```python
def compute_nsfd_coefficients(r: float, sigma: float, T: float, S_max: float, M: int, N: int) -> "np.ndarray":
    """Return the coefficient data required by the reference implicit discretization.

    Parameters
    ----------
    r : float
        Positive risk-free rate.
    sigma : float
        Positive volatility.
    T : float
        Positive time horizon.
    S_max : float
        Positive spatial-domain endpoint.
    M : int
        Number of spatial intervals; must be at least 2.
    N : int
        Number of time intervals; must be at least 1.

    Returns
    -------
    "np.ndarray"
        A float64 array of shape `(M-1, 4)`. Column 0 is the left coefficient, column 1 the center coefficient, column 2 the right coefficient, and column 3 the temporal quantity repeated on every row.

    Raises
    ------
    ValueError
        If an input violates the stated positivity or interval-count requirements, or if the reference coefficient construction becomes non-finite or singular.
    
    Notes
    -----
    The coefficient rows follow the positive interior-node convention stated in the task.
    """
    return None
```

### Step 3

assemble_nsfd_timestep

Goal
----
Construct the linear system associated with one implicit numerical update from the supplied previous solution and discretization data.






The construction must incorporate the spatial coefficients and the prescribed boundary information consistently with the reference formulation. This function is responsible only for forming the system representation required by the numerical solver; it must not perform the subsequent linear solve.

```python
def assemble_nsfd_timestep(previous: "np.ndarray", coefficients: "np.ndarray", S_max: float, K: float, r: float, t: float, M: int) -> "np.ndarray":
    """Assemble one interior implicit update system.

    Parameters
    ----------
    previous : numpy.ndarray
        Previous solution row with shape `(M+1,)`.
    coefficients : numpy.ndarray
        Coefficient table with shape `(M-1, 4)` produced by the coefficient stage.
    S_max, K, r, t : float
        Positive numerical/model parameters for the current update.
    M : int
        Number of spatial intervals; must be at least 2.

    Returns
    -------
    "np.ndarray"
        A float64 NumPy array of shape (M-1, 4) containing, in columns 0 to 3, the lower band, main diagonal, upper band, and right-hand-side values for the interior tridiagonal system. The lower band holds the sub-diagonal entries of interior rows 1 to M-2 in positions 0 to M-3, followed by a trailing zero; the upper band holds the super-diagonal entries of interior rows 0 to M-3 in positions 0 to M-3, followed by a trailing zero.

    Raises
    ------
    ValueError
        If shapes or scalar requirements are violated.
    
    Notes
    -----
    The returned bands are arranged for a standard tridiagonal solve of the interior unknowns.
    """
    return None
```

### Step 4

solve_nsfd_surface

Goal
----
Compute the complete numerical solution over the prescribed space-time mesh using the implicit structure-preserving discretization defined by the reference work.



The calculation must begin from the regularized initial state, repeatedly construct the appropriate implicit update, solve the resulting interior system, and enforce the required spatial boundary values at every time level. The returned object must represent the complete numerical solution rather than only the terminal state or interior unknowns.



This stage must be implemented through the preceding computational components rather than by substituting an unrelated numerical method.

```python
def solve_nsfd_surface(initial: "np.ndarray", coefficients: "np.ndarray", r: float, T: float, K: float, S_max: float, M: int, N: int) -> "np.ndarray":
    """Return the complete numerical surface generated by the reference implicit formulation.

    Parameters
    ----------
    initial : numpy.ndarray
        Regularized payoff table with shape `(M+1, 2)`; its second column contains the initial values.
    coefficients : numpy.ndarray
        Coefficient table with shape `(M-1, 4)` supplied by the coefficient stage.
    r, T, K, S_max : float
        Positive model and domain parameters.
    M : int
        Number of spatial intervals; must be at least 2.
    N : int
        Number of time intervals; must be at least 1.

    Returns
    -------
    numpy.ndarray
        Float64 surface of shape `(N+1, M+1)`.

    Raises
    ------
    ValueError
        If input shapes or scalar requirements are violated.
    
    Notes
    -----
    The initial row is the supplied regularized state; subsequent rows are generated from the supplied coefficient data.
    """
    return None
```

### Step 5

compute_black_scholes_reference

Goal
----
Evaluate the analytical reference valuation corresponding to the continuous model on the same space-time mesh used by the numerical solution.



The implementation must maintain the same parameter convention, temporal interpretation, spatial ordering, and mesh dimensions as the numerical calculation so that every analytical value corresponds to exactly one numerical node.

```python
def compute_black_scholes_reference(r: float, sigma: float, T: float, K: float, S_max: float, M: int, N: int) -> "np.ndarray":
    """Return the analytical reference surface on the benchmark mesh.

    Parameters
    ----------
    r, sigma, T, K, S_max : float
        Positive model and domain parameters.
    M : int
        Number of spatial intervals; must be at least 1.
    N : int
        Number of time intervals; must be at least 1.

    Returns
    -------
    numpy.ndarray
        Float64 reference surface of shape `(N+1, M+1)`.

    Raises
    ------
    ValueError
        If a scalar input violates the stated positivity or interval-count requirements.
    
    Notes
    -----
    The returned surface uses the same spatial and temporal indexing as the numerical surface.
    """
    return None
```

### Step 6

compute_max_nodal_error

Goal
----
Determine the global discrepancy between two solution surfaces defined on the same computational mesh.



The comparison must be performed node by node over the complete domain, without discarding boundary nodes or restricting the calculation to a selected region. The function should return the single scalar quantity specified by the benchmark's error definition.

```python
def compute_max_nodal_error(numerical: "np.ndarray", reference: "np.ndarray") -> float:
    """Return the maximum absolute nodal discrepancy.

    Parameters
    ----------
    numerical : numpy.ndarray
        Numerical surface.
    reference : numpy.ndarray
        Analytical reference surface.

    Returns
    -------
    float
        Maximum absolute difference between corresponding finite mesh values.

    Raises
    ------
    ValueError
        If the inputs are not finite non-empty two-dimensional arrays with equal shapes.
    
    Notes
    -----
    The inputs are treated as corresponding values on the same mesh.
    """
    return None
```

### Step 7

run_nsfd_benchmark

Goal
----
Execute the complete benchmark workflow for the fixed numerical instance specified by the task.



The final computation must be obtained by composing the preceding public stages in their intended dependency order: construction of the initial data, construction of the reference discretization quantities, formation and solution of the numerical evolution, construction of the analytical reference, and evaluation of the global nodal discrepancy.



Do not replace the preceding workflow with an independently hard-coded final calculation. The function represents the complete scientific experiment and must return only its requested deterministic result.

```python
def run_nsfd_benchmark(r: float, sigma: float, T: float, K: float, S_max: float, epsilon: float, M: int, N: int) -> float:
    """Return the complete deterministic benchmark error.

    Parameters
    ----------
    r, sigma, T, K, S_max, epsilon : float
        Positive benchmark parameters.
    M : int
        Number of spatial intervals; must be at least 2.
    N : int
        Number of time intervals; must be at least 1.

    Returns
    -------
    float
        The maximum absolute nodal discrepancy produced by the complete composed workflow.

    Raises
    ------
    ValueError
        If the supplied benchmark parameters violate the contracts of the preceding stages.
    
    Notes
    -----
    The final scalar is obtained by composing the preceding public numerical stages.
    """
    return None
```
