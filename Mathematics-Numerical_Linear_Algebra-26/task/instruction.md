# Mathematics-Numerical_Linear_Algebra-26

## Background

Dynamical low-rank approximation evolves large matrix-valued systems on a fixed-rank manifold, reducing memory use and computational cost. For nonlinear dynamics, interpolatory projections based on DEIM or QDEIM estimate the projected velocity from selected matrix entries, allowing projected Runge–Kutta methods to avoid forming the full matrix field while retaining their convergence order.

## Problem

Dynamical low-rank approximation (DLRA) integrates large-scale matrix differential equations $\dot{A}(t) = F(A(t))$ on the manifold $\mathcal{M}_r$ of rank-$r$ matrices. For nonlinear velocity fields, evaluating the standard orthogonal tangent-space projection requires forming the full $n \times n$ matrix $F$, negating the cost savings of low-rank representation. Orthogonal projections may be replaced with oblique, data-sparse projections based on a DEIM index selection procedure, yielding a new class of projected Runge–Kutta integrators (PRK-DEIM) that retain the convergence order of their orthogonal counterparts while requiring only $O(nr^2)$ operations per projection.

You are given the discrete nonlinear Schrödinger equation on a $32 \times 32$ grid:
$$i\dot{A}(t) = -\tfrac{1}{2}\bigl(BA(t) + A(t)B\bigr) - \alpha \cdot A(t) \odot A(t) \odot A(t),$$
with $B = \operatorname{tridiag}(1,0,1) \in \mathbb{R}^{n \times n}$, elementwise (Hadamard) product $\odot$, $n = 32$, and $\alpha = 0.5$. The initial condition is
$$A_{j,k}(0) = \exp\!\Bigl(-\frac{(j-\mu_1)^2 + (k-\nu_1)^2}{\sigma^2}\Bigr) + \exp\!\Bigl(-\frac{(j-\mu_2)^2 + (k-\nu_2)^2}{\sigma^2}\Bigr)$$
with $\sigma = 0.15n$, $\mu_1 = 0.7n$, $\mu_2 = 0.4n$, $\nu_1 = 0.6n$, $\nu_2 = 0.3n$, for $j,k = 1,\ldots,n$.

Determine the number of digits of relative Frobenius-norm accuracy that the source's second-order projected integrator with its interpolatory index selection achieves at time $T = 0.51$, when compared against a full-order reference trajectory. Use classical RK4 with step size $h_0 = 10^{-4}$ as the reference integrator, pre-propagating from $t=0$ to $t_0 = 0.01$ before initializing the rank-$r$ approximation ($r = 4$, via truncated SVD of the pre-propagated state). Evolve the low-rank approximation from $t_0$ to $T$ using that integrator with step size $h = 10^{-3}$.

Report $\lfloor -\log_{10}(\lVert A_{\mathrm{ref}}(T) - Y_N \rVert_F \,/\, \lVert A_{\mathrm{ref}}(T) \rVert_F) \rfloor$ as a single integer, where $\lfloor \cdot \rfloor$ is the floor function.
In the reasoning also report the unrounded relative Frobenius error before the floor is applied, the reference norm $\lVert A_{\mathrm{ref}}(T)\rVert_F$, the four retained singular values of the truncated pre-propagated state together with the largest discarded one, the QDEIM index set selected at $t_0$, and the number of DLRA steps taken.

As a source-specific implementation certificate, apply the paper's two-stage Kronecker QDEIM construction: select rows from both factor bases, orthonormalize the sampled Kronecker core, select its reduced rows, and map those rows back to full C-order Kronecker indices. This certificate must use the paper's smallest-index tie rule and must complete before the matrix trajectory is scored.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_schrodinger_rhs

Goal
----
Implement schrodinger_rhs for a discrete nonlinear Schrödinger lattice.

Given the complex state matrix A, a same-sized square coupling matrix B, and a nonnegative real coefficient alpha, return the complex velocity field F(A) = dA/dt. Reject incompatible matrix shapes or a negative coefficient with ValueError.

```python
import numpy as np

def schrodinger_rhs(A: np.ndarray, B: np.ndarray, alpha: float) -> np.ndarray:
    """Evaluate the velocity field F(A) = dA/dt for the DNLS lattice equation.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Current matrix state (complex).
    B : np.ndarray, shape (n, n)
        Tridiagonal coupling matrix.
    alpha : float
        Nonlinearity parameter (must be >= 0).

    Returns
    -------
    F_A : np.ndarray, shape (n, n)
        Velocity field dA/dt.
    """
    F_A = np.zeros_like(A, dtype=complex)
    return F_A
```

### Step 2

02_rk4_integrate

Goal
----
Implement rk4_integrate for a matrix-valued ordinary differential equation.

Advance the initial matrix A0 from t0 to t_end with the classical fixed-step four-stage Runge-Kutta method and return the matrix at the final time. The routine is used for both the short pre-propagation and the full-order reference trajectory; reject a nonpositive step size or a reversed interval with ValueError.

```python
import numpy as np

def rk4_integrate(A0: np.ndarray, rhs_fn, t0: float, t_end: float, h: float) -> np.ndarray:
    """Integrate a matrix ODE using the classical four-stage Runge-Kutta method.

    Parameters
    ----------
    A0 : np.ndarray, shape (n, n)
        Initial matrix state.
    rhs_fn : callable
        Right-hand side function F(A) returning an (n, n) array.
    t0 : float
        Start time.
    t_end : float
        End time.
    h : float
        Step size (must be strictly positive).

    Returns
    -------
    A : np.ndarray, shape (n, n)
        Matrix state at t_end.
    """
    A = A0.copy()
    return A
```

### Step 3

03_qdeim_select

Goal
----
Implement qdeim_select for a real or complex orthonormal basis.

For an m-by-r matrix U with r <= m, return the r selected row indices in the exact QDEIM pivot order so that U[indices, :] defines the interpolation submatrix. The selection procedure must be recovered from the QDEIM literature; reject nonmatrix or incompatible inputs with ValueError.

```python
import numpy as np

def qdeim_select(U: np.ndarray) -> np.ndarray:
    """Select r row indices from an orthonormal matrix following the QDEIM
    procedure from the literature.

    Parameters
    ----------
    U : np.ndarray, shape (m, r)
        Matrix with orthonormal columns.

    Returns
    -------
    indices : np.ndarray, shape (r,), dtype int
        Selected row indices in pivoting order.
    """
    indices = np.array([], dtype=int)
    return indices
```

### Step 4

04_oblique_project

Goal
----
Implement oblique_project for a rank-r matrix state.

Given orthonormal factors U and V of Y = U Sigma V* and a same-sized matrix Z, return the complex oblique projection of Z onto the tangent space at Y. The data-sparse interpolation formula must be recovered from the literature on interpolatory dynamical low-rank approximation.

```python
import numpy as np

def oblique_project(U: np.ndarray, V: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Compute the oblique tangent-space projection P_Y^angle[Z] using DEIM
    interpolation indices.

    Parameters
    ----------
    U : np.ndarray, shape (n, r)
        Left orthonormal factor of Y = U Sigma V*.
    V : np.ndarray, shape (n, r)
        Right orthonormal factor of Y = U Sigma V*.
    Z : np.ndarray, shape (n, n)
        Matrix to project.

    Returns
    -------
    P_Z : np.ndarray, shape (n, n)
        Oblique projection of Z onto T_Y M_r.
    """
    P_Z = np.zeros_like(Z, dtype=complex)
    return P_Z
```

### Step 5

05_prk2_qdeim_step

Goal
----
Implement one prk2_qdeim_step on the rank-r matrix manifold.

Advance Y_i = U diag(s) Vh by one positive step h under rhs_fn and return the updated factors (U_new, s_new, Vh_new) at the requested rank. Recover the stage structure, coefficients, oblique projections, and rank retractions from the literature on second-order PRK-DEIM integration.

```python
import numpy as np
from typing import Tuple

def prk2_qdeim_step(U: np.ndarray, s: np.ndarray, Vh: np.ndarray, rhs_fn, h: float, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform one PRK2-QDEIM time step on the rank-r manifold.

    Parameters
    ----------
    U : np.ndarray, shape (n, r)
        Left orthonormal factor of Y_i.
    s : np.ndarray, shape (r,)
        Singular values of Y_i.
    Vh : np.ndarray, shape (r, n)
        Right factor (conjugate transpose of V) of Y_i.
    rhs_fn : callable
        Velocity field F(A), maps (n, n) -> (n, n).
    h : float
        Step size.
    r : int
        Target rank for truncation.

    Returns
    -------
    U_new : np.ndarray, shape (n, r)
    s_new : np.ndarray, shape (r,)
    Vh_new : np.ndarray, shape (r, n)
    """
    n = U.shape[0]
    U_new = np.zeros((n, r), dtype=complex)
    s_new = np.zeros(r)
    Vh_new = np.zeros((r, n), dtype=complex)
    return (U_new, s_new, Vh_new)
```

### Step 6

06_prk2_qdeim_integrate

Goal
----
Implement prk2_qdeim_integrate over a fixed time interval.

Starting from Y_0 = U0 diag(s0) Vh0, repeatedly call the earlier prk2_qdeim_step routine to advance from t0 to t_end and return the rank-r factors at the final time. Reject a nonpositive step size or a reversed interval with ValueError.

```python
import numpy as np
from typing import Tuple

def prk2_qdeim_integrate(U0: np.ndarray, s0: np.ndarray, Vh0: np.ndarray, rhs_fn, t0: float, t_end: float, h: float, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate a low-rank ODE from t0 to t_end using PRK2-QDEIM.

Parameters
----------
U0 : np.ndarray, shape (n, r)
    Left orthonormal factor at t0.
s0 : np.ndarray, shape (r,)
    Singular values at t0.
Vh0 : np.ndarray, shape (r, n)
    Right factor (V*) at t0.
rhs_fn : callable
    Velocity field F(A), maps (n, n) array -> (n, n) array.
t0 : float
    Start time.
t_end : float
    End time.
h : float
    Step size (must be strictly positive).
r : int
    Target rank for truncation.

Returns
-------
U_N : np.ndarray, shape (n, r)
s_N : np.ndarray, shape (r,)
Vh_N : np.ndarray, shape (r, n)

Implementation requirement
--------------------------
Call the previously defined public functions ``prk2_qdeim_step`` rather than reproducing their algorithms locally."""
    n = U0.shape[0]
    return (np.zeros((n, r), dtype=complex), np.zeros(r), np.zeros((r, n), dtype=complex))
```

### Step 7

07_kron_qdeim_indices

Goal
----
Implement the two-stage Kronecker QDEIM sampling construction.

Use the orthonormal factor bases left and right together with the reduced core basis to return exactly r distinct full Kronecker-row indices in pivot order. Preserve the source construction's reduced selection and C-order row mapping, and reject nonfinite, nonorthonormal, rank-deficient, or incompatible inputs with ValueError.

```python
import numpy as np

def kron_qdeim_indices(left: np.ndarray, right: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Return full Kronecker-row indices from the source's two-stage sampler.

    ``left`` and ``right`` have orthonormal columns and shapes ``(m1,r)`` and
    ``(m2,r)``. ``core`` has orthonormal columns and shape ``(r*r,r)``.
    Call the earlier ``qdeim_select`` on ``left`` and ``right``. If their
    ordered index arrays are ``p`` and ``q``, form
    ``B = kron(left[p,:], right[q,:]) @ core``. Compute its reduced QR,
    canonicalize each column so the corresponding diagonal of ``R`` is
    positive, and call ``qdeim_select`` on that orthonormal factor. For each
    selected reduced row ``s``, with ``i=s//r`` and ``j=s%r``, return the
    full C-order Kronecker row ``p[i]*m2+q[j]``. Real and complex factor
    bases are accepted. Reject non-finite, non-orthonormal, rank-deficient,
    or incompatible inputs.

    Returns
    -------
    numpy.ndarray
        Exactly ``r`` distinct integer indices in QDEIM pivot order.
    """
    return np.empty(0, dtype=int)
```

### Step 8

08_dlra_deim_pipeline

Goal
----
Implement dlra_deim_pipeline as the final end-to-end orchestrator.

Build the discrete nonlinear Schrödinger initial state and coupling matrix, use the earlier full-order integrator to obtain the pre-propagated and reference states, initialise the rank-r factors, certify the Kronecker sampling construction, and call the earlier low-rank integrator through the final time. Return a dictionary containing the relative Frobenius error, its integer accuracy score floor(-log10(rel_error)), and the number of low-rank time steps.

```python
import numpy as np
from typing import Dict, Any, Tuple

def dlra_deim_pipeline(n: int, alpha: float, r: int, h_ref: float, h_dlra: float, t0_pre: float, T: float) -> Dict[str, Any]:
    """Run the complete DLRA-DEIM pipeline for the Schrodinger equation.

Parameters
----------
n : int
    Spatial grid size (A in C^{n x n}).
alpha : float
    Nonlinearity parameter.
r : int
    Approximation rank.
h_ref : float
    Step size for the RK4 reference integration.
h_dlra : float
    Step size for PRK2-QDEIM integration.
t0_pre : float
    Pre-propagation time (RK4 from 0 to t0_pre).
T : float
    Final time.

Returns
-------
result : dict
    Dictionary with keys:
    - 'rel_error': float, relative Frobenius error
    - 'answer': int, floor(-log10(rel_error))
    - 'n_steps': int, number of PRK2-QDEIM steps taken

Implementation requirement
--------------------------
Call the previously defined public functions ``prk2_qdeim_integrate``, ``rk4_integrate``, ``schrodinger_rhs`` rather than reproducing their algorithms locally.

Implementation requirement
--------------------------
Call ``kron_qdeim_indices`` after the initialization SVD to certify the source Algorithm 2 two-stage Kronecker sampler before advancing the matrix trajectory."""
    result = {}
    return result
```
