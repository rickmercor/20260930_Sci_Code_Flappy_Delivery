# Mathematics-Numerical_Linear_Algebra-38

## Background

Large-scale matrix-function computations arise when the action $f(A)b$ is needed for a sparse, non-Hermitian matrix $A$, but forming the full matrix function or maintaining a fully orthonormal Krylov basis is too expensive. Krylov subspace methods reduce the problem to matrix-vector products and a much smaller projected problem, while randomized sketching can reduce the cost of inner-product calculations and storage. However, truncated orthogonalization can produce a nonorthogonal basis whose conditioning deteriorates as the Krylov dimension grows, motivating adaptive control based on a sketched basis. Restarting methods provide another way to limit storage by representing the approximation error through an integral involving the spectral information of the reduced problem, allowing a new Krylov process to approximate the remaining error without retaining the previous full basis. The resulting framework is particularly relevant for matrix functions with integral representations, including Stieltjes functions such as inverse fractional powers.

## Problem

Computing the action of a matrix function on a vector is important for large-scale scientific problems where the matrix is sparse, non-Hermitian, and too large to factor or diagonalize explicitly. The paper develops a randomized sketch-and-restart framework that reduces the cost and storage associated with Krylov subspace methods while retaining a matrix-function approximation suitable for repeated restarting. The computational input is a sparse convection-diffusion operator, a normalized starting vector, and the scalar function $f(z)=z^{-1/2}$, while the required output is one specified entry of a once-restarted approximation to $f(A)b$. Solve one concrete deterministic instance of this framework with $n=7$, $N=n^2=49$, $h=1/(n+1)$, and $D=10^{-3}$, where $$ A=\frac{D}{h^2}(L\otimes I+I\otimes L)+\frac{1}{h}(C\otimes I+I\otimes C^\top), $$ $L$ is tridiagonal with diagonal entries $2$ and first sub- and superdiagonal entries $-1$, and $C$ is bidiagonal with diagonal entries $1$ and first subdiagonal entries $-1$; let $b$ be the all-ones vector normalized to unit Euclidean norm and let $f(z)=z^{-1/2}$ use the principal branch. Use sparse-sign sketching with sparsity $\zeta=\min(\text{number of sketch rows},8)$, generated from numpy.random.default_rng(2026) with the same generator stream continued for every subsequent random draw, together with truncation parameter $t=1$, condition-number threshold $\tau=10^4$, initial and incremental sketch dimension $s_0=10$, sketch-growth parameter $\eta=2$, and maximum Krylov dimension $m_{\max}=12$ for the first cycle, with the first-cycle dimension determining the fixed maximum dimension of the second cycle. For the restart correction, use the quadrature construction associated with the Stieltjes representation of $z^{-1/2}$, beginning with $r_1=4$ nodes and doubling the number of nodes until successive corrections agree to $10^{-6}$ or $64$ nodes are reached. Compute the truncated sketch-corrected Arnoldi-like decomposition for the first cycle, the resulting zero-restart approximation $f^{[0]}$, the harmonic Ritz values and quadrature-based error function used for exactly one restart, the second truncated sketch-corrected decomposition seeded by the corrected residual direction from the first cycle, and the resulting once-restarted approximation $f^{[1]}$; the final requested quantity is the **25th entry of** **$f^{[1]}$**.

Output Format Requirements: Emit <final_answer> immediately, then . Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules:

The tags are required. Do not omit them or leave them empty.
The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
Put only that one number between the tags. No units, no words, no extra lines. Keep short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_build_convdiff_operator

Goal
----
Construct the convection-diffusion operator and the normalized starting vector for the deterministic matrix-function computation.

```python
import numpy as np

def build_convdiff_operator(
    state: dict,
    n: int,
    D: float,
) -> float:
    """Construct the operator and normalized starting vector.

    Parameters
    ----------
    state : dict
        Mutable state for intermediate pipeline data.
    n : int
        Number of interior grid points per dimension.
    D : float
        Diffusion coefficient.

    Returns
    -------
    float
        Frobenius norm of the constructed operator.
    """
    result = 0.0
    return result
```

### Step 2

02_sparse_sign_sketch

Goal
----
Generate the sparse randomized embedding used by the adaptive Krylov computation.

```python
import numpy as np

def sparse_sign_sketch(
    state: dict,
    seed: int,
    rows: int,
) -> float:
    """Generate a deterministic sparse-sign sketch.

    Parameters
    ----------
    state : dict
        Mutable pipeline state containing the ambient dimension.
    seed : int
        Random seed for the sketch stream.
    rows : int
        Number of sketch rows.

    Returns
    -------
    float
        Frobenius norm of the sketch.
    """
    result = 0.0
    return result
```

### Step 3

03_truncated_arnoldi_cycle

Goal
----
Generate one adaptive Krylov decomposition using truncated orthogonalization and sketch-based conditioning.

```python
import numpy as np

def truncated_arnoldi_cycle(
    state: dict,
    t: int,
    tau: float,
    s0: int,
    eta: float,
    mmax: int,
) -> float:
    """Run one adaptive truncated-Arnoldi cycle.

    Parameters
    ----------
    state : dict
        Mutable pipeline state containing the operator, starting vector,
        sketch, and random-number generator.
    t : int
        Truncation parameter.
    tau : float
        Condition-number threshold.
    s0 : int
        Sketch rows added during growth.
    eta : float
        Sketch-growth control parameter.
    mmax : int
        Maximum Krylov dimension.

    Returns
    -------
    float
        Selected Krylov dimension.
    """
    result = 0.0
    return result
```

### Step 4

04_rank1_harmonic_update

Goal
----
Apply the correction required by the reduced representation after a truncated Krylov cycle.

```python
import numpy as np

def rank1_harmonic_update(
    state: dict,
) -> float:
    """Apply the rank-1 correction to the current Krylov cycle.

    Parameters
    ----------
    state : dict
        Mutable pipeline state containing the current decomposition.

    Returns
    -------
    float
        Norm of the reduced-matrix correction.
    """
    result = 0.0
    return result
```

### Step 5

05_arnoldi_like_approx

Goal
----
Evaluate the reduced inverse-square-root approximation associated with the corrected cycle.

```python
import numpy as np

def arnoldi_like_approx(
    state: dict,
) -> float:
    """Evaluate the reduced matrix-function approximation.

    Parameters
    ----------
    state : dict
        Mutable pipeline state containing the corrected reduced matrix.

    Returns
    -------
    float
        Euclidean norm of the approximation.
    """
    result = 0.0
    return result
```

### Step 6

06_restart_error_scalar_kernel

Goal
----
Evaluate one scalar value of the restart correction kernel associated with the previous reduced problem.

```python
import numpy as np

def restart_error_scalar_kernel(
    theta: np.ndarray,
    gamma: float,
    beta: float,
    z: complex,
    tol: float,
    maxnodes: int,
    r1: int,
) -> float:
    """Evaluate one scalar value of the restart error kernel.

    Parameters
    ----------
    theta : np.ndarray
        Reduced spectral values from the previous cycle.
    gamma : float
        Scalar factor associated with the previous cycle.
    beta : float
        Starting-vector scaling.
    z : complex
        Evaluation point.
    tol : float
        Quadrature convergence tolerance.
    maxnodes : int
        Maximum quadrature nodes.
    r1 : int
        Initial quadrature nodes.

    Returns
    -------
    float
        Real part of the kernel value.
    """
    result = 0.0
    return result
```

### Step 7

07_error_kernel_matrix_apply

Goal
----
Construct the matrix-valued restart correction on the reduced problem of the next cycle.

```python
import numpy as np

def error_kernel_matrix_apply(
    state: dict,
    theta_prev: np.ndarray,
    gamma_prev: float,
    beta_prev: float,
    tol: float,
    maxnodes: int,
    r1: int,
) -> float:
    """Evaluate the restart kernel on the current reduced matrix.

    Parameters
    ----------
    state : dict
        Mutable pipeline state containing the current corrected matrix.
    theta_prev : np.ndarray
        Previous-cycle reduced spectral values.
    gamma_prev : float
        Previous-cycle scalar factor.
    beta_prev : float
        Previous-cycle starting-vector scaling.
    tol : float
        Quadrature tolerance.
    maxnodes : int
        Maximum quadrature nodes.
    r1 : int
        Initial quadrature nodes.

    Returns
    -------
    float
        Frobenius norm of the matrix-valued correction.
    """
    result = 0.0
    return result
```

### Step 8

08_orchestrator_sketch_and_restart

Goal
----
Assemble the preceding numerical stages into the complete deterministic two-cycle computation.

```python
import numpy as np

def sketch_and_restart_pipeline(
    n: int,
    D: float,
    sketch_seed: int,
    t: int,
    tau: float,
    s0: int,
    eta: float,
    mmax0: int,
    entry_index: int,
) -> float:
    """Run the complete two-cycle computation.

    Parameters
    ----------
    n : int
        Grid size.
    D : float
        Diffusion coefficient.
    sketch_seed : int
        Initial random seed.
    t : int
        Truncation parameter.
    tau : float
        Condition-number threshold.
    s0 : int
        Initial and incremental sketch size.
    eta : float
        Sketch-growth parameter.
    mmax0 : int
        Initial maximum Krylov dimension.
    entry_index : int
        Zero-based requested output index.

    Returns
    -------
    float
        Requested entry of the final approximation.
    """
    result = 0.0
    return result
```
