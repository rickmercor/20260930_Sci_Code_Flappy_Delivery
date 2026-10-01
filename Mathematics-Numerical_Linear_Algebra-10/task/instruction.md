# Mathematics-Numerical_Linear_Algebra-10

## Background

Krylov subspace methods reduce large linear systems to smaller least-squares problems, but maintaining a fully orthogonal basis can dominate the cost as the subspace grows. Deterministic sketching replaces a generic random embedding with rows selected from the current non-orthogonal Krylov basis. Q-DEIM supplies an interpolation set at the basis rank, while GappyPOD+E can add rows to improve the spectral quality of the sampled basis. In sketched GMRES, the selected rows define a compressed residual minimization, whereas random orthonormal embeddings provide an exact finite-dimensional expectation for sketch-and-solve residual inflation. The additional uncertainty calculation examines sensitivity of the fixed deterministic sketch within the orthogonal residual space. A metric restricted to that space yields a fractional quadratic optimization; changing the denominator or maximizing over all ambient directions would measure a different sensitivity.

## Problem

Randomized and deterministic sketching provide different ways to compress least-squares problems originating from Krylov methods. The deterministic sampling can exploit the current non-orthogonal basis, while random orthonormal sketches admit sharp expected-error benchmarks. Consider the real $14\times14$ nonsymmetric matrix $K$, using one-based indices, with $K_{ii}=1+0.05i$, $K_{i,i+1}=0.8$, $K_{i+1,i}=0.01$, and $K_{i,i+2}=0.5$ whenever the indicated indices lie in $\{1,\ldots,14\}$, all other entries zero, and
$b=[-1.75034599,-2.39261019,-0.66844122,2.54142790,2.37457273,1.51299239,-0.75570191,-0.83104218,-1.29937175,-1.23666168,0.61959633,1.73510860,-0.25227877,-0.54794458]^\top$.

Using a $k=1$ truncated-Arnoldi basis of dimension $m=5$, construct the deterministically sketched GMRES approximation with sketch size $s=7$ by using Q-DEIM for the initial $m$ row indices and the eigenvalue-bound GappyPOD+E rule for the two oversampling indices. Whenever a Q-DEIM pivot score or GappyPOD+E score is tied within $10^{-12}$, choose the smallest original one-based row index. Let $\rho_{\mathrm{det}}$ be the squared residual of this deterministically sketched GMRES approximation divided by the minimum squared residual attainable over the same $m$-dimensional Krylov search space.

Freeze this basis, its image $M=KV_5$, and the selected row-extraction matrix $S$. Let $u=(b-MM^\dagger b)/\lVert b-MM^\dagger b\rVert_2$ and $W=\operatorname{diag}(1+0.1i)_{i=1}^{14}$. To test sensitivity of the fixed sketch to residual uncertainty, define the task-specific robust inflation

$$
\rho_{\mathrm{rob}}=\max_{e:\,M^\top e=0,\ u^\top e=0,\ e^\top We=0.65^2}
\frac{\lVert (I-M(SM)^\dagger S)(u+e)\rVert_2^2}{\lVert u+e\rVert_2^2}.
$$

The uncertainty constraint is an ellipsoid surface, not its interior. Orthogonality is Euclidean. Do not regenerate the Krylov basis or reselect rows for perturbed residuals. Compute the global maximum, including singular stationary cases if encountered. This uncertainty analysis is an extension defined here, not a result claimed by the source papers.

For a real uniformly random orthonormal embedding with ambient dimension $n=14$, rank $r=5$, and embedding dimension $\ell=7$, determine the exact expected residual-inflation factor $\rho_{\mathrm{orth}}$ analytically. Compare the deterministic worst value on this fixed uncertainty set to the random expectation for a fixed residual direction: the denominator is not an expectation of a worst-case random quantity. Using IEEE-754 binary64 arithmetic, report $R_{\mathrm{rob}}=\rho_{\mathrm{rob}}/\rho_{\mathrm{orth}}$ rounded to 8 digits after the decimal point. In the reasoning identify $\kappa_2(V_5)$, the Q-DEIM rows, the two GappyPOD+E additions, the nominal $\rho_{\mathrm{det}}$, the robust $\rho_{\mathrm{rob}}$, and $\rho_{\mathrm{orth}}$. Briefly explain the constrained reduction, why its denominator depends on direction, and how global optimality is certified.

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

step1_construct_krylov_operator

Goal
----
Construct the deterministic nonsymmetric banded operator.

```python
import numpy as np

def construct_krylov_operator(n: int = 14) -> np.ndarray:
    r"""Construct the deterministic nonsymmetric banded operator.

    Parameters
    ----------
    n : int, optional
        Positive matrix dimension. The default is 14.

    Returns
    -------
    K : np.ndarray
        Finite float64 square matrix of shape $n\times n$ with $K_{ii}=1+0.05\,i$,
        $K_{i,i+1}=0.8$, $K_{i+1,i}=0.01$, and $K_{i,i+2}=0.5$ (one-based indices).

    Raises
    ------
    ValueError
        If n is not a positive integer.
    """
    return K
```

### Step 2

step2_truncated_arnoldi_basis

Goal
----
Generate a truncated-Arnoldi basis and its matrix image.

```python
import numpy as np

def truncated_arnoldi_basis(A: np.ndarray, b: np.ndarray, m: int, k: int) -> tuple[np.ndarray,np.ndarray]:
    r"""Generate a truncated-Arnoldi basis and its matrix image.

    Parameters
    ----------
    A : np.ndarray
        Finite nonempty square matrix of shape $n\times n$.
    b : np.ndarray
        Finite nonzero starting vector of length n.
    m : int
        Number of Krylov basis vectors, satisfying $1\le m\le n$.
    k : int
        Positive truncation length, satisfying $1\le k\le m$; the new
        vector is orthogonalized against the $k$ most recent basis vectors
        only, with $v_1=b/\lVert b\rVert_2$. For each new column, compute
        every projection coefficient against the original image
        $Av_j$, then subtract the corresponding basis components
        (one pass of classical, not modified, Gram--Schmidt).
        Breakdown means that the new norm is at most binary64 machine
        epsilon times $\lVert Av_j\rVert_2$.

    Returns
    -------
    V : np.ndarray
        Float64 truncated-Arnoldi basis of shape $n\times m$.
    M : np.ndarray
        Float64 matrix image $M=AV$ of shape $n\times m$.

    Raises
    ------
    ValueError
        If A is not a finite nonempty square matrix, if b is not a
        finite nonzero vector of compatible length, if m or k is
        outside its admissible range, or if Arnoldi breakdown occurs.
    """
    return V, M
```

### Step 3

step3_qdeim_indices

Goal
----
Determine the Q-DEIM row indices for a basis matrix.

```python
import numpy as np

def qdeim_indices(V: np.ndarray, tie_tol: float = 1e-12) -> np.ndarray:
    r"""Determine the Q-DEIM row indices for a basis matrix.

    Parameters
    ----------
    V : np.ndarray
        Finite two-dimensional full-column-rank basis matrix with
        shape $n\times m$, where $n\ge m$.
    tie_tol : float, optional
        Finite nonnegative tolerance used to resolve numerically tied
        pivot scores. The default is 1e-12.

    Returns
    -------
    indices : np.ndarray
        One-dimensional integer array of length $m$ containing the
        selected one-based row indices in pivot order (pivoted QR of
        $V^{\top}$; ties within `tie_tol` resolve to the smallest index).

    Raises
    ------
    ValueError
        If V is not a finite nonempty two-dimensional matrix, if
        $n<m$, if V is not full column rank, if tie_tol is invalid,
        or if a rank-deficient pivot is encountered.
    """
    return indices
```

### Step 4

step4_gappypod_e_oversample

Goal
----
Extend a Q-DEIM row set using GappyPOD+E oversampling.

```python
import numpy as np

def gappypod_e_oversample(V: np.ndarray, initial_indices: np.ndarray, s: int, tie_tol: float = 1e-12) -> np.ndarray:
    r"""Extend a Q-DEIM row set using GappyPOD+E oversampling.

    Parameters
    ----------
    V : np.ndarray
        Finite full-column-rank basis matrix of shape $n\times m$. For a
        single-column basis ($m=1$) no eigengap exists; it is taken as
        zero, so every candidate scores zero and ties resolve to the
        smallest one-based index.
    initial_indices : np.ndarray
        One-dimensional integer array of exactly $m$ distinct one-based
        row indices whose sampled basis has full column rank.
    s : int
        Target sketch size satisfying $m\le s\le n$.
    tie_tol : float, optional
        Finite nonnegative tolerance used to resolve numerically tied
        oversampling scores. The default is 1e-12.

    Returns
    -------
    indices : np.ndarray
        One-dimensional integer array of length $s$ containing the
        one-based selected rows, preserving the initial index order.

    Raises
    ------
    ValueError
        If V is invalid or rank deficient, if initial_indices is not a
        valid full-rank $m$-row sample, if $s$ is outside $[m,n]$, if
        tie_tol is invalid, or if no admissible oversampling row exists.
    """
    return indices
```

### Step 5

step5_deterministic_sgmres_solve

Goal
----
Solve the deterministically sketched GMRES least-squares problem.

```python
import numpy as np

def deterministic_sgmres_solve(V: np.ndarray, M: np.ndarray, b: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray,float]:
    r"""Solve the deterministically sketched GMRES least-squares problem.

    Parameters
    ----------
    V : np.ndarray
        Finite full-column-rank Krylov basis of shape $n\times m$.
    M : np.ndarray
        Finite full-column-rank matrix image $M=AV$ of shape $n\times m$,
        compatible with V.
    b : np.ndarray
        Finite right-hand-side vector of length $n$.
    indices : np.ndarray
        One-dimensional integer array containing at least $m$ distinct
        one-based sampled row indices.

    Returns
    -------
    y : np.ndarray
        Float64 Krylov coefficient vector $y$ of length $m$.
    residual_sq : float
        Full-space squared residual $\lVert b-My\rVert_2^2$ as a native
        Python float.

    Raises
    ------
    ValueError
        If V or M is invalid, nonfinite, incompatible, or rank
        deficient; if b has invalid shape or values; if indices are
        invalid or repeated; or if the sampled matrix image is rank
        deficient.
    """
    return y, residual_sq
```

### Step 6

step6_deterministic_residual_inflation

Goal
----
Compute residual inflation relative to the best reduced-space solve.

```python
import numpy as np

def deterministic_residual_inflation(M: np.ndarray, b: np.ndarray, residual_sq: float) -> tuple[float,float]:
    r"""Compute residual inflation relative to the best reduced-space solve.

    Parameters
    ----------
    M : np.ndarray
        Finite full-column-rank reduced design matrix of shape $n\times m$.
    b : np.ndarray
        Finite right-hand-side vector of length $n$.
    residual_sq : float
        Finite nonnegative squared residual from the sketched solve.

    Returns
    -------
    rho_det : float
        Deterministic residual-inflation factor
        $\rho_{\mathrm{det}}=\mathrm{residual\_sq}/\min_x\lVert b-Mx\rVert_2^2$.
    optimal_residual_sq : float
        Minimum squared residual attainable over the same reduced
        search space.

    Raises
    ------
    ValueError
        If M is invalid or rank deficient, if b is invalid, if
        residual_sq is not finite and nonnegative, or if the optimal
        reduced-space residual is zero.
    """
    return rho_det, optimal_residual_sq
```

### Step 7

step7_random_orthonormal_benchmark

Goal
----
Evaluate the exact random-orthonormal residual-inflation benchmark.

```python
import numpy as np

def random_orthonormal_benchmark(n: int, r: int, ell: int, field: str = "real") -> float:
    r"""Evaluate the exact random-orthonormal residual-inflation benchmark.

    Parameters
    ----------
    n : int
        Positive ambient dimension.
    r : int
        Positive matrix rank satisfying $r<n$.
    ell : int
        Positive embedding dimension satisfying $\ell\le n$. When
        $\ell=n$ the embedding is square, the factor is exactly $1$, and no
        further condition is required; otherwise the finite-expectation
        condition $r<\ell-\alpha$ must hold, with $\alpha=1$ for "real"
        and $\alpha=0$ for "complex".
    field : str, optional
        Scalar field, either "real" or "complex". The default is "real".

    Returns
    -------
    rho : float
        Exact finite expected residual-inflation factor
        $1+\frac{n-\ell}{n-r}\,\frac{r}{\ell-r-\alpha}$ (equal to $1$ when $\ell=n$).

    Raises
    ------
    ValueError
        If n, r, or ell is not an admissible integer, if field is not
        "real" or "complex", or if $\ell<n$ and the expectation is not
        finite for the supplied dimensions.
    """
    return rho
```

### Step 8

step8_fractional_sphere_maximum

Goal
----
Compute a globally maximal fractional quadratic on a sphere.

```python
import numpy as np


def fractional_sphere_maximum(A: np.ndarray, g: np.ndarray, c: float,
                             B: np.ndarray, tau: float) -> float:
    r"""Maximize a fractional quadratic over a Euclidean sphere.

    Parameters
    ----------
    A : np.ndarray
        Finite real symmetric matrix of shape $d\times d$, $d\ge1$;
        it may be indefinite and may have repeated eigenvalues.
    g : np.ndarray
        Finite real vector of length $d$.
    c : float
        Finite real constant.
    B : np.ndarray
        Finite real symmetric positive-semidefinite matrix of shape
        $d\times d$. Thus the denominator below is strictly positive.
    tau : float
        Finite nonnegative sphere radius.

    Returns
    -------
    value : float
        $\max_{\lVert z\rVert_2=\tau}
        (c+2g^\top z+z^\top Az)/(1+z^\top Bz)$.
        For $\tau=0$, return $c$. The constraint is a sphere, including
        when $A$ is negative definite. Return the global value even
        when the maximizing direction is not unique. Numerical answers
        are compared at relative and absolute tolerance $10^{-9}$.

    Raises
    ------
    ValueError
        For nonfinite inputs, incompatible or empty shapes, nonsymmetric
        matrices, non-positive-semidefinite $B$, or negative $\tau$.
        Symmetry and semidefiniteness may be assessed at relative
        tolerance $10^{-12}$ against a scale of at least one.
    """
    return value
```

### Step 9

step9_deterministic_random_benchmark_ratio

Goal
----
Run the complete deterministic-versus-random benchmark pipeline.

```python
import numpy as np

def deterministic_random_benchmark_ratio(K, b: np.ndarray, m: int = 5, k: int = 1, s: int = 7, tie_tol: float = 1e-12, tau: float = 0.65, weights=None) -> float:
    r"""Run the complete deterministic-versus-random benchmark pipeline.

    Parameters
    ----------
    K : np.ndarray or int
        Finite real square matrix defining the linear system, or a
        positive integer $n$, in which case the deterministic banded
        operator of `construct_krylov_operator(n)` is used.
    b : np.ndarray
        Finite nonzero right-hand-side and Krylov starting vector.
    m : int, optional
        Krylov subspace dimension. The default is 5.
    k : int, optional
        Truncated-Arnoldi orthogonalization length. The default is 1.
    s : int, optional
        Deterministic sketch size and random embedding dimension.
        The default is 7.
    tie_tol : float, optional
        Finite nonnegative tolerance used in deterministic row-selection
        tie breaking. The default is 1e-12.

    tau : float, optional
        Finite nonnegative uncertainty radius, default $0.65$.
    weights : np.ndarray or None, optional
        Positive finite length-$n$ diagonal entries of $W$. If omitted,
        $W_{ii}=1+0.1i$ with one-based $i$.

    Notes
    -----
    Freeze the basis and selected rows from the original right-hand side.
    Put $M=KV$, $u=(b-MM^\dagger b)/\lVert b-MM^\dagger b\rVert_2$,
    and let $S$ extract the selected rows. The robust inflation is
    $\rho_{\mathrm{rob}}=\max_e
    \lVert (I-M(SM)^\dagger S)(u+e)\rVert_2^2/\lVert u+e\rVert_2^2$,
    subject to $M^\top e=0$, $u^\top e=0$, and $e^\top We=\tau^2$.
    The last constraint is equality. At zero radius use the nominal
    inflation. For positive radius require $n-m-1\ge1$. Whiten the
    metric only on this constrained subspace; Euclidean and weighted
    orthogonality are different. The denominator varies with direction.
    Use the fractional-sphere stage to obtain the global maximum.
    No Krylov regeneration or row reselection is performed for a perturbation.

    Returns
    -------
    value : float
        Unrounded ratio $\rho_{\mathrm{rob}}/\rho_{\mathrm{orth}}$ as a
        native Python float.

    Raises
    ------
    ValueError
        If K is neither a two-dimensional array nor a positive integer,
        or if any input violates the requirements of the Krylov-basis,
        Q-DEIM, GappyPOD+E, sketched-GMRES, residual-inflation, or
        random-orthonormal benchmark stages, if $\tau$ or the weights are invalid,
        or if positive $\tau$ has an empty tangent space.
    """
    return value
```
