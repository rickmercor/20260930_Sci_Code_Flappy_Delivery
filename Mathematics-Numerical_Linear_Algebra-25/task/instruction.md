# Mathematics-Numerical_Linear_Algebra-25

## Background

Regularized least-squares solvers can converge slowly when a few large singular values are separated from a mild trailing spectrum. Recent sketch-and-precondition research reuses one small embedding to build an adaptive low-rank surrogate for the augmented system `[A; mu I]`, then modifies only the currently unresolved dominant directions before a short iterative solve.

## Problem

Regularized linear least-squares problems of the form minimize ||A x - b||_2^2 + mu^2 ||x||_2^2 are central in inverse problems and scientific computing, but iterative convergence can be slow when the spectrum of A is clustered and ill-conditioned. Recent work avoids a large up-front full-rank factorization by reusing one small sketch to grow an adaptive row-and-column surrogate and modify only the captured regularized directions before a short Krylov solve.
Use numpy Generator default_rng(7) to form the Q factors returned directly by np.linalg.qr(..., mode="reduced") for standard_normal draws of shapes (12, 8) and (8, 8), without postprocessing their column signs, and set A = U diag(s) V^T with s = (10, 9, 8, 0.45, 0.3, 0.22, 0.15, 0.1); draw b as standard_normal(12) from the same generator. Using default_rng(11), draw a Gaussian sketch S of shape (3, 12). Take mu = 0.05, CUR block size ell = 2, and x0 = 0.
Apply the source-defined residual-update selection and compact spectral construction to the augmented operator A_mu = [A; mu I], using the rank-2 surrogate to flatten its captured regularized values to the smallest captured value. Starting from x0, execute exactly two right-preconditioned iterations of the associated least-squares Krylov recurrence. Your final answer must be a single number: the first coordinate of that two-iteration iterate.
Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
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

01_construct_clustered_ls_data

Goal
----
Build a clustered-spectrum least-squares instance [A | b]. Draw thin QR factors of Gaussians from numpy Generator default_rng(seed), set A = U diag(s) V^T with prescribed positive singular values s, then draw b from the same generator. Return an array of shape (m, n+1) whose first n columns are A and whose last column is b.

```python
import numpy as np

def construct_clustered_ls_data(
    m: int,
    n: int,
    s: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Build [A | b] with prescribed singular values.

    Parameters
    ----------
    m : int
        Row dimension, m >= n >= 1.
    n : int
        Column dimension.
    s : np.ndarray
        Positive singular values, shape (n,).
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    data : np.ndarray
        Array of shape (m, n+1) whose first n columns are A and whose last
        column is b.

    Raises
    ------
    ValueError
        If m or n is not an integer, if m >= n >= 1 does not hold, if s does not
        have shape (n,), or if s is not positive and finite.
    """
    return np.zeros((m, n + 1))
```

### Step 2

02_form_sketched_range

Goal
----
Draw one Gaussian sketch S of shape (n_sketch, m) from default_rng(seed) and return Y = S A. The sketch is computed once and reused for CUR index selection; no second embedding is drawn.

```python
import numpy as np

def form_sketched_range(A: np.ndarray, n_sketch: int, seed: int) -> np.ndarray:
    """Compute Y = S A with S ~ randn(n_sketch, m).

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    n_sketch : int
        Sketch dimension, 1 <= n_sketch.
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    Y : np.ndarray
        Sketched matrix of shape (n_sketch, n).

    Raises
    ------
    ValueError
        If A is not a 2D array with at least one row and one column, or if
        n_sketch is not an integer >= 1.
    """
    return np.zeros((n_sketch, A.shape[1]))
```

### Step 3

03_select_cur_indices

Goal
----
Select rank-ell CUR row and column indices from a reused sketch. With empty index sets the sketched residual is E_row = Y. Columns J are the first ell LUPP row pivots of Y^T; rows I are the first ell LUPP row pivots of A(:, J). Return the concatenated vector [I, J] as floats.

```python
import numpy as np

def select_cur_indices(A: np.ndarray, Y: np.ndarray, ell: int) -> np.ndarray:
    """Residual-update CUR indices from one sketch.

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    Y : np.ndarray
        Sketch Y = S A, shape (n_sketch, n) with n_sketch >= ell.
    ell : int
        Block size / target rank, 1 <= ell <= min(m, n, n_sketch).

    Returns
    -------
    index_vector : np.ndarray
        Length-2*ell vector [I, J] of 0-based indices stored as floats.

    Raises
    ------
    ValueError
        If A or Y is not 2D, if Y does not have n columns matching A, if ell is
        not an integer >= 1, or if ell exceeds min(m, n, n_sketch).
    """
    return np.zeros(2 * ell)
```

### Step 4

04_cur_core_matrix

Goal
----
Form the CUR-CA core from packed indices: unpack [I, J] and return U = A(I, J)^+, the Moore-Penrose pseudoinverse of the intersection submatrix.

```python
import numpy as np

def cur_core_matrix(A: np.ndarray, index_vector: np.ndarray) -> np.ndarray:
    """CUR-CA core U = A(I, J)^+.

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    index_vector : np.ndarray
        Length-2*ell vector [I, J] of 0-based indices.

    Returns
    -------
    U : np.ndarray
        Core matrix of shape (ell, ell).

    Raises
    ------
    ValueError
        If A is not a 2D array, if index_vector does not have even positive
        length, if any index is out of range, or if I and J do not each
        contain ell distinct indices.
    """
    return np.zeros((len(index_vector) // 2, len(index_vector) // 2))
```

### Step 5

05_cur_captured_singular_values

Goal
----
Compute the captured singular values of a CUR surrogate from its supplied core.

```python
import numpy as np

def cur_captured_singular_values(
    A: np.ndarray,
    index_vector: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """Return the singular values captured by the supplied CUR factors.

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    index_vector : np.ndarray
        Length-2*ell vector [I, J] of 0-based indices.
    U : np.ndarray
        CUR core from the preceding step, shape (ell, ell).

    Returns
    -------
    sigma : np.ndarray
        Captured singular values, shape (ell,), in descending order.

    Raises
    ------
    ValueError
        If A is not a 2D array, if index_vector does not have even positive
        length, if any index is out of range, if I and J do not each contain
        ell distinct indices, if U is not a finite array of shape
        (ell, ell), or if C^T C is not symmetric positive definite.
    """
    return np.zeros(len(index_vector) // 2)
```

### Step 6

06_build_spectral_pinv_factors

Goal
----
Build compact numerical factors for the implicit rank-ell spectral inverse.

```python
import numpy as np

def build_spectral_pinv_factors(
    A: np.ndarray,
    index_vector: np.ndarray,
    U: np.ndarray,
    sigma: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Pack the inverse scales and captured right basis.

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    index_vector : np.ndarray
        Length-2*ell vector [I, J] of 0-based indices.
    U : np.ndarray
        CUR core from the preceding step, shape (ell, ell).
    sigma : np.ndarray
        Captured singular values from the preceding step, shape (ell,).
    mu : float
        Regularization parameter, mu >= 0.

    Returns
    -------
    factors : np.ndarray
        Array of shape (n+1, ell). The first row stores inverse scaling
        coefficients and the remaining rows store the captured right basis.

    Raises
    ------
    ValueError
        If A is not a 2D array, if index_vector does not have even positive
        length, if any index is out of range, if I and J do not each contain
        ell distinct indices, if U is not a finite array of shape
        (ell, ell), if sigma does not contain ell positive finite values, if
        mu is not a finite number >= 0, if C^T C is not symmetric positive
        definite, or if sigma is inconsistent with A, index_vector and U.
    """
    return np.zeros((A.shape[1] + 1, len(index_vector) // 2))
```

### Step 7

07_preconditioned_lsqr

Goal
----
Run a fixed number of right-preconditioned LSQR iterations.

```python
import numpy as np

def preconditioned_lsqr(
    A: np.ndarray,
    b: np.ndarray,
    pinv_factors: np.ndarray,
    mu: float,
    niter: int,
) -> np.ndarray:
    """Solve the augmented problem using supplied inverse factors.

    Parameters
    ----------
    A : np.ndarray
        Data matrix, shape (m, n).
    b : np.ndarray
        Right-hand side, shape (m,).
    pinv_factors : np.ndarray
        Packed inverse coefficients and right basis, shape (n+1, ell).
    mu : float
        Regularization parameter, mu >= 0.
    niter : int
        Number of Golub-Kahan LSQR iterations, niter >= 1.

    Returns
    -------
    x : np.ndarray
        Approximate minimizer, shape (n,).

    Raises
    ------
    ValueError
        If A is not 2D, if b does not have shape (m,), if pinv_factors is not
        finite with shape (n+1, ell), if the captured right basis does not
        have orthonormal columns, if mu is not a finite number >= 0, or if
        niter is not an integer >= 1.
    """
    return np.zeros(A.shape[1])
```

### Step 8

08_run_cur_spectral_lsqr

Goal
----
Chain the seven numerical steps and return the requested coordinate.

```python
import numpy as np

def run_cur_spectral_lsqr(
    m: int,
    n: int,
    s: np.ndarray,
    data_seed: int,
    n_sketch: int,
    sketch_seed: int,
    ell: int,
    mu: float,
    niter: int,
) -> float:
    """Return the first coordinate produced by the complete pipeline.

    Parameters
    ----------
    m, n : int
        Dimensions with m >= n >= 1.
    s : np.ndarray
        Positive singular values, shape (n,).
    data_seed : int
        Seed for A and b.
    n_sketch : int
        Gaussian sketch rows.
    sketch_seed : int
        Seed for the sketch.
    ell : int
        CUR block size.
    mu : float
        Regularization parameter.
    niter : int
        Number of LSQR iterations.

    Returns
    -------
    x0 : float
        First coordinate of the resulting iterate.

    Raises
    ------
    ValueError
        Propagated from the underlying steps whenever any argument is invalid,
        for example when m >= n >= 1 does not hold, when s is not positive
        and finite, when ell exceeds min(m, n, n_sketch), when mu is
        negative, or when niter is not an integer >= 1.
    """
    return 0.0
```
