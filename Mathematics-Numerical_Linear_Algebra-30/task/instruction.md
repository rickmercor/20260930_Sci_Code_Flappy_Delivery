# Mathematics-Numerical_Linear_Algebra-30

## Background

Low-rank approximations of large symmetric matrices are often built from a small subset of their columns, because individual entries can be expensive to evaluate and a good column subset can be chosen from cheap side information such as leverage scores or pivoted factorisations. Once the columns are fixed, the quality of the approximation is decided by how the small middle matrix that couples them is chosen, and the natural choices differ in cost, in numerical stability, and in whether they require a full pass over the matrix. For positive semidefinite matrices the classical choice is well understood. For symmetric matrices whose eigenvalues have both signs, which arise from indefinite kernels, saddle-point systems, and centred or shifted covariance-type operators, the same question is more delicate and has drawn recent attention.

## Problem

A column-based reconstruction of a real symmetric matrix A has the form C M C^T, where C holds a chosen subset of the columns of A and M is a small square middle matrix. In this task A is indefinite and the column subset is fixed in advance. The source that studies this reconstruction states a multiplicative residual-bound factor for the sketched core when the reconstruction error is measured in the Frobenius norm. The wanted quantity is that factor, evaluated on the given column subset and the given unscaled draw, using the draw as given.

Use numpy Generator default_rng(7) to form a thin QR factor Q of a standard_normal draw of shape (12, 12), and set A = Q diag(s) Q^T with s = (6.5, 5.2, 4.1, -3.8, -2.9, -1.6, 0.9, 0.55, 0.35, -0.25, 0.18, 0.12). Let C consist of columns 5, 8, and 9 of A, indexing from 0. Using default_rng(11), form a standard_normal array of shape (6, 12). Do not rescale its entries. Your final answer must be a single number: that residual-bound factor.

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

01_construct_indefinite_symmetric_matrix

Goal
----
Build A = Q diag(s) Q^T from a thin QR factor of a Gaussian drawn by default_rng(seed). The eigenvalues s must contain both a positive and a negative entry. Require n >= 2.

```python
import numpy as np

def construct_indefinite_symmetric_matrix(n: int, s: np.ndarray, seed: int) -> np.ndarray:
    """Build A of shape (n, n) equal to Q diag(s) Q^T.

    Parameters
    ----------
    n : int
        Dimension, n >= 2.
    s : np.ndarray
        Eigenvalues of length n, with at least one positive and one negative entry.
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    A : np.ndarray
        Symmetric array of shape (n, n).

    Raises
    ------
    ValueError
        If n is not an integer, if n < 2, if s does not have shape (n,),
        if s has a non-finite entry, or if s does not contain both a
        positive and a negative entry.
    """
    return np.zeros((n, n))
```

### Step 2

02_extract_column_subset

Goal
----
Return C = A[:, I] for a 0-based index set I of distinct columns. Require 1 <= r < n and 0 <= I[j] < n for every index. Invalid shapes, repeated indices, and out-of-range indices raise ValueError.

```python
import numpy as np

def extract_column_subset(A: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return the columns of A listed in indices.

    Parameters
    ----------
    A : np.ndarray
        Square symmetric matrix of shape (n, n).
    indices : np.ndarray
        Distinct 0-based column indices, length r with 1 <= r < n.

    Returns
    -------
    C : np.ndarray
        Array of shape (n, r).

    Raises
    ------
    ValueError
        If A is not a square 2D array, if n < 2, if r is not in
        1 <= r < n, if indices are not distinct, or if any index lies
        outside 0, ..., n-1.
    """
    return np.zeros((1, 1))
```

### Step 3

03_draw_gaussian_sketch

Goal
----
Draw a standard_normal sketch X of shape (t, n) from default_rng(seed). Require 1 <= t < n.

```python
import numpy as np

def draw_gaussian_sketch(t: int, n: int, seed: int) -> np.ndarray:
    """Draw X of shape (t, n) with standard_normal entries.

    Parameters
    ----------
    t : int
        Sketch height, 1 <= t < n.
    n : int
        Sketch width (ambient dimension).
    seed : int
        RNG seed for numpy Generator default_rng.

    Returns
    -------
    X : np.ndarray
        Array of shape (t, n).

    Raises
    ------
    ValueError
        If t or n is not an integer, if n < 2, or if t is not in
        1 <= t < n.
    """
    return np.zeros((t, n))
```

### Step 4

04_residual_bound_factor

Goal
----
Return the specialised Frobenius-norm residual factor from the cited source's general bound (Source 1, Theorem 4.1), for this C and this sketch, using the sketch as given. Do not return the unitarily invariant factor. Do not orthonormalise the rows of X. Do not drop the square root from the Frobenius form. Require full column rank of C and t > r.

```python
import numpy as np

def residual_bound_factor(C: np.ndarray, X: np.ndarray) -> float:
    """Return the source's specialised Frobenius residual factor.

    The cited source (Source 1, Theorem 4.1) states a factor for every
    unitarily invariant norm and a specialised factor for the Frobenius
    norm. Return the Frobenius factor for this C and this sketch, using
    the sketch as given. Do not orthonormalise the rows of X. Do not
    return the unitarily invariant factor. Do not drop the square root
    from the Frobenius form. The factor is finite and strictly greater
    than one when the source's rank assumption holds.

    Parameters
    ----------
    C : np.ndarray
        Column subset of shape (n, r) with full column rank and 1 <= r < n.
    X : np.ndarray
        Sketch of shape (t, n) with t > r.

    Returns
    -------
    value : float
        The multiplicative factor.

    Raises
    ------
    ValueError
        If C is not a 2D array with 1 <= r < n, if X is not a 2D array with
        n columns, if t <= r, or if C is rank deficient.
    """
    return 0.0
```

### Step 5

05_sketched_middle_matrix

Goal
----
From A, the column subset C and the sketch X, return the r-by-r matrix M of the cited source's reconstruction C M C^T (Source 1, Algorithm 3.1). Require t > r and t < n. Return M symmetric to rounding.

```python
import numpy as np

def sketched_middle_matrix(A: np.ndarray, C: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return the r-by-r matrix M of the cited source's reconstruction C M C^T.

    The cited source (Source 1, Algorithm 3.1) determines a square middle
    matrix M from the column subset C and one sketch X. Return that
    matrix, of shape (r, r), symmetric to rounding.

    Parameters
    ----------
    A : np.ndarray
        Real symmetric matrix of shape (n, n).
    C : np.ndarray
        Column subset of shape (n, r) with 1 <= r < n.
    X : np.ndarray
        Sketch of shape (t, n) with r < t < n.

    Returns
    -------
    M : np.ndarray
        Array of shape (r, r).

    Raises
    ------
    ValueError
        If A is not a square 2D array, if C does not have shape (n, r) with
        1 <= r < n, if X does not have shape (t, n) with r < t < n, or if
        the sketched column block is rank deficient.
    """
    return np.zeros((1, 1))
```

### Step 6

06_reconstruction_entry

Goal
----
From the column subset C and the middle matrix M, return the 0-based (row, col) entry of the reconstruction C M C^T. Require C of shape (n, r) with 1 <= r < n, M of shape (r, r), and integer row and col in 0, ..., n-1.

```python
import numpy as np

def reconstruction_entry(C: np.ndarray, M: np.ndarray, row: int, col: int) -> float:
    """Return (C M C^T)[row, col], 0-based.

    Parameters
    ----------
    C : np.ndarray
        Column subset of shape (n, r).
    M : np.ndarray
        Middle matrix of shape (r, r).
    row, col : int
        0-based indices in 0, ..., n-1.

    Returns
    -------
    value : float
        One entry of C M C^T.

    Raises
    ------
    ValueError
        If C is not 2D, if n <= r or r < 1, if M does not have shape
        (r, r), if row or col is not an integer, or if row or col lies
        outside 0, ..., n-1.
    """
    return 0.0
```

### Step 7

07_run_sketched_core_entry

Goal
----
Build A, take the column subset, draw the sketch, evaluate the source's specialised Frobenius residual-bound factor, form the sketched middle matrix, confirm that one reconstruction entry is finite, and return the residual-bound factor.

```python
import numpy as np

def run_sketched_core_entry(
    n: int,
    s: np.ndarray,
    data_seed: int,
    sketch_seed: int,
    indices: np.ndarray,
    t: int,
    row: int,
    col: int,
) -> float:
    """End-to-end sketched symmetric reconstruction; return the residual factor.

    Chains the earlier steps: build A from (n, s, data_seed), take the
    column subset C = A[:, indices], draw the (t, n) Gaussian sketch from
    sketch_seed, evaluate the cited source's specialised Frobenius
    residual-bound factor on this column subset and this sketch, recover
    the middle matrix of the cited reconstruction from A, C and X, confirm
    that (C M C^T)[row, col] is finite, and return the residual-bound
    factor. The factor is finite and strictly greater than one when the
    source's rank assumption holds.

    Parameters
    ----------
    n : int
        Dimension, n >= 2.
    s : np.ndarray
        Eigenvalues of length n with both a positive and a negative entry.
    data_seed : int
        Seed for the matrix construction.
    sketch_seed : int
        Seed for the Gaussian sketch.
    indices : np.ndarray
        Distinct 0-based column indices, length r with 1 <= r < n.
    t : int
        Sketch height with r < t < n.
    row, col : int
        0-based indices in 0, ..., n-1.

    Returns
    -------
    value : float
        The specialised Frobenius residual-bound factor.

    Raises
    ------
    ValueError
        If t is not an integer or t <= r, if the residual-bound factor of
        the sketch on the column span is not finite and strictly greater
        than one, if the reconstruction entry is not finite, and any
        ValueError raised by the earlier steps for their invalid inputs
        (for example s with no negative entry, n < 2, repeated or
        out-of-range indices, t >= n, or a rank-deficient sketched
        column block).
    """
    return 0.0
```
