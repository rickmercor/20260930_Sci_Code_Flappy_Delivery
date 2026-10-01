# Mathematics-Numerical_Linear_Algebra-31

## Background

Backward error measures how much a linear system must be perturbed for a computed iterate to become exact, so it remains informative for severely ill-conditioned systems. Recent work derives conditioning-independent convergence for positive-semidefinite problems and obtains finite Krylov-space minimizers from reduced singular-value problems, using a symmetric recurrence for MINBERR and Golub--Kahan bidiagonalization for MINBERR-NE.

## Problem

Consider the matrix-only relative backward error $\operatorname{berr}_{A,b}(x)=\lVert Ax-b\rVert_2/(\lVert A\rVert_2\lVert x\rVert_2)$ and the paper's direct MINBERR and normal-equation MINBERR-NE Krylov constructions. Set $n=16$ and $\kappa=10^{12}$, define the orthogonal cosine basis by $U_{i0}=1/\sqrt{n}$ and $U_{ij}=\sqrt{2/n}\cos(\pi(i+1/2)j/n)$ for zero-based $i$ and $j>0$, and form $A=U\operatorname{diag}(\kappa^{-j/(n-1)})U^\top$. Define $c_j=1+0.25\cos(\pi(j+1)/(n+1))$ and $b=Uc/\lVert c\rVert_2$, so that $\lVert A\rVert_2=\lVert b\rVert_2=1$.

For the baseline, start from zero and perform at most seven fixed updates $x\leftarrow x+(b-Ax)/\lVert A\rVert_2$. For direct MINBERR, use the source's symmetric Krylov reduction with one full reorthogonalization at each step, update only the final column of Algorithm 4.1's upper-banded Cholesky factor of $\widetilde T_j^\top\widetilde T_j-10^{-10}I$, include the first rejected prefix, and otherwise stop after dimension seven. At every retained dimension, use exactly 30 inverse iterations from `numpy.random.default_rng(1729 + j).standard_normal(j)`, orient the unit vector so its first component of magnitude above $10^{-14}$ is positive, and scale it to cancel the leading residual coordinate. In parallel, apply the source's MINBERR-NE Golub--Kahan reduction for the same number of retained entries, use exactly 30 inverse iterations on its upper-bidiagonal reduced operator from `numpy.random.default_rng(11729 + j).standard_normal(j + 1)`, and recover each full-space iterate with the normal-equation residual-cancelling scale. Report the final MINBERR-NE backward error divided by the final direct-MINBERR backward error.

For numerical stability in MINBERR-NE, perform one full modified Gram--Schmidt reorthogonalization pass on every new left and right residual against all previously stored vectors of the same basis, in creation order, before computing its norm and normalizing. Include the initial left residual; use the post-reorthogonalization norms as the bidiagonal coefficients. This is the task's finite-precision stabilization of the source's reduction.

In the reasoning, identify both paper-specific reduced operators and scale formulas, state the first shifted-Cholesky rejection, and report the direct recurrence diagnostics, final direct singular data, all three backward-error histories, and the final ratio.

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

01_compute_relative_backward_error

Goal
----
Evaluate the matrix-only relative backward error of a candidate linear-system iterate.

```python
import numpy as np

def compute_relative_backward_error(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
) -> float:
    """Return ``||A x - b||_2 / (||A||_2 ||x||_2)``.

    Parameters
    ----------
    A : np.ndarray
        Finite, nonzero square matrix of shape ``(n, n)``.
    b : np.ndarray
        Finite right-hand side of shape ``(n,)``.
    x : np.ndarray
        Finite nonzero candidate solution of shape ``(n,)``.

    Returns
    -------
    float
        The nonnegative matrix-only relative backward error.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, an input is non-finite, or ``A`` or
        ``x`` has zero 2-norm.
    """
    return result
```

### Step 2

02_construct_controlled_psd_system

Goal
----
Construct the deterministic positive-definite spectral system used by the benchmark.

```python
import numpy as np

def construct_controlled_psd_system(
    n: int,
    condition_number: float,
) -> np.ndarray:
    """Return the augmented array ``[A | b]`` for a controlled spectral system.

    Let ``U`` be the orthogonal cosine basis with

    ``U[i, 0] = 1 / sqrt(n)`` and
    ``U[i, j] = sqrt(2/n) cos(pi (i + 1/2) j / n)`` for ``j > 0``.

    Define ``lambda[j] = condition_number**(-j/(n-1))`` and
    ``c[j] = 1 + 0.25 cos(pi (j+1)/(n+1))``. Form
    ``A = U diag(lambda) U.T`` and ``b = U c / ||c||_2``.

    Parameters
    ----------
    n : int
        System dimension, at least 4. Boolean values are invalid.
    condition_number : float
        Finite target condition number strictly greater than 1.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, n+1)`` whose first ``n`` columns are the
        symmetric positive-definite matrix ``A`` and whose last column is
        the unit-norm vector ``b``.

    Raises
    ------
    ValueError
        If ``n`` is not an integer at least 4 (booleans are invalid), or if
        ``condition_number`` is not finite and strictly greater than 1.
    """
    return result
```

### Step 3

03_build_orthogonal_recurrence

Goal
----
Build the reorthogonalized symmetric Krylov basis and finite recurrence matrix.

```python
import numpy as np

def build_orthogonal_recurrence(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable ``Q`` and ``T`` satisfying ``A Q[:, :k] = Q T``.

    Initialize ``q_prev = 0``, ``q = b / ||b||_2``, and ``beta = 0``.
    For columns ``j = 0, ..., k-1``, compute

    ``w = A q - beta q_prev``, ``alpha = q.T w``, and
    ``w = w - alpha q``. Reorthogonalize once against all columns already
    stored in ``Q`` by replacing ``w`` with
    ``w - Q[:, :j+1] @ (Q[:, :j+1].T @ w)``. Then set
    ``beta_next = ||w||_2`` and ``q_next = w / beta_next``.

    Store ``alpha`` on row ``j`` of column ``j`` in ``T``, the previous
    ``beta`` on row ``j-1`` when ``j > 0``, and ``beta_next`` on row
    ``j+1``. The off-diagonal coefficients are nonnegative by construction.

    Parameters
    ----------
    A : np.ndarray
        Finite symmetric matrix of shape ``(n, n)``.
    b : np.ndarray
        Finite nonzero vector of shape ``(n,)``.
    iterations : int
        Number ``k`` of recurrence columns, with ``1 <= k < n``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``Q`` with shape ``(n, k+1)`` and orthonormal columns, and ``T``
        with shape ``(k+1, k)``.

    Raises
    ------
    ValueError
        If ``A`` is not a nonempty square matrix, if ``A`` or ``b`` is
        non-finite or their shapes are incompatible, if ``A`` is not
        symmetric, if ``iterations`` does not satisfy ``1 <= iterations < n``,
        if ``b`` is zero, or if the recurrence breaks down before the
        requested iteration.
    """
    return result
```

### Step 4

04_extract_reduced_operator

Goal
----
Extract the paper-specific reduced operator and reproduce Algorithm 4.1's incremental, shifted upper-banded Cholesky state.

```python
import numpy as np

def extract_reduced_operator(
    T: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the reduced operator and its incremental Cholesky state.

    Delete the first row of the ``(k+1)``-by-``k`` symmetric recurrence
    matrix to obtain the paper's square upper-triangular matrix ``reduced``.
    It has upper bandwidth two. Form

    ``G = reduced.T @ reduced - threshold**2 * I``

    and update the upper-banded Cholesky factorization ``G = R.T @ R`` one
    prefix at a time. For prefix ``j``, form only the final three potentially
    nonzero entries ``G[i, j]`` for ``max(0, j-2) <= i <= j`` directly from
    column dot products of ``reduced``. Then compute only the final column of
    ``R``. Do not materialize the full Gram matrix and do not call a dense
    Cholesky factorization, eigensolver, singular-value decomposition, matrix
    inverse, or linear solver. Stop at the first diagonal pivot that is not
    greater than
    ``64 * eps * max(1, ||reduced||_inf**2, threshold**2)``.

    Return ``(reduced, certificate, R)``. The certificate has length ``k+2``:
    ``certificate[0]`` is the zero-based rejected-pivot index, or ``k`` when
    all pivots are accepted; ``certificate[1]`` is the rejected pivot, or the
    final accepted pivot when the factorization succeeds; the remaining
    entries are ``diag(R)``, with zeros from the first rejected pivot onward.
    ``R`` is the upper-triangular factor of shape ``(k, k)`` accumulated up to
    the rejection; its uncomputed entries are zero. This factor is part of the
    required state because Algorithm 4.1 updates its last column rather than
    refactorizing every prefix.

    Parameters
    ----------
    T : np.ndarray
        Finite recurrence matrix of shape ``(k+1, k)`` with ``k >= 1`` whose
        first-row deletion is upper triangular with upper bandwidth two.
    threshold : float
        Finite nonnegative singular-value threshold. Boolean values are
        invalid.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The ``(k, k)`` reduced operator, the length-``k+2`` certificate, and
        the ``(k, k)`` upper-banded factor state.

    Raises
    ------
    ValueError
        If ``T`` does not have shape ``(k+1, k)`` with ``k >= 1``, if ``T``
        is not finite, if ``threshold`` is not finite and nonnegative
        (booleans included), or if the first-row deletion of ``T`` is not
        upper triangular with upper bandwidth two.
    """
    return result
```

### Step 5

05_approximate_smallest_singular_pair

Goal
----
Approximate the smallest right singular pair of the paper's banded reduced operator without using dense factorizations.

```python
import numpy as np

def approximate_smallest_singular_pair(
    reduced: np.ndarray,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return ``[sigma, v...]`` after structured inverse iterations.

    ``reduced`` is the paper's square matrix obtained by deleting the first
    row of the extended symmetric recurrence. It is upper triangular with at
    most two nonzero superdiagonals. Draw ``v`` from
    ``np.random.default_rng(seed).standard_normal(k)`` and normalize it.

    Repeat exactly ``inverse_steps`` times. First solve
    ``reduced.T @ z = v`` by forward substitution using only the diagonal and
    two subdiagonals of ``reduced.T``. Then solve ``reduced @ v = z`` by
    backward substitution using only the diagonal and two superdiagonals of
    ``reduced``. Normalize after every pair of solves. Do not call a dense
    linear solver, matrix inverse, eigensolver, singular-value decomposition,
    or Cholesky factorization.

    After the iterations, orient ``v`` so its first component whose magnitude
    exceeds ``1e-14`` is positive, and set
    ``sigma = ||reduced @ v||_2``.

    Parameters
    ----------
    reduced : np.ndarray
        Finite nonsingular upper-triangular matrix of shape ``(k, k)`` with
        upper bandwidth two.
    inverse_steps : int
        Positive number of fixed iterations. Boolean values are invalid.
    seed : int
        Seed passed to ``np.random.default_rng``. Boolean values are invalid.

    Returns
    -------
    np.ndarray
        Vector of shape ``(k+1,)`` containing ``sigma`` followed by the
        deterministically oriented unit vector ``v``.

    Raises
    ------
    ValueError
        If ``reduced`` is not a nonempty, finite, square matrix, if it is
        not upper triangular with upper bandwidth two, or if it is singular;
        if ``inverse_steps`` is not a positive integer (booleans included);
        if ``seed`` is not an integer (booleans included); or if an inverse
        iteration produces an invalid (non-finite or non-normalizable)
        vector.
    """
    return result
```

### Step 6

06_compute_normal_equation_history

Goal
----
Compute the MINBERR-NE backward-error history from the paper's Golub--Kahan reduction.

```python
import numpy as np

def compute_normal_equation_history(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return the MINBERR-NE backward-error history.

    Start the Golub--Kahan process with ``u_1 = b / ||b||_2`` and
    ``alpha_1 v_1 = A.T @ u_1``. For every one-based step ``j``, append
    ``u_(j+1)`` and ``v_(j+1)`` using

    ``beta_(j+1) u_(j+1) = A @ v_j - alpha_j u_j`` and
    ``alpha_(j+1) v_(j+1) = A.T @ u_(j+1) - beta_(j+1) v_j``.

    Before taking each residual norm and normalizing, perform one full
    modified Gram--Schmidt pass against all previously stored vectors of
    the same (left or right) basis, in their creation order. Include the
    initial left update against ``u_1``. The post-reorthogonalization norms
    define the nonnegative ``alpha`` and ``beta`` coefficients.

    After forming ``beta_(j+2) u_(j+2)`` in the same way, build the
    ``(j+1)``-by-``(j+1)`` upper-bidiagonal reduced matrix whose diagonal is
    ``beta_2, ..., beta_(j+2)`` and whose superdiagonal is
    ``alpha_2, ..., alpha_(j+1)``. Call
    ``approximate_smallest_singular_pair`` on this matrix with seed
    ``seed + j`` and the supplied fixed iteration count. If ``y`` is the
    returned right singular-vector estimate and ``V`` contains
    ``v_1, ..., v_(j+1)``, use the paper's scale

    ``scale = ||b||_2**2 / ((b.T @ A) @ (V @ y))``

    and evaluate the full matrix-only relative backward error of
    ``x = V @ (scale * y)``. Do not replace the Golub--Kahan reduction by a
    normal-equations eigensolve, SVD, or dense least-squares solve.

    Parameters
    ----------
    A : np.ndarray
        Finite nonzero square matrix of shape ``(n, n)``.
    b : np.ndarray
        Finite nonzero vector of shape ``(n,)``.
    iterations : int
        Number of history entries, with ``1 <= iterations < n``.
    inverse_steps : int
        Positive fixed iteration count for every reduced problem.
    seed : int
        Base integer seed; reduced problem ``j`` uses ``seed + j``.

    Returns
    -------
    np.ndarray
        Array of shape ``(iterations,)`` containing the full-space
        MINBERR-NE backward errors.

    Raises
    ------
    ValueError
        If ``A`` is not a nonempty square matrix, if ``A`` or ``b`` is
        non-finite or their shapes are incompatible, if ``A`` or ``b`` is
        zero, if ``iterations`` does not satisfy ``1 <= iterations < n``, if
        ``inverse_steps`` is not a positive integer, if ``seed`` is not an
        integer, or if the Golub--Kahan recursion breaks down at any step
        (initial right/left vector or a later right/left update) or the
        MINBERR-NE direction cannot be scaled.
    """
    return result
```

### Step 7

07_compute_error_histories

Goal
----
Compute the fixed-update, direct-MINBERR, and MINBERR-NE backward-error histories.

```python
import numpy as np

def compute_error_histories(
    A: np.ndarray,
    b: np.ndarray,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> np.ndarray:
    """Return three finite-iteration backward-error histories.

    Row 0 uses ``x_0 = 0`` and the fixed updates
    ``x_j = x_(j-1) + (b - A x_(j-1)) / ||A||_2``.

    Row 1 uses the symmetric recurrence through each dimension ``j``. At
    every dimension, call ``extract_reduced_operator(Tj, 1e-5)`` and use its
    reduced operator in ``approximate_smallest_singular_pair`` with seed
    ``seed + j``. If ``v`` is the returned direction, set
    ``scale = ||b||_2 / (Tj[0, :] @ v)`` and evaluate the full-space error of
    ``Qj[:, :j] @ (scale * v)``. Include the first prefix whose shifted
    Cholesky certificate rejects a pivot, then stop both rows 0 and 1 there.

    Row 2 must be the first ``m`` entries returned by
    ``compute_normal_equation_history(A, b, m, inverse_steps, seed + 10000)``,
    where ``m`` is the number of retained direct-MINBERR prefixes. This keeps
    the normal-equation Golub--Kahan comparison independent of the symmetric
    inverse-iteration starts while using the same number of entries.

    Parameters
    ----------
    A : np.ndarray
        Finite symmetric matrix of shape ``(n, n)`` with nonzero 2-norm.
    b : np.ndarray
        Finite nonzero vector of shape ``(n,)``.
    iterations : int
        Maximum number of history entries, with ``1 <= iterations < n``.
    inverse_steps : int
        Positive fixed iteration count for every reduced problem.
    seed : int
        Base integer seed for the reduced iterations.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, m)``, where ``m <= iterations`` is the first
        rejected symmetric prefix or ``iterations`` if no prefix rejects.
        The rows contain fixed-update, direct-MINBERR, and MINBERR-NE errors.

    Raises
    ------
    ValueError
        If ``A`` is not a nonempty square matrix, if ``A`` or ``b`` is
        non-finite or their shapes are incompatible, if ``A`` is not
        symmetric, if ``A`` or ``b`` is zero, if ``iterations`` does not
        satisfy ``1 <= iterations < n``, if ``inverse_steps`` is not a
        positive integer, if ``seed`` is not an integer, or if the shifted
        Cholesky certificate/factor state is invalid or the direct-MINBERR
        direction cannot be scaled.
    """
    return result
```

### Step 8

08_run_backward_error_benchmark

Goal
----
Run the deterministic benchmark and return the final MINBERR-NE-over-direct ratio.

```python
import numpy as np

def run_backward_error_benchmark(
    n: int,
    condition_number: float,
    iterations: int,
    inverse_steps: int,
    seed: int,
) -> float:
    """Return the final MINBERR-NE error divided by the direct error.

    Construct the prescribed augmented positive-definite system, compute the
    fixed-update, direct-MINBERR, and MINBERR-NE histories, and return
    ``history[2, -1] / history[1, -1]``.

    Parameters
    ----------
    n : int
        System dimension, at least 4.
    condition_number : float
        Finite target condition number strictly greater than 1.
    iterations : int
        Number of history entries, with ``1 <= iterations < n``.
    inverse_steps : int
        Positive fixed iteration count for every reduced problem.
    seed : int
        Base integer seed for the reduced iterations.

    Returns
    -------
    float
        Positive final MINBERR-NE-over-direct error ratio.

    Raises
    ------
    ValueError
        If any earlier-stage validation fails for the constructed system or
        histories (invalid ``n``, ``condition_number``, ``iterations``,
        ``inverse_steps``, or ``seed``, per the earlier steps' contracts), or
        if the final minimum backward error or the final normal-equation
        backward error is not positive.
    """
    return result
```
