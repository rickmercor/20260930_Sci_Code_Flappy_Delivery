# Mathematics-Numerical_Linear_Algebra-21

## Background

Randomized projected solvers of this kind combine progressive sketching with randomized row- or column-based actions, allowing each iteration to use limited matrix information while retaining information from earlier correction directions. Such constructions are useful in large, sparse, ill-conditioned, or incrementally accessed linear systems, where repeatedly processing the full matrix can be costly.

Iterative least-squares computations are sensitive to conditioning. Algebraically equivalent formulas can behave differently in finite precision when nearly equal quantities are subtracted, so stable linear solves and least-squares projections are often preferable to explicit inverses in numerically delicate cases. The distinction matters when a deterministic numerical experiment requires a reproducible scalar at high precision.

## Problem

Randomized projected methods for large linear systems can reduce the amount of matrix information required at each iteration while retaining information from previously generated correction directions, making them useful for inconsistent systems and ill-conditioned problems. Consider the following full-column-rank inconsistent system:

$$
A =
\begin{bmatrix}
1.0 & 2.001 & -0.999 & 0.501 & 1.499 \\
2.0 & 3.999 & -2.001 & 1.002 & 3.001 \\
-1.0 & -2.002 & 1.0005 & -0.499 & -1.498 \\
0.5 & 1.0015 & -0.4995 & 0.2505 & 0.749 \\
3.0 & 5.999 & -2.998 & 1.501 & 4.502 \\
-2.0 & -4.001 & 1.999 & -1.0005 & -2.999 \\
1.2 & 2.401 & -1.1995 & 0.6004 & 1.799 \\
-0.7 & -1.399 & 0.699 & -0.3502 & -1.049
\end{bmatrix}.
$$

with

$$
x_{\rm ref} =
\begin{bmatrix}
0.4 \\
-0.8 \\
1.1 \\
-0.6 \\
0.9
\end{bmatrix},
\qquad
\eta =
\begin{bmatrix}
0.12 \\
-0.07 \\
0.05 \\
0.09 \\
-0.11 \\
0.08 \\
-0.04 \\
0.06
\end{bmatrix},
\qquad
x_0 =
\begin{bmatrix}
0.15 \\
-0.2 \\
0.35 \\
-0.1 \\
0.25
\end{bmatrix}.
$$

Define

$$
b = Ax_{\rm ref} + \eta.
$$

Treating every supplied entry as an exact decimal value, and exactly one `numpy.random.default_rng(271828)` instance, reproduce the residual-informed column-action variant of the randomized projected solver for four successive iterations, using `rng.choice(5, size=2, replace=False)` at each iteration with the same generator instance and retaining the accumulated correction directions throughout the computation. All numerical conventions are fixed by the statement, including the random generator, sampling rule, initial iterate, and iteration count. Carry out every arithmetic operation exactly (rational arithmetic, or at least 50 significant digits) with no intermediate rounding; the matrix is severely ill-conditioned, so binary64 floating-point evaluation does not determine the result to the requested precision. Show the key intermediate numerical quantities used to determine the final scalar, including the sampled column blocks and the update or residual quantities needed to justify the fourth-iteration result.

Compute the squared Euclidean residual of the fourth iterate,

$$
R_4 = \|Ax_4-b\|_2^2,
$$

and report the resulting deterministic scalar.

Return $$R_4$$ rounded to 8 decimal places. In the reasoning, also state the unrounded value of $R_4$ to at least 10 significant digits.

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

01_prepare_weight_matrix

Goal
----
Prepare the stable least-squares operator used by the numerical procedure.

```python
def prepare_weight_matrix(A: np.ndarray) -> np.ndarray:
    """Construct the stable least-squares operator A^+.

    Parameters
    ----------
    A : np.ndarray
        Finite matrix of shape (m, n), with m >= n and full column rank.

    Returns
    -------
    np.ndarray
        A float64 matrix with shape (n, m) representing the least-squares
        operator A^+.

    Raises
    ------
    ValueError
        If A is not two-dimensional, has fewer rows than columns, has zero
        columns, contains non-finite values, or is not full column rank.

    Notes
    -----
    The result is deterministic for identical inputs. The implementation should
    avoid forming A.T @ A and its explicit inverse.
    """
    return result
```

### Step 2

02_generate_column_blocks

Goal
----
Generate the deterministic sequence of sampled column blocks.

```python
import numpy as np

def generate_column_blocks(
    n_cols: int,
    block_size: int,
    n_iters: int,
    seed: int = 271828,
) -> np.ndarray:
    """Generate sequential random column-index blocks.

    Parameters
    ----------
    n_cols : int
        Number of available columns.
    block_size : int
        Number of distinct columns sampled per iteration.
    n_iters : int
        Number of iterations.
    seed : int
        Seed for numpy default_rng.

    Returns
    -------
    np.ndarray
        Integer array of shape (n_iters, block_size).

    Raises
    ------
    ValueError
        If n_cols is not positive, block_size is outside the range
        1 through n_cols, n_iters is not positive, or seed is not an
        integer.

    Notes
    -----
    The result is deterministic for identical inputs.
    """
    return result
```

### Step 3

03_form_residual_sketch

Goal
----
Construct the current residual quantities and the sampled residual-weighted sketch.

```python
def form_residual_sketch(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the current residual, normal-equation residual, and sketch.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix.
    b : np.ndarray
        Right-hand-side vector.
    x : np.ndarray
        Current iterate.
    tau : np.ndarray
        Distinct sampled column indices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The residual, normal-equation residual, and sampled sketch.

    Raises
    ------
    ValueError
        If the inputs have incompatible shapes, tau is empty, tau contains
        non-integer or invalid indices, tau contains duplicates, or any input
        contains non-finite values.

    Notes
    -----
    The result is deterministic for identical inputs.
    """
    return result
```

### Step 4

04_project_historical_direction

Goal
----
Project the current sketch direction against previously generated corrections.

```python
import numpy as np

def project_historical_direction(
    A: np.ndarray,
    A_plus: np.ndarray,
    P: np.ndarray,
    s: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project the current direction against historical corrections.

    Parameters
    ----------
    A : np.ndarray
        Coefficient matrix of shape (m, n).
    A_plus : np.ndarray
        Stable least-squares operator of shape (n, m) prepared from A.
    P : np.ndarray
        Previously generated correction directions of shape (n, r).
    s : np.ndarray
        Current sketch vector of shape (m,).

    Returns
    -------
    tuple[np.ndarray, float]
        The projected direction, and the historical correction scalar. Writing
        ``c`` for the coefficients of the least-squares fit of the sketch in the
        basis ``A @ P``, the projected direction is the refined least-squares
        direction less ``P @ c``, and the scalar is the squared Euclidean norm
        of ``A @ P @ c``. When the historical space is empty the projected
        direction is the refined least-squares direction itself and the scalar
        is 0.0.

    Raises
    ------
    ValueError
        If the inputs have incompatible shapes, contain non-finite values,
        A does not have full column rank, the historical space is rank-deficient,
        or the historical subspace has exhausted the available dimension.

    Notes
    -----
    The result is deterministic for identical inputs. The current least-squares
    direction is formed from A_plus @ s and refined once using the same operator
    on its residual before the historical projection is applied.
    """
    return result
```

### Step 5

05_compute_projected_update

Goal
----
Compute the scalar-normalized correction from the current projected direction.

```python
def compute_projected_update(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    s: np.ndarray,
    projected_direction: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, float]:
    """Compute the correction and scalar multiplier.

    ``delta`` is retained as the historical projection checkpoint. For stable
    numerical evaluation, use ``D = ||A @ projected_direction||_2^2`` directly
    and do not subtract nearly equal weighted quantities.

    Returns
    -------
    tuple[np.ndarray, float]
        The correction vector p and scalar multiplier gamma.

    Raises
    ------
    ValueError
        If inputs are invalid, the projected direction has zero image under
        A, or the multiplier/correction becomes non-finite.

    Notes
    -----
    The result is deterministic for identical inputs.
    """
    return result
```

### Step 6

06_update_iterate_state

Goal
----
Update the current iterate and append the newly computed correction direction.

```python
def update_iterate_state(
    x: np.ndarray,
    P: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Update the iterate and append the new correction direction.

    Parameters
    ----------
    x : np.ndarray
        Current iterate of shape (n,).
    P : np.ndarray
        Historical correction matrix of shape (n, r).
    p : np.ndarray
        New correction direction of shape (n,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated iterate x_new and updated historical matrix P_new.

    Raises
    ------
    ValueError
        If x or p is not one-dimensional, p and x have different shapes,
        P is not two-dimensional with a compatible row count, an input
        contains non-finite values, the correction is zero, the historical
        correction space has exhausted the available dimension, or the
        correction is numerically dependent on the historical correction
        space.

    Notes
    -----
    The result is deterministic for identical inputs.
    """
    return result
```

### Step 7

07_run_rplss_gcd

Goal
----
Run the complete deterministic randomized projected numerical experiment.

```python
def run_rplss_gcd(
    A: np.ndarray,
    x_ref: np.ndarray,
    eta: np.ndarray,
    x0: np.ndarray,
    seed: int = 271828,
    n_iters: int = 4,
    block_size: int = 2,
) -> float:
    """Run the complete deterministic numerical experiment.

    Parameters
    ----------
    A : np.ndarray
        Full-column-rank coefficient matrix of shape (m, n).
    x_ref : np.ndarray
        Reference vector used to construct the right-hand side.
    eta : np.ndarray
        Additive inconsistency vector.
    x0 : np.ndarray
        Initial iterate.
    seed : int
        Seed for the NumPy random number generator.
    n_iters : int
        Number of correction iterations.
    block_size : int
        Number of columns sampled at each iteration.

    Returns
    -------
    float
        Final squared Euclidean residual from the stable float64 recurrence.
        The exact-arithmetic benchmark value is specified by the golden solution.

    Raises
    ------
    ValueError
        If the inputs are invalid, the matrix does not define a valid
        full-column-rank problem, or the requested iteration count would
        exhaust the available correction-space dimension.

    Notes
    -----
    For n=1, one iteration is permitted because no historical correction exists
    before the first update. Otherwise the maximum supported iteration count is n-1.
    """
    return result
```
