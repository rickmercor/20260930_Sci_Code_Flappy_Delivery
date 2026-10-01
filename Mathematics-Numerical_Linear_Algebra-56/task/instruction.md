# Mathematics-Numerical_Linear_Algebra-56

## Background

RCGLS is a randomized conjugate least-squares method in which sketch information and conjugate-direction structure are coupled by a method-specific recurrence. Several familiar conjugate-gradient-style coefficients are dimensionally compatible with the same state variables, but they need not define the same randomized method. Separately, random-orthonormal sketch-and-solve has sharp finite-dimensional residual-inflation benchmarks that depend on the underlying field. This task requires resolving the RCGLS transition before comparing its deterministic conditioned trajectory with those benchmarks.

## Problem

Randomized sketching can be used either inside an iterative least-squares solver or to compress an entire least-squares problem. Here the iterative part contains a methodological ambiguity that must be resolved from the RCGLS definition before the numerical trajectory is evaluated. Given
$$
A=\left[\begin{matrix}
-0.98912135 & -0.36778665 & 1.28792526 & 0.19397442 \\
 0.92023090 & 0.57710379 & -0.63646365 & 0.54195222 \\
-0.31659545 & -0.32238912 & 0.09716732 & -1.52593041 \\
1.19216610 & -0.67108968 & 1.00026942 & 0.13632112 \\
1.53203308 & -0.65996941 & -0.31179486 & 0.33776913 \\
-2.20747110 & 0.82792144 & 1.54163039 & 1.12680679 \\
0.75476964 & -0.14597789 & 1.28190223 & 1.07403062 \\
0.39262084 & 0.00511431 & -0.36176687 & -1.23023220
\end{matrix}\right],\qquad
b=\left[\begin{matrix}
1.22622929 \\
-2.17204389 \\
-0.37014735 \\
0.16438007 \\
0.85988118 \\
1.76166124 \\
0.99332378 \\
-0.29152143
\end{matrix}\right],
$$
start from $x_0=0\in\mathbb{R}^4$ and condition on the realized one-based coordinate stream $(3,1,2,3,4,1)$. Use IEEE-754 binary64 arithmetic and do not stop early.

At each transition after the first update, consider two dimensionally valid conjugate-direction constructions built from the next coordinate-sketch gradient seed. **Candidate A** forms its correction coefficient from inner products involving the $A$-images of the new seed and preceding search direction. **Candidate B** forms its correction coefficient from a Fletcher--Reeves-style ratio of the squared norms of successive coordinate-sketch gradient seeds. Determine which candidate is the coordinate-sketch specialization of RCGLS. The choice must come from the RCGLS method, not from which candidate reduces the residual more on this instance. Use only the selected construction for the six-update trajectory.

Let $\rho_k=\|b-Ax_k\|_2^2/\min_x\|b-Ax\|_2^2$. For the same $A$ and $b$, evaluate the sharp finite-dimensional expected squared-residual inflation for sketch-and-solve with a uniformly random orthonormal embedding of dimension $\ell=7$ over both $\mathbb{R}$ and $\mathbb{C}$. For each field $\mathbb K\in\{\mathbb R,\mathbb C\}$, let $k_\star^{(\mathbb K)}$ be the smallest $k\in\{1,\ldots,6\}$ such that $\rho_k\le\rho_{\mathrm{orth}}^{(\mathbb K)}$, and define
$$
M_{\mathbb K}=\frac{\rho_{k_\star^{(\mathbb K)}}}{\rho_{\mathrm{orth}}^{(\mathbb K)}}.
$$
Report the method-certified field-sensitivity ratio
$$
Q=\frac{M_{\mathbb R}}{M_{\mathbb C}},
$$
rounded to 8 digits past the decimal point. In the reasoning, identify the admissible candidate and give only the scalar quantities needed to verify the choice and both first crossings. Do not print full iterates or residual vectors.

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

step1_prepare_least_squares_problem

Goal
----
Prepare the common least-squares inputs.

```python
import numpy as np

def prepare_least_squares_problem(
    A: np.ndarray, b: np.ndarray, x0: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the common least-squares inputs.

    Parameters
    ----------
    A : np.ndarray
        Two-dimensional real matrix with shape ``(n, d)``.
    b : np.ndarray
        Real right-hand side with shape ``(n,)``.
    x0 : np.ndarray
        Real initial iterate with shape ``(d,)``.

    Returns
    -------
    A64 : np.ndarray
        Finite float64 matrix with shape ``(n, d)``.
    b64 : np.ndarray
        Finite float64 vector with shape ``(n,)``.
    x064 : np.ndarray
        Finite float64 vector with shape ``(d,)``.

    Raises
    ------
    ValueError
        If conversion fails, dimensions are inconsistent, an array is empty,
        or any value is nonfinite.
    """
    return A64, b64, x064
```

### Step 2

step2_normalized_coordinate_sketch

Goal
----
Form the sketch associated with a selected coordinate.

```python
import numpy as np

def normalized_coordinate_sketch(A: np.ndarray, index: int) -> np.ndarray:
    """Construct a one-column normalized coordinate sketch vector.

    Parameters
    ----------
    A : np.ndarray
        Finite real matrix with shape ``(n, d)``.
    index : int
        One-based coordinate index in ``{1, ..., d}``.

    Returns
    -------
    S : np.ndarray
        Float64 vector of shape ``(d,)`` representing the normalized selected
        coordinate.

    Raises
    ------
    ValueError
        If ``A`` is invalid, ``index`` is not a valid one-based integer, or
        the selected column of ``A`` has zero norm.
    """
    return S
```

### Step 3

step3_rcgls_sketch_seed

Goal
----
Form the RCGLS information associated with the current sketch.

```python
import numpy as np

def rcgls_sketch_seed(A: np.ndarray, r: np.ndarray, S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Form the RCGLS information associated with the current sketch.

    Parameters
    ----------
    A : np.ndarray
        Finite real matrix with shape ``(n, d)``.
    r : np.ndarray
        Finite residual vector with shape ``(n,)``.
    S : np.ndarray
        Finite sketch matrix with shape ``(d, q)`` or a finite sketch vector
        with shape ``(d,)``.

    Returns
    -------
    seed : np.ndarray
        Float64 coefficient-space RCGLS sketched-gradient seed, shape ``(d,)``.
    image : np.ndarray
        Float64 vector ``A @ seed`` with shape ``(n,)``.

    Raises
    ------
    ValueError
        If conversion fails, dimensions are inconsistent, any value is
        nonfinite, the sketch has zero columns, or the resulting seed/image is
        zero or nonfinite.
    """
    return seed, image
```

### Step 4

step4_rcgls_conjugate_direction

Goal
----
Construct the RCGLS conjugate transition.

```python
import numpy as np

def rcgls_conjugate_direction(
    A: np.ndarray,
    r_next: np.ndarray,
    S_next: np.ndarray,
    p_prev: np.ndarray,
    v_prev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float, float, float]:
    """Construct the RCGLS conjugate transition.

    Parameters
    ----------
    A : np.ndarray
        Finite real matrix with shape ``(n, d)``.
    r_next : np.ndarray
        Finite residual after the preceding RCGLS update, shape ``(n,)``.
    S_next : np.ndarray
        Finite next sketch matrix with shape ``(d, q)`` or sketch vector with
        shape ``(d,)``.
    p_prev : np.ndarray
        Finite preceding RCGLS search direction with shape ``(d,)``.
    v_prev : np.ndarray
        Finite preceding search-direction image ``A @ p_prev``, shape ``(n,)``.

    Returns
    -------
    p_next : np.ndarray
        Corrected next RCGLS search direction with shape ``(d,)``.
    v_next : np.ndarray
        Image ``A @ p_next`` with shape ``(n,)``.
    tau : float
        Signed RCGLS conjugacy coefficient in the convention
        ``p_next = seed + tau * p_prev``.
    mu_next : float
        RCGLS step coefficient for the selected next direction. For a valid
        RCGLS trajectory state, it coincides with the exact line-search
        coefficient along that direction.
    r_after : np.ndarray
        Residual after applying the RCGLS step.
    residual_sq : float
        Squared Euclidean norm of ``r_after``.
    conjugacy_defect : float
        Inner product ``v_next @ v_prev``; it should vanish up to roundoff.
    line_search_defect : float
        Residual/search-image inner product after the update; it vanishes up
        to roundoff for a valid RCGLS trajectory state.

    Raises
    ------
    ValueError
        If inputs are invalid, ``v_prev`` is inconsistent with ``A @ p_prev``,
        a required denominator vanishes, or the coupled RCGLS transition is
        nonfinite or degenerate.
    """
    return p_next, v_next, tau, mu_next, r_after, residual_sq, conjugacy_defect, line_search_defect
```

### Step 5

step5_rcgls_residual_history

Goal
----
Build a scale-certified trajectory generated by the selected RCGLS transition.

```python
import numpy as np

def rcgls_residual_history(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    indices: tuple[int, ...] | list[int],
    scale_factors: tuple[float, ...] | list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a scale-certified trajectory generated by the selected RCGLS transition.

    Parameters
    ----------
    A : np.ndarray
        Finite real matrix with shape ``(n, d)``.
    b : np.ndarray
        Finite real vector with shape ``(n,)``.
    x0 : np.ndarray
        Finite real initial iterate with shape ``(d,)``.
    indices : tuple[int, ...] or list[int]
        Nonempty one-based conditioned coordinate stream.
    scale_factors : tuple[float, ...], list[float], or np.ndarray
        Finite nonzero sketch-rescaling factors, one per conditioned update.

    Returns
    -------
    residual_sq : np.ndarray
        Squared residual after each canonical RCGLS update, shape ``(m,)``.
    mu_values : np.ndarray
        Canonical RCGLS step coefficients, shape ``(m,)``.
    tau_values : np.ndarray
        Canonical conjugacy coefficients after the first update, shape ``(m-1,)``.
    scaled_mu_values : np.ndarray
        RCGLS step coefficients for the rescaled sketch stream, shape ``(m,)``.
    scaled_tau_values : np.ndarray
        Conjugacy coefficients for the rescaled sketch stream, shape ``(m-1,)``.
    iterate_defects : np.ndarray
        Euclidean differences between canonical and rescaled iterates after each update, shape ``(m,)``.
    residual_defects : np.ndarray
        Euclidean differences between canonical and rescaled residuals after each update, shape ``(m,)``.

    Raises
    ------
    ValueError
        If inputs, indices, or scale factors are invalid, a required sketch is undefined,
        or either RCGLS trajectory breaks down or becomes nonfinite.
    """
    return residual_sq, mu_values, tau_values, scaled_mu_values, scaled_tau_values, iterate_defects, residual_defects
```

### Step 6

step6_least_squares_orthonormal_reference

Goal
----
Establish the least-squares reference and orthonormal-sketch benchmark.

```python
import numpy as np

def least_squares_orthonormal_reference(
    A: np.ndarray, b: np.ndarray, ell: int, field: str = "real"
) -> tuple[float, int, float]:
    """Establish the least-squares reference and orthonormal-sketch benchmark.

    Parameters
    ----------
    A : np.ndarray
        Finite real matrix with shape ``(n, d)``.
    b : np.ndarray
        Finite real vector with shape ``(n,)``.
    ell : int
        Random-orthonormal sketch-and-solve embedding dimension.
    field : str, default="real"
        Either ``"real"`` or ``"complex"``.

    Returns
    -------
    optimal_residual_sq : float
        Minimum unsketched squared Euclidean residual.
    rank : int
        Numerical rank returned by ``numpy.linalg.lstsq`` with ``rcond=None``.
    rho_orth : float
        Exact finite-dimensional expected squared-residual inflation for the
        random-orthonormal sketch-and-solve benchmark.

    Raises
    ------
    ValueError
        If inputs are invalid, the least-squares solve fails, the optimum is
        zero/nonfinite, or ``n``, ``rank``, ``ell``, and ``field`` violate the
        paper's benchmark domain.
    """
    return optimal_residual_sq, rank, rho_orth
```

### Step 7

step7_rcgls_benchmark_crossing

Goal
----
Identify the first RCGLS attainment of a benchmark.

```python
import numpy as np

def rcgls_benchmark_crossing(
    residual_sq: np.ndarray, optimal_residual_sq: float, benchmark: float
) -> tuple[np.ndarray, int, float, float]:
    """Identify the first RCGLS attainment of a benchmark.

    Parameters
    ----------
    residual_sq : np.ndarray
        Nonempty one-dimensional finite array of nonnegative RCGLS squared
        residuals in update order.
    optimal_residual_sq : float
        Finite strictly positive minimum unsketched squared residual.
    benchmark : float
        Finite strictly positive residual-inflation threshold.

    Returns
    -------
    inflation : np.ndarray
        Float64 residual-inflation history ``residual_sq / optimal_residual_sq``.
    k_star : int
        One-based index of the first inflation not exceeding ``benchmark``.
    crossing_inflation : float
        Inflation factor at ``k_star``.
    margin : float
        Ratio ``crossing_inflation / benchmark``.

    Raises
    ------
    ValueError
        If inputs are invalid or if the RCGLS history never reaches the
        benchmark.
    """
    return inflation, k_star, crossing_inflation, margin
```

### Step 8

step8_rcgls_method_certified_field_sensitivity

Goal
----
Combine a scale-certified RCGLS trajectory with two field benchmarks.

```python
import numpy as np

def rcgls_method_certified_field_sensitivity(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    indices: tuple[int, ...] | list[int],
    ell: int,
) -> float:
    """Combine a scale-certified RCGLS trajectory with two field benchmarks.

    Parameters
    ----------
    A : np.ndarray
        Finite real matrix with shape ``(n, d)``.
    b : np.ndarray
        Finite real vector with shape ``(n,)``.
    x0 : np.ndarray
        Finite real initial iterate with shape ``(d,)``.
    indices : tuple[int, ...] or list[int]
        Realized one-based coordinate stream.
    ell : int
        Random-orthonormal sketch-and-solve embedding dimension.

    Returns
    -------
    value : float
        Ratio of the real and complex first-crossing margins for the selected
        RCGLS trajectory.

    Raises
    ------
    ValueError
        If any component input is invalid, the unsketched optimum has zero
        residual, either benchmark theorem domain is violated, RCGLS breaks
        down, either benchmark is not reached, or the final scalar is nonfinite.
    """
    return value
```
