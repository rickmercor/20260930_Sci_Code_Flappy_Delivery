# Mathematics-Numerical_Linear_Algebra-23

## Background

Polynomial preconditioning transforms the spectrum seen by a Krylov method in order to improve convergence. For an indefinite problem, an unsuitable residual polynomial can leave the transformed spectrum indefinite or can map spectral values very close to the origin, either of which may significantly degrade restarted GMRES.

For a GMRES residual polynomial $\pi$, the source uses

$$
\varphi(z)=1-\pi(z)
$$

as the polynomial applied to the matrix in the preconditioned system. Equivalently,

$$
\varphi(z)=z\,p(z),
$$

where $p$ is the right-preconditioning polynomial.

The roots of $\pi$ are harmonic Ritz values. For indefinite systems, the source introduces balanced polynomial constructions intended to make the transformed spectrum more favorable near the origin.

Balance Method 2 modifies the root set before introducing a balancing root. Its selection procedure treats real harmonic Ritz roots and complex-conjugate pairs differently.

After balancing, the source uses cubic Hermite splines between selected consecutive real roots of the residual polynomial to screen whether the transformed polynomial may lose positivity over the relevant spectral region. The screening uses endpoint derivative information together with the supplied extremal Ritz values.

The harmonic Ritz roots and extremal Ritz values are supplied directly in this task. No Arnoldi iteration, GMRES run, eigenvalue computation, or random starting vector is required.

## Problem

Using the supplied harmonic Ritz roots
$$
\theta_1=-5,\qquad
\theta_2=-1.3,\qquad
\theta_3=0.9,\qquad
\theta_4=2+1.5i,\qquad
\theta_5=2-1.5i,
$$
and the Ritz extrema
$$
\widetilde{\theta}_{\min}=-4.6,\qquad
\widetilde{\theta}_{\max}=0.75,
$$
compute the scalar diagnostic $Q$ obtained by applying Balance Method 2 and the Hermite-spline Ritz screening procedure from the cited source.

Apply the source algorithms exactly, including their treatment of real roots versus complex-conjugate pairs, the construction of the balanced residual polynomial, the eligibility of spline intervals, and the Ritz-extrema screening rules.

For every interval that is flagged by the source screening procedure, its contribution to the diagnostic is the excess of the selected interior Hermite-spline value above $1$. Define $Q$ as the largest such contribution; if no interval is flagged, set $Q=0$.

Use the cited source to determine all source-specific branching rules, interval conventions, and screening conditions that are not explicitly given above. Report $Q$ rounded to exactly 12 digits after the decimal point.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

evaluate_balance_candidates

Goal
----
Evaluate the Balance Method 2 candidates formed by the supplied harmonic Ritz roots, preserving complex-conjugate pairs as indivisible candidates and identifying the candidate closest to the initial derivative quantity.

```python
import numpy as np


def evaluate_balance_candidates(
    harmonic_ritz_roots: np.ndarray,
) -> np.ndarray:
    """
    Evaluate Balance Method 2 candidates from harmonic Ritz roots.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots. Every
        nonreal root must have a complex-conjugate partner.

    Returns
    -------
    np.ndarray
        Candidate table with one row per real root or conjugate pair
        and columns
        [primary_index, partner_index, multiplicity, contribution,
         difference, phi_prime_0, selected_flag].

    Raises
    ------
    ValueError
        If the array is empty or not one-dimensional, contains non-finite
        or zero entries, or a nonreal root has no conjugate partner.
    """
    return None
```

### Step 2

apply_balance_method2

Goal
----
Apply the Balance Method 2 remove-or-keep decision to the harmonic Ritz roots and construct the balanced root set by appending the corresponding balancing root.

```python
import numpy as np


def apply_balance_method2(
    harmonic_ritz_roots: np.ndarray,
) -> np.ndarray:
    """
    Apply Balance Method 2 and construct the balanced root set.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots with
        complex roots occurring in conjugate pairs.

    Returns
    -------
    np.ndarray
        Complex-valued one-dimensional array containing the retained
        original roots in input order followed by the real balancing
        root.

    Raises
    ------
    ValueError
        If the array is empty or not one-dimensional, contains non-finite
        entries, or the balancing denominator is numerically zero.
    """
    return None
```

### Step 3

build_balanced_residual_polynomial

Goal
----
Construct the balanced residual polynomial from the Balance Method 2 root set and return its coefficients in ascending order of polynomial degree.

```python
import numpy as np


def build_balanced_residual_polynomial(
    harmonic_ritz_roots: np.ndarray,
) -> np.ndarray:
    """
    Construct the balanced residual polynomial.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots with
        complex roots occurring in conjugate pairs.

    Returns
    -------
    np.ndarray
        Real one-dimensional coefficient array [c_0, ..., c_m]
        representing pi(alpha) in ascending powers.

    Raises
    ------
    ValueError
        If the array is empty or not one-dimensional, contains non-finite
        entries, or the product coefficients are not real to tolerance.
    """
    return None
```

### Step 4

evaluate_real_root_derivatives

Goal
----
Extract the real roots of the balanced residual polynomial, sort them on the real axis, and evaluate the residual-polynomial derivative at each real root.

```python
import numpy as np


def evaluate_real_root_derivatives(
    harmonic_ritz_roots: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the balanced residual-polynomial derivative at its real roots.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots with
        complex roots occurring in conjugate pairs.

    Returns
    -------
    np.ndarray
        Real array of shape (n_real, 2) with columns
        [sorted_real_root, derivative_value].

    Raises
    ------
    ValueError
        If the array is empty or not one-dimensional, contains non-finite
        entries, or fewer than two distinct real balanced roots exist.
    """
    return None
```

### Step 5

construct_hermite_spline_candidates

Goal
----
Construct the Hermite cubic-spline candidates on the eligible intervals between consecutive real balanced roots and determine the relevant interior critical point and spline value for each examined interval.

```python
import numpy as np


def construct_hermite_spline_candidates(
    real_root_derivatives: np.ndarray,
) -> np.ndarray:
    """
    Construct Hermite spline candidates on eligible real-root intervals.

    Parameters
    ----------
    real_root_derivatives : np.ndarray
        Real array of shape (n_real, 2) with columns
        [sorted_real_root, derivative_value].

    Returns
    -------
    np.ndarray
        Real array with one row per eligible interval and columns
        [left_root, right_root, left_derivative, right_derivative,
         a, b, c, x_hat, C_at_x_hat].

    Raises
    ------
    ValueError
        If the array is not of shape (n_real, 2), has fewer than two rows,
        contains non-finite entries, or roots are not strictly increasing.
    """
    return None
```

### Step 6

apply_ritz_interval_screening

Goal
----
Apply the Ritz-extrema screening procedure of Algorithm 4 to the supplied Hermite-spline candidates and return the screening result for every examined interval.

```python
import numpy as np


def apply_ritz_interval_screening(
    spline_candidates: np.ndarray,
    ritz_min: float,
    ritz_max: float,
) -> np.ndarray:
    """
    Apply Ritz-extrema screening to Hermite spline candidates.

    Parameters
    ----------
    spline_candidates : np.ndarray
        Real array of shape (n_examined, 9) with columns
        [left_root, right_root, left_derivative, right_derivative,
         a, b, c, x_hat, C_at_x_hat].
    ritz_min : float
        Ritz value with the smallest real part.
    ritz_max : float
        Ritz value with the largest real part.

    Returns
    -------
    np.ndarray
        Real array with one row per examined interval and columns
        [left_root, right_root, x_hat, C_at_x_hat,
         condition_1, condition_2, condition_3,
         flagged, exceedance_margin].

    Raises
    ------
    ValueError
        If the array is not of shape (n_examined, 9), contains non-finite
        entries, the Ritz extrema are non-finite or ritz_min > ritz_max,
        or a critical point is not strictly inside its interval.
    """
    return None
```

### Step 7

evaluate_exact_interval_maxima

Goal
----
Compute the exact maximum of the balanced residual polynomial on every examined real-root interval and the amount by which the Hermite-spline estimate overshoots it.

```python
import numpy as np


def evaluate_exact_interval_maxima(
    harmonic_ritz_roots: np.ndarray,
) -> np.ndarray:
    """
    Compute exact residual-polynomial maxima on the examined intervals.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots with
        complex roots occurring in conjugate pairs.

    Returns
    -------
    np.ndarray
        Real array of shape (n_examined, 4) with columns
        [left_root, right_root, exact_maximum, spline_overshoot].

    Raises
    ------
    ValueError
        If the root array is empty or not one-dimensional, contains
        non-finite entries, or an upstream stage rejects the input.
    """
    return None
```

### Step 8

evaluate_final_diagnostic

Goal
----
Run the complete Balance Method 2 and Hermite-spline screening pipeline and return the maximum flagged exceedance margin as the final scalar diagnostic.

```python
import numpy as np


def evaluate_final_diagnostic(
    harmonic_ritz_roots: np.ndarray,
    ritz_min: float,
    ritz_max: float,
) -> float:
    """
    Evaluate the complete balanced-polynomial screening diagnostic.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots with
        complex roots occurring in conjugate pairs.
    ritz_min : float
        Ritz value with the smallest real part.
    ritz_max : float
        Ritz value with the largest real part.

    Returns
    -------
    float
        Maximum flagged spline exceedance margin, or 0.0 if none
        is present.

    Raises
    ------
    ValueError
        If the root array is empty or not one-dimensional, contains
        non-finite entries, or the Ritz extrema are non-finite or
        ritz_min > ritz_max.
    """
    return None
```
