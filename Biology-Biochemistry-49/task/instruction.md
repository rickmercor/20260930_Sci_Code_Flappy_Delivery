# Biology-Biochemistry-49

## Background

Pathway-biased molecular simulations can concentrate sampling along distinct transition channels, but their resulting observations cannot be pooled as if they came from one equilibrium trajectory. Path collective variables provide smooth coordinates along and away from a discretized pathway, while bin-less reweighting assigns configuration-specific statistical weights without constructing histograms.

The reported equilibrium kinetic observable can depend on the geometry chosen to represent each pathway. Quantifying that dependence is useful for assessing whether modest uncertainty in interior pathway images could materially change a reweighted mechanistic conclusion. An analytic local sensitivity separates this geometric effect from sampling noise and from changes in the endpoint states.

## Problem

Independent pathway-biased simulations sample different transition channels and require configuration-specific reweighting before their committor statistics can be combined. For pathway $i$, use the supplied profiles $V_i(s)=a_i s+b_i s(1-s)$ and $u_i^{\perp}(\zeta)=c_i\zeta^2$, with the converged bias convention fixed explicitly as $w_i(\mathbf z)=-V_i[s_i(\mathbf z)]+u_i^{\perp}[\zeta_i(\mathbf z)]$. Recover from the cited article and Supporting Information the string path-collective coordinates, bin-less WHAM factors, self-consistent relative offsets, and normalized conditional committor correlation.

Treat every interior string-image coordinate as an independent differentiable variable while holding both endpoint images, the pooled configurations, committor traces, coefficients, counts, $\alpha$, and $\beta$ fixed. After solving the pathway-0-gauge WHAM equations, analytically evaluate the complete geometry gradient $G_{imr}=\partial C[q;\tau]/\partial Z_{imr}$ at `target_lag_step`, for pathways $i$, interior images $m=1,\ldots,M-2$, and Cartesian coordinates $r$. The requested scalar is the endpoint-constrained sensitivity
$$
S_{\mathrm{geom}}=\left(\sum_{i=0}^{W-1}\sum_{m=1}^{M-2}\sum_{r=0}^{D-1}G_{imr}^{2}\right)^{1/2},
$$
which is also the maximum first-order increase over unit-Frobenius interior-image perturbations. Use the Supporting Information committor estimator as printed, without the main-text factor $1/2$, averaging each trace over its own valid time origins before applying normalized WHAM weights. Do not estimate the requested gradient by finite differences.

Use IEEE-754 float64 arithmetic, Euclidean squared distance, stable log-sum-exp evaluations, zero initialization of all offsets, simultaneous offset updates, pathway 0 as the reporting reference, and the inclusive convergence condition that the maximum absolute offset change after an update is at most `tolerance`. Use zero-based pathway, image, and Cartesian-coordinate indices when identifying a gradient component.

Use the following numerical configuration:

```python
strings = np.array([
    [[-1.00, 0.00], [-0.55, 0.32], [0.00, 0.50], [0.55, 0.32], [1.00, 0.00]],
    [[-1.00, 0.00], [-0.55,-0.30], [0.00,-0.46], [0.55,-0.30], [1.00, 0.00]],
    [[-1.00, 0.00], [-0.50, 0.08], [0.00,-0.08], [0.48, 0.14], [1.00, 0.00]],
], dtype=np.float64)

points = np.array([
    [-0.92, 0.05], [-0.58, 0.29], [-0.10, 0.47], [0.44, 0.37], [0.91, 0.04],
    [-0.88,-0.04], [-0.60,-0.27], [-0.08,-0.43], [0.47,-0.33], [0.93,-0.03],
    [-0.90, 0.01], [-0.46, 0.12], [0.04,-0.04], [0.51, 0.10], [0.89, 0.02],
], dtype=np.float64)

committor_traces = np.array([
    [0.04,0.07,0.11,0.16,0.22,0.29,0.37],
    [0.12,0.16,0.21,0.27,0.34,0.42,0.51],
    [0.34,0.39,0.45,0.52,0.60,0.69,0.77],
    [0.55,0.62,0.69,0.76,0.82,0.87,0.91],
    [0.77,0.83,0.88,0.92,0.95,0.97,0.98],
    [0.05,0.08,0.12,0.17,0.23,0.30,0.38],
    [0.15,0.19,0.24,0.30,0.37,0.45,0.54],
    [0.36,0.40,0.45,0.51,0.58,0.66,0.75],
    [0.57,0.63,0.70,0.77,0.83,0.88,0.92],
    [0.79,0.84,0.89,0.93,0.96,0.98,0.99],
    [0.03,0.06,0.10,0.15,0.21,0.28,0.36],
    [0.14,0.18,0.23,0.29,0.36,0.44,0.53],
    [0.35,0.41,0.48,0.56,0.65,0.74,0.82],
    [0.58,0.64,0.71,0.78,0.84,0.89,0.93],
    [0.80,0.85,0.90,0.94,0.97,0.985,0.995],
], dtype=np.float64)

alpha = 18.0
linear_coefficients = np.array([0.70, -0.35, 0.25], dtype=np.float64)
curvature_coefficients = np.array([1.10, 0.85, 1.30], dtype=np.float64)
orthogonal_scales = np.array([2.2, 1.8, 2.5], dtype=np.float64)
sample_counts = np.array([5.0, 5.0, 5.0], dtype=np.float64)
beta = 1.4
lag_steps = np.array([1, 2, 3], dtype=int)
tolerance = 1.0e-12
max_iterations = 10000
target_lag_step = 3
```

In `<reasoning>`, identify the relevant Supporting Information equations, state the derived coordinate- and WHAM-response relations compactly, and report the relative offsets, three global correlations, three pathway-specific interior-gradient norms, and the zero-based index and signed value of the largest-magnitude lag-3 gradient component. Report all requested numerical checkpoints to at least 12 significant figures.

Output Format Requirements:
Emit `<final_answer>` immediately, then `<reasoning>`. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside `<final_answer>...</final_answer>`, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep `<reasoning>` short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, full Jacobian arrays, or per-window tables.

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

step_01_pcv_progress_geometry.py

Goal
----
Compute the paper-specific soft path progress values and their analytic derivatives with respect to all string-image coordinates.

```python
def pcv_progress_geometry(
    points: "np.ndarray",
    strings: "np.ndarray",
    alpha: float,
) -> tuple["np.ndarray", "np.ndarray"]:
    r"""Return soft progress coordinates and string-geometry derivatives.

    Parameters
    ----------
    points : np.ndarray
        Finite pooled configurations with shape ``(n, d)``.
    strings : np.ndarray
        Finite pathway images with shape ``(w, m, d)``, where ``m >= 2``.
    alpha : float
        Finite strictly positive kernel sharpness.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float64 progress values with shape ``(w, n)`` and local image-coordinate
        derivatives with shape ``(w, n, m, d)``. The derivative entry
        ``[i,t,m,r]`` is with respect to ``strings[i,m,r]``.

    Raises
    ------
    ValueError
        If arrays have incompatible shapes, contain nonfinite values, there are
        fewer than two images, or alpha is not finite and strictly positive.

    Notes
    -----
    Use fractional image positions from zero to one and a stable normalized
    exponential-distance kernel over all images.
    """
    return None
```

### Step 2

step_02_pcv_orthogonal_geometry.py

Goal
----
Compute the paper-specific orthogonal string variables and their analytic derivatives with respect to all string-image coordinates.

```python
def pcv_orthogonal_geometry(
    points: "np.ndarray",
    strings: "np.ndarray",
    alpha: float,
) -> tuple["np.ndarray", "np.ndarray"]:
    r"""Return orthogonal string variables and geometry derivatives.

    Parameters
    ----------
    points : np.ndarray
        Finite pooled configurations with shape ``(n, d)``.
    strings : np.ndarray
        Finite pathway images with shape ``(w, m, d)``, where ``m >= 2``.
    alpha : float
        Finite strictly positive kernel sharpness.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float64 orthogonal values with shape ``(w, n)`` and local
        image-coordinate derivatives with shape ``(w, n, m, d)``.

    Raises
    ------
    ValueError
        If arrays have incompatible shapes, contain nonfinite values, there are
        fewer than two images, or alpha is not finite and strictly positive.

    Notes
    -----
    Evaluate the logarithmic kernel sum with a stable log-sum-exp calculation.
    """
    return None
```

### Step 3

step_03_pathway_bias_geometry.py

Goal
----
Combine the progress and orthogonal coordinates with the fixed bias convention to produce pathway biases and their geometry Jacobians

```python
def pathway_bias_geometry(
    progress_values: "np.ndarray",
    progress_jacobian: "np.ndarray",
    orthogonal_values: "np.ndarray",
    orthogonal_jacobian: "np.ndarray",
    linear_coefficients: "np.ndarray",
    curvature_coefficients: "np.ndarray",
    orthogonal_scales: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray"]:
    r"""Return converged pathway biases and local geometry derivatives.

    Parameters
    ----------
    progress_values : np.ndarray
        Finite path progress values with shape ``(w, n)``.
    progress_jacobian : np.ndarray
        Finite local derivatives with shape ``(w, n, m, d)``.
    orthogonal_values : np.ndarray
        Finite orthogonal variables with shape ``(w, n)``.
    orthogonal_jacobian : np.ndarray
        Finite local derivatives with shape ``(w, n, m, d)``.
    linear_coefficients : np.ndarray
        Finite profile coefficients with shape ``(w,)``.
    curvature_coefficients : np.ndarray
        Finite curvature coefficients with shape ``(w,)``.
    orthogonal_scales : np.ndarray
        Finite nonnegative restraint scales with shape ``(w,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float64 bias values with shape ``(w, n)`` and local geometry
        derivatives with shape ``(w, n, m, d)``.

    Raises
    ------
    ValueError
        If shapes are incompatible, values are nonfinite, progress values lie
        outside ``[0,1]``, or an orthogonal scale is negative.

    Notes
    -----
    Use the supplied converged-bias convention with the profile subtracted and
    the quadratic orthogonal restraint added.
    """
    return None
```

### Step 4

step_04_binless_reweighting_factors.py

Goal
----
Evaluate the configuration-dependent reciprocal WHAM denominator for every pooled sample using stable log-sum-exp arithmetic.

```python
def binless_reweighting_factors(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
) -> "np.ndarray":
    r"""Evaluate one pointwise bin-less WHAM factor per pooled configuration.

    Parameters
    ----------
    bias_values : np.ndarray
        Finite bias matrix with shape ``(w, n)``.
    sample_counts : np.ndarray
        Finite positive pathway counts with shape ``(w,)``.
    beta : float
        Finite positive inverse temperature.
    free_energy_offsets : np.ndarray
        Finite relative offsets with shape ``(w,)``.

    Returns
    -------
    np.ndarray
        Finite positive float64 factors with shape ``(n,)``.

    Raises
    ------
    ValueError
        If shapes or values violate the contract, counts or beta are not
        positive, or the resulting factors are not finite and positive.

    Notes
    -----
    Evaluate the reciprocal mixture denominator in log space.
    """
    return None
```

### Step 5

step_05_wham_offset_update.py

Goal
----
Perform one simultaneous gauge-fixed relative free-energy update from the current WHAM factors.

```python
def wham_offset_update(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
    reference_index: int,
) -> "np.ndarray":
    r"""Perform one simultaneous relative free-energy-offset update.

    Parameters
    ----------
    bias_values : np.ndarray
        Finite bias matrix with shape ``(w, n)``.
    sample_counts : np.ndarray
        Finite positive pathway counts with shape ``(w,)``.
    beta : float
        Finite positive inverse temperature.
    free_energy_offsets : np.ndarray
        Finite current offsets with shape ``(w,)``.
    reference_index : int
        Integer pathway index whose returned offset is fixed to zero.

    Returns
    -------
    np.ndarray
        Float64 updated relative offsets with shape ``(w,)``.

    Raises
    ------
    ValueError
        If an input violates the reweighting-factor contract or the reference
        index is not an in-range integer.

    Notes
    -----
    Evaluate every new offset from the same old offset vector.
    """
    return None
```

### Step 6

step_06_solve_wham_offsets.py

Goal
----
Iterate simultaneous relative-offset updates from zero to the documented inclusive convergence threshold.

```python
def solve_wham_offsets(
    bias_values: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    tolerance: float,
    max_iterations: int,
    reference_index: int,
) -> tuple["np.ndarray", int]:
    r"""Iterate simultaneous updates to the gauge-fixed WHAM solution.

    Parameters
    ----------
    bias_values : np.ndarray
        Finite bias matrix with shape ``(w, n)``.
    sample_counts : np.ndarray
        Finite positive pathway counts with shape ``(w,)``.
    beta : float
        Finite positive inverse temperature.
    tolerance : float
        Finite strictly positive inclusive convergence tolerance.
    max_iterations : int
        Positive integer update cap.
    reference_index : int
        In-range integer pathway index fixed to zero.

    Returns
    -------
    tuple[np.ndarray, int]
        Converged float64 relative offsets with shape ``(w,)`` and the number
        of simultaneous updates performed.

    Raises
    ------
    ValueError
        If shapes or values violate an upstream contract, tolerance is not
        finite and strictly positive, max_iterations is not a positive
        integer, or reference_index is invalid.
    RuntimeError
        If convergence is not reached within max_iterations.

    Notes
    -----
    Initialize every offset to zero and test the maximum absolute change after
    each update with an inclusive comparison.
    """
    return None
```

### Step 7

step_07_wham_geometry_gradient.py

Goal
----
Implicitly differentiate the converged WHAM equations and normalized committor correlations with respect to every string-image coordinate.

```python
def wham_geometry_gradient(
    bias_values: "np.ndarray",
    bias_jacobian: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    free_energy_offsets: "np.ndarray",
    committor_traces: "np.ndarray",
    lag_steps: "np.ndarray",
    reference_index: int,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    r"""Return offset and correlation gradients for all string coordinates.

    Parameters
    ----------
    bias_values : np.ndarray
        Finite bias matrix with shape ``(w, n)``.
    bias_jacobian : np.ndarray
        Finite local derivatives with shape ``(w, n, m, d)``; pathway ``i``
        entries differentiate ``bias_values[i]`` with respect to its own string.
    sample_counts : np.ndarray
        Finite positive pathway counts with shape ``(w,)``.
    beta : float
        Finite strictly positive inverse temperature.
    free_energy_offsets : np.ndarray
        Finite converged offsets with shape ``(w,)`` and a zero reference.
    committor_traces : np.ndarray
        Finite committor values in ``[0,1]`` with shape ``(n,t)``, ``t >= 2``.
    lag_steps : np.ndarray
        Nonempty one-dimensional array with integer dtype and entries satisfying
        ``1 <= lag < t``. A floating-point array is invalid even when every
        value is mathematically integral.
    reference_index : int
        In-range integer pathway index defining the fixed gauge.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Float64 offset gradients with shape ``(w,w,m,d)``, correlations with
        shape ``(l,)``, and correlation gradients with shape ``(l,w,m,d)``.

    Raises
    ------
    ValueError
        If shapes or values violate the contract, counts or beta are not
        positive, the reference is invalid or nonzero, lag_steps lacks integer
        dtype or contains an invalid lag, the gauge-fixed system is singular,
        or an output is nonfinite.

    Notes
    -----
    Treat beta, committor traces, and sample counts as geometry-independent and
    differentiate the converged equations in the fixed reference gauge.
    """
    return None
```

### Step 8

step_08_binless_wham_geometry_sensitivity.py

Goal
----
Run the complete pipeline and return the Frobenius norm of the selected-lag correlation gradient over all interior string images.

```python
def binless_wham_geometry_sensitivity(
    points: "np.ndarray",
    strings: "np.ndarray",
    committor_traces: "np.ndarray",
    alpha: float,
    linear_coefficients: "np.ndarray",
    curvature_coefficients: "np.ndarray",
    orthogonal_scales: "np.ndarray",
    sample_counts: "np.ndarray",
    beta: float,
    lag_steps: "np.ndarray",
    tolerance: float,
    max_iterations: int,
    target_lag_step: int,
) -> float:
    r"""Return the full interior-string geometry sensitivity at one lag.

    Parameters
    ----------
    points : np.ndarray
        Finite pooled configurations with shape ``(n,d)``.
    strings : np.ndarray
        Finite pathway images with shape ``(w,m,d)``, where ``m >= 3``.
    committor_traces : np.ndarray
        Finite committor traces in ``[0,1]`` with shape ``(n,t)``.
    alpha : float
        Finite strictly positive string-kernel sharpness.
    linear_coefficients : np.ndarray
        Finite profile coefficients with shape ``(w,)``.
    curvature_coefficients : np.ndarray
        Finite curvature coefficients with shape ``(w,)``.
    orthogonal_scales : np.ndarray
        Finite nonnegative restraint scales with shape ``(w,)``.
    sample_counts : np.ndarray
        Finite positive pathway counts with shape ``(w,)``.
    beta : float
        Finite strictly positive inverse temperature.
    lag_steps : np.ndarray
        Nonempty valid one-dimensional integer-dtype lag array.
    tolerance : float
        Finite strictly positive inclusive convergence tolerance.
    max_iterations : int
        Positive integer update cap.
    target_lag_step : int
        Integer lag value occurring exactly once in lag_steps.

    Returns
    -------
    float
        Finite nonnegative native Python float equal to the Frobenius norm of
        the selected correlation gradient over interior string images.

    Raises
    ------
    ValueError
        If an upstream contract fails, fewer than three string images are
        supplied, the target lag is not an integer occurring exactly once, or
        the selected norm is nonfinite.
    RuntimeError
        If the WHAM offsets do not converge within max_iterations.

    Notes
    -----
    Use pathway 0 as the reporting reference and hold the first and last image
    of every pathway fixed when forming the final norm.
    """
    return None
```
