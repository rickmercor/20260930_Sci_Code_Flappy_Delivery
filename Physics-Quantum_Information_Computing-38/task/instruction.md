# Physics-Quantum_Information_Computing-38

## Background

Continuous weak measurement of a quantum system yields a noisy record whose drift depends on the state conditioned on that record, and the conditional state itself evolves by a diffusive stochastic master equation driven by the same record. Identifying Hamiltonian and measurement parameters from such records is therefore statistical inference for a partially observed diffusion, and repeating the experiment from a known preparation gives a many-trajectory setting in which the consistency and the uncertainty of an estimator can be analysed.

The task below applies such an analysis to a driven, continuously monitored qubit.

## Problem

I calibrate a continuously monitored, resonantly driven qubit from repeated runs, and I want to test my records against the nominal calibration with an uncertainty I can defend. With ħ = 1 the qubit has Hamiltonian H = (α/2)σ_x and is monitored in the standard diffusive unravelling of the Lindblad equation with the single measurement operator L = βσ_z, each run's record obeying dY = Tr(Lρ + ρL†) dt + dW, where ρ is the state conditioned on that record and W is a standard Wiener process. Every run starts from ρ₀ = (I − 0.8σ_y + 0.6σ_z)/2, and θ = (α, β) is the same in every run.

These are my measurements: N = 30 independent runs, each giving the integrated record Y at t = 0.4, 0.8, …, 4.0 (Y = 0 at t = 0), listed in the table below with one row per run. My nominal calibration is θ_nom = (α, β) = (−1.5, 0.55), with the sign of α as in H.

Estimate θ over the box −5 ≤ α ≤ 5, 0.05 ≤ β ≤ 3 with the filter-free many-run estimator whose contrast replaces each run's conditional state by the deterministic ensemble-averaged state, and take the covariance V̂ of √N(θ̂ − θ) from that method's own empirical estimator. Report the squared joint Studentized distance D² = N(θ̂ − θ_nom)ᵀ V̂⁻¹ (θ̂ − θ_nom) to four significant figures, and tell me what my records imply for the nominal calibration.

In `<reasoning>`, state: α̂ and β̂; the three distinct entries of V̂; the componentwise Studentized errors √N(θ̂_k − θ_nom,k)/√V̂_kk; D² and whether my records are consistent with θ_nom at the 5% level; the value D² would take if V̂ were replaced by the inverse of the per-run Fisher information at θ̂ obtained by treating the discretized contrast as an exact Gaussian log-likelihood, and why the two values differ; what the estimate and D² become when the same runs are used only at t = 0.8, 1.6, 2.4, 3.2 and 4.0; and whether my 0.4 sampling interval is fine enough to trust that verdict, in light of the method's large-sample theory for sampled records and its published simulation results. Those are the derived quantities behind the final number and its interpretation, so stating them is what the output requirements below call for; what those requirements exclude is restating my table or pasting per-candidate and per-iteration output.

| run | Y(0.4) | Y(0.8) | Y(1.2) | Y(1.6) | Y(2.0) | Y(2.4) | Y(2.8) | Y(3.2) | Y(3.6) | Y(4.0) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.057 | 0.536 | 0.472 | 0.901 | 0.951 | 1.111 | 0.463 | 0.251 | -0.484 | -0.755 |
| 2 | 0.354 | 0.700 | 0.885 | 1.082 | 0.795 | -0.441 | -0.032 | -0.514 | -0.778 | -0.085 |
| 3 | 1.112 | 1.834 | 1.821 | 2.242 | 1.391 | 1.234 | 1.739 | 1.825 | 1.428 | 2.231 |
| 4 | 0.383 | 1.615 | 3.222 | 5.142 | 4.666 | 5.428 | 6.352 | 5.471 | 4.854 | 4.752 |
| 5 | 1.085 | 1.926 | 3.015 | 3.792 | 3.021 | 2.686 | 2.776 | 2.373 | 2.827 | 3.662 |
| 6 | 1.064 | 2.237 | 3.358 | 4.848 | 5.271 | 6.587 | 7.454 | 6.749 | 7.591 | 7.991 |
| 7 | -1.186 | -1.341 | -1.492 | 0.713 | 0.275 | 0.904 | 0.695 | -0.692 | -0.285 | -0.787 |
| 8 | 0.937 | 1.705 | 1.358 | 1.211 | 1.412 | 0.520 | -0.041 | 0.361 | -0.189 | -0.275 |
| 9 | -0.087 | -0.609 | -0.545 | -0.038 | -0.789 | -1.514 | -1.654 | -0.083 | 0.694 | 1.694 |
| 10 | 1.051 | 2.449 | 3.672 | 3.545 | 2.940 | 0.747 | -0.608 | -1.184 | -1.956 | -1.091 |
| 11 | 0.577 | 0.740 | -0.238 | 0.212 | 0.370 | 0.658 | 0.702 | 1.230 | 2.645 | 2.840 |
| 12 | 1.106 | 1.410 | 1.715 | 2.557 | 2.513 | 2.944 | 2.081 | 2.908 | 2.096 | 0.993 |
| 13 | -0.083 | 1.177 | 2.991 | 3.400 | 5.465 | 5.955 | 6.024 | 5.880 | 4.994 | 4.605 |
| 14 | 1.141 | 0.260 | 0.618 | 1.727 | 0.996 | 0.637 | -0.422 | -1.344 | -1.343 | -2.364 |
| 15 | 1.229 | 2.791 | 3.215 | 3.813 | 4.257 | 5.944 | 7.022 | 8.047 | 9.166 | 9.895 |
| 16 | 0.127 | 0.621 | 0.504 | 1.264 | 0.446 | -0.121 | 0.067 | -0.860 | -1.863 | -1.558 |
| 17 | -0.830 | -1.596 | -2.590 | -3.462 | -3.944 | -4.083 | -2.916 | -2.071 | -0.887 | -0.479 |
| 18 | 1.242 | 0.542 | 0.798 | 1.179 | 0.240 | -0.835 | -1.871 | -2.401 | -4.260 | -3.874 |
| 19 | 0.355 | 0.461 | 0.927 | 2.465 | 2.734 | 3.007 | 2.628 | 1.734 | 1.447 | 0.079 |
| 20 | 0.184 | -0.317 | 0.396 | 0.234 | 1.278 | -0.102 | -0.262 | -0.033 | -0.599 | 0.200 |
| 21 | 1.609 | 2.536 | 3.100 | 3.793 | 3.969 | 3.707 | 3.682 | 3.316 | 2.341 | 1.913 |
| 22 | 0.544 | 2.100 | 2.333 | 3.348 | 3.550 | 3.722 | 5.023 | 6.229 | 7.256 | 7.473 |
| 23 | 1.235 | 1.557 | 0.980 | 1.143 | -0.310 | 0.574 | -0.260 | 0.643 | 0.969 | 1.791 |
| 24 | 0.929 | 2.082 | 3.714 | 4.108 | 5.370 | 5.709 | 7.072 | 7.129 | 6.489 | 5.830 |
| 25 | 0.383 | 2.260 | 2.446 | 1.717 | 2.875 | 2.069 | 1.328 | 0.554 | 0.333 | 0.297 |
| 26 | -0.279 | -0.265 | 0.413 | 1.202 | 2.222 | 2.960 | 3.981 | 3.145 | 2.365 | 1.214 |
| 27 | 0.673 | 0.737 | 0.928 | -0.304 | -0.768 | -1.652 | -2.321 | -2.960 | -2.440 | -1.386 |
| 28 | 0.953 | 1.323 | 1.174 | 2.001 | 3.216 | 3.971 | 3.908 | 3.854 | 2.925 | 1.559 |
| 29 | -0.396 | -0.686 | -0.045 | 0.877 | 1.861 | 2.659 | 1.946 | 1.477 | 0.480 | 0.562 |
| 30 | -0.310 | 0.490 | 0.475 | 0.906 | 0.879 | 1.854 | 2.158 | 5.135 | 5.542 | 6.784 |

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_build_bloch_generator

Goal
----
Build the real affine generator that propagates the Bloch vector of a qubit whose ensemble-averaged state follows a Lindblad master equation.

```python
def build_bloch_generator(hamiltonian: 'np.ndarray', jump_operators: 'np.ndarray') -> 'np.ndarray':
    """Return the affine generator of the Bloch-vector dynamics of a Lindblad qubit.

    The qubit state ``rho`` obeys (hbar = 1)
    ``d rho / d t = -i [H, rho] + sum_k (L_k rho L_k^dag - {L_k^dag L_k, rho} / 2)``
    with ``H = hamiltonian`` and ``L_k = jump_operators[k]``. Write
    ``u = (1, x, y, z)`` with ``x = Tr(rho sigma_x)``, ``y = Tr(rho sigma_y)``,
    ``z = Tr(rho sigma_z)`` and the Pauli matrices
    ``sigma_x = [[0, 1], [1, 0]]``, ``sigma_y = [[0, -1j], [1j, 0]]``,
    ``sigma_z = [[1, 0], [0, -1]]``. Return the real matrix ``G`` for which
    ``d u / d t = G u`` holds for every state; its first row is zero.

    Parameters
    ----------
    hamiltonian : np.ndarray
        Complex ``(2, 2)`` Hermitian matrix, with each entry's modulus at
        most 10 in the task's units.
    jump_operators : np.ndarray
        Complex array of shape ``(m, 2, 2)`` holding ``0 <= m <= 16`` jump
        operators, with each entry's modulus at most 10. A sequence of
        ``(2, 2)`` matrices is also accepted.

    Returns
    -------
    np.ndarray
        Real array of shape ``(4, 4)``.

    Raises
    ------
    ValueError
        If ``hamiltonian`` is not a finite ``(2, 2)`` matrix, if it differs
        from its conjugate transpose by more than ``1e-12`` in any entry, or
        if ``jump_operators`` cannot be read as a finite array of shape
        ``(m, 2, 2)``, either operator array exceeds the supported entry
        bound of 10, there are more than 16 jump operators, or numerical
        evaluation produces a nonfinite intermediate or result.
    """
    return generator
```

### Step 2

02_propagate_averaged_drift

Goal
----
Propagate the ensemble-averaged Bloch vector of the monitored qubit and return the averaged homodyne drift together with its exact parameter derivatives on the left ends of a uniform sampling grid.

```python
def propagate_averaged_drift(
    alpha: float,
    beta: float,
    bloch0: 'np.ndarray',
    delta: float,
    n_intervals: int,
    drive_generator: 'np.ndarray',
    dephasing_generator: 'np.ndarray',
) -> 'np.ndarray':
    """Return the averaged drift and its derivatives at the sampling-interval starts.

    The qubit has Hamiltonian ``H = (alpha / 2) sigma_x`` and measurement
    operator ``L = beta sigma_z``. Its ensemble-averaged state has Bloch
    vector ``u(t) = (1, x, y, z)`` obeying
    ``d u / d t = (alpha * drive_generator + beta**2 * dephasing_generator) u``
    with ``u(0) = (1, bloch0)``, and the averaged drift of the homodyne
    record is ``h(t) = Tr(L rho(t) + rho(t) L^dag)``. Row ``j`` of the
    result, ``j = 0, ..., n_intervals - 1``, holds
    ``[h(t_j), dh/dalpha (t_j), dh/dbeta (t_j)]`` at ``t_j = j * delta``,
    where both derivatives are total derivatives at fixed ``bloch0``.
    On the supported domain below, the accuracy target is a maximum-entry
    error of ``1e-11 * max(1, max(abs(exact_drift)))`` for the whole returned
    array. The derivatives are evaluated from the differentiated dynamics;
    this is a floating-point accuracy target, not exact arithmetic.

    Both generators must have a zero first row and entries of absolute
    value at most 2. For ``G = alpha * drive_generator + beta**2 *
    dephasing_generator``, the largest eigenvalue of the symmetric part of
    ``G[1:, 1:]`` must be at most ``1e-12``. This contract supports
    contractive Bloch dynamics, including an affine forcing column, and
    excludes exponentially growing homogeneous dynamics. The last returned
    sample time ``(n_intervals - 1) * delta`` must not exceed 4.

    Parameters
    ----------
    alpha : float
        Finite real drive parameter, ``-5 <= alpha <= 5``.
    beta : float
        Finite real measurement parameter, ``0 <= beta <= 3``.
    bloch0 : np.ndarray
        Initial Bloch vector ``(x, y, z)`` with Euclidean norm at most
        ``1 + 1e-12``.
    delta : float
        Sampling interval in ``[1e-4, 1]``.
    n_intervals : int
        Integer number of sampling intervals in ``[1, 128]``.
    drive_generator : np.ndarray
        Finite real ``(4, 4)`` matrix satisfying the bounds above.
    dephasing_generator : np.ndarray
        Finite real ``(4, 4)`` matrix satisfying the bounds above.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_intervals, 3)``.

    Raises
    ------
    ValueError
        If ``alpha``, ``beta`` or ``delta`` is not a finite real number,
        a parameter, sample count or time exceeds its supported range,
        ``bloch0`` is not a finite real length-3 vector of norm at most
        ``1 + 1e-12``, either generator fails its real-array, shape, entry,
        zero-first-row or contractivity requirements, or a numerical
        intermediate/result is nonfinite. Non-real entries are rejected.
        Boolean scalar parameters and sample counts are rejected.
    """
    return drift
```

### Step 3

03_evaluate_contrast_model

Goal
----
Evaluate the discretely sampled averaged-state contrast of the ensemble-mean record, its parameter gradient and its plug-in information matrix at one parameter value.

```python
def evaluate_contrast_model(
    alpha: float,
    beta: float,
    mean_increments: 'np.ndarray',
    delta: float,
    drift_fn: "Callable[[float, float], np.ndarray]",
) -> 'np.ndarray':
    """Return the contrast, its gradient and the plug-in information at (alpha, beta).

    ``drift_fn(alpha, beta)`` follows the contract of
    ``propagate_averaged_drift`` with every other argument fixed: row ``j``
    holds ``[h_j, dh_j/dalpha, dh_j/dbeta]``, the averaged drift and its
    gradient at the start ``t_j = j * delta`` of the ``j``-th sampling
    interval. ``mean_increments[j]`` is the ensemble-mean record increment
    ``Ybar(t_{j+1}) - Ybar(t_j)`` over that interval. With ``g_j`` the
    gradient pair of row ``j``, return the length-6 array
    ``[Phi, dPhi/dalpha, dPhi/dbeta, I_aa, I_ab, I_bb]`` where
    ``Phi = sum_j h_j * mean_increments[j] - (delta / 2) * sum_j h_j**2``,
    ``(dPhi/dalpha, dPhi/dbeta) = sum_j g_j * (mean_increments[j] - h_j * delta)``
    and ``I = delta * sum_j g_j g_j^T``.

    Parameters
    ----------
    alpha, beta : float
        Parameter value passed to ``drift_fn``.
    mean_increments : np.ndarray
        Finite array of shape ``(n,)``.
    delta : float
        Positive sampling interval.
    drift_fn : callable
        Function of ``(alpha, beta)`` returning a finite ``(n, 3)`` array.

    Returns
    -------
    np.ndarray
        Float array of shape ``(6,)``.

    Raises
    ------
    ValueError
        If ``alpha``, ``beta`` or ``delta`` is not a finite real number,
        ``delta <= 0``, ``mean_increments`` is not a non-empty finite 1-D
        array, ``drift_fn`` is not callable, or ``drift_fn`` does not return
        a finite array of shape ``(n, 3)`` with ``n = len(mean_increments)``.
    """
    return model
```

### Step 4

04_locate_contrast_candidates

Goal
----
Scan the contrast on a uniform parameter grid and select the highest, well-separated grid local maxima as starting points for continuous refinement.

```python
def locate_contrast_candidates(
    contrast_fn: "Callable[[float, float], float]",
    alpha_bounds: tuple,
    beta_bounds: tuple,
    spacing: float = 0.05,
    max_candidates: int = 8,
    min_separation: float = 0.15,
) -> 'np.ndarray':
    """Return the selected grid local maxima of the contrast.

    The grid is ``alpha_k = alpha_bounds[0] + k * spacing`` for
    ``k = 0, ..., K`` with ``K = round((alpha_bounds[1] - alpha_bounds[0]) / spacing)``,
    and likewise for ``beta``; each range must be a whole number of
    spacings to within ``1e-9 * max(1, range)``. The last point on each
    axis is set to its exact upper bound after this check. A grid point is a candidate
    when ``contrast_fn`` there is no smaller than at each of its (up to
    eight) neighbouring grid points. Candidates are ranked by decreasing
    contrast, ties keeping row-major grid order (``alpha`` index major).
    Walking down that ranking, a candidate is accepted when its Euclidean
    distance to every previously accepted point is at least
    ``min_separation - 1e-9``; the walk stops once ``max_candidates`` points
    are accepted.

    The supported numerical grid has bounds within ``[-100, 100]``,
    ``1e-4 <= spacing <= 200``, at most 2000 intervals on either axis,
    and at most 250000 grid points in total (endpoints included).

    Parameters
    ----------
    contrast_fn : callable
        Function of ``(alpha, beta)`` returning a finite real number.
    alpha_bounds, beta_bounds : tuple
        ``(low, high)`` with ``-100 <= low < high <= 100``.
    spacing : float
        Shared grid spacing in ``[1e-4, 200]``.
    max_candidates : int
        Maximum number of accepted points, ``>= 1``.
    min_separation : float
        Finite minimum distance between accepted points, ``>= 0``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(m, 2)`` holding the accepted ``(alpha, beta)``
        points in acceptance order, ``1 <= m <= max_candidates``.

    Raises
    ------
    ValueError
        If ``contrast_fn`` is not callable or returns a non-finite or
        non-scalar value, a bound pair is not two finite numbers with
        ``low < high``, ``spacing`` is not a finite positive number that
        divides both ranges, ``max_candidates`` is not an integer ``>= 1``
        (booleans are rejected), or ``min_separation`` is negative or not
        finite. Unsupported numerical bounds, spacing, axis sizes or total
        grid size, and nonfinite intermediate/result arrays also raise
        ``ValueError``.
    """
    return candidates
```

### Step 5

05_maximize_contrast

Goal
----
Refine each grid candidate by bounded Gauss-Newton ascent of the contrast and return the refined local maximizer with the largest contrast as the maximum-contrast estimate.

```python
def maximize_contrast(
    model_fn: "Callable[[float, float], np.ndarray]",
    starts: 'np.ndarray',
    alpha_bounds: tuple,
    beta_bounds: tuple,
    tolerance: float = 1e-10,
) -> 'np.ndarray':
    """Return the maximum-contrast point reached from the starting points.

    ``model_fn(alpha, beta)`` follows the contract of
    ``evaluate_contrast_model``: it returns
    ``[Phi, dPhi/dalpha, dPhi/dbeta, I_aa, I_ab, I_bb]`` with a symmetric
    positive-definite information matrix ``I``. From each start, clipped
    into the box, run bounded Gauss-Newton ascent: at the current point a
    coordinate is held fixed when it lies on a bound and its gradient
    component points out of the box, the free coordinates move along the
    direction ``d`` that solves ``I_ff d_f = g_f`` on the free block, the
    trial point ``theta + s d`` is clipped into the box, and ``s`` is halved
    from 1 until the contrast at the trial point is not below the current
    contrast. The ascent comes to rest at a local maximizer of the contrast
    in the box (every free coordinate has zero gradient there); locate it to
    within ``tolerance`` in each coordinate. Return the located maximizer
    with the largest contrast, the earliest start winning ties.

    Parameters
    ----------
    model_fn : callable
        Function of ``(alpha, beta)`` returning a finite length-6 array.
    starts : np.ndarray
        Finite array of shape ``(m, 2)`` with ``m >= 1`` starting points.
    alpha_bounds, beta_bounds : tuple
        ``(low, high)`` with finite ``low < high``.
    tolerance : float
        Positive accuracy of each located maximizer.

    Returns
    -------
    np.ndarray
        Float array ``[alpha_hat, beta_hat]`` of shape ``(2,)``.

    Raises
    ------
    ValueError
        If ``model_fn`` is not callable or returns anything other than a
        finite length-6 array, the information matrix on the free block is
        not positive definite, ``starts`` is not a finite ``(m, 2)`` array
        with ``m >= 1``, a bound pair is not two finite numbers with
        ``low < high``, or ``tolerance`` is not a finite positive number.
    """
    return theta_hat
```

### Step 6

06_estimate_sandwich_covariance

Goal
----
Estimate the covariance of the scaled estimation error from the per-record score contributions and the plug-in information at a parameter value.

```python
def estimate_sandwich_covariance(
    alpha: float,
    beta: float,
    increments: 'np.ndarray',
    delta: float,
    drift_fn: "Callable[[float, float], np.ndarray]",
) -> 'np.ndarray':
    """Return the empirical sandwich covariance of sqrt(N) (theta_hat - theta).

    ``increments[i, j]`` is record ``i``'s increment
    ``Y_i(t_{j+1}) - Y_i(t_j)`` over the ``j``-th sampling interval, and row
    ``j`` of ``drift_fn(alpha, beta)`` (contract of
    ``propagate_averaged_drift``) holds ``[h_j, g_j]`` at ``t_j = j * delta``,
    with ``g_j`` the gradient pair. Each record's score contribution is
    ``eta_i = sum_j g_j * (increments[i, j] - h_j * delta)``. Return
    ``V = I^{-1} S I^{-1}`` with ``I = delta * sum_j g_j g_j^T`` and
    ``S = (1 / N) sum_i (eta_i - eta_bar)(eta_i - eta_bar)^T``, where
    ``eta_bar`` is the mean of the ``eta_i`` over the ``N`` records.

    Parameters
    ----------
    alpha, beta : float
        Parameter value passed to ``drift_fn``.
    increments : np.ndarray
        Finite array of shape ``(N, n)`` with ``N >= 2``.
    delta : float
        Positive sampling interval.
    drift_fn : callable
        Function of ``(alpha, beta)`` returning a finite ``(n, 3)`` array.

    Returns
    -------
    np.ndarray
        Symmetric float array of shape ``(2, 2)``.

    Raises
    ------
    ValueError
        If ``alpha``, ``beta`` or ``delta`` is not a finite real number,
        ``delta <= 0``, ``increments`` is not a finite 2-D array with at
        least two rows and one column, ``drift_fn`` is not callable or does
        not return a finite ``(n, 3)`` array, or ``I`` is not positive
        definite (smallest eigenvalue at most ``1e-12`` times the largest).
    """
    return covariance
```

### Step 7

07_compute_studentized_statistics

Goal
----
Studentize the estimation error against a reference parameter value with a covariance estimate, componentwise and jointly, and form the squared joint distance.

```python
def compute_studentized_statistics(
    theta_hat: 'np.ndarray',
    theta_ref: 'np.ndarray',
    covariance: 'np.ndarray',
    n_records: int,
) -> 'np.ndarray':
    """Return the componentwise and joint Studentized errors and the squared distance.

    With ``d = sqrt(n_records) * (theta_hat - theta_ref)`` and ``V =
    covariance`` (the covariance of ``sqrt(N)(theta_hat - theta)``), return
    ``[T_1, T_2, J_1, J_2, D2]`` where ``T_k = d_k / sqrt(V_kk)``,
    ``(J_1, J_2) = V^{-1/2} d`` with ``V^{-1/2}`` the symmetric
    positive-definite inverse square root of ``V``, and ``D2 = J_1**2 + J_2**2``.

    Parameters
    ----------
    theta_hat, theta_ref : np.ndarray
        Finite length-2 parameter vectors.
    covariance : np.ndarray
        Finite ``(2, 2)`` symmetric positive-definite matrix.
    n_records : int
        Number of records ``N >= 1``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(5,)``.

    Raises
    ------
    ValueError
        If either parameter vector is not a finite length-2 vector,
        ``covariance`` is not a finite ``(2, 2)`` matrix whose entries match
        its transpose to ``1e-12`` times its largest absolute entry, or it is
        not positive definite, or ``n_records`` is not an integer ``>= 1``
        (booleans are rejected).
    """
    return statistics
```

### Step 8

08_assess_nominal_calibration

Goal
----
Compose every earlier step to estimate the drive and measurement parameters of a monitored qubit from repeated homodyne records and return the squared joint Studentized distance of the estimate from a nominal calibration.

```python
def assess_nominal_calibration(
    records: 'np.ndarray',
    delta: float = 0.4,
    bloch0: tuple = (0.0, -0.8, 0.6),
    theta_nominal: tuple = (-1.5, 0.55),
    alpha_bounds: tuple = (-5.0, 5.0),
    beta_bounds: tuple = (0.05, 3.0),
    grid_spacing: float = 0.05,
    tolerance: float = 1e-10,
) -> float:
    """Return the squared joint Studentized distance of the estimate from the nominal.

    ``records[i, j]`` is run ``i``'s integrated homodyne record ``Y_i`` at
    ``t = (j + 1) * delta`` (``Y_i(0) = 0``) for a qubit with Hamiltonian
    ``H = (alpha / 2) sigma_x``, measurement operator ``L = beta sigma_z``
    and initial Bloch vector ``bloch0``. Build the averaged-state drift of
    this model from the Lindblad generators of ``sigma_x / 2`` (no jump
    operators) and of the single jump operator ``sigma_z`` (zero
    Hamiltonian); maximize the contrast of the ensemble-mean record over the
    box ``alpha_bounds x beta_bounds``, starting from the grid candidates
    (spacing ``grid_spacing``, at most 8, separation 0.15) and refining them
    to ``tolerance``; estimate the sandwich covariance from the records at
    the maximizer; and return the squared joint Studentized distance of
    the maximizer from ``theta_nominal``. The defaults reproduce the problem
    statement apart from the records.

    Parameters
    ----------
    records : np.ndarray
        Finite array of shape ``(N, n)`` with ``N >= 2``, ``1 <= n <= 128``,
        and ``(n - 1) * delta <= 4``, as required by the drift step.
    delta : float
        Sampling interval in ``[1e-4, 1]``.
    bloch0 : tuple
        Initial Bloch vector ``(x, y, z)`` inside the Bloch ball.
    theta_nominal : tuple
        Nominal ``(alpha, beta)``.
    alpha_bounds, beta_bounds : tuple
        ``(low, high)`` search box contained in ``[-5, 5] x [0, 3]``.
    grid_spacing : float
        Candidate-grid spacing dividing both ranges, subject to the
        supported spacing and grid-size limits of the candidate step.
    tolerance : float
        Positive accuracy of the refined maximizers.

    Returns
    -------
    float
        The squared joint Studentized distance ``D2``.

    Raises
    ------
    ValueError
        If ``records`` is not a finite ``(N, n)`` array with ``N >= 2`` and
        ``1 <= n <= 128``, the search box exceeds the supported parameter
        domain, or any stage rejects its input (including the drift step's
        sampling-time and real Bloch-vector requirements).
    """
    return 0.0
```
