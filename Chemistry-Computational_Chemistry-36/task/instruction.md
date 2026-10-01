# Chemistry-Computational_Chemistry-36

## Background

Enhanced-sampling simulations accelerate rare molecular events by adding a collective-variable bias. Recovering an unbiased kinetic rate is difficult when that coordinate is imperfect and when trajectories finish at different times or are right-censored. A multi-condition rate analysis can infer a shared efficiency parameter by demanding agreement among independently biased simulation sets, while a conventional flooding estimate rescales individual durations using their accumulated acceleration. The comparison in this task tests how those two estimators respond to the same finite, censored archive.

The numerical challenge is not a generic regression. It combines ragged ensemble averages, exponential quantities that require stable log-domain evaluation, a bounded variance minimization, an ordered convergence analysis, a censored empirical CDF fit, and a leave-one-set-out scientific validity check. The signed final comparison is intentionally not restricted to the interval from zero to one hundred.

## Problem

A deterministic archive of right-censored biased molecular-dynamics trajectories is available. Execute the NumPy block once and quantify the signed percentage by which the source workflow's multi-set unbiased rate exceeds its conventional flooding comparator on the largest scientifically consistent prefix.

```python
import numpy as np

rng = np.random.default_rng(26091615)
barriers = np.array([3.0, 4.5, 6.0, 7.5, 9.0, 11.0, 13.5])
lengths = np.array([
    [18,18,17,18,16,18,18,15,18],
    [18,15,14,18,13,16,18,12,17],
    [16,14,15,13,14,12,16,13,13],
    [14,13,12,14,11,13,14,12,12],
    [13,12,11,13,10,12,13,10,11],
    [8,7,10,6,7,5,9,6,5],
    [7,6,8,5,6,4,7,5,4],
], dtype=int)
events = np.array([
    [0,0,1,0,1,0,0,1,0],
    [1,0,1,0,1,1,0,0,0],
    [1,1,0,1,1,0,0,1,0],
    [1,1,1,0,1,1,0,1,0],
    [1,1,1,1,1,1,0,1,0],
    [1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1],
], dtype=int)
raw_bias = np.full((7, 9, 18), np.nan)
for j in range(7):
    for i in range(9):
        m = int(lengths[j, i])
        u = np.linspace(0.0, 1.0, m)
        restored = (
            0.18 + 0.76 * barriers[j]
            + 0.18 * np.sin(2.0 * np.pi * (u + 0.07 * i))
            + 0.06 * i
            + rng.normal(scale=0.055, size=m)
        )
        raw_bias[j, i, :m] = restored - barriers[j]

dt = 0.08
temperature = 300.0
variance_limit = 0.003
gamma_margin = 0.02
influence_limit = 0.2
```

Each stored bias value covers one complete interval of width `dt`; `lengths` counts those intervals. An event value of zero means right-censoring at the recorded duration. The finite prefix of `raw_bias[j,i]` is the reported OPES bias and the remaining entries are padding. Use `R=0.008314462618 kJ mol^-1 K^-1` and restore the physical bias in set `j` as `raw_bias[j] + barriers[j]`.

Reconstruct and apply the primary article's EATR-flooding multi-set estimator, including its set-wise corrected rates, common efficiency fit and ensemble-before-time acceleration convention. For this benchmark, define each observed rate as the event count divided by the total recorded exposure, retaining the complete duration of every right-censored trajectory. Order the sets by increasing full-efficiency log acceleration and fit every requested prefix independently.

Reconstruct and apply the source workflow's conventional OPES-f comparator. Apply its time transformation separately to each selected trajectory before pooling. Construct the censored empirical event CDF using ordinates `m/N`, where right-censored trajectories remain in the all-trajectory denominator `N`, and obtain the positive rate with the source's unweighted nonlinear fit. A pooled-exposure maximum-likelihood rate is not this comparator.

For this benchmark, scan prefixes containing 3 through 7 sets. Select the largest prefix whose population variance does not exceed `variance_limit` and whose fitted efficiency lies in the closed interval `[gamma_margin, 1-gamma_margin]`. Reject a selected result if the maximum absolute change in its refitted log-rate after deleting any one selected set exceeds `influence_limit`. Return

`100 * (k_multiset / k_conventional - 1)`

without clipping or taking an absolute value.

In `<reasoning>`, cite the sources and briefly state the attached article's five-barrier Protein G result and its weak-cavity unbinding comparison. From the earlier EATR study, identify the three biased coordinates and replication count in its all-atom chignolin benchmark, then state its coordinate-quality and rate-estimation conclusions. From the original iMetaD article, state its prior-knowledge condition and what its three demonstrations recovered relative to long unbiased dynamics. Compactly report beta, the seven observed log-rates, the seven full-efficiency log accelerations, one-based set order, the five prefix rows, selected prefix, fitted efficiency and log-rate, conventional log-rate, five leave-one-set-out log-rates, maximum influence, and final signed percentage. Explain why the percentage may exceed 100. Report floating values to at least ten decimal places; do not paste the input tensors.

For numerical reproducibility, perform every bounded scalar minimization with SciPy's `minimize_scalar` using `method="bounded"` and `options={"xatol": 1e-12, "maxiter": 10000}`. Perform the conventional nonlinear least-squares fit with SciPy's `curve_fit` using `maxfev=100000` and `ftol=xtol=gtol=1e-12`.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the requested audit scalars.

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

restore_opes_bias

Goal
----
Validate a ragged OPES archive and restore the reported bias offset.

```python
def restore_opes_bias(
    raw_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    barriers: 'np.ndarray',
    dt: float,
    temperature: float,
) -> 'np.ndarray':
    """Return a finite bias tensor with the OPES BARRIER offset restored.

    Parameters
    ----------
    raw_bias : np.ndarray
        Real numeric array with shape (n_set, n_traj, n_frame), where
        n_set >= 3, n_traj >= 1, and n_frame >= 2. Samples at frame indices
        smaller than the corresponding value in ``lengths`` are active and
        must be finite. Padding samples at frame indices greater than or equal
        to ``lengths`` may be nonfinite.
    lengths : np.ndarray
        Integer array with shape (n_set, n_traj). Every entry must satisfy
        2 <= lengths[i, j] <= n_frame.
    events : np.ndarray
        Real numeric binary array with shape (n_set, n_traj). Every entry
        must be finite and equal to either 0 or 1.
    barriers : np.ndarray
        Real numeric array with shape (n_set,). All values must be finite
        and distinct.
    dt : float
        Finite positive time step.
    temperature : float
        Finite positive temperature.

    Returns
    -------
    np.ndarray
        Float array with the same shape as ``raw_bias``. The barrier for each
        set is added to every active bias sample. Padding samples are replaced
        by zero. All returned values are finite.

    Raises
    ------
    ValueError
        If the archive arrays are not real numeric data with the required
        shapes, if n_set < 3, n_traj < 1, or n_frame < 2, if ``lengths`` is
        not an integer array or contains values outside [2, n_frame], if
        ``events`` contains nonfinite or non-binary values, if ``barriers``
        contains nonfinite or duplicate values, if ``dt`` or ``temperature``
        is not finite and positive, if an active bias sample is nonfinite,
        if the arrays cannot be represented as float64 where required, or if
        restoring the barrier produces a nonfinite float64 result.
    """
    return result
```

### Step 2

censored_observed_log_rates

Goal
----
Estimate per-set observed rates with a right-censored exponential MLE.

```python
def censored_observed_log_rates(
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    dt: float,
) -> 'np.ndarray':
    """Return one observed log-rate per simulation set.

    A trajectory of ``lengths[j,i]`` stored intervals contributes exposure
    ``lengths[j,i]*dt`` whether it transitions or is right-censored.  The
    event count is the sum of the binary ``events`` row.

    Raises ``ValueError`` for invalid arrays, zero-event sets, or nonfinite
    rates.
    """
    return result
```

### Step 3

ensemble_time_log_acceleration

Goal
----
Compute the flooding acceleration with the source averaging order.

```python
def ensemble_time_log_acceleration(
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    gamma: float,
) -> 'np.ndarray':
    """Return log acceleration factors for ragged trajectory sets.

    At each stored frame, average the exponential only over trajectories still
    present, then average those frame means uniformly over time. Use
    ``R=0.008314462618`` kJ mol^-1 K^-1, matching the pinned implementation,
    and a stable log-domain evaluation. Padding in ``restored_bias`` is ignored
    through ``lengths``.

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes or unsupported dtypes, if any
        length lies outside the available frame range, if ``temperature`` is
        nonfinite or not strictly positive, if ``gamma`` is nonfinite or lies
        outside the closed interval [0, 1], if an active bias sample is
        nonfinite, or if the returned log acceleration is nonfinite.
    """
    return result
```

### Step 4

fit_eatr_flooding

Goal
----
Fit a common flooding efficiency and unbiased log-rate across sets.

```python
def fit_eatr_flooding(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    indices: 'np.ndarray',
    gamma_min: float = 0.0,
    gamma_max: float = 1.0,
) -> 'np.ndarray':
    """Return ``[gamma, mean_log_k0, variance]`` for selected sets.

    Minimize the population variance of the selected per-set corrected
    log-rates with bounded scalar minimization using ``xatol=1e-12`` and
    ``maxiter=10000``, then report their mean and variance at the optimum.
    At least three distinct set indices are needed.

    Raises ``ValueError`` for invalid selections, bounds, arrays, or a
    nonfinite optimization result.
    """
    return result
```

### Step 5

scan_eatr_prefixes

Goal
----
Scan least-to-most-biased set prefixes for flooding-fit convergence.

```python
def scan_eatr_prefixes(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    order: 'np.ndarray',
    min_sets: int = 3,
) -> 'np.ndarray':
    """Return rows ``[n_sets,gamma,mean_log_k0,variance]`` for prefixes.

    ``order`` must be a permutation of all set indices already sorted from
    the smallest to largest full-efficiency log acceleration.  Fit every
    prefix from ``min_sets`` through the complete order.

    Raises ``ValueError`` for an invalid permutation or prefix control.
    """
    return result
```

### Step 6

select_consistent_prefix

Goal
----
Select a converged prefix without admitting boundary gamma fits.

```python
def select_consistent_prefix(
    prefix_table: 'np.ndarray',
    variance_limit: float,
    gamma_margin: float = 0.02,
) -> 'np.ndarray':
    """Return the largest eligible prefix row.

    Rows are ``[n_sets,gamma,mean_log_k0,variance]`` in increasing consecutive
    ``n_sets`` order.  An eligible row has variance no larger than
    ``variance_limit`` and gamma inside the closed interval
    ``[gamma_margin,1-gamma_margin]``.

    Raises ``ValueError`` for malformed diagnostics or no eligible prefix.
    """
    return result
```

### Step 7

opesf_cdf_log_rate

Goal
----
Compute the conventional OPES-f comparator from rescaled event times.

```python
def opesf_cdf_log_rate(
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    dt: float,
    temperature: float,
    indices: 'np.ndarray',
) -> float:
    """Return the pooled conventional OPES-f CDF-fit log-rate.

    Rescale every selected trajectory duration by its own time-average of
    ``exp(beta*V)``.  Pool the selected trajectories, sort only transitioned
    rescaled times, use empirical ordinates ``1/N,...,M/N`` with all ``N``
    pooled trajectories in the denominator, and fit ``1-exp(-k*t)`` by
    unweighted nonlinear least squares using ``ftol=xtol=gtol=1e-12`` and
    ``maxfev=100000``.  Return ``log(k)``.

    Raises ``ValueError`` for invalid inputs, unsafe exponentiation, or a
    nonpositive/nonfinite fit.
    """
    return result
```

### Step 8

leave_one_set_out_log_rates

Goal
----
Measure flooding-estimate sensitivity to removing each selected set.

```python
def leave_one_set_out_log_rates(
    log_k_observed: 'np.ndarray',
    restored_bias: 'np.ndarray',
    lengths: 'np.ndarray',
    temperature: float,
    indices: 'np.ndarray',
) -> 'np.ndarray':
    """Return refitted unbiased log-rates after each one-set deletion.

    Preserve the order of ``indices`` in the output.  Each deletion must leave
    at least three sets and is refitted independently with the bounded common
    efficiency calculation.

    Raises ``ValueError`` unless at least four distinct valid sets are given.
    """
    return result
```

### Step 9

run_flooding_rate_audit

Goal
----
Run the complete deterministic censored flooding-rate comparison.

```python
def run_flooding_rate_audit(
    seed: int,
    variance_limit: float = 0.003,
    influence_limit: float = 0.2,
) -> float:
    """Return the signed percent difference between two rate estimates.

    Rebuild the supplied seven-set archive for ``seed`` and execute steps
    1--8. Sort sets by full-efficiency log acceleration; scan prefixes from
    three sets; select the largest prefix passing ``variance_limit`` with a
    0.02 interior-gamma margin. Require the maximum absolute leave-one-set-out
    log-rate influence not to exceed ``influence_limit``. Return
    ``100*(k_flooding/k_OPESf - 1)`` without clipping or absolute values.

    Raises
    ------
    ValueError
        If ``seed`` is not an integer or is a Boolean, if ``variance_limit`` is
        nonfinite or negative, if ``influence_limit`` is nonfinite or not
        strictly positive, if an earlier step rejects the generated archive,
        if no eligible prefix exists, if the selected prefix and its direct
        refit disagree, if the leave-one-set-out influence exceeds
        ``influence_limit``, or if the final result is nonfinite.
    """
    return result
```
