# Biology-Genetics-5

## Background

This is a synthetic sensitivity experiment, not a reconstruction of a measured locus. Correlated genetic variants near a gene can provide competing explanations for variation in that gene's expression. Functional annotations may contribute directional information, but conclusions from an approximate statistical model must be distinguished from experimentally established causality.

## Problem

For the synthetic cis-eQTL locus specified below, use the 2026 fine-mapping framework that combines signed functional priors with a primary cluster-weighted multi-basin ensemble to quantify the sensitivity of variant 2's ensemble posterior inclusion probability (PIP) to the sign of its functional annotation.
Use two single-effect components, fixed residual variance 1, uniform variant prior probabilities, and conditional prior effect mean `c*a[j]` with common prior variance `v0` in each fit.
Fit exactly the Cartesian grid `c = [0, 0.4, 0.8, 1.2]` and `v0 = [0.04, 0.12, 0.30]`, with `c` as the outer loop, starting every component's expected effect vector at zero and performing exactly 300 full cyclic IBSS sweeps in component order 1 then 2.
Keep all 12 fits and use the paper's primary cluster-weight aggregation, with its credible-shift dissimilarity, complete linkage cut at 0.05 and unit temperature; use the Gaussian ELBO including the `n=61` normalizing constant and normalized variant priors, with no empirical-Bayes updates, annealing, rescaling, pruning or early stopping.
Repeat the entire calculation after changing only `a[2]` from `-0.35` to `+0.35`, where variant indices here are one-based.
Return the flipped-minus-original difference in the ensemble PIP of variant 2, rounded to six decimal places.
In a short scientific explanation, give the single-effect conditional mean and variance and normalized inclusion-weight expressions, the uncertainty-aware expected residual sum of squares and ELBO expressions, the within-fit PIP expression, and the paper's dissimilarity and two-level weighting rule; also report the cluster count, maximum grid ELBO and ensemble PIP of variant 2 for each condition, retaining at least eight decimal places for the latter two quantities.
Identify the source sections supporting the aggregation rule, and interpret the contrast as model sensitivity rather than experimentally established causality.

The sufficient statistics are `G = X.T @ X`, `t = X.T @ y`, `n = 61` and `y.T @ y = 60`; use their supplied values directly.

```text
G =
[[60,     52.992,   5.76,    0.9,   16.62, -3.84],
 [52.992, 60,      11.1,    -5.4,   11.94, -8.28],
 [5.76,   11.1,    60,     -50.22,   9.57, -0.9 ],
 [0.9,    -5.4,   -50.22,   60,    -0.45,  7.2 ],
 [16.62,  11.94,    9.57,   -0.45,  60,   39   ],
 [-3.84,  -8.28,   -0.9,     7.2,   39,   60   ]]
t = [26, 24, -22, 20, 13, 10]
a = [0.45, -0.35, -0.40, 0.50, 0.15, -0.25]
```

Return exactly one finite decimal inside `<final_answer>...</final_answer>` first, followed by the requested short scientific explanation inside `<reasoning>...</reasoning>`. Do not put units, prose or other numbers inside the final-answer tags, or any text outside these two blocks.

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

step_01_single_effect_posterior

Goal
----
Compute one signed-prior single-effect posterior from residual sufficient statistics.

```python
def single_effect_posterior(residual_cross: 'np.ndarray | list', diagonal: 'np.ndarray | list', noise: float, prior_variance: float, prior_mean: 'np.ndarray | list') -> 'np.ndarray':
    """Return a float ndarray of shape (3,p).

    Parameters and contract
    -----------------------
    residual_cross, diagonal and prior_mean are finite float arrays of shape
    (p,), p >= 1. residual_cross is X.T @ residual; diagonal contains positive
    diagonal entries of X.T @ X. noise is a positive residual variance, and
    prior_variance is a positive common conditional effect variance.
    A single factor selects one variant with uniform prior probability 1/p.
    Row 0 is the posterior variant selection probability; rows 1 and 2 are
    the effect mean and variance conditional on selecting each variant.
    Preserve variant order and signed prior means. Return unrounded values.
    Inputs satisfy these domains; invalid-input behavior is not tested.

    Returns
    -------
    Float ndarray (3,p): selection probabilities, conditional means, conditional variances."""
    return None
```

### Step 2

step_02_expected_residual_ss

Goal
----
Evaluate expected Gaussian residual energy under independent single-effect factors.

```python
def expected_residual_ss(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, posterior: 'np.ndarray | list') -> float:
    """Return the unrounded expected residual sum of squares as a float.

    Parameters and contract
    -----------------------
    gram is a finite positive-definite (p,p) Gram matrix, cross is a finite
    (p,) vector, and yty is y.T @ y. These are feasible sufficient statistics.
    posterior is a finite array of shape (L,3,p), L,p >= 1. Within each factor,
    row 0 contains nonnegative selection probabilities summing to one, row 1
    contains conditional means, and row 2 contains nonnegative conditional
    variances. Factors are independent; each selects exactly one variant.
    Include posterior uncertainty. Zero probabilities and zero variances are
    valid here. Do not change the supplied Gram scaling or mutate inputs.
    Inputs satisfy these domains; invalid-input behavior is not tested.

    Returns
    -------
    Unrounded Python float, the expected residual sum of squares."""
    return None
```

### Step 3

step_03_variational_objective

Goal
----
Evaluate the complete Gaussian evidence lower bound of a supplied variational posterior.

```python
def variational_objective(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, noise: float, prior_variance: float, prior_mean: 'np.ndarray | list', posterior: 'np.ndarray | list') -> float:
    """Return the unrounded Gaussian variational objective in nats as a float.

    Parameters and contract
    -----------------------
    gram (p,p), cross (p,), yty and positive integer n describe feasible
    sufficient statistics; use gram directly without rescaling. noise and
    prior_variance are positive variance scalars; prior_mean has shape (p,).
    posterior has shape (L,3,p): selection probabilities, conditional means,
    and strictly positive conditional variances. Probabilities are
    nonnegative and sum to one within each independent single-effect factor.
    Each prior selects uniformly among p variants. Include the complete
    n-sample Gaussian likelihood and categorical/normal KL terms, with
    zero-probability entropy terms interpreted by continuity. Do not mutate
    inputs. Invalid-input behavior is not tested.

    Returns
    -------
    Unrounded Python float, complete Gaussian ELBO in nats."""
    return None
```

### Step 4

step_04_fit_single_setting

Goal
----
Fit one fixed signed-prior setting using a prescribed finite cyclic update schedule.

```python
def fit_single_setting(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, annotation: 'np.ndarray | list', scale: float, prior_variance: float, effects: int=2, noise: float=1.0, sweeps: int=300) -> 'tuple[np.ndarray, float, np.ndarray]':
    """Return (pip, elbo, posterior), without rounding.

    Parameters and contract
    -----------------------
    gram (p,p), cross (p,), yty and positive integer n are feasible finite
    sufficient statistics with positive-definite gram. annotation is (p,).
    scale is a finite scalar; prior_variance and noise are positive variance
    scalars. effects and sweeps are positive integers. The conditional prior
    mean is scale*annotation and variant selection priors are uniform.
    Initialize every component's expected coefficient vector to zero. Perform
    exactly sweeps full cyclic IBSS sweeps, in ascending component order,
    using each new update immediately. Do not rescale, update hyperparameters,
    prune, anneal or stop early. pip is a (p,) float ndarray; elbo is a float
    including the n-sample normalizer; posterior is (effects,3,p) with rows
    selection probability, conditional mean and conditional variance.
    Do not mutate inputs. Invalid-input behavior is not tested.

    Returns
    -------
    Tuple of (float ndarray (p,), Python float, float ndarray (effects,3,p)): PIP, ELBO, posterior."""
    return None
```

### Step 5

step_05_partition_fits

Goal
----
Partition fitted inclusion vectors by credible-shift complete-linkage clustering.

```python
def partition_fits(pips: 'np.ndarray | list', cutoff: float=0.05) -> 'np.ndarray':
    """Return an integer ndarray of canonical cluster labels, shape (M,).

    Parameters and contract
    -----------------------
    pips is a finite (M,p) array in [0,1], M,p >= 1; cutoff is nonnegative.
    Use the paper's credible-shift dissimilarity with complete linkage.
    Merge the closest pair while its linkage distance is <= cutoff.
    In an exact distance tie, choose the lexicographically smallest ordered
    pair of sorted original-member-index tuples. Clusters in such a pair
    are themselves ordered lexicographically. Label final clusters 0,1,...
    by ascending smallest original fit index, preserving input fit order.
    Do not mutate inputs. Invalid-input behavior is not tested.

    Returns
    -------
    Integer ndarray (M,), canonical zero-based cluster labels in input order."""
    return None
```

### Step 6

step_06_aggregate_ensemble

Goal
----
Aggregate fitted inclusion vectors using the primary cluster-weight ensemble.

```python
def aggregate_ensemble(pips: 'np.ndarray | list', elbos: 'np.ndarray | list', labels: 'np.ndarray | list') -> 'tuple[np.ndarray, np.ndarray]':
    """Return (aggregate_pip, fit_weights), both unrounded float ndarrays.

    Parameters and contract
    -----------------------
    pips is a finite probability matrix (M,p), M,p >= 1. elbos is a finite
    length-M float array, and labels is a length-M integer array identifying
    clusters. Labels may be nonconsecutive or negative and have no numerical
    meaning. Apply SuSiNE's primary cluster-weight rule at unit temperature
    at both levels, not its naive ensemble. aggregate_pip has shape (p,);
    fit_weights has shape (M,) in original fit order and sums to one.
    Do not mutate inputs. Invalid-input behavior is not tested.

    Returns
    -------
    Tuple of float ndarrays ((p,), (M,)): ensemble PIP and per-fit weights."""
    return None
```

### Step 7

step_07_annotation_sensitivity

Goal
----
Run both signed-annotation ensembles and compute the final inclusion-probability contrast.

```python
def annotation_sensitivity(gram: 'np.ndarray | list', cross: 'np.ndarray | list', yty: float, n: int, annotation: 'np.ndarray | list', variant: int, scales: 'np.ndarray | list', variances: 'np.ndarray | list', effects: int=2, noise: float=1.0, sweeps: int=300, cutoff: float=0.05) -> float:
    """Return the unrounded flipped-minus-original variant PIP as a float.

    Parameters and contract
    -----------------------
    gram, cross, yty, n, annotation, effects, noise and sweeps have the same
    valid domains as fit_single_setting. variant is a zero-based integer
    index in [0,p). scales and variances are nonempty finite one-dimensional
    arrays; variances are positive. cutoff is nonnegative.
    Fit all Cartesian pairs in both conditions, with scales as the outer
    loop and variances as the inner loop. Flip only annotation[variant] in
    the second condition. Retain every fit, use the stated zero-start finite
    cyclic schedule, partition with the preceding complete-linkage convention
    and apply the primary unit-temperature cluster-weight ensemble. The
    returned contrast refers to the same variant index in both conditions.
    Do not round intermediate or final results, or mutate any input.
    Invalid-input behavior is not tested.

    Returns
    -------
    Unrounded Python float, flipped-minus-original ensemble PIP at the zero-based variant index."""
    return None
```
