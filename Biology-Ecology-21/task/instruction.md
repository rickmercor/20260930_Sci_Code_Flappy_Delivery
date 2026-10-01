# Biology-Ecology-21

## Background

Functional diversity describes variation in ecological traits and how species occupy multidimensional trait space. Conventional functional-diversity indices often summarize this structure with a single value, whereas multi-scale functional-arrangement approaches examine how species relationships change across a continuum of functional distances.

Pairwise-neighbour and nearest-neighbour summaries capture complementary aspects of functional organization. Pairwise-neighbour relationships describe broader structure using all interspecific distances, while nearest-neighbour relationships emphasize local spacing around individual species. These patterns may indicate clustering, random arrangement, or regularity at different functional-distance scales.

The ecological interpretation of an observed spatial pattern also depends on the null model against which it is evaluated. A regional species-pool null model preserves the locations of species available in the larger pool and tests whether the local community differs from random species selection. In contrast, a random-displacement null model evaluates arrangement relative to randomized locations constrained by the limits of the functional space. The benchmark below replaces that random placement with a single deterministic low-discrepancy sequence so that the result is reproducible; because a low-discrepancy sequence spreads its points more evenly than independent random draws, it is not distributionally equivalent to the source's random displacement, and the prescribed construction is to be used as stated. Differences between conclusions obtained from these null models therefore reveal sensitivity to the ecological process represented by the null expectation.

Boundary-constrained random displacement additionally requires a representation of the functional-space boundary. Radial-basis-function kernels provide a flexible way to represent nonlinear boundaries, and a data-driven median heuristic can be used to set the kernel distance parameter. Combining multi-scale arrangement statistics, alternative null models, boundary constraints, and standardized effect sizes allows functional organization to be evaluated under different ecological expectations. Because functional-arrangement conclusions can be driven disproportionately by individual species, robustness can also be evaluated with a leave-one-out sensitivity analysis. In this approach, each species in the observed local community is omitted in turn while the regional species pool and functional-space boundary are held fixed. The reduced community is then re-evaluated against null communities having the corresponding reduced richness. Comparing the full-community and leave-one-out results reveals whether disagreement between alternative null models is robust or strongly dependent on particular community members.

## Problem

Using the provided research source, calculate a single null-model sensitivity burden for the local community below; obtain from the source the exact PNcp and NNcp definitions, the RBF median heuristic and kernel convention, the ecological meaning of the two null models, the role of OCSVM in defining functional-space boundaries, and the SES standardization/classification framework.

Regional species pool (2D functional coordinates):

```text
Species    Dimension 1    Dimension 2
A          0.96           0.04
B          0.25           0.36
C          0.57           0.46
D          0.94           0.19
E          0.31           0.08
F          0.18           0.60
G          0.05           0.94
H          0.76           0.09
I          0.79           0.12
J          0.52           0.21
K          0.17           0.53
```

The observed community is A, D, F, H, and I, and the tested thresholds are r = (0.08, 0.18, 0.28, 0.36, 0.44, 0.52, 0.56); use Euclidean distances and include distances exactly equal to r.

Use an exhaustive fixed-richness regional-pool null consisting of every unique k-species subset of the 11-species pool, where k is the current local-community richness.

For the displacement null, fit one OCSVM boundary to the unchanged 11-species regional pool using nu = 0.05 together with the source-study RBF convention and median heuristic. To make this benchmark deterministic, if K is the resulting regional-pool Gram matrix, define alpha as the minimizer of 0.5*alpha^T*K*alpha subject to sum(alpha)=1 and 0<=alpha_i<=1/(nu*N); define rho as the mean of (K*alpha)_i over free support species satisfying 0<alpha_i<1/(nu*N), and evaluate new points with f(x)=sum_i alpha_i*K(x,x_i)-rho. Solve for alpha and rho accurately enough that they satisfy the KKT optimality conditions to within 1e-10; a solution accurate only to a general-purpose solver's default tolerance is not sufficient, because which candidate points the boundary accepts turns on the fourth decimal of rho. The regional pool, fitted boundary, and kernel scale remain fixed during all leave-one-out analyses; the boundary is fitted once and is not refitted for any reduced community.

Generate standard unscrambled two-dimensional Halton points - the plain radical-inverse sequence, with no scrambling or randomisation - using bases 2 and 3 for indices 1 through 300, retain points with f(x)>=0, preserve their original order, partition accepted points into consecutive groups of k, and discard any incomplete final group.

Use sample standard deviations with denominator N-1, retain full numerical precision, and use the rounded critical SES magnitude stated by the source study for its conservative two-tailed p<0.01 band rather than recomputing an exact normal quantile; equality to either critical boundary is treated as random.

For the full community, count every PNcp- or NNcp-by-r cell whose ecological classification differs between the two null models.

Repeat the analysis after omitting A, D, F, H, and I one at a time, keeping the regional pool and fitted boundary fixed while allowing all richness-dependent null quantities to change, then sum the full-community disagreement count and the five leave-one-out disagreement counts.

Return one integer: the null-model sensitivity burden.

## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Report this instance's own intermediate results rather than a general description of the method: the regional-pool median pairwise distance and the resulting kernel parameter; the number of candidate points the fitted boundary accepts and the number of complete groups they form; the full-community observed PNcp and NNcp curves together with each null model's standardized effect sizes and classifications; the full-community disagreement count; the disagreement count for each of the five leave-one-out communities; and the final burden. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

01_compute_observed_arrangement_curves

Goal
----
Compute the two functional-arrangement curves defined in the provided research source for a community at the supplied distance thresholds. Preserve threshold order, including repeated thresholds, and return both curves together as a two-row numerical array. Derive the source-defined statistics from the research source rather than hard-coding benchmark-specific values.

```python
def compute_observed_arrangement_curves(
    coords: "np.ndarray",
    r_values: "np.ndarray"
) -> "np.ndarray":
    """Compute observed PNcp and NNcp curves.

    Parameters
    ----------
    coords : np.ndarray
        Finite array of shape (n_species, n_dimensions), with at
        least two species and one dimension.
    r_values : np.ndarray
        Non-empty one-dimensional array of finite, non-negative
        distance thresholds. Input order and repeated values must
        be preserved.

    Returns
    -------
    curves : np.ndarray
        Float array of shape (2, len(r_values)).
        Row 0 contains PNcp and row 1 contains NNcp.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    return curves
```

### Step 2

02_compute_exact_pool_null_certificate

Goal
----
Compute an exact, exhaustive second-moment certificate of the fixed-richness regional-pool null for the complete PNcp/NNcp threshold battery.



Let S range over every unique community_size-species subset of the regional pool. For threshold r_t, let X_t(S) be the number of ordered pairs of distinct species in S whose Euclidean distance is <= r_t (the PNcp numerator), and let Y_t(S) be the number of species in S whose nearest-neighbour distance within S is <= r_t (the NNcp numerator). With z(S) = (1, X_1(S), ..., X_T(S), Y_1(S), ..., Y_T(S)), return the integer matrix M = sum over all S of z(S) z(S)^T.



M must be exact. Do not enumerate, sample, or materialize subsets; the running time must be polynomial in the pool size, and inputs with more than 10^8 subsets must be handled. Thresholds keep their input order, repeated thresholds are allowed, and distances exactly equal to a threshold (including zero distances between duplicated coordinates) count as within the threshold.

```python
def compute_exact_pool_null_certificate(
    pool_coords: "np.ndarray",
    community_size: int,
    r_values: "np.ndarray"
) -> "np.ndarray":
    """Return the exact exhaustive pool-null second-moment certificate.

    Parameters
    ----------
    pool_coords : np.ndarray
        Finite regional-pool coordinates with shape (n_species, 2) and at
        least three species. Duplicated coordinates are allowed.
    community_size : int
        Fixed richness k of every null community; 2 <= k < n_species.
    r_values : np.ndarray
        Non-empty one-dimensional array of finite, non-negative thresholds.
        Input order and repeated values are preserved.

    Returns
    -------
    certificate : np.ndarray
        int64 array M of shape (1 + 2T, 1 + 2T), M = sum_S z(S) z(S)^T with
        z(S) = (1, X_1..X_T, Y_1..Y_T) as defined in the description.

    Raises
    ------
    ValueError
        If an input is invalid or an entry cannot be represented exactly.
    """
    return certificate
```

### Step 3

03_fit_regional_boundary_certificate

Goal
----
Fit the regional functional-space boundary and certify it.



Set sigma to the reciprocal of the median Euclidean distance over all unique regional-pool species pairs and use K(x,s)=exp(-sigma*||x-s||^2). With c = 1/(nu*N), alpha is the exact minimizer of 0.5*alpha^T K alpha subject to sum(alpha)=1 and 0<=alpha_i<=c; rho is the mean of (K alpha)_i over free support species (0<alpha_i<c). The returned alpha and rho must satisfy the KKT optimality conditions to within 1e-10; a solution accurate only to optimizer tolerance is not sufficient.



The optimal solution partitions the species into zero, free, and upper-bounded dual coefficients. Return also the exact endpoints nu_lo <= nu <= nu_hi of the maximal closed interval of nu on which this partition remains optimal (use nu_lo = 0 if the partition persists as nu -> 0, and cap nu_hi at 1).



Return [sigma, alpha_1, ..., alpha_N, rho, nu_lo, nu_hi].

```python
def fit_regional_boundary_certificate(
    pool_coords: "np.ndarray",
    nu: float = 0.05
) -> "np.ndarray":
    """Fit and certify the regional OCSVM boundary.

    Parameters
    ----------
    pool_coords : np.ndarray
        Finite regional-pool coordinates with shape (n_species, 2) and at
        least three species.
    nu : float, default=0.05
        OCSVM parameter strictly between 0 and 1; the dual upper bound is
        1/(nu*n_species).

    Returns
    -------
    certificate : np.ndarray
        Float array [sigma, alpha_1..alpha_N, rho, nu_lo, nu_hi].

    Raises
    ------
    ValueError
        If an input is invalid, the median pairwise distance is not positive,
        the optimum has no free support species, or the KKT conditions cannot
        be certified.
    """
    return certificate
```

### Step 4

04_analyze_full_community_dual_null

Goal
----
### Compute the observed PNcp/NNcp curves; derive the exhaustive regional-pool null means and sample standard deviations from the exact pool-null certificate; and compute the displacement-null moments by applying the certified regional boundary to ordered two-dimensional Halton candidates (bases 2 and 3, indices 1..n_candidates), retaining f(x)>=0 in order, forming consecutive groups of the community richness, and discarding an incomplete final group. Standardize both nulls with the supplied critical magnitude, classify with strict thresholds, and mark every statistic-by-threshold cell where the two classifications differ.

```python
def analyze_full_community_dual_null(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray",
    nu: float = 0.05,
    n_candidates: int = 300,
    critical: float = 2.58
) -> "np.ndarray":
    """Build the complete full-community dual-null evidence tensor.

    Parameters
    ----------
    pool_coords : np.ndarray
        Finite regional-pool coordinates with shape (n_species, 2)
        and at least four species.
    observed_indices : np.ndarray
        One-dimensional array of unique valid pool indices defining
        an observed community with at least three species and richness
        smaller than the regional pool.
    r_values : np.ndarray
        Non-empty one-dimensional array of finite, non-negative thresholds.
        Input order and repeated values are preserved.
    nu : float, default=0.05
        OCSVM parameter strictly between 0 and 1. The boundary is fitted from
        the unchanged regional pool.
    n_candidates : int, default=300
        Positive number of deterministic Halton candidates.
    critical : float, default=2.58
        Positive finite SES critical magnitude; equality to either
        boundary remains classified as random.

    Returns
    -------
    evidence : np.ndarray
        Float array of shape (2, 10, len(r_values)). Axis 1 is ordered as
        observed value, pool mean, pool sample SD, pool SES, pool class,
        displacement mean, displacement sample SD, displacement SES,
        displacement class, disagreement indicator.

    Raises
    ------
    ValueError
        If any input is invalid or a downstream null-model requirement fails.
    """
    return evidence
```

### Step 5

05_compute_all_leave_one_out_observed_curves

Goal
----
Compute the observed PNcp and NNcp curves for every leave-one-out community in one batched calculation.



Start from the observed community selected from the regional pool and return the two observed arrangement curves after omitting each observed species in turn, preserving observed_indices order. Reuse the full pairwise-distance structure rather than recomputing a complete pairwise analysis separately for every omission. The implementation must remain exact for repeated or unsorted thresholds and for tied nearest-neighbour distances.



The required running-time target is quadratic in observed-community richness for constructing the distance structure, plus work proportional to the returned richness-by-threshold output. A cubic leave-one-out recomputation is not acceptable.

```python
def compute_all_leave_one_out_observed_curves(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray"
) -> "np.ndarray":
    """Compute all leave-one-out observed arrangement curves efficiently.

    Parameters
    ----------
    pool_coords : np.ndarray
        Finite regional-pool coordinates with shape (n_species, 2).
    observed_indices : np.ndarray
        One-dimensional array of unique valid pool indices defining an
        observed community with at least four species and richness smaller
        than the regional pool.
    r_values : np.ndarray
        Non-empty one-dimensional array of finite, non-negative thresholds.
        Input order and repeated values are preserved.

    Returns
    -------
    curves : np.ndarray
        Float array of shape (k, 2, len(r_values)), where k is the observed
        richness. curves[j] contains PNcp then NNcp after omitting the
        j-th entry of observed_indices.

    Raises
    ------
    ValueError
        If the pool, observed indices, or thresholds are invalid.
    """
    return curves
```

### Step 6

06_compute_leave_one_out_disagreement_profile

Goal
----
Compute the complete cross-null disagreement profile for the full community and every leave-one-out perturbation while reusing richness-dependent null quantities.



Analyze the full observed community once. For the leave-one-out communities, first obtain every reduced observed PNcp/NNcp curve in one batched calculation. Because every leave-one-out community has the same richness and the regional pool and fitted OCSVM boundary remain fixed, compute the fixed-richness pool-null moments and displacement-null moments once for richness k-1, then standardize and classify all leave-one-out observed curves against those shared null moments.



Return the full-community disagreement count followed by one count for each omission in observed_indices order.

```python
def compute_leave_one_out_disagreement_profile(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray",
    nu: float = 0.05,
    n_candidates: int = 300,
    critical: float = 2.58
) -> "np.ndarray":
    """Return full-community and ordered leave-one-out disagreement counts.

    Parameters
    ----------
    pool_coords : np.ndarray
        Finite regional-pool coordinates with shape (n_species, 2).
    observed_indices : np.ndarray
        One-dimensional array of unique valid pool indices defining an
        observed community with at least four species and richness smaller
        than the regional pool.
    r_values : np.ndarray
        Non-empty one-dimensional array of finite, non-negative thresholds.
    nu : float, default=0.05
        OCSVM parameter strictly between 0 and 1. The boundary reference is
        determined from the unchanged regional pool.
    n_candidates : int, default=300
        Positive number of deterministic Halton candidates.
    critical : float, default=2.58
        Positive finite SES critical magnitude.

    Returns
    -------
    profile : np.ndarray
        Integer array of length 1 + len(observed_indices). Element 0 is the
        full-community disagreement count; subsequent elements correspond
        to omissions in observed_indices order.

    Raises
    ------
    ValueError
        If an input is invalid or a downstream null-model requirement fails.
    """
    return profile
```

### Step 7

07_compute_null_model_sensitivity_burden

Goal
----
Compute the final null-model sensitivity burden from the raw regional pool, observed-community definition, distance thresholds, and regional-pool-fitted OCSVM boundary.

Use the preceding task components to construct the full-community evidence and the batched leave-one-out analysis while preserving the fixed regional reference structures. Verify that independently assembled intermediate outputs agree, then sum the ordered full-plus-leave-one-out disagreement profile and return one integer burden. Do not hard-code intermediate classifications, disagreement counts, profiles, or final values.

```python
def compute_null_model_sensitivity_burden(
    pool_coords: "np.ndarray",
    observed_indices: "np.ndarray",
    r_values: "np.ndarray",
    nu: float = 0.05,
    n_candidates: int = 300,
    critical: float = 2.58
) -> int:
    """Compute the full and leave-one-out null-model sensitivity burden.

    Parameters
    ----------
    pool_coords : np.ndarray
        Finite regional-pool coordinates with shape (n_species, 2).
    observed_indices : np.ndarray
        One-dimensional array of unique valid pool indices defining an
        observed community with at least four species.
    r_values : np.ndarray
        Non-empty one-dimensional array of finite, non-negative thresholds.
    nu : float, default=0.05
        OCSVM parameter strictly between 0 and 1. The boundary reference is
        fitted from the unchanged regional pool.
    n_candidates : int, default=300
        Positive number of deterministic Halton candidates.
    critical : float, default=2.58
        Positive finite SES critical magnitude.

    Returns
    -------
    sensitivity_burden : int
        Sum of cross-null disagreement counts for the full community and
        every leave-one-out community in observed_indices order.

    Raises
    ------
    ValueError
        If an input is invalid or a downstream null-model requirement fails.
    RuntimeError
        If independently assembled intermediate results are inconsistent.
    """
    return sensitivity_burden
```
