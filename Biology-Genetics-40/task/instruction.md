# Biology-Genetics-40

## Background

A genetic variant rarely acts everywhere at once. It may change the expression of a gene in monocytes and not in T cells, or raise the risk of two cancers out of the five in a consortium, and a meta-analysis that pools every study with equal enthusiasm dilutes such a signal until it disappears. The standard answer is to stop assuming the effect is shared and to search for the sharing pattern instead: form the fixed-effect pooled statistic of every nonempty subset of the studies, or of the cell types, and keep the largest one in absolute value. This recovers the power a fixed-effect pool throws away and, as a by-product, names the subset that drives the association, which is what makes the result interpretable rather than merely significant. Subset-based scans of this kind are now routine in cross-trait genome-wide association studies, in multi-ancestry meta-analysis and, most recently, in single-cell expression quantitative trait locus mapping, where the cell types of a tissue play the role of the studies.

The price is a multiple testing problem with an unusual shape. There are exponentially many subsets, but they overlap heavily, and two subsets that differ by one member produce statistics that are almost the same number, so treating the scan as a family of independent tests is hopelessly conservative while ignoring the search altogether is anti-conservative. The correction in use handles this by treating the collection of subset statistics as a random field indexed by the subsets, with neighbours defined by adding or dropping one member, and by decomposing the event that the field exceeds a threshold over the sites at which the field is a local maximum. That decomposition converts an intractable maximum into a sum of one-dimensional integrals, one per subset, each involving only the correlations between a subset and its immediate neighbours. It is a genuinely good approximation, and the significance it produces has been checked against direct simulation in the range where direct simulation is affordable.

Every step of that argument, however, assumes the underlying study statistics are jointly normal. In a well powered genome-wide association study of a common variant that assumption is unimpeachable, and the correction can be trusted at the significance levels genetics actually uses, which reach ten to the minus eight and beyond. Single-cell expression data break it from three directions at once. Cohorts are small, often one or two hundred donors, because each donor is an expensive sequencing library rather than a genotyping array. Pseudobulk expression is not normal: a gene is silent in a large share of donors within a cell type, so its distribution is a point mass at the detection floor with a skewed continuous body above it, and no monotone transformation removes the point mass. And the variants that matter most are rare, so the genotype vector is a lattice variable that is almost always zero and occasionally one, meaning a single donor can move the statistic by a large fraction of its whole range. A sum of a hundred and twenty such terms is close to normal in the middle and nowhere near normal five standard deviations out, which is precisely where the correction is evaluated.

Checking the correction where it matters is itself the hard part, because the probabilities involved are far below what simulation can reach. Estimating a probability of ten to the minus ten to within ten per cent by drawing genotypes from the null needs of order ten to the twelve draws, each requiring the evaluation of every subset statistic. The way out is to change the law being simulated. If one simulates instead from a tilted version of the null under which the rare event is common, and divides each replicate by the likelihood ratio that generated it, the estimator remains unbiased for the original probability while its variance is set by how well the tilted law matches the event rather than by how rare the event is. Choosing the tilt so that the mean of the statistic sits on the threshold makes roughly half the replicates informative and collapses the variance, and because the statistic is a sum of independent per-subject contributions, the tilted law factorizes into independent per-subject laws that are as cheap to sample as the null. A scan over subsets needs a mixture of such tilted laws rather than a single one, since no subset dominates, and a two-sided scan needs both signs of the tilt.

What this makes possible is a like-for-like comparison. Two numbers describe the same tail event for the same data and differ only in what they assume about the randomness: one propagates a multivariate normal law through the subset scan, the other conditions on the observed expression and puts the randomness in the genotypes where it actually is. Their ratio is a direct measure of how much the normal approximation costs; its sign says whether the analytic correction is anti-conservative and manufacturing false discoveries or conservative and hiding real ones, and its dependence on allele frequency, cohort size and the shape of the expression distribution tells a practitioner when the cheap correction is safe and when it is not.

## Problem

Reliability of the all-subset significance correction in a single-cell eQTL scan

Pooling a variant's evidence across cell types by scanning all $2^C-1$ nonempty subsets and keeping the largest absolute fixed-effect meta-analysis statistic buys power against effects confined to a few cell types, at the price of a multiple-testing correction over a heavily overlapping family; the correction in standard use for this scan is analytic, derived for a Gaussian field of subset statistics, and has been validated mainly at moderate significance levels. Single-cell eQTL mapping applies it exactly where that premise is weakest, because cohorts are small, pseudobulk expression is zero-inflated rather than normal, and the variants of interest are rare: conditional on the observed expression matrix the only randomness under the global null is a genotype vector of independent three-point variables, so every subset statistic is a fixed linear form in that vector and has no reason to be Gaussian in the tail where the correction is being used. For the configuration below, report the base-ten logarithm of the analytic significance divided by a genotype-conditional estimate of the same probability, the estimate being required to be unbiased under the conditional null and to resolve a probability far beyond the reach of direct simulation.

Testbed. Expression panel: $N=120$ subjects and $C=7$ cell types; with `numpy.random.default_rng(20260824)` draw a subject factor `standard_normal(120)` and then an idiosyncratic array `standard_normal((120, 7))`, form the latent value of subject $i$ in cell type $c$ as $\sqrt{0.4}$ times the factor of subject $i$ plus $\sqrt{0.6}$ times the corresponding idiosyncratic entry, subtract from it the standard normal quantile of $0.6$ and clamp negative results to zero, then standardise every column to mean zero and unit variance. Variant: minor allele frequency $f=0.02$ with genotypes coded $0,1,2$ under Hardy-Weinberg equilibrium; the score statistic of cell type $c$ is $z_c=\sum_i y_{ic}\,(g_i-2f)/(\sqrt{N}\,\sigma_g)$ with $\sigma_g^2=2f(1-f)$, and $Z_A$ is the fixed-effect pooling of $\{z_c\}_{c\in A}$ that is efficient under a common effect across the subset and has unit null variance given the panel. Observed statistic: $b=7.0$. Estimator: $5\times10^{4}$ replicates driven by `numpy.random.default_rng(2026)`, each replicate drawing in this order a subset index with `integers`, then one `random()` whose value below one half selects the negative branch, then `random(120)` mapped to genotypes by the inverse cumulative rule; subsets are indexed so that subset $m-1$ contains cell type $c$ whenever bit $c$ of the integer $m$ is set, for $m=1,\ldots,2^{C}-1$.

Your final answer must be a single number: $\log_{10}$ of the analytic significance divided by the estimated significance.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules: - The tags are required. Do not omit them or leave them empty. - The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose. - Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_build_expression_panel

Goal
----
Generate the deterministic standardized single-cell expression panel that every later step conditions on.

```python
import numpy as np

def build_expression_panel(n_subjects: int, n_cell_types: int, zero_fraction: float,
                           factor_correlation: float, seed: int) -> np.ndarray:
    """Build a standardised zero-inflated expression panel with correlated cell types.

    A latent Gaussian value is formed for every subject and cell type as
    sqrt(factor_correlation) times a subject-level factor plus
    sqrt(1 - factor_correlation) times an independent idiosyncratic term. The
    latent values are censored from below at the standard normal quantile of
    zero_fraction, entries at or below it becoming exactly zero and entries
    above it being shifted down by that quantile. Each column is then
    standardised to mean zero and unit variance using the population standard
    deviation.

    Randomness comes from numpy.random.default_rng(seed), which draws the
    subject factor of length n_subjects first and the idiosyncratic array of
    shape (n_subjects, n_cell_types) second.

    Parameters
    ----------
    n_subjects : int
        Number of subjects, n_subjects >= 2.
    n_cell_types : int
        Number of cell types, 1 <= n_cell_types <= 14.
    zero_fraction : float
        Expected fraction of censored (zero) entries, 0 < zero_fraction < 1.
    factor_correlation : float
        Fraction of latent variance carried by the shared subject factor,
        0 <= factor_correlation < 1.
    seed : int
        Seed of the random generator.

    Returns
    -------
    panel : np.ndarray
        Array of shape (n_subjects, n_cell_types), each column having mean
        zero and unit variance.
    """
    return panel  # placeholder
```

### Step 2

02_compute_subset_weights

Goal
----
Express every subset meta-analysis statistic as a fixed linear form in the centred genotypes and return the table of per-subject coefficients.

```python
import numpy as np

def compute_subset_weights(panel: np.ndarray, maf: float) -> np.ndarray:
    """Return the per-subject coefficients of every subset meta-analysis statistic.

    Subsets are enumerated in binary order: row m - 1 of the returned table
    corresponds to the subset containing cell type c whenever bit c of the
    integer m is set, for m = 1, ..., 2**n_cell_types - 1.

    Parameters
    ----------
    panel : np.ndarray
        Standardised expression panel of shape (n_subjects, n_cell_types),
        n_subjects >= 2 and 1 <= n_cell_types <= 14.
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.

    Returns
    -------
    weights : np.ndarray
        Array of shape (2**n_cell_types - 1, n_subjects) whose row A holds the
        coefficient of every subject in the subset statistic of A.
    """
    return weights  # placeholder
```

### Step 3

03_compute_subset_correlation

Goal
----
Build the null correlation matrix of the family of subset meta-analysis statistics from the table of per-subject coefficients.

```python
import numpy as np

def compute_subset_correlation(weights: np.ndarray, maf: float) -> np.ndarray:
    """Return the null correlation matrix of the subset meta-analysis statistics.

    Parameters
    ----------
    weights : np.ndarray
        Table of per-subject coefficients of shape (n_subsets, n_subjects),
        n_subsets >= 1 and n_subjects >= 1.
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.

    Returns
    -------
    correlation : np.ndarray
        Symmetric array of shape (n_subsets, n_subsets) with unit diagonal and
        entries in the interval [-1, 1].
    """
    return correlation  # placeholder
```

### Step 4

04_build_neighbour_table

Goal
----
Enumerate, for every nonempty subset of cell types, the subsets reached by toggling one cell type, marking the toggles that leave the admissible set.

```python
import numpy as np

def build_neighbour_table(n_cell_types: int) -> np.ndarray:
    """Return the one-cell-type-toggle neighbour table of the nonempty subsets.

    Subsets are enumerated in binary order: row m - 1 corresponds to the subset
    containing cell type c whenever bit c of the integer m is set, for
    m = 1, ..., 2**n_cell_types - 1.

    Parameters
    ----------
    n_cell_types : int
        Number of cell types, 1 <= n_cell_types <= 14.

    Returns
    -------
    neighbours : np.ndarray
        Integer array of shape (2**n_cell_types - 1, n_cell_types) whose entry
        (a, c) is the index of the subset obtained from subset a by toggling
        cell type c, or -1 when that toggle is inadmissible.
    """
    return neighbours  # placeholder
```

### Step 5

05_compute_dlm_pvalue

Goal
----
Evaluate the analytic discrete local maxima significance of the all-subset maximum for a Gaussian family of subset statistics.

```python
import numpy as np

def compute_dlm_pvalue(correlation: np.ndarray, neighbours: np.ndarray, threshold: float,
                       n_nodes: int = 400, tail_width: float = 8.0) -> float:
    """Return the discrete local maxima significance of the all-subset maximum.

    The level integral of each subset is truncated at threshold + tail_width
    and evaluated with n_nodes Gauss-Legendre nodes on that interval.
    Correlations are clipped away from plus or minus one by 1e-12 before the
    conditional standard deviation is formed.

    Parameters
    ----------
    correlation : np.ndarray
        Symmetric unit-diagonal correlation matrix of the subset statistics,
        of shape (n_subsets, n_subsets).
    neighbours : np.ndarray
        Integer neighbour table of shape (n_subsets, n_cell_types); entries
        below zero mark inadmissible toggles.
    threshold : float
        Observed value of the all-subset maximum, threshold > 0.
    n_nodes : int
        Number of Gauss-Legendre nodes, n_nodes >= 2.
    tail_width : float
        Width of the truncated level interval, tail_width > 0.

    Returns
    -------
    pvalue : float
        Analytic significance as a native Python float.
    """
    return pvalue  # placeholder
```

### Step 6

06_compute_subset_cgf

Goal
----
Evaluate the conditional cumulant generating function of one subset statistic under the genotype null, together with its first two derivatives.

```python
import numpy as np

def compute_subset_cgf(weight_row: np.ndarray, maf: float, tilt: float) -> tuple:
    """Return the conditional cumulant generating function of one subset statistic.

    The statistic is the sum over subjects of weight_row[i] times the centred
    genotype g_i - 2 * maf, with g_i taking the values 0, 1 and 2 with the
    Hardy-Weinberg probabilities of the allele frequency maf.

    Parameters
    ----------
    weight_row : np.ndarray
        Per-subject coefficients of one subset, of shape (n_subjects,) with
        n_subjects >= 1.
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.
    tilt : float
        Argument of the cumulant generating function.

    Returns
    -------
    values : tuple
        Native Python floats (cgf, first_derivative, second_derivative), the
        value of the cumulant generating function at tilt and its first two
        derivatives with respect to tilt.
    """
    return values  # placeholder
```

### Step 7

07_solve_tilting_parameters

Goal
----
Find, for every subset and each sign of the threshold, the exponential tilt that moves the mean of the subset statistic onto that threshold.

```python
import numpy as np

def solve_tilting_parameters(weights: np.ndarray, maf: float, threshold: float) -> np.ndarray:
    """Return the two exponential tilts of every subset statistic.

    Column 0 holds the tilt at which the tilted mean of the subset statistic
    equals plus threshold and column 1 the tilt at which it equals minus
    threshold. Each root is found by a bracketed Newton iteration started from
    the bracket obtained by doubling the interval [-1, 1] outward, and is
    accepted once the tilted mean is within 1e-12 of its target or the bracket
    has collapsed to a relative width of 1e-14.

    Parameters
    ----------
    weights : np.ndarray
        Table of per-subject coefficients of shape (n_subsets, n_subjects).
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.
    threshold : float
        Observed value of the all-subset maximum, threshold > 0.

    Returns
    -------
    tilts : np.ndarray
        Array of shape (n_subsets, 2) holding the positive-branch and
        negative-branch tilts of every subset.
    """
    return tilts  # placeholder
```

### Step 8

08_sample_tilted_genotypes

Goal
----
Draw one genotype vector from the exponentially tilted Hardy-Weinberg law attached to a subset and a tilt, using a supplied vector of uniforms.

```python
import numpy as np

def sample_tilted_genotypes(weight_row: np.ndarray, maf: float, tilt: float,
                            uniforms: np.ndarray) -> np.ndarray:
    """Map a vector of uniforms to genotypes under the exponentially tilted law.

    Subject i receives genotype k with probability proportional to
    exp(tilt * weight_row[i] * (k - 2 * maf)) times the Hardy-Weinberg
    probability of k. The genotype returned is the smallest value whose
    cumulative tilted probability strictly exceeds uniforms[i].

    Parameters
    ----------
    weight_row : np.ndarray
        Per-subject coefficients of one subset, of shape (n_subjects,).
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.
    tilt : float
        Exponential tilt applied to the subset statistic.
    uniforms : np.ndarray
        Uniform variates of shape (n_subjects,) with entries in [0, 1).

    Returns
    -------
    genotypes : np.ndarray
        Array of shape (n_subjects,) holding the values 0.0, 1.0 or 2.0.
    """
    return genotypes  # placeholder
```

### Step 9

09_compute_is_weight

Goal
----
Evaluate the likelihood-ratio contribution of one simulated genotype vector to the estimated tail probability.

```python
import numpy as np

def compute_is_weight(weights: np.ndarray, tilts: np.ndarray, cgf_values: np.ndarray,
                      genotypes: np.ndarray, maf: float, threshold: float) -> float:
    """Return the importance sampling contribution of one genotype vector.

    Parameters
    ----------
    weights : np.ndarray
        Table of per-subject coefficients of shape (n_subsets, n_subjects).
    tilts : np.ndarray
        Array of shape (n_subsets, 2) holding the positive-branch and
        negative-branch tilts of every subset.
    cgf_values : np.ndarray
        Array of shape (n_subsets, 2) holding the cumulant generating function
        of every subset evaluated at the matching tilt.
    genotypes : np.ndarray
        Simulated genotypes of shape (n_subjects,), each equal to 0, 1 or 2.
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.
    threshold : float
        Observed value of the all-subset maximum, threshold > 0.

    Returns
    -------
    contribution : float
        Contribution of this replicate as a native Python float; zero when the
        all-subset maximum does not exceed the threshold.
    """
    return contribution  # placeholder
```

### Step 10

10_run_importance_sampling

Goal
----
Run the mixture importance sampling loop over replicates, drawing each replicate with sub-problem 08 and weighting it with sub-problem 09, and return the estimated tail probability with its standard error.

```python
import numpy as np 

def run_importance_sampling(weights: np.ndarray, tilts: np.ndarray, cgf_values: np.ndarray,
                            maf: float, threshold: float, n_sims: int, seed: int) -> tuple:
    """Estimate the all-subset tail probability by mixture importance sampling.

    Randomness comes from numpy.random.default_rng(seed). Each replicate draws,
    in this order, the driving subset index with integers(n_subsets), then a
    single random() whose value below one half selects the negative branch, then
    a vector of n_subjects uniforms with random(n_subjects). Genotypes follow
    the exponentially tilted Hardy-Weinberg law of the driving subset at the
    tilt of the selected branch, taken by the inverse cumulative rule.

    Parameters
    ----------
    weights : np.ndarray
        Table of per-subject coefficients of shape (n_subsets, n_subjects).
    tilts : np.ndarray
        Array of shape (n_subsets, 2) holding the positive-branch and
        negative-branch tilts of every subset.
    cgf_values : np.ndarray
        Array of shape (n_subsets, 2) holding the cumulant generating function
        of every subset evaluated at the matching tilt.
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.
    threshold : float
        Observed value of the all-subset maximum, threshold > 0.
    n_sims : int
        Number of replicates, n_sims >= 2.
    seed : int
        Seed of the random generator.

    Returns
    -------
    estimate : tuple
        Native Python floats (pvalue, standard_error), the estimated tail
        probability and the standard error of that estimate.
    """
    return estimate  # placeholder
```

### Step 11

11_run_asset_discrepancy_pipeline

Goal
----
Chain the sub-problem functions 01-10 end to end on the single-cell eQTL testbed and return the base-ten logarithm of the ratio of the analytic significance to the genotype-conditional estimate.




Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (build_expression_panel, compute_subset_weights, compute_subset_correlation, build_neighbour_table, compute_dlm_pvalue, compute_subset_cgf, solve_tilting_parameters, sample_tilted_genotypes, compute_is_weight, run_importance_sampling) rather than reimplementing them.

```python
import numpy as np 

def run_asset_discrepancy_pipeline(n_subjects: int = 120, n_cell_types: int = 7,
                                   zero_fraction: float = 0.6,
                                   factor_correlation: float = 0.4,
                                   panel_seed: int = 20260824, maf: float = 0.02,
                                   threshold: float = 7.0, n_sims: int = 50000,
                                   sampling_seed: int = 2026) -> float:
    """Run the whole comparison on the single-cell eQTL testbed.

    Parameters
    ----------
    n_subjects : int
        Number of subjects, n_subjects >= 2.
    n_cell_types : int
        Number of cell types, 1 <= n_cell_types <= 14.
    zero_fraction : float
        Expected fraction of censored expression entries, 0 < zero_fraction < 1.
    factor_correlation : float
        Fraction of latent expression variance carried by the shared subject
        factor, 0 <= factor_correlation < 1.
    panel_seed : int
        Seed of the generator that builds the expression panel.
    maf : float
        Minor allele frequency, 0 < maf <= 0.5.
    threshold : float
        Observed value of the all-subset maximum, threshold > 0.
    n_sims : int
        Number of importance sampling replicates, n_sims >= 2.
    sampling_seed : int
        Seed of the generator that drives the importance sampling loop.

    Returns
    -------
    discrepancy : float
        Base-ten logarithm of the analytic significance divided by the
        genotype-conditional estimate, as a native Python float.
    """
    return discrepancy  # placeholder
```
