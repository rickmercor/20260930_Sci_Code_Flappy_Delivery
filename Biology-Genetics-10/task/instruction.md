# Biology-Genetics-10

## Background

Regulatory sequence models consume long genomic windows rather than isolated variants. At a GWAS locus, linked alleles therefore form part of the perturbation background, and a lead-only edit can combine alleles in a way that is rare or absent among phased chromosomes.
Population-informed sequence construction treats phased partner alleles as correlated binary outcomes. A low-rank latent Gaussian model preserves their marginal allele frequencies while representing shared linkage structure through a small number of continuous factors.
Conditioning this model on either lead allele yields a distribution over complete partner configurations. Ranked configurations can support a single representative background or an uncertainty-aware summary, while comparison with observed haplotypes and residual correlations assesses whether the low-rank approximation is adequate.

## Problem

Sequence-to-function models evaluate complete genomic windows, so changing a lead GWAS allele while leaving linked partners on the reference background can create a haplotype that is implausible in the population. A population-informed perturbation instead conditions the partner-allele configuration on the chosen lead state and retains uncertainty across several high-probability configurations.

Apply the paper-grounded deterministic one-factor conditional-haplotype construction to the phased-panel summaries. The supplied `working_loadings` are the fitted working-scale coefficients required by that construction, and its conditional haplotype probabilities feed a benchmark-specific top-`L` probability-weighted mean partner ALT-allele burden for each lead state.

Solve one deterministic instance with the following configuration:

- `alt_counts = [6, 9, 29, 12, 31, 4, 23, 15]`, ordered as the lead followed by seven partners
- `n_haplotypes = 40`
- `working_loadings = [2.2, 1.7, -1.9, 1.25, -1.55, 0.85, -1.15, 1.45]`, in the same order
- `psi_min = 0.15`
- `quadrature_order = 64`, a benchmark integration control
- `factor_bound = 8.0`, giving the benchmark factor interval `[-8, 8]`
- `top_l = 4`
- exact probability ties are broken by ascending lexicographic order of the binary partner configuration

Use the construction's source-based model conventions together with the listed benchmark controls, retaining full precision throughout. Your final answer must be a single number: the benchmark-specific top-`L` mean partner ALT-allele burden for lead state 1 minus that for lead state 0.

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

01_compute_fixed_margins

Goal
----
Fixed allele-frequency margins for a binary haplotype panel.

For n phased haplotypes and ALT count m_j at variant j, the fixed-margin

HaploPerturb model uses the Jeffreys-corrected frequency

p_j = (m_j + 1/2) / (n + 1). The latent Gaussian threshold is

tau_j = Phi^{-1}(1 - p_j), so thresholding a standard normal above tau_j

reproduces p_j while keeping thresholds finite even for counts of zero or n.

Inputs

------

alt_counts : one-dimensional ALT counts ordered as lead then partners

n_haplotypes : number of phased haplotypes in the panel

Returns

-------

corrected_frequencies : corrected ALT frequencies

thresholds : standard-normal thresholds

```python
import numpy as np


def compute_fixed_margins(
    alt_counts: np.ndarray, n_haplotypes: int
) -> tuple[np.ndarray, np.ndarray]:
    '''Compute fixed ALT-frequency margins and latent Gaussian thresholds.

    Parameters
    ----------
    alt_counts : np.ndarray
        One-dimensional integer ALT counts, ordered as lead then partners.
    n_haplotypes : int
        Positive number of phased haplotypes used to form the counts.

    Returns
    -------
    corrected_frequencies : np.ndarray
        Jeffreys-corrected ALT frequencies as a float64 vector.
    thresholds : np.ndarray
        Standard-normal thresholds as a float64 vector.

    Raises
    ------
    ValueError
        If `alt_counts` is not a finite one-dimensional integer vector with at
        least two entries, if `n_haplotypes` is not a positive integer, or if
        any count lies outside the inclusive interval from zero to
        `n_haplotypes`.
    '''
    return result  # noqa: F821
```

### Step 2

02_map_factor_parameters

Goal
----
Map working-scale probit coefficients to a unit-variance factor model.

For working coefficient a_j and fixed threshold tau_j, the one-factor model

uses psi_j = 1 / (1 + a_j^2), b_j = a_j sqrt(psi_j), and

beta_j = -tau_j / sqrt(psi_j). This absorbs b_j^2 + psi_j = 1 exactly.

The factor reflection is fixed by orienting the lead loading b_0 to be positive,

and every uniqueness must remain at or above the prescribed floor psi_min.

Inputs

------

working_loadings : one-dimensional working-scale coefficients

thresholds : fixed latent Gaussian thresholds

psi_min : minimum admissible uniqueness

Returns

-------

loadings : oriented latent factor loadings

uniqueness : residual variances

intercepts : working-scale probit intercepts

```python
import numpy as np


def map_factor_parameters(
    working_loadings: np.ndarray, thresholds: np.ndarray, psi_min: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Map working coefficients to oriented unit-variance factor parameters.

    Parameters
    ----------
    working_loadings : np.ndarray
        One-dimensional fitted working-scale coefficients, lead first.
    thresholds : np.ndarray
        Fixed Gaussian thresholds in the same order.
    psi_min : float
        Strictly positive lower bound on every uniqueness.

    Returns
    -------
    loadings : np.ndarray
        Oriented one-factor loadings.
    uniqueness : np.ndarray
        Residual variances satisfying loadings squared plus uniqueness equals one.
    intercepts : np.ndarray
        Working-scale probit intercepts.

    Raises
    ------
    ValueError
        If `working_loadings` is not a finite one-dimensional vector with at
        least two entries; if `thresholds` is not finite with the same shape;
        if `psi_min` is not numeric or does not lie in `(0, 1)`; if the lead
        working loading is zero; or if any implied uniqueness is more than
        `1e-12` below `psi_min`.
    '''
    return result  # noqa: F821
```

### Step 3

03_build_conditional_mixture

Goal
----
Discretize the factor distribution conditional on a lead allele.

At factor value f, variant j has ALT probability

q_j(f) = Phi((b_j f - tau_j) / sqrt(psi_j)). Conditioning on lead state s

reweights the standard-normal factor density by q_0(f)^s[1-q_0(f)]^(1-s).

A fixed Gauss-Legendre rule on a symmetric finite interval turns this posterior

factor law into normalized mixture weights and partner Bernoulli probabilities.

Inputs

------

loadings : oriented one-factor loadings, lead first

uniqueness : residual variances

thresholds : fixed Gaussian thresholds

lead_state : conditioned lead allele, zero or one

quadrature_order : number of fixed Gauss-Legendre nodes

factor_bound : positive half-width of the integration interval

Returns

-------

factor_nodes : transformed quadrature nodes

mixture_weights : normalized lead-conditioned factor weights

partner_probabilities : ALT probabilities for every node and partner

```python
import numpy as np


def build_conditional_mixture(
    loadings: np.ndarray,
    uniqueness: np.ndarray,
    thresholds: np.ndarray,
    lead_state: int,
    quadrature_order: int,
    factor_bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Build a fixed-quadrature approximation to the lead-conditioned factor law.

    Parameters
    ----------
    loadings : np.ndarray
        One-dimensional oriented loadings, with the lead first.
    uniqueness : np.ndarray
        Positive residual variances in the same order.
    thresholds : np.ndarray
        Fixed Gaussian thresholds in the same order.
    lead_state : int
        Conditioned lead allele, either zero or one.
    quadrature_order : int
        Number of Gauss-Legendre nodes.
    factor_bound : float
        Positive half-width of the symmetric factor interval.

    Returns
    -------
    factor_nodes : np.ndarray
        Transformed quadrature nodes.
    mixture_weights : np.ndarray
        Normalized posterior factor weights.
    partner_probabilities : np.ndarray
        Partner ALT probabilities with shape (quadrature_order, p - 1).

    Raises
    ------
    ValueError
        If `loadings` is not a finite one-dimensional vector with at least two
        entries; if `uniqueness` or `thresholds` has a different shape; if a
        uniqueness is nonpositive or nonfinite; if a threshold is nonfinite;
        if `lead_state` is not zero or one; if `quadrature_order` is not an
        integer of at least eight; if `factor_bound` is not numeric, positive,
        and finite; or if the lead-conditioned quadrature has no positive
        finite mass.
    '''
    return result  # noqa: F821
```

### Step 4

04_enumerate_factor_modes

Goal
----
Enumerate conditional partner modes along a one-dimensional factor.

At a fixed factor value f, conditional independence makes the modal partner

allele x_j*(f) = 1{b_j f > tau_j}. Every nonzero loading contributes one

breakpoint t_j = tau_j / b_j. Sorting the distinct breakpoints partitions the

factor line into intervals on which the full modal configuration is constant,

so one interior representative per interval enumerates the candidate path.

Inputs

------

partner_loadings : one-factor loadings for partner variants

partner_thresholds : fixed thresholds for the same partners

Returns

-------

mode_candidates : unique binary modal configurations in lexicographic order

```python
import numpy as np


def enumerate_factor_modes(
    partner_loadings: np.ndarray, partner_thresholds: np.ndarray
) -> np.ndarray:
    '''Enumerate every partner configuration that is modal on a factor interval.

    Parameters
    ----------
    partner_loadings : np.ndarray
        One-dimensional partner loadings.
    partner_thresholds : np.ndarray
        One-dimensional partner thresholds in the same order.

    Returns
    -------
    mode_candidates : np.ndarray
        Binary array with one unique modal configuration per row.

    Raises
    ------
    ValueError
        If `partner_loadings` is not a nonempty finite one-dimensional vector,
        or if `partner_thresholds` is not finite with the same shape as
        `partner_loadings`.
    '''
    return mode_candidates  # noqa: F821
```

### Step 5

05_score_conditional_candidates

Goal
----
Evaluate candidate haplotypes under a lead-conditioned factor mixture.

Given normalized quadrature weights w_g and partner ALT probabilities q_j(f_g),

the conditional probability of configuration x is the weighted product-Bernoulli

sum over factor nodes: sum_g w_g product_j q_j(f_g)^x_j

[1-q_j(f_g)]^(1-x_j). This is the fixed-node approximation to the paper's

lead-conditioned latent-factor integral.

Inputs

------

configurations : binary partner configurations

mixture_weights : normalized conditioned factor weights

partner_probabilities : node-specific partner ALT probabilities

Returns

-------

conditional_probabilities : probability of each candidate configuration

```python
import numpy as np


def score_conditional_candidates(
    configurations: np.ndarray,
    mixture_weights: np.ndarray,
    partner_probabilities: np.ndarray,
) -> np.ndarray:
    '''Evaluate candidate probabilities under a conditioned product-Bernoulli mixture.

    Parameters
    ----------
    configurations : np.ndarray
        Binary array with one partner configuration per row.
    mixture_weights : np.ndarray
        One-dimensional normalized quadrature weights.
    partner_probabilities : np.ndarray
        ALT probabilities with one row per quadrature node.

    Returns
    -------
    conditional_probabilities : np.ndarray
        Conditional probability of every configuration.

    Raises
    ------
    ValueError
        If `configurations` is not a nonempty two-dimensional binary array; if
        `mixture_weights` is not a nonempty finite one-dimensional vector, has
        a negative entry, or does not sum to one within absolute tolerance
        `1e-10`; or if `partner_probabilities` does not have shape
        `(len(mixture_weights), configurations.shape[1])`, is nonfinite, or has
        an entry outside `[0, 1]`.
    '''
    return conditional_probabilities  # noqa: F821
```

### Step 6

06_expand_mode_candidates

Goal
----
Expand the deterministic factor-mode set around its leading configuration.

The one-factor breakpoint path contains only configurations that are modal at

some factor value. After these modes are scored for one lead state, the paper's

local repair selects the highest-probability mode and adjoins every configuration

obtained by flipping exactly one partner allele. The union removes duplicates and

is ordered lexicographically before final probability scoring.

Inputs

------

mode_candidates : binary configurations from factor-space enumeration

mode_probabilities : conditioned probabilities for those configurations

Returns

-------

expanded_candidates : mode set plus all one-allele neighbors of its leader

```python
import numpy as np


def expand_mode_candidates(
    mode_candidates: np.ndarray, mode_probabilities: np.ndarray
) -> np.ndarray:
    '''Add every one-allele neighbor of the highest-probability factor mode.

    Parameters
    ----------
    mode_candidates : np.ndarray
        Unique binary factor-mode configurations.
    mode_probabilities : np.ndarray
        Conditional probabilities in corresponding row order.

    Returns
    -------
    expanded_candidates : np.ndarray
        Lexicographically ordered unique binary configurations.

    Raises
    ------
    ValueError
        If `mode_candidates` is not a nonempty two-dimensional binary array, or
        if `mode_probabilities` is not a finite vector with one nonnegative
        entry per candidate row.
    '''
    return expanded_candidates  # noqa: F821
```

### Step 7

07_summarize_top_burden

Goal
----
Summarize the highest-ranked conditional haplotypes as an allele burden.

Candidates are ranked by decreasing conditional probability, with binary

lexicographic order breaking exact ties. For the first L configurations, the

reported burden is the number of partner ALT alleles averaged with probabilities

renormalized within that top-L set. The retained mass is also returned so the

truncation can be inspected.

Inputs

------

configurations : expanded binary partner configurations

conditional_probabilities : probability assigned to each configuration

top_l : number of leading configurations retained

Returns

-------

mean_burden : top-L probability-weighted partner ALT count

top_configurations : ranked leading configurations

top_probabilities : their unnormalized conditional probabilities

top_mass : total conditional mass of the retained configurations

```python
import numpy as np


def summarize_top_burden(
    configurations: np.ndarray, conditional_probabilities: np.ndarray, top_l: int
) -> tuple[float, np.ndarray, np.ndarray, float]:
    '''Rank candidates and compute the top-L probability-weighted ALT burden.

    Parameters
    ----------
    configurations : np.ndarray
        Binary candidate configurations.
    conditional_probabilities : np.ndarray
        Conditional probability of each candidate.
    top_l : int
        Positive number of leading candidates to retain.

    Returns
    -------
    mean_burden : float
        Probability-weighted partner ALT count within the retained set.
    top_configurations : np.ndarray
        Leading configurations in rank order.
    top_probabilities : np.ndarray
        Corresponding unnormalized conditional probabilities.
    top_mass : float
        Sum of the retained probabilities.

    Raises
    ------
    ValueError
        If `configurations` is not a nonempty two-dimensional binary array; if
        `conditional_probabilities` is not a finite nonnegative vector with one
        entry per candidate row; if `top_l` is not an integer between one and
        the number of candidates, inclusive; or if the retained candidates
        have zero total probability.
    '''
    return result  # noqa: F821
```

### Step 8

08_run_haploperturb_benchmark

Goal
----
Orchestrate the fixed-margin conditional haplotype benchmark.

The panel counts determine fixed Gaussian thresholds, working coefficients map

to an oriented unit-variance factor model, and a breakpoint sweep enumerates

partner modes. For each lead state, fixed quadrature builds the conditional

factor mixture, the modes are scored, the leading mode receives its one-allele

neighborhood expansion, and the expanded candidates are rescored and ranked.

The final scalar is the alternate-state top-L mean partner ALT burden minus the

corresponding reference-state burden.

Inputs

------

alt_counts : ALT counts ordered as lead then partners

n_haplotypes : number of phased haplotypes

working_loadings : fitted one-factor working coefficients

psi_min : uniqueness floor

quadrature_order : fixed Gauss-Legendre order

factor_bound : half-width of the factor interval

top_l : number of leading haplotypes retained per lead state

Returns

-------

burden_contrast : alternate-conditioned minus reference-conditioned mean burden

```python
import numpy as np


def run_haploperturb_benchmark(
    alt_counts: np.ndarray,
    n_haplotypes: int,
    working_loadings: np.ndarray,
    psi_min: float,
    quadrature_order: int,
    factor_bound: float,
    top_l: int,
) -> float:
    '''Run the complete conditional haplotype construction and return its burden contrast.

    Parameters
    ----------
    alt_counts : np.ndarray
        ALT counts ordered as lead then partners.
    n_haplotypes : int
        Number of phased haplotypes represented by the counts.
    working_loadings : np.ndarray
        Fitted one-factor working coefficients in the same order.
    psi_min : float
        Minimum admissible residual uniqueness.
    quadrature_order : int
        Number of fixed Gauss-Legendre nodes.
    factor_bound : float
        Positive half-width of the symmetric factor interval.
    top_l : int
        Number of leading configurations retained for each lead state.

    Returns
    -------
    burden_contrast : float
        Alternate-conditioned minus reference-conditioned top-L mean ALT burden.

    Raises
    ------
    ValueError
        If `alt_counts` is not a finite one-dimensional integer vector with at
        least two entries; if `n_haplotypes` is not a positive integer; if any
        count lies outside `[0, n_haplotypes]`; if `working_loadings` is not a
        same-length finite vector with a nonzero lead entry; if `psi_min` is not
        numeric in `(0, 1)` or an implied uniqueness is more than `1e-12` below
        it; if `quadrature_order` is not an integer of at least eight; if
        `factor_bound` is not numeric, positive, and finite; if `top_l` is not
        an integer between one and the expanded candidate count, inclusive; or
        if either lead-conditioned quadrature has no positive finite mass.
    '''
    return burden_contrast  # noqa: F821
```
