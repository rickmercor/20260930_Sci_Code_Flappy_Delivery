# Biology-Genetics-2

## Background

Gene flow between early modern humans and archaic hominins is currently read from a handful of sequenced archaic genomes, all of them Eurasian, which leaves admixture in Africa and at deeper timescales largely out of reach. Methods for detecting it need either an archaic reference sequence or a population assumed to carry none of the ancestry being searched for, and both requirements fail when the candidate source has never been sequenced.

Reconstructed genealogies offer a way past that, because ancestry from a long-separated population distorts the shape of local trees in ways that present-day samples alone make visible. Work in this area builds probabilistic models over the ordered sequence of local genealogies along a chromosome and assigns an ancestry state position by position.

Such methods are assessed by the precision and recall of the tracts they call against simulated truth, by their false discovery rate on data simulated without any admixture, and by whether the timing they recover agrees with the simulated history.

## Problem

Ancestral recombination graphs reconstructed from present-day genomes record the full sequence of coalescence and recombination events behind a sample, and a recent line of work exploits that record to detect gene flow from deeply divergent hominins without any sequenced archaic genome and without an outgroup population assumed to be free of archaic ancestry. The input is the ordered sequence of marginal trees along a chromosome, each with the physical and genetic span it covers, and the output is a set of called archaic tracts for one focal haplotype together with an estimate of when the gene flow happened.

Gene flow from a population that separated long ago places the focal lineage where it cannot coalesce with the recipient gene pool for a long stretch of the past, and over that same stretch the other lineages of the local tree go on coalescing at the ordinary rate, so each marginal tree is reduced to a single positive scalar built from what the focal lineage and the rest of the tree are each doing around a user-chosen time cutoff. Because introgressed material survives as contiguous haplotypes rather than isolated positions, the sequence of per-tree scalars is treated as the emission sequence of a two-state hidden Markov chain, modern human and archaic, with gamma emission densities written in the shape and rate parameterisation and a transition matrix carrying one entry into the archaic state and one out of it. No labelled data is available to anchor the two emissions, so the genome-wide distribution of the scalar is used directly after a one-sided generalized extreme Studentized deviate screen separates its upper tail from its bulk, and the model parameters are then refined by Baum-Welch before the chain is decoded with the forward-backward algorithm.

Tracts are then called from the posterior and filtered on length in both physical and recombination units, and the endpoints of the focal lineage branches inside the surviving tracts carry the information about the timing of the pulse.

Your task is to solve one concrete deterministic example of this pipeline. Use the following configuration:

- t_archaic = 15000.0 generations, the time cutoff defining an archaic event
- x_floor = 1e-10, a positive floor added to every per-tree scalar
- esd_alpha = 0.05 and esd_max_outlier_fraction = 0.2 for the outlier screen, which may therefore remove at most 9 of the 48 trees
- gamma shapes are fitted by maximum likelihood, solving the shape equation by bisection on the bracket [1e-6, 1e6] with 200 halvings
- p_init = 0.01, q_init = 0.1, pi_archaic = 0.05, the last held fixed throughout
- max_iter = 200 and loglik_tol = 1e-2, stopping once the log-likelihood of the observation sequence changes by less than the tolerance, with the reported parameters being those in force when the test fires rather than the update from that same iteration
- post_threshold = 0.9, min_bp = 50000.0, min_cm = 0.05
- the final average over marginal trees is weighted by their physical spans
- coal_times, the coalescence times in generations of each of the 48 marginal trees, ascending along each row:

```
coal_times = [
  [1300, 10930, 11700, 16900, 33800],
  [11320, 11500, 17880, 20960, 21250],
  [2000, 13400, 17340, 21860, 24400],
  [2200, 2970, 10500, 15320, 25700],
  [1300, 3730, 9500, 18600, 23300],
  [4530, 13000, 14510, 15680, 18650],
  [10760, 11130, 11700, 16230, 35200],
  [1500, 6610, 12400, 26630, 31600],
  [3100, 6230, 13400, 14460, 27500],
  [1900, 11450, 12200, 16430, 18200],
  [4700, 9200, 13200, 20850, 21700],
  [5210, 12400, 14510, 14810, 16450],
  [320, 3540, 9600, 10340, 15800],
  [2550, 10800, 13360, 20530, 21050],
  [2183, 4400, 7570, 32420, 48283],
  [1846, 6990, 20090, 32450, 51346],
  [2027, 4840, 14080, 36090, 46827],
  [2314, 17550, 19480, 45670, 49914],
  [870, 2461, 3740, 15900, 25461],
  [1800, 10100, 11540, 16160, 19250],
  [2730, 9550, 11020, 14000, 15300],
  [600, 7540, 11100, 13380, 23000],
  [2300, 12700, 15300, 17750, 21850],
  [700, 5700, 11100, 13150, 19300],
  [900, 1490, 12900, 18160, 27200],
  [1600, 9200, 15560, 19770, 20300],
  [1600, 6730, 9100, 11340, 26800],
  [2210, 2560, 11300, 20060, 35300],
  [7300, 20920, 28530, 30470, 37800],
  [2500, 11630, 12400, 12890, 23300],
  [4460, 5200, 7070, 18080, 28200],
  [1000, 4720, 13200, 15340, 25400],
  [2400, 13000, 15740, 17300, 22050],
  [2650, 10920, 35000, 35050, 47850],
  [9800, 10100, 11190, 16240, 17450],
  [1000, 8240, 12600, 17260, 18500],
  [2000, 11300, 11930, 15680, 17750],
  [1800, 4520, 10100, 19090, 32200],
  [6500, 11700, 13100, 19460, 19750],
  [5260, 6690, 9000, 14780, 21500],
  [5830, 11500, 11830, 16360, 19150],
  [2300, 3940, 11000, 17620, 22400],
  [4850, 13200, 14720, 18080, 18250],
  [8640, 9540, 13200, 24650, 28100],
  [800, 7050, 9300, 14420, 15300],
  [7450, 8690, 11300, 17850, 19900],
  [500, 2940, 11600, 18680, 28900],
  [2910, 11640, 13300, 24550, 26000],
]
```

- focal_mask, holding 1 where the coalescence lies on the focal haplotype's path to the root of that tree and 0 otherwise:

```
focal_mask = [
  [1, 0, 1, 0, 1],
  [0, 1, 0, 0, 1],
  [1, 1, 0, 0, 1],
  [1, 0, 1, 0, 1],
  [1, 0, 1, 0, 1],
  [0, 1, 0, 0, 1],
  [0, 0, 1, 0, 1],
  [1, 0, 1, 0, 1],
  [0, 0, 1, 0, 1],
  [1, 0, 1, 0, 1],
  [0, 0, 1, 0, 1],
  [0, 1, 0, 0, 1],
  [0, 0, 1, 0, 1],
  [0, 1, 0, 0, 1],
  [1, 0, 0, 0, 1],
  [1, 0, 0, 0, 1],
  [1, 0, 0, 0, 1],
  [1, 0, 0, 0, 1],
  [0, 1, 0, 0, 1],
  [1, 1, 0, 0, 1],
  [0, 1, 0, 0, 1],
  [1, 0, 1, 0, 1],
  [1, 1, 0, 0, 1],
  [1, 0, 1, 0, 1],
  [1, 0, 1, 0, 1],
  [1, 1, 0, 0, 1],
  [1, 0, 1, 0, 1],
  [0, 0, 1, 0, 1],
  [0, 1, 0, 0, 1],
  [1, 0, 1, 0, 1],
  [0, 1, 0, 0, 1],
  [1, 0, 1, 0, 1],
  [1, 1, 0, 0, 1],
  [1, 0, 0, 0, 1],
  [0, 1, 0, 0, 1],
  [1, 0, 1, 0, 1],
  [1, 1, 0, 0, 1],
  [1, 0, 1, 0, 1],
  [0, 1, 0, 0, 1],
  [0, 0, 1, 0, 1],
  [0, 1, 0, 0, 1],
  [1, 0, 1, 0, 1],
  [0, 1, 0, 0, 1],
  [0, 0, 1, 0, 1],
  [1, 0, 1, 0, 1],
  [0, 0, 1, 0, 1],
  [1, 0, 1, 0, 1],
  [0, 0, 1, 0, 1],
]
```

- span_bp, the physical span of each marginal tree in base pairs:

```
span_bp = [
  15800, 12500, 14300, 13000, 10200, 12600, 15600, 15900, 15800, 13300, 11200, 13000,
  13600, 11000, 13000, 12000, 13700, 13000, 12000, 13300, 11400, 15600, 11100, 10700,
  13200, 11100, 11400, 11700, 13800, 10900, 58000, 11400, 14400, 62000, 13900, 11600,
  13700, 14500, 14400, 9200, 9600, 11100, 11500, 13900, 9600, 12200, 14600, 15300,
]
```

- span_cm, the genetic span of each marginal tree in centimorgans:

```
span_cm = [
  0.0160, 0.0141, 0.0157, 0.0144, 0.0117, 0.0133, 0.0135, 0.0126, 0.0142, 0.0107, 0.0092, 0.0119,
  0.0150, 0.0105, 0.0130, 0.0120, 0.0147, 0.0130, 0.0120, 0.0145, 0.0138, 0.0175, 0.0102, 0.0139,
  0.0144, 0.0131, 0.0131, 0.0142, 0.0166, 0.0109, 0.0610, 0.0119, 0.0185, 0.0280, 0.0169, 0.0133,
  0.0161, 0.0128, 0.0115, 0.0084, 0.0108, 0.0109, 0.0104, 0.0137, 0.0114, 0.0110, 0.0157, 0.0152,
]
```

Report the range of the per-tree scalars, the trees flagged by the outlier screen, the two fitted gamma emissions, the fitted transition probabilities, the trees whose archaic posterior exceeds the threshold, and the archaic tracts that survive both length filters. Your final answer must be a single number: the estimated lower bound on the admixture time, in generations, implied by the surviving archaic tracts.

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

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_focal_branch_intervals

Goal
----
Locate, in every marginal tree of an ancestral recombination graph, the branch on

the focal haplotype's ancestral lineage whose time span contains the archaic time

cutoff, and return the lower and upper endpoint of that branch for each tree.

```python
def focal_branch_intervals(
    coal_times: "np.ndarray",
    focal_mask: "np.ndarray",
    t_archaic: float = 15000.0,
) -> "np.ndarray":
    """Return the focal branch endpoints spanning t_archaic in each marginal tree.

    Parameters
    ----------
    coal_times : np.ndarray
        Coalescence times of shape (m, c), ascending along each row.
    focal_mask : np.ndarray
        Binary indicator of shape (m, c) marking the coalescences that lie on the
        focal haplotype's path to the root.
    t_archaic : float
        Time cutoff in generations defining an archaic event.

    Returns
    -------
    intervals : np.ndarray
        Array of shape (m, 2) holding the lower and upper endpoint of the focal
        branch that spans t_archaic in each tree.

    Raises
    ------
    ValueError
        If coal_times is not two dimensional, if focal_mask has a different shape
        or holds entries outside {0, 1}, if a tree has no focal coalescence, if
        any coalescence time is not strictly positive, if the times do not
        increase strictly along a row, or if t_archaic is not positive and finite.
    """
    return intervals
```

### Step 2

02_tree_observation_statistic

Goal
----
Count, in each marginal tree, the coalescence events that fall strictly inside

the time span of the focal branch, and combine that count with the branch length

into the per-tree observation that the archaic ancestry model is fitted to.

```python
def tree_observation_statistic(
    coal_times: "np.ndarray",
    intervals: "np.ndarray",
    x_floor: float = 1e-10,
) -> tuple:
    """Return the interior coalescence counts and the per-tree observation.

    Parameters
    ----------
    coal_times : np.ndarray
        Coalescence times of shape (m, c), ascending along each row.
    intervals : np.ndarray
        Focal branch endpoints of shape (m, 2).
    x_floor : float
        Positive floor added to every observation.

    Returns
    -------
    result : tuple
        The interior coalescence counts of shape (m,) followed by the per-tree
        observations of shape (m,).

    Raises
    ------
    ValueError
        If coal_times is not two dimensional, if intervals does not have shape
        (m, 2) matching coal_times, if any upper endpoint lies below its lower
        endpoint, or if x_floor is not strictly positive.
    """
    return result
```

### Step 3

03_esd_outlier_flags

Goal
----
Flag the upper-tail outliers of the genome-wide observation distribution with a

one-sided generalized extreme Studentized deviate test, returning a binary

indicator per marginal tree.

```python
def esd_outlier_flags(
    observations: "np.ndarray",
    alpha: float = 0.05,
    max_outlier_fraction: float = 0.2,
) -> "np.ndarray":
    """Return a binary flag per observation marking the upper-tail outliers.

    Parameters
    ----------
    observations : np.ndarray
        Per-tree observations of shape (m,).
    alpha : float
        Significance level of the test.
    max_outlier_fraction : float
        Largest fraction of the sample that may be removed.

    Returns
    -------
    flags : np.ndarray
        Array of shape (m,) holding 1 for a flagged outlier and 0 otherwise.

    Raises
    ------
    ValueError
        If observations is not one dimensional, if it holds fewer than four
        points, if alpha does not lie strictly between 0 and 1, or if
        max_outlier_fraction does not lie strictly between 0 and 1.
    """
    return flags
```

### Step 4

04_weighted_gamma_mle

Goal
----
Fit the shape and rate of a gamma density to a weighted sample by maximum

likelihood, solving the shape equation by bisection on a fixed bracket.

```python
def weighted_gamma_mle(
    observations: np.ndarray,
    weights: np.ndarray,
    bracket: tuple = (1e-6, 1e6),
    n_bisect: int = 200,
) -> tuple:
    """Return the maximum-likelihood gamma shape and rate of a weighted sample.

    Parameters
    ----------
    observations : np.ndarray
        Strictly positive observations of shape (m,).
    weights : np.ndarray
        Non-negative membership weights of shape (m,).
    bracket : tuple
        Lower and upper bound of the bisection bracket for the shape.
    n_bisect : int
        Number of bisection halvings.

    Returns
    -------
    result : tuple
        The maximum-likelihood shape followed by the maximum-likelihood rate.

    Raises
    ------
    ValueError
        If observations or weights are not one dimensional or have different
        lengths, if any observation is not strictly positive, if any weight is
        negative, if the weights sum to zero, if n_bisect is not positive, if the
        bracket is not an increasing pair of positive numbers, or if the weighted
        sample has no dispersion in the log domain.
    """
    return result
```

### Step 5

05_gamma_emission_logpdf

Goal
----
Evaluate the log emission density of both hidden states at every marginal tree,

returning a two-row matrix of log densities.

```python
def gamma_emission_logpdf(
    observations: np.ndarray,
    shape_null: float,
    rate_null: float,
    shape_archaic: float,
    rate_archaic: float,
) -> np.ndarray:
    """Return the two-state log emission matrix for the observations.

    Parameters
    ----------
    observations : np.ndarray
        Strictly positive observations of shape (m,).
    shape_null : float
        Gamma shape of the modern human state.
    rate_null : float
        Gamma rate of the modern human state.
    shape_archaic : float
        Gamma shape of the archaic state.
    rate_archaic : float
        Gamma rate of the archaic state.

    Returns
    -------
    log_emissions : np.ndarray
        Array of shape (2, m) holding the log emission density of each state.

    Raises
    ------
    ValueError
        If observations is not one dimensional, if any observation is not
        strictly positive, or if any shape or rate is not strictly positive.
    """
    return log_emissions
```

### Step 6

06_forward_backward_posteriors

Goal
----
Run the forward and backward recursions of the two-state chain in the log domain

and return the posterior state probabilities at every marginal tree together with

the log-likelihood of the observation sequence.

```python
def forward_backward_posteriors(
    log_emissions: np.ndarray,
    p: float,
    q: float,
    pi_archaic: float = 0.05,
) -> tuple:
    """Return posterior state probabilities and the sequence log-likelihood.

    Parameters
    ----------
    log_emissions : np.ndarray
        Log emission densities of shape (2, m).
    p : float
        Transition probability into the archaic state.
    q : float
        Transition probability out of the archaic state.
    pi_archaic : float
        Prior probability of the archaic state at the first tree.

    Returns
    -------
    result : tuple
        The posterior state probabilities of shape (2, m) followed by the
        log-likelihood of the observation sequence.

    Raises
    ------
    ValueError
        If log_emissions does not have shape (2, m) with m at least two, or if p,
        q or pi_archaic does not lie strictly between 0 and 1.
    """
    return result
```

### Step 7

07_baum_welch_parameters

Goal
----
Re-estimate the archaic emission parameters and the two transition probabilities

by expectation maximisation, holding the modern human emission and the initial

state distribution fixed, and return the fitted parameters with the log-likelihood

reached.

```python
def baum_welch_parameters(
    observations: np.ndarray,
    shape_null: float,
    rate_null: float,
    shape_archaic: float,
    rate_archaic: float,
    p: float = 0.01,
    q: float = 0.1,
    pi_archaic: float = 0.05,
    max_iter: int = 200,
    loglik_tol: float = 1e-2,
) -> dict:
    """Return the expectation maximisation fit of the free model parameters.

    Parameters
    ----------
    observations : np.ndarray
        Strictly positive observations of shape (m,).
    shape_null : float
        Fixed gamma shape of the modern human state.
    rate_null : float
        Fixed gamma rate of the modern human state.
    shape_archaic : float
        Starting gamma shape of the archaic state.
    rate_archaic : float
        Starting gamma rate of the archaic state.
    p : float
        Starting transition probability into the archaic state.
    q : float
        Starting transition probability out of the archaic state.
    pi_archaic : float
        Fixed prior probability of the archaic state at the first tree.
    max_iter : int
        Largest number of expectation maximisation iterations.
    loglik_tol : float
        Convergence tolerance on the log-likelihood.

    Returns
    -------
    fitted : dict
        Dictionary holding shape_archaic, rate_archaic, p, q, n_iter and loglik.
        ``n_iter`` is the index, counting from zero, of the iteration the loop
        stopped on: the number of maximisation steps already applied when the
        convergence test fires, or ``max_iter - 1`` when the iteration cap binds.

    Raises
    ------
    ValueError
        If observations is not one dimensional or holds fewer than two points, if
        any observation is not strictly positive, if any gamma parameter is not
        strictly positive, if p, q or pi_archaic does not lie strictly between 0
        and 1, if max_iter is not positive, or if loglik_tol is not positive.
    """
    return fitted
```

### Step 8

08_archaic_segments

Goal
----
Turn the per-tree posterior probability of the archaic state into called ancestry

tracts by taking maximal runs of consecutive trees above a posterior threshold and

keeping only those runs that are long enough in both physical and genetic units.

```python
def archaic_segments(
    posterior_archaic: np.ndarray,
    span_bp: np.ndarray,
    span_cm: np.ndarray,
    post_threshold: float = 0.9,
    min_bp: float = 5e4,
    min_cm: float = 0.05,
) -> list:
    """Return the retained archaic tracts as inclusive tree index ranges.

    Parameters
    ----------
    posterior_archaic : np.ndarray
        Posterior probability of the archaic state, of shape (m,).
    span_bp : np.ndarray
        Physical span of each marginal tree in base pairs.
    span_cm : np.ndarray
        Genetic span of each marginal tree in centimorgans.
    post_threshold : float
        Posterior probability a tree must exceed to join a candidate tract.
    min_bp : float
        Smallest admissible physical length of a retained tract.
    min_cm : float
        Smallest admissible genetic length of a retained tract.

    Returns
    -------
    segments : list
        List of (start, end) inclusive tree index pairs of the retained tracts.

    Raises
    ------
    ValueError
        If the three arrays are not one dimensional of equal length, if any span
        is negative, if post_threshold does not lie strictly between 0 and 1, or
        if min_bp or min_cm is negative.
    """
    return segments
```

### Step 9

09_admixture_time_estimate

Goal
----
Combine the focal branch endpoints of the marginal trees that fall inside the

retained archaic tracts into a single estimate of the admixture time, weighting

each tree by the physical span it covers.

```python
def admixture_time_estimate(
    segments: list,
    intervals: "np.ndarray",
    span_bp: "np.ndarray",
) -> float:
    """Return the span-weighted admixture time implied by the retained tracts.

    Parameters
    ----------
    segments : list
        List of (start, end) inclusive tree index pairs of the retained tracts.
    intervals : np.ndarray
        Focal branch endpoints of shape (m, 2).
    span_bp : np.ndarray
        Physical span of each marginal tree in base pairs.

    Returns
    -------
    t_admix : float
        Span-weighted admixture time in generations.

    Raises
    ------
    ValueError
        If segments is empty, if intervals does not have shape (m, 2) matching
        span_bp, if a segment index falls outside the range of trees, if a segment
        end precedes its start, or if the covered trees carry no positive span.
    """
    return t_admix
```

### Step 10

10_run_full_pipeline

Goal
----
Run the whole archaic ancestry pipeline on one ancestral recombination graph and

return the resulting estimate of the admixture time in generations.

```python
def run_full_pipeline(
    coal_times: "np.ndarray",
    focal_mask: "np.ndarray",
    span_bp: "np.ndarray",
    span_cm: "np.ndarray",
    t_archaic: float = 15000.0,
    x_floor: float = 1e-10,
    esd_alpha: float = 0.05,
    esd_max_outlier_fraction: float = 0.2,
    p_init: float = 0.01,
    q_init: float = 0.1,
    pi_archaic: float = 0.05,
    max_iter: int = 200,
    loglik_tol: float = 1e-2,
    post_threshold: float = 0.9,
    min_bp: float = 5e4,
    min_cm: float = 0.05,
) -> float:
    """Return the admixture time estimate for one ancestral recombination graph.

    Parameters
    ----------
    coal_times : np.ndarray
        Coalescence times of shape (m, c), ascending along each row.
    focal_mask : np.ndarray
        Binary focal lineage indicator of shape (m, c).
    span_bp : np.ndarray
        Physical span of each marginal tree in base pairs.
    span_cm : np.ndarray
        Genetic span of each marginal tree in centimorgans.
    t_archaic : float
        Time cutoff in generations defining an archaic event.
    x_floor : float
        Positive floor added to every observation.
    esd_alpha : float
        Significance level of the outlier test.
    esd_max_outlier_fraction : float
        Largest fraction of trees the outlier test may remove.
    p_init : float
        Starting transition probability into the archaic state.
    q_init : float
        Starting transition probability out of the archaic state.
    pi_archaic : float
        Fixed prior probability of the archaic state at the first tree.
    max_iter : int
        Largest number of expectation maximisation iterations.
    loglik_tol : float
        Convergence tolerance on the log-likelihood.
    post_threshold : float
        Posterior probability a tree must exceed to join a candidate tract.
    min_bp : float
        Smallest admissible physical length of a retained tract.
    min_cm : float
        Smallest admissible genetic length of a retained tract.

    Returns
    -------
    t_admix : float
        Span-weighted admixture time estimate in generations.

    Raises
    ------
    ValueError
        If span_bp or span_cm is not one dimensional of length m matching
        coal_times, or if any stage of the pipeline rejects its own inputs, which
        includes an outlier test that flags no tree and a decoding that retains no
        tract.
    """
    return t_admix
```
