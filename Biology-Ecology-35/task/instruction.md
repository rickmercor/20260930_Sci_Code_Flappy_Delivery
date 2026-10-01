# Biology-Ecology-35

## Background

Effective population size measures how fast a population loses genetic diversity to drift, and conservation policy now uses it directly: the headline genetic indicator of the global biodiversity framework counts the populations whose effective size exceeds 500. Genetic monitoring often estimates the contemporary effective size from a single sample, through the linkage disequilibrium that drift creates between unlinked loci.

Sequencing now yields thousands of SNPs even for non-model species, but when they sit on only a few chromosomes the many pairs of loci behind such an estimate share loci and linkage, so the estimate carries far less information than the number of pairs suggests. Deciding how many individuals a monitoring survey must sample then depends on confidence intervals that allow for this pseudo-replication. A survey is commissioned before its data exist, so what matters is not only where the interval is expected to fall but how often a survey of that size would in fact place its lower limit above the threshold.

## Problem

The contemporary effective size of a population can be estimated from linkage disequilibrium between unlinked loci: the mean squared allelic correlation r² over pairs of SNPs on different chromosomes, minus the part expected from sampling alone, is converted to Ne. With genome-wide SNPs on few chromosomes the r² values of those pairs are not independent, because pairs share SNPs and SNPs on the same chromosome are in drift-induced linkage disequilibrium, so an interval that treats them as independent is far too narrow. Use the correction proposed recently for this pseudo-replication, which expresses the correlations among the unlinked r² values through the drift-induced r² among SNPs on the same chromosome, summarises them in a pseudo-replication parameter ρ, and builds from ρ a confidence interval for the expected mean r².

An isolated population of a threatened insect with three chromosome pairs is assessed against the criterion that its Ne exceeds 500. A pilot sample of 45 diploid individuals was genotyped at 3,076, 2,273 and 1,955 pruned SNPs on chromosomes 1, 2 and 3. The mean r² over all pairs of SNPs on different chromosomes was 0.0242, and the mean raw r² over all pairs of SNPs on the same chromosome was 0.0357, 0.0352 and 0.0345 on chromosomes 1, 2 and 3; each raw value contains a sampling component with the same expectation as for a pair of unlinked SNPs.

For a sample of S ≥ 30 individuals use Waples' relations: the r² expected from sampling alone is 1/S + 3.19/S², the drift part x of a mean r² is the mean minus that expectation, and Ne = (1/3 + √(1/9 − 2.76x))/(2x), equivalently x = 1/(3Ne) − 0.69/Ne², with Ne infinite when x ≤ 0. Average r² over the unlinked pairs of all three chromosomes together, take the drift-induced r² between SNPs on different chromosomes as zero in the correlation terms, use z = 1.96 for two-sided 95 per cent limits, and convert each confidence limit for the expected mean r² into a limit for Ne with Waples' relations for the same S.

The full survey is now being sized, with the pilot's point estimate taken as the true Ne and the pilot's drift-induced same-chromosome r² as fixed. A survey certifies the population when its own corrected lower 95 per cent limit for Ne exceeds 500. The survey's mean r² over the unlinked pairs is not its expectation but a draw around it: in the source's model that mean is normally distributed about 1/S + 3.19/S² + 1/(3Ne) − 0.69/Ne², with a standard deviation of √(2(1 + 2ρ)/N) times that expectation, where N is the number of unlinked pairs. Find S_A, the smallest S ≥ 30 whose probability of certifying the population is at least 0.9.

In the reasoning, report the pilot's point estimate of Ne; the drift-induced mean same-chromosome r² on chromosome 3; ρ, and the part of ρ contributed by pairs of unlinked pairs that share no SNP; the pilot's corrected 95 per cent limits for Ne; the smallest S ≥ 30 whose expected corrected upper limit is finite; the smallest S ≥ 30 whose expected corrected lower limit exceeds 500, with the probability that a survey of that size would certify; the probability that a repeat of the pilot's own size would certify; S_A, with the expected mean unlinked r² and the expected corrected lower limit there; and, with the same planning Ne, the smallest S ≥ 30 that would reach the same assurance if the panel were reduced to chromosomes 1 and 2. State briefly how you obtained the correlations between unlinked r² values, the definition of ρ you used, how the confidence interval was built, and how you turned the assurance into a sample size. Your final answer is the expected corrected upper 95 per cent limit for Ne at S_A.

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

01_waples_ne_estimate.py

Goal
----
Effective population size from a mean r^2 over unlinked pairs of loci, by Waples' relations for 30 or more individuals.

```python
def waples_ne_estimate(r2_mean: float, sample_size: int) -> float:
    '''Effective population size from a mean r^2 over unlinked pairs of loci, by Waples' relations for 30 or more individuals.

    The expected r^2 from sampling alone in a sample of S diploid
    individuals is 1/S + 3.19/S^2. The drift part x of a mean r^2 is the
    mean minus that expectation, and the effective size is
    Ne = (1/3 + sqrt(1/9 - 2.76 x)) / (2 x), which is the root of
    x = 1/(3 Ne) - 0.69/Ne^2 on the branch that grows as x falls.

    Parameters
    ----------
    r2_mean : float
        Mean r^2 over pairs of unlinked loci, between 0 and 1.
    sample_size : int
        Number of diploid individuals S, an integer from 30 to 100000.

    Returns
    -------
    ne : float
        The estimated effective population size, as a native Python float.

    Raises
    ------
    ValueError
        If sample_size is not an integer from 30 to 100000, if r2_mean is
        not a finite number between 0 and 1, if the drift part x is below
        1e-6 (the estimate would be larger than about 333,000 or infinite),
        or if x exceeds 1/(9 * 2.76), where the relation has no real
        solution.
    '''
    return ne  # placeholder
```

### Step 2

02_expected_unlinked_r2.py

Goal
----
Expected mean r^2 over unlinked pairs of loci for a population of effective size Ne sampled with S diploid individuals.

```python
def expected_unlinked_r2(ne: float, sample_size: int) -> float:
    '''Expected mean r^2 over unlinked pairs of loci for a population of effective size Ne sampled with S diploid individuals.

    The expectation is the sampling part 1/S + 3.19/S^2 plus the drift part
    1/(3 Ne) - 0.69/Ne^2, so that Waples' estimator applied to the result
    returns Ne.

    Parameters
    ----------
    ne : float
        Effective population size, a finite number from 5 to 300000.
    sample_size : int
        Number of diploid individuals S, an integer from 30 to 100000.

    Returns
    -------
    r2 : float
        The expected mean r^2 over unlinked pairs, as a native Python float.

    Raises
    ------
    ValueError
        If ne is not a finite number from 5 to 300000, or if sample_size is
        not an integer from 30 to 100000.
    '''
    return r2  # placeholder
```

### Step 3

03_pseudo_replication_rho.py

Goal
----
Pseudo-replication parameter rho of the mean r^2 over all pairs of SNPs on different chromosomes.

```python
def pseudo_replication_rho(raw_same_chromosome_r2: "np.ndarray", n_snps: "np.ndarray", sample_size: int) -> float:
    '''Pseudo-replication parameter rho of the mean r^2 over all pairs of SNPs on different chromosomes.

    The panel has C chromosomes, with L_c SNPs on chromosome c, genotyped in a
    sample of S diploid individuals. The unlinked pairs are all N = sum over
    a < b of L_a L_b pairs of SNPs on different chromosomes, and their r^2
    values are averaged. rho is the source's parameter: the sum, over all
    unordered pairs of distinct unlinked pairs, of the correlation between
    their r^2 values, divided by N, so that the variance of the mean is
    1 + 2 rho times its value if the N r^2 values were independent. The
    correlations follow the source's model, which expresses them through the
    drift-induced r^2 among SNPs that lie on the same chromosome;
    drift-induced LD between SNPs on different chromosomes is zero.

    Entry c of raw_same_chromosome_r2 is the sample's mean raw r^2 over all
    L_c (L_c - 1) / 2 unordered pairs of distinct SNPs on chromosome c. Its
    sampling component has the same expectation as for an unlinked pair,
    1/S + 3.19/S^2, and the rest is the drift-induced part.

    Parameters
    ----------
    raw_same_chromosome_r2 : np.ndarray
        Shape (C,), C >= 2; the sample's mean raw r^2 over the pairs of
        distinct SNPs on each chromosome, each a finite number between
        1/S + 3.19/S^2 and 1.
    n_snps : np.ndarray
        Shape (C,); integer numbers of SNPs L_c, each from 2 to 10**6.
    sample_size : int
        Number of diploid individuals S, an integer from 30 to 100000.

    Returns
    -------
    rho : float
        The pseudo-replication parameter, as a native Python float.

    Raises
    ------
    ValueError
        If the arrays are not one-dimensional with the same length C >= 2, if
        n_snps holds a value that is not an integer from 2 to 10**6, if
        sample_size is not an integer from 30 to 100000, or if an entry of
        raw_same_chromosome_r2 is not finite, is below the sampling
        expectation or is above 1.
    '''
    return rho  # placeholder
```

### Step 4

04_r2_confidence_limits.py

Goal
----
Two-sided confidence limits for the expected mean r^2 over unlinked pairs, allowing for pseudo-replication.

```python
def r2_confidence_limits(r2_mean: float, rho: float, n_snps: "np.ndarray", z: float) -> "np.ndarray":
    '''Two-sided confidence limits for the expected mean r^2 over unlinked pairs, allowing for pseudo-replication.

    The observed mean r2_mean is taken over all N = sum over a < b of
    L_a L_b pairs of SNPs on different chromosomes, and rho is the source's
    pseudo-replication parameter of that mean for the given panel. The source
    takes r2_mean / E(r^2) to have mean 1 and variance 2 (1 + 2 rho) / N, and
    inverts 1 - w <= r2_mean / E(r^2) <= 1 + w, where
    w = z sqrt(2 (1 + 2 rho) / N): the lower limit is r2_mean / (1 + w), and
    the upper limit is r2_mean / (1 - w), or infinite when w >= 1. Each limit
    is clamped to the range 0 to 1 of r^2.

    Parameters
    ----------
    r2_mean : float
        Observed mean r^2 over the N unlinked pairs, a finite number between
        0 and 1, not 0.
    rho : float
        The panel's pseudo-replication parameter, a finite number that is not
        negative.
    n_snps : np.ndarray
        Shape (C,), C >= 2; integer numbers of SNPs L_c on each chromosome,
        each from 2 to 10**6.
    z : float
        Standard normal quantile, a finite number from 0.5 to 5.

    Returns
    -------
    limits : np.ndarray
        Shape (2,), float; the lower and the upper confidence limit for the
        expected mean r^2, in that order.

    Raises
    ------
    ValueError
        If r2_mean is not a finite number in (0, 1], if rho is not a finite
        number that is at least 0, if z is not a finite number from 0.5 to 5,
        or if n_snps is not a one-dimensional array of at least two integers
        from 2 to 10**6.
    '''
    return limits  # placeholder
```

### Step 5

05_ne_confidence_limits.py

Goal
----
Two-sided confidence limits for the effective population size from a mean r^2 over unlinked pairs, allowing for pseudo-replication.

```python
def ne_confidence_limits(r2_mean: float, sample_size: int, rho: float, n_snps: "np.ndarray", z: float) -> "np.ndarray":
    '''Two-sided confidence limits for the effective population size from a mean r^2 over unlinked pairs, allowing for pseudo-replication.

    The confidence limits for the expected mean unlinked r^2 are those of the
    source's interval for the given panel and pseudo-replication parameter (as
    in the r^2 limits step), and each is converted to an effective size with
    Waples' relations for S diploid individuals (as in the estimate step): the
    sampling expectation is 1/S + 3.19/S^2 and
    Ne = (1/3 + sqrt(1/9 - 2.76 x)) / (2 x) for the drift part x of the limit.

    Parameters
    ----------
    r2_mean : float
        Observed mean r^2 over all pairs of SNPs on different chromosomes, a
        finite number between 0 and 1.
    sample_size : int
        Number of diploid individuals S, an integer from 30 to 100000.
    rho : float
        The panel's pseudo-replication parameter, a finite number that is not
        negative.
    n_snps : np.ndarray
        Shape (C,), C >= 2; integer numbers of SNPs L_c on each chromosome,
        each from 2 to 10**6.
    z : float
        Standard normal quantile, a finite number from 0.5 to 5.

    Returns
    -------
    limits : np.ndarray
        Shape (2,), float; the lower and the upper confidence limit for the
        effective population size, in that order.

    Raises
    ------
    ValueError
        If any argument is invalid as in the r^2 limits step or the estimate
        step, or if either limit is not a finite effective size: the drift
        part of the lower r^2 limit must be at least 1e-6, and the drift part
        of the upper r^2 limit at most 1/(9 * 2.76).
    '''
    return limits  # placeholder
```

### Step 6

06_certification_probability.py

Goal
----
Probability that a future sample shows the effective population size to exceed a threshold.

```python
def certification_probability(ne: float, ne_threshold: float, sample_size: int, rho: float, n_snps: "np.ndarray", z: float) -> float:
    '''Probability that a future sample shows the effective population size to exceed a threshold.

    A future sample of S diploid individuals is genotyped at the given panel,
    and the corrected two-sided confidence limits for the effective size are
    computed from its own mean r^2 over the panel's unlinked pairs, at the
    same sample size, rho and z (as in the Ne limits step). The sample
    certifies the population when that lower limit is strictly greater than
    ne_threshold. A sample whose upper confidence limit for the expected mean
    r^2 has a drift part at or below zero leaves the effective size unbounded
    from above and so certifies; one whose upper limit has a drift part above
    1/(9 * 2.76), where Waples' relation has no solution, does not.

    The population's effective size is ne. Under the source's model the future
    sample's mean r^2 over the unlinked pairs is normally distributed, with
    mean the expected mean unlinked r^2 for ne and S (as in the expected r^2
    step) and with standard deviation that expectation times
    sqrt(2 (1 + 2 rho) / N), where N is the panel's number of unlinked pairs.
    Return the probability, under that distribution of the sample's mean r^2,
    that the sample certifies the population.

    Parameters
    ----------
    ne : float
        The population's effective size, a finite number from 5 to 300000.
    ne_threshold : float
        Threshold the lower limit must exceed, a finite number from 5 to ne.
    sample_size : int
        Number of diploid individuals S in the future sample, an integer from
        30 to 100000.
    rho : float
        The panel's pseudo-replication parameter, a finite number that is not
        negative.
    n_snps : np.ndarray
        Shape (C,), C >= 2; integer numbers of SNPs L_c on each chromosome,
        each from 2 to 10**6.
    z : float
        Standard normal quantile of the confidence limits, a finite number
        from 0.5 to 5.

    Returns
    -------
    probability : float
        The probability that the future sample certifies the population, a
        native Python float between 0 and 1.

    Raises
    ------
    ValueError
        If ne, ne_threshold, sample_size or z are outside their ranges, if rho
        or n_snps are invalid as for the r^2 limits step, or if
        z sqrt(2 (1 + 2 rho) / N) is 1 or more, where the upper confidence
        limit carries no information.
    '''
    return probability  # placeholder
```

### Step 7

07_smallest_assured_sample.py

Goal
----
Smallest number of diploid individuals whose probability of showing the effective size to exceed a threshold reaches an assurance level.

```python
def smallest_assured_sample(ne: float, ne_threshold: float, assurance: float, rho: float, n_snps: "np.ndarray", z: float, max_sample: int) -> int:
    '''Smallest number of diploid individuals whose probability of showing the effective size to exceed a threshold reaches an assurance level.

    For each integer S from 30 upwards, take the probability that a future
    sample of S individuals genotyped at the given panel certifies a
    population of effective size ne against ne_threshold, as the
    certification-probability step defines it for the same rho and z. Return
    the first S at which that probability is greater than or equal to
    assurance.

    Parameters
    ----------
    ne : float
        The population's effective size, a finite number from 5 to 300000.
    ne_threshold : float
        Threshold the lower limit must exceed, a finite number from 5 to ne.
    assurance : float
        Required probability of certifying, a finite number from 0.5 to 0.999.
    rho : float
        The panel's pseudo-replication parameter, a finite number that is not
        negative.
    n_snps : np.ndarray
        Shape (C,), C >= 2; integer numbers of SNPs L_c on each chromosome,
        each from 2 to 10**6.
    z : float
        Standard normal quantile of the confidence limits, a finite number
        from 0.5 to 5.
    max_sample : int
        Largest sample size to consider, an integer from 30 to 100000.

    Returns
    -------
    sample_size : int
        The smallest qualifying S, as a native Python int.

    Raises
    ------
    ValueError
        If assurance or max_sample are outside their ranges, if any other
        argument is invalid as for the certification-probability step, or if
        no S up to max_sample qualifies.
    '''
    return sample_size  # placeholder
```

### Step 8

08_planned_upper_ne_limit.py

Goal
----
Expected upper confidence limit for the effective size at the smallest sample assured of showing it exceeds a threshold.

```python
def planned_upper_ne_limit(pilot_r2_mean: float, pilot_size: int, raw_same_chromosome_r2: "np.ndarray", n_snps: "np.ndarray", ne_threshold: float, assurance: float, z: float, max_sample: int) -> float:
    '''Expected upper confidence limit for the effective size at the smallest sample assured of showing it exceeds a threshold.

    The pilot sample of pilot_size diploid individuals has mean r^2
    pilot_r2_mean over all pairs of SNPs on different chromosomes and, on
    chromosome c, mean raw r^2 raw_same_chromosome_r2[c] over all pairs of its
    n_snps[c] SNPs. The pilot's point estimate of Ne (as in the estimate step)
    is taken as the true effective size, and the panel's pseudo-replication
    parameter (as in the rho step, for the pilot's own sample size) as fixed.
    The smallest assured sample is found as in the assured-sample step with
    this Ne, ne_threshold, assurance, the panel, z and max_sample. At that
    sample size the mean r^2 over unlinked pairs is set to its expectation (as
    in the expected r^2 step), and the confidence limits for the effective
    size follow as in the Ne-limits step. Return the upper limit.

    Parameters
    ----------
    pilot_r2_mean : float
        The pilot's mean r^2 over unlinked pairs, a finite number between 0
        and 1.
    pilot_size : int
        Number of diploid individuals in the pilot, an integer from 30 to
        100000.
    raw_same_chromosome_r2 : np.ndarray
        Shape (C,), C >= 2; the pilot's mean raw r^2 over pairs of SNPs on
        each chromosome, each between the pilot's sampling expectation and 1.
    n_snps : np.ndarray
        Shape (C,); integer numbers of SNPs L_c on each chromosome, each from
        2 to 10**6.
    ne_threshold : float
        Threshold the lower limit must exceed, a finite number from 5 to the
        pilot estimate.
    assurance : float
        Required probability of certifying, a finite number from 0.5 to 0.999.
    z : float
        Standard normal quantile, a finite number from 0.5 to 5.
    max_sample : int
        Largest sample size to consider, an integer from 30 to 100000.

    Returns
    -------
    upper_limit : float
        The expected upper confidence limit for the effective size at the
        assured sample size, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is invalid as in the steps it is passed to, if the
        pilot's point estimate exceeds 300000, if no sample size up to
        max_sample reaches the assurance, or if the upper confidence limit at
        that sample size is not finite.
    '''
    return upper_limit  # placeholder
```
