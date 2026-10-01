# Biology-Genetics-16

## Problem

Population adaptation rate is generally set by the additive genetic variance for relative fitness, denoted by V_A. Measuring V_A using conventional means necessitates fitness metrics on many individuals of documented relatedness. This is most infeasible, and is highly dependent on the metric used for fitness. As an alternative, consider abandoning individual fitness, and calculate V_A from the change in genetic composition of the population itself. After a few generations of evolution, replicates from a common starting base leave a genomic signature of selection in their allele frequencies. After accounting for diversity and linkage structure, this should be sufficient to recover V_A. This algorithm would require as input: a sequenced base population, a recombination map, and the allele-frequency change for each replicate. The algorithm would output a single variance value.

The calculations here would be genome-wide, and conditioned on the diversity and linkage of the base population/recombination map. Here, the replicate structure identifies the selection component, because allele-frequency change alone confounds selection with drift.

Use this configuration:
* Base population: 10 biallelic autosomal loci across 2000 diploid individuals, physically ordered along one chromosome.
* Replicates: six. Each was founded from the offspring of one round of random mating in the base population; its allele frequencies were measured in that founding generation and again 2 generations later.
* Census sizes: 2000, 1200, 700, 400, 250 and 150 individuals.
* Inbreeding effective size: the round of random mating that produced each replicate's founders passed through a bottleneck with an inbreeding effective size of 50. In every later round of mating, treat the replicate's census size as its inbreeding effective size.
* Pairing: individual k carries haplotype rows 2k and 2k+1.
* Recombination: crossover probability 0.05 between adjacent loci; assume no crossover interference.
* Haplotypes: threshold a latent Gaussian AR(1) process along the locus axis (unit variance, stationary, lag-1 correlation 0.90). Innovations are a single (4000, 10) standard-normal array drawn from `numpy.random.default_rng(2043)`. The first column initializes the process. Locus i carries the reference allele on a haplotype when its latent value exceeds the standard-normal quantile of 1 - t_i, where t_i is the i-th of 10 values evenly spaced from 0.15 to 0.85 inclusive.
* Variance in offspring number: 6.0 in every replicate.
* Observed allele-frequency changes between the two measurements, locus by locus:

* R1: [-0.080508, -0.088382, -0.092691, -0.079427, -0.029581, +0.007478, +0.057723, +0.081924, +0.063215, +0.058229]
* R2: [-0.087222, -0.098856, -0.084800, -0.045451, -0.039817, -0.008562, +0.026788, +0.041907, +0.049555, +0.057158]
* R3: [-0.104796, -0.131458, -0.065336, -0.087962, -0.065417, -0.008970, +0.015046, +0.058879, +0.065851, +0.072945]
* R4: [-0.076533, -0.060456, -0.062452, -0.045712, -0.024884, +0.022225, +0.068221, +0.082374, +0.104465, +0.091738]
* R5: [-0.063602, -0.083285, -0.104352, -0.102128, -0.025376, +0.053171, +0.009892, +0.118097, +0.088889, +0.056976]
* R6: [-0.079714, -0.159038, -0.173297, -0.097500, -0.076308, +0.005407, +0.028781, +0.104331, +0.044838, +0.049813]

Assume the average effects for relative fitness scale linearly with the reference-allele frequency contrast p - q of the base population, where p and q are the realised frequencies of the simulated haplotypes. Further assume no across-replicate variance in the average effects, and correct the resulting quadratic form for the sampling error in the fitted scale coefficient. Report the estimated additive genetic variance for relative fitness of the base population. In the reasoning, report the realised reference-allele frequencies of the base population, the fitted scale coefficient and its sampling variance, and the quadratic form before and after the correction. Your final answer must be a single number: V_A.

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

01_simulate_base_population.py

Goal
----
'''Materialise the phased base population under the reference construction.




    Draw Z = rng.standard_normal((2 * n_individuals, n_loci)) from

    numpy.random.default_rng(seed). Set X[:, 0] = Z[:, 0] and, for i >= 1,

    X[:, i] = rho * X[:, i-1] + sqrt(1 - rho**2) * Z[:, i]. Locus i carries the

    reference allele on a haplotype when X[:, i] exceeds the standard-normal

    quantile of (1 - t_i), where t_i is the i-th of n_loci values evenly spaced

    from p_low to p_high inclusive. Individual k owns rows 2k and 2k+1.




    Parameters

    ----------

    n_loci : int

        Number of biallelic loci, in physical order along one chromosome. >= 2.

    n_individuals : int

        Number of diploid individuals in the base population. >= 2.

    rho : float

        Lag-1 correlation of the latent AR(1) process. In [0, 1).

    p_low : float

        Target reference-allele frequency at the first locus. In (0, 1).

    p_high : float

        Target reference-allele frequency at the last locus. In (0, 1).

    seed : int

        Seed passed to numpy.random.default_rng.




    Returns

    -------

    haplotypes : np.ndarray

        (2 * n_individuals, n_loci) array of 0/1 integers.




    Raises

    ------

    ValueError

        If n_loci < 2, if n_individuals < 2, if rho is not in [0, 1),

        if p_low or p_high is not in (0, 1), or if seed is not an integer.

    '''

```python
def simulate_base_population(n_loci: int, n_individuals: int, rho: float,
                             p_low: float, p_high: float, seed: int) -> np.ndarray:
    '''Materialise the phased base population under the reference construction.

    Draw Z = rng.standard_normal((2 * n_individuals, n_loci)) from
    numpy.random.default_rng(seed). Set X[:, 0] = Z[:, 0] and, for i >= 1,
    X[:, i] = rho * X[:, i-1] + sqrt(1 - rho**2) * Z[:, i]. Locus i carries the
    reference allele on a haplotype when X[:, i] exceeds the standard-normal
    quantile of (1 - t_i), where t_i is the i-th of n_loci values evenly spaced
    from p_low to p_high inclusive. Individual k owns rows 2k and 2k+1.

    Parameters
    ----------
    n_loci : int
        Number of biallelic loci, in physical order along one chromosome. >= 2.
    n_individuals : int
        Number of diploid individuals in the base population. >= 2.
    rho : float
        Lag-1 correlation of the latent AR(1) process. In [0, 1).
    p_low : float
        Target reference-allele frequency at the first locus. In (0, 1).
    p_high : float
        Target reference-allele frequency at the last locus. In (0, 1).
    seed : int
        Seed passed to numpy.random.default_rng.

    Returns
    -------
    haplotypes : np.ndarray
        (2 * n_individuals, n_loci) array of 0/1 integers.

    Raises
    ------
    ValueError
        If n_loci < 2, if n_individuals < 2, if rho is not in [0, 1),
        if p_low or p_high is not in (0, 1), or if seed is not an integer.
    '''
    return haplotypes  # placeholder
```

### Step 2

02_individual_diversity_matrix.py

Goal
----
'''Covariance of per-individual reference-allele PROPORTION across individuals.




    Individual k owns haplotype rows 2k and 2k+1, so its allele content at locus i

    is c[k, i] = (haplotypes[2k, i] + haplotypes[2k+1, i]) / 2, taking values in

    {0, 1/2, 1}. Return the population covariance of c across individuals, i.e.

    normalised by the number of individuals (ddof = 0), not by n - 1.




    Parameters

    ----------

    haplotypes : np.ndarray

        (2N, n_loci) array of 0/1 values with N >= 2 individuals.




    Returns

    -------

    L : np.ndarray

        (n_loci, n_loci) symmetric covariance matrix.




    Raises

    ------

    ValueError

        If haplotypes is not a 2D array, if it has fewer than 4 rows, if its row

        count is odd, if it has fewer than 1 column, or if it contains a value

        other than 0 or 1.

    '''

```python
def individual_diversity_matrix(haplotypes: np.ndarray) -> np.ndarray:
    '''Covariance of per-individual reference-allele PROPORTION across individuals.

    Individual k owns haplotype rows 2k and 2k+1, so its allele content at locus i
    is c[k, i] = (haplotypes[2k, i] + haplotypes[2k+1, i]) / 2, taking values in
    {0, 1/2, 1}. Return the population covariance of c across individuals, i.e.
    normalised by the number of individuals (ddof = 0), not by n - 1.

    Parameters
    ----------
    haplotypes : np.ndarray
        (2N, n_loci) array of 0/1 values with N >= 2 individuals.

    Returns
    -------
    L : np.ndarray
        (n_loci, n_loci) symmetric covariance matrix.

    Raises
    ------
    ValueError
        If haplotypes is not a 2D array, if it has fewer than 4 rows, if its row
        count is odd, if it has fewer than 1 column, or if it contains a value
        other than 0 or 1.
    '''
    return L  # placeholder
```

### Step 3

03_pairwise_recombination_matrix.py

Goal
----
'''Recombination probability between every pair of loci, assuming no interference.




    interval_rates[i] is the crossover probability between locus i and locus i+1,

    for n_loci - 1 intervals along one chromosome in physical order. Assuming

    crossovers occur without interference, convert each interval probability to an

    additive map distance, accumulate distances along the chromosome, and convert

    absolute pairwise distance differences back to recombination probabilities.




    Parameters

    ----------

    interval_rates : np.ndarray

        (n_loci - 1,) array of crossover probabilities, each in [0, 0.5).




    Returns

    -------

    R : np.ndarray

        (n_loci, n_loci) symmetric matrix with a zero diagonal and all entries

        in [0, 0.5).




    Raises

    ------

    ValueError

        If interval_rates is not a 1D array, if it is empty, or if any entry is

        not in [0, 0.5).

    '''

```python
def pairwise_recombination_matrix(interval_rates: np.ndarray) -> np.ndarray:
    '''Recombination probability between every pair of loci, assuming no interference.

    interval_rates[i] is the crossover probability between locus i and locus i+1,
    for n_loci - 1 intervals along one chromosome in physical order. Assuming
    crossovers occur without interference, convert each interval probability to an
    additive map distance, accumulate distances along the chromosome, and convert
    absolute pairwise distance differences back to recombination probabilities.

    Parameters
    ----------
    interval_rates : np.ndarray
        (n_loci - 1,) array of crossover probabilities, each in [0, 0.5).

    Returns
    -------
    R : np.ndarray
        (n_loci, n_loci) symmetric matrix with a zero diagonal and all entries
        in [0, 0.5).

    Raises
    ------
    ValueError
        If interval_rates is not a 1D array, if it is empty, or if any entry is
        not in [0, 0.5).
    '''
    return R  # placeholder
```

### Step 4

04_weighted_base_diversity_matrix.py

Goal
----
'''Base-population matrix that the forward projection of diversity acts on.




    Split the allele-content covariance of the base population into its

    gametic-phase component (pairs of alleles on the same gamete) and its

    non-gametic-phase component (pairs of alleles on the two different gametes of

    an individual). Combine them into the matrix L_tilde defined by this property:

    in the absence of selection and drift, one round of random mating produces a

    generation whose expected gametic-phase component is

    (1 - recombination) * L_tilde, elementwise.




    Parameters

    ----------

    haplotypes : np.ndarray

        (2N, n_loci) array of 0/1 values, individual k owning rows 2k and 2k+1.

    recombination : np.ndarray

        (n_loci, n_loci) symmetric matrix of pairwise recombination probabilities

        with a zero diagonal and entries in [0, 0.5).




    Returns

    -------

    L_tilde : np.ndarray

        (n_loci, n_loci) symmetric matrix.




    Raises

    ------

    ValueError

        If haplotypes is not a 2D array with an even number of rows >= 4, if it

        contains a value other than 0 or 1, if recombination is not a square 2D

        array whose size matches the locus count, if recombination is not

        symmetric within atol=1e-12, or if any entry of recombination is outside

        [0, 0.5).

    '''

```python
def weighted_base_diversity_matrix(haplotypes: np.ndarray,
                                   recombination: np.ndarray) -> np.ndarray:
    '''Base-population matrix that the forward projection of diversity acts on.

    Split the allele-content covariance of the base population into its
    gametic-phase component (pairs of alleles on the same gamete) and its
    non-gametic-phase component (pairs of alleles on the two different gametes of
    an individual). Combine them into the matrix L_tilde defined by this property:
    in the absence of selection and drift, one round of random mating produces a
    generation whose expected gametic-phase component is
    (1 - recombination) * L_tilde, elementwise.

    Parameters
    ----------
    haplotypes : np.ndarray
        (2N, n_loci) array of 0/1 values, individual k owning rows 2k and 2k+1.
    recombination : np.ndarray
        (n_loci, n_loci) symmetric matrix of pairwise recombination probabilities
        with a zero diagonal and entries in [0, 0.5).

    Returns
    -------
    L_tilde : np.ndarray
        (n_loci, n_loci) symmetric matrix.

    Raises
    ------
    ValueError
        If haplotypes is not a 2D array with an even number of rows >= 4, if it
        contains a value other than 0 or 1, if recombination is not a square 2D
        array whose size matches the locus count, if recombination is not
        symmetric within atol=1e-12, or if any entry of recombination is outside
        [0, 0.5).
    '''
    return L_tilde  # placeholder
```

### Step 5

05_variance_effective_size.py

Goal
----
'''Variance effective population size from census size and offspring variance.




    Use the large-population form N_E = 4 N / (2 + V_o) (Wright 1938), with N the

    census size and V_o the variance in offspring number.




    Parameters

    ----------

    census_size : int

        Census number of diploid individuals, N >= 1.

    offspring_variance : float

        Variance in offspring number in the absence of additive genetic variance

        for fitness, V_o >= 0.




    Returns

    -------

    n_effective : float

        Variance effective size as a native Python float.




    Raises

    ------

    ValueError

        If census_size is not an integer >= 1, or if offspring_variance is not a

        finite real number >= 0.

    '''

```python
def variance_effective_size(census_size: int, offspring_variance: float) -> float:
    '''Variance effective population size from census size and offspring variance.

    Use the large-population form N_E = 4 N / (2 + V_o) (Wright 1938), with N the
    census size and V_o the variance in offspring number.

    Parameters
    ----------
    census_size : int
        Census number of diploid individuals, N >= 1.
    offspring_variance : float
        Variance in offspring number in the absence of additive genetic variance
        for fitness, V_o >= 0.

    Returns
    -------
    n_effective : float
        Variance effective size as a native Python float.

    Raises
    ------
    ValueError
        If census_size is not an integer >= 1, or if offspring_variance is not a
        finite real number >= 0.
    '''
    return n_effective  # placeholder
```

### Step 6

06_expected_change_operator.py

Goal
----
'''Operator mapping base-population average effects onto expected frequency change.




    Return the matrix L_m such that L_m @ alpha_bar is the expected

    allele-frequency change due to selection between the two measurements of a

    replicate, where alpha_bar holds the mean average effects for relative

    fitness. The replicate was founded from the offspring of one round of random

    mating in the base population; the first measurement is taken in that

    founding generation and the second n_generations generations later. From the

    base population onward, the structure carried by base_matrix is eroded by

    recombination and by drift. Drift acts at founding_effective_size in the round

    of random mating that produced the founders and at inbreeding_effective_size in

    every later round.




    Parameters

    ----------

    base_matrix : np.ndarray

        (n_loci, n_loci) symmetric weighted base-population matrix, as returned

        by weighted_base_diversity_matrix.

    recombination : np.ndarray

        (n_loci, n_loci) symmetric matrix of pairwise recombination probabilities

        with a zero diagonal and entries in [0, 0.5).

    n_generations : int

        Number of generations between the two measurements, >= 1.

    founding_effective_size : float

        Inbreeding effective size of the round of random mating that produced the

        replicate's founders. Must be > 0.5.

    inbreeding_effective_size : float

        Inbreeding effective size of the replicate in every round of mating after

        the founding round. Must be > 0.5.




    Returns

    -------

    operator : np.ndarray

        (n_loci, n_loci) symmetric matrix.




    Raises

    ------

    ValueError

        If base_matrix or recombination is not a square 2D array, if their shapes

        disagree, if either is not symmetric within atol=1e-12, if any entry of

        recombination is outside [0, 0.5), if n_generations is not an integer >= 1,

        or if founding_effective_size or inbreeding_effective_size is not a finite

        number > 0.5.

    '''

```python
def expected_change_operator(base_matrix: np.ndarray, recombination: np.ndarray,
                             n_generations: int, founding_effective_size: float,
                             inbreeding_effective_size: float) -> np.ndarray:
    '''Operator mapping base-population average effects onto expected frequency change.

    Return the matrix L_m such that L_m @ alpha_bar is the expected
    allele-frequency change due to selection between the two measurements of a
    replicate, where alpha_bar holds the mean average effects for relative
    fitness. The replicate was founded from the offspring of one round of random
    mating in the base population; the first measurement is taken in that
    founding generation and the second n_generations generations later. From the
    base population onward, the structure carried by base_matrix is eroded by
    recombination and by drift. Drift acts at founding_effective_size in the round
    of random mating that produced the founders and at inbreeding_effective_size in
    every later round.

    Parameters
    ----------
    base_matrix : np.ndarray
        (n_loci, n_loci) symmetric weighted base-population matrix, as returned
        by weighted_base_diversity_matrix.
    recombination : np.ndarray
        (n_loci, n_loci) symmetric matrix of pairwise recombination probabilities
        with a zero diagonal and entries in [0, 0.5).
    n_generations : int
        Number of generations between the two measurements, >= 1.
    founding_effective_size : float
        Inbreeding effective size of the round of random mating that produced the
        replicate's founders. Must be > 0.5.
    inbreeding_effective_size : float
        Inbreeding effective size of the replicate in every round of mating after
        the founding round. Must be > 0.5.

    Returns
    -------
    operator : np.ndarray
        (n_loci, n_loci) symmetric matrix.

    Raises
    ------
    ValueError
        If base_matrix or recombination is not a square 2D array, if their shapes
        disagree, if either is not symmetric within atol=1e-12, if any entry of
        recombination is outside [0, 0.5), if n_generations is not an integer >= 1,
        or if founding_effective_size or inbreeding_effective_size is not a finite
        number > 0.5.
    '''
    return operator  # placeholder
```

### Step 7

07_drift_covariance_matrix.py

Goal
----
'''Covariance across loci of the drift contribution to observed frequency change.




    Return the covariance, taken over the evolutionary process, of the part of the

    allele-frequency change between the two measurements of a replicate that is

    due to genetic drift. The replicate history is the one described for

    expected_change_operator: founded from the offspring of one round of random

    mating in the base population, measured in that founding generation and again

    n_generations generations later, with the structure carried by base_matrix

    eroded by recombination and by drift, drift acting at founding_effective_size

    in the round of random mating that produced the founders and at

    inbreeding_effective_size in every later round. The sampling of allele

    frequencies in each generation of the window is governed by the variance

    effective size n_effective.




    Parameters

    ----------

    base_matrix : np.ndarray

        (n_loci, n_loci) symmetric weighted base-population matrix, as returned

        by weighted_base_diversity_matrix.

    recombination : np.ndarray

        (n_loci, n_loci) symmetric matrix of pairwise recombination probabilities

        with a zero diagonal and entries in [0, 0.5).

    n_generations : int

        Number of generations between the two measurements, >= 1.

    founding_effective_size : float

        Inbreeding effective size of the round of random mating that produced the

        replicate's founders. Must be > 0.5.

    inbreeding_effective_size : float

        Inbreeding effective size of the replicate in every round of mating after

        the founding round. Must be > 0.5.

    n_effective : float

        Variance effective size governing the sampling variance of allele

        frequency. Must be > 0.




    Returns

    -------

    covariance : np.ndarray

        (n_loci, n_loci) symmetric matrix.




    Raises

    ------

    ValueError

        If base_matrix or recombination is not a square 2D array, if their shapes

        disagree, if either is not symmetric within atol=1e-12, if any entry of

        recombination is outside [0, 0.5), if n_generations is not an integer >= 1,

        if founding_effective_size or inbreeding_effective_size is not a finite

        number > 0.5, or if n_effective is not a finite number > 0.

    '''

```python
def drift_covariance_matrix(base_matrix: np.ndarray, recombination: np.ndarray,
                            n_generations: int, founding_effective_size: float,
                            inbreeding_effective_size: float,
                            n_effective: float) -> np.ndarray:
    '''Covariance across loci of the drift contribution to observed frequency change.

    Return the covariance, taken over the evolutionary process, of the part of the
    allele-frequency change between the two measurements of a replicate that is
    due to genetic drift. The replicate history is the one described for
    expected_change_operator: founded from the offspring of one round of random
    mating in the base population, measured in that founding generation and again
    n_generations generations later, with the structure carried by base_matrix
    eroded by recombination and by drift, drift acting at founding_effective_size
    in the round of random mating that produced the founders and at
    inbreeding_effective_size in every later round. The sampling of allele
    frequencies in each generation of the window is governed by the variance
    effective size n_effective.

    Parameters
    ----------
    base_matrix : np.ndarray
        (n_loci, n_loci) symmetric weighted base-population matrix, as returned
        by weighted_base_diversity_matrix.
    recombination : np.ndarray
        (n_loci, n_loci) symmetric matrix of pairwise recombination probabilities
        with a zero diagonal and entries in [0, 0.5).
    n_generations : int
        Number of generations between the two measurements, >= 1.
    founding_effective_size : float
        Inbreeding effective size of the round of random mating that produced the
        replicate's founders. Must be > 0.5.
    inbreeding_effective_size : float
        Inbreeding effective size of the replicate in every round of mating after
        the founding round. Must be > 0.5.
    n_effective : float
        Variance effective size governing the sampling variance of allele
        frequency. Must be > 0.

    Returns
    -------
    covariance : np.ndarray
        (n_loci, n_loci) symmetric matrix.

    Raises
    ------
    ValueError
        If base_matrix or recombination is not a square 2D array, if their shapes
        disagree, if either is not symmetric within atol=1e-12, if any entry of
        recombination is outside [0, 0.5), if n_generations is not an integer >= 1,
        if founding_effective_size or inbreeding_effective_size is not a finite
        number > 0.5, or if n_effective is not a finite number > 0.
    '''
    return covariance  # placeholder
```

### Step 8

08_genic_variance_from_change.py

Goal
----
'''Additive GENIC variance for relative fitness, assuming linkage equilibrium.




    Sum the squared expected per-locus allele-frequency change divided by the

    corresponding per-locus entry on the diagonal of the diversity matrix. Loci

    whose diagonal entry is zero are monomorphic and contribute nothing; they must

    be skipped rather than producing a division by zero.




    Parameters

    ----------

    delta_p : np.ndarray

        (n_loci,) expected allele-frequency change attributable to selection.

    diversity_matrix : np.ndarray

        (n_loci, n_loci) base-population diversity matrix; only its diagonal is

        used here.




    Returns

    -------

    genic_variance : float

        Native Python float, >= 0.




    Raises

    ------

    ValueError

        If delta_p is not a finite 1D array, if diversity_matrix is not a square

        2D array whose size matches delta_p, if any diagonal entry of

        diversity_matrix is negative, or if a locus has a zero diagonal entry but

        a non-zero delta_p.

    '''

```python
def genic_variance_from_change(delta_p: np.ndarray,
                               diversity_matrix: np.ndarray) -> float:
    '''Additive GENIC variance for relative fitness, assuming linkage equilibrium.

    Sum the squared expected per-locus allele-frequency change divided by the
    corresponding per-locus entry on the diagonal of the diversity matrix. Loci
    whose diagonal entry is zero are monomorphic and contribute nothing; they must
    be skipped rather than producing a division by zero.

    Parameters
    ----------
    delta_p : np.ndarray
        (n_loci,) expected allele-frequency change attributable to selection.
    diversity_matrix : np.ndarray
        (n_loci, n_loci) base-population diversity matrix; only its diagonal is
        used here.

    Returns
    -------
    genic_variance : float
        Native Python float, >= 0.

    Raises
    ------
    ValueError
        If delta_p is not a finite 1D array, if diversity_matrix is not a square
        2D array whose size matches delta_p, if any diagonal entry of
        diversity_matrix is negative, or if a locus has a zero diagonal entry but
        a non-zero delta_p.
    '''
    return genic_variance  # placeholder
```

### Step 9

09_average_effect_scale.py

Goal
----
'''Generalised-least-squares scale coefficient for the mean average effects.




    The mean average effects are modelled as the coefficient times `contrast`, so

    replicate m has expected allele-frequency change equal to the coefficient times

    operators[m] applied to `contrast`, with residual covariance

    drift_covariances[m]. Fit the single coefficient across all replicates jointly

    by generalised least squares, weighting each replicate by the inverse of its

    own residual covariance.




    Parameters

    ----------

    delta_p_by_replicate : np.ndarray

        (n_replicates, n_loci) observed allele-frequency changes, one row per

        replicate, loci in a consistent order.

    operators : list

        Length-n_replicates list of (n_loci, n_loci) expected-change operators.

    drift_covariances : list

        Length-n_replicates list of (n_loci, n_loci) symmetric positive-definite

        drift covariance matrices.

    contrast : np.ndarray

        (n_loci,) reference-allele frequency contrast p - q in the base population.




    Returns

    -------

    coefficient : float

        Native Python float.




    Raises

    ------

    ValueError

        If delta_p_by_replicate is not a finite 2D array with at least one row, if

        operators or drift_covariances is not a list whose length matches the

        number of replicates, if any matrix is not square with size equal to the

        locus count, if contrast is not a finite 1D array of that length, if any

        drift covariance is not symmetric within atol=1e-12 or is not positive

        definite, or if the accumulated information is zero.

    '''

```python
def average_effect_scale(delta_p_by_replicate: np.ndarray,
                         operators: list, drift_covariances: list,
                         contrast: np.ndarray) -> float:
    '''Generalised-least-squares scale coefficient for the mean average effects.

    The mean average effects are modelled as the coefficient times `contrast`, so
    replicate m has expected allele-frequency change equal to the coefficient times
    operators[m] applied to `contrast`, with residual covariance
    drift_covariances[m]. Fit the single coefficient across all replicates jointly
    by generalised least squares, weighting each replicate by the inverse of its
    own residual covariance.

    Parameters
    ----------
    delta_p_by_replicate : np.ndarray
        (n_replicates, n_loci) observed allele-frequency changes, one row per
        replicate, loci in a consistent order.
    operators : list
        Length-n_replicates list of (n_loci, n_loci) expected-change operators.
    drift_covariances : list
        Length-n_replicates list of (n_loci, n_loci) symmetric positive-definite
        drift covariance matrices.
    contrast : np.ndarray
        (n_loci,) reference-allele frequency contrast p - q in the base population.

    Returns
    -------
    coefficient : float
        Native Python float.

    Raises
    ------
    ValueError
        If delta_p_by_replicate is not a finite 2D array with at least one row, if
        operators or drift_covariances is not a list whose length matches the
        number of replicates, if any matrix is not square with size equal to the
        locus count, if contrast is not a finite 1D array of that length, if any
        drift covariance is not symmetric within atol=1e-12 or is not positive
        definite, or if the accumulated information is zero.
    '''
    return coefficient  # placeholder
```

### Step 10

10_additive_genetic_variance_for_fitness.py

Goal
----
'''Additive genetic variance for relative fitness of the base population.




    Each replicate was founded from the offspring of one round of random mating in

    the base population, a round that passed through a bottleneck with inbreeding

    effective size founding_effective_size. Its allele frequencies were measured

    in that founding generation and again n_generations generations later, and

    delta_p_by_replicate holds the differences. Assume the mean average effects

    for relative fitness are proportional to the reference-allele frequency

    contrast p - q of the base population, using its realised frequencies, with no

    across-replicate variance in the average effects, and correct the resulting

    quadratic form for the sampling error in the fitted proportionality

    coefficient. Treat each replicate's census size as its inbreeding effective

    size in every round of mating after the founding round, and derive its

    variance effective size from that census size and offspring_variance.




    Parameters

    ----------

    delta_p_by_replicate : np.ndarray

        (n_replicates, n_loci) observed allele-frequency changes, one row per

        replicate, loci in the same physical order throughout.

    census_sizes : list

        Length-n_replicates list of int census sizes, one per replicate, in the

        same order as the rows of delta_p_by_replicate.

    founding_effective_size : float

        Inbreeding effective size of the founding round, shared by all

        replicates. Must be > 0.5.

    interval_rates : np.ndarray

        (n_loci - 1,) crossover probabilities between adjacent loci.

    n_generations : int

        Number of generations between the two measurements, >= 1.

    offspring_variance : float

        Variance in offspring number, shared across replicates, >= 0.

    population_spec : dict

        Exactly the keys 'n_loci', 'n_individuals', 'rho', 'p_low', 'p_high' and

        'seed', giving the reference construction of the base population.




    Returns

    -------

    v_a : float

        Estimated additive genetic variance for relative fitness, as a native

        Python float.




    Raises

    ------

    ValueError

        If delta_p_by_replicate is not a finite 2D array with at least one row, if

        census_sizes is not a list whose length matches the number of rows, if

        founding_effective_size is not a finite number > 0.5, if

        population_spec is not a dict holding exactly the six required keys, if

        population_spec['n_loci'] does not equal the number of columns of

        delta_p_by_replicate, or if interval_rates does not have length

        n_loci - 1. Conditions raised by the earlier steps propagate unchanged.

    '''

```python
def additive_genetic_variance_for_fitness(delta_p_by_replicate: np.ndarray,
                                          census_sizes: list,
                                          founding_effective_size: float,
                                          interval_rates: np.ndarray,
                                          n_generations: int,
                                          offspring_variance: float,
                                          population_spec: dict) -> float:
    '''Additive genetic variance for relative fitness of the base population.

    Each replicate was founded from the offspring of one round of random mating in
    the base population, a round that passed through a bottleneck with inbreeding
    effective size founding_effective_size. Its allele frequencies were measured
    in that founding generation and again n_generations generations later, and
    delta_p_by_replicate holds the differences. Assume the mean average effects
    for relative fitness are proportional to the reference-allele frequency
    contrast p - q of the base population, using its realised frequencies, with no
    across-replicate variance in the average effects, and correct the resulting
    quadratic form for the sampling error in the fitted proportionality
    coefficient. Treat each replicate's census size as its inbreeding effective
    size in every round of mating after the founding round, and derive its
    variance effective size from that census size and offspring_variance.

    Parameters
    ----------
    delta_p_by_replicate : np.ndarray
        (n_replicates, n_loci) observed allele-frequency changes, one row per
        replicate, loci in the same physical order throughout.
    census_sizes : list
        Length-n_replicates list of int census sizes, one per replicate, in the
        same order as the rows of delta_p_by_replicate.
    founding_effective_size : float
        Inbreeding effective size of the founding round, shared by all
        replicates. Must be > 0.5.
    interval_rates : np.ndarray
        (n_loci - 1,) crossover probabilities between adjacent loci.
    n_generations : int
        Number of generations between the two measurements, >= 1.
    offspring_variance : float
        Variance in offspring number, shared across replicates, >= 0.
    population_spec : dict
        Exactly the keys 'n_loci', 'n_individuals', 'rho', 'p_low', 'p_high' and
        'seed', giving the reference construction of the base population.

    Returns
    -------
    v_a : float
        Estimated additive genetic variance for relative fitness, as a native
        Python float.

    Raises
    ------
    ValueError
        If delta_p_by_replicate is not a finite 2D array with at least one row, if
        census_sizes is not a list whose length matches the number of rows, if
        founding_effective_size is not a finite number > 0.5, if
        population_spec is not a dict holding exactly the six required keys, if
        population_spec['n_loci'] does not equal the number of columns of
        delta_p_by_replicate, or if interval_rates does not have length
        n_loci - 1. Conditions raised by the earlier steps propagate unchanged.
    '''
    return v_a  # placeholder
```
