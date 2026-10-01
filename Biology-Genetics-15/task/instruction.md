# Biology-Genetics-15

## Background

Quantitative traits such as body size are typically controlled by many loci of small effect and held near an intermediate optimum by stabilizing selection, while mutation pushes them in a direction of its own and genetic drift erodes their variation. How much genetic variance such a trait keeps at equilibrium, how far its mean settles from the optimum, and how quickly the mean recovers from a chance excursion all follow from the balance of these forces at every locus at once, and that balance can change qualitatively with the strength of selection. The framework used here describes the trait from the perspective of its loci, following each allele frequency to a stationary balance that depends on the rest of the genome only through the mean trait, and the source scales mutation rates, selection and time by the population size in a convention of its own.

## Problem

Many quantitative traits are held near an intermediate optimum by stabilizing selection while mutation pushes them in a direction of its own, and how much genetic variance such a trait keeps at equilibrium is not a monotone function of how strongly it is selected: selection against the mutational bias can recentre allele frequencies and raise the variance before selection against heterozygotes depletes it. In this problem the inputs are a genetic architecture with dominance, a population size and a grid of selection strengths; the output is a single number, the relaxation time of the mean trait at the first strength on the grid at which the equilibrium variance falls below its neutral value.

Model a randomly mating population of N diploid individuals whose trait is controlled by L unlinked biallelic loci in three classes. The trait value of a genotype is the sum over loci of alpha_l g_l + D_l [g_l = 1], where g_l in {0, 1, 2} is the number of copies of the trait-increasing allele at locus l, alpha_l is its additive effect and D_l the dominance deviation of the heterozygote, so that the three genotypes at a locus have trait values 0, alpha_l + D_l and 2 alpha_l. Log fitness is -(z - eta)^2 / (2 omega^2), with optimum eta and selection strength omega^-2. At each locus the trait-decreasing allele mutates to the trait-increasing allele at rate mu+ per generation and back at rate mu-. Neglect linkage disequilibrium and take the diffusion limit of weak selection, in which the frequency of the trait-increasing allele at each locus is a diffusion with reversible mutation and drift whose selection coefficient is the regression coefficient of log fitness on the allele count under Hardy-Weinberg proportions, the other loci entering only through the mean trait value of the population. Take the population at stationarity, each locus at the stationary distribution of its diffusion, and require the mean trait value that enters the selection coefficients to equal the expectation of the trait over the stationary distributions of all loci; this fixes the equilibrium deviation Delta* of the mean trait from the optimum. Where a derivation from this model disagrees with a published approximation or closed form, the derivation governs.

Compute the equilibrium total genetic variance sigma^2 of the trait, the additive plus the dominance component under Hardy-Weinberg proportions, at every selection strength of the grid omega^-2 = 10^(k/10) for integer k from -20 to 10, together with the neutral variance sigma_0^2 of the same architecture under mutation and drift alone. Let k_hat be the smallest k at which sigma^2 is strictly below sigma_0^2. At omega^-2 = 10^(k_hat/10), describe the stationary fluctuations of the mean trait about its equilibrium value as an Ornstein-Uhlenbeck process and report the e-folding time of its autocorrelation, in generations. Evaluate every stationary expectation by a quadrature accurate to a relative error of 1e-8 or better, and every equilibrium deviation to an absolute accuracy of 1e-10 or better.

Use this configuration:
* Population: N = 10,000 diploid individuals.
* Class A: 60 loci, alpha = 0.012, D = 0, mu+ = 3 x 10^-6 and mu- = 2 x 10^-5 per locus per generation.
* Class B: 40 loci, alpha = 0.008, D = +0.007, mu+ = 5 x 10^-6 and mu- = 1.5 x 10^-5.
* Class C: 20 loci, alpha = 0.010, D = -0.008, mu+ = 10^-5 and mu- = 10^-5.
* Optimum: eta = 1.15, in the trait units of alpha and D.
* Selection strengths: the grid omega^-2 = 10^(k/10) for integer k from -20 to 10, in inverse squared trait units, and the reference strength omega^-2 = 1.

In the reasoning, report sigma_0^2; the largest value of sigma^2 / sigma_0^2 on the grid; at the reference strength omega^-2 = 1: Delta*, the additive genetic variance, the dominance share of the total genetic variance, the mean frequency of the trait-increasing allele at a class-A locus, and the e-folding time of the mean trait's autocorrelation in generations; and k_hat. State briefly the model you used and each convention you adopted wherever the configuration leaves a choice open. Your final answer must be a single number: the e-folding time, in generations, of the autocorrelation of the mean trait's deviation from its equilibrium value at omega^-2 = 10^(k_hat/10).


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

01_locus_stationary_moments.py

Goal
----
Stationary moments of the trait-increasing allele frequency at one locus under stabilizing selection with dominance.

```python
def locus_stationary_moments(delta: float, alpha: float, dom: float, theta_plus: float, theta_minus: float,
                             sel_ratio: float, n_nodes: int) -> "np.ndarray":
    '''Stationary moments of the trait-increasing allele frequency at one locus under stabilizing selection with dominance.

    The locus has additive effect alpha and dominance deviation dom, so that the trait
    values of the genotypes carrying 0, 1 and 2 copies of the trait-increasing allele
    are 0, alpha + dom and 2 alpha; mutation is reversible at scaled rates theta_plus
    (towards the trait-increasing allele) and theta_minus; the trait is under Gaussian
    stabilizing selection with scaled selection-drift ratio sel_ratio, and the mean
    trait of the population sits delta above the selection optimum. theta_plus,
    theta_minus and sel_ratio are in the source's scaling of mutation, selection and
    time. The selection coefficient at the locus is the regression coefficient of log
    fitness on the count of the trait-increasing allele under Hardy-Weinberg and
    linkage equilibrium, the other loci entering only through delta, and the allele
    frequency follows the diffusion with that frequency-dependent coefficient. Return
    moments of Wright's stationary distribution of that diffusion, evaluated by
    quadrature after the endpoint singularities of the density have been removed by
    substitution.

    Parameters
    ----------
    delta : float
        Deviation of the mean trait value from the selection optimum, in trait units.
    alpha : float
        Additive effect of the trait-increasing allele, in trait units, > 0.
    dom : float
        Dominance deviation of the heterozygote, in trait units.
    theta_plus : float
        Scaled mutation rate towards the trait-increasing allele, > 0.
    theta_minus : float
        Scaled mutation rate towards the trait-decreasing allele, > 0.
    sel_ratio : float
        Scaled selection-drift ratio, >= 0; 0 is mutation and drift alone.
    n_nodes : int
        Number of quadrature nodes per sub-interval, >= 2; with 400 nodes every
        entry is accurate to a relative error below 1e-10 over the parameter ranges
        of the tests, which compare at a relative tolerance of 1e-7.

    Returns
    -------
    moments : np.ndarray
        Shape (4,): the expectations of p, p(1 - p), beta(p)^2 p(1 - p) and
        (p(1 - p))^2 under the stationary distribution of the frequency p, where
        beta(p) is the average effect of an allele substitution at the locus.

    Raises
    ------
    ValueError
        If delta or dom is not a finite number, if alpha is not a finite number > 0,
        if theta_plus or theta_minus is not a finite number > 0, if sel_ratio is not
        a finite number >= 0, or if n_nodes is not an integer >= 2.
    '''
    return moments  # placeholder
```

### Step 2

02_architecture_mean_and_variances.py

Goal
----
Mean trait value and the additive and dominance components of the genetic variance of a locus-class architecture at a given mean-trait deviation.

```python
def architecture_mean_and_variances(delta: float, sel_ratio: float, alpha: "np.ndarray", dom: "np.ndarray",
                                    theta_plus: "np.ndarray", theta_minus: "np.ndarray", counts: "np.ndarray",
                                    n_nodes: int) -> "np.ndarray":
    '''Mean trait value and the additive and dominance components of the genetic variance of a locus-class architecture at a given mean-trait deviation.

    Class k consists of counts[k] loci with additive effect alpha[k], dominance
    deviation dom[k] and scaled mutation rates theta_plus[k], theta_minus[k]; every
    locus of the class has the stationary frequency distribution of the previous step
    at the deviation delta and the scaled selection-drift ratio sel_ratio. The mean
    trait value is the sum over all loci of the expected trait contribution of a
    locus, and the additive and dominance components of the genetic variance are the
    sums over all loci of the expected additive and dominance variances of a locus
    under Hardy-Weinberg proportions, every expectation taken under the stationary
    distribution of the locus.

    Parameters
    ----------
    delta : float
        Deviation of the mean trait value from the selection optimum, in trait units.
    sel_ratio : float
        Scaled selection-drift ratio, >= 0.
    alpha : np.ndarray
        Shape (K,): additive effects of the K classes, in trait units, each > 0.
    dom : np.ndarray
        Shape (K,): dominance deviations of the heterozygotes, in trait units.
    theta_plus : np.ndarray
        Shape (K,): scaled mutation rates towards the trait-increasing allele, each > 0.
    theta_minus : np.ndarray
        Shape (K,): scaled mutation rates towards the trait-decreasing allele, each > 0.
    counts : np.ndarray
        Shape (K,): numbers of loci in the classes, positive integers.
    n_nodes : int
        Number of quadrature nodes passed to the previous step, >= 2.

    Returns
    -------
    out : np.ndarray
        Shape (3,): the mean trait value, the additive genetic variance and the
        dominance genetic variance of the population, in trait units and squared
        trait units.

    Raises
    ------
    ValueError
        If the five class arrays are not one-dimensional of the same length K >= 1
        with finite entries, if any alpha, theta_plus or theta_minus entry is not > 0,
        if any count is not a positive integer, or on any condition raised by the
        previous step for delta, sel_ratio or n_nodes.
    '''
    return out  # placeholder
```

### Step 3

03_equilibrium_trait_deviation.py

Goal
----
Self-consistent equilibrium deviation of the mean trait value from the selection optimum.

```python
def equilibrium_trait_deviation(sel_ratio: float, eta: float, alpha: "np.ndarray", dom: "np.ndarray", theta_plus: "np.ndarray", theta_minus: "np.ndarray", counts: "np.ndarray", n_nodes: int, tol: float) -> float:
    '''Return the self-consistent equilibrium deviation of the mean trait from the optimum. Use the minimum and maximum of the three genotype values 0, alpha + dom, and 2 alpha in each class to bracket the unique root. Raise ValueError for invalid eta or tol, or for invalid inputs reported by earlier steps.'''
    return delta_star  # placeholder
```

### Step 4

04_neutral_variance_components.py

Goal
----
Additive and dominance components of the genetic variance under mutation and drift alone, in closed form.

```python
def neutral_variance_components(alpha: "np.ndarray", dom: "np.ndarray", theta_plus: "np.ndarray",
                                theta_minus: "np.ndarray", counts: "np.ndarray") -> "np.ndarray":
    '''Additive and dominance components of the genetic variance under mutation and drift alone, in closed form.

    Each locus of class k has additive effect alpha[k], dominance deviation dom[k] and
    scaled mutation rates theta_plus[k], theta_minus[k] in the source's scaling; with
    no selection its frequency has the Beta stationary distribution of reversible
    mutation and drift. Return the sums over all loci of the expected additive and
    dominance variances of a locus under Hardy-Weinberg proportions, using the exact
    moments of that distribution.

    Parameters
    ----------
    alpha : np.ndarray
        Shape (K,): additive effects of the K classes, in trait units, each > 0.
    dom : np.ndarray
        Shape (K,): dominance deviations of the heterozygotes, in trait units.
    theta_plus : np.ndarray
        Shape (K,): scaled mutation rates towards the trait-increasing allele, each > 0.
    theta_minus : np.ndarray
        Shape (K,): scaled mutation rates towards the trait-decreasing allele, each > 0.
    counts : np.ndarray
        Shape (K,): numbers of loci in the classes, positive integers.

    Returns
    -------
    out : np.ndarray
        Shape (2,): the additive and the dominance genetic variance under mutation and
        drift alone, in squared trait units.

    Raises
    ------
    ValueError
        If the five class arrays are not one-dimensional of the same length K >= 1
        with finite entries, if any alpha, theta_plus or theta_minus entry is not > 0,
        or if any count is not a positive integer.
    '''
    return out  # placeholder
```

### Step 5

05_variance_ratio_curve.py

Goal
----
Equilibrium genetic variance relative to its neutral value along a grid of selection strengths.

```python
def variance_ratio_curve(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray", alpha: "np.ndarray",
                         dom: "np.ndarray", counts: "np.ndarray", eta: float, omega_inv2: "np.ndarray",
                         n_nodes: int, tol: float) -> "np.ndarray":
    '''Equilibrium genetic variance relative to its neutral value along a grid of selection strengths.

    The population has n_diploid diploid individuals; class k has counts[k] loci with
    additive effect alpha[k], dominance deviation dom[k] and per-generation mutation
    rates mu_plus[k] (towards the trait-increasing allele) and mu_minus[k]; log fitness
    is Gaussian in the trait with optimum eta and inverse squared width omega_inv2. For
    each entry of omega_inv2, convert the population size, the mutation rates and the
    selection strength to the source's scaled parameters, solve the equilibrium
    deviation of the mean trait as in the earlier steps, and return the total genetic
    variance (additive plus dominance) at that equilibrium divided by the total genetic
    variance under mutation and drift alone.

    Parameters
    ----------
    n_diploid : int
        Number of diploid individuals, >= 1.
    mu_plus : np.ndarray
        Shape (K,): mutation rates per locus per generation towards the
        trait-increasing allele, each > 0.
    mu_minus : np.ndarray
        Shape (K,): mutation rates per locus per generation towards the
        trait-decreasing allele, each > 0.
    alpha : np.ndarray
        Shape (K,): additive effects of the K classes, in trait units, each > 0.
    dom : np.ndarray
        Shape (K,): dominance deviations of the heterozygotes, in trait units.
    counts : np.ndarray
        Shape (K,): numbers of loci in the classes, positive integers.
    eta : float
        Selection optimum, in trait units, finite.
    omega_inv2 : np.ndarray
        Shape (G,): selection strengths, the inverse squared widths of the Gaussian
        fitness function in inverse squared trait units, each >= 0.
    n_nodes : int
        Number of quadrature nodes passed to the earlier steps, >= 2.
    tol : float
        Absolute accuracy of each equilibrium deviation, in trait units, > 0.

    Returns
    -------
    ratio : np.ndarray
        Shape (G,): the equilibrium total genetic variance divided by the neutral
        total genetic variance, one entry per selection strength.

    Raises
    ------
    ValueError
        If n_diploid is not an integer >= 1, if mu_plus or mu_minus is not a
        one-dimensional array of finite numbers > 0 of the same length as the other
        class arrays, if omega_inv2 is not a one-dimensional array of finite numbers
        >= 0 with at least one entry, or on any condition raised by the earlier steps.
    '''
    return ratio  # placeholder
```

### Step 6

06_equilibrium_state_at_strength.py

Goal
----
Equilibrium summaries of the polygenic system at one selection strength, including the relaxation time of the mean trait in generations.

```python
def equilibrium_state_at_strength(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray", alpha: "np.ndarray",
                                  dom: "np.ndarray", counts: "np.ndarray", eta: float, omega_inv2: float,
                                  n_nodes: int, tol: float) -> "np.ndarray":
    '''Equilibrium summaries of the polygenic system at one selection strength, including the relaxation time of the mean trait in generations.

    The population and the architecture are as in the previous step, and omega_inv2 is
    one selection strength. Convert to the source's scaled parameters, solve the
    equilibrium deviation of the mean trait, and return the deviation, the additive
    genetic variance, the share of the total genetic variance that is dominance
    variance, the mean frequency of the trait-increasing allele at a locus of the first
    class, and the e-folding time, in generations, of the autocorrelation of the
    deviation of the mean trait from its equilibrium value under the source's
    Ornstein-Uhlenbeck description of the stationary fluctuations of the mean trait.

    Parameters
    ----------
    n_diploid : int
        Number of diploid individuals, >= 1.
    mu_plus : np.ndarray
        Shape (K,): mutation rates per locus per generation towards the
        trait-increasing allele, each > 0.
    mu_minus : np.ndarray
        Shape (K,): mutation rates per locus per generation towards the
        trait-decreasing allele, each > 0.
    alpha : np.ndarray
        Shape (K,): additive effects of the K classes, in trait units, each > 0.
    dom : np.ndarray
        Shape (K,): dominance deviations of the heterozygotes, in trait units.
    counts : np.ndarray
        Shape (K,): numbers of loci in the classes, positive integers.
    eta : float
        Selection optimum, in trait units, finite.
    omega_inv2 : float
        Selection strength, the inverse squared width of the Gaussian fitness
        function in inverse squared trait units, > 0.
    n_nodes : int
        Number of quadrature nodes passed to the earlier steps, >= 2.
    tol : float
        Absolute accuracy of the equilibrium deviation, in trait units, > 0.

    Returns
    -------
    state : np.ndarray
        Shape (5,): the equilibrium deviation of the mean trait from the optimum (trait
        units), the additive genetic variance (squared trait units), the dominance
        share of the total genetic variance (a fraction), the mean frequency of the
        trait-increasing allele in the first class, and the e-folding time of the
        autocorrelation of the mean trait's deviation, in generations.

    Raises
    ------
    ValueError
        If omega_inv2 is not a finite number > 0, or on any condition raised by the
        earlier steps for the other inputs.
    '''
    return state  # placeholder
```

### Step 7

07_relaxation_time_at_variance_crossing.py

Goal
----
E-folding time of the mean trait's autocorrelation at the first grid strength where the equilibrium genetic variance falls below its neutral value.

```python
def relaxation_time_at_variance_crossing(n_diploid: int, mu_plus: "np.ndarray", mu_minus: "np.ndarray",
                                         alpha: "np.ndarray", dom: "np.ndarray", counts: "np.ndarray", eta: float,
                                         omega_inv2_grid: "np.ndarray", n_nodes: int, tol: float) -> float:
    '''E-folding time of the mean trait's autocorrelation at the first grid strength where the equilibrium genetic variance falls below its neutral value.

    The population and the architecture are as in the earlier steps. Compute the ratio
    of the equilibrium total genetic variance to its neutral value at every entry of
    omega_inv2_grid, in the order given, and select the first entry whose ratio is
    strictly below 1. Return the e-folding time, in generations, of the autocorrelation
    of the deviation of the mean trait from its equilibrium value at that selection
    strength, as given by the previous step.

    Parameters
    ----------
    n_diploid : int
        Number of diploid individuals, >= 1.
    mu_plus : np.ndarray
        Shape (K,): mutation rates per locus per generation towards the
        trait-increasing allele, each > 0.
    mu_minus : np.ndarray
        Shape (K,): mutation rates per locus per generation towards the
        trait-decreasing allele, each > 0.
    alpha : np.ndarray
        Shape (K,): additive effects of the K classes, in trait units, each > 0.
    dom : np.ndarray
        Shape (K,): dominance deviations of the heterozygotes, in trait units.
    counts : np.ndarray
        Shape (K,): numbers of loci in the classes, positive integers.
    eta : float
        Selection optimum, in trait units, finite.
    omega_inv2_grid : np.ndarray
        Shape (G,): the grid of selection strengths, inverse squared widths of the
        Gaussian fitness function in inverse squared trait units, each > 0.
    n_nodes : int
        Number of quadrature nodes passed to the earlier steps, >= 2.
    tol : float
        Absolute accuracy of each equilibrium deviation, in trait units, > 0.

    Returns
    -------
    relaxation_time : float
        The e-folding time of the autocorrelation of the mean trait's deviation at
        the selected strength, in generations, as a native Python float.

    Raises
    ------
    ValueError
        If omega_inv2_grid is not a one-dimensional array of finite numbers > 0 with
        at least one entry, if no entry of the grid gives a variance ratio strictly
        below 1, or on any condition raised by the earlier steps for the other inputs.
    '''
    return relaxation_time  # placeholder
```
