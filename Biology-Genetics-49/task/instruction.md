# Biology-Genetics-49

## Background

Many microbes secrete compounds that benefit every cell nearby, such as siderophores and extracellular enzymes, and genotypes also differ in how much of a shared resource they consume, so a mutation can change how many individuals an environment supports as well as how well its carriers compete. The classical theory of fixation probabilities and fixation times treats population size as fixed, or as changing for reasons external to the population, which ties the strength of genetic drift to time rather than to the genetic composition of the population.

When the total size depends on the frequency of a spreading allele, drift weakens or strengthens as the allele spreads, which changes both the chance that a single new mutant escapes early loss and the time its sweep takes, in ways that depend on how the size responds to the mixture of genotypes. Repeated introductions of single mutant cells measure establishment directly, and the establishment rates of several strains carry information about that response.

## Problem

Many microbes secrete public goods, such as iron-scavenging siderophores, that let a population support more cells, so a mutation that changes a cell's production changes the census size of the whole population as it spreads, and with it the strength of genetic drift. The classical fixation probability of a new mutant, and the classical duration of its sweep, assume a population size that does not depend on the mutant's frequency. In this problem the inputs are an ancestral population, three mutant strains and the outcomes of replicate single-cell introductions of each strain; the output is a single number, the expected duration of the first selective sweep under the size response that the introductions support.

Model an asexual haploid population in the diffusion approximation of the Wright-Fisher model. The ancestral population has 800 cells. A strain has a selective advantage s over the ancestor, and a population fixed for it would have Nm cells; while the strain segregates at frequency p, the census size is a deterministic function N(p) with N(0) = 800 and N(1) = Nm, of the same form for every strain. The number of offspring of a cell has variance 2 in every generation, so genetic drift acts at the effective size N(p)/2, and selection changes p by s p (1 - p) per generation. An introduced cell starts its strain at frequency 1/800. Five forms of N(p) are candidates, listed in this order: the frequency-weighted arithmetic mean (1 - p) 800 + p Nm, the geometric mean 800^(1 - p) Nm^p, the harmonic mean ((1 - p)/800 + p/Nm)^(-1), the root-mean-square mean ((1 - p) 800^2 + p Nm^2)^(1/2), and the symmetric S-shaped response between 800 and Nm that a recent diffusion analysis of fixation probabilities, in populations whose total size depends on the genotypes they contain, uses alongside these four means.

Strain A would reach Nm = 12,800 cells and has s = 0.002, strain B would reach Nm = 200 cells and has s = 0.004, and strain C would reach Nm = 3,200 cells and has s = 0.003. Each strain was introduced 200,000 times as a single cell into the ancestral population, and it fixed 888 times for strain A, 771 times for strain B and 809 times for strain C. Decide the size response by the joint binomial likelihood of the three counts, with each strain's fixation probability taken from the diffusion under that response; take the response with the largest likelihood, the one listed first on a tie. Under the decided response, compute for each strain the fixation probability of a single introduced cell and its expected number of generations to fixation, conditional on fixation. If new cells of the three strains arose at equal rates and rarely enough that two strains never segregate together, the first strain to fix would be drawn in proportion to the fixation probabilities, so the expected duration of the first sweep is the average of the three conditional fixation times weighted by the three fixation probabilities under the decided response. Where a derivation from the model stated here disagrees with a published closed form, the derivation governs.

Report the expected duration, in generations, of the first sweep under the decided response. In the reasoning, report the decided response and the natural-log likelihood margin by which it exceeds the second-best response; the fixation probability and the conditional mean fixation time of each strain under the decided response; and the expected duration of the first sweep that the geometric-mean response would give. State briefly the model you used, including the expression you used for the fixation probability under the decided response, and each convention you adopted wherever the configuration leaves a choice open, and relate your results briefly to what is known about fixation when a mutant changes the size of its population. Your final answer must be a single number: the expected duration of the first sweep, in generations, under the decided response.

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

01_size_response.py

Goal
----
Census size of a population at given mutant frequencies, and its integral over the frequency, for five forms of the size response.

```python
def size_response(frequency: "np.ndarray", ancestral_size: float, mutant_size: float, response: str) -> "np.ndarray":
    '''Census size of a population at given mutant frequencies, and its integral over the frequency, for five forms of the size response.

    A population has census size Na = ancestral_size when the mutant is absent
    and Nm = mutant_size when the mutant is fixed. At mutant frequency p its
    census size N(p) follows `response`:
    "arithmetic", (1 - p) Na + p Nm;
    "geometric", Na^(1 - p) Nm^p;
    "harmonic", ((1 - p)/Na + p/Nm)^(-1);
    "root_mean_square", ((1 - p) Na^2 + p Nm^2)^(1/2);
    "s_shaped", the symmetric S-shaped response of the source,
    Na + (Nm - Na) p^2 / (p^2 + (1 - p)^2), which moves from Na at p = 0 to
    Nm at p = 1 with zero slope at both ends and satisfies
    N(p) + N(1 - p) = Na + Nm.
    When Na equals Nm every response is the constant Na.

    Parameters
    ----------
    frequency : np.ndarray
        Shape (n,), n >= 1; mutant frequencies, each a finite number in [0, 1].
    ancestral_size : float
        Census size Na with the mutant absent, a finite number from 1 to 1e7.
    mutant_size : float
        Census size Nm with the mutant fixed, a finite number from 1 to 1e7.
    response : str
        One of "arithmetic", "geometric", "harmonic", "root_mean_square" and
        "s_shaped".

    Returns
    -------
    size_table : np.ndarray
        Shape (n, 2). size_table[j, 0] is the census size N at frequency[j];
        size_table[j, 1] is the integral of N(p) dp from p = 0 to
        p = frequency[j]. Every entry is accurate to a relative error below
        1e-12.

    Raises
    ------
    ValueError
        If frequency is not a one-dimensional array of at least one finite
        number in [0, 1], if ancestral_size or mutant_size is not a finite
        number from 1 to 1e7, or if response is not one of the five names.
    '''
    return size_table  # placeholder
```

### Step 2

02_fixation_probability.py

Goal
----
Probability that a mutant starting at a given frequency is eventually fixed, in the diffusion approximation for a haploid population whose census size depends on the mutant's frequency.

```python
def fixation_probability(initial_frequency: float, selection: float, ancestral_size: float, mutant_size: float,
                         response: str, offspring_variance: float) -> float:
    '''Probability that a mutant starting at a given frequency is eventually fixed, in the diffusion approximation for a haploid population whose census size depends on the mutant's frequency.

    The population reproduces as a haploid Wright-Fisher population in which
    the number of offspring of an individual has variance offspring_variance,
    so genetic drift acts at the effective size N(p) / offspring_variance,
    where N(p) is the census size of the previous step for `response`,
    ancestral_size and mutant_size at mutant frequency p. Selection changes
    the mutant frequency by selection * p * (1 - p) per generation. Return the
    probability of ultimate fixation of the mutant from initial_frequency, in
    the diffusion approximation.

    Parameters
    ----------
    initial_frequency : float
        Mutant frequency at the start, a finite number with
        0 < initial_frequency <= 0.1.
    selection : float
        Selective advantage of the mutant per generation, a finite number from
        0 to 0.05.
    ancestral_size : float
        Census size with the mutant absent, a finite number from 2 to 1e6.
    mutant_size : float
        Census size with the mutant fixed, a finite number from 2 to 1e6, at
        most 100 times and at least 1/100 of ancestral_size.
    response : str
        One of "arithmetic", "geometric", "harmonic", "root_mean_square" and
        "s_shaped", as in the previous step.
    offspring_variance : float
        Variance of the number of offspring of an individual, a finite number
        from 0.1 to 10. The product
        selection * max(ancestral_size, mutant_size) / offspring_variance must
        not exceed 200.

    Returns
    -------
    probability : float
        The fixation probability, as a native Python float, accurate to a
        relative error below 1e-10.

    Raises
    ------
    ValueError
        If initial_frequency is not a finite number with
        0 < initial_frequency <= 0.1, if selection is not a finite number from
        0 to 0.05, if ancestral_size or mutant_size is not a finite number from
        2 to 1e6, if the larger size exceeds 100 times the smaller, if
        response is not one of the five names, if offspring_variance is not a
        finite number from 0.1 to 10, or if
        selection * max(ancestral_size, mutant_size) / offspring_variance
        exceeds 200.
    '''
    return probability  # placeholder
```

### Step 3

03_introduction_log_likelihood.py

Goal
----
Natural logarithm of the probability of a count of fixations among independent single-copy introductions with a common fixation probability.

```python
def introduction_log_likelihood(fixations: int, introductions: int, probability: float) -> float:
    '''Natural logarithm of the probability of a count of fixations among independent single-copy introductions with a common fixation probability.

    Each of `introductions` independent introductions fixes with probability
    `probability`. Return the natural logarithm of the binomial probability of
    exactly `fixations` fixations, the binomial coefficient included.

    Parameters
    ----------
    fixations : int
        Number of introductions that fixed, an integer from 0 to introductions.
    introductions : int
        Number of independent introductions, an integer >= 1.
    probability : float
        Fixation probability of one introduction, a finite number with
        0 < probability < 1.

    Returns
    -------
    log_likelihood : float
        The natural-log binomial probability, as a native Python float,
        accurate to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If introductions is not an integer >= 1, if fixations is not an
        integer from 0 to introductions, or if probability is not a finite
        number with 0 < probability < 1.
    '''
    return log_likelihood  # placeholder
```

### Step 4

04_response_log_likelihoods.py

Goal
----
Joint natural-log likelihood of the fixation counts of several mutant strains under each of the five size responses.

```python
def response_log_likelihoods(fixations: "np.ndarray", introductions: int, mutant_sizes: "np.ndarray",
                             selections: "np.ndarray", ancestral_size: float, offspring_variance: float) -> "np.ndarray":
    '''Joint natural-log likelihood of the fixation counts of several mutant strains under each of the five size responses.

    Every strain is introduced separately into the same ancestral population,
    of census size ancestral_size, with the offspring-number variance
    offspring_variance of the previous steps. Strain i has the census size
    mutant_sizes[i] when fixed and the selective advantage selections[i]; each
    of its `introductions` independent introductions starts from a single
    cell, at frequency 1 / ancestral_size, and fixations[i] of them fixed. For
    each response, return the sum over the strains of the log-likelihood of
    the strain's count, with the strain's fixation probability under that
    response from the previous steps.

    Parameters
    ----------
    fixations : np.ndarray
        Shape (m,), m >= 1; integer counts of fixations, each from 0 to
        introductions.
    introductions : int
        Number of introductions per strain, an integer >= 1.
    mutant_sizes : np.ndarray
        Shape (m,); census sizes with each strain fixed, within the ranges of
        the previous steps.
    selections : np.ndarray
        Shape (m,); selective advantages of the strains, from 0 to 0.05.
    ancestral_size : float
        Census size of the ancestral population, a finite number from 10 to
        1e6.
    offspring_variance : float
        Variance of the number of offspring of an individual, a finite number
        from 0.1 to 10.

    Returns
    -------
    log_likelihoods : np.ndarray
        Shape (5,). Entry j is the joint log-likelihood under the j-th
        response in the order "arithmetic", "geometric", "harmonic",
        "root_mean_square", "s_shaped"; accurate to a relative error below
        1e-9.

    Raises
    ------
    ValueError
        If fixations, mutant_sizes and selections are not one-dimensional
        arrays of equal length >= 1, if fixations holds a value that is not an
        integer from 0 to introductions, if introductions is not an integer
        >= 1, or if ancestral_size is not a finite number from 10 to 1e6.
        Conditions raised by the earlier steps propagate unchanged, including a
        fixation probability that is not strictly between 0 and 1.
    '''
    return log_likelihoods  # placeholder
```

### Step 5

05_conditional_fixation_time.py

Goal
----
Expected number of generations until a mutant fixes, conditional on its fixation, in the diffusion approximation for a haploid population whose census size depends on the mutant's frequency.

```python
def conditional_fixation_time(initial_frequency: float, selection: float, ancestral_size: float, mutant_size: float,
                              response: str, offspring_variance: float) -> float:
    '''Expected number of generations until a mutant fixes, conditional on its fixation, in the diffusion approximation for a haploid population whose census size depends on the mutant's frequency.

    The population, its census size N(p) for `response`, ancestral_size and
    mutant_size, the effective size N(p) / offspring_variance and the change
    selection * p * (1 - p) per generation are those of the previous steps.
    Starting from initial_frequency, return the mean number of generations
    until the mutant is fixed, averaged over the histories that end in
    fixation, in the diffusion approximation.

    Parameters
    ----------
    initial_frequency : float
        Mutant frequency at the start, a finite number with
        0 < initial_frequency <= 0.1.
    selection : float
        Selective advantage of the mutant per generation, a finite number from
        0 to 0.05.
    ancestral_size : float
        Census size with the mutant absent, a finite number from 2 to 1e6.
    mutant_size : float
        Census size with the mutant fixed, a finite number from 2 to 1e6, at
        most 100 times and at least 1/100 of ancestral_size.
    response : str
        One of "arithmetic", "geometric", "harmonic", "root_mean_square" and
        "s_shaped", as in the first step.
    offspring_variance : float
        Variance of the number of offspring of an individual, a finite number
        from 0.1 to 10. The product
        selection * max(ancestral_size, mutant_size) / offspring_variance must
        not exceed 200.

    Returns
    -------
    generations : float
        The conditional mean time to fixation in generations, as a native
        Python float, accurate to a relative error below 1e-8.

    Raises
    ------
    ValueError
        If initial_frequency is not a finite number with
        0 < initial_frequency <= 0.1, if selection is not a finite number from
        0 to 0.05, if ancestral_size or mutant_size is not a finite number from
        2 to 1e6, if the larger size exceeds 100 times the smaller, if
        response is not one of the five names, if offspring_variance is not a
        finite number from 0.1 to 10, or if
        selection * max(ancestral_size, mutant_size) / offspring_variance
        exceeds 200.
    '''
    return generations  # placeholder
```

### Step 6

06_strain_sweep_table.py

Goal
----
Fixation probability and conditional mean fixation time of a single cell of each of several mutant strains, under one size response.

```python
def strain_sweep_table(mutant_sizes: "np.ndarray", selections: "np.ndarray", ancestral_size: float,
                       offspring_variance: float, response: str) -> "np.ndarray":
    '''Fixation probability and conditional mean fixation time of a single cell of each of several mutant strains, under one size response.

    Every strain is introduced separately, as a single cell at frequency
    1 / ancestral_size, into the ancestral population of census size
    ancestral_size, with the offspring-number variance offspring_variance and
    the diffusion approximation of the previous steps. Strain i has the census
    size mutant_sizes[i] when fixed and the selective advantage selections[i],
    and the census size follows `response`.

    Parameters
    ----------
    mutant_sizes : np.ndarray
        Shape (m,), m >= 1; census sizes with each strain fixed, within the
        ranges of the previous steps.
    selections : np.ndarray
        Shape (m,); selective advantages of the strains, from 0 to 0.05.
    ancestral_size : float
        Census size of the ancestral population, a finite number from 10 to
        1e6.
    offspring_variance : float
        Variance of the number of offspring of an individual, a finite number
        from 0.1 to 10.
    response : str
        One of "arithmetic", "geometric", "harmonic", "root_mean_square" and
        "s_shaped".

    Returns
    -------
    sweep_table : np.ndarray
        Shape (m, 2). sweep_table[i, 0] is the fixation probability of a
        single cell of strain i; sweep_table[i, 1] is its conditional mean
        time to fixation in generations. Entries accurate to a relative error
        below 1e-8.

    Raises
    ------
    ValueError
        If mutant_sizes and selections are not one-dimensional numeric arrays
        of equal length >= 1, or if ancestral_size is not a finite number from
        10 to 1e6. Conditions raised by the earlier steps propagate unchanged.
    '''
    return sweep_table  # placeholder
```

### Step 7

07_first_sweep_duration.py

Goal
----
Expected duration of the first selective sweep among several mutant strains, under the size response their establishment counts support.

```python
def first_sweep_duration(fixations: "np.ndarray", introductions: int, mutant_sizes: "np.ndarray",
                         selections: "np.ndarray", ancestral_size: float, offspring_variance: float) -> float:
    '''Expected duration of the first selective sweep among several mutant strains, under the size response their establishment counts support.

    A haploid ancestral population of census size ancestral_size reproduces
    as a Wright-Fisher population with offspring-number variance
    offspring_variance, in the diffusion approximation of the earlier steps.
    Strain i has the selective advantage selections[i] and the census size
    mutant_sizes[i] when fixed; while a strain segregates, the census size
    follows one of the five responses of the first step, the same for every
    strain. Each strain was introduced `introductions` times, each time as a
    single cell at frequency 1 / ancestral_size, and fixed fixations[i] times.
    Decide the response by the largest joint binomial log-likelihood of the
    counts, the earliest in the order "arithmetic", "geometric", "harmonic",
    "root_mean_square", "s_shaped" on a tie. Under the decided response, with
    u_i the fixation probability of a single cell of strain i and t_i its
    conditional mean time to fixation, return sum_i u_i t_i / sum_i u_i, the
    expected duration of the first sweep when cells of the strains arise at
    equal rates and two strains never segregate together.

    Parameters
    ----------
    fixations : np.ndarray
        Shape (m,), m >= 1; integer counts of fixations, each from 0 to
        introductions.
    introductions : int
        Number of introductions per strain, an integer >= 1.
    mutant_sizes : np.ndarray
        Shape (m,); census sizes with each strain fixed, within the ranges of
        the earlier steps.
    selections : np.ndarray
        Shape (m,); selective advantages of the strains, from 0 to 0.05.
    ancestral_size : float
        Census size of the ancestral population, a finite number from 10 to
        1e6.
    offspring_variance : float
        Variance of the number of offspring of an individual, a finite number
        from 0.1 to 10.

    Returns
    -------
    generations : float
        The expected duration of the first sweep in generations, as a native
        Python float, accurate to a relative error below 1e-8.

    Raises
    ------
    ValueError
        If the inputs violate the conditions of the earlier steps, which
        propagate unchanged: equal-length one-dimensional arrays, integer
        counts from 0 to introductions, the ranges of the sizes, selections and
        offspring variance, selection * max(ancestral_size, mutant_size) /
        offspring_variance at most 200, and fixation probabilities strictly
        between 0 and 1.
    '''
    return generations  # placeholder
```
