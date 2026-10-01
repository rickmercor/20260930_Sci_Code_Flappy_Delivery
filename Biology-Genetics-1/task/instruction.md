# Biology-Genetics-1

## Background

Runs of homozygosity are the visible trace of shared ancestry within one individual's genome: stretches over which both haplotypes carry the same allele at every typed marker because they descend from a common ancestral copy. Their lengths carry information about time, since recombination and new mutations erode long shared segments generation after generation, and their abundance carries information about population size, since a small number of breeders makes recent common ancestry likely. Population geneticists therefore read the length distribution of these runs as a record of demographic history, from long-term population size to bottlenecks and recent inbreeding, and use it in conservation, animal breeding and human population genetics. Two things complicate that reading. Genotyping delivers not the segment of identity itself but its rendering on a finite panel of markers, so the theory that connects observed runs to the underlying genealogy has to account for how the markers are spaced and how often they are heterozygous. And selection at the many sites of a chromosome that carry deleterious mutations speeds up the drift of the neutral positions linked to them, so that the effective size a position experiences depends on how long its ancestry has been associated with the same background, and in a small population the occasional fixation of a deleterious mutation erodes the very fitness variance that drives that extra drift. Positive selection leaves a sharper and more local trace: a favourable mutation that sweeps through the population drags the neutral variation linked to it along, so that near the selected site many individuals share recent ancestry, and the runs of homozygosity on the two sides of a nearby neutral position are distorted differently.

## Problem

Runs of homozygosity are the stretches over which the two haplotypes of a diploid individual carry the same allele at every typed marker, and the share of the genome that lies in runs of a given length class is the standard way to summarise them: long runs point to recent common ancestry, short runs to ancient ancestry. A run is not the underlying segment of identity by descent. That segment ends where recombination, mutation or gene conversion broke the identity along either lineage since the common ancestor, but a genotyping panel sees only its typed markers, so the ends of a run are displaced from the ends of the segment. In this problem the input is the size history of a population, the deleterious mutations its chromosome carries, a favourable mutation sweeping at one site, its genotyping panel and two observations of its runs; the output is a single number, the mean age of the common ancestry behind short runs on one side of a position near the sweep.

Model the two haplotypes of a random individual as a random pair from a randomly mating Wright-Fisher population of diploid individuals whose census N(g) at generation g before the present (g = 1 is the parental generation) is constant within each of three epochs. Along each lineage, in each generation, identity at a position is broken by recombination at a rate of 1 per Morgan and by de novo mutation and gene conversion at their combined rate per Morgan, all as independent Poisson events on an unbounded chromosome. Typed markers lie at random positions with the stated mean spacing d (in Morgans), and each is heterozygous between two random haplotypes with probability H, independently of the others. Construct the observable boundary as follows: on each lineage, in each generation, only the first recombination breakpoint on a side counts, and it is observed at the first heterozygous marker beyond it, a displacement that is exponentially distributed with rate H/d per Morgan and drawn independently for each lineage and generation, whereas mutation and gene-conversion breaks are observed where they occur; on a given side of a focal position the run extends as far as no observable break of any lineage in any generation has occurred, the two sides are independent, and the length of the run is the sum of its two sides. The coverage of a length class is the expected fraction of genome positions of a random individual that lie inside a run whose length is in the class.

The chromosome is under background selection. Deleterious mutations arise along it at a total rate U per chromosome per generation, each with a multiplicative fitness effect s in heterozygotes, and the population of each epoch is at mutation-selection-drift balance. In an infinitely large population the standing genetic variance for fitness would be U s; in a finite one deleterious mutations fix recurrently, at an expected interval of T generations, and each fixation lowers mean fitness by s, so the standing variance is V_w = U s - s/T, with T = (exp(2 s n) - 1)/(2 U s n) and n the asymptotic effective number of haploid genomes at a neutral position of the chromosome, n = 2 N times the limit for old associations of the reduction factor F defined below; V_w, T and n are the unique joint solution of these relations for the census N of the epoch, and every generation of an epoch sees its own epoch's standing variance. The mutational input of fitness variance per generation is V_M = U s^2. A neutral copy at map distance x from a position under selection loses its association with the fitness deviation of its background in each generation by the recombination fraction c(x) between the two positions, given by Haldane's map function c(x) = (1 - exp(-2x))/2, and by the fraction V_M/V_w of the fitness variance that mutation renews, so that its cumulative expected contribution over g generations of association is Q_x(g) = sum from i = 0 to g - 1 of [(1 - c(x))(1 - V_M/V_w)]^i. Under multiplicative fitness the effective size of a neutral position after g generations of association is the census times the reduction factor F(g) = exp(-(2/L) times the integral from 0 to L/2 of V_w Q_x(g)^2 dx), the average over the map distances from the focal position to the positions of a chromosome of map length L. A pair that has not coalesced in any more recent generation coalesces in generation g with probability 1/(2 N(g) F(g)); treat the generations as discrete, with no continuous-time approximation, and count coalescence up to and including 3,000 generations before the present. Where a derivation from the model stated here disagrees with a published approximation or closed form, the derivation governs for every model computation, and no small-displacement approximation is to be used in them; the published closed-form expression enters only where the next paragraph says.

At one focal position a favourable mutation is sweeping. It arose as a single copy A generations before the present, after the bottleneck ended (A is not given), at a site whose recombination fraction with the focal position is r, and individuals carrying one or two copies have fitness 1 + alpha and 1 + 2 alpha relative to non-carriers. With N the census since the bottleneck ended, its expected frequency q_k, k generations after it arose, starts at q_0 = 1/(2N); while q_k < 1/(2N alpha) it gains exactly one copy per generation, q_{k+1} = q_k + 1/(2N), and from the first generation with q_k >= 1/(2N alpha) on it follows q_{k+1} = q_k (1 + alpha + alpha q_k)/(1 + 2 alpha q_k); q_A is its frequency today. For the neutral copies linked to the favourable allele in generation k, let y_t be their expected frequency, and w_t the expected square of that frequency, among the carriers of generation t >= k, with y_k = w_k = 1 and, for t = k to A - 1, y_{t+1} = y_t (1 - (1 - q_t) r) and w_{t+1} = w_t (1 - 2 (1 - q_t) r) + (y_t - w_t)/(2N q_t). The variance of the long-term contributions of generation k is V_k = w_A q_A^2/q_k + (1 - 2 y_A q_A + w_A q_A^2)/(1 - q_k) - 1, and the sweep multiplies the effective size of the focal position at generation g = A - k before the present by 1/(1 + V_{A-g}), for g = 1 to A: there the effective size is N(g) F(g)/(1 + V_{A-g}), with F(g) for the census of generation g, and beyond generation A it is N(g) F(g); pairs coalesce at the focal position by the same discrete rule with that effective size. Only the side of the run at the focal position away from the selected site is considered: recombination on that side leaves the association with the selected site intact, so for a pair that coalesced g generations ago that side follows the one-sided construction above, and the probability that it has a length in a class is the sum over g of the pair's coalescence probability at g times the probability that one side of such a run extends at least the lower end of the class but not its upper end. Take A to be the integer from 1 to E whose model probability for the 1-2 cM class of that side is closest to the observed value (the smaller integer if two are equally close).

The observed coverage of the 2-4 cM class is not reported directly: the study reports instead the constant population size that the published closed-form steady-state expression for the fraction of the genome covered by runs of homozygosity of a given length, for this panel and these break rates and integrated over the 2-4 cM class, returns for its data. Recover the observed coverage of the 2-4 cM class by evaluating that expression at the reported size. The number of generations E since the bottleneck ended is not given either. Compute the model coverage of the 2-4 cM class as a function of E and take E to be the integer from 1 to 200 whose model coverage is closest to the recovered observed coverage (the smaller integer if two are equally close); evaluate everything else at that E.

Use this configuration:
* Size history: 8,000 breeding individuals in every generation before the bottleneck; 150 during the bottleneck, which lasted exactly 35 generations; 600 in every generation since it ended. Generations 1 to E before the present had 600 breeders, generations E + 1 to E + 35 had 150, and all earlier generations had 8,000.
* Chromosome: map length L = 1 Morgan, at 1 cM per Mb throughout.
* Background selection: U = 0.5 deleterious mutations per chromosome per generation, each with s = 0.02.
* Sweep: heterozygous advantage alpha = 0.3 (0.6 for two copies), at recombination fraction r = 0.01 from the focal position.
* Genotyping panel: typed markers 100 kb apart on average, with heterozygosity 0.30 at the typed markers.
* Breaks of identity: de novo mutation at 1.2 x 10^-8 per base pair per generation and gene-conversion breaks at 3 x 10^-9 per base pair per generation, on each lineage.
* Observations: for runs of homozygosity of 2 to 4 cM the study reports a constant population size of 199.753 breeding individuals, obtained from the published closed-form steady-state expression described above; at the focal position, the fraction of individuals whose run extends 1 to 2 cM on the side away from the selected site is 0.0792658.
* Length classes, in centiMorgans: 1 to 2, 2 to 4, and 4 and longer for whole runs; 0.5 to 1 and 1 to 2 for the side of the run away from the selected site.

Report the mean coalescence generation of the pairs whose run at the focal position extends 0.5 to 1 cM on the side away from the selected site. In the reasoning, report the observed coverage of the 2-4 cM class implied by the reported size; the expected number of generations between fixations of deleterious mutations at the bottleneck census and the standing genetic variance for fitness there; the reduction factor F after 30 generations of association and its limit for old associations, both at the bottleneck census; E; the effective size of a neutral position at generation E; the genome-wide coverage of the 1-2 cM class and of runs of 4 cM and longer; the mean coalescence generation of the positions inside 2-4 cM runs and inside 1-2 cM runs, genome-wide; the fraction of the 1-2 cM coverage due to coalescence before the bottleneck began; A; the expected frequency of the favourable allele today; the factor by which the sweep multiplies the effective size of the focal position in the generation the mutation arose; the probability that the side of the run away from the selected site is 0.5 to 1 cM, and the fraction of that probability due to coalescence in generations 1 to A. State the condition under which the published closed-form expression applies, and whether a single constant effective size could reproduce the length-class coverages of a chromosome under background selection, with the reason. State briefly the model you used and each convention you adopted wherever the configuration leaves a choice open. Your final answer must be a single number: the mean coalescence generation, in generations before the present, of the pairs whose run at the focal position extends 0.5 to 1 cM on the side away from the selected site, at the recovered E and A.

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

Implement **all 13 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_flank_survival_and_density.py

Goal
----
Survival function and density of one side of the run of homozygosity around a focal position.

```python
def flank_survival_and_density(x: "np.ndarray", t: int, break_rate: float, marker_spacing: float,
                               heterozygosity: float) -> "np.ndarray":
    '''Survival function and density of one side of the run of homozygosity around a focal position.

    Two haplotypes of a randomly mating population coalesced at the focal
    position t generations ago. Along each lineage, in each generation, identity
    is broken by recombination at rate 1 per Morgan and by de novo mutation and
    gene conversion at a combined rate of break_rate per Morgan, all as
    independent Poisson events on an unbounded chromosome. Typed markers lie at
    random along the chromosome with mean spacing marker_spacing, and each is
    heterozygous between two random haplotypes with probability heterozygosity,
    independently of the others. On each lineage, in each generation, only the
    first recombination breakpoint on a side counts: it is observed at the first
    heterozygous marker beyond it, a displacement that is exponentially
    distributed with rate heterozygosity/marker_spacing per Morgan and drawn
    independently for each lineage and generation, whereas mutation and
    gene-conversion breaks are observed where they occur. The run extends on a
    side as far as no observable break of any lineage in any generation has
    occurred. For each distance in x, return the probability that the run
    extends at least that far beyond the focal position on a given side, and
    the density of that side's length at that distance.

    Parameters
    ----------
    x : np.ndarray
        Shape (n,): distances from the focal position in Morgans, each >= 0.
    t : int
        Number of generations since the two lineages coalesced, >= 1.
    break_rate : float
        Combined rate of mutation and gene-conversion breaks per Morgan per
        generation along one lineage, >= 0.
    marker_spacing : float
        Mean spacing of the typed markers in Morgans, in [0, heterozygosity).
        With marker_spacing = 0 every break is observed where it occurs.
    heterozygosity : float
        Probability that a typed marker is heterozygous between two random
        haplotypes, in (0, 1].

    Returns
    -------
    out : np.ndarray
        Shape (2, n). out[0, i] is the probability that the run extends at least
        x[i] Morgans on the given side; out[1, i] is the density of that side's
        length at x[i], per Morgan. Every entry is accurate to an absolute error
        below 1e-12.

    Raises
    ------
    ValueError
        If x is not a one-dimensional array of finite numbers >= 0 with at least
        one entry, if t is not an integer >= 1, if break_rate is not a finite
        number >= 0, if heterozygosity is not a finite number in (0, 1], or if
        marker_spacing is not a finite number in [0, heterozygosity).
    '''
    return out  # placeholder
```

### Step 2

02_class_mass_given_time.py

Goal
----
Probability that the run of homozygosity around a focal position has a length in a given class, for each coalescence time.

```python
def class_mass_given_time(lower: float, upper: float, t: "np.ndarray", break_rate: float,
                          marker_spacing: float, heterozygosity: float) -> "np.ndarray":
    '''Probability that the run of homozygosity around a focal position has a length in a given class, for each coalescence time.

    The run is made of its two sides, which are independent given the
    coalescence time and are each distributed as in the previous step; its
    length is the sum of the two side lengths. For each entry of t, return the
    probability that the length lies in the class [lower, upper] Morgans.

    Parameters
    ----------
    lower : float
        Lower end of the length class in Morgans, >= 0.
    upper : float
        Upper end of the length class in Morgans, > lower; may be infinite.
    t : np.ndarray
        Shape (k,): numbers of generations since coalescence, integers >= 1.
    break_rate : float
        Combined rate of mutation and gene-conversion breaks per Morgan per
        generation along one lineage, >= 0.
    marker_spacing : float
        Mean spacing of the typed markers in Morgans, in [0, heterozygosity).
    heterozygosity : float
        Probability that a typed marker is heterozygous between two random
        haplotypes, in (0, 1].

    Returns
    -------
    mass : np.ndarray
        Shape (k,): mass[j] is the probability that the run's length lies in the
        class, given a coalescence t[j] generations ago. Every entry is accurate
        to an absolute error below 1e-12.

    Raises
    ------
    ValueError
        If lower is not a finite number >= 0, if upper is not a number > lower
        (infinite allowed), if t is not a one-dimensional array of integers >= 1
        with at least one entry, or on any condition raised by the previous step
        for break_rate, marker_spacing and heterozygosity.
    '''
    return mass  # placeholder
```

### Step 3

03_fitness_variance_under_ratchet.py

Goal
----
Standing genetic variance for fitness at mutation-selection-drift balance, reduced by the recurrent fixation of deleterious mutations.

```python
def fitness_variance_under_ratchet(census: float, mutation_rate: float, selection: float,
                                   chromosome_length: float) -> float:
    '''Standing genetic variance for fitness at mutation-selection-drift balance, reduced by the recurrent fixation of deleterious mutations.

    A randomly mating Wright-Fisher population of census diploid individuals
    (2 * census haploid genomes) carries a chromosome of map length
    chromosome_length Morgans on which deleterious mutations arise at a total
    rate of mutation_rate per chromosome per generation, each with a
    multiplicative fitness effect of selection in heterozygotes. Write U for
    mutation_rate, s for selection, L for chromosome_length and N for census.
    In an infinitely large population the standing genetic variance for
    relative fitness would be U s. In the finite population deleterious
    mutations fix at an expected interval of T generations, each fixation
    lowers mean fitness by s, and the decline of mean fitness per generation,
    -s/T, equals the loss through new mutations, -U s, plus the response to
    selection, which is the variance itself, so the standing variance is
    V_w = U s - s/T. The interval between fixations is
    T = (exp(2 s n) - 1) / (2 U s n), where n is the asymptotic effective
    number of haploid genomes at a neutral position of the chromosome. That
    asymptotic number is n = 2 N exp(-V_w * Qbar), where
    Qbar = (2/L) * integral from 0 to L/2 of Q(x)^2 dx,
    Q(x) = 1 / (1 - (1 - c(x)) (1 - V_M/V_w)) is the cumulative effect of an
    association of unlimited age with a position at map distance x,
    c(x) = (1 - exp(-2 x))/2 is Haldane's recombination fraction, and
    V_M = U s^2 is the mutational input of fitness variance per generation.
    The three relations determine n, T and V_w uniquely: as n increases T
    increases, V_w increases towards U s, V_M/V_w decreases, Q(x) and Qbar
    increase, and 2 N exp(-V_w Qbar) decreases, so n = 2 N exp(-V_w Qbar) has
    exactly one solution in (0, 2 N), and V_w is positive there because
    T > 1/U for every n > 0. Solve for n to a relative precision of 1e-12 or
    better, evaluating s/T in the form 2 U s^2 n exp(-2 s n) / (1 - exp(-2 s n))
    so that large arguments do not overflow, and return V_w.

    Parameters
    ----------
    census : float
        Number of breeding diploid individuals, >= 1.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.

    Returns
    -------
    variance : float
        The standing genetic variance for relative fitness V_w, in
        (0, mutation_rate * selection], as a native Python float accurate to a
        relative error below 1e-11.

    Raises
    ------
    ValueError
        If census is not a finite number >= 1, if mutation_rate is not a finite
        number > 0, if selection is not a finite number in (0, 1), or if
        chromosome_length is not a finite number > 0.
    '''
    return variance  # placeholder
```

### Step 4

04_selection_reduction_factor.py

Goal
----
Factor by which background selection reduces the effective size of a neutral position after a given number of generations of association.

```python
def selection_reduction_factor(generations: "np.ndarray", mutation_rate: float, selection: float,
                               chromosome_length: float, census: float) -> "np.ndarray":
    '''Factor by which background selection reduces the effective size of a neutral position after a given number of generations of association.

    Deleterious mutations arise along a chromosome of map length
    chromosome_length Morgans at a total rate of mutation_rate per chromosome
    per generation, each with a multiplicative fitness effect of selection in
    heterozygotes, in a population of census breeding diploid individuals at
    mutation-selection-drift balance, where the standing genetic variance for
    fitness V_w is the value of the previous step for this census (the
    infinite-population value mutation_rate * selection reduced by the
    recurrent fixation of deleterious mutations) and the mutational input of
    variance per generation is V_M = mutation_rate * selection^2. A neutral copy at map distance x Morgans from a position under
    selection loses its association with the fitness deviation of its
    background in each generation by the recombination fraction c(x) between the
    two positions, given by Haldane's map function c(x) = (1 - exp(-2 x))/2, and
    by the fraction V_M/V_w of the fitness variance that mutation renews, so its
    cumulative expected contribution over g generations of association is
    Q_x(g) = sum from i = 0 to g - 1 of [(1 - c(x)) (1 - V_M/V_w)]^i. Under
    multiplicative fitness the effective size after g generations of
    association is the census size times
    F(g) = exp(-(2/chromosome_length) * integral from 0 to chromosome_length/2
    of V_w * Q_x(g)^2 dx), the average over the map distances from the focal
    position to the positions of a chromosome of that length. Return F(g) for
    each entry of generations.

    Parameters
    ----------
    generations : np.ndarray
        Shape (k,): numbers of generations of association, integers >= 1.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.
    census : float
        Number of breeding diploid individuals whose standing variance
        applies, >= 1.

    Returns
    -------
    factor : np.ndarray
        Shape (k,): factor[j] is F(generations[j]), in (0, 1]. Every entry is
        accurate to an absolute error below 1e-12.

    Raises
    ------
    ValueError
        If generations is not a one-dimensional array of integers >= 1 with at
        least one entry, if mutation_rate is not a finite number > 0, if
        selection is not a finite number in (0, 1), if chromosome_length is
        not a finite number > 0, or if census is not a finite number >= 1.
    '''
    return factor  # placeholder
```

### Step 5

05_coalescence_weights.py

Goal
----
Probability that a random pair of haplotypes first coalesces at each generation before the present.

```python
def coalescence_weights(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                        since_end: int, mutation_rate: float, selection: float, chromosome_length: float, horizon: int) -> "np.ndarray":
    '''Probability that a random pair of haplotypes first coalesces at each generation before the present.

    The population is a randomly mating Wright-Fisher population of diploid
    individuals. Generation g before the present (g = 1 is the parents of the
    present generation) has N(g) breeding individuals, that is 2 N(g) haploid
    genomes, with N(g) = n_recent for 1 <= g <= since_end, N(g) = n_bottleneck
    for since_end < g <= since_end + bottleneck_length, and N(g) = n_ancestral
    for larger g. Background selection reduces the effective size of the
    neutral position at generation g to N(g) F(g), where F(g) is the reduction
    factor of the previous step after g generations of association for the
    census N(g) of that generation (so each epoch's census sets its own
    standing variance), for the given mutation_rate, selection and
    chromosome_length. Two haplotypes sampled
    in the present that have not coalesced in any generation more recent than g
    coalesce in generation g with probability 1 / (2 N(g) F(g)), independently
    across generations and without any continuous-time approximation.

    Parameters
    ----------
    n_ancestral : float
        Number of breeding individuals before the bottleneck, >= 1.
    n_bottleneck : float
        Number of breeding individuals during the bottleneck, >= 1.
    n_recent : float
        Number of breeding individuals since the bottleneck ended, >= 1.
    bottleneck_length : int
        Number of generations the bottleneck lasted, >= 0.
    since_end : int
        Number of generations since the bottleneck ended, >= 0.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.
    horizon : int
        Last generation before the present that is included, >= 1.

    Returns
    -------
    weights : np.ndarray
        Shape (horizon,): weights[g - 1] is the probability that the pair first
        coalesces exactly g generations before the present, for g = 1 to
        horizon. The entries sum to less than 1 when coalescence beyond the
        horizon has positive probability.

    Raises
    ------
    ValueError
        If any of the three sizes is not a finite number >= 1, if
        bottleneck_length or since_end is not an integer >= 0, if horizon is
        not an integer >= 1, or on any condition raised by the previous step for
        mutation_rate, selection and chromosome_length.
    '''
    return weights  # placeholder
```

### Step 6

06_class_coverage_from_reported_size.py

Goal
----
Coverage of a length class given by the source's closed-form steady-state expression at a constant population size.

```python
def class_coverage_from_reported_size(reported_size: float, lower: float, upper: float, break_rate: float,
                                      marker_spacing: float, heterozygosity: float) -> float:
    '''Coverage of a length class given by the source's closed-form steady-state expression at a constant population size.

    The expression is the source's closed form for the fraction of the genome
    covered by runs of homozygosity of length x, per Morgan of run length, in a
    randomly mating population of constant size N breeding individuals:
    4 x (1 + m)^2 / (N (2 x (1 + m) + 1/(2 N) - 4 d/H)^3), with m the combined
    rate of mutation and gene-conversion breaks per Morgan per generation on one
    lineage, d the mean marker spacing in Morgans and H the heterozygosity of
    the typed markers. It holds for lengths well above the mean displacement
    d/H of the observable boundaries, and it requires
    2 lower (1 + m) + 1/(2 N) - 4 d/H > 0. Return that expression integrated
    over the class [lower, upper] Morgans, with N = reported_size,
    m = break_rate, d = marker_spacing and H = heterozygosity.

    Parameters
    ----------
    reported_size : float
        Constant number of breeding individuals, >= 1.
    lower : float
        Lower end of the length class in Morgans, > 0.
    upper : float
        Upper end of the length class in Morgans, > lower; may be infinite.
    break_rate : float
        Combined rate of mutation and gene-conversion breaks per Morgan per
        generation along one lineage, >= 0.
    marker_spacing : float
        Mean spacing of the typed markers in Morgans, in [0, heterozygosity).
    heterozygosity : float
        Probability that a typed marker is heterozygous between two random
        haplotypes, in (0, 1].

    Returns
    -------
    coverage : float
        The closed-form coverage of the class, a fraction of the genome, as a
        native Python float.

    Raises
    ------
    ValueError
        If reported_size is not a finite number >= 1, if lower is not a finite
        number > 0, if upper is not a number > lower (infinite allowed), if
        break_rate is not a finite number >= 0, if heterozygosity is not a
        finite number in (0, 1], if marker_spacing is not a finite number in
        [0, heterozygosity), or if the expression does not apply to the class
        because lower is not large enough relative to the mean displacement.
    '''
    return coverage  # placeholder
```

### Step 7

07_bottleneck_end_from_class_coverage.py

Goal
----
Number of generations since the bottleneck ended whose model coverage of a length class is closest to an observed value.

```python
def bottleneck_end_from_class_coverage(observed: float, lower: float, upper: float, n_ancestral: float,
                                       n_bottleneck: float, n_recent: float, bottleneck_length: int, mutation_rate: float, selection: float, chromosome_length: float, break_rate: float,
                                       marker_spacing: float, heterozygosity: float, horizon: int,
                                       max_since_end: int) -> int:
    '''Number of generations since the bottleneck ended whose model coverage of a length class is closest to an observed value.

    For each candidate number of generations since the end of the bottleneck,
    from 1 to max_since_end, the model coverage of the class [lower, upper] is
    the expected fraction of genome positions of a random individual that lie
    inside a run of homozygosity whose length is in the class: the sum over
    coalescence generations g = 1 to horizon of the first-coalescence
    probability of the coalescence-weights step at g for that candidate, times
    the class probability of the class-mass step for a pair that coalesced g
    generations ago. Return the candidate whose model coverage is closest to
    observed; if two candidates are equally close, return the smaller.

    Parameters
    ----------
    observed : float
        Observed coverage of the class, a fraction of the genome, in [0, 1].
    lower : float
        Lower end of the length class in Morgans, >= 0.
    upper : float
        Upper end of the length class in Morgans, > lower; may be infinite.
    n_ancestral : float
        Number of breeding individuals before the bottleneck, >= 1.
    n_bottleneck : float
        Number of breeding individuals during the bottleneck, >= 1.
    n_recent : float
        Number of breeding individuals since the bottleneck ended, >= 1.
    bottleneck_length : int
        Number of generations the bottleneck lasted, >= 0.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.
    break_rate : float
        Combined rate of mutation and gene-conversion breaks per Morgan per
        generation along one lineage, >= 0.
    marker_spacing : float
        Mean spacing of the typed markers in Morgans, in [0, heterozygosity).
    heterozygosity : float
        Probability that a typed marker is heterozygous between two random
        haplotypes, in (0, 1].
    horizon : int
        Last coalescence generation before the present that is counted, >= 1.
    max_since_end : int
        Largest candidate number of generations since the bottleneck ended,
        >= 1.

    Returns
    -------
    since_end : int
        The selected number of generations since the bottleneck ended, as a
        native Python int.

    Raises
    ------
    ValueError
        If observed is not a finite number in [0, 1], if max_since_end is not
        an integer >= 1, or on any condition raised by the previous steps for
        the other inputs.
    '''
    return since_end  # placeholder
```

### Step 8

08_sweep_frequency_trajectory.py

Goal
----
Expected frequency of a favourable mutation in each generation since it arose as a single copy.

```python
def sweep_frequency_trajectory(census: float, advantage: float, generations: int) -> "np.ndarray":
    '''Expected frequency of a favourable mutation in each generation since it arose as a single copy.

    A randomly mating population has census diploid individuals, that is 2 *
    census haploid genomes, in every generation. A favourable mutation arises
    as a single copy, so its frequency in the generation it arises is
    q_0 = 1 / (2 * census). Individuals carrying zero, one and two copies have
    relative fitness 1, 1 + advantage and 1 + 2 * advantage. Its expected
    trajectory has two phases. While q_k < 1 / (2 * census * advantage), the
    number of copies grows by exactly one per generation,
    q_{k+1} = q_k + 1 / (2 * census). From the first generation with
    q_k >= 1 / (2 * census * advantage) on, the frequency follows the
    deterministic selection recursion
    q_{k+1} = q_k * (1 + advantage + advantage * q_k) / (1 + 2 * advantage * q_k).
    Return q_k for k = 0 to generations, k counting the generations since the
    mutation arose.

    Parameters
    ----------
    census : float
        Number of breeding diploid individuals, >= 1.
    advantage : float
        Fitness advantage of a heterozygous carrier, > 0.
    generations : int
        Number of generations since the mutation arose, >= 0.

    Returns
    -------
    q : np.ndarray
        Shape (generations + 1,): q[k] is the expected frequency of the
        favourable mutation k generations after it arose, in (0, 1).

    Raises
    ------
    ValueError
        If census is not a finite number >= 1, if advantage is not a finite
        number > 0, if generations is not an integer >= 0, or if the expected
        frequency reaches 1 within the requested generations.
    '''
    return q  # placeholder
```

### Step 9

09_sweep_size_reduction.py

Goal
----
Factor by which a sweeping favourable mutation reduces the effective size of a linked neutral position in each past generation.

```python
def sweep_size_reduction(frequencies: "np.ndarray", census: float, recombination: float) -> "np.ndarray":
    '''Factor by which a sweeping favourable mutation reduces the effective size of a linked neutral position in each past generation.

    frequencies[k] is the expected frequency q_k of a favourable mutation k
    generations after it arose, for k = 0 to A, where generation A is the
    present; census is the number N of diploid individuals, constant over the
    sweep, and recombination is the recombination fraction r between the
    selected site and the neutral position. For each generation k from 0 to
    A - 1, consider the neutral copies that are linked to the favourable allele
    in generation k. Let y_t be their expected frequency, and w_t the expected
    square of their frequency, among the chromosomes that carry the favourable
    allele in generation t >= k, with y_k = w_k = 1. For t = k to A - 1,
    y_{t+1} = y_t * (1 - (1 - q_t) * r) and
    w_{t+1} = w_t * (1 - 2 * (1 - q_t) * r) + (y_t - w_t) / (2 * N * q_t),
    the last term being the random sampling of copies among the carriers. The
    variance of the expected long-term contributions of the neutral copies of
    generation k to the present is
    V_k = w_A * q_A^2 / q_k + (1 - 2 * y_A * q_A + w_A * q_A^2) / (1 - q_k) - 1,
    and the sweep multiplies the effective size of the neutral position in
    generation k by 1 / (1 + V_k). Return these factors by the number of
    generations before the present, g = A - k, for g = 1 to A.

    Parameters
    ----------
    frequencies : np.ndarray
        Shape (A + 1,) with A >= 1: the expected frequencies q_0 to q_A, each
        in (0, 1).
    census : float
        Number of breeding diploid individuals, >= 1.
    recombination : float
        Recombination fraction between the selected site and the neutral
        position, in [0, 0.5].

    Returns
    -------
    factor : np.ndarray
        Shape (A,): factor[g - 1] = 1 / (1 + V_{A-g}) for g = 1 to A, the
        factor for the generation g generations before the present; each entry
        is positive.

    Raises
    ------
    ValueError
        If frequencies is not a one-dimensional array of at least two finite
        numbers in (0, 1), if census is not a finite number >= 1, or if
        recombination is not a finite number in [0, 0.5].
    '''
    return factor  # placeholder
```

### Step 10

10_focal_coalescence_weights.py

Goal
----
Probability that a random pair of haplotypes first coalesces at each generation before the present, at a neutral position linked to a sweeping favourable mutation.

```python
def focal_coalescence_weights(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                              since_end: int, mutation_rate: float, selection: float, chromosome_length: float,
                              advantage: float, recombination: float, sweep_age: int, horizon: int) -> "np.ndarray":
    '''Probability that a random pair of haplotypes first coalesces at each generation before the present, at a neutral position linked to a sweeping favourable mutation.

    Generation g before the present (g = 1 is the parents of the present
    generation) has N(g) breeding individuals, with N(g) = n_recent for
    1 <= g <= since_end, N(g) = n_bottleneck for
    since_end < g <= since_end + bottleneck_length, and N(g) = n_ancestral for
    larger g. Background selection multiplies the effective size of the
    neutral position at generation g by F(g), the factor of the
    reduction-factor step after g generations of association for the census
    N(g). A favourable mutation arose as a single copy sweep_age generations
    before the present, after the bottleneck ended, at a site at recombination
    fraction recombination from the neutral position; its expected trajectory
    is that of the trajectory step for census n_recent and the given advantage
    over sweep_age generations, and the sweep multiplies the effective size at
    generation g by S(g), the factor of the sweep size-reduction step for that
    trajectory, census n_recent and recombination fraction, for
    1 <= g <= sweep_age, and by S(g) = 1 for larger g. The effective size of
    the neutral position at generation g is N(g) * F(g) * S(g). Two haplotypes
    sampled in the present that have not coalesced in any generation more
    recent than g coalesce in generation g with probability
    1 / (2 * N(g) * F(g) * S(g)), independently across generations and without
    any continuous-time approximation.

    Parameters
    ----------
    n_ancestral : float
        Number of breeding individuals before the bottleneck, >= 1.
    n_bottleneck : float
        Number of breeding individuals during the bottleneck, >= 1.
    n_recent : float
        Number of breeding individuals since the bottleneck ended, >= 1.
    bottleneck_length : int
        Number of generations the bottleneck lasted, >= 0.
    since_end : int
        Number of generations since the bottleneck ended, >= 1.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.
    advantage : float
        Fitness advantage of a heterozygous carrier of the favourable
        mutation, > 0.
    recombination : float
        Recombination fraction between the selected site and the neutral
        position, in [0, 0.5].
    sweep_age : int
        Number of generations since the favourable mutation arose, an integer
        in [1, since_end].
    horizon : int
        Last generation before the present that is included, >= 1.

    Returns
    -------
    weights : np.ndarray
        Shape (horizon,): weights[g - 1] is the probability that the pair
        first coalesces exactly g generations before the present, for g = 1 to
        horizon.

    Raises
    ------
    ValueError
        If n_ancestral, n_bottleneck or n_recent is not a finite number >= 1,
        if bottleneck_length is not an integer >= 0, if since_end is not an
        integer >= 1, if sweep_age is not an integer in [1, since_end], if
        horizon is not an integer >= 1, if the coalescence probability of some
        generation exceeds 1, or on any condition raised by the previous steps
        for the other inputs.
    '''
    return weights  # placeholder
```

### Step 11

11_far_side_class_profile.py

Goal
----
Probability, mean coalescence time and sweep share of a length class of the side of the run of homozygosity away from the selected site.

```python
def far_side_class_profile(lower: float, upper: float, n_ancestral: float, n_bottleneck: float, n_recent: float,
                           bottleneck_length: int, since_end: int, mutation_rate: float, selection: float,
                           chromosome_length: float, advantage: float, recombination: float, sweep_age: int,
                           break_rate: float, marker_spacing: float, heterozygosity: float,
                           horizon: int) -> "np.ndarray":
    '''Probability, mean coalescence time and sweep share of a length class of the side of the run of homozygosity away from the selected site.

    At a neutral position linked to a sweeping favourable mutation, consider
    only the side of the run of homozygosity away from the selected site. For
    a pair of haplotypes that coalesced g generations ago, the probability
    that this side extends at least a distance x is S_g(x), the one-side
    survival of the flank step for t = g with the given break rate, marker
    spacing and heterozygosity, independently of the sweep; the pair's first
    coalescence generation g has the probabilities of the focal-weights step
    for the given history, background selection and sweep. The probability
    that this side of the run of a random individual has a length in
    [lower, upper] Morgans is the sum over g = 1 to horizon of the
    first-coalescence probability at g times S_g(lower) - S_g(upper), with
    S_g(upper) = 0 for an infinite upper. Return that probability; the mean
    coalescence generation of the pairs it counts, each generation weighted by
    its contribution to the probability; and the fraction of the probability
    due to coalescence in generations 1 to sweep_age, during the sweep.

    Parameters
    ----------
    lower : float
        Lower end of the length class in Morgans, >= 0.
    upper : float
        Upper end of the length class in Morgans, > lower; may be infinite.
    n_ancestral : float
        Number of breeding individuals before the bottleneck, >= 1.
    n_bottleneck : float
        Number of breeding individuals during the bottleneck, >= 1.
    n_recent : float
        Number of breeding individuals since the bottleneck ended, >= 1.
    bottleneck_length : int
        Number of generations the bottleneck lasted, >= 0.
    since_end : int
        Number of generations since the bottleneck ended, >= 1.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.
    advantage : float
        Fitness advantage of a heterozygous carrier of the favourable
        mutation, > 0.
    recombination : float
        Recombination fraction between the selected site and the neutral
        position, in [0, 0.5].
    sweep_age : int
        Number of generations since the favourable mutation arose, an integer
        in [1, since_end].
    break_rate : float
        Combined rate of mutation and gene-conversion breaks per Morgan per
        generation along one lineage, >= 0.
    marker_spacing : float
        Mean spacing of the typed markers in Morgans, in [0, heterozygosity).
    heterozygosity : float
        Probability that a typed marker is heterozygous between two random
        haplotypes, in (0, 1].
    horizon : int
        Last coalescence generation before the present that is counted, >= 1.

    Returns
    -------
    profile : np.ndarray
        Shape (3,): profile[0] is the probability that the side away from the
        selected site has a length in the class, profile[1] the mean
        coalescence generation of the pairs counted in it, and profile[2] the
        fraction of the probability from coalescence in generations 1 to
        sweep_age.

    Raises
    ------
    ValueError
        If lower and upper do not satisfy 0 <= lower < upper with lower
        finite, if the probability of the class within the horizon is zero, or
        on any condition raised by the previous steps for the other inputs.
    '''
    return profile  # placeholder
```

### Step 12

12_sweep_age_from_far_side.py

Goal
----
Number of generations since a favourable mutation arose whose model far-side class probability is closest to an observed value.

```python
def sweep_age_from_far_side(observed: float, lower: float, upper: float, n_ancestral: float, n_bottleneck: float,
                            n_recent: float, bottleneck_length: int, since_end: int, mutation_rate: float,
                            selection: float, chromosome_length: float, advantage: float, recombination: float,
                            break_rate: float, marker_spacing: float, heterozygosity: float, horizon: int) -> int:
    '''Number of generations since a favourable mutation arose whose model far-side class probability is closest to an observed value.

    The favourable mutation arose after the bottleneck ended, so the candidate
    numbers of generations since it arose are the integers from 1 to
    since_end. For each candidate, the model probability that the side of the
    run of homozygosity away from the selected site has a length in
    [lower, upper] Morgans is the first entry of the far-side profile step for
    that candidate and the other inputs. Return the candidate whose model
    probability is closest to observed; if two candidates are equally close,
    return the smaller.

    Parameters
    ----------
    observed : float
        Observed probability of the class, in [0, 1].
    lower : float
        Lower end of the length class in Morgans, >= 0.
    upper : float
        Upper end of the length class in Morgans, > lower; may be infinite.
    n_ancestral : float
        Number of breeding individuals before the bottleneck, >= 1.
    n_bottleneck : float
        Number of breeding individuals during the bottleneck, >= 1.
    n_recent : float
        Number of breeding individuals since the bottleneck ended, >= 1.
    bottleneck_length : int
        Number of generations the bottleneck lasted, >= 0.
    since_end : int
        Number of generations since the bottleneck ended, >= 1.
    mutation_rate : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length : float
        Map length of the chromosome in Morgans, > 0.
    advantage : float
        Fitness advantage of a heterozygous carrier of the favourable
        mutation, > 0.
    recombination : float
        Recombination fraction between the selected site and the neutral
        position, in [0, 0.5].
    break_rate : float
        Combined rate of mutation and gene-conversion breaks per Morgan per
        generation along one lineage, >= 0.
    marker_spacing : float
        Mean spacing of the typed markers in Morgans, in [0, heterozygosity).
    heterozygosity : float
        Probability that a typed marker is heterozygous between two random
        haplotypes, in (0, 1].
    horizon : int
        Last coalescence generation before the present that is counted, >= 1.

    Returns
    -------
    sweep_age : int
        The selected number of generations since the favourable mutation
        arose, as a native Python int.

    Raises
    ------
    ValueError
        If observed is not a finite number in [0, 1], if since_end is not an
        integer >= 1, or on any condition raised by the previous steps for the
        other inputs.
    '''
    return sweep_age  # placeholder
```

### Step 13

13_swept_focal_mean_coalescence_time.py

Goal
----
Mean coalescence time of the pairs whose run extends a reported length on the side of a swept neutral position away from the selected site, at the recovered bottleneck end and sweep age.

```python
def swept_focal_mean_coalescence_time(n_ancestral: float, n_bottleneck: float, n_recent: float, bottleneck_length: int,
                                      mutation_rate_per_chromosome: float, selection_coefficient: float,
                                      chromosome_length_morgans: float,
                                      marker_spacing_kb: float, map_cM_per_Mb: float, heterozygosity: float,
                                      mutation_rate_per_bp: float, conversion_rate_per_bp: float,
                                      observed_class_cM: "np.ndarray", reported_size: float,
                                      advantage: float, recombination_fraction: float,
                                      far_observed_class_cM: "np.ndarray", far_observed_probability: float,
                                      far_report_class_cM: "np.ndarray", max_since_end: int, horizon: int) -> float:
    '''Mean coalescence time of the pairs whose run extends a reported length on the side of a swept neutral position away from the selected site, at the recovered bottleneck end and sweep age.

    A randomly mating diploid population had n_ancestral breeding individuals
    until a bottleneck of n_bottleneck individuals that lasted bottleneck_length
    generations, after which it has had n_recent individuals up to the present;
    the number of generations since the bottleneck ended is unknown. Deleterious
    mutations arise along the chromosome, of map length
    chromosome_length_morgans, at mutation_rate_per_chromosome per chromosome
    per generation with a multiplicative fitness effect selection_coefficient in
    heterozygotes, and the background selection they cause reduces the
    effective size of a neutral position as in the reduction-factor step.
    Individuals are genotyped at markers spaced marker_spacing_kb kilobases
    apart on average, on a map of map_cM_per_Mb centiMorgans per megabase, with
    heterozygosity at the typed markers; identity along a lineage is broken by
    recombination and by de novo mutation and gene conversion at the stated
    rates per base pair per generation. For the runs of lengths in
    observed_class_cM the study reports, in place of their coverage, the
    constant population size that the source's closed-form steady-state
    expression for the coverage of runs of a given length on this panel returns
    for that class: reported_size breeding individuals. At one focal position, a
    favourable mutation with heterozygous advantage advantage arose, after the
    bottleneck ended, at a site at recombination fraction
    recombination_fraction from the focal position; the number of generations
    since it arose is unknown. The fraction of individuals whose run of
    homozygosity around the focal position has a length in
    far_observed_class_cM on the side away from the selected site is
    far_observed_probability. Recover the coverage of the observed class from
    reported_size as in the closed-form step; recover the number of generations
    since the bottleneck ended from that coverage as in the bottleneck-end step,
    searching 1 to max_since_end; recover the number of generations since the
    favourable mutation arose from far_observed_probability as in the
    sweep-age step; then return the mean coalescence generation of the pairs
    counted by the class far_report_class_cM of the side away from the selected
    site at those two values, as in the far-side profile step, counting
    coalescence generations 1 to horizon.

    Parameters
    ----------
    n_ancestral : float
        Number of breeding individuals before the bottleneck, >= 1.
    n_bottleneck : float
        Number of breeding individuals during the bottleneck, >= 1.
    n_recent : float
        Number of breeding individuals since the bottleneck ended, >= 1.
    bottleneck_length : int
        Number of generations the bottleneck lasted, >= 0.
    mutation_rate_per_chromosome : float
        Deleterious mutation rate per chromosome per generation, > 0.
    selection_coefficient : float
        Multiplicative fitness effect of a deleterious mutation in
        heterozygotes, in (0, 1).
    chromosome_length_morgans : float
        Map length of the chromosome in Morgans, > 0.
    marker_spacing_kb : float
        Mean spacing of the typed markers in kilobases, >= 0.
    map_cM_per_Mb : float
        Genetic map density in centiMorgans per megabase, > 0.
    heterozygosity : float
        Probability that a typed marker is heterozygous between two random
        haplotypes, in (0, 1].
    mutation_rate_per_bp : float
        De novo mutation rate per base pair per generation, >= 0.
    conversion_rate_per_bp : float
        Rate of gene-conversion breaks per base pair per generation, >= 0.
    observed_class_cM : np.ndarray
        Shape (2,): lower and upper ends of the observed length class in
        centiMorgans, 0 <= lower < upper; the upper end may be infinite.
    reported_size : float
        Constant number of breeding individuals reported for that class, >= 1.
    advantage : float
        Fitness advantage of a heterozygous carrier of the favourable
        mutation, > 0.
    recombination_fraction : float
        Recombination fraction between the selected site and the focal
        position, in [0, 0.5].
    far_observed_class_cM : np.ndarray
        Shape (2,): lower and upper ends, in centiMorgans, of the length class
        of the side of the run away from the selected site whose frequency is
        observed, 0 <= lower < upper; the upper end may be infinite.
    far_observed_probability : float
        Observed fraction of individuals whose run has a length in that class
        on that side, in [0, 1].
    far_report_class_cM : np.ndarray
        Shape (2,): lower and upper ends, in centiMorgans, of the length class
        of the side away from the selected site that is reported,
        0 <= lower < upper; the upper end may be infinite.
    max_since_end : int
        Largest candidate number of generations since the bottleneck ended,
        >= 1.
    horizon : int
        Last coalescence generation before the present that is counted, >= 1.

    Returns
    -------
    mean_generation : float
        Mean coalescence generation of the pairs counted by the reported class
        of the side away from the selected site at the recovered bottleneck end
        and sweep age, in generations before the present, as a native Python
        float.

    Raises
    ------
    ValueError
        If marker_spacing_kb is not a finite number >= 0, if map_cM_per_Mb is
        not a finite number > 0, if either rate per base pair is not a finite
        number >= 0, if any class is not a pair of numbers with
        0 <= lower < upper, or on any condition raised by the earlier steps for
        the derived inputs.
    '''
    return mean_generation  # placeholder
```
