# Biology-Ecology-22

## Background

Integral projection models describe populations structured by a continuous trait such as body length: a
kernel built from length-dependent survival, growth and reproduction maps the trait distribution at one
census to the distribution at the next, and discretising the kernel integrals on a fixed mesh turns the
projection into a matrix-vector product. They are widely used for fish and for invasive species, whose
establishment after an accidental introduction is a question about the fate of a small founding
population. Two ingredients that a plain integral projection model lacks decide that fate. The first is
density dependence with an Allee effect: in the Ricker family of spawner-recruit relations used throughout
fisheries science, an Allee threshold makes the empty state stable for populations below the threshold
and the carrying capacity stable above it, and applying the relation to spawning biomass rather than to
spawner counts reflects that fecundity scales with body size. The second is demographic stochasticity,
which a branching process captures by giving every individual a small set of mutually exclusive outcomes
over a census interval (dying, surviving, reproducing, or both) with length-dependent probabilities; the
extinction of a small population is then a question about the branching process, while for large
populations the law of large numbers makes the expected trait distribution follow a deterministic
projection equation, and for a branching process of independent individuals the full probability distribution of a
census count can be obtained exactly rather than by replicate simulation, once the density-dependent
spawning level is frozen at its expected value. Linking the two scales requires a rule that translates a population-level biomass
target into the level of reproduction of individuals, so that survivors, growth and new recruits together
track the density-dependent target. With the vital rates of an invasive carp, such a coupled model
projects how many founders of a given size are needed for establishment and how the expected numbers of
mature females evolve over the years after a containment breach.

## Problem

Integral projection models (IPMs) carry the continuous length distribution of a population from one
annual census to the next through length-dependent survival, growth and maturity, but a plain IPM has no
mechanism for demographic stochasticity or for the density dependence and Allee effect that decide
whether a small founding population of an invader establishes. A recent source addresses this for the
females of an invasive carp by coupling an IPM to a Ricker model with an Allee effect that acts on
spawning biomass rather than on spawner counts, by translating the population-level biomass target into
per-individual reproduction through a yearly spawning proportion fixed by a biomass-balance rule, and by
deriving each female's contribution to the next census from a branching process with four mutually
exclusive outcomes (dies, survives, reproduces, or survives and reproduces). Taking expectations of that
branching process gives a deterministic projection equation for the expected female length distribution
whose only density dependence is the yearly spawning proportion. Because the underlying model is a
branching process, the number of females at a later census is a random variable, and its spread across
realizations measures the demographic stochasticity that the source studies by replicate simulation. The
inputs are the vital-rate functions, the Ricker-Allee parameters and the founding cohort; the outputs are the expected length distribution at later censuses and the probability distribution of the
census counts around it.

Lengths z are total lengths in metres on the domain [0.01, 1.50] m and every quantity refers to females.
Body weight in kilograms follows log10(W) = 1.02 + 3.02 log10(z). Annual survival is
s(z) = exp(-2.7 w^(-0.315)) with w the body weight in grams. The probability of being mature is the
increasing logistic m(z) = 1 / (1 + exp(7.41 - 24.98 z)). Growth is a normal transition density in the
next-census length with mean 0.194 + 0.841 z and standard deviation 0.1 m. Age-1 recruits have a normal
length density with mean 0.319 m and standard deviation 0.040 m, and each spawning female produces k = 2
female recruits that enter the next census. The Ricker-Allee model has intrinsic growth rate
gamma = 1.837, Allee threshold C equal to the biomass of 10 average adults of length 0.867 m, and carrying
capacity K equal to the biomass of 500 such adults; take it in the form the source adopts, applied to the
total female biomass b_t = integral of W(z) n(z, t) dz, to predict a target biomass for the next census.
The spawning proportion A_t of each year follows from the source's biomass-balance rule, which compares
that target with the biomass the present females themselves carry into the next census and with the
biomass that recruitment can add, and the expected length distribution advances by the source's expected
projection kernel, in which survival with growth and recruitment are gated by s(z), m(z) and A_t.

Discretise the length domain with the midpoint rule on 300 bins of equal width: evaluate every function at
the bin midpoints, replace each integral over z by the sum of the integrand at the midpoints times the bin
width, evaluate the normal densities of the growth kernel and of the recruits as they are (no
renormalisation over the finite domain), and treat the expected length distribution as a density (females
per metre) on the midpoints. The founding cohort at census 0 consists of 800 subadult females whose lengths
follow a normal density with mean 0.300 m and standard deviation 0.020 m, so that n(z, 0) is 800 times
that density. Project the expected distribution forward through 12 annual censuses; each year the biomass
target, the spawning proportion and the new distribution are computed from the current expected
distribution. For the variability, treat the realized population as the source's branching process: every
female present at a census independently dies, survives, reproduces, or survives and reproduces, with the
length-dependent probabilities built from s(z), m(z) and the spawning proportion of that year (survival
and reproduction independent); a survivor's next-census length is drawn from the growth kernel (a survivor
whose length falls outside the domain leaves the population); each recruit of a spawning female receives
an independent length from the recruit density; and at a census each female present is mature
independently with probability m(z). Take the spawning proportion of every year from the expected
trajectory (the density-dependent feedback frozen at its expected value, a mean-field simplification for a
founding cohort this large). The founders are whole individuals: convert the expected number of founders in
each bin (n(z, 0) times the bin width) to individuals with the source's rule for the fractional individual
of a bin, and let every founder start a lineage independent of the others.

Report the probability that at least 500 mature females are present at census 12 in this process,
computed exactly (no simulation), to at least five decimal places.

State the conventions you adopted and justify each from the source. In the reasoning give the target
biomass for census 1 predicted from the founding cohort, the biomass at census 1 of the founders that
survive, the biomass that the recruits of one spawning female add to the next census, the spawning
proportions applied in the projections from censuses 0 to 3 together with what sets each of them, the reason
no spawning occurs in the projections from census 6 onwards, the expected number and the standard deviation of the number of mature females at census 12, and the
construction you used to obtain the exact distribution (biomasses in kilograms).

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

length_to_weight

Goal
----
Return the body weight in kilograms of female fish of the given total lengths in metres from the allometric length-weight relation log10(weight in kg) = intercept + slope * log10(length in m), evaluated element-wise (base-10 logarithms).

```python
def length_to_weight(lengths: "numpy.ndarray", intercept: float = 1.02,
                     slope: float = 3.02) -> "numpy.ndarray":
    """Return the body weight in kilograms of female fish of the given total lengths in metres from the allometric length-weight relation log10(weight in kg) = intercept + slope * log10(length in m), evaluated element-wise (base-10 logarithms).

    Parameters
    ----------
    lengths : numpy.ndarray
        One-dimensional array of finite, strictly positive total lengths in metres.
    intercept : float
        Intercept of the log10-log10 relation; default 1.02.
    slope : float
        Slope of the log10-log10 relation; default 3.02.

    Returns
    -------
    weights : numpy.ndarray
        Array of the same shape as lengths holding the weights in kilograms (float64).

    Raises
    ------
    ValueError
        If lengths is not a non-empty one-dimensional array of finite positive values, or a coefficient is not finite.
    """
    return weights
```

### Step 2

survival_probability

Goal
----
Return the annual survival probability of female fish of the given total lengths in metres, s(z) = exp(-slope * w ** exponent) with w the body weight in grams, obtained from the length-weight relation of step 01 with its default coefficients (kilograms converted to grams), evaluated element-wise.

```python
def survival_probability(lengths: "numpy.ndarray", slope: float = 2.7,
                         exponent: float = -0.315) -> "numpy.ndarray":
    """Return the annual survival probability of female fish of the given total lengths in metres, s(z) = exp(-slope * w ** exponent) with w the body weight in grams, obtained from the length-weight relation of step 01 with its default coefficients (kilograms converted to grams), evaluated element-wise.

    Parameters
    ----------
    lengths : numpy.ndarray
        One-dimensional array of finite, strictly positive total lengths in metres.
    slope : float
        Positive coefficient multiplying the weight power; default 2.7.
    exponent : float
        Exponent of the weight in grams; default -0.315.

    Returns
    -------
    survival : numpy.ndarray
        Array of the same shape as lengths holding annual survival probabilities in [0, 1] (float64).

    Raises
    ------
    ValueError
        If lengths is not a non-empty one-dimensional array of finite positive values, slope is not positive and finite, or exponent is not finite.
    """
    return survival
```

### Step 3

maturity_probability

Goal
----
Return the probability that a female of the given total length in metres is mature, the logistic function m(z) = 1 / (1 + exp(-(intercept + slope * z))), which increases with length when slope is positive, evaluated element-wise.

```python
def maturity_probability(lengths: "numpy.ndarray", intercept: float = -7.41,
                         slope: float = 24.98) -> "numpy.ndarray":
    """Return the probability that a female of the given total length in metres is mature, the logistic function m(z) = 1 / (1 + exp(-(intercept + slope * z))), which increases with length when slope is positive, evaluated element-wise.

    Parameters
    ----------
    lengths : numpy.ndarray
        One-dimensional array of finite, nonnegative total lengths in metres.
    intercept : float
        Intercept of the logistic argument; default -7.41.
    slope : float
        Slope of the logistic argument per metre; default 24.98.

    Returns
    -------
    maturity : numpy.ndarray
        Array of the same shape as lengths holding maturity probabilities in (0, 1) (float64).

    Raises
    ------
    ValueError
        If lengths is not a non-empty one-dimensional array of finite nonnegative values, or a coefficient is not finite.
    """
    return maturity
```

### Step 4

midpoint_mesh

Goal
----
Return the midpoint-rule mesh of n_bins equal bins on [z_min, z_max]: an array of shape (2, n_bins) whose row 0 holds the bin midpoints in increasing order and whose row 1 holds the bin widths, all equal to (z_max - z_min) / n_bins, so that sum(row 1 * f(row 0)) approximates the integral of f over the interval.

```python
def midpoint_mesh(z_min: float, z_max: float, n_bins: int) -> "numpy.ndarray":
    """Return the midpoint-rule mesh of n_bins equal bins on [z_min, z_max]: an array of shape (2, n_bins) whose row 0 holds the bin midpoints in increasing order and whose row 1 holds the bin widths, all equal to (z_max - z_min) / n_bins, so that sum(row 1 * f(row 0)) approximates the integral of f over the interval.

    Parameters
    ----------
    z_min : float
        Lower end of the length interval (metres).
    z_max : float
        Upper end of the length interval (metres); must exceed z_min.
    n_bins : int
        Number of equal bins, at least 1.

    Returns
    -------
    mesh : numpy.ndarray
        Array of shape (2, n_bins): row 0 the bin midpoints, row 1 the bin widths (float64).

    Raises
    ------
    ValueError
        If z_max does not exceed z_min, either end is not finite, or n_bins is not a positive integer.
    """
    return mesh
```

### Step 5

growth_kernel

Goal
----
Return the midpoint-rule growth matrix on a mesh of bin midpoints: G[i, j] is the normal probability density of the next-census length z_i of a female whose current length is z_j, with mean mu0 + mu1 * z_j and standard deviation sd (metres), multiplied by the bin width, so that G @ f approximates the integral of the growth kernel times f(z) over z on the mesh. The density is evaluated as is: the columns are not renormalised over the finite mesh, so probability mass that falls outside the mesh is lost.

```python
def growth_kernel(midpoints: "numpy.ndarray", width: float, mu0: float = 0.194,
                  mu1: float = 0.841, sd: float = 0.1) -> "numpy.ndarray":
    """Return the midpoint-rule growth matrix on a mesh of bin midpoints: G[i, j] is the normal probability density of the next-census length z_i of a female whose current length is z_j, with mean mu0 + mu1 * z_j and standard deviation sd (metres), multiplied by the bin width, so that G @ f approximates the integral of the growth kernel times f(z) over z on the mesh. The density is evaluated as is: the columns are not renormalised over the finite mesh, so probability mass that falls outside the mesh is lost.

    Parameters
    ----------
    midpoints : numpy.ndarray
        One-dimensional array of finite bin midpoints in metres (row 0 of step 04).
    width : float
        Positive bin width in metres (the common value in row 1 of step 04).
    mu0 : float
        Intercept of the mean next-census length in metres; default 0.194.
    mu1 : float
        Slope of the mean next-census length per metre of current length; default 0.841.
    sd : float
        Positive standard deviation of the next-census length in metres; default 0.1.

    Returns
    -------
    growth : numpy.ndarray
        Array of shape (n, n) with G[i, j] the density at z_i given z_j times the bin width (float64).

    Raises
    ------
    ValueError
        If midpoints is not a non-empty one-dimensional finite array, width or sd is not positive and finite, or mu0 or mu1 is not finite.
    """
    return growth
```

### Step 6

recruit_length_density

Goal
----
Return the length density of age-1 female recruits at the given mesh midpoints: the normal probability density in metres with the given mean and standard deviation, evaluated as is, without renormalisation over the mesh.

```python
def recruit_length_density(midpoints: "numpy.ndarray", mean: float = 0.319,
                           sd: float = 0.040) -> "numpy.ndarray":
    """Return the length density of age-1 female recruits at the given mesh midpoints: the normal probability density in metres with the given mean and standard deviation, evaluated as is, without renormalisation over the mesh.

    Parameters
    ----------
    midpoints : numpy.ndarray
        One-dimensional array of finite bin midpoints in metres (row 0 of step 04).
    mean : float
        Mean recruit length in metres; default 0.319.
    sd : float
        Positive standard deviation of the recruit length in metres; default 0.040.

    Returns
    -------
    recruit_density : numpy.ndarray
        Array of the same shape as midpoints holding the density values per metre (float64).

    Raises
    ------
    ValueError
        If midpoints is not a non-empty one-dimensional finite array, mean is not finite, or sd is not positive and finite.
    """
    return recruit_density
```

### Step 7

ricker_allee_target

Goal
----
Return the target biomass (kg) for the next census predicted from the current female biomass (kg) by the discrete-time Ricker model with an Allee effect in the form the source adopts (its Eq. 2), with intrinsic growth rate gamma, Allee threshold allee_threshold and carrying capacity carrying_capacity (both in kg), applied to biomass rather than to counts. The target equals the current biomass when the current biomass lies exactly at the Allee threshold or at the carrying capacity, exceeds it in between and falls below it outside that range.

```python
def ricker_allee_target(biomass: float, gamma: float, allee_threshold: float,
                        carrying_capacity: float) -> float:
    """Return the target biomass (kg) for the next census predicted from the current female biomass (kg) by the discrete-time Ricker model with an Allee effect in the form the source adopts (its Eq. 2), with intrinsic growth rate gamma, Allee threshold allee_threshold and carrying capacity carrying_capacity (both in kg), applied to biomass rather than to counts. The target equals the current biomass when the current biomass lies exactly at the Allee threshold or at the carrying capacity, exceeds it in between and falls below it outside that range.

    Parameters
    ----------
    biomass : float
        Current total female biomass in kg, nonnegative.
    gamma : float
        Positive intrinsic growth rate of the Ricker model.
    allee_threshold : float
        Allee threshold C in kg, with 0 < C < K.
    carrying_capacity : float
        Carrying capacity K in kg, above the Allee threshold.

    Returns
    -------
    target : float
        Target biomass for the next census in kg, as a native Python float.

    Raises
    ------
    ValueError
        If any input is not finite, biomass is negative, gamma is not positive, or the Allee threshold and carrying capacity do not satisfy 0 < C < K.
    """
    return target
```

### Step 8

spawning_proportion

Goal
----
Return the proportion A_t of the mature females that spawn in the current year according to the source's biomass-balance rule (its Fig. 1, phases 1 and 2), given the expected female length density on the mesh, the mesh of step 04, the growth matrix of step 05, the survival probabilities, weights (kg) and maturity probabilities at the midpoints, the recruit length density of step 06 with recruits_per_spawner female recruits per spawning female, and the target biomass for the next census from step 07. The rule sets the spawning level from how the target compares with the biomass the present females themselves contribute to the next census and with the biomass that recruitment can add; the result lies in [0, 1], with no spawning when the target is already met, universal spawning when recruitment cannot reach it, and 0 whenever no female can spawn (zero maximum recruit biomass, e.g. an empty population).

```python
def spawning_proportion(density: "numpy.ndarray", mesh: "numpy.ndarray", growth: "numpy.ndarray",
                        survival: "numpy.ndarray", weights: "numpy.ndarray",
                        maturity: "numpy.ndarray", recruit_density: "numpy.ndarray",
                        recruits_per_spawner: int, target_biomass: float) -> float:
    """Return the proportion A_t of the mature females that spawn in the current year according to the source's biomass-balance rule (its Fig. 1, phases 1 and 2), given the expected female length density on the mesh, the mesh of step 04, the growth matrix of step 05, the survival probabilities, weights (kg) and maturity probabilities at the midpoints, the recruit length density of step 06 with recruits_per_spawner female recruits per spawning female, and the target biomass for the next census from step 07. The rule sets the spawning level from how the target compares with the biomass the present females themselves contribute to the next census and with the biomass that recruitment can add; the result lies in [0, 1], with no spawning when the target is already met, universal spawning when recruitment cannot reach it, and 0 whenever no female can spawn (zero maximum recruit biomass, e.g. an empty population).

    Parameters
    ----------
    density : numpy.ndarray
        One-dimensional array of length n: the expected number of females per metre at each midpoint, nonnegative.
    mesh : numpy.ndarray
        Array of shape (2, n) from step 04: midpoints and bin widths.
    growth : numpy.ndarray
        Array of shape (n, n) from step 05: the midpoint-rule growth matrix.
    survival : numpy.ndarray
        One-dimensional array of length n: annual survival probabilities at the midpoints.
    weights : numpy.ndarray
        One-dimensional array of length n: body weights in kg at the midpoints.
    maturity : numpy.ndarray
        One-dimensional array of length n: maturity probabilities at the midpoints.
    recruit_density : numpy.ndarray
        One-dimensional array of length n: recruit length density at the midpoints (per metre).
    recruits_per_spawner : int
        Positive number of female recruits produced by each spawning female.
    target_biomass : float
        Target biomass for the next census in kg from step 07, nonnegative.

    Returns
    -------
    proportion : float
        The spawning proportion A_t in [0, 1], as a native Python float.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent with the density length, any array contains a non-finite or negative value, a bin width is not positive, recruits_per_spawner is not a positive integer, or target_biomass is negative or not finite.
    """
    return proportion
```

### Step 9

expected_trajectory

Goal
----
Starting from a founding cohort of n0 females whose lengths follow a normal density with mean mean_len0 and standard deviation sd_len0 (metres), project the expected female length distribution forward through n_years annual censuses with the source's expected projection equation on the midpoint mesh of n_bins equal bins on [z_min, z_max] (step 04): the vital rates of steps 01-03 with their default coefficients, the growth matrix of step 05, the recruit length density of step 06 with mean recruit_mean and standard deviation recruit_sd, the Ricker-Allee target of step 07 with gamma, allee_threshold and carrying_capacity, and the spawning proportion of step 08 with recruits_per_spawner female recruits per spawning female. Each year the density advances by the source's expected kernel: the survivors propagated by the growth matrix plus the recruits produced by the mature females that spawn. Return the array of expected densities (females per metre at the midpoints) with row t the census t, t = 0 to n_years. Call the earlier step functions rather than reimplementing them.

```python
def expected_trajectory(n0: float, mean_len0: float, sd_len0: float, n_years: int, n_bins: int,
                        gamma: float, allee_threshold: float, carrying_capacity: float,
                        recruits_per_spawner: int, recruit_mean: float, recruit_sd: float,
                        z_min: float = 0.01, z_max: float = 1.5) -> "numpy.ndarray":
    """Starting from a founding cohort of n0 females whose lengths follow a normal density with mean mean_len0 and standard deviation sd_len0 (metres), project the expected female length distribution forward through n_years annual censuses with the source's expected projection equation on the midpoint mesh of n_bins equal bins on [z_min, z_max] (step 04): the vital rates of steps 01-03 with their default coefficients, the growth matrix of step 05, the recruit length density of step 06 with mean recruit_mean and standard deviation recruit_sd, the Ricker-Allee target of step 07 with gamma, allee_threshold and carrying_capacity, and the spawning proportion of step 08 with recruits_per_spawner female recruits per spawning female. Each year the density advances by the source's expected kernel: the survivors propagated by the growth matrix plus the recruits produced by the mature females that spawn. Return the array of expected densities (females per metre at the midpoints) with row t the census t, t = 0 to n_years. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    n0 : float
        Positive number of founding females.
    mean_len0 : float
        Mean length of the founders in metres.
    sd_len0 : float
        Positive standard deviation of the founder lengths in metres.
    n_years : int
        Number of annual censuses to project, at least 1.
    n_bins : int
        Number of equal midpoint bins on the length domain, at least 1.
    gamma : float
        Positive intrinsic growth rate of the Ricker-Allee model.
    allee_threshold : float
        Allee threshold in kg, with 0 < C < K.
    carrying_capacity : float
        Carrying capacity in kg.
    recruits_per_spawner : int
        Positive number of female recruits per spawning female.
    recruit_mean : float
        Mean recruit length in metres.
    recruit_sd : float
        Positive standard deviation of the recruit length in metres.
    z_min : float
        Lower end of the length domain in metres; default 0.01.
    z_max : float
        Upper end of the length domain in metres; default 1.5.

    Returns
    -------
    trajectory : numpy.ndarray
        Array of shape (n_years + 1, n_bins): row t holds the expected density (females per metre) at the midpoints at census t (float64).

    Raises
    ------
    ValueError
        If any real-valued input is not finite, n0 or sd_len0 is not positive, n_years or recruits_per_spawner is not a positive integer, or the mesh or Ricker-Allee parameters are invalid for the earlier steps.
    """
    return trajectory
```

### Step 10

mature_count_probability

Goal
----
Orchestrator. For the founding cohort and parameters of step 09, return the exact probability that at least threshold mature females are present at census n_years in the source's discrete-individual branching process, computed without simulation. In that process every female present at a census independently follows the source's four outcomes (dies; survives; reproduces; survives and reproduces), survival with the probability of step 02 and reproduction with the maturity probability of step 03 times the spawning proportion of that year, the two events independent; the spawning proportion of every year is the one of the expected trajectory of step 09 (steps 07-08 applied to its rows). A survivor's next-census length is drawn from the growth kernel of step 05 (a survivor whose length falls outside the mesh leaves the population); each of the recruits_per_spawner recruits of a spawning female receives an independent length from the recruit density of step 06; at the final census each female present is mature independently with the probability of step 03. The founders are whole individuals: the expected number in each bin (row 0 of step 09 times the bin width) is converted to individuals by the source's rule for the fractional individual of a bin, and every founder starts a lineage independent of the others. Call the earlier step functions rather than reimplementing them.

```python
def mature_count_probability(n0: float, mean_len0: float, sd_len0: float, n_years: int, n_bins: int,
                             gamma: float, allee_threshold: float, carrying_capacity: float,
                             recruits_per_spawner: int, recruit_mean: float, recruit_sd: float,
                             threshold: int, z_min: float = 0.01, z_max: float = 1.5) -> float:
    """Orchestrator. For the founding cohort and parameters of step 09, return the exact probability that at least threshold mature females are present at census n_years in the source's discrete-individual branching process, computed without simulation. In that process every female present at a census independently follows the source's four outcomes (dies; survives; reproduces; survives and reproduces), survival with the probability of step 02 and reproduction with the maturity probability of step 03 times the spawning proportion of that year, the two events independent; the spawning proportion of every year is the one of the expected trajectory of step 09 (steps 07-08 applied to its rows). A survivor's next-census length is drawn from the growth kernel of step 05 (a survivor whose length falls outside the mesh leaves the population); each of the recruits_per_spawner recruits of a spawning female receives an independent length from the recruit density of step 06; at the final census each female present is mature independently with the probability of step 03. The founders are whole individuals: the expected number in each bin (row 0 of step 09 times the bin width) is converted to individuals by the source's rule for the fractional individual of a bin, and every founder starts a lineage independent of the others. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    n0 : float
        Positive number of founding females.
    mean_len0 : float
        Mean length of the founders in metres.
    sd_len0 : float
        Positive standard deviation of the founder lengths in metres.
    n_years : int
        Number of annual censuses to project, at least 1.
    n_bins : int
        Number of equal midpoint bins on the length domain, at least 1.
    gamma : float
        Positive intrinsic growth rate of the Ricker-Allee model.
    allee_threshold : float
        Allee threshold in kg, with 0 < C < K.
    carrying_capacity : float
        Carrying capacity in kg.
    recruits_per_spawner : int
        Positive number of female recruits per spawning female.
    recruit_mean : float
        Mean recruit length in metres.
    recruit_sd : float
        Positive standard deviation of the recruit length in metres.
    threshold : int
        Nonnegative integer number of mature females; the probability of at least this many is returned.
    z_min : float
        Lower end of the length domain in metres; default 0.01.
    z_max : float
        Upper end of the length domain in metres; default 1.5.

    Returns
    -------
    probability : float
        Probability that at least threshold mature females are present at census n_years, as a native Python float.

    Raises
    ------
    ValueError
        If threshold is not a nonnegative integer, any real-valued input is not finite, n0 or sd_len0 is not positive, n_years or recruits_per_spawner is not a positive integer, or the mesh or Ricker-Allee parameters are invalid for the earlier steps.
    """
    return probability
```
