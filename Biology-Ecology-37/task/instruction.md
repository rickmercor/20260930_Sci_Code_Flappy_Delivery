# Biology-Ecology-37

## Background

Ice-associated marine mammals are hard to count from aircraft, and genetic sampling offers another route: a skin biopsy identifies an individual, so the same animal sampled in two years is a recapture, and kinship between samples links each animal to its mother and to its siblings. Close-kin mark-recapture turns how often sampled animals are close relatives into estimates of adult abundance and survival, and analysing close kin together with individual recaptures can sharpen a survey without more field effort.

Such surveys are costly, so how many animals to sample, and in which years, has to be settled in advance from the information the planned samples are expected to carry. Species whose females skip breeding years need a model of the breeding cycle inside the kinship probabilities, and because the ages of wild animals come from epigenetic clocks fitted to animals of known age, the clock's error has to be carried into every kinship probability.

## Problem

Close-kin mark-recapture estimates abundance from how often genetic samples turn out to be close relatives, and it can be combined in one pseudo-likelihood with genetic self-recaptures, the same animal sampled in two different years. For an ice-associated pinniped whose females skip breeding years, a biopsy survey is to be extended, and the design question is how many animals a new programme must sample each year for the estimate of adult female abundance to reach a target precision, a question the expected pseudo-Fisher information of the planned samples answers without simulating data. In this problem the inputs are a female life history, the past and planned sampling, the age clock's known-age reference animals and a precision target, and the output is a single number: how many kinship findings the chosen design can expect its genotyping to yield.

Model females only, in two stages: juveniles aged 1 to 5, whose annual survival of 0.85 applies to every year spent at those ages, and adults aged 6 and older, with annual survival 0.955 and no maximum age. There are 48,000 adult females in 2016, adult female abundance changes by the factor e^(-0.012) per year, the age composition within each stage is the quasi-equilibrium one, and the population dynamics are modelled from 2002. Each year an adult female is pregnant, has a calf born that year, or rests: a pregnant female always has a calf the next year, a female with a calf becomes pregnant again with probability 0.16 and otherwise rests, and a resting female becomes pregnant with probability 0.38; females enter this cycle resting at age 4.

Biopsies, which kill no animal, were taken from 800 animals in each year from 2014 to 2018, and a new programme will sample n animals in each year from 2024 to 2027. Each year's sample is drawn without regard to sex or age from the animals present, females aged 1 to 37 and males aged 1 to 5 (the two sexes equally numerous at each juvenile age), so the expected number sampled in each sex and age is proportional to its quasi-equilibrium abundance; use these expected numbers as the sample sizes, with kinship detected without error and every expected sample counted as a distinct animal in every comparison it can enter. Use the published kinship probabilities, comparison rules and Poisson pseudo-likelihood of a recent stage-structured model that adds self-recaptures to close-kin mark-recapture for a skip-breeding marine mammal: mother-offspring pairs, cross-cohort maternal half-sibling pairs and self-recaptures of females sampled in different years, whatever their birth years, with the kinship probabilities built on expected relative reproductive output and the breeding cycle. Ages come from an epigenetic clock: an animal of true age a receives the estimate a + e rounded to the nearest whole year, with e normal with mean 0 and standard deviation sigma, conditioned on the rounded estimate lying between 1 and 37; estimate sigma by maximum likelihood from these known-age females, listed as true age: clock estimates, 1: 1, 2, 2, 4; 2: 2, 3, 4; 3: 4, 5, 6; 4: 3, 4; 5: 3, 5; 6: 4; 7: 9; 8: 6; 9: 8; 10: 9; 12: 14; 14: 15; 16: 17; 18: 19; 20: 18; 22: 20; 25: 28; 28: 28; 31: 31; 34: 32; 36: 36. As the source proposes for aging error, classify every sample by its sampling year, sex and estimated age, and average each kinship probability over the two animals' true ages, weighting each possible true age of an animal in proportion to the expected number sampled of its sex and that true age in its year times the probability of its estimate (weights fixed by the design, not differentiated); apply the comparison rules to estimated ages, let true birth years before 2002 contribute nothing, and take the second sample of a self-recapture to have its stage known without error. Take the expected information over the grouped comparisons for the parameters (the natural logarithm of adult female abundance in 2016, the growth rate, the logits of the two survival rates and the logits of the two breeding probabilities), its inverse as their covariance, and the delta method for derived quantities.

Find the smallest n, a multiple of 50 no larger than 3,000, for which the expected CV of adult female abundance in 2027 is at most 0.10. In the reasoning, report sigma and the probability that a sampled female with estimated age 6 is truly an adult; the long-run proportion of adult females with a calf of the year; the fecundity of an 8-year-old female relative to an average adult female; juvenile female abundance in 2027; that smallest n and its expected CV; the expected CV for the same samples from self-recaptures alone, estimating only abundance, growth rate and the two survival rates; and, at that n, the expected standard errors of adult female survival and of the long-run proportion of adult females with a calf, and the expected numbers of mother-offspring pairs, half-sibling pairs and self-recaptures. State briefly the model and the comparison rules you used. Your final answer must be a single number: the total expected number of mother-offspring pairs, half-sibling pairs and self-recaptures among the compared samples at that n.

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

Implement **all 12 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_breeding_cycle_fecundity.py

Goal
----
Long-run proportion of adult females with a calf of the year, followed by the fecundity of a female at each requested age relative to an average adult female.

```python
def breeding_cycle_fecundity(psi2: float, psi3: float, entry_age: int, ages: "np.ndarray") -> "np.ndarray":
    '''Long-run proportion of adult females with a calf of the year, followed by the fecundity of a female at each requested age relative to an average adult female.

    Adult females move once a year between three breeding states: pregnant,
    with a calf born that year, and resting. A pregnant female always has a
    calf the next year; a female with a calf becomes pregnant again the next
    year with probability psi2 and otherwise rests; a resting female becomes
    pregnant with probability psi3 and otherwise keeps resting. A female
    enters the cycle in the resting state at age entry_age. The long-run
    proportion is the calf-state probability of the stationary distribution
    of the cycle. The fecundity at age a is the probability that a female of
    age a is in the calf state, divided by that long-run proportion; it is
    zero at every age below entry_age.

    Parameters
    ----------
    psi2 : float
        Probability of becoming pregnant in the year after a calf, a finite
        number with 0 < psi2 < 1.
    psi3 : float
        Probability that a resting female becomes pregnant, a finite number
        with 0 < psi3 < 1.
    entry_age : int
        Age in years at which a female enters the cycle, resting, an integer
        from 1 to 20.
    ages : np.ndarray
        Shape (K,), K >= 1; ages in years, integers from 0 to 200, in any
        order.

    Returns
    -------
    summary : np.ndarray
        Shape (K + 1,). Element 0 is the long-run proportion of adult females
        with a calf of the year; element 1 + k is the fecundity at age
        ages[k].

    Raises
    ------
    ValueError
        If psi2 or psi3 is not a finite number strictly between 0 and 1, if
        entry_age is not an integer from 1 to 20, or if ages is not a
        non-empty one-dimensional array of integers from 0 to 200.
    '''
    return summary  # placeholder
```

### Step 2

02_stage_abundances.py

Goal
----
Adult and juvenile female abundance in each requested year under quasi-equilibrium stage-structured dynamics.

```python
def stage_abundances(theta: "np.ndarray", ref_year: int, years: "np.ndarray") -> "np.ndarray":
    '''Adult and juvenile female abundance in each requested year under quasi-equilibrium stage-structured dynamics.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3):
    N_ref is the number of adult females (age 6 and older, with no maximum
    age) in ref_year, adult female abundance changes by the factor exp(r)
    per year, phi_A is the annual survival of adults and phi_J that of
    juveniles (ages 1 to 5; the year from age 5 to age 6 is a juvenile
    year). The last two entries are breeding probabilities and do not enter
    this step. Juvenile female abundance is the number of females aged 1 to
    5 when the age composition within each stage is the quasi-equilibrium
    (stable-age) one implied by the survival rates and the rate of change
    exp(r).

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,): ln N_ref from 0 to 30, r from -0.5 to 0.5, and each of
        the four logits from -15 to 15, with exp(r) > phi_A so that the
        adult age distribution is summable. Every later step that takes
        theta uses this domain.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    years : np.ndarray
        Shape (Y,), Y >= 1; integers from 1900 to 2200, in any order.

    Returns
    -------
    abundance : np.ndarray
        Shape (2, Y). Row 0 is adult female abundance and row 1 juvenile
        female abundance in years[i].

    Raises
    ------
    ValueError
        If theta is not an array of shape (6,) inside the domain above
        (exp(r) <= phi_A included), if ref_year is not an integer from 1900
        to 2200, or if years is not a non-empty one-dimensional array of
        integers from 1900 to 2200.
    '''
    return abundance  # placeholder
```

### Step 3

03_sample_composition.py

Goal
----
Expected number of sampled animals of each sex and age in each sampling year, for sampling that does not select by sex or age among the animals available.

```python
def sample_composition(sample_sizes: "np.ndarray", growth_rate: float, adult_survival: float,
                       juvenile_survival: float, max_female_age: int, max_male_age: int) -> "np.ndarray":
    '''Expected number of sampled animals of each sex and age in each sampling year, for sampling that does not select by sex or age among the animals available.

    In each sampling year, sample_sizes[i] animals are taken from the
    animals available: females aged 1 to max_female_age and males aged 1 to
    max_male_age. The numbers at age are those of the quasi-equilibrium age
    structure with annual survival juvenile_survival at ages 1 to 5 (the year
    from age 5 to age 6 included) and adult_survival from age 6 on, and
    population change exp(growth_rate) per year; each sex has the same
    numbers at every juvenile age. The expected number sampled in a class is
    sample_sizes[i] times that class's share of the available animals.

    Parameters
    ----------
    sample_sizes : np.ndarray
        Shape (Y,), Y >= 1; number of animals sampled in each sampling year,
        finite and non-negative.
    growth_rate : float
        Annual rate of change r, a finite number from -0.5 to 0.5.
    adult_survival : float
        Annual survival from age 6 on, a finite number strictly between 0
        and 1 with adult_survival < exp(growth_rate).
    juvenile_survival : float
        Annual survival at ages 1 to 5, a finite number strictly between 0
        and 1.
    max_female_age : int
        Oldest age at which females are sampled, an integer from 6 to 100.
    max_male_age : int
        Oldest age at which males are sampled, an integer from 0 to 5 (0
        means that no males are sampled).

    Returns
    -------
    samples : np.ndarray
        Shape (Y, 2, max_female_age). Entry [i, 0, a - 1] is the expected
        number of females of age a and entry [i, 1, a - 1] the expected
        number of males of age a sampled in year i (zero above max_male_age).

    Raises
    ------
    ValueError
        If sample_sizes is not a non-empty one-dimensional array of finite
        non-negative numbers, if growth_rate is not a finite number from -0.5
        to 0.5, if either survival is not a finite number strictly between 0
        and 1, if adult_survival >= exp(growth_rate), if max_female_age is
        not an integer from 6 to 100, or if max_male_age is not an integer
        from 0 to 5.
    '''
    return samples  # placeholder
```

### Step 4

04_age_error_sd.py

Goal
----
Maximum-likelihood standard deviation of the error of epigenetic age estimates, from reference animals of known age.

```python
def age_error_sd(true_ages: "np.ndarray", estimated_ages: "np.ndarray", max_age: int) -> float:
    '''Maximum-likelihood standard deviation of the error of epigenetic age estimates, from reference animals of known age.

    An animal of true age a receives the estimated age obtained by adding to
    a a normal error with mean 0 and standard deviation sigma, rounding the
    result to the nearest whole year, and conditioning on the rounded value
    lying between 1 and max_age inclusive. The reference animals'
    estimates are independent given their true ages. Return the sigma
    between 0.05 and 50 that maximises the likelihood of the estimated ages
    given the true ages, to a relative accuracy of 1e-10.

    Parameters
    ----------
    true_ages : np.ndarray
        Shape (K,), K >= 1; true ages of the reference animals, integers
        from 1 to max_age.
    estimated_ages : np.ndarray
        Shape (K,); their estimated ages, integers from 1 to max_age, with at
        least one different from its true age.
    max_age : int
        Oldest possible estimated age, an integer from 2 to 100.

    Returns
    -------
    sigma : float
        The maximum-likelihood standard deviation, as a native Python float.

    Raises
    ------
    ValueError
        If true_ages and estimated_ages are not one-dimensional integer
        arrays of the same non-zero length with values from 1 to max_age, if
        every estimated age equals its true age, if max_age is not an integer
        from 2 to 100, or if the likelihood over sigma from 0.05 to 50 is
        largest at sigma = 0.05 or sigma = 50.
    '''
    return sigma  # placeholder
```

### Step 5

05_estimated_age_numbers.py

Goal
----
Expected numbers of sampled animals by sex, true age and estimated age in each sampling year.

```python
def estimated_age_numbers(samples: "np.ndarray", error_sd: float) -> "np.ndarray":
    '''Expected numbers of sampled animals by sex, true age and estimated age in each sampling year.

    samples[i, s, a - 1] is the expected number of animals of sex s (0
    female, 1 male) and true age a sampled in year i, as in the third step,
    for ages 1 to A. Each animal's estimated age follows the error model of
    the fourth step with standard deviation error_sd and max_age = A:
    the true age plus a normal error, rounded to the nearest whole year and
    conditioned on lying between 1 and A.

    Parameters
    ----------
    samples : np.ndarray
        Shape (Y, 2, A), Y >= 1, 2 <= A <= 100; finite, non-negative
        expected numbers.
    error_sd : float
        Standard deviation of the error, a finite number from 0.05 to 50.

    Returns
    -------
    numbers : np.ndarray
        Shape (Y, 2, A, A). Entry [i, s, a - 1, e - 1] is the expected number
        of animals of sex s sampled in year i with true age a and estimated
        age e.

    Raises
    ------
    ValueError
        If samples is not a finite, non-negative array of shape (Y, 2, A)
        with Y >= 1 and 2 <= A <= 100, or if error_sd is not a finite number
        from 0.05 to 50.
    '''
    return numbers  # placeholder
```

### Step 6

06_mother_offspring_probabilities.py

Goal
----
Probability that a sampled female is the mother of a sampled animal, for samples classified by sampling year, sex and estimated age.

```python
def mother_offspring_probabilities(theta: "np.ndarray", ref_year: int, entry_age: int, sample_years: "np.ndarray",
                                   first_year: int, numbers: "np.ndarray") -> "np.ndarray":
    '''Probability that a sampled female is the mother of a sampled animal, for samples classified by sampling year, sex and estimated age.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3)
    as in the second step, with the breeding cycle of the first step
    (entering it resting at entry_age). numbers[i, s, a - 1, e - 1] is the
    expected number of animals of sex s (0 female, 1 male) sampled in
    sample_years[i] with true age a and estimated age e, as returned by the
    fifth step, for ages 1 to A. Entry [i, e - 1, j, s, f - 1] of the result
    is the probability that a female sampled in sample_years[i] with
    estimated age e is the mother of an animal of sex s sampled in
    sample_years[j] with estimated age f.

    Given its sampling year, sex and estimated age, an animal's true age a
    has probability proportional to numbers[i, s, a - 1, e - 1], and the two
    animals' true ages are independent. The entry is the average, over both
    true ages, of the probability for those true ages, and it is zero when
    either class has no expected animals. For a female of true age a sampled
    in year y and an animal whose true birth year is b (its sampling year
    minus its true age), the probability is zero when b is before
    first_year; otherwise it is by expected relative reproductive output:
    her expected number of calves of the year in year b, divided by the
    total for all adult females, which is taken as the long-run proportion
    of adult females with a calf (first step) times adult female abundance
    in year b (second step). A female sampled before year b must survive from
    her sampling year to year b, at the juvenile rate for each year she
    spends at ages 1 to 5 and the adult rate after; a female sampled in or
    after year b needs no survival term.

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,), inside the domain of the second step.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    entry_age : int
        Age at which a female enters the breeding cycle, resting, an integer
        from 1 to 20.
    sample_years : np.ndarray
        Shape (Y,), Y >= 1; strictly increasing integers from 1900 to 2200.
    first_year : int
        First modelled birth year, an integer from 1900 to 2200, below
        max(sample_years).
    numbers : np.ndarray
        Shape (Y, 2, A, A), 6 <= A <= 100; finite, non-negative expected
        numbers, with A + max(sample_years) - min(sample_years) <= 200.

    Returns
    -------
    probabilities : np.ndarray
        Shape (Y, A, Y, 2, A), as described above; a probability too small
        for double precision may be returned as zero.

    Raises
    ------
    ValueError
        If theta is not an array of shape (6,) inside the domain of the second
        step, if ref_year is not an integer from 1900 to 2200, if entry_age is
        not an integer from 1 to 20, if sample_years is not a non-empty,
        strictly increasing one-dimensional array of integers from 1900 to
        2200, if first_year is not an integer from 1900 to 2200 below
        max(sample_years), or if numbers is not a finite, non-negative array
        of shape (Y, 2, A, A) with 6 <= A <= 100 and
        A + max(sample_years) - min(sample_years) <= 200.
    '''
    return probabilities  # placeholder
```

### Step 7

07_half_sibling_probabilities.py

Goal
----
Probability that two sampled animals are maternal half-siblings, for samples classified by sampling year, sex and estimated age.

```python
def half_sibling_probabilities(theta: "np.ndarray", ref_year: int, sample_years: "np.ndarray", first_year: int,
                               numbers: "np.ndarray") -> "np.ndarray":
    '''Probability that two sampled animals are maternal half-siblings, for samples classified by sampling year, sex and estimated age.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3)
    as in the second step, with the breeding cycle of the first step.
    numbers[i, s, a - 1, e - 1] is as returned by the fifth step for animals
    sampled in sample_years (ages 1 to A). Entry [i, s, e - 1, j, t, f - 1]
    of the result is the probability that an animal of sex s sampled in
    sample_years[i] with estimated age e and an animal of sex t sampled in
    sample_years[j] with estimated age f have the same mother.

    Given its sampling year, sex and estimated age, an animal's true age a
    has probability proportional to numbers[i, s, a - 1, e - 1], and the two
    animals' true ages are independent. The entry is the average, over both
    true ages, of the probability for those true ages, and it is zero when
    either class has no expected animals. For true birth years b and b'
    (sampling year minus true age), the probability is zero when b = b' or
    when either is before first_year. Otherwise let b < b' be the earlier and
    the later of the two, whichever animal was born first. The earlier-born
    animal's mother had a calf of the year in year b and was alive one year
    later. She is the later-born animal's mother with her expected share of
    the calves of the year born in year b': she must survive, at the adult
    rate, from year b + 1 to year b' + 1, and have a calf of the year in
    year b'; the total over all adult females is taken as the long-run
    proportion of adult females with a calf (first step) times adult female
    abundance in year b' (second step).

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,), inside the domain of the second step.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    sample_years : np.ndarray
        Shape (Y,), Y >= 1; strictly increasing integers from 1900 to 2200.
    first_year : int
        First modelled birth year, an integer from 1900 to 2200, below
        max(sample_years).
    numbers : np.ndarray
        Shape (Y, 2, A, A), 6 <= A <= 100; finite, non-negative expected
        numbers.

    Returns
    -------
    probabilities : np.ndarray
        Shape (Y, 2, A, Y, 2, A), as described above; a probability too
        small for double precision may be returned as zero.

    Raises
    ------
    ValueError
        If theta is not an array of shape (6,) inside the domain of the second
        step, if ref_year is not an integer from 1900 to 2200, if sample_years
        is not a non-empty, strictly increasing one-dimensional array of
        integers from 1900 to 2200, if first_year is not an integer from 1900
        to 2200 below max(sample_years), or if numbers is not a finite,
        non-negative array of shape (Y, 2, A, A) with 6 <= A <= 100.
    '''
    return probabilities  # placeholder
```

### Step 8

08_self_recapture_probabilities.py

Goal
----
Probability that a female sampled in one year is the same animal as a female sampled in a later year, by the first sample's estimated age and the second sample's stage.

```python
def self_recapture_probabilities(theta: "np.ndarray", ref_year: int, sample_years: "np.ndarray",
                                 numbers: "np.ndarray") -> "np.ndarray":
    '''Probability that a female sampled in one year is the same animal as a female sampled in a later year, by the first sample's estimated age and the second sample's stage.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3)
    as in the second step. numbers[i, s, a - 1, e - 1] is as returned by the
    fifth step for animals sampled in sample_years (ages 1 to A). Entry
    [i, e - 1, m, d] of the result is the probability that a female sampled
    without harm in sample_years[i] with estimated age e is the same animal
    as a female sampled in sample_years[m] in stage d (d = 0 juvenile, ages 1
    to 5; d = 1 adult, 6 or older), whose stage is known without error.

    Given its sampling year and estimated age, the first female's true age a
    has probability proportional to numbers[i, 0, a - 1, e - 1], and the
    entry is the average over a of the probability for that true age; it is
    zero when the class has no expected animals and whenever
    sample_years[m] <= sample_years[i]. For true age a, the first female must
    survive the years between the two samples (the juvenile rate for each
    year spent at ages 1 to 5, the adult rate after) and be in stage d at
    the second sample, and is then equally likely to be any female of that
    stage in that year (stage abundances from the second step).

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,), inside the domain of the second step.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    sample_years : np.ndarray
        Shape (Y,), Y >= 1; strictly increasing integers from 1900 to 2200.
    numbers : np.ndarray
        Shape (Y, 2, A, A), 6 <= A <= 100; finite, non-negative expected
        numbers.

    Returns
    -------
    probabilities : np.ndarray
        Shape (Y, A, Y, 2), as described above; a probability too small for
        double precision may be returned as zero.

    Raises
    ------
    ValueError
        If theta is not an array of shape (6,) inside the domain of the second
        step, if ref_year is not an integer from 1900 to 2200, if sample_years
        is not a non-empty, strictly increasing one-dimensional array of
        integers from 1900 to 2200, or if numbers is not a finite,
        non-negative array of shape (Y, 2, A, A) with 6 <= A <= 100.
    '''
    return probabilities  # placeholder
```

### Step 9

09_comparison_counts.py

Goal
----
Number of pairwise comparisons of one kinship type in each class of estimated ages, leaving out the pairs that are not compared.

```python
def comparison_counts(numbers: "np.ndarray", sample_years: "np.ndarray", first_year: int, entry_age: int,
                      kind: str) -> "np.ndarray":
    '''Number of pairwise comparisons of one kinship type in each class of estimated ages, leaving out the pairs that are not compared.

    numbers[i, s, a - 1, e - 1] is the expected number of animals of sex s (0
    female, 1 male) sampled in sample_years[i] with true age a and estimated
    age e, as returned by the fifth step, for ages 1 to A. Every expected
    sample is a distinct animal and enters every comparison type it is
    eligible for; the number of comparisons between two classes is the
    product of their expected numbers of animals by estimated age (summed
    over true ages). Only females can be mothers or self-recaptures. The
    rules use estimated ages: an animal's estimated birth year is its
    sampling year minus its estimated age, and the age at first birth is
    entry_age + 2.
    kind "MOP": shape (Y, A, Y, 2, A); entry [i, e - 1, j, s, f - 1] counts
    comparisons of a female sampled in sample_years[i] with estimated age e
    with an animal of sex s sampled in sample_years[j] with estimated age f,
    when the latter's estimated birth year is first_year or later, leaving
    out every pair sampled in the same year in which f is below the age at
    first birth.
    kind "HSP": shape (Y, 2, A, Y, 2, A); entry [i, s, e - 1, j, t, f - 1]
    counts comparisons of an animal of sex s sampled in sample_years[i] with
    estimated age e with an animal of sex t sampled in sample_years[j] with
    estimated age f, each pair once, when both estimated birth years are
    first_year or later and the second animal's estimated birth year minus
    the first's is positive and below twice the age at first birth plus two
    years, that is, below 2 (entry_age + 2) + 2; zero otherwise.
    kind "SP": shape (Y, A, Y, 2); entry [i, e - 1, m, d] counts comparisons
    of a female sampled in sample_years[i] with estimated age e with a female
    sampled in a later year sample_years[m] whose stage is d (d = 0 true ages
    1 to 5, d = 1 true ages 6 or older; the stage is known without error),
    whatever their birth years; zero when sample_years[m] <= sample_years[i].

    Parameters
    ----------
    numbers : np.ndarray
        Shape (Y, 2, A, A), 6 <= A <= 100; finite, non-negative expected
        numbers.
    sample_years : np.ndarray
        Shape (Y,), Y >= 1; strictly increasing integers from 1900 to 2200.
    first_year : int
        First modelled birth year, an integer from 1900 to 2200.
    entry_age : int
        Age at which females enter the breeding cycle, an integer from 1 to
        20.
    kind : str
        "MOP", "HSP" or "SP".

    Returns
    -------
    counts : np.ndarray
        Shape as given for kind; non-negative.

    Raises
    ------
    ValueError
        If numbers is not a finite, non-negative array of shape (Y, 2, A, A)
        with 6 <= A <= 100 matching sample_years, if sample_years is not a
        non-empty, strictly increasing one-dimensional array of integers from
        1900 to 2200, if first_year is not an integer from 1900 to 2200, if
        entry_age is not an integer from 1 to 20, or if kind is not "MOP",
        "HSP" or "SP".
    '''
    return counts  # placeholder
```

### Step 10

10_pseudo_fisher_information.py

Goal
----
Expected pseudo-Fisher information about the six demographic parameters from all pairwise comparisons of a sampling design with estimated ages.

```python
def pseudo_fisher_information(theta: "np.ndarray", ref_year: int, entry_age: int, numbers: "np.ndarray",
                              sample_years: "np.ndarray", first_year: int, use_kin: bool) -> "np.ndarray":
    '''Expected pseudo-Fisher information about the six demographic parameters from all pairwise comparisons of a sampling design with estimated ages.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3)
    as in the second step, and the information is about theta on exactly
    this scale. numbers describes the design by sampling year, sex, true age
    and estimated age, as returned by the fifth step for sample_years; it is
    fixed by the design and does not depend on theta. Each comparison's
    kinship outcome is a Poisson count with mean equal to its kinship
    probability for the two classes of estimated age (the sixth, seventh and
    eighth steps, with first_year the first modelled birth year), in a
    pseudo-likelihood over all compared pairs (the numbers of comparisons
    from the ninth step). Return the expected information of that
    pseudo-likelihood, summed over the mother-offspring, half-sibling and
    self-recapture comparisons when use_kin is True, and over the
    self-recapture comparisons alone when it is False. Derivatives must be
    exact or accurate to a relative 1e-8.

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,), inside the domain of the second step.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    entry_age : int
        Age at which females enter the breeding cycle, resting, an integer
        from 1 to 20.
    numbers : np.ndarray
        Shape (Y, 2, A, A), 6 <= A <= 100; finite, non-negative expected
        numbers, with A + max(sample_years) - min(sample_years) <= 200.
    sample_years : np.ndarray
        Shape (Y,), Y >= 1; strictly increasing integers from 1900 to 2200.
    first_year : int
        First modelled birth year, an integer from 1900 to 2200, below
        max(sample_years).
    use_kin : bool
        True to include the mother-offspring and half-sibling comparisons.

    Returns
    -------
    information : np.ndarray
        Shape (6, 6), symmetric positive semi-definite, entries in the order
        of theta.

    Raises
    ------
    ValueError
        If any argument is invalid as in the sixth to ninth steps, or if
        use_kin is not a bool.
    '''
    return information  # placeholder
```

### Step 11

11_design_precision.py

Goal
----
Expected precision of a sampling design with estimated ages for adult female abundance, adult survival and the calving proportion, and the expected numbers of kin pairs of each type.

```python
def design_precision(theta: "np.ndarray", ref_year: int, entry_age: int, numbers: "np.ndarray",
                     sample_years: "np.ndarray", first_year: int, target_year: int) -> "np.ndarray":
    '''Expected precision of a sampling design with estimated ages for adult female abundance, adult survival and the calving proportion, and the expected numbers of kin pairs of each type.

    Arguments are as in the tenth step, with target_year the year of the
    abundance of interest. The covariance of the estimates is the inverse of
    the expected information of the tenth step; with self-recaptures alone
    only ln N_ref, r, logit phi_A and logit phi_J are estimated, so the
    inverse of that 4 x 4 block is used. Standard errors of derived
    quantities follow from the delta method, and the CV of an abundance
    estimate is its standard error divided by the abundance. Expected numbers
    of kin pairs are sums, over classes of estimated age, of numbers of
    comparisons (ninth step) times kinship probabilities (sixth to eighth
    steps).

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,), inside the domain of the second step.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    entry_age : int
        Age at which females enter the breeding cycle, resting, an integer
        from 1 to 20.
    numbers : np.ndarray
        Shape (Y, 2, A, A), 6 <= A <= 100; finite, non-negative expected
        numbers, with A + max(sample_years) - min(sample_years) <= 200.
    sample_years : np.ndarray
        Shape (Y,), Y >= 1; strictly increasing integers from 1900 to 2200.
    first_year : int
        First modelled birth year, an integer from 1900 to 2200, below
        max(sample_years).
    target_year : int
        Year of the adult female abundance of interest, an integer from 1900
        to 2200.

    Returns
    -------
    precision : np.ndarray
        Shape (7,): [0] the expected CV of adult female abundance in
        target_year from all three kinship types; [1] the same CV from
        self-recaptures alone; [2] the expected standard error of adult
        survival phi_A and [3] of the long-run proportion of adult females
        with a calf, both from all three kinship types; [4], [5] and [6] the
        expected numbers of mother-offspring pairs, half-sibling pairs and
        self-recaptures among the compared pairs.

    Raises
    ------
    ValueError
        If any argument is invalid as in the tenth step, if target_year is
        not an integer from 1900 to 2200, or if either information matrix
        used is not positive definite.
    '''
    return precision  # placeholder
```

### Step 12

12_smallest_design_expected_kin_pairs.py

Goal
----
Total expected number of mother-offspring pairs, half-sibling pairs and self-recaptures at the smallest annual sample size of a new survey programme that reaches a target CV for adult female abundance, with ages from a fitted clock.

```python
def smallest_design_expected_kin_pairs(theta: "np.ndarray", ref_year: int, entry_age: int, first_year: int,
                                       past_years: "np.ndarray", past_size: float, new_years: "np.ndarray",
                                       size_step: int, max_size: int, max_female_age: int, max_male_age: int,
                                       target_year: int, cv_target: float, reference_true_ages: "np.ndarray",
                                       reference_estimated_ages: "np.ndarray") -> float:
    '''Total expected number of mother-offspring pairs, half-sibling pairs and self-recaptures at the smallest annual sample size of a new survey programme that reaches a target CV for adult female abundance, with ages from a fitted clock.

    theta = (ln N_ref, r, logit phi_A, logit phi_J, logit psi2, logit psi3)
    holds the true values, as in the second step. The standard deviation of
    the clock's error is the maximum-likelihood value of the fourth step from
    the reference animals' true and estimated ages, with max_age =
    max_female_age. The design samples past_size animals in each of
    past_years and n animals in each of new_years, each year's sample
    composed by true age as in the third step at the true growth rate and
    survival rates, with females aged 1 to max_female_age and males aged 1
    to max_male_age, and each animal's estimated age distributed as in the
    fifth step with that standard deviation. For n = size_step,
    2 size_step, ..., max_size, compute the design precision of the eleventh
    step with first_year as the first modelled birth year and entry_age as
    the age at which females enter the breeding cycle, and take the smallest
    n whose expected CV of adult female abundance in target_year from all
    three kinship types is at most cv_target. Return the sum of the expected
    numbers of mother-offspring pairs, half-sibling pairs and self-recaptures
    of the eleventh step at that n.

    Parameters
    ----------
    theta : np.ndarray
        Shape (6,), inside the domain of the second step.
    ref_year : int
        Year to which N_ref refers, an integer from 1900 to 2200.
    entry_age : int
        Age at which females enter the breeding cycle, resting, an integer
        from 1 to 20.
    first_year : int
        First modelled birth year, an integer from 1900 to 2200, below
        max(new_years).
    past_years : np.ndarray
        Shape (P,), P >= 1; strictly increasing integers from 1900 to 2200.
    past_size : float
        Animals sampled in each past year, a finite number from 0 to 1e6.
    new_years : np.ndarray
        Shape (Q,), Q >= 1; strictly increasing integers from 1900 to 2200,
        all later than every past year.
    size_step : int
        Spacing of the annual sample sizes tried, an integer from 1 to 10000.
    max_size : int
        Largest annual sample size tried, a multiple of size_step from
        size_step to 100000.
    max_female_age : int
        Oldest age at which females are sampled, an integer from 6 to 100,
        with max_female_age + max(new_years) - min(past_years) <= 200.
    max_male_age : int
        Oldest age at which males are sampled, an integer from 0 to 5.
    target_year : int
        Year of the adult female abundance of interest, an integer from 1900
        to 2200.
    cv_target : float
        Target CV, a finite number with 0 < cv_target < 1.
    reference_true_ages : np.ndarray
        Shape (K,), K >= 1; true ages of the reference animals, as in the
        fourth step.
    reference_estimated_ages : np.ndarray
        Shape (K,); their estimated ages, as in the fourth step.

    Returns
    -------
    expected_pairs : float
        The total expected number of mother-offspring pairs, half-sibling
        pairs and self-recaptures at the smallest adequate annual sample size,
        as a native Python float.

    Raises
    ------
    ValueError
        If any argument is outside its stated range or invalid as in the
        third, fourth, fifth or eleventh step, if a past year is not earlier
        than every new year, or if no annual sample size up to max_size
        reaches cv_target.
    '''
    return expected_pairs  # placeholder
```
