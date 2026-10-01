# Biology-Ecology-46

## Background

Close-kin mark-recapture estimates the abundance and survival of wild animals from how often genetically identified relatives turn up among the animals sampled, and the probability that two sampled animals are related depends on when each was born. Birth years come from age measurements, such as growth bands in hard parts or DNA-methylation age scores, which are noisy and usually have to be calibrated against animals of known age, something rarely available for wild species. Kinship data carry some of that information themselves, because relatives born further apart are less often found, and using them to anchor an age score is an active topic in the monitoring of long-lived fish, sharks and marine mammals.

## Problem

Close-kin mark-recapture needs the birth year of every sampled animal, and wild animals are aged with scores such as band counts or DNA-methylation clocks that normally have to be calibrated against animals of known age. A recent approach calibrates the score from the kin themselves: maternal half-sibling pairs (MHSPs, two animals with the same mother) become rarer as the gap between their birth years widens, so their rates across survey gaps and across score gaps fix how the score maps onto age. In this problem the inputs are close-kin surveys of a shark population, the scores of a few newborns and the population's life history, and the output is a single number: a predicted MHSP rate for a class of comparisons the surveys summarised here do not contain.

A recovering population of a long-lived coastal shark was surveyed in 2016, 2020 and 2024. An animal dies at the instantaneous rate 0.30 per year until it matures, which it does on reaching age 8, and at the instantaneous rate ζ per year from then on, whatever the year, so that a mature animal survives t further years with probability e^(−ζt); the number of births, and with it the number of adult females, has grown by the factor e^0.05 every year for as long as the population has existed, adult females numbering N_2024 e^(0.05(t − 2024)) in year t. Each adult female breeds once every three years: the years in which she breeds are fixed for her lifetime, and the three breeding schedules are equally common among the adult females of every age. Each newborn's mother is equally likely to be any adult female that breeds in its birth year, and two animals born in the same year have the same mother c times as often as that rule alone implies. Each survey sampled at random among the animals alive in its year, except that the 2024 gear retained no animal younger than 4 years; genotyping found every MHSP among the sampled animals without error, and each pair of animals was compared once. Every animal carries a strictly positive epigenetic age score: for an animal aged a whole years (there is no maximum age) the score is gamma distributed with mean α + βa and coefficient of variation √φ, independently of everything else, and it was recorded only as the 2-unit bin (lower, upper] into which it fell. No sampled animal is of known age, but twelve newborns caught outside the surveys, all of age 0, had unbinned scores 1.59, 1.54, 1.51, 1.60, 1.50, 1.69, 1.61, 1.59, 1.35, 1.26, 1.26 and 1.36.

The surveys are summarised in eight contrast classes, each a class of comparisons with its count of MHSPs. Among the 520 animals of 2016 scoring in (3, 5] there were 108 MHSPs; between those 520 and the 480 animals of 2020 scoring in (3, 5], 71; between the 520 and the 300 animals of 2024 scoring in (3, 5], 21; between the 520 and the 150 animals of 2016 scoring in (9, 11], 14; among the 480 animals of 2020 scoring in (3, 5], 84; between those 480 and the 300 animals of 2024 scoring in (3, 5], 39; among the 150 animals of 2016 scoring in (9, 11], 12; and between those 150 and the 140 animals of 2020 scoring in (9, 11], 9.

Following the source's approach, estimate α and φ by maximum likelihood from the newborns, and then fit β, ζ, N_2024 and c to the eight counts: every comparison of a class is an independent trial whose chance of being an MHSP the model supplies, and the estimate is the parameter set that makes the eight observed counts most likely. The annual mortality of mature animals is known to be at least 0.05, and c is positive.

Under that fit, compute the expected age of a 2024 animal whose score fell in (19, 21], the expected age of a 2024 animal scoring in (7, 9] that turns out to be a maternal half-sibling of a 2016 animal scoring in (17, 19], the expected number of MHSPs per million comparisons among 2024 animals scoring in (13, 15], and the expected number per million comparisons between 2016 animals scoring in (17, 19] and 2024 animals scoring in (7, 9]. In the reasoning, report α and φ, the observed MHSP rate within the 2016 animals scoring in (3, 5], the fitted β, ζ, N_2024 and c, how many MHSPs the fit expects between the 2020 and 2024 animals scoring in (3, 5] against the 39 observed, whether any other parameter set with c > 0 and mortality at least 0.05 fits as well and why yours is the one, both expected ages and both per-million predictions, and state briefly the model you used. Also state, from the source, (i) how the bias of its low-noise pairwise-difference slope changes as score noise grows and how the full-marginalisation slope behaves across those noise scenarios, (ii) what evidence it gives for preferring a gamma to a normal error model for a strictly positive score, and (iii) what it found for c when its simulated data held no half-siblings born in the same year. Your final answer must be a single number: the expected number of MHSPs per million comparisons between 2016 animals scoring in (17, 19] and 2024 animals scoring in (7, 9].

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

01_newborn_calibration.py

Goal
----
Maximum-likelihood intercept and dispersion of a gamma age score, from the scores of newborns.

```python
def newborn_calibration(scores: "np.ndarray") -> "np.ndarray":
    '''Maximum-likelihood intercept and dispersion of a gamma age score, from the scores of newborns.

    Each newborn's score is an independent gamma variable with mean alpha and
    coefficient of variation sqrt(dispersion), so that its shape is
    1 / dispersion and its scale is dispersion * alpha. Return the pair
    (alpha, dispersion) that maximises the joint likelihood of the scores,
    each entry to a relative accuracy of 1e-11.

    Parameters
    ----------
    scores : np.ndarray
        Shape (K,), K >= 2; the newborns' scores, positive and finite, not all
        equal.

    Returns
    -------
    calibration : np.ndarray
        Shape (2,): the maximum-likelihood alpha and dispersion, in that order.

    Raises
    ------
    ValueError
        If scores is not a one-dimensional array of at least two positive,
        finite values, or if all its values are equal.
    '''
    return calibration  # placeholder
```

### Step 2

02_sampled_age_distribution.py

Goal
----
Age distribution of animals sampled at random from a stable population, at or above a gear's minimum age.

```python
def sampled_age_distribution(mortality: float, juvenile_mortality: float, maturity_age: int, growth: float, min_age: int, max_age: int) -> "np.ndarray":
    '''Age distribution of animals sampled at random from a stable population, at or above a gear's minimum age.

    An animal dies at the annual rate juvenile_mortality until it matures,
    which it does on reaching the age maturity_age, and at the annual rate
    mortality from then on. The number of births per year has grown by the
    factor exp(growth) every year for as long as the population has existed,
    and a survey samples at random among the animals alive in its year that
    are at least min_age whole years old.
    Ages are whole years 0, 1, ..., max_age, where max_age is a numerical
    cutoff: older animals are ignored and the probabilities are normalised
    over the ages retained.

    Parameters
    ----------
    mortality : float
        Annual mortality rate of mature animals, positive and finite.
    juvenile_mortality : float
        Annual mortality rate before maturity, positive and finite.
    maturity_age : int
        Age at which an animal matures, a whole number from 0 to max_age.
    growth : float
        Annual growth rate of the number of births, finite, with
        mortality + growth > 0.
    min_age : int
        Youngest age the survey gear retains, an integer from 0 to max_age.
    max_age : int
        Oldest age kept, an integer from 1 to 400.

    Returns
    -------
    distribution : np.ndarray
        Shape (max_age + 1,): the probability of each age 0, 1, ..., max_age,
        zero below min_age, summing to 1.

    Raises
    ------
    ValueError
        If mortality or juvenile_mortality is not positive and finite, if
        growth is not finite, if mortality + growth is not positive, if
        maturity_age is not an integer from 0 to max_age, or if min_age and
        max_age are not integers with 0 <= min_age <= max_age and
        1 <= max_age <= 400.
    '''
    return distribution  # placeholder
```

### Step 3

03_score_bin_probabilities.py

Goal
----
Probability that the age score of an animal of each age 0..max_age falls in the bin (lower, upper].

```python
def score_bin_probabilities(lower: float, upper: float, alpha: float, beta: float, dispersion: float, max_age: int) -> "np.ndarray":
    '''Probability that the age score of an animal of each age 0..max_age falls in the bin (lower, upper].

    The score of an animal aged a whole years is gamma distributed with mean
    alpha + beta * a and coefficient of variation sqrt(dispersion), the same
    at every age. Return, for a = 0, 1, ..., max_age, the probability that
    the score lies in (lower, upper].

    Parameters
    ----------
    lower : float
        Lower bin edge, finite, 0 <= lower < upper.
    upper : float
        Upper bin edge, finite.
    alpha : float
        Mean score at age 0, positive and finite.
    beta : float
        Increase of the mean score per year of age, positive and finite.
    dispersion : float
        Squared coefficient of variation of the score, positive and finite.
    max_age : int
        Oldest age, an integer from 0 to 400.

    Returns
    -------
    probabilities : np.ndarray
        Shape (max_age + 1,): the bin probability at each age 0, 1, ..., max_age.

    Raises
    ------
    ValueError
        If the edges are not finite with 0 <= lower < upper, if alpha, beta or
        dispersion is not positive and finite, or if max_age is not an integer
        from 0 to 400.
    '''
    return probabilities  # placeholder
```

### Step 4

04_maternal_half_sibling_matrix.py

Goal
----
Probability that two sampled animals of given ages have the same mother.

```python
def maternal_half_sibling_matrix(year_1: int, year_2: int, mortality: float, reference_females: float, growth: float, reference_year: int, within_cohort_factor: float, breeding_period: int, max_age: int) -> "np.ndarray":
    '''Probability that two sampled animals of given ages have the same mother.

    Animal 1 is sampled in year_1 and animal 2 in year_2; an animal sampled
    in year y at age a was born in year y - a. The adult females alive in year
    t number reference_females * exp(growth * (t - reference_year)), and a
    female that has matured dies at the annual rate mortality from then on. Each adult female
    breeds once every breeding_period years, the years in which she breeds
    are fixed for her lifetime, and the breeding_period breeding schedules
    are equally common among the adult females of every age. Each newborn's
    mother is equally likely to be any adult female that breeds in its birth
    year, and two animals born in the same year have the same mother
    within_cohort_factor times as often as that rule alone implies. Return
    the matrix whose entry [a1, a2] is the probability that animal 1, aged a1,
    and animal 2, aged a2, have the same mother, for a1, a2 = 0, 1, ..., max_age.

    Parameters
    ----------
    year_1 : int
        Survey year of animal 1.
    year_2 : int
        Survey year of animal 2.
    mortality : float
        Annual mortality rate of adult females, positive and finite.
    reference_females : float
        Number of adult females alive in reference_year, positive and finite.
    growth : float
        Annual growth rate of the number of adult females, finite.
    reference_year : int
        Year in which the number of adult females equals reference_females.
    within_cohort_factor : float
        Multiplier for animals born in the same year, finite and >= 0.
    max_age : int
        Oldest age, an integer from 0 to 400.

    Returns
    -------
    probabilities : np.ndarray
        Shape (max_age + 1, max_age + 1): rows indexed by the age of animal 1,
        columns by the age of animal 2.

    Raises
    ------
    ValueError
        If a year is not an integer, if mortality or reference_females is not
        positive and finite, if growth is not finite, if within_cohort_factor
        is negative or not finite, if breeding_period is not an integer of at
        least 1, or if max_age is not an integer from 0 to 400.
    '''
    return probabilities  # placeholder
```

### Step 5

05_class_pair_mhsp_probability.py

Goal
----
Probability that an animal of one survey class and a different animal of another class have the same mother.

```python
def class_pair_mhsp_probability(class_1: "np.ndarray", class_2: "np.ndarray", calibration: "np.ndarray", demography: "np.ndarray", reference_year: int, breeding_period: int, max_age: int) -> float:
    '''Probability that an animal of one survey class and a different animal of another class have the same mother.

    A class is given as [year, lower, upper, min_age]: its animals were
    sampled in survey year `year` by a gear retaining only animals at least
    min_age whole years old, and their scores fell in the bin (lower, upper].
    calibration = [alpha, beta, dispersion]: the score of an animal aged a
    whole years is gamma distributed with mean alpha + beta * a and
    coefficient of variation sqrt(dispersion), the same at every age.
    demography = [mortality, reference_females, growth,
    within_cohort_factor, juvenile_mortality, maturity_age]:
    - an animal dies at the annual rate juvenile_mortality until it matures,
      which it does on reaching the age maturity_age, and at the annual rate
      mortality from then on; a mother is mature whenever she breeds;
    - the number of births, and with it the number of adult females, has
      grown by the factor exp(growth) every year for as long as the
      population has existed, the adult females alive in year t numbering
      reference_females * exp(growth * (t - reference_year));
    - each adult female breeds once every breeding_period years, the years
      in which she breeds are fixed for life, and the breeding_period
      breeding schedules are equally common among adult females at every
      age;
    - each newborn's mother is equally likely to be any adult female that
      breeds in its birth year;
    - two animals born in the same year have the same mother
      within_cohort_factor times as often as that rule alone implies.
    Each survey samples at random among the animals alive in its year that
    its gear retains, and an animal sampled in year y at age a was born in
    year y - a. Neither animal's age is recorded: its class is all that is
    known about it, and the two animals are drawn independently. Ages run
    over the whole years 0, 1, ..., max_age, where max_age is a numerical
    cutoff. Return the probability that the two animals have the same
    mother.

    Parameters
    ----------
    class_1, class_2 : np.ndarray
        Shape (4,) each: [year, lower, upper, min_age], the year and min_age
        whole numbers, 0 <= lower < upper and 0 <= min_age <= max_age.
    calibration : np.ndarray
        Shape (3,): alpha, beta and dispersion, each positive and finite.
    demography : np.ndarray
        Shape (6,): mortality (positive), reference_females (positive),
        growth (finite, with mortality + growth positive),
        within_cohort_factor (non-negative), juvenile_mortality (positive)
        and maturity_age (a whole number from 0 to max_age), all finite.
    reference_year : int
        The year in which the adult females number reference_females.
    breeding_period : int
        The number of years between one breeding and the next, at least 1.
    max_age : int
        The oldest age carried, a whole number from 1 to 400.

    Returns
    -------
    float
        The probability that the two animals have the same mother, as a
        native Python float.

    Raises
    ------
    ValueError
        If class_1 or class_2 does not have shape (4,), calibration shape
        (3,) or demography shape (6,); if a class year or min_age is not a
        whole number, a bin is not 0 <= lower < upper, or min_age is not in
        0 <= min_age <= max_age; if an entry of calibration is not positive
        and finite; if mortality or reference_females is not positive,
        mortality + growth is not positive, or within_cohort_factor is
        negative, or juvenile_mortality is not positive; if maturity_age is
        not a whole number in 0 <= maturity_age <= max_age; if reference_year
        is not an integer; if breeding_period is
        not an integer of at least 1; if max_age is not an integer from 1 to
        400; or if the class's score bin has probability zero at every age
        its gear retains.
    '''
    return x  # placeholder
```

### Step 6

06_contrast_fit.py

Goal
----
Calibration slope, mortality, abundance and within-cohort factor most likely to have produced the observed half-sibling counts.

```python
def contrast_fit(classes: "np.ndarray", sizes: "np.ndarray", counts: "np.ndarray", newborn_estimates: "np.ndarray", growth: float, juvenile_mortality: float, maturity_age: int, reference_year: int, breeding_period: int, min_mortality: float, max_age: int) -> "np.ndarray":
    '''Calibration slope, mortality, abundance and within-cohort factor most likely to have produced the observed half-sibling counts.

    classes[k] holds the two classes of class pair k, each given as
    [year, lower, upper, min_age]; sizes[k] holds the numbers of animals in
    those two classes; and counts[k] is the number of maternal half-sibling
    pairs found among the pair's comparisons. When the two classes of a pair
    are identical (all four entries equal), the comparisons are the unordered
    pairs of distinct animals within that class; otherwise they are all pairs
    with one animal from each class. Each comparison is an independent trial
    whose probability of being a maternal half-sibling pair is the one the
    fifth step defines, at alpha and dispersion fixed to newborn_estimates
    and at the candidate beta, mortality, reference_females and
    within_cohort_factor, with the same growth, juvenile_mortality,
    maturity_age, reference_year, breeding_period and max_age throughout. The
    mortality the fit returns is the rate of mature animals: the young die at
    juvenile_mortality until they reach maturity_age. Return the
    (beta, mortality, reference_females, within_cohort_factor) that maximise
    the probability of the observed counts over beta > 0,
    mortality >= min_mortality, reference_females > 0 and
    within_cohort_factor > 0, which is unique, each entry to a relative
    accuracy of 1e-6. Parameters under which some class pair has no
    admissible probability are not candidates and do not stop the search.

    Parameters
    ----------
    classes : np.ndarray
        Shape (K, 2, 4) with K >= 1: the two classes of each class pair.
    sizes : np.ndarray
        Shape (K, 2): the number of animals in each class of each pair, whole
        numbers of at least 2 within a pair of identical classes and at least
        1 otherwise.
    counts : np.ndarray
        Shape (K,): the half-sibling pairs found, whole numbers with
        0 < counts[k] < the number of comparisons of pair k.
    newborn_estimates : np.ndarray
        Shape (2,): alpha and dispersion, each positive and finite.
    growth : float
        Annual growth rate of births and of adult females, finite.
    juvenile_mortality : float
        Annual mortality rate before maturity, positive and finite.
    maturity_age : int
        Age at which an animal matures, a whole number from 0 to max_age.
    reference_year : int
        The year in which the adult females number reference_females.
    breeding_period : int
        The number of years between one breeding and the next, at least 1.
    min_mortality : float
        The lowest annual mortality allowed, positive and finite.
    max_age : int
        The oldest age carried, a whole number from 1 to 400.

    Returns
    -------
    np.ndarray
        Shape (4,): beta, mortality, reference_females and
        within_cohort_factor, in that order.

    Raises
    ------
    ValueError
        If classes does not have shape (K, 2, 4) with K >= 1, sizes shape
        (K, 2), counts shape (K,) or newborn_estimates shape (2,); if a class
        year or min_age is not a whole number, a bin is not
        0 <= lower < upper, or min_age is not in 0 <= min_age <= max_age; if
        a size or count is not a whole number, a size is too small, or a
        count is not strictly between 0 and the pair's number of
        comparisons; if an entry of newborn_estimates is not positive and
        finite; if growth is not finite, juvenile_mortality is not positive
        and finite, maturity_age is not a whole number from 0 to max_age, or
        min_mortality is not positive and finite; if reference_year is not an integer, breeding_period is not
        an integer of at least 1, or max_age is not an integer from 1 to
        400; or if no admissible parameters give every class pair a
        probability strictly between 0 and 1.
    '''
    return x  # placeholder
```

### Step 7

07_kin_rate_per_million.py

Goal
----
Predicted maternal half-sibling pairs per million comparisons between two target classes, after calibrating the score from kin.

```python
def kin_rate_per_million(newborn_scores: "np.ndarray", contrast_classes: "np.ndarray", contrast_sizes: "np.ndarray", contrast_counts: "np.ndarray", growth: float, juvenile_mortality: float, maturity_age: int, reference_year: int, breeding_period: int, min_mortality: float, target_classes: "np.ndarray", max_age: int) -> float:
    '''Predicted maternal half-sibling pairs per million comparisons between two target classes, after calibrating the score from kin.

    Scores are gamma distributed with mean alpha + beta * a at age a and
    coefficient of variation sqrt(dispersion). The pipeline:
    1. alpha and dispersion are the maximum-likelihood values from the
       newborns' scores, as in the first step;
    2. beta, mortality, the number of adult females in reference_year and
       the within-cohort factor are the values that make the observed
       half-sibling counts of the contrast class pairs most likely, as in
       the sixth step;
    3. the result is one million times the probability that an animal of
       target_classes[0] and a different animal of target_classes[1] have
       the same mother under those values, as in the fifth step.
    Classes are given as [year, lower, upper, min_age]; births and adult
    females grow at the rate growth; an animal dies at the annual rate
    juvenile_mortality until it matures at maturity_age and at the fitted
    mortality from then on; each adult female breeds once every
    breeding_period years; and each survey samples at random among the
    animals alive in its year that its gear retains. Ages run over the whole years 0, ..., max_age.

    Parameters
    ----------
    newborn_scores : np.ndarray
        Shape (n,) with n >= 2: the unbinned scores of animals of age 0, each
        positive and finite.
    contrast_classes : np.ndarray
        Shape (K, 2, 4) with K >= 1: the two classes of each contrast pair.
    contrast_sizes : np.ndarray
        Shape (K, 2): the number of animals in each class of each pair.
    contrast_counts : np.ndarray
        Shape (K,): the half-sibling pairs found in each pair's comparisons.
    growth : float
        Annual growth rate of births and of adult females, finite.
    juvenile_mortality : float
        Annual mortality rate before maturity, positive and finite.
    maturity_age : int
        Age at which an animal matures, a whole number from 0 to max_age.
    reference_year : int
        The year whose adult females the fitted abundance counts.
    breeding_period : int
        The number of years between one breeding and the next, at least 1.
    min_mortality : float
        The lowest annual mortality allowed, positive and finite.
    target_classes : np.ndarray
        Shape (2, 4): the two classes whose comparisons are predicted.
    max_age : int
        The oldest age carried, a whole number from 1 to 400.

    Returns
    -------
    float
        One million times the predicted probability that a comparison of the
        two target classes is a maternal half-sibling pair, as a native
        Python float.

    Raises
    ------
    ValueError
        If target_classes does not have shape (2, 4), or if any argument
        fails the conditions of the step it feeds: the newborn scores, the
        contrast classes, sizes and counts, growth, juvenile_mortality,
        maturity_age, reference_year, breeding_period, min_mortality and
        max_age.
    '''
    return x  # placeholder
```
