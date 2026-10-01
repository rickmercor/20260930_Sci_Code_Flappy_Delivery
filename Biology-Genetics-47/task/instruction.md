# Biology-Genetics-47

## Background

Genetic diversity within a population is summarised by the site frequency spectrum, and the estimators of the mutation parameter and the neutrality tests built on it are interpreted against the expectation of the standard neutral model. Structural variants such as inversions, deletions and insertions are common in genomes, and a region completely linked to one carries the imprint of the variant's own history: its carriers and non-carriers form two backgrounds whose genealogies are not independent of the variant's frequency, so the neutral expectations that underlie the standard tools no longer hold there even when nothing is under selection. Whether such a region appears more or less diverse than its surroundings, and whether a neutrality test inside it signals anything at all, therefore depends on accounting for the variant, its frequency and its type, and for a presence/absence variant also on which of its two states is the derived one.

## Problem

The site frequency spectrum, the number of polymorphic sites at each derived-allele count in a sample, underlies the standard estimators of genetic diversity and the classical neutrality tests, all of which are read against the expectation of the standard neutral model, inversely proportional to the count. A structural variant that segregates in the sample and recombines with nothing in its region breaks that expectation even under strict neutrality: conditioning on the variant's sample count partitions the haplotypes into carriers and non-carriers and distorts the spectrum of every linked neutral site in a way that depends on the type of variant, and recent theory gives the exact finite-sample expectation of that conditional spectrum, its form inside inversions, deletions and insertions, and adjusted estimators that are unbiased in the presence of the variant. In this problem the inputs are the spectra of a presence/absence segment and of its neutral flanks in one sample; the output is a single number, the diversity of the segment relative to its flanks with the variant taken into account.

Model the sample of n = 34 haplotypes as drawn from a population at mutation-drift equilibrium under the standard neutral coalescent, with mutation under the infinite-sites model, no recombination inside the region, and a segment of 9,000 bp that is present on 13 of the haplotypes and absent from the other 21. Define the mutation parameter theta per site so that a neutral region of L sites is expected to hold theta L / k polymorphic sites of derived count k. Treat the presence/absence variant as neutral, biallelic, of single origin and completely linked to every site inside the segment, so that the expected spectrum of the segment's sites given the variant's count is the conditional spectrum of neutral sites linked to a focal neutral variant, which decomposes by the relation between a site's carrier set and the variant's carrier set into strictly nested, co-occurring, enclosing, complementary and strictly disjoint components; the segment's sites exist only on the haplotypes that carry the segment. Which state is derived is unknown: no outgroup sequence aligns to the segment, so neither the polarity of the variant nor the ancestral state of any site inside it is known, and the spectrum of those sites among the 13 carriers is folded into minor-allele classes 1 to 6, the class of minor count j merging the derived counts j and 13 - j. The flanking region of 60,000 bp is polarised, its spectrum over derived counts 1 to 33 is given below, and the neutral mutation parameter per site is Watterson's estimate from it.

Decide the polarity from the segment's spectrum: under each polarity (the absent state derived, a deletion whose carriers are the ancestral background; or the present state derived, an insertion whose carriers are the derived background) take the expected count of every folded class as the mutation parameter times the segment length times the folded conditional expectation of that class among the carriers, treat the classes as independent Poisson counts, and take as derived the state with the larger log-likelihood, the absent state on a tie. Under the decided polarity compute the expected values, as multiples of theta times the segment length, of Watterson's estimator and of the pairwise diversity among the 13 carriers, the expected Tajima's D obtained by substituting the expected spectrum into the statistic with its variance evaluated at the expected number of sites, and the expected number of sites in the singleton minor-allele class. Then estimate theta per site inside the segment from the folded spectrum by the class-wise adjustment, in which each class is divided by its own conditional expectation and a folded class carries the sum of the weights of the two derived counts it merges, with the weights of Watterson's estimator and with those of the pairwise diversity; give also the pairwise estimate under the rescaling adjustment, in which the standard estimator is divided by the ratio of its conditional to its neutral expectation, and the pairwise class-wise estimate under the rejected polarity. Finally, for a neutral inversion in a sample of the same size with both arrangements sequenced and the ancestral arrangement known, give the smallest count of the inverted arrangement from which the expected value of Fay and Wu's H, the pairwise diversity minus the estimator that weights each class by the square of its derived count, is negative for that count and every larger one.

Data:
* Segment: 9,000 bp, present on 13 of the 34 haplotypes; folded spectrum among the carriers, minor counts 1 to 6: 29, 14, 9, 11, 3, 6.
* Flanking region: 60,000 bp; spectrum over derived counts 1 to 33: 292, 159, 91, 71, 57, 62, 41, 35, 37, 31, 25, 20, 15, 24, 15, 19, 26, 14, 17, 13, 7, 11, 12, 16, 12, 6, 11, 16, 12, 7, 7, 14, 13.

Report as the final answer the class-wise pairwise estimate of theta per site inside the segment under the decided polarity divided by the flanking estimate. In the reasoning, report the flanking mutation parameter per site; the log-likelihood of the folded spectrum under each polarity and their difference; the sample count of the derived allele of the variant; the expected Watterson and pairwise values among the carriers as multiples of theta times the segment length, the expected Tajima's D and the expected number of singleton-class sites under the decided polarity; the class-wise estimates with Watterson and pairwise weights, the pairwise estimate under the rescaling adjustment and the pairwise class-wise estimate under the rejected polarity; and the inverted-arrangement count from which the expected H stays negative. State briefly the model you used and each convention you adopted wherever the configuration leaves a choice open.

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

01_linked_site_spectrum_components.py

Goal
----
Expected spectrum of neutral sites completely linked to a focal neutral variant of known sample count, split by the relation of each site to the variant.

```python
def linked_site_spectrum_components(n: int, i: int) -> "np.ndarray":
    '''Expected spectrum of neutral sites completely linked to a focal neutral variant of known sample count, split by the relation of each site to the variant.

    A sample of n haplotypes is drawn from a population at mutation-drift
    equilibrium that follows the standard neutral coalescent, with no
    recombination anywhere in the region and mutation under the infinite-sites
    model. One focal neutral mutation is carried by exactly i of the n
    haplotypes. Every other polymorphic site in the region is classified by
    the set of haplotypes carrying its derived allele, relative to the focal
    carrier set: strictly nested (a proper subset of the focal set),
    co-occurring (the same set), enclosing (a proper superset), complementary
    (exactly the haplotypes outside the focal set), or strictly disjoint (no
    haplotype in common with the focal set, and not complementary). Return, for
    each relation and each derived count k, the expected number of such sites
    averaged over genealogies, in units of theta L (the population-scaled
    mutation rate per site times the region length), scaled so that without
    any conditioning the expected number of sites of derived count k is
    theta L / k. The expectations are exact for the finite sample, not a
    large-sample or diffusion limit.

    Parameters
    ----------
    n : int
        Number of haplotypes in the sample, >= 2.
    i : int
        Number of haplotypes carrying the focal variant, from 1 to n - 1.

    Returns
    -------
    components : np.ndarray
        Shape (5, n - 1). Row 0 strictly nested, row 1 co-occurring, row 2
        enclosing, row 3 complementary, row 4 strictly disjoint; column k - 1
        holds the expected number of sites of derived count k (k from 1 to
        n - 1) per unit theta L. Entries outside a relation's admissible counts
        are 0. Every entry is accurate to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If n is not an integer >= 2, or if i is not an integer from 1 to n - 1.
    '''
    return components  # placeholder
```

### Step 2

02_indel_null_spectrum.py

Goal
----
Expected spectrum of neutral sites inside a presence/absence segment, among the haplotypes that carry the segment, for a stated polarity of the variant.

```python
def indel_null_spectrum(n: int, carriers: int, absence_derived: bool) -> "np.ndarray":
    '''Expected spectrum of neutral sites inside a presence/absence segment, among the haplotypes that carry the segment, for a stated polarity of the variant.

    The sample holds n haplotypes; a segment is present on exactly `carriers`
    of them and absent from the others, the presence/absence variant being a
    single-origin, neutral, biallelic variant in complete linkage with every
    site inside the segment, under the model of the previous step. Sites
    inside the segment exist only on the carriers, so they are analysed in the
    sample of `carriers` sequences and only sites that are polymorphic among
    the carriers are counted. If absence_derived is True the absent state
    arose once by a deletion, so the carriers are the ancestral background;
    otherwise the present state arose once by an insertion, so the carriers
    are the derived background. Return the expected number of polymorphic
    sites inside the segment classified by the number k of carriers that hold
    the derived allele of the site (k from 1 to carriers - 1), the derived
    allele of a site being the allele that arose by mutation, per unit
    theta L as in the previous step.

    Parameters
    ----------
    n : int
        Number of haplotypes in the sample, >= 3.
    carriers : int
        Number of haplotypes carrying the segment, from 2 to n - 1.
    absence_derived : bool
        True if the absent state is the derived state (a deletion), False if
        the present state is the derived state (an insertion).

    Returns
    -------
    spectrum : np.ndarray
        Shape (carriers - 1,). Entry k - 1 is the expected number of sites
        inside the segment whose derived allele is carried by exactly k of the
        carriers, per unit theta L. Every entry is accurate to a relative error
        below 1e-12.

    Raises
    ------
    ValueError
        If n is not an integer >= 3, if carriers is not an integer from 2 to
        n - 1, or if absence_derived is not a bool.
    '''
    return spectrum  # placeholder
```

### Step 3

03_fold_spectrum.py

Goal
----
Fold a spectrum over derived counts 1 to m - 1 into minor-allele classes 1 to floor(m / 2).

```python
def fold_spectrum(spectrum: "np.ndarray", m: int) -> "np.ndarray":
    '''Fold a spectrum over derived counts 1 to m - 1 into minor-allele classes 1 to floor(m / 2).

    Parameters
    ----------
    spectrum : np.ndarray
        Shape (m - 1,): the number (or expected number) of sites of derived
        count k at index k - 1, finite entries.
    m : int
        Number of sequences in the sample the spectrum refers to, >= 2.

    Returns
    -------
    folded : np.ndarray
        Shape (m // 2,). Entry j - 1 is the number of sites of minor count j:
        the sum of the entries for derived counts j and m - j when j < m / 2,
        and the entry for derived count m / 2 alone, counted once, when m is
        even and j = m / 2.

    Raises
    ------
    ValueError
        If m is not an integer >= 2, or if spectrum is not a one-dimensional
        array of finite numbers of length m - 1.
    '''
    return folded  # placeholder
```

### Step 4

04_spectrum_statistics.py

Goal
----
Number of sites, Watterson's estimator, pairwise diversity, Fay and Wu's estimator and Tajima's D of an unfolded spectrum, as region totals.

```python
def spectrum_statistics(counts: "np.ndarray", m: int) -> "np.ndarray":
    '''Number of sites, Watterson's estimator, pairwise diversity, Fay and Wu's estimator and Tajima's D of an unfolded spectrum, as region totals.

    counts gives the number of sites of derived count k in a sample of m
    sequences; it may hold expected (non-integer) numbers. Return the total
    number of sites S; Watterson's estimator of theta for the region, S over
    the harmonic number of m - 1 terms; the average number of pairwise
    differences pi; Fay and Wu's theta_H, the estimator that weights each
    class by the square of its derived count; and Tajima's D, the difference
    between pi and Watterson's estimator divided by the square root of its
    variance estimate as defined by Tajima (1989), with S(S - 1) in the
    second-order term, every quantity evaluated from counts as given. The
    three estimators of theta are totals over the region, not per site.

    Parameters
    ----------
    counts : np.ndarray
        Shape (m - 1,): the number of sites of derived count k at index k - 1,
        finite entries >= 0.
    m : int
        Number of sequences in the sample, >= 4.

    Returns
    -------
    stats : np.ndarray
        Shape (5,): [S, theta_W, pi, theta_H, D], with the first four as region
        totals and D dimensionless. Accurate to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If m is not an integer >= 4, if counts is not a one-dimensional array
        of finite numbers >= 0 of length m - 1, or if the total number of sites
        is below 2, for which D is not defined.
    '''
    return stats  # placeholder
```

### Step 5

05_polarity_log_likelihoods.py

Goal
----
Poisson log-likelihoods of the observed folded spectrum of a presence/absence segment under each polarity of the variant.

```python
def polarity_log_likelihoods(folded_counts: "np.ndarray", n: int, carriers: int, theta_per_site: float,
                             segment_length: float) -> "np.ndarray":
    '''Poisson log-likelihoods of the observed folded spectrum of a presence/absence segment under each polarity of the variant.

    The segment of segment_length base pairs is present on `carriers` of the n
    haplotypes; the population-scaled mutation rate per site is
    theta_per_site; the sites inside the segment are unpolarised, so their
    spectrum among the carriers is folded into minor-allele classes 1 to
    floor(carriers / 2) as in the folding step. Under each polarity the
    expected number of sites in class j is theta_per_site times segment_length
    times the expected spectrum of the segment for that polarity (the indel
    spectrum of the earlier step) folded the same way, and the classes are
    treated as independent Poisson counts. Return the log-likelihood of
    folded_counts under each polarity as the sum over classes of the observed
    count times the logarithm of the expected count, minus the expected count;
    the term in the factorials of the observed counts is the same for both
    polarities and is omitted.

    Parameters
    ----------
    folded_counts : np.ndarray
        Shape (carriers // 2,): the observed number of sites of minor count j
        at index j - 1, finite entries >= 0.
    n : int
        Number of haplotypes in the sample, >= 3.
    carriers : int
        Number of haplotypes carrying the segment, from 2 to n - 1.
    theta_per_site : float
        Population-scaled mutation rate per site, > 0.
    segment_length : float
        Length of the segment in base pairs, > 0.

    Returns
    -------
    log_likelihoods : np.ndarray
        Shape (2,): entry 0 for the absent state derived (a deletion), entry 1
        for the present state derived (an insertion). Accurate to an absolute
        error below 1e-9.

    Raises
    ------
    ValueError
        If folded_counts is not a one-dimensional array of finite numbers >= 0
        of length carriers // 2, if theta_per_site or segment_length is not a
        finite number > 0, or on any condition raised by the earlier steps for
        n and carriers.
    '''
    return log_likelihoods  # placeholder
```

### Step 6

06_sv_aware_theta_folded.py

Goal
----
Variant-aware estimate of the population-scaled mutation rate per site from the folded spectrum of a presence/absence segment.

```python
def sv_aware_theta_folded(folded_counts: "np.ndarray", n: int, carriers: int, absence_derived: bool,
                          segment_length: float, weighting: str, correction: int) -> float:
    '''Variant-aware estimate of the population-scaled mutation rate per site from the folded spectrum of a presence/absence segment.

    The segment of segment_length base pairs is present on `carriers` of the
    n haplotypes, with the polarity given by absence_derived, and its sites
    are unpolarised, so folded_counts holds their spectrum among the carriers
    in minor-allele classes 1 to floor(carriers / 2) as in the folding step.
    The estimator starts from a linear estimator of theta over the derived
    counts 1 to carriers - 1 of the carriers' sample: the sum over counts k of
    a weight times the number of sites of count k divided by the baseline
    expectation of that number per unit theta, which is segment_length / k,
    the weights summing to one; weighting "watterson" uses the weights of
    Watterson's estimator and weighting "pairwise" those of the average
    pairwise difference. With correction 1 the baseline expectation of every
    class is replaced by the class's expectation under the indel spectrum of
    the earlier step for this polarity (the expected number of sites per unit
    theta L times segment_length), and the estimate is formed on the folded
    classes: the weight of a minor-allele class is the sum of the weights of
    the two derived counts it merges (the weight of the single derived count
    for the class of equal counts), and its expectation is the indel spectrum
    folded the same way. With correction 2 the standard estimator, evaluated
    on the folded counts, is divided by the ratio of its expectation under the
    indel spectrum to its expectation under the baseline. Return the estimate
    per site.

    Parameters
    ----------
    folded_counts : np.ndarray
        Shape (carriers // 2,): the observed number of sites of minor count j
        at index j - 1, finite entries >= 0.
    n : int
        Number of haplotypes in the sample, >= 3.
    carriers : int
        Number of haplotypes carrying the segment, from 2 to n - 1.
    absence_derived : bool
        True if the absent state is derived (a deletion), False if the present
        state is derived (an insertion).
    segment_length : float
        Length of the segment in base pairs, > 0.
    weighting : str
        "watterson" or "pairwise".
    correction : int
        1 for the class-wise replacement of the baseline expectations, 2 for
        the rescaling of the standard estimator.

    Returns
    -------
    theta_hat : float
        The variant-aware estimate of theta per site, as a native Python
        float, accurate to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If folded_counts is not a one-dimensional array of finite numbers >= 0
        of length carriers // 2, if segment_length is not a finite number > 0,
        if weighting is not "watterson" or "pairwise", if correction is not 1
        or 2, or on any condition raised by the earlier steps for n, carriers
        and absence_derived.
    '''
    return theta_hat  # placeholder
```

### Step 7

07_sv_aware_relative_diversity.py

Goal
----
Variant-aware diversity of a presence/absence segment relative to its neutral flanks, at the polarity decided from the segment's spectrum.

```python
def sv_aware_relative_diversity(n: int, carriers: int, folded_counts: "np.ndarray", segment_length: float,
                                flank_counts: "np.ndarray", flank_length: float) -> float:
    '''Variant-aware diversity of a presence/absence segment relative to its neutral flanks, at the polarity decided from the segment's spectrum.

    n haplotypes are sampled from a neutrally evolving population that
    follows the standard neutral coalescent, with no recombination inside the
    region, mutation under the infinite-sites model, and a segment of
    segment_length base pairs present on `carriers` of the haplotypes and
    absent from the others, the presence/absence variant being neutral,
    biallelic, of single origin and in complete linkage with every site
    inside the segment. Sites inside the segment are unpolarised, so
    folded_counts holds their spectrum among the carriers in minor-allele
    classes 1 to floor(carriers / 2); flank_counts holds the polarised
    spectrum of a neutral flanking region of flank_length base pairs in the
    same n haplotypes, by derived count 1 to n - 1. Take the mutation
    parameter per site to be Watterson's estimate from the flanking spectrum
    (the statistics step, divided by flank_length). Decide the polarity of the
    variant from the Poisson log-likelihoods of the segment's folded spectrum
    under the two polarities at that mutation parameter (the polarity step):
    the absent state is derived when its log-likelihood is at least as large
    as the other's. Then compute the variant-aware estimate of the mutation
    parameter per site from the folded segment spectrum under the decided
    polarity, with pairwise weights and correction 1 of the estimation step,
    and return it divided by the flanking estimate.

    Parameters
    ----------
    n : int
        Number of haplotypes in the sample, >= 4.
    carriers : int
        Number of haplotypes carrying the segment, from 2 to n - 1.
    folded_counts : np.ndarray
        Shape (carriers // 2,): the observed number of sites inside the
        segment of minor count j among the carriers, at index j - 1, finite
        entries >= 0.
    segment_length : float
        Length of the segment in base pairs, > 0.
    flank_counts : np.ndarray
        Shape (n - 1,): the observed number of sites of derived count k in the
        flanking region, at index k - 1, finite entries >= 0, with at least 2
        sites in total.
    flank_length : float
        Length of the flanking region in base pairs, > 0.

    Returns
    -------
    relative_diversity : float
        The variant-aware estimate of theta per site inside the segment
        divided by the flanking estimate, as a native Python float.

    Raises
    ------
    ValueError
        If flank_length is not a finite number > 0, or on any condition raised
        by the earlier steps for the given or derived inputs.
    '''
    return relative_diversity  # placeholder
```
