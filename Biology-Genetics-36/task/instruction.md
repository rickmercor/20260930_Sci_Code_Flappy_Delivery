# Biology-Genetics-36

## Background

The variation observed in a sample of DNA sequences from a short, non-recombining locus is the outcome of two random processes: the genealogy that connects the sampled sequences to their common ancestor, and the mutations that fall on its branches. Population history leaves its signature in the genealogy, because the size of the population while a given number of ancestral lineages existed sets how long those lineages waited before coalescing, and therefore how many mutations they collected and how many sampled sequences inherit each of them. Using observed patterns of segregating sites to discriminate between candidate histories, and computing how probable a particular configuration of allele counts is, are central tasks of population genetics, and both rest on the distribution of the lengths of the branches of a sample genealogy.

## Problem

The number of segregating sites at a short non-recombining locus and the frequencies of their derived alleles are both shaped by the genealogy of the sample: the number of sites is a Poisson count whose mean is proportional to the total branch length of the genealogy, and the derived count of a site is the number of sampled sequences that descend from the branch on which its mutation arose. A change in population size during the history of the sample changes the lengths of the branches of particular sizes, so the joint probability of an observed pattern of sites and derived counts discriminates between size histories, and recent theory provides the moments of the total branch length and the probabilities of single-mutation patterns under a model in which the effective size is constant while a given number of ancestral lineages exists. In this problem the inputs are the patterns observed at three loci and two candidate histories; the output is a single number, the probability of the observed derived counts given the observed numbers of segregating sites, under the history the data support.

Model the sample of 14 sequences under the coalescent in which the effective size is constant while a given number of ancestral lineages exists and can change only when two lineages coalesce, with mutation under the infinite-sites model at the rate theta = 4 N_ref mu = 8 x 10^-4 per site, N_ref being the present-day effective size, so that a locus of L sites has the mutation parameter 8 x 10^-4 L. Three unlinked loci are sequenced in the same 14 sequences, so their genealogies are independent draws from the same history, and an outgroup gives the ancestral state at every site. Locus A (550 bp) shows exactly two segregating sites, whose derived alleles are carried by 6 sequences at one site and by 8 sequences at the other, in either order. Locus B (400 bp) shows exactly one segregating site, whose derived allele is carried by 12 sequences. Locus C (500 bp) shows exactly one segregating site, whose derived allele is carried by a single sequence. Compare two histories: history 1 keeps the size N_ref throughout, and history 2 has the size 0.15 N_ref while 3, 4 or 5 ancestral lineages existed and the size N_ref while any other number existed.

Decide between the histories by the joint probability of the three observations, the product of the probability that locus A shows exactly two segregating sites with derived counts 6 and 8, the probability that locus B shows exactly one segregating site with derived count 12, and the probability that locus C shows exactly one segregating site with derived count 1; take the history with the larger joint probability, history 1 on a tie. Under the decided history, also evaluate the probability of exactly two segregating sites at locus A by the alternating series obtained by expanding the exponential factor of the Poisson probability in powers of theta and taking expectations term by term, so that the partial sum of index h contains the terms of order theta^2 to theta^(2 + h), and find the smallest h whose partial sum is within 1 per cent of the exact probability; and derive the fifth central moment of the number of segregating sites at locus A from the model stated here. Where a derivation from this model disagrees with a published formula, the derivation governs.

Report as the final answer, under the decided history, the product of three conditional probabilities: the probability of the derived counts 6 and 8 at locus A given that it shows exactly two segregating sites, the probability of the derived count 12 at locus B given that it shows exactly one, and the probability of the derived count 1 at locus C given that it shows exactly one. In the reasoning, report under each history the probability of exactly two segregating sites at locus A, the probability of the two-site pattern at locus A, the probability of the single site of count 12 at locus B and the probability of the single site of count 1 at locus C; the ratio of the joint probabilities of the two histories, history 2 over history 1, and the decided history; and, under the decided history, the probability of the count 12 at locus B given one site, the probability of the count 1 at locus C given one site, the fifth central moment of the number of segregating sites at locus A, and the smallest series index that reaches 1 per cent. State briefly the model you used and each convention you adopted wherever the configuration leaves a choice open.

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

01_tree_length_moments.py

Goal
----
Power moments of the total branch length of a sample genealogy under a size history indexed by the number of ancestral lineages.

```python
def tree_length_moments(relative_sizes: "np.ndarray", max_order: int) -> "np.ndarray":
    '''Power moments of the total branch length of a sample genealogy under a size history indexed by the number of ancestral lineages.

    The sample holds n = len(relative_sizes) + 1 sequences. Their genealogy
    passes through the states with n, n - 1, ..., 2 ancestral lineages.
    relative_sizes[j] is the effective population size while j + 2 ancestral
    lineages exist, relative to a reference size N_ref, so the entry for the
    state with two lineages comes first and the entry for the state with n
    lineages last. Time is measured in units of 4 N_ref generations: while k
    lineages exist, the waiting time until two of them coalesce is
    exponentially distributed with rate k (k - 1) / relative_sizes[k - 2],
    independently of the waiting times of the other states. The total branch
    length L of the genealogy is the sum over the states of k times the waiting
    time spent with k lineages. Return the power moments E[L^m] for
    m = 0, 1, ..., max_order.

    Parameters
    ----------
    relative_sizes : np.ndarray
        Shape (n - 1,), n >= 2; finite entries > 0, one per coalescent state as
        described above.
    max_order : int
        Highest moment order to return, >= 0.

    Returns
    -------
    moments : np.ndarray
        Shape (max_order + 1,). moments[m] = E[L^m]; moments[0] = 1. Every entry
        is accurate to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If relative_sizes is not a one-dimensional array with at least one
        entry, if any of its entries is not a finite number > 0, or if
        max_order is not an integer >= 0.
    '''
    return moments  # placeholder
```

### Step 2

02_lineage_size_pair_counts.py

Goal
----
Expected numbers of ordered pairs of lineages of given sizes in pairs of coalescent states.

```python
def lineage_size_pair_counts(sample_size: int) -> "np.ndarray":
    '''Expected numbers of ordered pairs of lineages of given sizes in pairs of coalescent states.

    The genealogy of sample_size sequences follows the standard coalescent
    topology: at every coalescence a uniformly chosen pair of the lineages then
    present merges, independently of the waiting times and of all earlier
    merges. A lineage in the state with k ancestral lineages is of size i when
    exactly i of the sampled sequences descend from it, so the sizes in that
    state are positive and sum to sample_size. Let l_k(i) be the number of
    lineages of size i in the state with k lineages. Return the expected
    products E[l_k(i) l_k'(j)] for every pair of states and every pair of
    sizes: the expected number of ordered pairs (a lineage of size i in the
    state with k lineages, a lineage of size j in the state with k' lineages),
    in which the pair of a lineage with itself is counted when the two states
    coincide and i = j.

    Parameters
    ----------
    sample_size : int
        Number of sampled sequences n, >= 2.

    Returns
    -------
    counts : np.ndarray
        Shape (n - 1, n - 1, n - 1, n - 1). counts[a, b, i - 1, j - 1] is
        E[l_k(i) l_k'(j)] for the states with k = a + 2 and k' = b + 2 lineages
        and the sizes i and j from 1 to n - 1. The array is symmetric under the
        exchange of (a, i) with (b, j). Every entry is an exact rational number
        represented to a relative error below 1e-12.

    Raises
    ------
    ValueError
        If sample_size is not an integer >= 2.
    '''
    return counts  # placeholder
```

### Step 3

03_two_site_pattern_probability.py

Goal
----
Probability that a non-recombining locus shows exactly two segregating sites with given derived counts.

```python
def two_site_pattern_probability(relative_sizes: "np.ndarray", theta: float, count_a: int, count_b: int) -> float:
    '''Probability that a non-recombining locus shows exactly two segregating sites with given derived counts.

    The sample holds n = len(relative_sizes) + 1 sequences whose genealogy
    follows the coalescent of the previous steps: relative_sizes[j] is the
    effective size while j + 2 ancestral lineages exist, relative to N_ref;
    time is in units of 4 N_ref generations, so that with k lineages the
    waiting time to the next coalescence is exponential with rate
    k (k - 1) / relative_sizes[k - 2]; and at every coalescence a uniformly
    chosen pair of lineages merges. Mutations arise along the genealogy as a
    Poisson process of rate theta per unit branch length, theta being
    4 N_ref mu for the whole locus, under the infinite-sites model, so that
    every mutation is a segregating site whose derived allele is carried by the
    sequences descending from the branch on which it arose. Return the
    probability that the locus shows exactly two segregating sites and that
    their derived counts are count_a and count_b as an unordered pair: when
    count_a != count_b, one site has derived count count_a and the other
    count_b, in either assignment; when count_a == count_b, both sites have
    that derived count.

    Parameters
    ----------
    relative_sizes : np.ndarray
        Shape (n - 1,), n >= 2; finite entries > 0.
    theta : float
        Mutation parameter of the locus, 4 N_ref mu, a finite number > 0.
    count_a : int
        Derived count of one site, from 1 to n - 1.
    count_b : int
        Derived count of the other site, from 1 to n - 1.

    Returns
    -------
    probability : float
        The pattern probability as a native Python float, accurate to a
        relative error below 1e-10.

    Raises
    ------
    ValueError
        If relative_sizes is not a one-dimensional array with at least one
        finite entry > 0, if theta is not a finite number > 0, or if count_a or
        count_b is not an integer from 1 to n - 1.
    '''
    return probability  # placeholder
```

### Step 4

04_single_site_probability.py

Goal
----
Probability that a non-recombining locus shows exactly one segregating site, with a given derived count and with any count.

```python
def single_site_probability(relative_sizes: "np.ndarray", theta: float, count: int) -> "np.ndarray":
    '''Probability that a non-recombining locus shows exactly one segregating site, with a given derived count and with any count.

    The sample of n = len(relative_sizes) + 1 sequences, its genealogy, the
    size history relative_sizes (entry j for the state with j + 2 lineages,
    relative to N_ref), the time unit of 4 N_ref generations with the
    coalescence rate k (k - 1) / relative_sizes[k - 2] while k lineages exist,
    the uniform merging of lineages, and the infinite-sites mutation process of
    rate theta per unit branch length are those of the previous steps. Return
    the probability that the locus shows exactly one segregating site whose
    derived allele is carried by exactly `count` sequences, and the probability
    that it shows exactly one segregating site of any derived count.

    Parameters
    ----------
    relative_sizes : np.ndarray
        Shape (n - 1,), n >= 2; finite entries > 0.
    theta : float
        Mutation parameter of the locus, 4 N_ref mu, a finite number > 0.
    count : int
        Derived count of the site, from 1 to n - 1.

    Returns
    -------
    probabilities : np.ndarray
        Shape (2,). probabilities[0] is the probability of exactly one
        segregating site of derived count `count`; probabilities[1] is the
        probability of exactly one segregating site. Both accurate to a
        relative error below 1e-10.

    Raises
    ------
    ValueError
        If relative_sizes is not a one-dimensional array with at least one
        finite entry > 0, if theta is not a finite number > 0, or if count is
        not an integer from 1 to n - 1.
    '''
    return probabilities  # placeholder
```

### Step 5

05_site_count_statistics.py

Goal
----
Exact probability of a given number of segregating sites, the accuracy of its series evaluation, and a central moment of the number of sites.

```python
def site_count_statistics(relative_sizes: "np.ndarray", theta: float, n_sites: int, tolerance: float,
                          moment_order: int) -> "np.ndarray":
    '''Exact probability of a given number of segregating sites, the accuracy of its series evaluation, and a central moment of the number of sites.

    The sample of n = len(relative_sizes) + 1 sequences, its genealogy, the
    size history relative_sizes (entry j for the state with j + 2 lineages,
    relative to N_ref), the time unit of 4 N_ref generations with the
    coalescence rate k (k - 1) / relative_sizes[k - 2] while k lineages exist,
    and the infinite-sites mutation process of rate theta per unit branch
    length are those of the previous steps; K denotes the number of
    segregating sites of the locus. Three quantities are returned. First, the
    exact probability that K = n_sites. Second, the accuracy behaviour of the
    series evaluation of that probability: expanding the exponential factor of
    the Poisson probability of n_sites mutations in powers of theta times the
    total length and taking the expectation term by term expresses the
    probability as an alternating series in the power moments of the length,
    whose partial sum of index h keeps the terms of order theta^n_sites to
    theta^(n_sites + h); return the smallest h >= 0 at which that partial sum
    differs from the exact probability by less than `tolerance` in relative
    terms, and the partial sum itself. Third, the central moment of K of order
    moment_order, E[(K - E[K])^moment_order].

    Parameters
    ----------
    relative_sizes : np.ndarray
        Shape (n - 1,), n >= 2; finite entries > 0.
    theta : float
        Mutation parameter of the locus, 4 N_ref mu, a finite number > 0.
    n_sites : int
        Number of segregating sites whose probability is returned, >= 0.
    tolerance : float
        Relative accuracy required of the series partial sum, a finite number
        in (0, 1).
    moment_order : int
        Order of the central moment of K to return, >= 0.

    Returns
    -------
    statistics : np.ndarray
        Shape (4,). statistics[0] is the exact probability that K = n_sites;
        statistics[1] is the smallest partial-sum index h that reaches the
        tolerance, as a float holding an integer value; statistics[2] is the
        partial sum of that index; statistics[3] is the central moment of K of
        order moment_order. The exact probability and the central moment are
        accurate to a relative error below 1e-10.

    Raises
    ------
    ValueError
        If relative_sizes is not a one-dimensional array with at least one
        finite entry > 0, if theta is not a finite number > 0, if n_sites is
        not an integer >= 0, if tolerance is not a finite number in (0, 1), if
        moment_order is not an integer >= 0, or if the partial sums have not
        reached the tolerance by index 150.
    '''
    return statistics  # placeholder
```

### Step 6

06_history_decision.py

Goal
----
Joint probability of a three-locus observation under two size histories, and the decision between them.

```python
def history_decision(sizes_null: "np.ndarray", sizes_alt: "np.ndarray", theta_a: float, count_a1: int,
                     count_a2: int, theta_b: float, count_b: int, theta_c: float, count_c: int) -> "np.ndarray":
    '''Joint probability of a three-locus observation under two size histories, and the decision between them.

    Three unlinked non-recombining loci are sequenced in the same sample of
    n = len(sizes_null) + 1 sequences, so their genealogies are independent
    draws from the same history. Locus A shows exactly two segregating sites
    with derived counts count_a1 and count_a2 (an unordered pair) and has the
    mutation parameter theta_a; locus B shows exactly one segregating site
    with derived count count_b and has the mutation parameter theta_b; locus C
    shows exactly one segregating site with derived count count_c and has the
    mutation parameter theta_c. The size histories sizes_null and sizes_alt
    follow the convention of the earlier steps (entry j is the relative size
    while j + 2 lineages exist), as do the coalescent, its time unit and the
    mutation process. Return the probability of the joint observation under
    each history, and the decision: 1.0 when the joint probability under
    sizes_alt is strictly larger than under sizes_null, 0.0 otherwise.

    Parameters
    ----------
    sizes_null : np.ndarray
        Shape (n - 1,), n >= 2; finite entries > 0.
    sizes_alt : np.ndarray
        Shape (n - 1,), the same length as sizes_null; finite entries > 0.
    theta_a : float
        Mutation parameter of locus A, a finite number > 0.
    count_a1 : int
        Derived count of one site of locus A, from 1 to n - 1.
    count_a2 : int
        Derived count of the other site of locus A, from 1 to n - 1.
    theta_b : float
        Mutation parameter of locus B, a finite number > 0.
    count_b : int
        Derived count of the site of locus B, from 1 to n - 1.
    theta_c : float
        Mutation parameter of locus C, a finite number > 0.
    count_c : int
        Derived count of the site of locus C, from 1 to n - 1.

    Returns
    -------
    result : np.ndarray
        Shape (3,). result[0] and result[1] are the joint probabilities of the
        observation under sizes_null and sizes_alt, accurate to a relative
        error below 1e-10; result[2] is the decision, 1.0 for sizes_alt and
        0.0 for sizes_null.

    Raises
    ------
    ValueError
        If sizes_null or sizes_alt is not a one-dimensional array with at
        least one finite entry > 0, if their lengths differ, if theta_a,
        theta_b or theta_c is not a finite number > 0, or if count_a1,
        count_a2, count_b or count_c is not an integer from 1 to n - 1.
    '''
    return result  # placeholder
```

### Step 7

07_configuration_probability_given_site_counts.py

Goal
----
Probability of the observed derived counts at three loci given their numbers of segregating sites, under the history the observation supports.

```python
def configuration_probability_given_site_counts(sample_size: int, reduced_fraction: float, reduced_from: int,
                                                reduced_to: int, theta_per_site: float, length_a: float,
                                                count_a1: int, count_a2: int, length_b: float, count_b: int,
                                                length_c: float, count_c: int) -> float:
    '''Probability of the observed derived counts at three loci given their numbers of segregating sites, under the history the observation supports.

    A sample of sample_size sequences is sequenced at three unlinked
    non-recombining loci of length_a, length_b and length_c base pairs, whose
    genealogies are independent draws from the same history. Locus A shows
    exactly two segregating sites with derived counts count_a1 and count_a2
    (an unordered pair); locus B shows exactly one segregating site with
    derived count count_b; locus C shows exactly one segregating site with
    derived count count_c. The mutation parameter is theta_per_site =
    4 N_ref mu per site, N_ref being the present-day effective size, so a
    locus of length L has the mutation parameter theta_per_site * L. Two
    histories are compared, both in the model of the earlier steps in which
    the effective size is constant while a given number of ancestral lineages
    exists: history 1 keeps the size N_ref throughout; history 2 has the size
    reduced_fraction * N_ref while k lineages exist for every k from
    reduced_from to reduced_to inclusive, and N_ref otherwise. The history is
    decided as in the previous step, by the larger joint probability of the
    three loci's patterns, history 1 on a tie. Return, under the decided
    history, the product of three conditional probabilities: the probability
    of the derived counts count_a1 and count_a2 at locus A given that it shows
    exactly two segregating sites, the probability of the derived count
    count_b at locus B given that it shows exactly one, and the probability of
    the derived count count_c at locus C given that it shows exactly one.

    Parameters
    ----------
    sample_size : int
        Number of sampled sequences n, >= 3.
    reduced_fraction : float
        Size of history 2 while reduced_from to reduced_to lineages exist,
        relative to N_ref; a finite number > 0.
    reduced_from : int
        First number of lineages of the reduced range, from 2 to reduced_to.
    reduced_to : int
        Last number of lineages of the reduced range, from reduced_from to n.
    theta_per_site : float
        Mutation parameter per site, 4 N_ref mu, a finite number > 0.
    length_a : float
        Length of locus A in base pairs, a finite number > 0.
    count_a1 : int
        Derived count of one site of locus A, from 1 to n - 1.
    count_a2 : int
        Derived count of the other site of locus A, from 1 to n - 1.
    length_b : float
        Length of locus B in base pairs, a finite number > 0.
    count_b : int
        Derived count of the site of locus B, from 1 to n - 1.
    length_c : float
        Length of locus C in base pairs, a finite number > 0.
    count_c : int
        Derived count of the site of locus C, from 1 to n - 1.

    Returns
    -------
    probability : float
        The product of the three conditional probabilities under the decided
        history, as a native Python float, accurate to a relative error below
        1e-9.

    Raises
    ------
    ValueError
        If sample_size is not an integer >= 3, if reduced_fraction,
        theta_per_site, length_a, length_b or length_c is not a finite number
        > 0, if reduced_from and reduced_to are not integers with
        2 <= reduced_from <= reduced_to <= sample_size, or if count_a1,
        count_a2, count_b or count_c is not an integer from 1 to n - 1.
        Conditions raised by the earlier steps propagate unchanged; in
        particular the series evaluation of the site-count step raises if its
        partial sums have not reached one per cent by index 150, which needs a
        locus mutation parameter far above those of this problem.
    '''
    return probability  # placeholder
```
