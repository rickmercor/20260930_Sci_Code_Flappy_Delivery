# Biology-Genetics-31

## Background

Gene flow between diverged populations is pervasive, and separating it from the allele sharing that incomplete lineage sorting produces is a standing problem in population genetics. The site-pattern tests that dominate the field were designed to summarize whole genomes rather than to say where along a chromosome foreign ancestry sits.

Attention has since shifted to locating individual introgressed segments, which exposes two weaknesses of those tests: they need an outgroup to assign ancestral states, and they are sensitive to how many lineages each population contributed. Recent methods address both by counting alleles restricted to particular subsets of populations and by borrowing a subsampling correction from the estimation of allelic diversity.

The result is a signed imbalance whose sign identifies which pair of populations exchanged genetic material. Such methods are judged by precision and recall against coalescent simulations with known introgressed tracts.

## Problem

Introgression between diverged lineages leaves an excess of alleles shared between the recipient population and the donor, and the classical four-taxon site-pattern test detects that excess genome wide. It needs an outgroup to polarize every site into ancestral and derived states, it discards sites carrying more than two alleles, and it is unreliable on segments short enough to be interesting as candidate introgressed haplotypes. A recent statistic drops all three requirements by working inside a rooted three-population tree ((P1, P2), P3), by counting alleles confined to a pair of populations, and by applying an allelic rarefaction correction so that populations sequenced to different depths are compared on equal terms.

Incomplete lineage sorting produces sharing of both kinds even without gene flow, and the statistic is signed so that its value identifies which sister population received material from P3. Uneven and incomplete sampling across the three populations would bias any such comparison, and removing that bias is what the allelic rarefaction correction is for.

Your task is to solve one concrete deterministic example of this pipeline on the data below, in which each string gives one site with one character per haploid lineage, digits naming the alleles observed there and `.` marking a lineage whose genotype is missing. Sampling is uneven, with 6 lineages in P1, 5 in P2 and 4 in P3, and the configuration is as follows:

- `positions = [40, 180, 430, 620, 880, 1150, 1320, 1490, 1780, 2050, 2260, 2410, 2640, 2870, 3090, 3240, 3510, 3660, 3920, 4130, 4380, 4520, 4740, 4960, 5080, 5230, 5470, 5610, 5820, 5950]`
- `P1 = ["000000", "100000", ".101.1", "000..0", "000000", ".0.110", "0000..", "0000.0", "022000", "000000", "000000", "000111", "000000", "110010", "111111", "000000", "000.00", "111111", "020000", "000000", "1.1111", "011010", "000000", "00.000", "000000", "020000", "00.220", ".0.000", "0111.1", "000000"]`
- `P2 = ["0000.", "00000", "00000", "0.110", "00000", "00000", "00010", "0010.", "00010", "10110", "00000", "10111", "000.0", "00000", "00000", "0.000", "11001", "000.0", "11110", "11.00", "000.0", "10100", "11111", "00010", "10101", "10001", "11111", "0100.", "11010", "00000"]`
- `P3 = ["010.", "1100", "0000", "1011", ".000", "0010", "0100", "1111", "1110", "0110", "0000", "0110", "0011", "0000", "11.1", "1101", "1111", "1001", "11.1", "1111", "0111", "0101", "0110", "....", ".11.", "0111", "1011", "1100", ".000", "0000"]`
- `window_size = 500`, non-overlapping windows anchored at position 0, windows for which the statistic is undefined do not serve as blocks
- `n_replicates = 2000`
- `seed = 2026`
- block resampling: draw the replicates from a single `numpy.random.default_rng(2026)` stream, one replicate at a time, each drawing `B` window indices with `integers(0, B, size=B)`, where `B` is the number of windows used

Report the genome-wide value of the statistic and the standard error the block resampling gives it. Your final answer must be a single number: the mean of the 2000 block bootstrap replicates of the genome-wide statistic.

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

01_site_allele_counts

Goal
----
Parse the per-site genotype strings of three populations into allele count

tables and non-missing sample sizes, one table per site.

```python
def site_allele_counts(P1_sites: list[str], P2_sites: list[str], P3_sites: list[str]) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Tabulate allele counts and non-missing sample sizes at every site.

    Parameters
    ----------
    P1_sites : sequence of str
        Genotype strings for P1, one per site, one character per lineage. A
        digit names an allele and '.' marks a missing lineage.
    P2_sites : sequence of str
        Genotype strings for P2, in the same format and the same site order.
    P3_sites : sequence of str
        Genotype strings for P3, in the same format and the same site order.

    Returns
    -------
    site_tables : tuple
        The pair (counts, totals). counts holds, per site, one row
        [N_i1, N_i2, N_i3] for each allele observed at that site, ordered by the
        sorted allele labels. totals holds, per site, the triple
        [N_1, N_2, N_3] of non-missing sample sizes.

    Raises
    ------
    ValueError
        If the three sequences differ in length or are empty, if a population
        does not use the same number of lineages at every site, or if a genotype
        string contains a character that is neither a digit nor '.'.
    """
    return site_tables
```

### Step 2

02_rarefaction_subsample_sizes

Goal
----
Fix the rarefaction subsample size g used at each site from that site's

non-missing sample sizes in the three populations.

```python
def rarefaction_subsample_sizes(totals: list[list[int]]) -> list[int]:
    """Determine the per-site rarefaction subsample size.

    Parameters
    ----------
    totals : sequence of sequence of int
        Per site, the triple [N_1, N_2, N_3] of non-missing sample sizes in P1,
        P2 and P3.

    Returns
    -------
    subsample_sizes : list of int
        The subsample size g used at each site.

    Raises
    ------
    ValueError
        If totals is empty, if a site does not carry exactly three sample sizes,
        or if any sample size is negative.
    """
    return subsample_sizes
```

### Step 3

03_rarefied_pair_private_counts

Goal
----
Compute, for one site, the two rarefied counts of alleles confined to a pair of

populations that includes P3.

```python
def rarefied_pair_private_counts(counts: list[list[int]], totals: list[int], g: int) -> tuple[float, float]:
    """Rarefied counts of alleles confined to P1 with P3 and to P2 with P3.

    Parameters
    ----------
    counts : sequence of sequence of int
        One row [N_i1, N_i2, N_i3] per allele observed at the site.
    totals : sequence of int
        The triple [N_1, N_2, N_3] of non-missing sample sizes at the site.
    g : int
        The rarefaction subsample size at the site.

    Returns
    -------
    pair_counts : tuple of float
        The pair (pi_P1P3, pi_P2P3), where pi_P1P3 is the rarefied number of
        alleles present in P1 and P3 and absent from P2, and pi_P2P3 is the
        rarefied number present in P2 and P3 and absent from P1.

    Raises
    ------
    ValueError
        If counts is empty or a row does not carry three entries, if any count is
        negative, if the per-population column sums do not match totals, or if g
        is negative or exceeds one of the sample sizes.
    """
    return pair_counts
```

### Step 4

04_rarefied_allelic_richness

Goal
----
Compute the rarefied allelic richness carried jointly by the three populations

at one site, which is the normalizer of the introgression statistic.

```python
def rarefied_allelic_richness(counts: list[list[int]], totals: list[int], g: int) -> float:
    """Rarefied allelic richness of one site, the normalizer of the statistic.

    Parameters
    ----------
    counts : sequence of sequence of int
        One row [N_i1, N_i2, N_i3] per allele observed at the site.
    totals : sequence of int
        The triple [N_1, N_2, N_3] of non-missing sample sizes at the site.
    g : int
        The rarefaction subsample size at the site.

    Returns
    -------
    kappa : float
        The rarefied allelic richness at the site.

    Raises
    ------
    ValueError
        If counts is empty or a row does not carry three entries, if any count is
        negative, if the per-population column sums do not match totals, or if g
        is negative or exceeds one of the sample sizes.
    """
    return kappa
```

### Step 5

05_window_dstar_sums

Goal
----
Assign informative sites to non-overlapping genomic windows and accumulate the

numerator and denominator of the introgression statistic within each window.

```python
def window_dstar_sums(positions: list[int], pair_counts: list[list[float]], richness: list[float], window_size: int) -> tuple[list[int], list[float], list[float]]:
    """Accumulate the statistic's numerator and denominator per window.

    Parameters
    ----------
    positions : sequence of int
        Chromosomal position of each site, in base pairs.
    pair_counts : sequence of sequence of float
        Per site, the pair [pi_P1P3, pi_P2P3] of rarefied private allele counts.
    richness : sequence of float
        Per site, the rarefied allelic richness kappa.
    window_size : int
        Width of the non-overlapping windows, in base pairs.

    Returns
    -------
    window_sums : tuple
        The triple (window_indices, numerators, denominators), holding the
        indices of the retained windows in increasing order, the summed signed
        imbalance in each retained window, and the summed rarefied allelic
        richness in each retained window.

    Raises
    ------
    ValueError
        If the three per-site sequences differ in length or are empty, if
        window_size is not a positive integer, if any position is negative, or if
        any allelic richness is negative.
    """
    return window_sums
```

### Step 6

06_block_bootstrap_moments

Goal
----
Resample the retained windows with replacement to obtain the mean and standard

deviation of the genome-wide introgression statistic.

```python
def block_bootstrap_moments(numerators: list[float], denominators: list[float], n_replicates: int, seed: int) -> tuple[float, float]:
    """Mean and standard deviation of the block bootstrap replicates.

    Parameters
    ----------
    numerators : sequence of float
        Per-window numerator sums of the statistic, one per retained window.
    denominators : sequence of float
        Per-window denominator sums, one per retained window.
    n_replicates : int
        Number of bootstrap replicates to draw.
    seed : int
        Seed passed to numpy.random.default_rng.

    Returns
    -------
    moments : tuple of float
        The pair (mean, sd), where mean is the mean of the replicate values of
        the genome-wide statistic and sd is their sample standard deviation with
        one degree of freedom removed.

    Raises
    ------
    ValueError
        If the two window sequences differ in length or are empty, if any
        denominator is not positive, or if n_replicates is not at least two.
    """
    return moments
```

### Step 7

07_run_full_pipeline

Goal
----
Chain every step to turn the three population genotype data into the genome-wide

block bootstrap estimate of the introgression statistic.

```python
def run_full_pipeline(positions: list[int], P1_sites: list[str], P2_sites: list[str], P3_sites: list[str], window_size: int = 500,
                      n_replicates: int = 2000, seed: int = 2026) -> float:
    """Genome-wide block bootstrap estimate of the introgression statistic.

    Parameters
    ----------
    positions : sequence of int
        Chromosomal position of each site, in base pairs.
    P1_sites : sequence of str
        Genotype strings for P1, one per site, one character per lineage.
    P2_sites : sequence of str
        Genotype strings for P2, in the same format and the same site order.
    P3_sites : sequence of str
        Genotype strings for P3, in the same format and the same site order.
    window_size : int
        Width of the non-overlapping windows, in base pairs.
    n_replicates : int
        Number of bootstrap replicates to draw.
    seed : int
        Seed passed to numpy.random.default_rng.

    Returns
    -------
    estimate : float
        Mean of the block bootstrap replicates of the genome-wide statistic.

    Raises
    ------
    ValueError
        If positions does not cover the same sites as the genotype strings, or if
        no window survives the requirement of a positive denominator.
    """
    return estimate
```
