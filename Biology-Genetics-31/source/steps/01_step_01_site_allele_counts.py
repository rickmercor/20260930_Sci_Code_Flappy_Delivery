"""
Parse the per-site genotype strings of three populations into allele count

tables and non-missing sample sizes, one table per site.

A site is described by the number of copies N_ij of allele i observed in

population j, where j indexes P1, P2 and P3 and i runs over the alleles seen at

that site in any of the three populations. The non-missing sample size of

population j at the site is N_j = sum_i N_ij, which is smaller than the number

of sampled lineages wherever genotypes are missing. Nothing about the site is

polarized into ancestral and derived states, so alleles carry no ordering beyond

the arbitrary labels used to write them down, and a site may carry any number of

distinct alleles rather than only two. Missing lineages contribute to no allele

and reduce N_j, which is what later steps use to set the rarefaction subsample

size.

Returns
-------
tuple (counts, totals): counts is a list of n_sites lists of [int, int, int] rows ordered by sorted allele label, totals is a list of n_sites [int, int, int] triples
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _validate_population(sites, name, n_sites):
    """Check one population's genotype strings and return its lineage count."""
    if len(sites) != n_sites:
        raise ValueError("all three populations must cover the same sites")
    widths = {len(s) for s in sites}
    if len(widths) != 1:
        raise ValueError("%s must use the same number of lineages at every site" % name)
    width = widths.pop()
    if width == 0:
        raise ValueError("%s must sample at least one lineage" % name)
    for s in sites:
        for ch in s:
            if ch != "." and not ch.isdigit():
                raise ValueError("genotype characters must be digits or '.'")
    return width


def _oracle_site_allele_counts(P1_sites: list[str], P2_sites: list[str], P3_sites: list[str]) -> tuple[list[list[list[int]]], list[list[int]]]:
    """Reference implementation."""
    P1_sites = list(P1_sites)
    P2_sites = list(P2_sites)
    P3_sites = list(P3_sites)
    n_sites = len(P1_sites)
    if n_sites == 0:
        raise ValueError("at least one site is required")
    _validate_population(P1_sites, "P1", n_sites)
    _validate_population(P2_sites, "P2", n_sites)
    _validate_population(P3_sites, "P3", n_sites)

    counts = []
    totals = []
    for site in range(n_sites):
        strings = (P1_sites[site], P2_sites[site], P3_sites[site])
        labels = sorted({ch for s in strings for ch in s if ch != "."})
        table = [[0, 0, 0] for _ in labels]
        total = [0, 0, 0]
        index = {label: k for k, label in enumerate(labels)}
        for j, s in enumerate(strings):
            for ch in s:
                if ch == ".":
                    continue
                table[index[ch]][j] += 1
                total[j] += 1
        counts.append(table)
        totals.append(total)
    return counts, totals

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: multiallelic sites with missing lineages ---
        {
            "setup": """def pack(result):
    counts, totals = result
    packed = [len(counts)]
    for table, total in zip(counts, totals):
        packed.append(len(table))
        for row in table:
            packed.extend(row)
        packed.extend(total)
    return packed
P1_sites = ["022000", "0000.0", "00.220"]
P2_sites = ["00010", "0010.", "11111"]
P3_sites = ["1110", "1111", "1011"]
""",
            "call": "pack(site_allele_counts(P1_sites, P2_sites, P3_sites))",
            "gold_call": "pack(_oracle_site_allele_counts(P1_sites, P2_sites, P3_sites))",
        },
        # --- Boundary case: one site in which P3 is entirely missing ---
        {
            "setup": """def pack(result):
    counts, totals = result
    packed = [len(counts)]
    for table, total in zip(counts, totals):
        packed.append(len(table))
        for row in table:
            packed.extend(row)
        packed.extend(total)
    return packed
P1_sites = ["00.000"]
P2_sites = ["00010"]
P3_sites = ["...."]
""",
            "call": "pack(site_allele_counts(P1_sites, P2_sites, P3_sites))",
            "gold_call": "pack(_oracle_site_allele_counts(P1_sites, P2_sites, P3_sites))",
        },
        # --- Edge case: ragged lineage counts within a population must raise ---
        {
            "setup": """P1_sites = ["000000", "00000"]
P2_sites = ["00000", "00000"]
P3_sites = ["0000", "0000"]
def run_model():
    try:
        site_allele_counts(P1_sites, P2_sites, P3_sites)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_site_allele_counts(P1_sites, P2_sites, P3_sites)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
