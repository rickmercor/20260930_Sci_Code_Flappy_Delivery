"""
Compute, for one site, the two rarefied counts of alleles confined to a pair of

populations that includes P3.

An allele is private to a combination of populations when it is present in every

member of that combination and absent from every population outside it. Counting

such alleles from the data as they stand is biased by how deeply each population

was sequenced, because a population examined in more lineages reveals more of its

alleles, and correcting that bias by allelic rarefaction at the site's subsample

size g is what this step computes: the rarefied number of alleles private to P1

with P3, and the rarefied number private to P2 with P3. In a three population tree

those are the two combinations that carry information about gene flow with P3.

Without gene flow the two have equal expectation, because an allele that P3 shares

with a sister population through incomplete lineage sorting is as likely to have

persisted in P1 as in P2, so the difference between them is what isolates

introgression. Rarefied counts are real numbers rather than integers, the site is

not restricted to two alleles, and a site at which no population can supply a

lineage leaves nothing to compare.

Returns
-------
tuple of two native Python floats, (pi_P1P3, pi_P2P3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from math import comb


def _check_site_table(counts, totals, g):
    """Validate one site's count table against its sample sizes and g."""
    counts = [list(row) for row in counts]
    totals = list(totals)
    if len(counts) == 0:
        raise ValueError("at least one allele is required")
    if len(totals) != 3:
        raise ValueError("totals must hold three non-missing sample sizes")
    if int(g) != g or g < 0:
        raise ValueError("g must be a non-negative integer")
    for row in counts:
        if len(row) != 3:
            raise ValueError("each allele needs three counts")
        for value in row:
            if int(value) != value or value < 0:
                raise ValueError("allele counts must be non-negative integers")
    for j in range(3):
        if sum(row[j] for row in counts) != totals[j]:
            raise ValueError("allele counts must sum to the non-missing sample size")
        if g > totals[j]:
            raise ValueError("g cannot exceed a non-missing sample size")
    return counts, totals, int(g)


def _oracle_rarefied_pair_private_counts(counts: list[list[int]], totals: list[int], g: int) -> tuple[float, float]:
    """Reference implementation."""
    counts, totals, g = _check_site_table(counts, totals, g)
    pi_P1P3 = 0.0
    pi_P2P3 = 0.0
    for row in counts:
        absent = [comb(totals[j] - row[j], g) / comb(totals[j], g) for j in range(3)]
        present = [1.0 - x for x in absent]
        pi_P1P3 += present[0] * absent[1] * present[2]
        pi_P2P3 += absent[0] * present[1] * present[2]
    return float(pi_P1P3), float(pi_P2P3)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: a triallelic site rarefied to g = 4 ---
        {
            "setup": """sites = [([[4, 4, 1], [0, 1, 3], [2, 0, 0]], [6, 5, 4], 4),
         ([[1, 2, 3], [4, 3, 0]], [5, 5, 3], 3)]
""",
            "call": ("[[round(v, 12) for v in rarefied_pair_private_counts(c, t, g)]"
                     " for c, t, g in sites]"),
            "gold_call": ("[[round(v, 12) for v in _oracle_rarefied_pair_private_counts(c, t, g)]"
                          " for c, t, g in sites]"),
        },
        # --- Boundary case: g = 0 at a site with an uncalled population ---
        {
            "setup": """counts = [[5, 4, 0], [0, 1, 0]]
totals = [5, 5, 0]
g = 0
""",
            "call": "[round(v, 12) for v in rarefied_pair_private_counts(counts, totals, g)]",
            "gold_call": "[round(v, 12) for v in _oracle_rarefied_pair_private_counts(counts, totals, g)]",
        },
        # --- Edge case: g larger than a sample size must raise ValueError ---
        {
            "setup": """counts = [[4, 4, 1], [0, 1, 3], [2, 0, 0]]
totals = [6, 5, 4]
g = 5
def run_model():
    try:
        rarefied_pair_private_counts(counts, totals, g)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_rarefied_pair_private_counts(counts, totals, g)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
