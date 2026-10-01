"""
Compute the rarefied allelic richness carried jointly by the three populations

at one site, which is the normalizer of the introgression statistic.

The imbalance between the two pair-specific private allele counts has to be put on

a common scale, otherwise a window rich in variation dominates a window that is

nearly monomorphic. The scale the method uses is allelic richness, meaning the

number of distinct alleles that a single subsample of the site's called lineages

would be expected to recover, rarefied at the same subsample size g the pair

counts use so that both sides of the comparison are measured at one depth. It is

a property of the site as a whole rather than of any one population, and it is

the denominator the statistic is divided by. A monomorphic site scores 1, a site

scores more as its alleles become both numerous and common enough to survive

subsampling, and a site with no subsample to take scores 0.

Returns
-------
float, the rarefied allelic richness kappa as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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


def _oracle_rarefied_allelic_richness(counts: list[list[int]], totals: list[int], g: int) -> float:
    """Reference implementation."""
    counts, totals, g = _check_site_table(counts, totals, g)
    pooled = sum(totals)
    kappa = 0.0
    for row in counts:
        kappa += 1.0 - comb(pooled - sum(row), g) / comb(pooled, g)
    return float(kappa)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: a triallelic site rarefied to g = 4 ---
        {
            "setup": """counts = [[4, 4, 1], [0, 1, 3], [2, 0, 0]]
totals = [6, 5, 4]
g = 4
""",
            "call": "round(rarefied_allelic_richness(counts, totals, g), 12)",
            "gold_call": "round(_oracle_rarefied_allelic_richness(counts, totals, g), 12)",
        },
        # --- Boundary case: a monomorphic site must give kappa = 1 ---
        {
            "setup": """counts = [[6, 5, 4]]
totals = [6, 5, 4]
g = 4
""",
            "call": "round(rarefied_allelic_richness(counts, totals, g), 12)",
            "gold_call": "round(_oracle_rarefied_allelic_richness(counts, totals, g), 12)",
        },
        # --- Edge case: counts inconsistent with totals must raise ValueError ---
        {
            "setup": """counts = [[4, 4, 1], [0, 1, 3]]
totals = [6, 5, 4]
def run_model():
    try:
        rarefied_allelic_richness(counts, totals, 4)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_rarefied_allelic_richness(counts, totals, 4)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
