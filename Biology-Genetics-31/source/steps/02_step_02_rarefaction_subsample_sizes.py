"""
Fix the rarefaction subsample size g used at each site from that site's

non-missing sample sizes in the three populations.

Rarefaction compares populations at a common subsample size rather than at the

sizes that happen to have been sequenced, because counts of alleles restricted to

a population or to a pair of populations grow with the number of lineages

examined. The method fixes that size, g, separately at every site from the

lineages the three populations have called there, so it is a property of the site

rather than a parameter chosen once for the dataset, and it moves with the missing

data along the sequence. A site where some population contributes no called

lineage leaves nothing to compare and falls out of the analysis without needing a

special case.

Returns
-------
list of n_sites native Python ints, the subsample size g at each site
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rarefaction_subsample_sizes(totals: list[list[int]]) -> list[int]:
    """Reference implementation."""
    totals = [list(t) for t in totals]
    if len(totals) == 0:
        raise ValueError("at least one site is required")
    for t in totals:
        if len(t) != 3:
            raise ValueError("each site needs three non-missing sample sizes")
        for value in t:
            if int(value) != value or value < 0:
                raise ValueError("sample sizes must be non-negative integers")
    return [int(min(t)) for t in totals]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: uneven sampling across the three populations ---
        {
            "setup": """totals = [[6, 5, 4], [5, 5, 3], [6, 4, 4], [5, 5, 0]]
""",
            "call": "rarefaction_subsample_sizes(totals)",
            "gold_call": "_oracle_rarefaction_subsample_sizes(totals)",
        },
        # --- Boundary case: a single site at which one population is uncalled ---
        {
            "setup": """totals = [[6, 5, 0]]
""",
            "call": "rarefaction_subsample_sizes(totals)",
            "gold_call": "_oracle_rarefaction_subsample_sizes(totals)",
        },
        # --- Edge case: a negative sample size must raise ValueError ---
        {
            "setup": """totals = [[6, 5, 4], [6, -1, 4]]
def run_model():
    try:
        rarefaction_subsample_sizes(totals)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_rarefaction_subsample_sizes(totals)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
