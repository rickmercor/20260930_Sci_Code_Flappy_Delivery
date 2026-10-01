"""
Assign informative sites to non-overlapping genomic windows and accumulate the

numerator and denominator of the introgression statistic within each window.

Within a window the statistic weighs the imbalance between the two pair counts

against the allelic diversity the window carries. It is formed from the totals the

window accumulates rather than from the per-site values it averages, so a site

contributes to the comparison in proportion to how much diversity it holds and a

nearly monomorphic site cannot move the window as much as a variable one. The

imbalance is oriented so that an excess of alleles confined to P2 with P3 gives a

positive value, which points to gene flow between P3 and P2, and an excess

confined to P1 with P3 gives a negative one. Not every site takes part: the

method admits only some of them into the two totals.

Windows are non-overlapping intervals of a fixed number of base pairs anchored at

position 0, and a window whose participating sites leave no allelic diversity has

no defined value.

Returns
-------
tuple (window_indices, numerators, denominators) of three lists of equal length, holding native Python ints and floats respectively
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_window_dstar_sums(positions: list[int], pair_counts: list[list[float]], richness: list[float], window_size: int) -> tuple[list[int], list[float], list[float]]:
    """Reference implementation."""
    positions = [p for p in positions]
    pair_counts = [list(pc) for pc in pair_counts]
    richness = [float(k) for k in richness]
    n_sites = len(positions)
    if n_sites == 0:
        raise ValueError("at least one site is required")
    if len(pair_counts) != n_sites or len(richness) != n_sites:
        raise ValueError("per-site inputs must cover the same sites")
    if int(window_size) != window_size or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    for p in positions:
        if int(p) != p or p < 0:
            raise ValueError("positions must be non-negative integers")
    for pc in pair_counts:
        if len(pc) != 2:
            raise ValueError("each site needs two private allele counts")
    for k in richness:
        if k < 0.0:
            raise ValueError("allelic richness must be non-negative")

    window_size = int(window_size)
    numerator_by_window = {}
    denominator_by_window = {}
    for site in range(n_sites):
        pi_P1P3, pi_P2P3 = pair_counts[site]
        if pi_P1P3 <= 0.0 and pi_P2P3 <= 0.0:
            continue
        window = int(positions[site]) // window_size
        numerator_by_window[window] = (
            numerator_by_window.get(window, 0.0) + (pi_P2P3 - pi_P1P3))
        denominator_by_window[window] = (
            denominator_by_window.get(window, 0.0) + richness[site])

    window_indices = sorted(
        w for w in denominator_by_window if denominator_by_window[w] > 0.0)
    numerators = [float(numerator_by_window[w]) for w in window_indices]
    denominators = [float(denominator_by_window[w]) for w in window_indices]
    return window_indices, numerators, denominators

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    positions_data = [40, 180, 430, 620, 880, 1150, 1320, 1490, 1780, 2050,
                      2260, 2410, 2640, 2870, 3090, 3240, 3510, 3660]
    pair_counts_data = [[0.0, 0.0], [0.6666666666666667, 0.0], [0.0, 0.0],
                        [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.8],
                        [0.0, 1.0], [0.0, 0.8], [0.0, 1.0], [0.0, 0.0],
                        [0.2, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0],
                        [0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    richness_data = [1.2307692307692308, 1.6373626373626373,
                     1.7062937062937062, 1.9191919191919191, 1.0,
                     1.7062937062937062, 1.5384615384615383,
                     1.895104895104895, 2.2234432234432235,
                     1.8424908424908426, 1.0, 1.8967032967032966,
                     1.5054945054945055, 1.6373626373626373,
                     1.7417582417582418, 1.6703296703296704,
                     1.93006993006993, 1.9150849150849152]
    return [
        # --- Normal scenario: eight sites spread over three windows ---
        {
            "setup": f"""def _rounded(window_sums):
    indices, numerators, denominators = window_sums
    return [list(indices), [round(v, 12) for v in numerators],
            [round(v, 12) for v in denominators]]
positions = {positions_data!r}
pair_counts = {pair_counts_data!r}
richness = {richness_data!r}
window_size = 500
""",
            "call": "_rounded(window_dstar_sums(positions, pair_counts, richness, window_size))",
            "gold_call": "_rounded(_oracle_window_dstar_sums(positions, pair_counts, richness, window_size))",
        },
        # --- Boundary case: every site uninformative leaves no window ---
        {
            "setup": """def _rounded(window_sums):
    indices, numerators, denominators = window_sums
    return [list(indices), [round(v, 12) for v in numerators],
            [round(v, 12) for v in denominators]]
positions = [40, 180, 620]
pair_counts = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
richness = [1.5, 1.25, 2.0]
window_size = 500
""",
            "call": "[len(part) for part in window_dstar_sums(positions, pair_counts, richness, window_size)]",
            "gold_call": "[len(part) for part in _oracle_window_dstar_sums(positions, pair_counts, richness, window_size)]",
        },
        # --- Edge case: a window size of zero must raise ValueError ---
        {
            "setup": """positions = [40, 180]
pair_counts = [[0.0, 1.0], [0.5, 0.0]]
richness = [1.5, 1.25]
def run_model():
    try:
        window_dstar_sums(positions, pair_counts, richness, 0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_window_dstar_sums(positions, pair_counts, richness, 0)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
