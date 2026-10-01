"""
Chain every step to turn the three population genotype data into the genome-wide

block bootstrap estimate of the introgression statistic.

The full calculation moves from genotypes to a single genome-wide number in four

stages. Each site is reduced to allele counts and non-missing sample sizes and is

given its own rarefaction subsample size. At that subsample size the site yields

the two rarefied counts of alleles confined to P1 with P3 and to P2 with P3, and a

rarefied allelic richness that serves as the normalizer. The sites the method

admits are then accumulated into non-overlapping windows, and the windows that come out defined

act as the blocks of a bootstrap whose replicate mean is the genome-wide point

estimate and whose spread is its standard error. A positive estimate indicates gene

flow between P3 and P2 and a negative one gene flow between P3 and P1, and no

outgroup and no assignment of ancestral states enters anywhere.

Returns
-------
float, the mean of the block bootstrap replicates as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_full_pipeline(positions: list[int], P1_sites: list[str], P2_sites: list[str], P3_sites: list[str], window_size: int = 500,
                              n_replicates: int = 2000, seed: int = 2026) -> float:
    """Reference implementation."""
    positions = [p for p in positions]
    counts, totals = _oracle_site_allele_counts(P1_sites, P2_sites, P3_sites)
    if len(positions) != len(counts):
        raise ValueError("positions must cover the same sites as the genotypes")

    subsample_sizes = _oracle_rarefaction_subsample_sizes(totals)

    pair_counts = []
    richness = []
    for site in range(len(counts)):
        g = subsample_sizes[site]
        pair_counts.append(
            list(_oracle_rarefied_pair_private_counts(counts[site], totals[site], g)))
        richness.append(_oracle_rarefied_allelic_richness(counts[site], totals[site], g))

    window_indices, numerators, denominators = _oracle_window_dstar_sums(
        positions, pair_counts, richness, window_size)
    if len(window_indices) == 0:
        raise ValueError("no window has a positive denominator")

    mean, _sd = _oracle_block_bootstrap_moments(
        numerators, denominators, n_replicates, seed)
    return float(mean)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases(
    _POSITIONS=[40, 180, 430, 620, 880, 1150, 1320, 1490, 1780, 2050, 2260,
                2410, 2640, 2870, 3090, 3240, 3510, 3660, 3920, 4130, 4380,
                4520, 4740, 4960, 5080, 5230, 5470, 5610, 5820, 5950],
    _P1=["000000", "100000", ".101.1", "000..0", "000000", ".0.110", "0000..",
         "0000.0", "022000", "000000", "000000", "000111", "000000", "110010",
         "111111", "000000", "000.00", "111111", "020000", "000000", "1.1111",
         "011010", "000000", "00.000", "000000", "020000", "00.220", ".0.000",
         "0111.1", "000000"],
    _P2=["0000.", "00000", "00000", "0.110", "00000", "00000", "00010", "0010.",
         "00010", "10110", "00000", "10111", "000.0", "00000", "00000", "0.000",
         "11001", "000.0", "11110", "11.00", "000.0", "10100", "11111", "00010",
         "10101", "10001", "11111", "0100.", "11010", "00000"],
    _P3=["010.", "1100", "0000", "1011", ".000", "0010", "0100", "1111", "1110",
         "0110", "0000", "0110", "0011", "0000", "11.1", "1101", "1111", "1001",
         "11.1", "1111", "0111", "0101", "0110", "....", ".11.", "0111", "1011",
         "1100", ".000", "0000"]):
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: the full benchmark configuration ---
        {
            "setup": "positions = %r\nP1_sites = %r\nP2_sites = %r\nP3_sites = %r\n" %
                     (_POSITIONS, _P1, _P2, _P3),
            "call": "round(run_full_pipeline(positions, P1_sites, P2_sites, P3_sites, 500, 2000, 2026), 12)",
            "gold_call": "round(_oracle_run_full_pipeline(positions, P1_sites, P2_sites, P3_sites, 500, 2000, 2026), 12)",
        },
        # --- Boundary case: eight sites, a wider window and few replicates ---
        {
            "setup": "positions = %r\nP1_sites = %r\nP2_sites = %r\nP3_sites = %r\n" %
                     (_POSITIONS[:8], _P1[:8], _P2[:8], _P3[:8]),
            "call": "round(run_full_pipeline(positions, P1_sites, P2_sites, P3_sites, 1000, 50, 11), 12)",
            "gold_call": "round(_oracle_run_full_pipeline(positions, P1_sites, P2_sites, P3_sites, 1000, 50, 11), 12)",
        },
        # --- Edge case: positions that do not match the sites must raise ---
        {
            "setup": ("positions = %r\nP1_sites = %r\nP2_sites = %r\nP3_sites = %r\n" %
                      (_POSITIONS[:5], _P1[:8], _P2[:8], _P3[:8])) + """\ndef run_model():
    try:
        run_full_pipeline(positions, P1_sites, P2_sites, P3_sites, 500, 50, 3)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_run_full_pipeline(positions, P1_sites, P2_sites, P3_sites, 500, 50, 3)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
