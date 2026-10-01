"""
Coverage of a length class given by the source's closed-form steady-state expression at a constant population size.

Studies of runs of homozygosity often summarise a length class not by its coverage but

by the constant population size that would produce that coverage at steady state, read

off a closed-form expression in which the marker panel and the break rates are

parameters. The source publishes such an expression for runs of homozygosity, in the

form that holds for lengths well above the mean displacement of the observable

boundaries. When a study reports the size, the coverage behind it is recovered by

evaluating that expression. This step evaluates the expression, integrated over a

length class, for a stated constant size.

Returns
-------
float, the closed-form coverage of the class, a fraction of the genome, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def class_coverage_from_reported_size(reported_size: float, lower: float, upper: float, break_rate: float,
                                      marker_spacing: float, heterozygosity: float) -> float:
    '''Coverage of a length class given by the source's closed-form steady-state expression at a constant population size.

    The expression is the source's closed form for the fraction of the genome
    covered by runs of homozygosity of length x, per Morgan of run length, in a
    randomly mating population of constant size N breeding individuals:
    4 x (1 + m)^2 / (N (2 x (1 + m) + 1/(2 N) - 4 d/H)^3), with m the combined
    rate of mutation and gene-conversion breaks per Morgan per generation on one
    lineage, d the mean marker spacing in Morgans and H the heterozygosity of
    the typed markers. It holds for lengths well above the mean displacement
    d/H of the observable boundaries, and it requires
    2 lower (1 + m) + 1/(2 N) - 4 d/H > 0. Return that expression integrated
    over the class [lower, upper] Morgans, with N = reported_size,
    m = break_rate, d = marker_spacing and H = heterozygosity.

    Parameters
    ----------
    reported_size : float
        Constant number of breeding individuals, >= 1.
    lower : float
        Lower end of the length class in Morgans, > 0.
    upper : float
        Upper end of the length class in Morgans, > lower; may be infinite.
    break_rate : float
        Combined rate of mutation and gene-conversion breaks per Morgan per
        generation along one lineage, >= 0.
    marker_spacing : float
        Mean spacing of the typed markers in Morgans, in [0, heterozygosity).
    heterozygosity : float
        Probability that a typed marker is heterozygous between two random
        haplotypes, in (0, 1].

    Returns
    -------
    coverage : float
        The closed-form coverage of the class, a fraction of the genome, as a
        native Python float.

    Raises
    ------
    ValueError
        If reported_size is not a finite number >= 1, if lower is not a finite
        number > 0, if upper is not a number > lower (infinite allowed), if
        break_rate is not a finite number >= 0, if heterozygosity is not a
        finite number in (0, 1], if marker_spacing is not a finite number in
        [0, heterozygosity), or if the expression does not apply to the class
        because lower is not large enough relative to the mean displacement.
    '''
    return coverage  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_class_coverage_from_reported_size(reported_size: float, lower: float, upper: float, break_rate: float,
                                              marker_spacing: float, heterozygosity: float) -> float:
    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating))

    if not _num(reported_size) or not np.isfinite(float(reported_size)) or float(reported_size) < 1.0:
        raise ValueError("reported_size must be a finite number >= 1")
    if not _num(lower) or not np.isfinite(float(lower)) or float(lower) <= 0.0:
        raise ValueError("lower must be a finite number > 0")
    if not _num(upper) or np.isnan(float(upper)) or not float(upper) > float(lower):
        raise ValueError("upper must be a number > lower (infinite allowed)")
    if not _num(break_rate) or not np.isfinite(float(break_rate)) or float(break_rate) < 0.0:
        raise ValueError("break_rate must be a finite number >= 0")
    if not _num(heterozygosity) or not np.isfinite(float(heterozygosity)) or not 0.0 < float(heterozygosity) <= 1.0:
        raise ValueError("heterozygosity must be a finite number in (0, 1]")
    if not _num(marker_spacing) or not np.isfinite(float(marker_spacing)) or not 0.0 <= float(marker_spacing) < float(heterozygosity):
        raise ValueError("marker_spacing must be a finite number in [0, heterozygosity)")
    n, m, d, h = float(reported_size), float(break_rate), float(marker_spacing), float(heterozygosity)
    a, b = float(lower), float(upper)
    # the source's steady-state density of coverage by runs of length x under a constant size:
    # 4 x (1 + m)^2 / (N (2 x (1 + m) + 1/(2N) - 4 d/H)^3), the displacement entering through -4 d/H
    beta = 2.0 * (1.0 + m)
    shift = 1.0 / (2.0 * n) - 4.0 * d / h
    if beta * a + shift <= 0.0:
        raise ValueError("the expression does not apply to the class: lower is not large enough relative to the mean displacement")
    def _antiderivative(u):
        return -1.0 / u + shift / (2.0 * u * u)
    upper_term = 0.0 if np.isinf(b) else _antiderivative(beta * b + shift)
    return float((upper_term - _antiderivative(beta * a + shift)) / n)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the reported size of the shipped configuration and the 2-4 cM class ---
        {
            "setup": "import numpy as np\n",
            "call": "class_coverage_from_reported_size(352.998, 0.02, 0.04, 1.5, 0.001, 0.30)",
            "gold_call": "_oracle_class_coverage_from_reported_size(352.998, 0.02, 0.04, 1.5, 0.001, 0.30)",
        },
        # --- boundary: markers so dense that the displacement term vanishes ---
        {
            "setup": "import numpy as np\n",
            "call": "class_coverage_from_reported_size(1000.0, 0.01, 0.02, 1.5, 0.0, 0.30)",
            "gold_call": "_oracle_class_coverage_from_reported_size(1000.0, 0.01, 0.02, 1.5, 0.0, 0.30)",
        },
        # --- edge: an open class of long runs at a large constant size on a sparse panel ---
        {
            "setup": "import numpy as np\n",
            "call": "class_coverage_from_reported_size(8000.0, 0.04, np.inf, 2.0, 0.004, 0.25)",
            "gold_call": "_oracle_class_coverage_from_reported_size(8000.0, 0.04, np.inf, 2.0, 0.004, 0.25)",
        },
    ]
