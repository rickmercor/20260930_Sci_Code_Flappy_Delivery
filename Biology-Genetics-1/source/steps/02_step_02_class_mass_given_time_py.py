"""
Probability that the run of homozygosity around a focal position has a length in a given class, for each coalescence time.

A run of homozygosity is reported by its length, and demographic inference sorts runs

into length classes: long runs point to recent common ancestry, short runs to ancient

ancestry. The run around a focal position extends to one side and to the other, and

what is observed is the total. This step turns the one-sided description of the

previous step into the probability that the whole run falls into a length class, for

each possible coalescence time of the pair of haplotypes.

Returns
-------
np.ndarray, shape (k,): the probability that the run's length lies in the class, for each coalescence time in t
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def class_mass_given_time(lower: float, upper: float, t: "np.ndarray", break_rate: float,
                          marker_spacing: float, heterozygosity: float) -> "np.ndarray":
    '''Probability that the run of homozygosity around a focal position has a length in a given class, for each coalescence time.

    The run is made of its two sides, which are independent given the
    coalescence time and are each distributed as in the previous step; its
    length is the sum of the two side lengths. For each entry of t, return the
    probability that the length lies in the class [lower, upper] Morgans.

    Parameters
    ----------
    lower : float
        Lower end of the length class in Morgans, >= 0.
    upper : float
        Upper end of the length class in Morgans, > lower; may be infinite.
    t : np.ndarray
        Shape (k,): numbers of generations since coalescence, integers >= 1.
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
    mass : np.ndarray
        Shape (k,): mass[j] is the probability that the run's length lies in the
        class, given a coalescence t[j] generations ago. Every entry is accurate
        to an absolute error below 1e-12.

    Raises
    ------
    ValueError
        If lower is not a finite number >= 0, if upper is not a number > lower
        (infinite allowed), if t is not a one-dimensional array of integers >= 1
        with at least one entry, or on any condition raised by the previous step
        for break_rate, marker_spacing and heterozygosity.
    '''
    return mass  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_class_mass_given_time(lower: float, upper: float, t: "np.ndarray", break_rate: float,
                                  marker_spacing: float, heterozygosity: float) -> "np.ndarray":
    from numpy.polynomial.legendre import leggauss

    def _num(v):
        return (not isinstance(v, bool)) and isinstance(v, (int, float, np.integer, np.floating))

    if not _num(lower) or not np.isfinite(float(lower)) or float(lower) < 0.0:
        raise ValueError("lower must be a finite number >= 0")
    if not _num(upper) or np.isnan(float(upper)) or not float(upper) > float(lower):
        raise ValueError("upper must be a number > lower (infinite allowed)")
    tt = np.asarray(t) if not isinstance(t, (str, bytes)) else None
    if (tt is None or tt.ndim != 1 or tt.size < 1 or tt.dtype.kind == "b"
            or not np.all(np.isfinite(tt.astype(float))) or np.any(tt.astype(float) != np.floor(tt.astype(float)))
            or np.any(tt.astype(float) < 1)):
        raise ValueError("t must be a one-dimensional array of integers >= 1 with at least one entry")
    a, b = float(lower), float(upper)
    # the two sides are independent, so with T(x) = P(L + R >= x) = S(x) + int_0^x f(u) S(x - u) du,
    # a sum of non-negative terms, the class probability is T(lower) - T(upper), with T(inf) = 0
    nodes, weights = leggauss(128)

    def _tail_grid(x):
        u = 0.5 * x * (nodes + 1.0)
        return u, 0.5 * x * weights

    ua, wa = _tail_grid(a)
    finite = np.isfinite(b)
    if finite:
        ub, wb = _tail_grid(b)
        grid = np.concatenate([[a], ua, a - ua, [b], ub, b - ub])
    else:
        grid = np.concatenate([[a], ua, a - ua])
    n = ua.size
    out = np.empty(tt.size)
    for j, tj in enumerate(tt.astype(float)):
        sd = _oracle_flank_survival_and_density(grid, int(tj), break_rate, marker_spacing, heterozygosity)
        tail_a = sd[0, 0] + float((sd[1, 1:n + 1] * sd[0, n + 1:2 * n + 1]) @ wa)
        if finite:
            off = 2 * n + 1
            tail_b = sd[0, off] + float((sd[1, off + 1:off + n + 1] * sd[0, off + n + 1:off + 2 * n + 1]) @ wb)
        else:
            tail_b = 0.0
        out[j] = tail_a - tail_b
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the 2-4 cM class on the shipped panel across recent to old coalescence times ---
        {
            "setup": "import numpy as np\nt = np.array([1, 3, 10, 30, 100, 400])\n",
            "call": "class_mass_given_time(0.02, 0.04, t, 1.5, 0.001, 0.30)",
            "gold_call": "_oracle_class_mass_given_time(0.02, 0.04, t, 1.5, 0.001, 0.30)",
        },
        # --- boundary: every break observed where it occurs, the 1-2 cM class ---
        {
            "setup": "import numpy as np\nt = np.array([2, 20, 200])\n",
            "call": "class_mass_given_time(0.01, 0.02, t, 1.5, 0.0, 0.30)",
            "gold_call": "_oracle_class_mass_given_time(0.01, 0.02, t, 1.5, 0.0, 0.30)",
        },
        # --- edge: an open class of long runs, 4 cM and above ---
        {
            "setup": "import numpy as np\nt = np.array([1, 5, 50, 200])\n",
            "call": "class_mass_given_time(0.04, np.inf, t, 1.5, 0.001, 0.30)",
            "gold_call": "_oracle_class_mass_given_time(0.04, np.inf, t, 1.5, 0.001, 0.30)",
        },
        # --- edge: a class starting at zero length on a sparse panel ---
        {
            "setup": "import numpy as np\nt = np.array([4, 40])\n",
            "call": "class_mass_given_time(0.0, 0.005, t, 2.0, 0.005, 0.2)",
            "gold_call": "_oracle_class_mass_given_time(0.0, 0.005, t, 2.0, 0.005, 0.2)",
        },
    ]
