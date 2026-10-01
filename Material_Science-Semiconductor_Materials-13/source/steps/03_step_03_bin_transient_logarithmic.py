"""
Return the logarithmically binned form of one capacitance transient, giving for each retained bin its mean sampling time, its mean capacitance and the number of raw samples it averages.

A transient acquired on a uniform time grid spanning five decades is enormously redundant at long times and sparse at short ones: half of the samples describe the last half of the record, where an exponential that has already decayed carries no information, while the first decade, which fixes the fastest emission rates, is described by a handful of points. Re-expressing the record on a logarithmically spaced time axis restores the balance, gives every decade the same number of points, and reduces the data volume by three orders of magnitude before any inversion is attempted.




The re-expression is an average, not a resampling: every raw sample falls into exactly one logarithmic interval and each retained interval is represented by the mean of the times and the mean of the capacitances it contains. Averaging is what makes the reduction lossless in the statistical sense, because the mean of a bin holding many independent samples has a standard error smaller than that of a single sample by the square root of the count. The counts are therefore not bookkeeping: they are the relative statistical weights of the binned points, and the binned record is strongly heteroscedastic, with the short-time bins holding one raw sample each and the long-time bins holding thousands.




Because the earliest logarithmic intervals are narrower than the uniform sampling interval, they contain no raw sample at all and must be dropped rather than filled, so the binned record is shorter than the nominal number of intervals. Ignoring the weights and treating the surviving points as equally reliable would let the noisiest short-time bins dominate any subsequent least-squares problem, which is precisely where a multi-exponential decomposition is most fragile.

Returns
-------
np.ndarray of shape (n_kept, 3), float: bin mean time (s), bin mean capacitance (pF) and raw sample count, one row per retained bin.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def bin_transient_logarithmic(times: np.ndarray, capacitance: np.ndarray,
                              n_bins: int) -> np.ndarray:
    """Return one capacitance transient re-expressed on a logarithmic time axis.

    Parameters
    ----------
    times : np.ndarray
        Strictly increasing, strictly positive sampling times in s.
    capacitance : np.ndarray
        Capacitance in pF sampled at those times, same length as times.
    n_bins : int
        Number of logarithmically spaced intervals spanning the first to the
        last sampling time (n_bins >= 1).

    Returns
    -------
    binned : np.ndarray
        Array of shape (n_kept, 3) whose columns are the mean sampling time of
        the bin in s, the mean capacitance of the bin in pF, and the number of
        raw samples averaged in that bin. Empty bins are omitted, so n_kept may
        be smaller than n_bins.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return binned  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_bin_transient_logarithmic(times: np.ndarray, capacitance: np.ndarray,
                                      n_bins: int) -> np.ndarray:
    import numpy as np

    times = np.asarray(times, dtype=float)
    capacitance = np.asarray(capacitance, dtype=float)

    if not (isinstance(n_bins, (int, np.integer)) and not isinstance(n_bins, bool)
            and int(n_bins) >= 1):
        raise ValueError("n_bins must be an integer >= 1")
    if times.ndim != 1 or times.size < 2:
        raise ValueError("times must be a one-dimensional array of at least 2 points")
    if times.shape != capacitance.shape:
        raise ValueError("times and capacitance must have the same shape")
    if np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly positive and strictly increasing")

    n_bins = int(n_bins)

    # Interval edges are geometric between the first and the last sample, so
    # every decade of the record receives the same number of intervals.
    edges = np.geomspace(times[0], times[-1], n_bins + 1)

    # digitize against the interior edges assigns the first and last samples to
    # the first and last interval without any out-of-range index.
    index = np.clip(np.digitize(times, edges[1:-1]), 0, n_bins - 1)

    rows = []
    for j in range(n_bins):
        mask = index == j
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append((float(times[mask].mean()),
                     float(capacitance[mask].mean()),
                     float(count)))

    return np.array(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark acquisition grid and interval count ---
        {
            "setup": """import numpy as np
times = np.arange(1, 100001) / 1.0e5
capacitance = 204.5 - 0.75 * np.exp(-61.468 * times) - 0.05 * np.exp(-1451.65 * times)
""",
            "call": "bin_transient_logarithmic(times, capacitance, 100) * np.array([1.0e8, 1.0, 1.0]) + 1000.0",
            "gold_call": "_oracle_bin_transient_logarithmic(times, capacitance, 100) * np.array([1.0e8, 1.0, 1.0]) + 1000.0",
        },
        # --- Valid: retained bin count when intervals outnumber early samples ---
        {
            "setup": """import numpy as np
times = np.arange(1, 30001) / 4.0e4
capacitance = 160.0 - 0.5 * np.exp(-85.0 * times)
""",
            "call": "bin_transient_logarithmic(times, capacitance, 75) * np.array([1.0e8, 1.0, 1.0]) + 1000.0",
            "gold_call": "_oracle_bin_transient_logarithmic(times, capacitance, 75) * np.array([1.0e8, 1.0, 1.0]) + 1000.0",
        },
        # --- Valid: total sample count is conserved by the binning ---
        {
            "setup": """import numpy as np
times = np.arange(1, 20001) / 5.0e4
capacitance = 180.0 - 0.4 * np.exp(-250.0 * times)
""",
            "call": "bin_transient_logarithmic(times, capacitance, 60) * np.array([1.0e8, 1.0, 1.0]) + 1000.0",
            "gold_call": "_oracle_bin_transient_logarithmic(times, capacitance, 60) * np.array([1.0e8, 1.0, 1.0]) + 1000.0",
        },
        # --- Boundary: a single interval collapses the record to its mean ---
        {
            "setup": """import numpy as np
times = np.arange(1, 1001) / 1.0e4
capacitance = 100.0 - 0.3 * np.exp(-40.0 * times)
""",
            "call": "bin_transient_logarithmic(times, capacitance, 1) * np.array([1.0e8, 1.0, 1.0]) + 1000.0",
            "gold_call": "_oracle_bin_transient_logarithmic(times, capacitance, 1) * np.array([1.0e8, 1.0, 1.0]) + 1000.0",
        },
        # --- Edge: more intervals than samples leaves many bins empty ---
        {
            "setup": """import numpy as np
times = np.arange(1, 51) / 1.0e3
capacitance = 50.0 - 0.2 * np.exp(-100.0 * times)
""",
            "call": "bin_transient_logarithmic(times, capacitance, 400) * np.array([1.0e8, 1.0, 1.0]) + 1000.0",
            "gold_call": "_oracle_bin_transient_logarithmic(times, capacitance, 400) * np.array([1.0e8, 1.0, 1.0]) + 1000.0",
        },
        # --- Invalid: mismatched input lengths ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        bin_transient_logarithmic(np.arange(1, 11) / 100.0, np.zeros(9), 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bin_transient_logarithmic(np.arange(1, 11) / 100.0, np.zeros(9), 5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-increasing sampling times ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        bin_transient_logarithmic(np.array([0.1, 0.05, 0.2]), np.zeros(3), 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bin_transient_logarithmic(np.array([0.1, 0.05, 0.2]), np.zeros(3), 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
