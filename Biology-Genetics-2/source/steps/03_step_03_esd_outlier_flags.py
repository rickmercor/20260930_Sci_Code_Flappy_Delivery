"""
Flag the upper-tail outliers of the genome-wide observation distribution with a

one-sided generalized extreme Studentized deviate test, returning a binary

indicator per marginal tree.

Fitting a two-state model of archaic ancestry needs a starting split of the data

into a bulk that stands for ordinary variation and a tail that stands for the

candidate introgressed trees, and no external reference is available to make that

split. Because introgressed trees are a small minority, the genome-wide

distribution of the per-tree observation is itself a workable approximation to

the null once its upper tail has been removed. The generalized extreme

Studentized deviate procedure removes that tail without being told in advance how

many points belong to it. At iteration i the sample mean and the sample standard

deviation with one degree of freedom removed are recomputed on the points that

survive, the statistic R_i is the largest remaining deviation above the mean

divided by that standard deviation, and R_i is compared with the critical value

the procedure attaches to that iteration, which depends on the size n of the

original sample and on the level alpha. Only the upper tail is of interest, so

the deviation is taken signed rather than absolute and the critical value carries

no two-sided correction. Testing stops at the first iteration whose statistic

does not exceed its critical value, and a cap on the fraction of the sample that

may be removed keeps the procedure from eroding the bulk of the distribution. If

the surviving points carry no dispersion the sample standard deviation is zero,

the statistic is undefined, and the procedure stops without flagging a further

point.

Returns
-------
np.ndarray of shape (m,), the outlier indicator as a float64 array of 0.0 and 1.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def esd_outlier_flags(
    observations: "np.ndarray",
    alpha: float = 0.05,
    max_outlier_fraction: float = 0.2,
) -> "np.ndarray":
    """Return a binary flag per observation marking the upper-tail outliers.

    Parameters
    ----------
    observations : np.ndarray
        Per-tree observations of shape (m,).
    alpha : float
        Significance level of the test.
    max_outlier_fraction : float
        Largest fraction of the sample that may be removed.

    Returns
    -------
    flags : np.ndarray
        Array of shape (m,) holding 1 for a flagged outlier and 0 otherwise.

    Raises
    ------
    ValueError
        If observations is not one dimensional, if it holds fewer than four
        points, if alpha does not lie strictly between 0 and 1, or if
        max_outlier_fraction does not lie strictly between 0 and 1.
    """
    return flags

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.stats import t as student_t


def _oracle_esd_outlier_flags(
    observations: "np.ndarray",
    alpha: float = 0.05,
    max_outlier_fraction: float = 0.2,
) -> "np.ndarray":
    """Reference implementation."""
    observations = np.asarray(observations, dtype=float)
    if observations.ndim != 1:
        raise ValueError("observations must be one dimensional")
    if observations.size < 4:
        raise ValueError("observations must hold at least four points")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between 0 and 1")
    if not (0.0 < max_outlier_fraction < 1.0):
        raise ValueError("max_outlier_fraction must lie strictly between 0 and 1")

    n = observations.size
    flags = np.zeros(n, dtype=float)
    remaining = observations.copy()
    index = np.arange(n)
    max_outliers = max(1, int(n * max_outlier_fraction))
    for i in range(1, max_outliers + 1):
        sd = remaining.std(ddof=1)
        if sd <= 0.0:
            break
        deviation = remaining - remaining.mean()
        j = int(np.argmax(deviation))
        statistic = deviation[j] / sd
        prob = 1.0 - alpha / (n - i + 1)
        t_c = student_t.ppf(prob, df=n - i - 1)
        critical = (n - i) * t_c / np.sqrt((n - i - 1 + t_c ** 2) * (n - i + 1))
        if statistic <= critical:
            break
        flags[index[j]] = 1.0
        remaining = np.delete(remaining, j)
        index = np.delete(index, j)
    return flags

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    fixture = '''import numpy as np

_OBS = np.array([
    22100.0, 19500.0, 22000.0, 15200.0, 13800.0, 11300.0, 23500.0, 19200.0,
    14100.0, 6000.0, 8500.0, 8100.0, 6200.0, 20500.0, 138300.0, 148500.0,
    134400.0, 142800.0, 46000.0, 18300.0, 11500.0, 11900.0, 18300.0, 8200.0,
    14300.0, 22200.0, 17700.0, 24000.0, 20920.0, 10900.0, 46000.0, 12200.0,
    18100.0, 135600.0, 14700.0, 5900.0, 12900.0, 22100.0, 16100.0, 12500.0,
    15300.0, 11400.0, 10100.0, 14900.0, 6000.0, 8600.0, 17300.0, 12700.0,
])
'''
    return [
        # --- Normal scenario: the full genome-wide observation vector ---
        {
            "setup": fixture + """observations = _OBS.copy()
alpha = 0.05
max_outlier_fraction = 0.2
""",
            "call": "esd_outlier_flags(observations, alpha=alpha, max_outlier_fraction=max_outlier_fraction).tolist()",
            "gold_call": "_oracle_esd_outlier_flags(observations, alpha=alpha, max_outlier_fraction=max_outlier_fraction).tolist()",
        },
        # --- Boundary case: a sample with no dispersion at all ---
        {
            "setup": fixture + """observations = np.full(8, 9000.0)
alpha = 0.05
max_outlier_fraction = 0.2
""",
            "call": "esd_outlier_flags(observations, alpha=alpha, max_outlier_fraction=max_outlier_fraction).tolist()",
            "gold_call": "_oracle_esd_outlier_flags(observations, alpha=alpha, max_outlier_fraction=max_outlier_fraction).tolist()",
        },
        # --- Edge case: a significance level outside the unit interval ---
        {
            "setup": fixture + """observations = _OBS.copy()
def run_model():
    try:
        esd_outlier_flags(observations, alpha=1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_esd_outlier_flags(observations, alpha=1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
