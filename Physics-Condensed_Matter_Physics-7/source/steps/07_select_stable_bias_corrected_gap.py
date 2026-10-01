"""
Select the first ordered rank pair whose bootstrap support and robust logarithmic spread satisfy the inclusive stability thresholds, then return its positive additive bootstrap-bias-corrected intersector gap.

The scan rows are already ordered from the preferred largest retained subspaces to smaller alternatives. Selection therefore stops at the first stable row and does not reorder candidates by their numerical gap.

Returns
-------
float, the positive bias-corrected gap from the first stable candidate pair
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def select_stable_bias_corrected_gap(
    rank_pair_scan: np.ndarray,
    min_valid_fraction: float,
    max_log_mad: float,
) -> float:
    """Return the bias-corrected gap from the first stable scan row.

    Parameters
    ----------
    rank_pair_scan : np.ndarray
        Finite float array of shape ``(candidates, 7)`` with the column order
        documented by ``compute_bootstrap_rank_pair_scan``. Candidate pairs
        must be unique positive integer-valued labels.
    min_valid_fraction : float
        Inclusive support threshold in ``(0, 1]``.
    max_log_mad : float
        Inclusive finite nonnegative robust-spread threshold.

    Returns
    -------
    gap : float
        Positive finite bias-corrected intersector gap from the first stable
        candidate row.

    Raises
    ------
    ValueError
        If the scan or thresholds are invalid or no row is stable.
    """
    return gap

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
from numbers import Real

import numpy as np


def _oracle_select_stable_bias_corrected_gap(
    rank_pair_scan: np.ndarray,
    min_valid_fraction: float,
    max_log_mad: float,
) -> float:
    import math
    from numbers import Real
    import numpy as np

    scan = np.asarray(rank_pair_scan, dtype=float)
    if scan.ndim != 2 or scan.shape[0] < 1 or scan.shape[1] != 7 or not np.all(np.isfinite(scan)):
        raise ValueError("rank_pair_scan must have finite shape (candidates,7)")
    if (
        isinstance(min_valid_fraction, bool)
        or not isinstance(min_valid_fraction, Real)
        or not math.isfinite(float(min_valid_fraction))
        or not (0.0 < float(min_valid_fraction) <= 1.0)
    ):
        raise ValueError("min_valid_fraction must lie in (0,1]")
    if (
        isinstance(max_log_mad, bool)
        or not isinstance(max_log_mad, Real)
        or not math.isfinite(float(max_log_mad))
        or float(max_log_mad) < 0.0
    ):
        raise ValueError("max_log_mad must be finite and nonnegative")

    labels = scan[:, :2]
    rounded = np.rint(labels)
    if np.any(labels != rounded) or np.any(rounded < 2):
        raise ValueError("candidate rank labels must be integer-valued and at least two")
    if len({(int(a), int(b)) for a, b in rounded}) != scan.shape[0]:
        raise ValueError("candidate rank pairs must be unique")
    if np.any(scan[:, 2] < 0.0) or np.any(scan[:, 2] > 1.0):
        raise ValueError("valid fractions must lie in [0,1]")
    if np.any(scan[:, 3] < 0.0) or np.any(scan[:, 4] <= 0.0):
        raise ValueError("spreads must be nonnegative and central gaps positive")

    max_float = np.finfo(float).max
    for row in scan:
        corrected = float(row[6])
        if (
            row[2] >= float(min_valid_fraction)
            and row[3] <= float(max_log_mad)
            and corrected > 0.0
            and corrected < max_float
        ):
            return corrected
    raise ValueError("no candidate rank pair satisfies the stability thresholds")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\ns=np.array([[4,8,0.0,np.finfo(float).max,0.8,np.finfo(float).max,np.finfo(float).max],[3,6,1.0,0.13,0.81,0.79,0.83],[2,6,1.0,0.12,0.80,0.78,0.82]],dtype=float)",
            "call": "select_stable_bias_corrected_gap(s.copy(),0.95,0.125)",
            "gold_call": "_oracle_select_stable_bias_corrected_gap(s.copy(),0.95,0.125)",
        },
        {
            "setup": "import numpy as np\ns=np.array([[3,6,0.95,0.125,0.8,0.79,0.81],[2,6,1.0,0.1,0.7,0.69,0.71]],dtype=float)",
            "call": "select_stable_bias_corrected_gap(s.copy(),0.95,0.125)",
            "gold_call": "_oracle_select_stable_bias_corrected_gap(s.copy(),0.95,0.125)",
        },
        {
            "setup": "import numpy as np\ns=np.array([[4,7,0.94,0.01,0.8,0.79,0.81],[3,6,1.0,0.2,0.8,0.79,0.81],[2,5,1.0,0.1,0.7,0.69,0.71]],dtype=float)",
            "call": "select_stable_bias_corrected_gap(s.copy(),0.95,0.15)",
            "gold_call": "_oracle_select_stable_bias_corrected_gap(s.copy(),0.95,0.15)",
        },
        {
            "setup": """import numpy as np
s=np.array([[3,6,0.5,0.8,0.8,0.8,0.8],[2,5,0.6,0.7,0.7,0.7,0.7]],dtype=float)
def run_model():
 try:
  select_stable_bias_corrected_gap(s.copy(),0.95,0.2)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_select_stable_bias_corrected_gap(s.copy(),0.95,0.2)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
s=np.array([[3,6,1.0,0.1,0.8,0.79,-0.2]],dtype=float)
def run_model():
 try:
  select_stable_bias_corrected_gap(s.copy(),0.95,0.2)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_select_stable_bias_corrected_gap(s.copy(),0.95,0.2)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
s=np.array([[3,6,1.0,0.1,0.8,0.79,np.nan]],dtype=float)
def run_model():
 try:
  select_stable_bias_corrected_gap(s.copy(),0.95,0.2)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_select_stable_bias_corrected_gap(s.copy(),0.95,0.2)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
