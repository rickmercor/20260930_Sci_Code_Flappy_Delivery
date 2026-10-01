"""
Compute the acceptance probability for one continuous-time diagonal-link insertion or removal. Use the pre-move local count and return a native float in the closed interval [0, 1].

The continuous-time expansion has unequal insertion and removal proposal multiplicities. Detailed balance fixes their two branch-dependent probabilities through the inverse temperature, local matrix-element magnitude, and current local expansion order.

Returns
-------
float, the continuous-time link-move acceptance probability in [0, 1]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_link_acceptance(
    beta: float,
    local_weight: float,
    local_count: int,
    move: int,
) -> float:
    """Compute a continuous-time link-move acceptance probability.

    Parameters
    ----------
    beta : float
        Strictly positive inverse temperature.
    local_weight : float
        Strictly positive magnitude of the selected diagonal matrix element.
    local_count : int
        Nonnegative selected-location count before the proposal.
    move : int
        ``+1`` for insertion or ``-1`` for removal.

    Returns
    -------
    acceptance : float
        Acceptance probability in ``[0, 1]``.

    Raises
    ------
    ValueError
        If an input is non-finite or outside its domain, or if removal is
        requested from zero count.
    """
    return acceptance

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
from numbers import Integral, Real


def _oracle_compute_link_acceptance(
    beta: float,
    local_weight: float,
    local_count: int,
    move: int,
) -> float:
    import math
    from numbers import Integral, Real

    if isinstance(beta, bool) or not isinstance(beta, Real) or not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be a finite real scalar > 0")
    if isinstance(local_weight, bool) or not isinstance(local_weight, Real) or not math.isfinite(float(local_weight)) or float(local_weight) <= 0.0:
        raise ValueError("local_weight must be a finite real scalar > 0")
    if isinstance(local_count, bool) or not isinstance(local_count, Integral) or int(local_count) < 0:
        raise ValueError("local_count must be a nonnegative integer")
    if isinstance(move, bool) or not isinstance(move, Integral) or int(move) not in (-1, 1):
        raise ValueError("move must be +1 or -1")
    n = int(local_count)
    if int(move) == -1 and n == 0:
        raise ValueError("cannot remove a link from zero count")
    raw = float(beta) * float(local_weight) / float(n + 1) if int(move) == 1 else float(n) / (float(beta) * float(local_weight))
    return float(min(1.0, raw))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\nbeta=2.7\nw=0.56\nn=16\nmove=1", "call": "compute_link_acceptance(beta,w,n,move)", "gold_call": "_oracle_compute_link_acceptance(beta,w,n,move)"},
        {"setup": "import numpy as np\nbeta=2.7\nw=1.33\nn=19\nmove=-1", "call": "compute_link_acceptance(beta,w,n,move)", "gold_call": "_oracle_compute_link_acceptance(beta,w,n,move)"},
        {"setup": "import numpy as np\nbeta=2.0\nw=5.0\nn=0\nmove=1", "call": "compute_link_acceptance(beta,w,n,move)", "gold_call": "_oracle_compute_link_acceptance(beta,w,n,move)"},
        {"setup": "import numpy as np\nbeta=4.0\nw=0.5\nn=2\nmove=-1", "call": "compute_link_acceptance(beta,w,n,move)", "gold_call": "_oracle_compute_link_acceptance(beta,w,n,move)"},
        {"setup": """import numpy as np
def run_model():
 try:
  compute_link_acceptance(1.0,1.0,0,-1)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_compute_link_acceptance(1.0,1.0,0,-1)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""", "call": "run_model()", "gold_call": "run_gold()"},
        {"setup": """import numpy as np
def run_model():
 try:
  compute_link_acceptance(float('nan'),1.0,1,1)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_compute_link_acceptance(float('nan'),1.0,1,1)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""", "call": "run_model()", "gold_call": "run_gold()"},
    ]
