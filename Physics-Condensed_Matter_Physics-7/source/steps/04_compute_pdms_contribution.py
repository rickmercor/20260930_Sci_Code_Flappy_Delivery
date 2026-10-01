"""
Construct the signed normalized projected-density and shifted-energy contribution from one post-decision configuration. The two matrix contributions share the same oriented boundary-amplitude outer product.

The path sign restores the original Monte Carlo weight. The shifted-energy estimator multiplies the density contribution by the post-decision total expansion order and the continuous-time factor $-1/\beta$.

Returns
-------
np.ndarray, a float array of shape (2, d, d), with density then shifted-energy contributions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_pdms_contribution(
    sign: int,
    final_amplitudes: np.ndarray,
    initial_amplitudes: np.ndarray,
    total_links: int,
    beta: float,
) -> np.ndarray:
    """Compute one signed projected-matrix sample.

    Parameters
    ----------
    sign : int
        Path sign, either ``-1`` or ``+1``.
    final_amplitudes, initial_amplitudes : np.ndarray
        Matched nonempty one-dimensional finite projection-amplitude vectors.
    total_links : int
        Nonnegative total number of links after the proposal decision.
    beta : float
        Strictly positive finite inverse temperature.

    Returns
    -------
    contributions : np.ndarray
        Float array of shape ``(2, d, d)``. Entry 0 is the density contribution;
        entry 1 is the shifted-energy contribution.

    Raises
    ------
    ValueError
        If a sign, vector, count, or inverse temperature is invalid.
    """
    return contributions

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
from numbers import Integral, Real

import numpy as np


def _oracle_compute_pdms_contribution(
    sign: int,
    final_amplitudes: np.ndarray,
    initial_amplitudes: np.ndarray,
    total_links: int,
    beta: float,
) -> np.ndarray:
    import math
    from numbers import Integral, Real
    import numpy as np

    if isinstance(sign, bool) or not isinstance(sign, Integral) or int(sign) not in (-1, 1):
        raise ValueError("sign must be -1 or +1")
    f = np.asarray(final_amplitudes, dtype=float)
    i = np.asarray(initial_amplitudes, dtype=float)
    if f.ndim != 1 or i.ndim != 1 or f.size < 1 or f.shape != i.shape or not np.all(np.isfinite(f)) or not np.all(np.isfinite(i)):
        raise ValueError("amplitude vectors must be matched nonempty finite one-dimensional arrays")
    if isinstance(total_links, bool) or not isinstance(total_links, Integral) or int(total_links) < 0:
        raise ValueError("total_links must be a nonnegative integer")
    if isinstance(beta, bool) or not isinstance(beta, Real) or not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be a finite real scalar > 0")
    density = float(int(sign)) * np.outer(f, i)
    shifted = -(float(int(total_links)) / float(beta)) * density
    return np.stack((density, shifted), axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\nf=np.array([0.5,-0.25])\ni=np.array([0.25,0.75])", "call": "compute_pdms_contribution(1,f.copy(),i.copy(),7,2.4)", "gold_call": "_oracle_compute_pdms_contribution(1,f.copy(),i.copy(),7,2.4)"},
        {"setup": "import numpy as np\nf=np.array([2.0])\ni=np.array([1.0])", "call": "compute_pdms_contribution(-1,f.copy(),i.copy(),0,3.0)", "gold_call": "_oracle_compute_pdms_contribution(-1,f.copy(),i.copy(),0,3.0)"},
        {"setup": "import numpy as np\nf=np.array([1.0,-2.0])\ni=np.array([3.0,-4.0])", "call": "compute_pdms_contribution(-1,f.copy(),i.copy(),18,1.0)", "gold_call": "_oracle_compute_pdms_contribution(-1,f.copy(),i.copy(),18,1.0)"},
        {"setup": """import numpy as np
f=np.array([1.0,2.0]); i=np.array([1.0])
def run_model():
 try:
  compute_pdms_contribution(1,f.copy(),i.copy(),1,1.0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_compute_pdms_contribution(1,f.copy(),i.copy(),1,1.0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""", "call": "run_model()", "gold_call": "run_gold()"},
        {"setup": """import numpy as np
f=np.array([1.0]); i=np.array([1.0])
def run_model():
 try:
  compute_pdms_contribution(0,f.copy(),i.copy(),1,1.0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_compute_pdms_contribution(0,f.copy(),i.copy(),1,1.0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""", "call": "run_model()", "gold_call": "run_gold()"},
    ]
