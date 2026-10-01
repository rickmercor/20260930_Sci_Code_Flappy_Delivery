"""
Apply one supplied strict Metropolis decision to an independent copy of a local link-count vector. The returned numeric array contains the updated counts followed by the 0/1 decision.

Accepted changes immediately alter the expansion order seen by later proposals, whereas rejected proposals leave the state unchanged. Copy semantics prevent a caller's input from being mutated implicitly.

Returns
-------
np.ndarray, the updated integer counts followed by one acceptance flag
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def apply_link_move(
    counts: np.ndarray,
    location: int,
    move: int,
    acceptance: float,
    uniform: float,
) -> np.ndarray:
    """Apply one accepted or rejected link-count mutation.

    Parameters
    ----------
    counts : np.ndarray
        Nonempty one-dimensional nonnegative integer count vector.
    location : int
        Zero-based selected location.
    move : int
        ``+1`` for insertion or ``-1`` for removal.
    acceptance : float
        Acceptance probability in ``[0, 1]``.
    uniform : float
        Supplied deviate in ``[0, 1]``; acceptance is strictly ``uniform < acceptance``.

    Returns
    -------
    packed_result : np.ndarray
        Integer vector of length ``counts.size + 1``. The first entries are an
        independent updated count vector and the last is the decision flag.

    Raises
    ------
    ValueError
        If the count vector or a scalar argument is invalid, or removal is
        proposed at a zero-count location.
    """
    return packed_result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
from numbers import Integral, Real

def _oracle_apply_link_move(
    counts: np.ndarray,
    location: int,
    move: int,
    acceptance: float,
    uniform: float,
) -> np.ndarray:
    import math
    from numbers import Integral, Real
    import numpy as np

    raw = np.asarray(counts)
    if raw.ndim != 1 or raw.size < 1 or not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_) or np.any(raw < 0):
        raise ValueError("counts must be a nonempty one-dimensional nonnegative integer array")
    if isinstance(location, bool) or not isinstance(location, Integral) or not (0 <= int(location) < raw.size):
        raise ValueError("location is out of range")
    if isinstance(move, bool) or not isinstance(move, Integral) or int(move) not in (-1, 1):
        raise ValueError("move must be +1 or -1")
    for name, value in (("acceptance", acceptance), ("uniform", uniform)):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name} must be finite and in [0,1]")
    loc = int(location)
    if int(move) == -1 and int(raw[loc]) == 0:
        raise ValueError("cannot remove a link from zero count")
    updated = raw.astype(int, copy=True)
    accepted = int(float(uniform) < float(acceptance))
    if accepted:
        updated[loc] += int(move)
    out = np.empty(updated.size + 1, dtype=int)
    out[:-1] = updated
    out[-1] = accepted
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\ncounts=np.array([16,18,20,22,17,19],dtype=int)", "call": "(lambda x: np.concatenate((apply_link_move(x,0,1,0.08894117647058825,0.037165510406342916),x)))(counts.copy())", "gold_call": "(lambda x: np.concatenate((_oracle_apply_link_move(x,0,1,0.08894117647058825,0.037165510406342916),x)))(counts.copy())"},
        {"setup": "import numpy as np\ncounts=np.array([1,2],dtype=int)", "call": "(lambda x: np.concatenate((apply_link_move(x,0,-1,0.6,0.6),x)))(counts.copy())", "gold_call": "(lambda x: np.concatenate((_oracle_apply_link_move(x,0,-1,0.6,0.6),x)))(counts.copy())"},
        {"setup": "import numpy as np\ncounts=np.array([0],dtype=int)", "call": "(lambda x: np.concatenate((apply_link_move(x,0,1,1.0,1.0),x)))(counts.copy())", "gold_call": "(lambda x: np.concatenate((_oracle_apply_link_move(x,0,1,1.0,1.0),x)))(counts.copy())"},
        {"setup": "import numpy as np\ncounts=np.array([3,4],dtype=int)", "call": "(lambda x: np.concatenate((apply_link_move(x,1,-1,1.0,0.0),x)))(counts.copy())", "gold_call": "(lambda x: np.concatenate((_oracle_apply_link_move(x,1,-1,1.0,0.0),x)))(counts.copy())"},
        {"setup": """import numpy as np
counts=np.array([0,1],dtype=int)
def run_model():
 try:
  apply_link_move(counts.copy(),0,-1,0.0,1.0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_apply_link_move(counts.copy(),0,-1,0.0,1.0)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""", "call": "run_model()", "gold_call": "run_gold()"},
        {"setup": """import numpy as np
counts=np.array([1,2],dtype=int)
def run_model():
 try:
  apply_link_move(counts.copy(),2,1,0.5,0.1)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_apply_link_move(counts.copy(),2,1,0.5,0.1)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""", "call": "run_model()", "gold_call": "run_gold()"},
    ]
