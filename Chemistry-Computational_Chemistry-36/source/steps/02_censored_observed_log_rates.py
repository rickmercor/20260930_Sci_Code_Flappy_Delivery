"""
Estimate per-set observed rates with a right-censored exponential MLE.

Right-censored trajectories contribute exposure even though they do not add
an observed event to the likelihood.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def censored_observed_log_rates(
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    dt: float,
) -> 'np.ndarray':
    """Return one observed log-rate per simulation set.

    A trajectory of ``lengths[j,i]`` stored intervals contributes exposure
    ``lengths[j,i]*dt`` whether it transitions or is right-censored.  The
    event count is the sum of the binary ``events`` row.

    Raises ``ValueError`` for invalid arrays, zero-event sets, or nonfinite
    rates.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_censored_observed_log_rates(
    lengths: 'np.ndarray',
    events: 'np.ndarray',
    dt: float,
) -> 'np.ndarray':
    try:
        lens = np.asarray(lengths)
        ev = np.asarray(events)
        step = float(dt)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("lengths, events, and dt must be numeric") from exc
    if lens.ndim != 2 or lens.size == 0 or ev.shape != lens.shape:
        raise ValueError("lengths and events must be nonempty matching matrices")
    if not np.issubdtype(lens.dtype, np.integer) or not np.issubdtype(ev.dtype, np.number):
        raise ValueError("lengths must be integers and events numeric")
    ev = ev.astype(float, copy=False)
    if (
        np.any(lens <= 0) or np.any(~np.isfinite(ev))
        or np.any((ev != 0.0) & (ev != 1.0))
        or not np.isfinite(step) or step <= 0.0
    ):
        raise ValueError("invalid lengths, events, or dt")
    counts = np.sum(ev, axis=1)
    if np.any(counts <= 0.0):
        raise ValueError("every set must contain at least one transition")
    exposure = np.sum(lens.astype(float), axis=1) * step
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.log(counts / exposure)
    if np.any(~np.isfinite(result)):
        raise ValueError("observed log-rate is not finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nl=np.array([[5,7,6],[3,4,5],[9,8,7]]); e=np.array([[1,0,1],[1,1,0],[0,1,0]])", "call": "censored_observed_log_rates(l,e,.2)", "gold_call": "_oracle_censored_observed_log_rates(l,e,.2)", "tol": 1e-12},
        {"setup": "import numpy as np\nl=np.ones((3,4),dtype=int); e=np.ones((3,4),dtype=int)", "call": "censored_observed_log_rates(l,e,1.)", "gold_call": "_oracle_censored_observed_log_rates(l,e,1.)", "tol": 1e-12},
        {"setup": "import numpy as np\nl=np.array([[1000000],[2000000],[4000000]]); e=np.ones((3,1),dtype=int)", "call": "censored_observed_log_rates(l,e,1e-9)", "gold_call": "_oracle_censored_observed_log_rates(l,e,1e-9)", "tol": 1e-12},
        {"setup": "import numpy as np\ndef check(fn):\n out=[]\n for l,e,d in ((np.ones((3,2),int),np.zeros((3,2)),.1),(np.ones((3,2),int),np.full((3,2),np.nan),.1),(np.ones((3,2),int),np.ones((3,2)),0.)):\n  try: fn(l,e,d)\n  except ValueError: out.append(1)\n  except Exception: out.append(2)\n  else: out.append(0)\n return np.array(out)", "call": "check(censored_observed_log_rates)", "gold_call": "check(_oracle_censored_observed_log_rates)", "tol": 0.0},
    ]
