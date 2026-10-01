"""
Convert residual unit clocks into physical completion waits.

Each pending transcript retains its age and unused unit-clock interval, so the cumulative hazard inversion starts from its current age rather than zero.

Returns
-------
np.ndarray: one non-negative physical waiting time per pending transcript.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def invert_completion_waits(ages: "np.ndarray", residuals: "np.ndarray",
                            state: "np.ndarray",
                            parameters: "np.ndarray") -> "np.ndarray":
    """Return physical waiting times that exhaust the pending residual clocks.

    Raises
    ------
    ValueError
        If arrays have inconsistent shapes, ages are negative, residuals are
        non-positive, or the state and delay parameters are inadmissible.
    """
    return waiting_times  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_invert_completion_waits(ages: "np.ndarray", residuals: "np.ndarray",
                                     state: "np.ndarray",
                                     parameters: "np.ndarray") -> "np.ndarray":
    a = np.asarray(ages, dtype=float)
    r = np.asarray(residuals, dtype=float)
    x = np.asarray(state, dtype=float)
    p = np.asarray(parameters, dtype=float)
    if a.ndim != 1 or r.shape != a.shape or x.shape != (3,) or p.shape != (10,):
        raise ValueError("input shapes are inconsistent")
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("ages must be finite and non-negative")
    if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
        raise ValueError("residual clocks must be finite and positive")
    if np.any(x < 0.0) or not np.all(np.isfinite(x)) or p[1] <= 0 or p[5] <= 0 or p[6] <= 0 or p[7] < 0:
        raise ValueError("state or parameters are inadmissible")
    coefficient = p[6] ** p[5] * p[1] ** p[5]
    coefficient /= p[6] ** p[5] + p[7] * (x[0] + x[2]) ** p[5]
    waits = (a ** p[5] + r / coefficient) ** (1.0 / p[5]) - a
    return np.maximum(waits, 0.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\na=np.array([0.,1.,3.]); r=np.array([.2,1.,2.]); x=np.array([5,8,3]); p=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])",
            "call": "float(np.sum(invert_completion_waits(a,r,x,p)))",
            "gold_call": "float(np.sum(_oracle_invert_completion_waits(a,r,x,p)))",
        },
        {
            "setup": "import numpy as np\na=np.array([0.]); r=np.array([1.]); x=np.array([0,0,1]); p=np.array([10,.175,1,.08,.05,1,10,0,1.5,5.])",
            "call": "float(invert_completion_waits(a,r,x,p)[0])",
            "gold_call": "float(_oracle_invert_completion_waits(a,r,x,p)[0])",
        },
        {
            "setup": "import numpy as np\na=np.array([5.]); r=np.array([1e-12]); x=np.array([1,1,1]); p=np.array([1,.2,1,.1,.1,1,10,0,1,1.])",
            "call": "float(invert_completion_waits(a,r,x,p)[0])",
            "gold_call": "float(_oracle_invert_completion_waits(a,r,x,p)[0])",
        },
        {"setup": "import numpy as np\np=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])\nx=np.array([5,8,3])\ndef bad():\n    try: invert_completion_waits(np.array([0.]),np.array([0.]),x,p); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef bad_gold():\n    try: _oracle_invert_completion_waits(np.array([0.]),np.array([0.]),x,p); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "bad()", "gold_call": "bad_gold()"},
    ]
