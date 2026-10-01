"""
Advance all residual clocks to the next exact event.

When one event wins the race, every other channel keeps its unused internal interval after subtracting only the integrated hazard accumulated meanwhile.

Returns
-------
np.ndarray: length 4 + 2 * len(ages) holding the four channel residuals, then the completion residuals, then the advanced ages.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def advance_remaining_clocks(reaction_residuals: "np.ndarray",
                             ages: "np.ndarray", completion_residuals: "np.ndarray",
                             dt: float, state: "np.ndarray",
                             parameters: "np.ndarray") -> "np.ndarray":
    """Deduct accumulated hazards and advance delay ages over ``dt``.

    Returns the four reaction residuals, then completion residuals, then ages.

    Raises
    ------
    ValueError
        If clock arrays are inconsistent or contain inadmissible values.
    """
    return packed_clocks  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_advance_remaining_clocks(reaction_residuals: "np.ndarray",
                                      ages: "np.ndarray",
                                      completion_residuals: "np.ndarray",
                                      dt: float, state: "np.ndarray",
                                      parameters: "np.ndarray") -> "np.ndarray":
    rr = np.asarray(reaction_residuals, dtype=float)
    a = np.asarray(ages, dtype=float)
    cr = np.asarray(completion_residuals, dtype=float)
    if rr.shape != (4,) or a.ndim != 1 or cr.shape != a.shape:
        raise ValueError("clock arrays have inconsistent shapes")
    if np.any(rr <= 0.0) or np.any(cr <= 0.0) or np.any(a < 0.0):
        raise ValueError("residuals must be positive and ages non-negative")
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    prop = _oracle_gene_channel_propensities(state, parameters)
    used_completion = _oracle_completion_hazard_increment(a, dt, state, parameters)
    next_rr = rr - prop * float(dt)
    next_cr = cr - used_completion
    tolerance = 1e-10
    if np.any(next_rr < -tolerance) or np.any(next_cr < -tolerance):
        raise ValueError("dt advances beyond the next scheduled event")
    next_rr = np.maximum(next_rr, 0.0)
    next_cr = np.maximum(next_cr, 0.0)
    return np.concatenate((next_rr, next_cr, a + float(dt)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nrr=np.array([2.,3.,4.,5.]); a=np.array([1.,2.]); cr=np.array([2.,3.]); x=np.array([2,3,2]); p=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])",
            "call": "float(np.sum(advance_remaining_clocks(rr,a,cr,.01,x,p)))",
            "gold_call": "float(np.sum(_oracle_advance_remaining_clocks(rr,a,cr,.01,x,p)))",
        },
        {
            "setup": "import numpy as np\nrr=np.ones(4); a=np.array([]); cr=np.array([]); x=np.zeros(3); p=np.array([1,.2,1,.1,.1,1,10,0,1,1.])",
            "call": "float(np.sum(advance_remaining_clocks(rr,a,cr,.1,x,p)))",
            "gold_call": "float(np.sum(_oracle_advance_remaining_clocks(rr,a,cr,.1,x,p)))",
        },
        {
            "setup": "import numpy as np\nrr=np.ones(4); a=np.array([0.]); cr=np.ones(1); x=np.array([0,0,1]); p=np.array([1,.2,1,.1,.1,2,10,0,1,1.])",
            "call": "float(np.dot(advance_remaining_clocks(rr,a,cr,.1,x,p),np.arange(1,7)))",
            "gold_call": "float(np.dot(_oracle_advance_remaining_clocks(rr,a,cr,.1,x,p),np.arange(1,7)))",
        },
        {"setup": "import numpy as np\np=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])\nx=np.array([5,8,3])\ndef bad():\n    try: advance_remaining_clocks(np.ones(3),np.empty(0),np.empty(0),0.,x,p); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef bad_gold():\n    try: _oracle_advance_remaining_clocks(np.ones(3),np.empty(0),np.empty(0),0.,x,p); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "bad()", "gold_call": "bad_gold()"},
    ]
