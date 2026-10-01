"""
Integrate pending completion hazards over a fixed-state interval.

The state is frozen during a race interval while every pending transcript's age advances, so its non-exponential hazard must be integrated over age.

Returns
-------
np.ndarray: integrated completion hazard per pending transcript, the same shape as ages.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def completion_hazard_increment(ages: "np.ndarray", dt: float,
                                state: "np.ndarray",
                                parameters: "np.ndarray") -> "np.ndarray":
    """Integrate every pending transcript's completion hazard over ``dt``.

    The biochemical state is frozen over the interval, but each transcript's
    age advances continuously.

    Raises
    ------
    ValueError
        If ages, state, parameters, or ``dt`` are non-finite or inadmissible.
    """
    return increments  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_completion_hazard_increment(ages: "np.ndarray", dt: float,
                                         state: "np.ndarray",
                                         parameters: "np.ndarray") -> "np.ndarray":
    a = np.asarray(ages, dtype=float)
    x = np.asarray(state, dtype=float)
    p = np.asarray(parameters, dtype=float)
    if a.ndim != 1 or x.shape != (3,) or p.shape != (10,):
        raise ValueError("ages, state, or parameters have the wrong shape")
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("ages must be finite and non-negative")
    if not isinstance(dt, (int, float, np.integer, np.floating)) or not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    if not np.all(np.isfinite(x)) or np.any(x < 0.0) or p[1] <= 0 or p[5] <= 0 or p[6] <= 0 or p[7] < 0:
        raise ValueError("state or parameters are inadmissible")
    scale = p[6] ** p[5] * p[1] ** p[5]
    scale /= p[6] ** p[5] + p[7] * (x[0] + x[2]) ** p[5]
    return scale * ((a + float(dt)) ** p[5] - a ** p[5])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\na=np.array([0.,1.,3.]); x=np.array([5,8,3]); p=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])",
            "call": "float(np.sum(completion_hazard_increment(a,2.5,x,p)))",
            "gold_call": "float(np.sum(_oracle_completion_hazard_increment(a,2.5,x,p)))",
        },
        {
            "setup": "import numpy as np\na=np.array([0.]); x=np.array([0,0,1]); p=np.array([10,.175,1,.08,.05,1,10,0,1.5,5.])",
            "call": "float(completion_hazard_increment(a,.25,x,p)[0])",
            "gold_call": "float(_oracle_completion_hazard_increment(a,.25,x,p)[0])",
        },
        {
            "setup": "import numpy as np\na=np.array([20.]); x=np.array([30,0,1]); p=np.array([1,.2,1,.1,.1,3,10,2,1,1.])",
            "call": "float(completion_hazard_increment(a,1e-3,x,p)[0])",
            "gold_call": "float(_oracle_completion_hazard_increment(a,1e-3,x,p)[0])",
        },
        {"setup": "import numpy as np\np=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])\nx=np.array([5,8,3])\ndef bad():\n    try: completion_hazard_increment(np.array([0.]),-1.,x,p); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef bad_gold():\n    try: _oracle_completion_hazard_increment(np.array([0.]),-1.,x,p); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "bad()", "gold_call": "bad_gold()"},
    ]
