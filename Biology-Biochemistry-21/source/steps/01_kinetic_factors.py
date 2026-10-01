"""
Calculate source-defined kinetic factors for the enzyme panel.

The kinetic factor combines substrate affinity, catalytic turnover and pathway weight before nutrient restriction is applied.

Returns
-------
a float64 vector with one kinetic factor per reaction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def kinetic_factors(
    affinity: "np.ndarray",
    turnover: "np.ndarray",
    pathway_weight: "np.ndarray",
) -> "np.ndarray":
    """Calculate kinetic factors.

    Parameters
    ----------
    affinity, turnover, pathway_weight
        Finite positive one-dimensional arrays with identical shape.

    Returns
    -------
    np.ndarray
        A float64 vector with one kinetic factor per reaction.

    Raises
    ------
    ValueError
        If an input is not one-dimensional, finite and positive, or if the
        three shapes differ.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_kinetic_factors(
    affinity: "np.ndarray",
    turnover: "np.ndarray",
    pathway_weight: "np.ndarray",
) -> "np.ndarray":
    affinity = np.asarray(affinity, dtype=np.float64)
    turnover = np.asarray(turnover, dtype=np.float64)
    pathway_weight = np.asarray(pathway_weight, dtype=np.float64)
    if (
        affinity.ndim != 1
        or affinity.size == 0
        or turnover.shape != affinity.shape
        or pathway_weight.shape != affinity.shape
        or not np.isfinite(affinity).all()
        or not np.isfinite(turnover).all()
        or not np.isfinite(pathway_weight).all()
        or np.any(affinity <= 0.0)
        or np.any(turnover <= 0.0)
        or np.any(pathway_weight <= 0.0)
    ):
        raise ValueError("kinetic inputs must be aligned finite positive vectors")
    return np.sqrt(affinity * turnover / pathway_weight).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
a=np.array([0.18,0.055,0.092,0.041,0.077,0.032,0.068,0.049],float)
t=np.array([54.,31.,47.,28.,39.,22.,18.,25.])
b=np.array([1.,0.82,0.82,0.73,0.73,0.95,0.61,0.61])
ca,ct,cb=a.copy(),t.copy(),b.copy()
ga,gt,gb=a.copy(),t.copy(),b.copy()""",
            "call": "kinetic_factors(ca,ct,cb)",
            "gold_call": "_oracle_kinetic_factors(ga,gt,gb)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
a=np.ones(4); t=np.array([1.,4.,9.,16.]); b=np.ones(4)
ca,ct,cb=a.copy(),t.copy(),b.copy(); ga,gt,gb=a.copy(),t.copy(),b.copy()""",
            "call": "kinetic_factors(ca,ct,cb)",
            "gold_call": "_oracle_kinetic_factors(ga,gt,gb)",
            "tol": 1e-12,
        },
        {
            "setup": """import numpy as np
a=np.array([0.25]); t=np.array([16.]); b=np.array([4.])
ca,ct,cb=a.copy(),t.copy(),b.copy(); ga,gt,gb=a.copy(),t.copy(),b.copy()""",
            "call": "kinetic_factors(ca,ct,cb)",
            "gold_call": "_oracle_kinetic_factors(ga,gt,gb)",
            "tol": 0.0,
        },
        {
            "setup": """import numpy as np
a=np.array([0.1,-0.2]); t=np.array([2.,3.]); b=np.ones(2)
ca,ct,cb=a.copy(),t.copy(),b.copy(); ga,gt,gb=a.copy(),t.copy(),b.copy()
def case_raises(fn,*args):
    try: fn(*args)
    except ValueError: return 1.0
    return 0.0""",
            "call": "case_raises(kinetic_factors,ca,ct,cb)",
            "gold_call": "case_raises(_oracle_kinetic_factors,ga,gt,gb)",
            "tol": 0.0,
        },
    ]
