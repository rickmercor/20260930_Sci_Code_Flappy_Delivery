"""
Compute aggregate completion means for existing delay groups.

A group shares one initiation time and age. Its completion process scales the single-transcript integrated hazard by the unfinished population of the group.

Returns
-------
np.ndarray: one Poisson completion mean per pre-existing delay group.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_group_completion_means(group_ages: "np.ndarray",
                                   group_sizes: "np.ndarray", dt: float,
                                   state: "np.ndarray",
                                   parameters: "np.ndarray") -> "np.ndarray":
    """Return Poisson means for pre-existing delay groups over one leap.

    Raises
    ------
    ValueError
        If group arrays disagree, group sizes are not positive counts, or
        the hazard inputs are inadmissible.
    """
    return means  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_group_completion_means(group_ages: "np.ndarray",
                                            group_sizes: "np.ndarray", dt: float,
                                            state: "np.ndarray",
                                            parameters: "np.ndarray") -> "np.ndarray":
    ages = np.asarray(group_ages, dtype=float)
    sizes = np.asarray(group_sizes, dtype=float)
    if ages.ndim != 1 or sizes.shape != ages.shape:
        raise ValueError("group arrays must be one-dimensional and aligned")
    if not np.all(np.isfinite(sizes)) or np.any(sizes <= 0.0) or np.any(sizes != np.floor(sizes)):
        raise ValueError("group sizes must be positive integer counts")
    means = sizes * _oracle_completion_hazard_increment(ages, dt, state, parameters)
    if np.any(means < 0.0) or not np.all(np.isfinite(means)):
        raise ValueError("completion means must be finite and non-negative")
    return means

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\na=np.array([0.,2.,5.]); s=np.array([3,1,7]); x=np.array([4,9,11]); p=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])",
            "call": "float(np.dot(compute_group_completion_means(a,s,2.5,x,p),s))",
            "gold_call": "float(np.dot(_oracle_compute_group_completion_means(a,s,2.5,x,p),s))",
        },
        {
            "setup": "import numpy as np\na=np.array([0.]); s=np.array([1]); x=np.array([0,0,1]); p=np.array([1,.2,1,.1,.1,1,4,0,1,2.])",
            "call": "float(compute_group_completion_means(a,s,.5,x,p)[0])",
            "gold_call": "float(_oracle_compute_group_completion_means(a,s,.5,x,p)[0])",
        },
        {
            "setup": "import numpy as np\na=np.array([4.]); s=np.array([50]); x=np.array([10,0,50]); p=np.array([1,.2,1,.1,.1,2,4,1,1,2.])",
            "call": "float(compute_group_completion_means(a,s,.01,x,p)[0])",
            "gold_call": "float(_oracle_compute_group_completion_means(a,s,.01,x,p)[0])",
        },
        {"setup": "import numpy as np\np=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])\nx=np.array([5,8,3])\ndef bad():\n    try: compute_group_completion_means(np.array([0.]),np.array([0]),1.,x,p); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef bad_gold():\n    try: _oracle_compute_group_completion_means(np.array([0.]),np.array([0]),1.,x,p); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "bad()", "gold_call": "bad_gold()"},
    ]
