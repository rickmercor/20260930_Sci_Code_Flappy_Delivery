"""
Quantify the amplification of significant catalytic-efficiency epistasis produced by increasing the kinetic-cycle complexity. Given the significant-interaction prevalences obtained independently for the simple and extended mechanisms, return their dimensionless prevalence ratio.

A central result of the study is that kinetic-mechanism complexity can amplify non-specific epistasis even when the underlying microscopic mutation effects remain additive. The simple and extended catalytic cycles transform the same type of additive energetic perturbation through different kinetic mappings. Comparing the resulting prevalence of significant catalytic-efficiency interactions isolates the effect of mechanism complexity from the definition of the microscopic null model. The resulting ratio is therefore a direct quantitative measure of epistasis amplification caused by the additional kinetic states and transitions.

Returns
-------
float, the dimensionless ratio of significant catalytic-efficiency epistasis prevalence in the extended mechanism to that in the simple mechanism.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_complexity_ratio(
    simple_stats: "np.ndarray",
    complete_stats: "np.ndarray",
) -> float:
    """
    Compute the prevalence ratio between the reduced and complete
    representations.

    Parameters
    ----------
    simple_stats : np.ndarray
        Length-4 summary array for the reduced representation.

    complete_stats : np.ndarray
        Length-4 summary array for the complete representation.

    Returns
    -------
    float
        Complete-representation significant-interaction prevalence divided
        by reduced-representation significant-interaction prevalence.

    Raises
    ------
    ValueError
        If either statistics array has an invalid shape or contains
        invalid values, or if the reduced prevalence is not positive.
    """
    return ratio

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_complexity_ratio(
    simple_stats: "np.ndarray",
    complete_stats: "np.ndarray",
) -> float:
    s = np.asarray(simple_stats, dtype=float)
    c = np.asarray(complete_stats, dtype=float)
    if s.shape != (4,) or c.shape != (4,) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(c)) or s[0] <= 0 or c[0] < 0:
        raise ValueError("invalid statistics")
    if s[3] <= 0:
        raise ValueError("simple prevalence must be positive")
    out = float(c[3] / s[3])
    if not np.isfinite(out) or out <= 0:
        raise ValueError("invalid ratio")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\ns=np.array([38.,26.,12.,38/120]); c=np.array([45.,35.,10.,45/120])\n",
            "call": "compute_complexity_ratio(s,c)",
            "gold_call": "_oracle_compute_complexity_ratio(s,c)"
        },
        {
            "setup": "import numpy as np\ns=np.array([1.,1.,0.,1.]); c=np.array([2.,1.,1.,.5])\n",
            "call": "compute_complexity_ratio(s,c)",
            "gold_call": "_oracle_compute_complexity_ratio(s,c)"
        },
        {
            "setup": "import numpy as np\ns=np.array([120.,60.,60.,.5]); c=np.array([120.,60.,60.,.5])\n",
            "call": "compute_complexity_ratio(s,c)",
            "gold_call": "_oracle_compute_complexity_ratio(s,c)"
        },
        {
            "setup": "import numpy as np\ns=np.array([0.,0.,0.,0.]); c=np.array([1.,1.,0.,1.])\ndef run_model():\n    try: compute_complexity_ratio(s,c); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef run_gold():\n    try: _oracle_compute_complexity_ratio(s,c); return 0\n    except ValueError: return 1\n    except Exception: return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
    ]
