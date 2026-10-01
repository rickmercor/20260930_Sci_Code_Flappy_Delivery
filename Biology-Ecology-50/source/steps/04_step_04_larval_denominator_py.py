"""
Calculate equilibrium surviving larval output at a sampled node.

At constant adult population size, all possible larval ages contribute to the number of surviving larvae at a site. Egg and larval survival multiply the expected output from the local adult females.

Returns
-------
float: equilibrium expected surviving larval output at the site.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def larval_denominator(nf: float, beta: float, te: int, tl: int, mu_e: float, mu_l: float) -> float:
    """Calculate equilibrium surviving larval output at a sampled node.

    Parameters
    ----------
    nf : equilibrium adult female count at node
    beta : eggs per female per day
    te : egg-stage duration in days
    tl : larval-stage duration in days
    mu_e : daily egg mortality
    mu_l : daily larval mortality

    Returns
    -------
    float, expected surviving larval offspring at the site

    Notes
    -----
    Return the stationary expected surviving larval output over all TL admissible egg-laying days using the paper's egg and larval survival factors.
    E_L = nf * beta * (1-mu_e)**te * sum((1-mu_l)**a for a=0,...,tl-1).
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math

def _oracle_larval_denominator(nf: float, beta: float, te: int, tl: int, mu_e: float, mu_l: float) -> float:
    return nf * beta * (1-mu_e)**te * sum((1-mu_l)**a for a in range(tl))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Three independent source-method scenarios."""
    return [
        {"setup": 'import numpy as np\ncoords=np.array([[0.,0.],[0.,20.],[20.,0.],[20.,20.]])\n', "call": 'larval_denominator(25,20,2,5,.175,.554)', "gold_call": '_oracle_larval_denominator(25,20,2,5,.175,.554)', "tol": 1e-09},
        {"setup": 'import numpy as np\ncoords=np.array([[0.,0.],[0.,20.],[20.,0.],[20.,20.]])\n', "call": 'larval_denominator(25,20,2,1,.175,.554)', "gold_call": '_oracle_larval_denominator(25,20,2,1,.175,.554)', "tol": 1e-09},
        {"setup": 'import numpy as np\ncoords=np.array([[0.,0.],[0.,20.],[20.,0.],[20.,20.]])\n', "call": 'larval_denominator(10,15,1,3,.1,.2)', "gold_call": '_oracle_larval_denominator(10,15,1,3,.1,.2)', "tol": 1e-09},
    ]
