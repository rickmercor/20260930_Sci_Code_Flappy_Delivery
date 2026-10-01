"""
Convert an energy release rate (J-integral value) into the mode-I stress intensity factor (the paper's Eq. 9 when dynamic=True, its quasi-static counterpart Eq. 16 when dynamic=False). The dynamic conversion divides by the velocity-dependent factor A_I; the quasi-static conversion uses the plane-strain effective modulus and is independent of the crack speed. Recover both relations from the paper.

The J-integral measures energy flux; fracture criteria are stated in terms of the stress intensity factor, so the velocity-dependent conversion between them is what makes a dynamic-fracture prediction, and using the quasi-static conversion for a fast crack is a definite error.

Returns
-------
return float: the mode-I stress intensity factor
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def dsif(J, A_I, E, nu, dynamic=True):
    """J: energy release rate (>= 0); A_I: the velocity factor from wave_factor; E, nu:
    plane-strain constants; dynamic: if True use the dynamic conversion (paper Eq. 9),
    else the quasi-static one (paper Eq. 16). Returns float: the mode-I stress intensity
    factor. Raises ValueError if J < 0."""
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 7: dynamic and quasi-static stress intensity factors (Eqs. 9, 16)."""

import numpy as np


def _oracle_dsif(J, A_I, E, nu, dynamic=True):
    if J < 0:
        raise ValueError("J must be non-negative")
    if dynamic:
        return float(np.sqrt(E * J / ((1.0 + nu) * A_I)))
    return float(np.sqrt(E * J / (1.0 - nu ** 2)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'J=0.29;A_I=1.0837;E=1.0;nu=0.3', "call": 'dsif(J, A_I, E, nu, True)', "gold_call": '_oracle_dsif(J, A_I, E, nu, True)', "tol": 1e-10},
        {"setup": 'J=0.23;A_I=1.0837;E=1.0;nu=0.3', "call": 'dsif(J, A_I, E, nu, False)', "gold_call": '_oracle_dsif(J, A_I, E, nu, False)', "tol": 1e-10},
        {"setup": 'J=0.5;A_I=1.876;E=1.5;nu=0.35', "call": 'dsif(J, A_I, E, nu, True)', "gold_call": '_oracle_dsif(J, A_I, E, nu, True)', "tol": 1e-10},
    ]
