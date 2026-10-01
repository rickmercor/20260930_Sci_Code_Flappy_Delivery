"""
Compute the isotropic real-space damping factor S^G for one or more QM-site--MM-site centre-of-mass separations, given the damping parameter beta.  The result is a dimensionless scalar in [0, 1] for each separation; downstream it multiplies the bare QM--MM multipole interaction (and the field / field gradient it produces) for every near-field QM--MM pair, including periodic images.

Point-multipole polarizable-embedding QM/MM must screen the QM--MM electrostatic coupling at short range, or the induced-moment iteration diverges (a polarization catastrophe) when the QM density and an MM site come into close contact.  Earlier schemes damp this coupling per real-space grid point, which makes the screening depend on molecular orientation rather than on separation alone.  The method used here instead damps the QM/MM boundary isotropically: the screening depends only on the distance between the two centres of mass, independent of how either object is oriented.  When the QM region is reduced to a single fixed expansion centre, the spatially resolved damping function collapses to this scalar envelope of the QM--MM centre-of-mass distance alone.

Returns
-------
return np.zeros_like(np.asarray(r_com, dtype=float))
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def isotropic_qmmm_damping(r_com, beta: float) -> np.ndarray:
    '''Isotropic real-space QM-MM boundary damping factor S^G.

    Parameters
    ----------
    r_com : array_like
        QM-site--MM-site centre-of-mass separations, shape (...,).  Must be finite and
        non-negative.  Same length unit as ``1 / beta``.
    beta : float
        Isotropic damping parameter (strictly positive, inverse length).

    Returns
    -------
    S_G : np.ndarray
        Array with the same shape as ``r_com`` holding the scalar damping prefactor
        S^G(d) = erf(beta d) - (2/sqrt(pi)) beta d exp(-(beta d)^2), clipped to [0, 1].

    Raises
    ------
    ValueError
        If ``r_com`` contains a non-finite value or a negative separation, or if
        ``beta`` is not finite or is not strictly positive.
    '''
    return np.zeros_like(np.asarray(r_com, dtype=float))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf

def _oracle_isotropic_qmmm_damping(r_com, beta: float) -> np.ndarray:
    """Reference implementation of the isotropic QM-MM real-space damping envelope."""
    r = np.asarray(r_com, dtype=float)
    if not np.all(np.isfinite(r)):
        raise ValueError("r_com must be finite")
    if np.any(r < 0.0):
        raise ValueError("r_com must be non-negative (COM separations)")
    if not np.isfinite(beta) or float(beta) <= 0.0:
        raise ValueError("beta must be a finite strictly-positive number")

    x = float(beta) * r
    s = erf(x) - (2.0 / np.sqrt(np.pi)) * x * np.exp(-(x ** 2))
    # numerically S^G can leave [0, 1] by ~1e-16 near the end points
    s = np.clip(s, 0.0, 1.0)
    return np.asarray(s, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test-case specifications."""
    return [
        {
            "setup": """import numpy as np
r_com = np.array([1.5, 3.0, 5.2, 8.0, 12.0], dtype=float)
beta = 0.291 / 1.8897261246  # 0.291 1/Angstrom expressed in 1/Bohr
""",
            "call": "isotropic_qmmm_damping(r_com, beta)",
            "gold_call": "_oracle_isotropic_qmmm_damping(r_com, beta)",
        },
        {
            "setup": """import numpy as np
r_com = np.array([0.0, 1e3], dtype=float)
beta = 0.4
""",
            "call": "isotropic_qmmm_damping(r_com, beta)",
            "gold_call": "_oracle_isotropic_qmmm_damping(r_com, beta)",
        },
        {
            "setup": """import numpy as np
r_com = 4.0
beta = 0.15399
""",
            "call": "np.atleast_1d(isotropic_qmmm_damping(r_com, beta))",
            "gold_call": "np.atleast_1d(_oracle_isotropic_qmmm_damping(r_com, beta))",
        },
        {
            "setup": """import numpy as np
r_com = np.array([1.0, 2.0])
def run_model():
    try:
        isotropic_qmmm_damping(r_com, -0.1); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_isotropic_qmmm_damping(r_com, -0.1); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
r_com = np.array([1.0, -2.0])
def run_model():
    try:
        isotropic_qmmm_damping(r_com, 0.3); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_isotropic_qmmm_damping(r_com, 0.3); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
