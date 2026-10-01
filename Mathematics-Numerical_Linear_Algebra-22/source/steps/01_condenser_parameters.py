"""
Compute the parameters of the Mobius transformation that reduces the two-plate condenser formed by a negative pole interval and the nonnegative real axis to a symmetric condenser, together with the resulting elliptic parameter.



Given an interval [c, d] with c < d < 0, the condenser formed by the union of [c, d] and the nonnegative real axis is mapped by the transformation z -> (z - varsigma)/(z - varrho) onto a symmetric condenser consisting of the union of [-eta2, -eta1] and [eta1, eta2], with 0 < eta1 < eta2, where eta1 = -(d - varsigma)/(d - varrho) and eta2 = -(c - varsigma)/(c - varrho) are the negated images of the right and left endpoints of [c, d]. The fifth quantity is the elliptic parameter mu = 1 - (eta1/eta2)^2 of the symmetric condenser, that is, the square of the Jacobi modulus, in the convention of scipy.special.ellipk and scipy.special.ellipj; it is not the modulus itself.



Return the five quantities varsigma, varrho, eta1, eta2, mu in that order as a one-dimensional float array of length 5.



The function raises ValueError if either input is not finite, or if the ordering c < d < 0 does not hold.

A condenser is a pair of disjoint compact sets in the extended complex plane, and the third Zolotarev problem asks for the rational function of given degree whose modulus is largest on one plate relative to the other. Its extremal solution is expressed through Jacobi elliptic functions whose modulus is determined by the conformal capacity of the condenser, and closed-form node placements are classically available only for symmetric condensers, in which one plate is the reflection of the other about the origin.

The condenser arising in time-uniform real-pole approximation of $\exp(-tz)$ is not symmetric: one plate is a bounded interval on the negative real axis where the poles must lie, and the other is the unbounded nonnegative real axis where the approximation error is measured. Because Möbius transformations preserve both the class of rational functions of a given degree and the extremal ratio being optimized, the asymmetric problem can be transported to a symmetric one, solved there in closed form, and transported back. Fixing the transformation requires choosing its two free parameters so that the images of the interval endpoints are placed symmetrically about the origin relative to the image of the unbounded plate; the elliptic modulus of the transported condenser then controls every subsequent elliptic-function evaluation and approaches unity as the original interval widens.

Returns
-------
np.ndarray, a one-dimensional float64 array of length 5 containing [varsigma, varrho, eta1, eta2, mu]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def condenser_parameters(c: float, d: float) -> np.ndarray:
    '''Compute Möbius and elliptic parameters reducing [c,d] u [0,inf) to a symmetric condenser.

    Parameters
    ----------
    c : float
        Left endpoint of the pole interval. Must be finite with c < d < 0.
    d : float
        Right endpoint of the pole interval. Must be finite with c < d < 0.

    Returns
    -------
    params : np.ndarray
        One-dimensional float array of length 5 holding, in order, the two Möbius
        parameters varsigma and varrho, the two symmetric-condenser endpoints
        eta1 and eta2, and the elliptic parameter mu = 1 - (eta1/eta2)**2.

    Raises
    ------
    ValueError
        If either input is not finite, or if the ordering c < d < 0 does not hold.
    '''
    return params  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_condenser_parameters(c: float, d: float) -> np.ndarray:
    """Reference implementation."""
    c = float(c)
    d = float(d)
    if not (np.isfinite(c) and np.isfinite(d)):
        raise ValueError("c and d must be finite")
    if not (c < d < 0.0):
        raise ValueError("endpoints must satisfy c < d < 0")

    varsigma = c + np.sqrt(c * (c - d))
    varrho = 2.0 * c - varsigma

    eta1 = (varsigma - d) / (d - varrho)
    eta2 = (varsigma - c) / (c - varrho)
    mu = 1.0 - (eta1 / eta2) ** 2

    return np.array([varsigma, varrho, eta1, eta2, mu], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the refined production interval ---
        {
            "setup": """import numpy as np
c = -551.5183157669
d = -10.1118823417
""",
            "call": "condenser_parameters(c, d)",
            "gold_call": "_oracle_condenser_parameters(c, d)",
        },
        # --- normal: the unrefined initial interval ---
        {
            "setup": """import numpy as np
c = -1484.9242404917
d = -14.8492424049
""",
            "call": "condenser_parameters(c, d)",
            "gold_call": "_oracle_condenser_parameters(c, d)",
        },
        # --- normal: small integer interval, modulus well away from 1 ---
        {
            "setup": """import numpy as np
c = -2.0
d = -1.0
""",
            "call": "condenser_parameters(c, d)",
            "gold_call": "_oracle_condenser_parameters(c, d)",
        },
        # --- boundary: very narrow interval, eta1 close to eta2 ---
        {
            "setup": """import numpy as np
c = -1.0000001
d = -1.0
""",
            "call": "condenser_parameters(c, d)",
            "gold_call": "_oracle_condenser_parameters(c, d)",
        },
        # --- edge: extremely wide interval, modulus saturates at 1 in double precision ---
        {
            "setup": """import numpy as np
c = -1.0e6
d = -1.0e-3
""",
            "call": "condenser_parameters(c, d)",
            "gold_call": "_oracle_condenser_parameters(c, d)",
        },
        # --- edge: interval far from the origin ---
        {
            "setup": """import numpy as np
c = -1.0e8
d = -9.9e7
""",
            "call": "condenser_parameters(c, d)",
            "gold_call": "_oracle_condenser_parameters(c, d)",
        },
        # --- invalid: endpoints equal ---
        {
            "setup": """import numpy as np
c = -3.0
d = -3.0
def run_model():
    try:
        condenser_parameters(c, d)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_condenser_parameters(c, d)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: right endpoint not negative ---
        {
            "setup": """import numpy as np
c = -5.0
d = 0.0
def run_model():
    try:
        condenser_parameters(c, d)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_condenser_parameters(c, d)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: reversed ordering ---
        {
            "setup": """import numpy as np
c = -1.0
d = -8.0
def run_model():
    try:
        condenser_parameters(c, d)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_condenser_parameters(c, d)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-finite input ---
        {
            "setup": """import numpy as np
c = -np.inf
d = -1.0
def run_model():
    try:
        condenser_parameters(c, d)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_condenser_parameters(c, d)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
