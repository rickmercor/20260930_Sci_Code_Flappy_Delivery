"""
Return the electron repulsion integrals of the same primitive basis in chemists' notation.

For four s primitives of a common exponent the repulsion integral in chemists'

notation reduces to a product of the two Gaussian product prefactors and one

zeroth order Boys function whose argument is set by the separation of the two

product centres. The resulting tensor carries the usual permutational symmetry

in the first pair, in the second pair, and between the two pairs. The exchange

type integrals between orbitals sitting on different parts of the molecule are

the ones that make the higher seniority excitations couple to the reference at

all, so they must not be discarded on the grounds that they look small.

Returns
-------
np.ndarray of shape (N, N, N, N): the electron repulsion integrals (mn|ab) in chemists' notation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ao_two_electron_integrals(centres: np.ndarray, exponent: float) -> np.ndarray:
    '''Electron repulsion integrals of the primitive basis.

    Parameters
    ----------
    centres : np.ndarray
        Positions of the nuclei along a line, in bohr, an even number of them.
    exponent : float
        Gaussian exponent shared by every primitive, strictly positive.

    Returns
    -------
    eri : np.ndarray
        Array of shape (N, N, N, N) holding (mn|ab) in chemists' notation.
    
    Raises
    ------
    ValueError
        If `centres` is not a one dimensional array holding an even number of
        distinct positions, or if `exponent` is not a positive scalar.
'''
    return eri

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf


def _boys0(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    m = x > 1e-12
    out[m] = 0.5 * np.sqrt(np.pi / x[m]) * erf(np.sqrt(x[m]))
    return out


def _check_model(centres, exponent):
    R = np.asarray(centres, dtype=float)
    if R.ndim != 1 or R.size < 2 or R.size % 2:
        raise ValueError("centres must be a 1D array of an even number of positions")
    if np.any(np.abs(R[:, None] - R[None, :]) + np.eye(R.size) < 1e-8):
        raise ValueError("centres must be distinct")
    if not np.isscalar(exponent) or float(exponent) <= 0.0:
        raise ValueError("exponent must be a positive scalar")
    return R, float(exponent)


def _oracle_ao_two_electron_integrals(centres, exponent):
    R = np.asarray(centres, dtype=float)
    if R.ndim != 1 or R.size < 2 or R.size % 2:
        raise ValueError("centres must be a 1D array of an even number of positions")
    if np.any(np.abs(R[:, None] - R[None, :]) + np.eye(R.size) < 1e-8):
        raise ValueError("centres must be distinct")
    if not np.isscalar(exponent) or float(exponent) <= 0.0:
        raise ValueError("exponent must be a positive scalar")
    a = float(exponent)
    p = 2.0 * a
    nrm = (2.0 * a / np.pi) ** 0.75
    d2 = (R[:, None] - R[None, :]) ** 2
    E = np.exp(-0.5 * a * d2)
    P = 0.5 * (R[:, None] + R[None, :])
    pre = 2.0 * np.pi ** 2.5 / (p * p * np.sqrt(2.0 * p)) * nrm ** 4
    Q = (P[:, :, None, None] - P[None, None, :, :]) ** 2
    return pre * E[:, :, None, None] * E[None, None, :, :] * _boys0(0.5 * p * Q)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
X = np.array([0.0, 3.0, 6.6, 9.8, 13.5, 16.4, 20.2, 23.6])
""",
            "call": "ao_two_electron_integrals(X, 0.30)",
            "gold_call": "_oracle_ao_two_electron_integrals(X, 0.30)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 3.0, 6.6, 9.8, 13.5, 16.4])
""",
            "call": "ao_two_electron_integrals(X, 0.30)",
            "gold_call": "_oracle_ao_two_electron_integrals(X, 0.30)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 2.8, 6.2, 8.9])
""",
            "call": "ao_two_electron_integrals(X, 0.85)",
            "gold_call": "_oracle_ao_two_electron_integrals(X, 0.85)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 1.0])
""",
            "call": "ao_two_electron_integrals(X, 2.5)",
            "gold_call": "_oracle_ao_two_electron_integrals(X, 2.5)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 20.0, 40.0, 60.0])
""",
            "call": "ao_two_electron_integrals(X, 0.30)",
            "gold_call": "_oracle_ao_two_electron_integrals(X, 0.30)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 3.0, 3.0, 6.0])

def run_model():
    try:
        ao_two_electron_integrals(X, 0.3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_ao_two_electron_integrals(X, 0.3)
        return 0
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
X = np.array([0.0, 3.0, 6.0, 9.0])

def run_model():
    try:
        ao_two_electron_integrals(X, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_ao_two_electron_integrals(X, 0.0)
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
