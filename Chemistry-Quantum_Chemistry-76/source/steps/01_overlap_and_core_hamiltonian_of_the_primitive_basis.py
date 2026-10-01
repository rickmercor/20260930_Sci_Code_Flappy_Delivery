"""
Return the overlap matrix and the core Hamiltonian of a minimal basis of one normalised s Gaussian primitive per collinear centre, stacked into a single array.

Every centre of the model carries one normalised s type primitive of the same

exponent, so all one electron integrals have closed forms. With p the sum of

the two exponents and P the Gaussian product centre, the overlap carries the

factor exp(-a R^2 / 2), the kinetic energy is the overlap times a(3 - a R^2)/2,

and each nuclear attraction term brings in the zeroth order Boys function

F0(x) = sqrt(pi / 4x) erf(sqrt x), which tends to one as its argument tends to

zero. The step returns the overlap matrix and the core Hamiltonian, the sum of

kinetic energy and nuclear attraction, stacked into one array so that later

steps receive both in a single object.

Returns
-------
np.ndarray of shape (2, N, N): the overlap matrix stacked on top of the core Hamiltonian, both in the primitive basis
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ao_overlap_and_core(centres: np.ndarray, exponent: float,
                        charges: np.ndarray) -> np.ndarray:
    '''Overlap and core Hamiltonian of the primitive basis.

    Parameters
    ----------
    centres : np.ndarray
        Positions of the nuclei along a line, in bohr. One normalised s type
        primitive sits on each centre. The number of centres must be even.
    exponent : float
        Gaussian exponent shared by every primitive, strictly positive.
    charges : np.ndarray
        Nuclear charge of each centre, one entry per centre.

    Returns
    -------
    stacked : np.ndarray
        Array of shape (2, N, N) whose first slice is the overlap matrix and
        whose second slice is the core Hamiltonian.
    
    Raises
    ------
    ValueError
        If `centres` is not a one dimensional array holding an even number of
        distinct positions, if `exponent` is not a positive scalar, or if
        `charges` does not hold one entry per centre.
'''
    return stacked

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


def _oracle_ao_overlap_and_core(centres, exponent, charges):
    R, a = _check_model(centres, exponent)
    Z = np.asarray(charges, dtype=float)
    if Z.shape != R.shape:
        raise ValueError("charges must have one entry per centre")
    N = R.size
    d2 = (R[:, None] - R[None, :]) ** 2
    p = 2.0 * a
    nrm = (2.0 * a / np.pi) ** 0.75
    E = np.exp(-0.5 * a * d2)
    S = nrm * nrm * (np.pi / p) ** 1.5 * E
    T = 0.5 * a * (3.0 - a * d2) * S
    P = 0.5 * (R[:, None] + R[None, :])
    Vne = np.zeros((N, N))
    for c in range(N):
        Vne -= Z[c] * (2.0 * np.pi / p) * nrm * nrm * E * _boys0(p * (P - R[c]) ** 2)
    return np.stack([S, T + Vne])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
X = np.array([0.0, 3.0, 6.6, 9.8, 13.5, 16.4, 20.2, 23.6])
Z = np.ones(8)
""",
            "call": "ao_overlap_and_core(X, 0.30, Z)",
            "gold_call": "_oracle_ao_overlap_and_core(X, 0.30, Z)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 3.0, 6.6, 9.8, 13.5, 16.4])
Z = np.ones(6)
""",
            "call": "ao_overlap_and_core(X, 0.30, Z)",
            "gold_call": "_oracle_ao_overlap_and_core(X, 0.30, Z)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 2.8, 6.2, 8.9])
Z = np.array([1.0, 1.0, 2.0, 2.0])
""",
            "call": "ao_overlap_and_core(X, 0.85, Z)",
            "gold_call": "_oracle_ao_overlap_and_core(X, 0.85, Z)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 2.9, 6.4, 9.5, 13.1, 15.9, 19.6, 22.3])
Z = np.ones(8)
""",
            "call": "ao_overlap_and_core(X, 0.22, Z)",
            "gold_call": "_oracle_ao_overlap_and_core(X, 0.22, Z)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 12.0, 24.0, 36.0])
Z = np.ones(4)
""",
            "call": "ao_overlap_and_core(X, 0.30, Z)",
            "gold_call": "_oracle_ao_overlap_and_core(X, 0.30, Z)",
        },
        {
            "setup": """import numpy as np
X = np.array([0.0, 2.0, 2.0, 5.0])
Z = np.ones(4)

def run_model():
    try:
        ao_overlap_and_core(X, 0.3, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_ao_overlap_and_core(X, 0.3, Z)
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
X = np.array([0.0, 2.0, 4.0, 7.0])
Z = np.ones(4)

def run_model():
    try:
        ao_overlap_and_core(X, -0.3, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_ao_overlap_and_core(X, -0.3, Z)
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
X = np.array([0.0, 2.0, 4.0])
Z = np.ones(3)

def run_model():
    try:
        ao_overlap_and_core(X, 0.3, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_ao_overlap_and_core(X, 0.3, Z)
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
X = np.array([0.0, 2.0, 4.0, 7.0])
Z = np.ones(3)

def run_model():
    try:
        ao_overlap_and_core(X, 0.3, Z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_ao_overlap_and_core(X, 0.3, Z)
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
