"""
Apply the wider amplifier Z to the rangefinder Y for exactly q >= 1 steps. Each step takes a thin QR of Z^T Yhat and replaces Yhat by Z X. Reject q = 0: an unamplified rangefinder is a different method.

The Gram operator of the wider sketch stands in for A A^T on the rangefinder. Re-orthonormalizing the small core at each step keeps the same column space as (Z Z^T)^q Y while avoiding a large intermediate product. One or two steps are the usual budget; zero steps is the plain one-pass rangefinder.

Returns
-------
ndarray of shape (m, s_width): the amplified rangefinder Yhat
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def amplify_rangefinder(Y: np.ndarray, Z: np.ndarray, q: int) -> np.ndarray:
    """Apply q re-orthonormalized amplifier steps to Y using Z.

    Raises
    ------
    ValueError
        If `Y` or `Z` is not 2D, if `Y` and `Z` do not have the same
        number of rows, if `require l >= s_width >= 1` does not hold,
        if `q` is not an integer, if `q < 1`, or if the amplifier core
        is rank deficient.
    """
    return np.zeros_like(Y)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_amplify_rangefinder(Y, Z, q):
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if Y.ndim != 2 or Z.ndim != 2:
        raise ValueError("Y and Z must be 2D")
    if Y.shape[0] != Z.shape[0]:
        raise ValueError("Y and Z must have the same number of rows")
    if Y.shape[1] < 1 or Z.shape[1] < Y.shape[1]:
        raise ValueError("require l >= s_width >= 1")
    if not isinstance(q, (int, np.integer)):
        raise ValueError("q must be an integer")
    q = int(q)
    if q < 1:
        raise ValueError("require q >= 1")
    Yhat = np.array(Y, dtype=float, copy=True)
    for _ in range(q):
        core = Z.T @ Yhat
        X, R = np.linalg.qr(core, mode="reduced")
        if np.any(np.abs(np.diag(R)) < 1e-14):
            raise ValueError("amplifier core is rank deficient")
        Yhat = Z @ X
    return Yhat

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
m, n, s_width, l = 12, 10, 4, 8
rng = np.random.default_rng(7)
U, _ = np.linalg.qr(rng.standard_normal((m, n)), mode='reduced')
V, _ = np.linalg.qr(rng.standard_normal((n, n)), mode='reduced')
s = np.array([4.0, 3.2, 2.6, 2.1, 1.7, 1.4, 1.15, 0.95, 0.8, 0.7])
A = U @ np.diag(s) @ V.T
rng = np.random.default_rng(11)
Omega = rng.standard_normal((n, s_width))
_ = rng.standard_normal((7, m))
Phi = rng.standard_normal((n, l))
Y = A @ Omega
Z = A @ Phi
q = 1
""",
            "call": "amplify_rangefinder(Y, Z, q)",
            "gold_call": "_oracle_amplify_rangefinder(Y, Z, q)",
        },
        {
            "setup": """import numpy as np
Z = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=float)
Y = np.array([[1.0], [0.0], [0.0]], dtype=float)
q = 1
""",
            "call": "amplify_rangefinder(Y, Z, q)",
            "gold_call": "_oracle_amplify_rangefinder(Y, Z, q)",
        },
        {
            "setup": """import numpy as np
Z = np.eye(3)
Y = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)
q = 2
""",
            "call": "amplify_rangefinder(Y, Z, q)",
            "gold_call": "_oracle_amplify_rangefinder(Y, Z, q)",
        },
        {
            "setup": """import numpy as np
Y = np.ones((3, 1))
Z = np.ones((3, 2))
def run_model():
    try:
        amplify_rangefinder(Y, Z, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_amplify_rangefinder(Y, Z, 0)
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
Y = np.ones((3, 2))
Z = np.ones((3, 1))
def run_model():
    try:
        amplify_rangefinder(Y, Z, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_amplify_rangefinder(Y, Z, 1)
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
