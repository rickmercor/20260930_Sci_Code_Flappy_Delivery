"""
Builds the two-qubit fractional gate fSim(θ, φ) = CP(−φ) R_XX(θ) R_YY(θ) with the standard half-angle generators.

Continuously parameterized two-qubit gates interpolate between familiar Clifford points such as CZ and iSWAP. The excitation-preserving family used here is the product of a controlled phase and simultaneous XX and YY rotations.

Returns
-------
A complex ndarray of shape (4, 4).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def fsim_unitary(theta: float, phi: float) -> np.ndarray:
    """Return the 4×4 unitary fSim(theta, phi).

    Use R_XX(θ) = exp(−i θ XX / 2), R_YY(θ) = exp(−i θ YY / 2), and
    CP(φ) = diag(1, 1, 1, exp(i φ)). Then fSim(0, π) is exactly CZ.

    Parameters
    ----------
    theta : float
        Exchange angle, finite.
    phi : float
        Conditional phase, finite.

    Returns
    -------
    unitary : np.ndarray
        Complex array of shape (4, 4).

    Raises
    ------
    ValueError
        If theta or phi is not a finite number.
    """
    return np.eye(4, dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fsim_unitary(theta: float, phi: float) -> np.ndarray:
    import numpy as np

    theta = float(theta)
    phi = float(phi)
    if theta != theta or phi != phi or abs(theta) == float("inf") or abs(phi) == float("inf"):
        raise ValueError("theta and phi must be finite")
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    XX = np.kron(X, X)
    YY = np.kron(Y, Y)
    eye = np.eye(4, dtype=complex)
    rxx = np.cos(theta / 2.0) * eye - 1.0j * np.sin(theta / 2.0) * XX
    ryy = np.cos(theta / 2.0) * eye - 1.0j * np.sin(theta / 2.0) * YY
    cp = np.diag([1.0, 1.0, 1.0, np.exp(-1.0j * phi)]).astype(complex)
    return cp @ rxx @ ryy

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
theta, phi = 0.0, np.pi
""",
            "call": "fsim_unitary(theta, phi)",
            "gold_call": "_oracle_fsim_unitary(theta, phi)",
        },
        {
            "setup": """import numpy as np
theta, phi = 0.0, 0.0
""",
            "call": "fsim_unitary(theta, phi)",
            "gold_call": "_oracle_fsim_unitary(theta, phi)",
        },
        {
            "setup": """import numpy as np
theta, phi = np.pi / 2.0, 0.0
""",
            "call": "fsim_unitary(theta, phi)",
            "gold_call": "_oracle_fsim_unitary(theta, phi)",
        },
        {
            "setup": """
def run_model():
    try:
        fsim_unitary(float('nan'), 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_fsim_unitary(float('nan'), 0.0)
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
