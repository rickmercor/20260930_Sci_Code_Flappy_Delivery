"""
Step 02: Response to a residual-lead mode over one step.

Response of the dot and reaction coordinates to one residual-lead mode during a step of constant coupling.

While the switching factors are constant, the drift matrix M of the reaction-coordinate step is constant. A residual-lead
mode of energy omega enters the Heisenberg equations of (d, r_1, r_2) as a source oscillating as exp(-i omega s). The
response accumulated over a step that starts at s = 0 and lasts tau is

  R(omega) = integral from 0 to tau of exp[M (tau - s)] exp(-i omega s) ds,

a 3 x 3 matrix for every omega; column b is the response to a source acting on component b. This step returns R for a
whole array of mode energies. It must be exact up to rounding for any tau >= 0, including omega close to an eigenvalue
frequency of M.

Inputs: drift, a complex 3 x 3 array whose eigenvalues all have strictly negative real part; tau >= 0; omega, a
one-dimensional array of real mode energies. Output: complex ndarray of shape (len(omega), 3, 3). A drift of the wrong
shape or with an eigenvalue of non-negative real part, a negative or non-finite tau, or an omega array that is not
one-dimensional and finite raises ValueError.

Returns
-------
numpy.ndarray of shape (len(omega), 3, 3), complex step responses R(omega) over a step of constant drift
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.linalg import expm


def segment_response(drift: np.ndarray, tau: float, omega: np.ndarray) -> np.ndarray:
    '''Accumulated response matrices R(omega) over a step of constant drift and duration tau.

    Parameters
    ----------
    drift : np.ndarray
        Complex array of shape (3, 3); every eigenvalue has negative real part.
    tau : float
        Step duration, finite and non-negative.
    omega : np.ndarray
        One-dimensional array of real mode energies.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (len(omega), 3, 3) with R(omega) = int_0^tau exp[M (tau - s)] exp(-i omega s) ds.

    Raises
    ------
    ValueError
        If drift has the wrong shape or an eigenvalue with non-negative real part, tau is negative or not finite, or
        omega is not a finite one-dimensional array.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_segment_response(drift: np.ndarray, tau: float, omega: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    import numpy as np
    from scipy.linalg import expm
    try:
        m = np.asarray(drift, dtype=complex)
        om = np.asarray(omega, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("drift and omega must be numeric arrays")
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError("drift must be a finite 3 x 3 array")
    if np.max(np.linalg.eigvals(m).real) >= 0.0:
        raise ValueError("every eigenvalue of the drift must have negative real part")
    tau = _real_float(tau, "tau")
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    if om.ndim != 1 or not np.all(np.isfinite(om)):
        raise ValueError("omega must be a finite one-dimensional array")
    eye = np.eye(3)
    em = expm(m * tau)
    lhs = m[None, :, :] + 1j * om[:, None, None] * eye[None, :, :]
    rhs = em[None, :, :] - np.exp(-1j * om * tau)[:, None, None] * eye[None, :, :]
    return np.linalg.solve(lhs, rhs)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: step-1 drift of the benchmark device, a spread of mode energies ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "k1 = 1.0 * np.sqrt(p[3] * p[2] / 2.0)\n"
                     "k2 = 0.0 * np.sqrt(p[6] * p[5] / 2.0)\n"
                     "m = -np.diag([1j * p[0], p[2] + 1j * p[1], p[5] + 1j * p[4]]).astype(complex)\n"
                     "m[0, 1] = m[1, 0] = -1j * k1\n"
                     "m[0, 2] = m[2, 0] = -1j * k2\n"
                     "w = np.array([-30.0, -3.0, -0.5, 0.0, 0.37, 1.2])\n"
                     "m_g = m.copy()\n"
                     "w_g = w.copy()\n",
            "call": "segment_response(m, 1.128, w)",
            "gold_call": "_oracle_segment_response(m_g, 1.128, w_g)",
            "tol": 1e-10,
        },
        # --- Boundary: long step, where the free part has decayed, mode energies at the hybrid-level frequencies ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([1.2, 0.0, 0.1, 2.5, 1.2, 0.25, 0.1])\n"
                     "k1 = 0.0 * np.sqrt(p[3] * p[2] / 2.0)\n"
                     "k2 = 1.0 * np.sqrt(p[6] * p[5] / 2.0)\n"
                     "m = -np.diag([1j * p[0], p[2] + 1j * p[1], p[5] + 1j * p[4]]).astype(complex)\n"
                     "m[0, 1] = m[1, 0] = -1j * k1\n"
                     "m[0, 2] = m[2, 0] = -1j * k2\n"
                     "w = np.linalg.eigvals(1j * m).real\n"
                     "m_g = m.copy()\n"
                     "w_g = w.copy()\n",
            "call": "segment_response(m, 40.0, w)",
            "gold_call": "_oracle_segment_response(m_g, 40.0, w_g)",
            "tol": 1e-10,
        },
        # --- Edge: zero duration gives zero response ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([-0.4, -1.5, 0.8, 0.6, 2.0, 0.05, 3.0])\n"
                     "k1 = 0.3 * np.sqrt(p[3] * p[2] / 2.0)\n"
                     "k2 = 0.7 * np.sqrt(p[6] * p[5] / 2.0)\n"
                     "m = -np.diag([1j * p[0], p[2] + 1j * p[1], p[5] + 1j * p[4]]).astype(complex)\n"
                     "m[0, 1] = m[1, 0] = -1j * k1\n"
                     "m[0, 2] = m[2, 0] = -1j * k2\n"
                     "w = np.array([-1.0, 0.5])\n"
                     "m_g = m.copy()\n"
                     "w_g = w.copy()\n",
            "call": "segment_response(m, 0.0, w)",
            "gold_call": "_oracle_segment_response(m_g, 0.0, w_g)",
            "tol": 1e-12,
        },
        # --- Normal: both leads partly coupled, fine energy grid across a resonance ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "p = np.array([-0.4, -1.5, 0.8, 0.6, 2.0, 0.05, 3.0])\n"
                     "k1 = 0.3 * np.sqrt(p[3] * p[2] / 2.0)\n"
                     "k2 = 0.7 * np.sqrt(p[6] * p[5] / 2.0)\n"
                     "m = -np.diag([1j * p[0], p[2] + 1j * p[1], p[5] + 1j * p[4]]).astype(complex)\n"
                     "m[0, 1] = m[1, 0] = -1j * k1\n"
                     "m[0, 2] = m[2, 0] = -1j * k2\n"
                     "w = np.linspace(1.8, 2.2, 9)\n"
                     "m_g = m.copy()\n"
                     "w_g = w.copy()\n",
            "call": "segment_response(m, 3.3, w)",
            "gold_call": "_oracle_segment_response(m_g, 3.3, w_g)",
            "tol": 1e-10,
        },
        # --- Error: a drift with a purely oscillating eigenvalue must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.linalg import expm\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.diag([-1j, -0.1 - 0.2j, -0.3 + 0.0j]), 1.0, np.array([0.0]))\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(segment_response)",
            "gold_call": "_probe(_oracle_segment_response)",
        },
    ]
