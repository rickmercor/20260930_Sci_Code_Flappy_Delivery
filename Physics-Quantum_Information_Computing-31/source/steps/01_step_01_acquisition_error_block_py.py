"""
Compute the part of a bounded-joint-measurement tomography estimator's averaging error that lies strictly outside the true state's support, for a fixed sequence of round outcomes.

The source estimator expresses this support-orthogonal block through a centered complex

Wishart term in coordinates adapted to the target state's support.

Returns
-------
(d, d) complex array, the embedded out-of-support acquisition error.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pi_perp_error_block(Z: "np.ndarray", K: int, N: int, V_perp: "np.ndarray") -> "np.ndarray":
    """Embed the exact out-of-support averaging error of a bounded-joint-measurement
    tomography estimator into the full Hilbert space.

    Parameters
    ----------
    Z : np.ndarray
        (m, K) complex matrix, standard complex Gaussian in the coordinates of an
        orthonormal basis of the orthogonal complement of the true state's support
        (m is the dimension of that complement).
    K : int
        Sum of the per-round outcome values across all measurement rounds.
    N : int
        Total number of copies measured (rounds times copies per round).
    V_perp : np.ndarray
        (d, m) complex matrix whose columns are an orthonormal basis of the
        orthogonal complement of the true state's support, in the full d-dimensional space.

    Returns
    -------
    result : np.ndarray
        (d, d) complex Hermitian matrix, the error block embedded in the full space.

    Raises
    ------
    ValueError
        If Z's row count does not match V_perp's column count, or N <= 0.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pi_perp_error_block(Z: "np.ndarray", K: int, N: int, V_perp: "np.ndarray") -> "np.ndarray":
    m = V_perp.shape[1]
    if Z.shape[0] != m:
        raise ValueError("Z's row count must match V_perp's column count")
    if N <= 0:
        raise ValueError("N must be positive")
    E_coords = (Z @ Z.conj().T - K * np.eye(m)) / N
    return V_perp @ E_coords @ V_perp.conj().T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: small m=2, K=5, generic Z and V_perp (orthonormal columns of a random unitary) ---
        {
            "setup": """
import numpy as np
Z = np.array([[0.3+0.1j, -0.2+0.4j, 0.1-0.1j, 0.5+0.0j, -0.3+0.2j],
              [0.1-0.2j, 0.4+0.1j, -0.2+0.3j, 0.0-0.4j, 0.2+0.1j]])
K = 5
N = 10
rng = np.random.default_rng(7)
Araw = rng.standard_normal((4,4)) + 1j*rng.standard_normal((4,4))
Qm, _ = np.linalg.qr(Araw)
V_perp = Qm[:, :2]
""",
            "call": "pi_perp_error_block(np.array(Z, copy=True), K, N, np.array(V_perp, copy=True))",
            "gold_call": "_oracle_pi_perp_error_block(np.array(Z, copy=True), K, N, np.array(V_perp, copy=True))",
        },
        # --- Boundary: Z is all zero, isolating the -K*I/N term exactly ---
        {
            "setup": """
import numpy as np
Z = np.zeros((2,4), dtype=complex)
K = 4
N = 8
rng = np.random.default_rng(11)
Araw = rng.standard_normal((4,4)) + 1j*rng.standard_normal((4,4))
Qm, _ = np.linalg.qr(Araw)
V_perp = Qm[:, :2]
""",
            "call": "pi_perp_error_block(np.array(Z, copy=True), K, N, np.array(V_perp, copy=True))",
            "gold_call": "_oracle_pi_perp_error_block(np.array(Z, copy=True), K, N, np.array(V_perp, copy=True))",
        },
        # --- Edge: the actual task instance (d=4, r=2, m=2, K=6, N=12) ---
        {
            "setup": """
import numpy as np
rng0 = np.random.default_rng(31)
eigvals = np.array([0.6, 0.4, 0.0, 0.0])
Araw = rng0.standard_normal((4,4)) + 1j*rng0.standard_normal((4,4))
Q, _ = np.linalg.qr(Araw)
V_perp = Q[:, 2:]
rng1 = np.random.default_rng(4100)
Z = (rng1.standard_normal((2,6)) + 1j*rng1.standard_normal((2,6)))/np.sqrt(2)
K = 6
N = 12
""",
            "call": "pi_perp_error_block(np.array(Z, copy=True), K, N, np.array(V_perp, copy=True))",
            "gold_call": "_oracle_pi_perp_error_block(np.array(Z, copy=True), K, N, np.array(V_perp, copy=True))",
        },
    ]
