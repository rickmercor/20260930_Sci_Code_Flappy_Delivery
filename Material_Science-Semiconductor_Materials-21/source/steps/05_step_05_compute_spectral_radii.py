"""
Read the convergence rate of each scheme off the spectral radius of its error amplification map.

Both schemes reduce one inner iteration to a linear recursion on the two-component vector of energy-density error amplitudes. The unaccelerated loop states its new amplitudes explicitly, so its amplification map is the matrix of step 03 as it stands. The accelerated loop delivers its new amplitudes only implicitly, through the macroscopic system, so its map is the operator acting on the new amplitudes inverted onto the operator acting on the old ones. After k iterations the error is the k-th power of the map applied to the first one, so its norm decays like the k-th power of the spectral radius: a radius above one diverges, a radius near one converges but stalls, and a small radius kills the error in a handful of iterations. Only the modulus of the dominant eigenvalue matters, since a negative or complex-conjugate dominant eigenvalue flips the sign or rotates the phase of the error each iteration without changing how fast it shrinks. The inverse of the implicit operator should never be formed explicitly; solving the linear system is cheaper and better conditioned, which matters here because that operator degenerates as the wavenumber vanishes.

Returns
-------
np.ndarray of shape (2,), float: the unaccelerated then the accelerated convergence rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_spectral_radii(unaccelerated: np.ndarray,
                           operators: np.ndarray) -> np.ndarray:
    """Return the convergence rate of each scheme as a spectral radius.

    Parameters
    ----------
    unaccelerated : np.ndarray
        Shape (2, 2) as returned by ``build_unaccelerated_matrix``.
    operators : np.ndarray
        Shape (2, 2, 2) as returned by ``build_accelerated_operators``, the
        operator on the new amplitudes first and non-singular.

    Returns
    -------
    rates : np.ndarray
        Shape (2,): the unaccelerated convergence rate followed by the
        accelerated one, each the largest eigenvalue modulus of its map.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return np.zeros(2, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_spectral_radii(unaccelerated: np.ndarray,
                                   operators: np.ndarray) -> np.ndarray:
    import numpy as np

    unaccelerated = np.asarray(unaccelerated, dtype=float)
    operators = np.asarray(operators, dtype=float)
    if unaccelerated.shape != (2, 2) or not np.all(np.isfinite(unaccelerated)):
        raise ValueError("unaccelerated must be a finite array of shape (2, 2)")
    if operators.shape != (2, 2, 2) or not np.all(np.isfinite(operators)):
        raise ValueError("operators must be a finite array of shape (2, 2, 2)")
    left, right = operators[0], operators[1]
    if np.linalg.det(left) == 0.0:
        raise ValueError("the operator on the new amplitudes must be non-singular")

    rho_plain = float(np.max(np.abs(np.linalg.eigvals(unaccelerated))))
    accelerated = np.linalg.solve(left, right)
    return np.array([rho_plain, float(np.max(np.abs(np.linalg.eigvals(accelerated))))],
                    dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: a transition-regime operator triple.
        {"setup": ("import numpy as np\nU = np.array([[0.186, 0.186], [0.188, 0.471]])\n"
                   "O = np.array([[[0.610, 0.402], [-0.098, 0.505]],\n"
                   "              [[0.081, 0.235], [0.037, 0.052]]])\n"),
         "call": "compute_spectral_radii(U, O)",
         "gold_call": "_oracle_compute_spectral_radii(U, O)"},
        # Edge: a complex-conjugate dominant pair, where only the modulus counts.
        {"setup": ("import numpy as np\nU = np.array([[0.2, -0.6], [0.6, 0.2]])\n"
                   "O = np.array([np.eye(2), np.array([[0.1, -0.4], [0.4, 0.1]])])\n"),
         "call": "compute_spectral_radii(U, O)",
         "gold_call": "_oracle_compute_spectral_radii(U, O)"},
        # Edge: a negative dominant eigenvalue; the sign flips, the rate is positive.
        {"setup": ("import numpy as np\nU = np.array([[-0.8, 0.1], [0.05, 0.2]])\n"
                   "O = np.array([[[2.0, 0.5], [-0.5, 1.5]], [[-1.2, 0.3], [0.2, 0.4]]])\n"),
         "call": "compute_spectral_radii(U, O)",
         "gold_call": "_oracle_compute_spectral_radii(U, O)"},
        # Boundary: a nearly degenerate implicit operator, as met near zero wavenumber.
        {"setup": ("import numpy as np\nU = np.array([[0.5, 0.5], [0.25, 0.75]])\n"
                   "O = np.array([[[100.0, -99.999], [-100.0, 100.001]],\n"
                   "              [[0.004, -0.004], [-0.002, 0.002]]])\n"),
         "call": "compute_spectral_radii(U, O)",
         "gold_call": "_oracle_compute_spectral_radii(U, O)"},
    ]
