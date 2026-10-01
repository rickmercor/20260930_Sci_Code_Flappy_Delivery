"""
Compute the Hilbert-Schmidt adjoint of a linear measurement map applied to a given noise vector, and return its operator norm.

For Hermitian sensing operators and real noise coefficients, the adjoint image is Hermitian,

so its spectral norm is the largest absolute eigenvalue.

Returns
-------
float, the weighted adjoint-noise operator norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adjoint_operator_norm(xi: "np.ndarray", A_ops: "np.ndarray") -> float:
    """Compute the operator norm of the adjoint measurement map applied to a noise vector.

    Parameters
    ----------
    xi : np.ndarray
        (m,) real array, the measurement noise/residual associated with each operator.
    A_ops : np.ndarray
        (m, d, d) complex array, m Hermitian measurement operators.

    Returns
    -------
    result : float
        The operator norm of sum_i xi_i * A_ops[i], as a native Python float.

    Raises
    ------
    ValueError
        If xi and A_ops have mismatched lengths.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_adjoint_operator_norm(xi: "np.ndarray", A_ops: "np.ndarray") -> float:
    if xi.shape[0] != A_ops.shape[0]:
        raise ValueError("xi and A_ops must have the same length")
    d = A_ops.shape[1]
    A_star_xi = np.zeros((d, d), dtype=complex)
    for i in range(xi.shape[0]):
        A_star_xi = A_star_xi + xi[i] * A_ops[i]
    eigs = np.linalg.eigvalsh(A_star_xi)
    return float(np.max(np.abs(eigs)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: small d=2, m=3, generic real xi and Hermitian PSD operators ---
        {
            "setup": """
import numpy as np
rng = np.random.default_rng(4)
vecs = (rng.standard_normal((3,2)) + 1j*rng.standard_normal((3,2)))
A_ops = np.array([np.outer(v, v.conj()) for v in vecs])
xi = np.array([0.02, -0.03, 0.01])
""",
            "call": "adjoint_operator_norm(np.array(xi, copy=True), np.array(A_ops, copy=True))",
            "gold_call": "_oracle_adjoint_operator_norm(np.array(xi, copy=True), np.array(A_ops, copy=True))",
        },
        # --- Boundary: xi is all zero, operator norm must be exactly 0 ---
        {
            "setup": """
import numpy as np
rng = np.random.default_rng(6)
vecs = (rng.standard_normal((3,2)) + 1j*rng.standard_normal((3,2)))
A_ops = np.array([np.outer(v, v.conj()) for v in vecs])
xi = np.zeros(3)
""",
            "call": "adjoint_operator_norm(np.array(xi, copy=True), np.array(A_ops, copy=True))",
            "gold_call": "_oracle_adjoint_operator_norm(np.array(xi, copy=True), np.array(A_ops, copy=True))",
        },
        # --- Edge: an unweighted-basis comparison instance's xi and A_ops (orthonormal Hermitian operator basis) ---
        {
            "setup": """
import numpy as np
rng0 = np.random.default_rng(31)
eigvals = np.array([0.6, 0.4, 0.0, 0.0])
Araw = rng0.standard_normal((4,4)) + 1j*rng0.standard_normal((4,4))
Q, _ = np.linalg.qr(Araw)
rho_star = (Q * eigvals) @ Q.conj().T
rho_star = (rho_star + rho_star.conj().T)/2

def _basis(d):
    b = []
    for i in range(d):
        E = np.zeros((d,d), dtype=complex); E[i,i]=1.0
        b.append(E)
    for i in range(d):
        for j in range(i+1,d):
            E1 = np.zeros((d,d), dtype=complex); E1[i,j]=1/np.sqrt(2); E1[j,i]=1/np.sqrt(2)
            b.append(E1)
            E2 = np.zeros((d,d), dtype=complex); E2[i,j]=1j/np.sqrt(2); E2[j,i]=-1j/np.sqrt(2)
            b.append(E2)
    return np.array(b)

A_ops = _basis(4)
rng_noise = np.random.default_rng(2026)
xi = rng_noise.normal(scale=0.01, size=16)
""",
            "call": "adjoint_operator_norm(np.array(xi, copy=True), np.array(A_ops, copy=True))",
            "gold_call": "_oracle_adjoint_operator_norm(np.array(xi, copy=True), np.array(A_ops, copy=True))",
        },
    ]
