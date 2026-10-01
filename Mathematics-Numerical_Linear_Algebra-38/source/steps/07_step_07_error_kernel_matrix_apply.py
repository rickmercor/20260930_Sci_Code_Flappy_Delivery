"""
Construct the matrix-valued restart correction on the reduced problem of the next cycle.

The scalar restart quantity from the previous cycle must be transferred to the reduced matrix generated during the next cycle. This produces the small matrix correction used to recover the contribution of the restarted approximation.

Returns
-------
float, Frobenius norm of err(H) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def error_kernel_matrix_apply(
    state: dict,
    theta_prev: np.ndarray,
    gamma_prev: float,
    beta_prev: float,
    tol: float,
    maxnodes: int,
    r1: int,
) -> float:
    """Evaluate the restart kernel on the current reduced matrix.

    Parameters
    ----------
    state : dict
        Mutable pipeline state containing the current corrected matrix.
    theta_prev : np.ndarray
        Previous-cycle reduced spectral values.
    gamma_prev : float
        Previous-cycle scalar factor.
    beta_prev : float
        Previous-cycle starting-vector scaling.
    tol : float
        Quadrature tolerance.
    maxnodes : int
        Maximum quadrature nodes.
    r1 : int
        Initial quadrature nodes.

    Returns
    -------
    float
        Frobenius norm of the matrix-valued correction.
    """
    result = 0.0
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _scalar_kernel_internal(
    z: complex,
    theta: np.ndarray,
    gamma: float,
    beta: float,
    tol: float,
    maxnodes: int,
    r1: int,
) -> complex:
    theta = np.asarray(theta, dtype=complex)

    def phi(v):
        out = np.ones(v.shape, dtype=complex)
        for th in theta:
            out *= v - th
        return out

    def estimate(nodes):
        x, w = np.polynomial.legendre.leggauss(nodes)

        q = 0.5 * (x + 1.0)
        weights = 0.5 * w

        v = q / (1.0 - q)
        jac = 1.0 / (1.0 - q) ** 2

        integrand = 1.0 / (phi(-v**2) * (v**2 + z))
        return (2.0 * gamma * beta / np.pi) * np.sum(
            weights * jac * integrand
        )

    nodes = int(r1)
    previous = estimate(nodes)
    nodes *= 2

    while nodes <= maxnodes:
        current = estimate(nodes)

        if abs(current - previous) <= tol:
            return current

        previous = current
        nodes *= 2

    return previous


def _oracle_error_kernel_matrix_apply(
    state: dict,
    theta_prev: np.ndarray,
    gamma_prev: float,
    beta_prev: float,
    tol: float,
    maxnodes: int,
    r1: int,
) -> float:
    import numpy as np

    if not isinstance(state, dict):
        raise ValueError("state must be a dict")

    if "second_corrected_Hm" not in state:
        raise ValueError("second corrected reduced matrix is missing")

    Hm = np.asarray(state["second_corrected_Hm"])

    if Hm.ndim != 2 or Hm.shape[0] != Hm.shape[1]:
        raise ValueError("current reduced matrix must be square")

    eigvals, V = np.linalg.eig(Hm)

    if np.linalg.cond(V) > 1e14:
        raise ValueError("current reduced matrix is numerically diagonalization-singular")

    Vinv = np.linalg.inv(V)

    values = np.array(
        [
            _scalar_kernel_internal(
                z,
                np.asarray(theta_prev, dtype=complex),
                gamma_prev,
                beta_prev,
                tol,
                maxnodes,
                r1,
            )
            for z in eigvals
        ],
        dtype=complex,
    )

    errHm = V @ np.diag(values) @ Vinv

    state["errHm"] = np.real_if_close(errHm)

    return float(np.linalg.norm(errHm, ord="fro"))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
state = {
    "second_corrected_Hm": np.diag([3.0, 4.0]),
}
theta_prev = np.array([5.0])
gamma_prev = 0.8
beta_prev = 1.0
tol = 1e-6
maxnodes = 64
r1 = 4
""",
            "call": "error_kernel_matrix_apply(state, theta_prev, gamma_prev, beta_prev, tol, maxnodes, r1)",
            "gold_call": "_oracle_error_kernel_matrix_apply(state, theta_prev, gamma_prev, beta_prev, tol, maxnodes, r1)",
        },
        {
            "setup": """import numpy as np
state = {
    "second_corrected_Hm": np.array([[2.0]]),
}
theta_prev = np.array([1.0, 6.0])
gamma_prev = 1.0
beta_prev = 1.0
tol = 1e-6
maxnodes = 64
r1 = 4
""",
            "call": "error_kernel_matrix_apply(state, theta_prev, gamma_prev, beta_prev, tol, maxnodes, r1)",
            "gold_call": "_oracle_error_kernel_matrix_apply(state, theta_prev, gamma_prev, beta_prev, tol, maxnodes, r1)",
        },
        {
            "setup": """import numpy as np
state = {
    "second_corrected_Hm": np.array([[2.0, -1.0], [1.0, 2.0]]),
}
theta_prev = np.array([4.0])
gamma_prev = 0.5
beta_prev = 1.5
tol = 1e-6
maxnodes = 64
r1 = 4
""",
            "call": "error_kernel_matrix_apply(state, theta_prev, gamma_prev, beta_prev, tol, maxnodes, r1)",
            "gold_call": "_oracle_error_kernel_matrix_apply(state, theta_prev, gamma_prev, beta_prev, tol, maxnodes, r1)",
        },
    ]
