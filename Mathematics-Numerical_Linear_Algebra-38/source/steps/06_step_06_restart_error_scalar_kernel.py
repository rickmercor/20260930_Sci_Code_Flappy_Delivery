"""
Evaluate one scalar value of the restart correction kernel associated with the previous reduced problem.

Restarting a matrix-function approximation requires estimating the contribution that remains after a finite Krylov computation. For functions admitting suitable integral representations, this contribution can be expressed through a scalar kernel that depends on the reduced problem from the preceding cycle.

Returns
-------
float, real part of err(z) as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def restart_error_scalar_kernel(
    theta: np.ndarray,
    gamma: float,
    beta: float,
    z: complex,
    tol: float,
    maxnodes: int,
    r1: int,
) -> float:
    """Evaluate one scalar value of the restart error kernel.

    Parameters
    ----------
    theta : np.ndarray
        Reduced spectral values from the previous cycle.
    gamma : float
        Scalar factor associated with the previous cycle.
    beta : float
        Starting-vector scaling.
    z : complex
        Evaluation point.
    tol : float
        Quadrature convergence tolerance.
    maxnodes : int
        Maximum quadrature nodes.
    r1 : int
        Initial quadrature nodes.

    Returns
    -------
    float
        Real part of the kernel value.
    """
    result = 0.0
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _phi(values: np.ndarray, theta: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=complex)
    out = np.ones(values.shape, dtype=complex)

    for th in theta:
        out *= values - th

    return out


def _quadrature_value(
    z: complex,
    theta: np.ndarray,
    gamma: float,
    beta: float,
    nodes: int,
) -> complex:
    x, w = np.polynomial.legendre.leggauss(nodes)

    q = 0.5 * (x + 1.0)
    weights = 0.5 * w

    v = q / (1.0 - q)
    jac = 1.0 / (1.0 - q) ** 2

    denominator = _phi(-v**2, theta) * (v**2 + z)

    integral = np.sum(weights * jac / denominator)

    return (2.0 * gamma * beta / np.pi) * integral


def _oracle_restart_error_scalar_kernel(
    theta: np.ndarray,
    gamma: float,
    beta: float,
    z: complex,
    tol: float,
    maxnodes: int,
    r1: int,
) -> float:
    import numpy as np

    theta = np.asarray(theta, dtype=complex)

    if theta.ndim != 1 or theta.size == 0:
        raise ValueError("theta must be a nonempty one-dimensional array")
    if r1 < 2 or maxnodes < r1:
        raise ValueError("invalid quadrature sizes")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if not np.isfinite(gamma) or not np.isfinite(beta):
        raise ValueError("gamma and beta must be finite")

    nodes = int(r1)

    previous = _quadrature_value(
        complex(z), theta, float(gamma), float(beta), nodes
    )

    nodes *= 2

    while nodes <= maxnodes:
        current = _quadrature_value(
            complex(z), theta, float(gamma), float(beta), nodes
        )

        if abs(current - previous) <= tol:
            return float(np.real(current))

        previous = current
        nodes *= 2

    return float(np.real(previous))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
theta = np.array([5.0])
gamma = 0.8
beta = 1.0
z = 3.0 + 0.0j
tol = 1e-6
maxnodes = 64
r1 = 4
""",
            "call": "restart_error_scalar_kernel(theta, gamma, beta, z, tol, maxnodes, r1)",
            "gold_call": "_oracle_restart_error_scalar_kernel(theta, gamma, beta, z, tol, maxnodes, r1)",
        },
        {
            "setup": """import numpy as np
theta = np.array([2.0, 3.0])
gamma = 1.0
beta = 1.0
z = 1.0 + 0.0j
tol = 1e-10
maxnodes = 64
r1 = 4
""",
            "call": "restart_error_scalar_kernel(theta, gamma, beta, z, tol, maxnodes, r1)",
            "gold_call": "_oracle_restart_error_scalar_kernel(theta, gamma, beta, z, tol, maxnodes, r1)",
        },
        {
            "setup": """import numpy as np
theta = np.array([1.0 + 1.0j, 1.0 - 1.0j])
gamma = 0.5
beta = 2.0
z = 2.0 + 0.5j
tol = 1e-6
maxnodes = 64
r1 = 4
""",
            "call": "restart_error_scalar_kernel(theta, gamma, beta, z, tol, maxnodes, r1)",
            "gold_call": "_oracle_restart_error_scalar_kernel(theta, gamma, beta, z, tol, maxnodes, r1)",
        },
    ]
