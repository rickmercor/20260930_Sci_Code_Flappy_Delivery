"""
Build the imaginary-time fluctuation propagator of one coordinate about a discretized orbit, given the curvature of the potential in that coordinate at each bead.

The propagator is the inverse of the Hessian of the discretized imaginary-time action of that coordinate, for a closed chain of N beads in the standard primitive discretization with time step beta hbar / N. It is the object whose entries give the second moments of the fluctuations, and it exists only when that Hessian is positive definite, that is when the coordinate is stable around the whole chain.

Returns
-------
np.ndarray of shape (N, N): the symmetric inverse action Hessian of one fluctuation coordinate on the closed chain (bohr^2 per unit of action)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ring_polymer_propagator(curvatures: "np.ndarray", beta: float, m: float) -> "np.ndarray":
    '''Inverse action Hessian of one fluctuation coordinate on a closed chain.

    Parameters
    ----------
    curvatures : np.ndarray
        One-dimensional array of length N >= 3 with the second derivative of
        the potential with respect to the fluctuation coordinate, in
        hartree / bohr^2, at each of the N beads, in chain order.
    beta : float
        Inverse temperature in 1/hartree (hbar = 1); the imaginary-time step
        is beta / N.
    m : float
        Mass of the coordinate in electron masses.

    Returns
    -------
    G : np.ndarray
        Symmetric (N, N) array, the inverse of the action Hessian, in
        bohr^2 per unit of action. Entry [i, j] refers to beads i and j of
        the same chain order as `curvatures`.

    Raises
    ------
    ValueError
        If curvatures is not one-dimensional with N >= 3, beta or m is not
        positive, or the action Hessian is not positive definite.
    '''
    return G

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ring_polymer_propagator(curvatures: "np.ndarray", beta: float, m: float) -> "np.ndarray":
    c = np.asarray(curvatures, dtype=float)
    if c.ndim != 1 or c.size < 3:
        raise ValueError("curvatures must be a one-dimensional array with N >= 3")
    beta, m = float(beta), float(m)
    if not (beta > 0.0 and m > 0.0):
        raise ValueError("beta and m must be positive")
    dtau = beta / c.size
    jac = (m / dtau) * _cyclic_laplacian(c.size) + dtau * np.diag(c)
    try:
        chol = np.linalg.cholesky(jac)
    except np.linalg.LinAlgError:
        raise ValueError("action Hessian is not positive definite")
    inv_chol = np.linalg.solve(chol, np.eye(c.size))
    g = inv_chol.T @ inv_chol
    return 0.5 * (g + g.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: curvature modulated along a 32-bead chain.
        {
            "setup": """import numpy as np
m, we, beta = 1836.15267, 0.00415, 2975.0
n = 32
curv = m * we ** 2 * (1.0 + 0.3 * np.cos(2.0 * np.pi * np.arange(n) / n))
""",
            "call": "ring_polymer_propagator(curv.copy(), beta, m)",
            "gold_call": "_oracle_ring_polymer_propagator(curv.copy(), beta, m)",
        },
        # Normal: uniform curvature, 16 beads.
        {
            "setup": """import numpy as np
m, we, beta = 1836.15267, 0.00415, 2975.0
curv = np.full(16, m * we ** 2)
""",
            "call": "ring_polymer_propagator(curv.copy(), beta, m)",
            "gold_call": "_oracle_ring_polymer_propagator(curv.copy(), beta, m)",
        },
        # Boundary: smallest closed chain (N = 3) in reduced units.
        {
            "setup": """import numpy as np
curv = np.array([1.0, 0.7, 1.3])
""",
            "call": "ring_polymer_propagator(curv.copy(), 4.0 * np.pi, 1.0)",
            "gold_call": "_oracle_ring_polymer_propagator(curv.copy(), 4.0 * np.pi, 1.0)",
        },
        # Edge: nearly free coordinate, where the propagator is dominated by the chain centroid.
        {
            "setup": """import numpy as np
curv = np.full(12, 1e-6)
""",
            "call": "ring_polymer_propagator(curv.copy(), 2975.0, 1836.15267)",
            "gold_call": "_oracle_ring_polymer_propagator(curv.copy(), 2975.0, 1836.15267)",
        },
        # Invalid: a strongly negative curvature makes the action Hessian indefinite.
        {
            "setup": """import numpy as np
curv = np.array([0.01, -0.5, 0.01, 0.01])
def run_model():
    try:
        ring_polymer_propagator(curv.copy(), 2975.0, 1836.15267)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_ring_polymer_propagator(curv.copy(), 2975.0, 1836.15267)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
