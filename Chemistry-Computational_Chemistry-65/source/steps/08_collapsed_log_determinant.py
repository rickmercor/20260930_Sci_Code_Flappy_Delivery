"""
Logarithm of the determinant of the second-derivative matrix of the discretised action for a closed chain collapsed onto a harmonic minimum. SPECIFICATION: the chain has n_ring rows, all sitting at one minimum whose normal-mode frequencies are omegas, with spacing eps of the evolution parameter between consecutive rows and unit mass, and consecutive rows are coupled cyclically so that the chain is closed. The matrix is scaled as eps over the mass times the second derivative of the discretised action, so that each diagonal block is twice the identity plus eps squared times the Hessian at the minimum and each cyclic off-diagonal block is minus the identity. Return the natural logarithm of its determinant as a single real number. n_ring is supplied by the caller.

A chain collapsed onto a single point is the reference against which the fluctuations around the tunnelling trajectory are normalised: it supplies the statistical weight of a well on its own, with no crossing. Its second-derivative matrix is highly regular, so the determinant can be evaluated without any factorisation. Keeping the result as a logarithm matters because the determinant itself overflows at the chain lengths used here.

Returns
-------
float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def collapsed_log_determinant(n_ring: int, eps: float, omegas: "np.ndarray") -> float:
    """Logarithm of the determinant of the second-derivative matrix of the discretised action for a closed chain collapsed onto a harmonic minimum. SPECIFICATION: the chain has n_ring rows, all sitting at one minimum whose normal-mode frequencies are omegas, with spacing eps of the evolution parameter between consecutive rows and unit mass, and consecutive rows are coupled cyclically so that the chain is closed. The matrix is scaled as eps over the mass times the second derivative of the discretised action, so that each diagonal block is twice the identity plus eps squared times the Hessian at the minimum and each cyclic off-diagonal block is minus the identity. Return the natural logarithm of its determinant as a single real number. n_ring is supplied by the caller.

    Returns
    -------
    float.

    Raises
    ------
    ValueError: if n_ring is not a positive integer, if eps is not positive and finite, or if omegas is empty or holds a value that is not finite.
    """
    return log_det  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq


def _oracle_collapsed_log_determinant(n_ring: int, eps: float, omegas: "np.ndarray") -> float:
    n_ring = int(n_ring)
    if n_ring < 1:
        raise ValueError("n_ring must be a positive integer")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    om = np.atleast_1d(np.asarray(omegas, dtype=float))
    if om.size == 0 or not np.all(np.isfinite(om)):
        raise ValueError("omegas must be a non-empty array of finite frequencies")
    k = np.arange(n_ring)
    base = 4.0 * np.sin(np.pi * k / float(n_ring)) ** 2
    return float(sum(np.sum(np.log(base + (eps * w) ** 2)) for w in om))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\nom = np.array([0.4, 2.0])',
         "call": 'collapsed_log_determinant(38, 120.0 / 64, om.copy())',
         "gold_call": '_oracle_collapsed_log_determinant(38, 120.0 / 64, om.copy())'},   # normal: a left-well arc with two frequencies
        {"setup": 'import numpy as np\nom = np.array([0.55, 1.9])',
         "call": 'collapsed_log_determinant(90, 120.0 / 64, om.copy())',
         "gold_call": '_oracle_collapsed_log_determinant(90, 120.0 / 64, om.copy())'},   # edge: a longer right-well arc, same spacing
        {"setup": 'import numpy as np\nom = np.array([1.0])',
         "call": 'collapsed_log_determinant(7, 0.5, om.copy())',
         "gold_call": '_oracle_collapsed_log_determinant(7, 0.5, om.copy())'},   # edge: short chain, single frequency
        {"setup": 'import numpy as np\nom = np.array([1.0])',
         "call": 'collapsed_log_determinant(1, 0.5, om.copy())',
         "gold_call": '_oracle_collapsed_log_determinant(1, 0.5, om.copy())'},   # boundary: a one-row chain
        {"setup": 'import numpy as np\nom = np.array([0.4, 2.0])\ndef _c():\n    try:\n        collapsed_log_determinant(0, 120.0 / 64, om.copy())\n        return 0\n    except ValueError:\n        return 1\ndef _g():\n    try:\n        _oracle_collapsed_log_determinant(0, 120.0 / 64, om.copy())\n        return 0\n    except ValueError:\n        return 1',
         "call": '_c()',
         "gold_call": '_g()'},   # invalid: n_ring not positive
    ]
