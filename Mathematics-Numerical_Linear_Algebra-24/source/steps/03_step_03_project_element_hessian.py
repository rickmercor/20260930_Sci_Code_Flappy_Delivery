"""
Apply the local spectral projection to one element Hessian.



For ``H_e = Q_e D_e Q_e.T``, replace each diagonal value by

`$Dhat_e[ii] = max(D_e[ii], epsilon)$` and form ``Hhat_e = Q_e Dhat_e Q_e.T``.

The positive floor differs from a zero-clamp positive-semidefinite projection.

Returns
-------
np.ndarray with the input shape and eigenvalues at least eigenvalue_floor
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def project_element_hessian(
    hessian: np.ndarray, eigenvalue_floor: float = 1e-8
) -> np.ndarray:
    """Clamp every eigenvalue below ``eigenvalue_floor``.

    Raises ``ValueError`` unless every one of the following holds: ``hessian``
    is a nonempty square two-dimensional array; all its entries are finite;
    it equals its own transpose to within ``1e-12`` absolute, which is
    verified rather than assumed, so a nonsymmetric argument is rejected
    instead of being symmetrized or passed to the eigendecomposition; and
    ``eigenvalue_floor`` is a finite strictly positive scalar.

    Parameters
    ----------
    hessian : np.ndarray
        Finite square element Hessian. Symmetry is a checked requirement, not
        a caller guarantee.
    eigenvalue_floor : float
        Strictly positive eigenvalue floor.

    Returns
    -------
    np.ndarray
        Symmetric projected Hessian with the same shape.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_project_element_hessian(hessian, eigenvalue_floor=1e-8):
    """Reference symmetric eigenvalue-clamping projection."""
  
    hessian = np.asarray(hessian, dtype=float)
    if (
        hessian.ndim != 2
        or hessian.shape[0] < 1
        or hessian.shape[0] != hessian.shape[1]
    ):
        raise ValueError("hessian must be a nonempty square matrix")
    if not np.all(np.isfinite(hessian)):
        raise ValueError("hessian must be finite")
    if not np.allclose(hessian, hessian.T, rtol=0.0, atol=1e-12):
        raise ValueError("hessian must be symmetric")
    if not np.isscalar(eigenvalue_floor) or not np.isfinite(eigenvalue_floor):
        raise ValueError("eigenvalue_floor must be finite")
    if float(eigenvalue_floor) <= 0.0:
        raise ValueError("eigenvalue_floor must be positive")
    values, vectors = np.linalg.eigh(hessian)
    projected = (vectors * np.maximum(values, float(eigenvalue_floor))) @ vectors.T
    return 0.5 * (projected + projected.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return strongly indefinite, mildly indefinite, and nonsymmetric cases."""
    return [
        {
            "setup": """
import numpy as np
H = np.array([[-4.0, 0.5], [0.5, 1.5]])
floor = 1e-8
""",
            "call": "project_element_hessian(H, floor)",
            "gold_call": "_oracle_project_element_hessian(H, floor)",
        },
        {
            "setup": """
import numpy as np
H = np.array([[-0.25, 0.25], [0.25, 1.0]])
floor = 1e-8
""",
            "call": "project_element_hessian(H, floor)",
            "gold_call": "_oracle_project_element_hessian(H, floor)",
        },
        {
            "setup": """
import numpy as np
H = np.array([[1.0, 2.0], [0.0, 1.0]])
def run_model():
    try:
        project_element_hessian(H)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_project_element_hessian(H)
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
