"""
Compute a matrix-function approximation associated with the global reduced Krylov representation. The routine validates the reduced representation and the unnormalized starting block, evaluates the reduced matrix function, and maps the result back to the original space.

The global Krylov formulation uses the same reduced matrix-function principle as the classical formulation, but its basis and its scaling come from different objects. The basis is built from the block normalized to unit Frobenius norm, while the scaling that maps the reduced result back to the original space is the Frobenius norm of the unnormalized starting block. Taking the scaling from the normalized block instead would give 1 and leave the reconstruction short by exactly that factor.

Returns
-------
np.ndarray, the global approximation to exp(-A)b in the original vector space, one-dimensional with as many entries as the rows of Vq.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def global_fAb_approx(

    Vq: "np.ndarray",

    Hqq: "np.ndarray",

    Omega_ell: "np.ndarray",

) -> "np.ndarray":

    """Compute the global Krylov approximation to exp(-A)b.

    Args:
        Vq: Two-dimensional reduced Krylov basis.
        Hqq: Two-dimensional square reduced matrix.
        Omega_ell: Two-dimensional starting block associated with the
            reduction.

    Returns:
        The matrix-function approximation in the original vector space.

    Raises:
        ValueError: If ``Vq`` or ``Hqq`` is not two-dimensional, is empty,
            contains non-finite values, or has incompatible dimensions.
        ValueError: If ``Omega_ell`` is not two-dimensional, is empty,
            contains non-finite values, or is incompatible with the supplied
            reduction.
    """

    return approximation  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_global_fAb_approx(
    Vq: "np.ndarray",
    Hqq: "np.ndarray",
    Omega_ell: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the global matrix-function approximation."""
    Vq = np.asarray(Vq, dtype=float)
    Hqq = np.asarray(Hqq, dtype=float)
    Omega_ell = np.asarray(Omega_ell, dtype=float)

    if Vq.ndim != 2:
        raise ValueError("Vq must be two-dimensional")
    if Hqq.ndim != 2 or Hqq.shape[0] != Hqq.shape[1]:
        raise ValueError("Hqq must be square")
    if Omega_ell.ndim != 2:
        raise ValueError("Omega_ell must be two-dimensional")

    if not np.all(np.isfinite(Vq)):
        raise ValueError("Vq must contain only finite values")
    if not np.all(np.isfinite(Hqq)):
        raise ValueError("Hqq must contain only finite values")
    if not np.all(np.isfinite(Omega_ell)):
        raise ValueError("Omega_ell must contain only finite values")

    if Vq.shape[1] != Hqq.shape[0]:
        raise ValueError("Vq and Hqq dimensions are incompatible")
    if Omega_ell.shape[0] != Vq.shape[0]:
        raise ValueError("Omega_ell and Vq dimensions are incompatible")
    if Omega_ell.shape[1] == 0:
        raise ValueError("Omega_ell must be non-empty")
    if Hqq.shape[0] == 0:
        raise ValueError("Hqq must be non-empty")

    xi_G = np.linalg.norm(Omega_ell, ord="fro")

    if not np.isfinite(xi_G) or xi_G <= 0.0:
        raise ValueError("Omega_ell must have positive Frobenius norm")

    e1 = np.zeros(Hqq.shape[0], dtype=float)
    e1[0] = 1.0

    fH = expm(-Hqq)

    return Vq @ (fH @ e1) * xi_G

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    return [

        {
            "setup": """
import numpy as np

Vq = np.eye(3)

Hqq = np.diag([1.0, 2.0, 3.0])

Omega_ell = np.ones((3, 1))

""",

            "call": """
global_fAb_approx(Vq, Hqq, Omega_ell)
""",

            "gold_call": """
_oracle_global_fAb_approx(Vq, Hqq, Omega_ell)
""",

        },

        {
            "setup": """
import numpy as np

Vq = np.eye(2)

Hqq = np.array([
    [2.0, 0.5],
    [0.0, 3.0],
])

Omega_ell = np.array([
    [2.0],
    [0.0],
])

""",

            "call": """
global_fAb_approx(Vq, Hqq, Omega_ell)
""",

            "gold_call": """
_oracle_global_fAb_approx(Vq, Hqq, Omega_ell)
""",

        },

        {
            "setup": """
import numpy as np

Vq = np.eye(4)

Hqq = np.array([
    [1.0, 0.1, 0.0, 0.0],
    [0.0, 2.0, 0.2, 0.0],
    [0.0, 0.0, 3.0, 0.3],
    [0.1, 0.0, 0.0, 4.0],
])

Omega_ell = np.ones((4, 2))

""",

            "call": """
global_fAb_approx(Vq, Hqq, Omega_ell)
""",

            "gold_call": """
_oracle_global_fAb_approx(Vq, Hqq, Omega_ell)
""",

        },

        {
            "setup": """
import numpy as np

Vq = np.eye(2)

Hqq = np.eye(2)

Omega_ell = np.array([
    [2.0],
    [0.0],
])

""",

            "call": """
global_fAb_approx(Vq, Hqq, Omega_ell)
""",

            "gold_call": """
_oracle_global_fAb_approx(Vq, Hqq, Omega_ell)
""",

        },

    ]
