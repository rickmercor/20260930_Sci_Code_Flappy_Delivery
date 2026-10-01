"""
Compute a matrix-function approximation associated with the classical reduced Krylov representation.

Krylov methods replace a large matrix-function evaluation with a matrix function applied to a much smaller projected matrix. The resulting reduced-space vector is then lifted through the Krylov basis, with the target-vector scaling retained. For the classical formulation that scaling is the Euclidean norm of the target vector, because the first column of the classical basis is the target vector normalized to unit length.

Returns
-------
np.ndarray, the classical approximation to exp(-A)b in the original vector space, one-dimensional with as many entries as b.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def classical_fAb_approx(

    Vq: "np.ndarray",

    Hqq: "np.ndarray",

    b: "np.ndarray",

) -> "np.ndarray":

    """Compute the classical Krylov approximation to exp(-A)b.

    Args:
        Vq: Two-dimensional reduced Krylov basis.
        Hqq: Two-dimensional square reduced matrix.
        b: One-dimensional target vector.

    Returns:
        The matrix-function approximation in the original vector space.

    Raises:
        ValueError: If ``Vq`` or ``Hqq`` is not two-dimensional, is empty,
            contains non-finite values, or has incompatible dimensions.
        ValueError: If ``b`` is not one-dimensional, contains non-finite
            values, or is incompatible with the supplied reduction.
    """

    return approximation  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_classical_fAb_approx(
    Vq: "np.ndarray",
    Hqq: "np.ndarray",
    b: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of the classical matrix-function approximation."""
    Vq = np.asarray(Vq, dtype=float)
    Hqq = np.asarray(Hqq, dtype=float)
    b = np.asarray(b, dtype=float)

    if Vq.ndim != 2:
        raise ValueError("Vq must be two-dimensional")
    if Hqq.ndim != 2 or Hqq.shape[0] != Hqq.shape[1]:
        raise ValueError("Hqq must be square")
    if b.ndim != 1:
        raise ValueError("b must be one-dimensional")
    if not np.all(np.isfinite(Vq)):
        raise ValueError("Vq must contain only finite values")
    if not np.all(np.isfinite(Hqq)):
        raise ValueError("Hqq must contain only finite values")
    if not np.all(np.isfinite(b)):
        raise ValueError("b must contain only finite values")

    if Vq.shape[1] != Hqq.shape[0]:
        raise ValueError("Vq and Hqq dimensions are incompatible")
    if Vq.shape[0] != b.size:
        raise ValueError("Vq and b dimensions are incompatible")
    if Hqq.shape[0] == 0:
        raise ValueError("Hqq must be non-empty")

    xi = np.linalg.norm(b, ord=2)

    if not np.isfinite(xi):
        raise ValueError("b norm must be finite")

    e1 = np.zeros(Hqq.shape[0], dtype=float)
    e1[0] = 1.0

    fH = expm(-Hqq)

    return Vq @ (fH @ e1) * xi

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
b = np.array([1.0, 2.0, 3.0])
""",
            "call": """
classical_fAb_approx(Vq, Hqq, b)
""",
            "gold_call": """
_oracle_classical_fAb_approx(Vq, Hqq, b)
""",
        },
        {
            "setup": """
import numpy as np
Vq = np.array([[1.0]])
Hqq = np.array([[2.0]])
b = np.array([3.0])
""",
            "call": """
classical_fAb_approx(Vq, Hqq, b)
""",
            "gold_call": """
_oracle_classical_fAb_approx(Vq, Hqq, b)
""",
        },
        {
            "setup": """
import numpy as np
Vq = np.eye(4)
Hqq = np.array([
    [1.0, 0.2, 0.0, 0.0],
    [0.0, 2.0, 0.3, 0.0],
    [0.1, 0.0, 3.0, 0.4],
    [0.0, 0.1, 0.0, 4.0],
])
b = np.array([1.0, -1.0, 2.0, 0.5])
""",
            "call": """
classical_fAb_approx(Vq, Hqq, b)
""",
            "gold_call": """
_oracle_classical_fAb_approx(Vq, Hqq, b)
""",
        },
        {
            "setup": """
import numpy as np
Vq = np.eye(2)
Hqq = np.eye(2)
b = np.array([1.0, 2.0])
""",
            "call": """
classical_fAb_approx(Vq, Hqq, b)
""",
            "gold_call": """
_oracle_classical_fAb_approx(Vq, Hqq, b)
""",
        },
    ]
