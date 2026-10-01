"""
Prepare the augmented starting block for the global block Krylov strategy.

The global block Krylov formulation treats an entire block as one object under a scalar inner-product convention. Its normalization is therefore based on the Frobenius norm of the complete block rather than independent column normalization. The scale removed here, the Frobenius norm of the unnormalized block, is needed again when the reduced result is mapped back to the original space, so the unnormalized block is retained separately rather than discarded.

Returns
-------
np.ndarray, the input block normalized to unit Frobenius norm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def global_normalize(

    Omega_ell: "np.ndarray",

) -> "np.ndarray":

    """Construct the normalized global starting block.

    Args:
        Omega_ell: Two-dimensional starting block.

    Returns:
        The normalized starting block with the same shape as
        ``Omega_ell``.

    Raises:
        ValueError: If ``Omega_ell`` is not two-dimensional, is empty,
            contains non-finite values, or cannot be normalized.
    """

    return normalized_block  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_global_normalize(
    Omega_ell: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation of global initialization."""
    Omega_ell = np.asarray(Omega_ell, dtype=float)

    if Omega_ell.ndim != 2:
        raise ValueError("Omega_ell must be two-dimensional")
    if Omega_ell.size == 0:
        raise ValueError("Omega_ell must be non-empty")
    if not np.all(np.isfinite(Omega_ell)):
        raise ValueError("Omega_ell must contain only finite values")

    norm_f = np.linalg.norm(Omega_ell, ord="fro")

    if not np.isfinite(norm_f) or norm_f <= 0.0:
        raise ValueError("Omega_ell must have positive Frobenius norm")

    return Omega_ell / norm_f

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
Omega_ell = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
])
""",
            "call": """
global_normalize(Omega_ell)
""",
            "gold_call": """
_oracle_global_normalize(Omega_ell)
""",
        },
        {
            "setup": """
import numpy as np
Omega_ell = np.eye(1)
""",
            "call": """
global_normalize(Omega_ell)
""",
            "gold_call": """
_oracle_global_normalize(Omega_ell)
""",
        },
        {
            "setup": """
import numpy as np
Omega_ell = np.array([
    [2.0, -1.0, 3.0],
    [4.0, 0.5, -2.0],
])
""",
            "call": """
global_normalize(Omega_ell)
""",
            "gold_call": """
_oracle_global_normalize(Omega_ell)
""",
        },
        {
            "setup": """
import numpy as np
Omega_ell = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
])
""",
            "call": """
global_normalize(Omega_ell)
""",
            "gold_call": """
_oracle_global_normalize(Omega_ell)
""",
        },
    ]
