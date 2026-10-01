"""
Compute the orthogonal polar factor of a nonsingular real skew-symmetric matrix and return the structure-preserving factor used by the later reduction.

For $A=PY$ with $Y$ symmetric positive definite, the orthogonal factor $P$ is unique whenever $A$ is nonsingular, and the singular value decomposition supplies it directly. The two tolerance arguments decide which inputs are accepted, not how $P$ is computed.

Returns
-------
a finite real array P with A.shape, P.T @ P = I, and P.T = -P up to roundoff; ValueError when the documented input contract fails
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_skew_polar_factor(
    A: np.ndarray,
    skew_tolerance: float = 1e-12,
    singularity_tolerance: float = 1e-12,
) -> np.ndarray:
    """Return the unique orthogonal polar factor of ``A``.

    Raises ``ValueError`` unless every one of the following holds: ``A`` is real
    rather than complex; ``A`` is a two-dimensional square array of even order at
    least two; every entry of ``A`` is finite; ``skew_tolerance`` and
    ``singularity_tolerance`` are each finite and strictly positive; the
    normalized defect ``norm(A + A.T) / max(1.0, norm(A))`` is at most
    ``skew_tolerance``; and the smallest singular value of ``A`` exceeds
    ``singularity_tolerance`` times the largest, so a numerically singular ``A``
    is rejected rather than factored.

    Parameters
    ----------
    A : np.ndarray
        Finite real square array of even order at least two.
    skew_tolerance : float, optional
        Maximum normalized Frobenius defect in ``A + A.T``.
    singularity_tolerance : float, optional
        Minimum allowed ratio of smallest to largest singular value.

    Returns
    -------
    np.ndarray
        Real orthogonal polar factor with the same shape as ``A``.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_skew_polar_factor(
    A: np.ndarray,
    skew_tolerance: float = 1e-12,
    singularity_tolerance: float = 1e-12,
) -> np.ndarray:
    """Reference polar-factor calculation."""
    raw = np.asarray(A)
    if np.iscomplexobj(raw):
        raise ValueError("A must be real")
    matrix = np.asarray(raw, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 2
        or matrix.shape[0] % 2
    ):
        raise ValueError("A must be a nonempty even-order square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("A must contain only finite entries")
    if not np.isfinite(skew_tolerance) or skew_tolerance <= 0.0:
        raise ValueError("skew_tolerance must be positive and finite")
    if not np.isfinite(singularity_tolerance) or singularity_tolerance <= 0.0:
        raise ValueError("singularity_tolerance must be positive and finite")

    scale = max(1.0, float(np.linalg.norm(matrix, ord="fro")))
    defect = float(np.linalg.norm(matrix + matrix.T, ord="fro") / scale)
    if defect > skew_tolerance:
        raise ValueError("A must be skew-symmetric within skew_tolerance")

    left, singular_values, right_t = np.linalg.svd(matrix, full_matrices=False)
    if singular_values[-1] <= singularity_tolerance * singular_values[0]:
        raise ValueError("A is numerically singular")
    return left @ right_t

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, scaled, rotated, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
sigma = np.diag([3.0, 0.5])
A = np.block([[np.zeros((2,2)), -sigma], [sigma, np.zeros((2,2))]])
""",
            "call": "compute_skew_polar_factor(A)",
            "gold_call": "_oracle_compute_skew_polar_factor(A)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0, -1e-8], [1e-8, 0.0]])
""",
            "call": "compute_skew_polar_factor(A)",
            "gold_call": "_oracle_compute_skew_polar_factor(A)",
        },
        {
            "setup": """import numpy as np
sigma = np.diag([5.0, 1.0, 0.02])
S = np.block([[np.zeros((3,3)), -sigma], [sigma, np.zeros((3,3))]])
G = np.eye(6)
c, s = np.cos(0.43), np.sin(0.43)
G[0,0] = G[4,4] = c
G[0,4], G[4,0] = -s, s
A = G @ S @ G.T
""",
            "call": "compute_skew_polar_factor(A)",
            "gold_call": "_oracle_compute_skew_polar_factor(A)",
        },
        {
            "setup": """import numpy as np
S = np.array([[0.0,-1.0,0.4,0.2],[1.0,0.0,-0.7,0.3],[-0.4,0.7,0.0,-0.9],[-0.2,-0.3,0.9,0.0]])
E = 0.5 * (np.full((4,4), 1e-8) + np.full((4,4), 1e-8).T)
A = 1e6 * S + E
""",
            "call": "compute_skew_polar_factor(A)",
            "gold_call": "_oracle_compute_skew_polar_factor(A)",
        },
        {
            "setup": """import numpy as np
A = 1e-13 * np.array([[0.0, -1.0], [1.0, 0.0]])
""",
            "call": "compute_skew_polar_factor(A)",
            "gold_call": "_oracle_compute_skew_polar_factor(A)",
        },
        {
            "setup": """import numpy as np
S = np.array([[0.0,-1.0,0.4,0.2],[1.0,0.0,-0.7,0.3],[-0.4,0.7,0.0,-0.9],[-0.2,-0.3,0.9,0.0]])
E = 0.5 * (np.full((4,4), 2.5e-14) + np.full((4,4), 2.5e-14).T)
A = 1e-3 * S + E
""",
            "call": "compute_skew_polar_factor(A)",
            "gold_call": "_oracle_compute_skew_polar_factor(A)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0, 1.0], [1.0, 0.0]])
def run_model():
    try:
        compute_skew_polar_factor(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_polar_factor(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0,-1.0,0.0,0.0],[1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0]])
def run_model():
    try:
        compute_skew_polar_factor(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_polar_factor(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0, -1.0], [1.0, 0.0]])
def run_model():
    try:
        compute_skew_polar_factor(A, 1e-12, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_polar_factor(A, 1e-12, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0, -1.0], [1.0, 0.0]])
def run_model():
    try:
        compute_skew_polar_factor(A, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_polar_factor(A, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0, -1.0], [1.0, np.nan]])
def run_model():
    try:
        compute_skew_polar_factor(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_polar_factor(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=complex)
def run_model():
    try:
        compute_skew_polar_factor(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_polar_factor(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
A = np.zeros((3, 3))
def run_model():
    try:
        compute_skew_polar_factor(A)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_skew_polar_factor(A)
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
