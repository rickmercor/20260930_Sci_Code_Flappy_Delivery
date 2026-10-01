"""
Reduce the two deterministic real bases to weighted checksums that preserve sensitivity to sign, phase, and ordering conventions.

The first checksum reads the first half $Z_1$ of $Z$, and the second reads every entry of $Q$. With one-based indices,



$$

C_Z=\sum_{r=1}^{2n}\sum_{j=1}^{n}(13r+7j)(Z_1)_{rj},

$$



and



$$

C_Q=\sum_{r=1}^{2n}\sum_{c=1}^{2n}(11r+5c)Q_{rc}.

$$



The unequal row and column weights expose otherwise hidden sign, phase, ordering, and negative-imaginary convention errors.

Returns
-------
a finite real length-two array [C_Z, C_Q] using one-based checksum indices; ValueError when the documented input contract fails
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_basis_checksums(Z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return ``[C_Z, C_Q]`` using the prescribed one-based weights.

    Raises ``ValueError`` unless every one of the following holds: ``Z`` is a
    two-dimensional square array of even order at least two; ``Q`` has the same
    shape as ``Z``; every entry of both arrays is finite; and ``Z`` and ``Q`` are
    each orthogonal to an absolute tolerance of ``1e-10``, so a scaled basis such
    as ``2 * I`` is rejected rather than summed.

    Parameters
    ----------
    Z : np.ndarray
        Finite real orthogonal array of shape ``(2n, 2n)``.
    Q : np.ndarray
        Finite real orthogonal array with the same shape as ``Z``.

    Returns
    -------
    np.ndarray
        Real vector ``[C_Z, C_Q]`` of length two.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_basis_checksums(Z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Reference weighted basis checksums."""
    first_basis = np.asarray(Z, dtype=float)
    spectral_basis = np.asarray(Q, dtype=float)
    if (
        first_basis.ndim != 2
        or first_basis.shape[0] != first_basis.shape[1]
        or first_basis.shape[0] < 2
        or first_basis.shape[0] % 2
    ):
        raise ValueError("Z must be an even-order square matrix")
    if spectral_basis.shape != first_basis.shape:
        raise ValueError("Q must have the same shape as Z")
    if not np.all(np.isfinite(first_basis)) or not np.all(np.isfinite(spectral_basis)):
        raise ValueError("Z and Q must contain only finite entries")
    identity = np.eye(first_basis.shape[0])
    if not np.allclose(first_basis.T @ first_basis, identity, atol=1e-10, rtol=0.0):
        raise ValueError("Z must be orthogonal")
    if not np.allclose(
        spectral_basis.T @ spectral_basis,
        identity,
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError("Q must be orthogonal")

    order = first_basis.shape[0]
    n = order // 2
    rows = np.arange(1, order + 1, dtype=float)[:, None]
    first_columns = np.arange(1, n + 1, dtype=float)[None, :]
    all_columns = np.arange(1, order + 1, dtype=float)[None, :]
    c_z = np.sum((13.0 * rows + 7.0 * first_columns) * first_basis[:, :n])
    c_q = np.sum((11.0 * rows + 5.0 * all_columns) * spectral_basis)
    return np.array([c_z, c_q], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return identity, signed, rotated, and invalid basis pairs."""
    return [
        {
            "setup": """import numpy as np
Z = np.eye(4)
Q = np.eye(4)
""",
            "call": "compute_basis_checksums(Z, Q)",
            "gold_call": "_oracle_compute_basis_checksums(Z, Q)",
        },
        {
            "setup": """import numpy as np
Z = np.diag([1.0, -1.0, 1.0, -1.0])
Q = np.diag([-1.0, 1.0, -1.0, 1.0])
""",
            "call": "compute_basis_checksums(Z, Q)",
            "gold_call": "_oracle_compute_basis_checksums(Z, Q)",
        },
        {
            "setup": """import numpy as np
theta = 0.61
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
Z = np.block([[R, np.zeros((2,2))], [np.zeros((2,2)), R]])
Q = Z.T
""",
            "call": "compute_basis_checksums(Z, Q)",
            "gold_call": "_oracle_compute_basis_checksums(Z, Q)",
        },
        {
            "setup": """import numpy as np
Z = np.eye(3)
Q = np.eye(3)
def run_model():
    try:
        compute_basis_checksums(Z, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_basis_checksums(Z, Q)
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
Z = 2.0 * np.eye(4)
Q = np.eye(4)
def run_model():
    try:
        compute_basis_checksums(Z, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_basis_checksums(Z, Q)
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
Z = np.eye(4)
Q = np.eye(2)
def run_model():
    try:
        compute_basis_checksums(Z, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_basis_checksums(Z, Q)
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
Z = np.eye(4)
Q = np.eye(4)
Q[0, 0] = np.nan
def run_model():
    try:
        compute_basis_checksums(Z, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_basis_checksums(Z, Q)
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
Z = np.eye(4)
Q = 2.0 * np.eye(4)
def run_model():
    try:
        compute_basis_checksums(Z, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_basis_checksums(Z, Q)
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
