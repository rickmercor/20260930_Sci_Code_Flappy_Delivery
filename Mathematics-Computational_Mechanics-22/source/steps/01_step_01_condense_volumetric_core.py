"""
Factor and condense the local pressure contribution without losing its metric.

The local block contributes through a compliance-weighted row factor, not

through the raw coupling. Preserve that factor because its Gram product is a

useful independent check on the Schur complement: the factor, its displacement

Gram matrix, and the condensed core must satisfy two coupled identities. The

local block is symmetric positive definite but not necessarily diagonal, so the

compliance scaling is fixed by its lower-triangular Cholesky factor rather than

by any other square root; a symmetric or eigenvector square root produces the

same Gram matrix but a different factor and is not the requested convention. A

system with no local field is admissible and contributes exactly nothing.

Reject a non-positive condensed core.

Returns
-------
tuple of three finite float arrays with shapes (m, n), (n, n), and (n, n): Cholesky-scaled pressure rows, induced Gram matrix, condensed SPD core
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def condense_volumetric_core(
    material_stiffness: np.ndarray,
    pressure_coupling: np.ndarray,
    pressure_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the compliance-scaled rows, their Gram matrix, and the core.

    Write $L$ for the lower-triangular Cholesky factor of the local block, so
    that the block equals $LL^\top$. The scaled rows are the unique solution
    $V$ of $LV=B$, the induced Gram matrix is $V^\top V$, and the condensed
    core is the material stiffness plus that Gram matrix. Symmetrize the Gram
    matrix and the core by averaging with their transposes before returning
    them.

    Raises ValueError unless every input is finite; material_stiffness is a
    nonempty symmetric square matrix of shape (n, n); pressure_coupling has
    shape (m, n) with m at least zero; pressure_block is a symmetric positive-
    definite matrix of shape (m, m); and the condensed core is symmetric
    positive definite. Positive definiteness is decided by whether a Cholesky
    factorization succeeds. With m equal to zero the scaled rows have shape
    (0, n), the Gram matrix is exactly zero, and the core is the symmetrized
    material stiffness.

    Parameters
    ----------
    material_stiffness : np.ndarray
        Symmetric displacement matrix of shape (n, n).
    pressure_coupling : np.ndarray
        Signed row-oriented local-field coupling of shape (m, n).
    pressure_block : np.ndarray
        Symmetric positive-definite local block of shape (m, m).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The scaled pressure rows of shape (m, n), their displacement Gram
        matrix of shape (n, n), and the condensed SPD core of shape (n, n).
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_condense_volumetric_core(
    material_stiffness: np.ndarray,
    pressure_coupling: np.ndarray,
    pressure_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference factor-preserving pressure condensation."""
    stiffness = np.asarray(material_stiffness, dtype=float)
    coupling = np.asarray(pressure_coupling, dtype=float)
    block = np.asarray(pressure_block, dtype=float)
    if (
        stiffness.ndim != 2
        or stiffness.shape[0] != stiffness.shape[1]
        or stiffness.shape[0] == 0
    ):
        raise ValueError("material_stiffness must be a nonempty square matrix")
    n_dof = stiffness.shape[0]
    if coupling.ndim != 2 or coupling.shape[1] != n_dof:
        raise ValueError("pressure_coupling must have shape (m, n)")
    n_pressure = coupling.shape[0]
    if block.ndim != 2 or block.shape != (n_pressure, n_pressure):
        raise ValueError("pressure_block must have shape (m, m)")
    if not all(np.all(np.isfinite(x)) for x in (stiffness, coupling, block)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(stiffness, stiffness.T, rtol=0.0, atol=1e-12):
        raise ValueError("material_stiffness must be symmetric")
    if not np.allclose(block, block.T, rtol=0.0, atol=1e-12):
        raise ValueError("pressure_block must be symmetric")
    try:
        factor = np.linalg.cholesky(block)
    except np.linalg.LinAlgError as exc:
        raise ValueError("pressure_block must be positive definite") from exc

    if n_pressure == 0:
        scaled_rows = np.zeros((0, n_dof), dtype=float)
        volumetric = np.zeros((n_dof, n_dof), dtype=float)
    else:
        scaled_rows = np.linalg.solve(factor, coupling)
        volumetric = scaled_rows.T @ scaled_rows
        volumetric = 0.5 * (volumetric + volumetric.T)
    core = stiffness + volumetric
    core = 0.5 * (core + core.T)
    try:
        np.linalg.cholesky(core)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the condensed core must be positive definite") from exc
    return scaled_rows, volumetric, core

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return diagonal, coupled-block, wide, empty, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
K0 = np.array([[40,-1,.2,0,0],[-1,12,-.3,.1,0],[.2,-.3,2.5,-.2,.1],[0,.1,-.2,1.2,-.1],[0,0,.1,-.1,.8]], dtype=float)
B = np.array([[1,-1,0,0,.5],[0,.5,-1,1,0]], dtype=float)
D = np.diag([2.0, 1.5])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in condense_volumetric_core(K0, B, D)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_condense_volumetric_core(K0, B, D)])",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[6.0,-.5,.2],[-.5,3.0,-.1],[.2,-.1,1.4]])
B = np.array([[1.0,-2.0,.5],[0.0,1.0,-1.5]])
D = np.array([[2.5, -0.9], [-0.9, 1.6]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in condense_volumetric_core(K0, B, D)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_condense_volumetric_core(K0, B, D)])",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[3.0, -0.4], [-0.4, 1.0]])
B = np.array([[1.0, -2.0]])
D = np.array([[0.75]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in condense_volumetric_core(K0, B, D)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_condense_volumetric_core(K0, B, D)])",
        },
        {
            "setup": """import numpy as np
K0 = np.diag([5.0, 2.0, 0.8])
B = np.array([[0.2, 0.0, 1.0], [1.0, -0.5, 0.0]])
D = np.diag([1.0e3, 4.0])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in condense_volumetric_core(K0, B, D)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_condense_volumetric_core(K0, B, D)])",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[9.0, .3], [.3, 4.0]])
B = np.array([[1.0, 0.0], [0.5, -1.0], [-0.25, 2.0]])
D = np.array([[3.0, .5, -.2], [.5, 2.0, .4], [-.2, .4, 1.25]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in condense_volumetric_core(K0, B, D)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_condense_volumetric_core(K0, B, D)])",
        },
        {
            "setup": """import numpy as np
K0 = np.array([[2.0, -0.3, 0.1], [-0.3, 1.5, 0.0], [0.1, 0.0, 0.9]])
B = np.zeros((0, 3))
D = np.zeros((0, 0))
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in condense_volumetric_core(K0, B, D)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_condense_volumetric_core(K0, B, D)])",
        },
        {
            "setup": """import numpy as np
K0 = np.eye(2)
B = np.eye(2)
D = np.array([[2.0, 0.4], [-0.4, 1.0]])
def run_model():
    try:
        condense_volumetric_core(K0, B, D)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_condense_volumetric_core(K0, B, D)
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
K0 = np.eye(2)
B = np.eye(2)
D = np.array([[1.0, 2.0], [2.0, 1.0]])
def run_model():
    try:
        condense_volumetric_core(K0, B, D)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_condense_volumetric_core(K0, B, D)
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
K0 = np.array([[1.0, 0.0], [0.0, -3.0]])
B = np.array([[0.5, 0.25]])
D = np.array([[2.0]])
def run_model():
    try:
        condense_volumetric_core(K0, B, D)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_condense_volumetric_core(K0, B, D)
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
