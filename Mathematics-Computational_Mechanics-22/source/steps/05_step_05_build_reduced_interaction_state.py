"""
Build the compact retained-coordinate state used by the inverse action.

The retained interaction basis is useful only after it has been coupled to the

displacement core. Algorithm 1 of the source keeps that coupling compact: the

retained rows, their core responses, and the small reduced matrix are stored,

while a dense displacement-space inverse or filter is never formed. This

representation is also the one whose order matters in a flexible solve. A

rank-zero basis is the exact contact-free state and therefore has genuinely

empty retained arrays rather than dummy zero columns.

Returns
-------
tuple of four finite float np.ndarray values with shapes (n, n), (r, n), (n, r), and (r, r): the symmetric reduced preconditioner and the compact retained-row, core-response, and reduced-matrix state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_reduced_interaction_state(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    retained_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Return the reduced preconditioner and its compact Woodbury state.

    Let the columns of ``retained_basis`` be the supplied orthonormal
    interaction-space basis. Return the symmetric reduced preconditioner,
    the retained row matrix, the corresponding core-response block, and the
    reduced matrix required by the source's retained-basis inverse action.
    The basis may have zero columns; in that case the last three returns have
    shapes (0, n), (n, 0), and (0, 0), respectively.

    Raises ValueError unless condensed_core is finite symmetric positive
    definite with shape (n, n); interaction_rows is finite with shape (m, n),
    including (0, n) and m greater than n; retained_basis is finite with shape
    (m, r), 0 <= r <= m, and has orthonormal columns to rtol 1e-10 and atol
    1e-12; and both the reduced matrix and reduced preconditioner are positive
    definite when nonempty.

    Parameters
    ----------
    condensed_core : np.ndarray
        Symmetric positive-definite displacement core of shape (n, n).
    interaction_rows : np.ndarray
        Ordered interaction factor of shape (m, n).
    retained_basis : np.ndarray
        Supplied orthonormal interaction-space basis of shape (m, r).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Reduced preconditioner P of shape (n, n), retained rows R of shape
        (r, n), core responses Z of shape (n, r), and reduced matrix S of
        shape (r, r).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_reduced_interaction_state(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    retained_basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference construction of the source's compact retained state."""
    core = np.asarray(condensed_core, dtype=float)
    rows = np.asarray(interaction_rows, dtype=float)
    basis = np.asarray(retained_basis, dtype=float)
    if core.ndim != 2 or core.shape[0] != core.shape[1] or core.shape[0] == 0:
        raise ValueError("condensed_core must be a nonempty square matrix")
    n_dof = core.shape[0]
    if rows.ndim != 2 or rows.shape[1] != n_dof:
        raise ValueError("interaction_rows must have shape (m, n)")
    n_rows = rows.shape[0]
    if basis.ndim != 2 or basis.shape[0] != n_rows or basis.shape[1] > n_rows:
        raise ValueError("retained_basis must have shape (m, r), 0 <= r <= m")
    if not all(np.all(np.isfinite(x)) for x in (core, rows, basis)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(core, core.T, rtol=0.0, atol=1e-12):
        raise ValueError("condensed_core must be symmetric")
    try:
        np.linalg.cholesky(core)
    except np.linalg.LinAlgError as exc:
        raise ValueError("condensed_core must be positive definite") from exc

    rank = basis.shape[1]
    if rank and not np.allclose(basis.T @ basis, np.eye(rank), rtol=1e-10, atol=1e-12):
        raise ValueError("retained_basis must have orthonormal columns")

    retained_rows = basis.T @ rows
    if rank == 0:
        return (
            0.5 * (core + core.T),
            np.zeros((0, n_dof), dtype=float),
            np.zeros((n_dof, 0), dtype=float),
            np.zeros((0, 0), dtype=float),
        )

    core_responses = np.linalg.solve(core, retained_rows.T)
    reduced_matrix = np.eye(rank, dtype=float) + retained_rows @ core_responses
    reduced_matrix = 0.5 * (reduced_matrix + reduced_matrix.T)
    try:
        np.linalg.cholesky(reduced_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the reduced matrix must be positive definite") from exc

    preconditioner = core + retained_rows.T @ retained_rows
    preconditioner = 0.5 * (preconditioner + preconditioner.T)
    try:
        np.linalg.cholesky(preconditioner)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the preconditioner must be positive definite") from exc
    return preconditioner, retained_rows, core_responses, reduced_matrix

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return rank-two, zero-rank, full-rank, wide, stiff, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
M = np.array([[40.5,-1.5,.2,0,.25],[-1.5,12.666666666666666,-.633333333333333,.433333333333333,-.25],[.2,-.633333333333333,3.166666666666667,-.866666666666667,.1],[0,.433333333333333,-.866666666666667,1.866666666666667,-.1],[.25,-.25,.1,-.1,.925]])
U = np.array([[8,0,0,0,0],[0,4,0,0,0],[0,0,2,.2,0],[0,0,.1,1.5,1.2]], dtype=float)
G = U @ np.linalg.solve(M, U.T)
w, q = np.linalg.eigh((G + G.T) / 2)
Q = q[:, np.argsort(w)[::-1][:2]]
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in build_reduced_interaction_state(M, U, Q)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_build_reduced_interaction_state(M, U, Q)])",
        },
        {
            "setup": """import numpy as np
M = np.diag([4.0, 2.0, 1.0])
U = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.5]])
Q = np.zeros((2, 0))
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in build_reduced_interaction_state(M, U, Q)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_build_reduced_interaction_state(M, U, Q)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[3.0, 0.2], [0.2, 1.0]])
U = np.array([[1.0, -0.5], [0.25, 2.0]])
Q = np.eye(2)
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in build_reduced_interaction_state(M, U, Q)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_build_reduced_interaction_state(M, U, Q)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[5.0, .4], [.4, 2.0]])
U = np.array([[3.0, 0.0], [0.0, 1.0], [-1.0, .5], [.25, 2.0]])
G = U @ np.linalg.solve(M, U.T)
w, q = np.linalg.eigh((G + G.T) / 2)
Q = q[:, np.argsort(w)[::-1][:2]]
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in build_reduced_interaction_state(M, U, Q)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_build_reduced_interaction_state(M, U, Q)])",
        },
        {
            "setup": """import numpy as np
M = np.diag([1.0e8, 1.0])
U = np.array([[1.0, 0.0], [0.0, 2.0]])
Q = np.array([[1.0], [0.0]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in build_reduced_interaction_state(M, U, Q)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_build_reduced_interaction_state(M, U, Q)])",
        },
        {
            "setup": """import numpy as np
M = np.eye(2)
U = np.eye(2)
Q = np.array([[1.0, 0.5], [0.0, 1.0]])
def run_model():
    try:
        build_reduced_interaction_state(M, U, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_reduced_interaction_state(M, U, Q)
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
M = np.eye(3)
U = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
Q = np.eye(3)
def run_model():
    try:
        build_reduced_interaction_state(M, U, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_reduced_interaction_state(M, U, Q)
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
M = np.array([[1.0, 2.0], [2.0, 1.0]])
U = np.eye(2)
Q = np.eye(2)
def run_model():
    try:
        build_reduced_interaction_state(M, U, Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_reduced_interaction_state(M, U, Q)
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
