"""
Rank the interaction rows by the response they drive through the core.

Contact rows are not equally worth keeping. What matters is the displacement

response a row combination transmits through the condensed core, so the ranking

is set by an energy the core defines and not by Euclidean row size; a selector

built from raw row products is the standard near-miss and it retains different

rows. The retained subspace is fixed variationally rather than by a recipe:

among all orthogonal projectors of the requested rank in interaction space,

exactly one captures the greatest total response, and it is unique precisely

when the cutoff is strictly separated. Return the response operator, its

ordered levels, that projector, and the level ratio across the cutoff. A

projector is returned in place of a basis because the retained subspace, and

not any particular set of vectors spanning it, is what the later operators

depend on.

Returns
-------
tuple of finite float arrays with shapes (m, m), (m,), (m, m) and one finite nonnegative float: response operator, nonincreasing levels, retained orthogonal projector, cutoff level ratio
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_response_gramian(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    r"""Return the response operator, its levels, the retained projector, ratio.

    The response operator $G$ is the symmetric interaction-space operator whose
    quadratic form $c^\top Gc$ is the core energy $d^\top M^{-1}d$ of the
    displacement load $d=U^\top c$ that the row combination $c$ applies. Its
    levels are the eigenvalues of $G$ in nonincreasing order, clipped at zero.
    The retained projector is the orthogonal projector $\Pi$ of rank $r$ on
    interaction space maximizing $\operatorname{tr}(\Pi G)$. The cutoff ratio
    is the first omitted level divided by the last retained level, and is
    exactly 0.0 when nothing is omitted or nothing is retained.

    Raises ValueError unless condensed_core is finite, symmetric positive
    definite, and nonempty; interaction_rows is finite with shape (m, n),
    including (0, n) and m greater than n; rank is a Python or NumPy integer,
    not a bool, lying in [0, m]; the response is positive semidefinite to
    tolerance; the maximizing projector is unique; and the retained rank is
    supported by the numerically positive part of the response, so a rank
    reaching into its null space is rejected.

    Parameters
    ----------
    condensed_core : np.ndarray
        Symmetric positive-definite core of shape (n, n).
    interaction_rows : np.ndarray
        Ordered row-oriented interaction factor of shape (m, n).
    rank : int
        Retained response dimension.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, float]
        Response operator of shape (m, m), nonincreasing levels of shape (m,),
        retained orthogonal projector of shape (m, m), and the cutoff ratio.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_response_gramian(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Reference metric response and variational dominant selection."""
    core = np.asarray(condensed_core, dtype=float)
    rows = np.asarray(interaction_rows, dtype=float)
    if core.ndim != 2 or core.shape[0] != core.shape[1] or core.shape[0] == 0:
        raise ValueError("condensed_core must be a nonempty square matrix")
    n_dof = core.shape[0]
    if rows.ndim != 2 or rows.shape[1] != n_dof:
        raise ValueError("interaction_rows must have shape (m, n)")
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be an integer")  # noqa: TRY004 - API contract
    rank = int(rank)
    n_rows = rows.shape[0]
    if rank < 0 or rank > n_rows:
        raise ValueError("rank must lie in [0, m]")
    if not np.all(np.isfinite(core)) or not np.all(np.isfinite(rows)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(core, core.T, rtol=0.0, atol=1e-12):
        raise ValueError("condensed_core must be symmetric")
    try:
        np.linalg.cholesky(core)
    except np.linalg.LinAlgError as exc:
        raise ValueError("condensed_core must be positive definite") from exc
    if n_rows == 0:
        empty_square = np.zeros((0, 0), dtype=float)
        return empty_square, np.zeros(0), empty_square, 0.0

    response = rows @ np.linalg.solve(core, rows.T)
    response = 0.5 * (response + response.T)
    eigenvalues, eigenvectors = np.linalg.eigh(response)
    if eigenvalues[0] < -1e-10 * max(1.0, float(np.max(np.abs(eigenvalues)))):
        raise ValueError("the response must be positive semidefinite")
    order = np.argsort(eigenvalues)[::-1]
    levels = np.maximum(eigenvalues[order], 0.0)
    vectors = eigenvectors[:, order]
    scale = max(1.0, float(levels[0]))
    positive_tolerance = max(n_rows, n_dof, 1) * np.finfo(float).eps * scale
    numerical_rank = int(np.count_nonzero(levels > positive_tolerance))
    if rank > numerical_rank:
        raise ValueError("rank exceeds the numerical rank of the response")
    gap_tolerance = 1e-10 * scale
    if (
        0 < rank < n_rows
        and abs(float(levels[rank - 1] - levels[rank])) <= gap_tolerance
    ):
        raise ValueError("the maximizing projector is not unique")
    retained = vectors[:, :rank]
    projector = retained @ retained.T
    projector = 0.5 * (projector + projector.T)
    cutoff_ratio = float(levels[rank] / levels[rank - 1]) if 0 < rank < n_rows else 0.0
    return response, levels, projector, cutoff_ratio

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return task-scale, zero-rank, empty, wide, deficient, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
M = np.array([[40.5,-1.5,.2,0,.25],[-1.5,12.666666666666666,-.633333333333333,.433333333333333,-.25],[.2,-.633333333333333,3.166666666666667,-.866666666666667,.1],[0,.433333333333333,-.866666666666667,1.866666666666667,-.1],[.25,-.25,.1,-.1,.925]])
U = np.array([[8,0,0,0,0],[0,4,0,0,0],[0,0,2,.2,0],[0,0,.1,1.5,1.2]], dtype=float)
r = 2
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in build_response_gramian(M, U, r)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_build_response_gramian(M, U, r)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[3.0, .2, 0.0], [.2, 1.5, -.1], [0.0, -.1, .8]])
U = np.array([[1.0, -2.0, .5], [0.0, 1.0, 3.0]])
r = 0
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in build_response_gramian(M, U, r)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_build_response_gramian(M, U, r)])",
        },
        {
            "setup": """import numpy as np
M = np.diag([3.0, 1.0])
U = np.zeros((0, 2))
r = 0
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in build_response_gramian(M, U, r)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_build_response_gramian(M, U, r)])",
        },
        {
            "setup": """import numpy as np
M = np.diag([2.0, 3.0, 4.0])
U = np.diag([3.0, 2.0, 1.0])
r = 3
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in build_response_gramian(M, U, r)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_build_response_gramian(M, U, r)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[5.0, .4], [.4, 2.0]])
U = np.array([[3.0, 0.0], [0.0, 1.0], [-1.0, .5], [.25, 2.0]])
r = 2
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in build_response_gramian(M, U, r)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_build_response_gramian(M, U, r)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[2.0, .2], [.2, 1.0]])
U = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, .5]])
r = 1
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in build_response_gramian(M, U, r)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_build_response_gramian(M, U, r)])",
        },
        {
            "setup": """import numpy as np
M = np.eye(2)
U = np.diag([2.0, 2.0])
r = 1
def run_model():
    try:
        build_response_gramian(M, U, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_response_gramian(M, U, r)
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
M = np.diag([1.0, 4.0])
U = np.array([[1.5, 0.0], [0.0, 0.0]])
r = 2
def run_model():
    try:
        build_response_gramian(M, U, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_response_gramian(M, U, r)
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
U = np.diag([3.0, 2.0, 1.0])
r = 2.0
def run_model():
    try:
        build_response_gramian(M, U, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_response_gramian(M, U, r)
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
M = np.array([[1.0, 3.0], [3.0, 1.0]])
U = np.eye(2)
r = 1
def run_model():
    try:
        build_response_gramian(M, U, r)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_response_gramian(M, U, r)
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
