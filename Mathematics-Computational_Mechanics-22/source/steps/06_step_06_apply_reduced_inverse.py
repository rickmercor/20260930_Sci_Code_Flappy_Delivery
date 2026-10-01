"""
Apply a compact retained-coordinate inverse and its true adjoint.

The source's preconditioner is an action, not a stored dense inverse.  A core

solve is corrected through the retained row block, its already-computed core

responses, and one small reduced solve.  The adjoint reverses that complete

product: for a nonsymmetric core or reduced state it cannot be obtained by

merely toggling the core solve or reusing the forward correction in the same

order.  The compact identities are checked because later flexible iterations

must consume the state actually built for their own retained spaces.

Returns
-------
one finite float np.ndarray with exactly the input right-side shape (n,) or (n, k), equal to the compact inverse action or its complete adjoint
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def apply_reduced_inverse(
    base_operator: np.ndarray,
    retained_rows: np.ndarray,
    core_responses: np.ndarray,
    reduced_matrix: np.ndarray,
    right_hand_sides: np.ndarray,
    transposed: bool,
) -> np.ndarray:
    r"""Apply the compact Woodbury action, or its adjoint, to a vector/block.

    The three compact arrays must constitute one valid retained state for the
    supplied base operator: the response block solves the retained transpose
    loads through the base, and the reduced matrix is the identity plus their
    retained-coordinate coupling.  Apply Algorithm 1 of the source when
    ``transposed`` is false and the adjoint of that complete action otherwise.

    Raises ValueError unless all arrays are finite; base_operator is a nonempty
    nonsingular square matrix of shape (n, n); retained_rows, core_responses,
    and reduced_matrix have shapes (r, n), (n, r), and (r, r), including r=0;
    the two compact-state identities hold to rtol 1e-10 and atol 1e-12;
    reduced_matrix is nonsingular; right_hand_sides has shape (n,) or (n, k)
    with k at least one; and transposed is a Python or NumPy bool.  Preserve
    the exact vector-or-block shape of the supplied right side.

    Parameters
    ----------
    base_operator : np.ndarray
        Nonsingular base operator M of shape (n, n).
    retained_rows : np.ndarray
        Retained row matrix R of shape (r, n).
    core_responses : np.ndarray
        Core-response block Z of shape (n, r).
    reduced_matrix : np.ndarray
        Reduced matrix S of shape (r, r).
    right_hand_sides : np.ndarray
        One vector (n,) or a column block (n, k).
    transposed : bool
        Select the adjoint of the complete inverse action when True.

    Returns
    -------
    np.ndarray
        Compact inverse action with the same shape as right_hand_sides.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_apply_reduced_inverse(
    base_operator: np.ndarray,
    retained_rows: np.ndarray,
    core_responses: np.ndarray,
    reduced_matrix: np.ndarray,
    right_hand_sides: np.ndarray,
    transposed: bool,
) -> np.ndarray:
    """Reference compact forward or adjoint Woodbury action."""
    base = np.asarray(base_operator, dtype=float)
    rows = np.asarray(retained_rows, dtype=float)
    responses = np.asarray(core_responses, dtype=float)
    reduced = np.asarray(reduced_matrix, dtype=float)
    rhs = np.asarray(right_hand_sides, dtype=float)
    if not isinstance(transposed, (bool, np.bool_)):
        raise ValueError("transposed must be a bool")  # noqa: TRY004
    if base.ndim != 2 or base.shape[0] != base.shape[1] or base.shape[0] == 0:
        raise ValueError("base_operator must be a nonempty square matrix")
    n_dof = base.shape[0]
    if rows.ndim != 2 or rows.shape[1] != n_dof:
        raise ValueError("retained_rows must have shape (r, n)")
    rank = rows.shape[0]
    if responses.shape != (n_dof, rank):
        raise ValueError("core_responses must have shape (n, r)")
    if reduced.shape != (rank, rank):
        raise ValueError("reduced_matrix must have shape (r, r)")
    if rhs.ndim == 1:
        if rhs.shape != (n_dof,):
            raise ValueError("a vector right side must have shape (n,)")
    elif rhs.ndim == 2:
        if rhs.shape[0] != n_dof or rhs.shape[1] == 0:
            raise ValueError("a block right side must have shape (n, k), k >= 1")
    else:
        raise ValueError("right_hand_sides must be a vector or column block")
    if not all(np.all(np.isfinite(x)) for x in (base, rows, responses, reduced, rhs)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(base @ responses, rows.T, rtol=1e-10, atol=1e-12):
        raise ValueError("core_responses do not solve the retained transpose loads")
    expected_reduced = np.eye(rank, dtype=float) + rows @ responses
    if not np.allclose(reduced, expected_reduced, rtol=1e-10, atol=1e-12):
        raise ValueError("reduced_matrix is inconsistent with the compact state")

    try:
        if bool(transposed):
            corrected = rhs - rows.T @ np.linalg.solve(reduced.T, responses.T @ rhs)
            result = np.linalg.solve(base.T, corrected)
        else:
            core_action = np.linalg.solve(base, rhs)
            result = core_action - responses @ np.linalg.solve(
                reduced, rows @ core_action
            )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "base_operator and reduced_matrix must be nonsingular"
        ) from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("the inverse action must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return forward, adjoint, nonsymmetric, rank-zero, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
M = np.array([[40.5,-1.5,.2,0,.25],[-1.5,12.666666666666666,-.633333333333333,.433333333333333,-.25],[.2,-.633333333333333,3.166666666666667,-.866666666666667,.1],[0,.433333333333333,-.866666666666667,1.866666666666667,-.1],[.25,-.25,.1,-.1,.925]])
R = np.array([[-.331419211787616,.086034102962501,.822780175278025,1.4677952822772,1.115852830670655],[7.607543403265093,1.165070606175298,.207739110938568,.012975632230369,-.00628049304442]])
Z = np.linalg.solve(M, R.T)
S = np.eye(2) + R @ Z
V = np.column_stack([np.array([1.0, -.5, .75, .2, -1.1]), np.eye(5)[:, 2]])
""",
            "call": "apply_reduced_inverse(M, R, Z, S, V, False)",
            "gold_call": "_oracle_apply_reduced_inverse(M, R, Z, S, V, False)",
        },
        {
            "setup": """import numpy as np
M = np.array([[40.5,-1.5,.2,0,.25],[-1.5,12.666666666666666,-.633333333333333,.433333333333333,-.25],[.2,-.633333333333333,3.166666666666667,-.866666666666667,.1],[0,.433333333333333,-.866666666666667,1.866666666666667,-.1],[.25,-.25,.1,-.1,.925]])
R = np.array([[-.331419211787616,.086034102962501,.822780175278025,1.4677952822772,1.115852830670655],[7.607543403265093,1.165070606175298,.207739110938568,.012975632230369,-.00628049304442]])
Z = np.linalg.solve(M, R.T)
S = np.eye(2) + R @ Z
V = np.column_stack([np.array([1.0, -.5, .75, .2, -1.1]), np.eye(5)[:, 2]])
""",
            "call": "apply_reduced_inverse(M, R, Z, S, V, True)",
            "gold_call": "_oracle_apply_reduced_inverse(M, R, Z, S, V, True)",
        },
        {
            "setup": """import numpy as np
M = np.array([[2.0, 1.0], [.2, 1.5]])
R = np.array([[1.0, -.5]])
Z = np.linalg.solve(M, R.T)
S = np.eye(1) + R @ Z
v = np.array([.3, -1.2])
""",
            "call": "apply_reduced_inverse(M, R, Z, S, v, False)",
            "gold_call": "_oracle_apply_reduced_inverse(M, R, Z, S, v, False)",
        },
        {
            "setup": """import numpy as np
M = np.array([[2.0, 1.0], [.2, 1.5]])
R = np.array([[1.0, -.5]])
Z = np.linalg.solve(M, R.T)
S = np.eye(1) + R @ Z
v = np.array([.3, -1.2])
""",
            "call": "apply_reduced_inverse(M, R, Z, S, v, True)",
            "gold_call": "_oracle_apply_reduced_inverse(M, R, Z, S, v, True)",
        },
        {
            "setup": """import numpy as np
M = np.array([[3.0, -0.2], [-0.2, 1.0]])
R = np.zeros((0, 2))
Z = np.zeros((2, 0))
S = np.zeros((0, 0))
V = np.array([[1.0, 0.0], [-2.0, 1.0]])
""",
            "call": "apply_reduced_inverse(M, R, Z, S, V, False)",
            "gold_call": "_oracle_apply_reduced_inverse(M, R, Z, S, V, False)",
        },
        {
            "setup": """import numpy as np
M = np.array([[4.0, .7, -.2], [.1, 2.0, .3], [0.0, -.5, 1.2]])
R = np.array([[1.0, -.5, .25], [0.0, 2.0, -1.0]])
Z = np.linalg.solve(M, R.T)
S = np.eye(2) + R @ Z
V = np.column_stack([np.array([1.0, 0.0, -1.0]), np.array([.2, .4, .6])])
""",
            "call": "apply_reduced_inverse(M, R, Z, S, V, True)",
            "gold_call": "_oracle_apply_reduced_inverse(M, R, Z, S, V, True)",
        },
        {
            "setup": """import numpy as np
M = np.zeros((2, 2))
R = np.zeros((0, 2))
Z = np.zeros((2, 0))
S = np.zeros((0, 0))
v = np.ones(2)
def run_model():
    try:
        apply_reduced_inverse(M, R, Z, S, v, False)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_apply_reduced_inverse(M, R, Z, S, v, False)
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
M = np.eye(2)
R = np.zeros((0, 2))
Z = np.zeros((2, 0))
S = np.zeros((0, 0))
V = np.zeros((2, 0))
def run_model():
    try:
        apply_reduced_inverse(M, R, Z, S, V, False)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_apply_reduced_inverse(M, R, Z, S, V, False)
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
M = np.eye(2)
R = np.zeros((0, 2))
Z = np.zeros((2, 0))
S = np.zeros((0, 0))
v = np.ones(2)
def run_model():
    try:
        apply_reduced_inverse(M, R, Z, S, v, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_apply_reduced_inverse(M, R, Z, S, v, 1)
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
M = np.eye(2)
R = np.array([[1.0, 0.0]])
Z = np.array([[0.0], [1.0]])
S = np.eye(1)
v = np.ones(2)
def run_model():
    try:
        apply_reduced_inverse(M, R, Z, S, v, False)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_apply_reduced_inverse(M, R, Z, S, v, False)
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
