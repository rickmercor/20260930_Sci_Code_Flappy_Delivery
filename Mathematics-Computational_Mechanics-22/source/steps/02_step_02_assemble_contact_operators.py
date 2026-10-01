"""
Keep the interaction energy, reference operator, and full operator distinct.

The ordered contact rows induce a displacement energy through their row Gram

matrix. That energy augments the condensed core to form the symmetric reference

operator. A separate Newton correction may be nonsymmetric and is admitted only

to the complete residual operator. Field-of-values arguments for the complete

operator are driven by its symmetric part, so that part is reported as a fourth

object; it is the reference operator plus the symmetric part of the correction,

never a symmetrization of a quantity that already dropped the correction.

Return all four objects so the assembly identities remain observable; empty

interaction data must produce an exact zero energy matrix without suppressing

the complete correction.

Returns
-------
tuple of four finite float np.ndarray values of shape (n, n): interaction row Gram matrix, symmetric reference, complete operator, symmetric part
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def assemble_contact_operators(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Return interaction energy, symmetric reference, full and symmetric part.

    With core $M$, ordered rows $U$, and correction $N$, the returned objects
    are the interaction energy $U^\top U$, the symmetric reference
    $H=M+U^\top U$, the complete operator $A=H+N$, and the symmetric part
    $\tfrac{1}{2}(A+A^\top)=H+\tfrac{1}{2}(N+N^\top)$, in that order. The
    interaction energy and the reference are symmetrized by averaging with
    their transposes; the complete operator is returned unsymmetrized.

    Raises ValueError unless all inputs are finite; condensed_core is a
    nonempty symmetric positive-definite square matrix; interaction_rows has
    row-oriented shape (m, n), including (0, n) and m greater than n; and
    correction has shape (n, n). The correction need not be symmetric.

    Parameters
    ----------
    condensed_core : np.ndarray
        Symmetric positive-definite displacement core of shape (n, n).
    interaction_rows : np.ndarray
        Ordered row-oriented interaction factor of shape (m, n).
    correction : np.ndarray
        Complete-system correction of shape (n, n).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Interaction energy, symmetric reference operator, complete operator,
        and the symmetric part of the complete operator, each of shape (n, n).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_contact_operators(
    condensed_core: np.ndarray,
    interaction_rows: np.ndarray,
    correction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference four-way contact assembly."""
    core = np.asarray(condensed_core, dtype=float)
    rows = np.asarray(interaction_rows, dtype=float)
    extra = np.asarray(correction, dtype=float)
    if core.ndim != 2 or core.shape[0] != core.shape[1] or core.shape[0] == 0:
        raise ValueError("condensed_core must be a nonempty square matrix")
    n_dof = core.shape[0]
    if rows.ndim != 2 or rows.shape[1] != n_dof:
        raise ValueError("interaction_rows must have shape (m, n)")
    if extra.ndim != 2 or extra.shape != (n_dof, n_dof):
        raise ValueError("correction must have shape (n, n)")
    if not all(np.all(np.isfinite(x)) for x in (core, rows, extra)):
        raise ValueError("all inputs must be finite")
    if not np.allclose(core, core.T, rtol=0.0, atol=1e-12):
        raise ValueError("condensed_core must be symmetric")
    try:
        np.linalg.cholesky(core)
    except np.linalg.LinAlgError as exc:
        raise ValueError("condensed_core must be positive definite") from exc

    interaction_energy = rows.T @ rows
    interaction_energy = 0.5 * (interaction_energy + interaction_energy.T)
    reference = core + interaction_energy
    reference = 0.5 * (reference + reference.T)
    full_operator = reference + extra
    symmetric_part = reference + 0.5 * (extra + extra.T)
    return interaction_energy, reference, full_operator, symmetric_part

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return contacted, contact-free, wide, skew, and invalid cases."""
    return [
        {
            "setup": """import numpy as np
M = np.array([[4.0, -0.4, 0.1], [-0.4, 2.0, -0.2], [0.1, -0.2, 1.0]])
U = np.array([[3.0, 0.0, 0.0], [0.0, 1.5, 0.4]])
N = np.array([[0.1, 0.02, 0.0], [-0.03, -0.04, 0.01], [0.0, -0.02, 0.03]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in assemble_contact_operators(M, U, N)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_assemble_contact_operators(M, U, N)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[2.0, -0.25], [-0.25, 1.0]])
U = np.zeros((0, 2))
N = np.array([[.1, .2], [-.3, .05]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in assemble_contact_operators(M, U, N)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_assemble_contact_operators(M, U, N)])",
        },
        {
            "setup": """import numpy as np
M = np.diag([6.0, 2.0])
U = np.array([[1.0, -1.0], [-.5, 2.0]])
N = np.array([[0.0, 2.0], [-0.5, 0.0]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in assemble_contact_operators(M, U, N)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_assemble_contact_operators(M, U, N)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[5.0, .3], [.3, 2.0]])
U = np.array([[1.0, 0.0], [0.0, 2.0], [-1.5, .5], [.25, -.75]])
N = np.array([[0.0, 1.2], [-1.2, 0.0]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in assemble_contact_operators(M, U, N)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_assemble_contact_operators(M, U, N)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[40.5,-1.5,.2,0,.25],[-1.5,12.666666666666666,-.633333333333333,.433333333333333,-.25],[.2,-.633333333333333,3.166666666666667,-.866666666666667,.1],[0,.433333333333333,-.866666666666667,1.866666666666667,-.1],[.25,-.25,.1,-.1,.925]])
U = np.array([[8,0,0,0,0],[0,4,0,0,0],[0,0,2,.2,0],[0,0,.1,1.5,1.2]], dtype=float)
N = np.array([[.08,.02,0,0,0],[-.01,-.04,.03,0,0],[0,-.02,.05,.01,0],[0,0,-.03,-.02,.04],[.01,0,0,-.02,.03]])
""",
            "call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in assemble_contact_operators(M, U, N)])",
            "gold_call": "np.concatenate([np.asarray(x, dtype=float).ravel() for x in _oracle_assemble_contact_operators(M, U, N)])",
        },
        {
            "setup": """import numpy as np
M = np.array([[1.0, 2.0], [2.0, 1.0]])
U = np.zeros((0, 2))
N = np.zeros((2, 2))
def run_model():
    try:
        assemble_contact_operators(M, U, N)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_contact_operators(M, U, N)
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
U = np.ones((2, 3))
N = np.ones((3, 2))
def run_model():
    try:
        assemble_contact_operators(M, U, N)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_contact_operators(M, U, N)
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
M = np.array([[2.0, 0.3], [0.1, 1.0]])
U = np.array([[1.0, 0.0]])
N = np.zeros((2, 2))
def run_model():
    try:
        assemble_contact_operators(M, U, N)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_contact_operators(M, U, N)
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
