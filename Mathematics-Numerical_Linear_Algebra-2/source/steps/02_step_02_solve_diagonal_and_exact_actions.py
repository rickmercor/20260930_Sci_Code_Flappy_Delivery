"""
Given the SPD block $E$, constraint matrix $C$, and sketch matrix $\Omega$, compute the inverse actions required by the subsequent randomized preconditioning subproblems and return the resulting state.

The randomized preconditioning construction uses an SPD matrix block together with a constraint matrix and a fixed sketch. For this benchmark, the required actions are obtained through direct numerical linear solves so that the resulting state is deterministic.

Returns
-------
Tuple[np.ndarray, np.ndarray] — $(Y,Y_D)$, both shape $(p,k)$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_diagonal_and_exact_actions(
    E: np.ndarray,
    C: np.ndarray,
    Omega: np.ndarray,
) -> tuple:
    r"""
    Compute the numerical action states required by the subsequent
    randomized preconditioning subproblems.

    Raises
    ------
    ValueError
        If $E$ is not square, symmetric, and SPD, $C$ has an incompatible
        shape with $E$, or $\Omega$ has an incompatible shape with $C$.
    """
    return Y, Y_D

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_diagonal_and_exact_actions(
    E: np.ndarray,
    C: np.ndarray,
    Omega: np.ndarray,
) -> tuple:
    E = np.asarray(E, dtype=float)
    C = np.asarray(C, dtype=float)
    Omega = np.asarray(Omega, dtype=float)

    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError("E must be square")

    if not np.allclose(E, E.T, rtol=0.0, atol=1e-12):
        raise ValueError("E must be symmetric")

    try:
        np.linalg.cholesky(E)
    except np.linalg.LinAlgError as exc:
        raise ValueError("E must be SPD") from exc

    if C.ndim != 2 or C.shape[0] != E.shape[0]:
        raise ValueError("C shape is incompatible with E")

    if Omega.ndim != 2 or Omega.shape[0] != C.shape[1]:
        raise ValueError("Omega shape is incompatible with C")

    E_D = np.diag(np.diag(E))
    C_Omega = C @ Omega

    Y = np.linalg.solve(E, C_Omega)
    Y_D = np.linalg.solve(E_D, C_Omega)

    return Y.astype(float), Y_D.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: ordinary dense SPD case ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(11)
p, m, k = 5, 9, 3
G = rng.standard_normal((p, p))
E = G @ G.T + p * np.eye(p)
C = rng.standard_normal((p, m))
Omega = rng.standard_normal((m, k))
""",
            "call": "np.concatenate([a.ravel() for a in solve_diagonal_and_exact_actions(E, C, Omega)])",
            "gold_call": "np.concatenate([a.ravel() for a in _oracle_solve_diagonal_and_exact_actions(E, C, Omega)])",
        },

        # --- Valid: E already diagonal ---
        {
            "setup": """import numpy as np
E = np.diag([2.0, 3.0, 5.0])
C = np.array([
    [1.0, 0.0, 2.0, 1.0],
    [0.0, 1.0, 1.0, -1.0],
    [1.0, 1.0, 0.0, 2.0]
])
Omega = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [-1.0, 0.5]
])
""",
            "call": "np.concatenate([a.ravel() for a in solve_diagonal_and_exact_actions(E, C, Omega)])",
            "gold_call": "np.concatenate([a.ravel() for a in _oracle_solve_diagonal_and_exact_actions(E, C, Omega)])",
        },

        # --- Invalid: mismatched C ---
        {
            "setup": """import numpy as np
E = np.eye(4)
C = np.random.default_rng(0).standard_normal((3, 6))
Omega = np.random.default_rng(0).standard_normal((6, 2))

def run_model():
    try:
        solve_diagonal_and_exact_actions(E, C, Omega)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_solve_diagonal_and_exact_actions(E, C, Omega)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },

        # --- Invalid: non-SPD E ---
        {
            "setup": """import numpy as np
E = np.array([[1.0, 2.0], [2.0, 1.0]])
C = np.random.default_rng(0).standard_normal((2, 5))
Omega = np.random.default_rng(0).standard_normal((5, 2))

def run_model():
    try:
        solve_diagonal_and_exact_actions(E, C, Omega)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_solve_diagonal_and_exact_actions(E, C, Omega)
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
