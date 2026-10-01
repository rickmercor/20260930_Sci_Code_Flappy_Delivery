"""
Apply the irreversible bond-history and phase update.

The history variable stores the largest driving force reached by each bond,

so unloading cannot heal damage. Given $Y_c>0$, the update is



$$

\mathcal{Y}^{n+1}_{kn}=\max(\mathcal{Y}^{n}_{kn},Y_{kn}),\qquad

s^{n+1}_{kn}=\min\left(1,

\frac{\mathcal{Y}^{n+1}_{kn}}{\mathcal{Y}^{n+1}_{kn}+Y_c}\right).

$$



The matrices are symmetric because a geometric bond has no preferred

orientation, and their diagonals are zero because self-bonds are excluded.

Returns
-------
Two float arrays of shape (N, N): updated history in MPa and phase in [0, 1].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def update_bond_phase_history(
    trial_driving_force: np.ndarray,
    previous_history: np.ndarray,
    critical_driving_force: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Update irreversible history and the bond phase field.

    Raises ``ValueError`` unless the two input matrices have the same square
    shape of order at least two, are finite, nonnegative, symmetric, and have
    zero diagonals, and ``critical_driving_force`` is finite and strictly
    positive.

    Parameters
    ----------
    trial_driving_force : np.ndarray
        Symmetric $(N,N)$ trial energy-density matrix in MPa.
    previous_history : np.ndarray
        Symmetric $(N,N)$ previous maximum energy-density matrix in MPa.
    critical_driving_force : float
        Positive threshold $Y_c$ in MPa.

    Returns
    -------
    history_new : np.ndarray
        Symmetric $(N,N)$ irreversible history in MPa.
    phase_new : np.ndarray
        Symmetric $(N,N)$ dimensionless phase field in $[0,1]$.
    """
    return NotImplemented

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _validated_bond_matrix(values, name):
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"{name} must be a square matrix of order at least two")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must have a zero diagonal")
    return matrix
def _oracle_update_bond_phase_history(
    trial_driving_force: np.ndarray,
    previous_history: np.ndarray,
    critical_driving_force: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference irreversible maximum-history update."""
    trial = _validated_bond_matrix(trial_driving_force, "trial_driving_force")
    previous = _validated_bond_matrix(previous_history, "previous_history")
    if trial.shape != previous.shape:
        raise ValueError(
            "trial_driving_force and previous_history must have equal shapes"
        )
    if not np.isfinite(critical_driving_force) or critical_driving_force <= 0.0:
        raise ValueError("critical_driving_force must be positive and finite")
    history_new = np.maximum(previous, trial)
    phase_new = history_new / (history_new + float(critical_driving_force))
    np.fill_diagonal(history_new, 0.0)
    np.fill_diagonal(phase_new, 0.0)
    return history_new, np.minimum(1.0, phase_new)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return loading, unloading, and invalid-asymmetry cases."""
    return [
        {
            "setup": """import numpy as np
trial_driving_force = np.array([[0.0,0.30,0.02],[0.30,0.0,0.12],[0.02,0.12,0.0]])
previous_history = np.array([[0.0,0.10,0.08],[0.10,0.0,0.04],[0.08,0.04,0.0]])
critical_driving_force = 0.25
""",
            "call": "update_bond_phase_history(trial_driving_force, previous_history, critical_driving_force)",
            "gold_call": "_oracle_update_bond_phase_history(trial_driving_force, previous_history, critical_driving_force)",
        },
        {
            "setup": """import numpy as np
trial_driving_force = np.zeros((2, 2))
previous_history = np.array([[0.0,0.5],[0.5,0.0]])
critical_driving_force = 1e-12
""",
            "call": "update_bond_phase_history(trial_driving_force, previous_history, critical_driving_force)",
            "gold_call": "_oracle_update_bond_phase_history(trial_driving_force, previous_history, critical_driving_force)",
        },
        {
            "setup": """import numpy as np
trial_driving_force = np.array([[0.0,0.2],[0.1,0.0]])
previous_history = np.zeros((2, 2))
critical_driving_force = 0.25
def run_model():
    try:
        update_bond_phase_history(trial_driving_force, previous_history, critical_driving_force)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_bond_phase_history(trial_driving_force, previous_history, critical_driving_force)
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
