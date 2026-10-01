"""
Solve the tangent system for the Newton increment of the nodal velocities, shifting the tangent onto the positive definite cone when it is not already there.

A hyperelastic tangent is indefinite whenever a material point is compressed or rotated enough, so the raw Newton system can return a direction along which the incremental potential rises. Adding the smallest convenient multiple of the identity that restores a Cholesky factorization keeps the direction a descent direction without changing where the residual vanishes.

Returns
-------
np.ndarray of shape (n_nodes, 2): the Newton increment of the nodal velocities.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_newton_direction(residual: "np.ndarray", hessian: "np.ndarray") -> "np.ndarray":
    """Solve the shifted tangent system for the Newton increment.

    The matrix is first replaced by its symmetric part. A Cholesky
    factorization of that symmetric part is attempted; if it fails, a multiple
    of the identity is added and the attempt is repeated. The multiple starts
    at one thousandth of the mean of the diagonal entries of the symmetric
    part, or, when that mean is not strictly positive, at one thousandth of the
    largest absolute entry of the symmetric part, and it is doubled on every
    further failure. The increment returned solves the first system that
    factorizes, with the negative of the residual as the right-hand side.

    Parameters
    ----------
    residual : "np.ndarray"
        Array of shape (n_nodes, 2) holding the gradient of the incremental
        potential with respect to the nodal velocities. Every entry must be
        finite.
    hessian : "np.ndarray"
        Array of shape (2 * n_nodes, 2 * n_nodes) holding the tangent matrix,
        with the two components of node i in rows and columns 2 i and 2 i + 1.
        Every entry must be finite.

    Returns
    -------
    direction : "np.ndarray"
        Array of shape (n_nodes, 2) holding the Newton increment of the nodal
        velocities, in m s^-1.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, if the shapes
        are mutually inconsistent, or if no shift within sixty doublings makes
        the matrix factorizable.
    """
    return direction  #placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_newton_direction(residual: "np.ndarray", hessian: "np.ndarray") -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    gradient = np.asarray(residual, dtype=float)
    if gradient.ndim != 2 or gradient.shape[1] != 2:
        raise ValueError("residual must have shape (n_nodes, 2)")
    n_nodes = gradient.shape[0]
    if n_nodes == 0:
        raise ValueError("residual must hold at least one node")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("residual must contain only finite entries")

    matrix = np.asarray(hessian, dtype=float)
    if matrix.shape != (2 * n_nodes, 2 * n_nodes):
        raise ValueError("hessian must have shape (2 * n_nodes, 2 * n_nodes)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("hessian must contain only finite entries")

    symmetric = 0.5 * (matrix + matrix.T)
    order = symmetric.shape[0]
    diagonal = np.arange(order)
    mean_diagonal = float(np.mean(symmetric[diagonal, diagonal]))
    if mean_diagonal > 0.0:
        base = 1.0e-3 * mean_diagonal
    else:
        largest = float(np.max(np.abs(symmetric)))
        base = 1.0e-3 * largest if largest > 0.0 else 1.0
    right_hand_side = -gradient.ravel()

    shift = 0.0
    for attempt in range(61):
        shifted = symmetric.copy()
        if shift > 0.0:
            shifted[diagonal, diagonal] += shift
        try:
            # The factorization is used only as the test for positive
            # definiteness; the increment itself comes from the shifted system.
            np.linalg.cholesky(shifted)
            increment = np.linalg.solve(shifted, right_hand_side)
        except np.linalg.LinAlgError:
            shift = base if attempt == 0 else 2.0 * shift
            continue
        if not np.all(np.isfinite(increment)):
            raise ValueError("the tangent system produced a non-finite increment")
        return increment.reshape(n_nodes, 2)

    raise ValueError("no shift within sixty doublings made the tangent factorizable")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    """Return list of test case specifications."""
    # Every case returns a plain float: the valid cases reduce the returned array
    # through a position weighted digest, the invalid cases return a status code.
    return [
        # --- Valid: a well conditioned mass dominated tangent (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(81)
n_nodes = 9
raw = rng.standard_normal((2 * n_nodes, 2 * n_nodes))
hessian = 0.02 * (raw @ raw.T) + np.diag(np.repeat(np.linspace(0.03, 0.19, n_nodes), 2))
residual = 0.01 * rng.standard_normal((n_nodes, 2))

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(solve_newton_direction(residual, hessian))",
            "gold_call": "pin(_oracle_solve_newton_direction(residual, hessian))",
        },
        # --- Valid: an indefinite tangent, which forces the shift to act ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(82)
n_nodes = 7
raw = rng.standard_normal((2 * n_nodes, 2 * n_nodes))
symmetric = 0.5 * (raw + raw.T)
hessian = symmetric + np.diag(np.repeat(np.linspace(0.5, 1.5, n_nodes), 2))
residual = 0.02 * rng.standard_normal((n_nodes, 2))

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(solve_newton_direction(residual, hessian))",
            "gold_call": "pin(_oracle_solve_newton_direction(residual, hessian))",
        },
        # --- Valid: the increment solves the system it was built from ---
        # For a positive definite tangent no shift is applied, so the increment
        # must satisfy the unshifted system and must point downhill against the
        # residual.
        {
            "setup": """import numpy as np
rng = np.random.default_rng(83)
n_nodes = 8
raw = rng.standard_normal((2 * n_nodes, 2 * n_nodes))
hessian = 0.05 * (raw @ raw.T) + np.diag(np.repeat(np.linspace(0.05, 0.2, n_nodes), 2))
residual = 0.03 * rng.standard_normal((n_nodes, 2))

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)

def descent(fn):
    direction = fn(residual, hessian)
    leftover = hessian @ direction.ravel() + residual.ravel()
    slope = float(direction.ravel() @ residual.ravel())
    flags = float(int(np.abs(leftover).max() < 1.0e-9 * np.abs(residual).max())
                  + 2 * int(slope < 0.0))
    return flags + pin(direction) + 5.0 * pin(leftover) + 1.0e3 * slope
""",
            "call": "descent(solve_newton_direction)",
            "gold_call": "descent(_oracle_solve_newton_direction)",
        },
        # --- Boundary: a vanishing residual, so the increment vanishes ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(84)
n_nodes = 5
raw = rng.standard_normal((2 * n_nodes, 2 * n_nodes))
hessian = 0.03 * (raw @ raw.T) + np.eye(2 * n_nodes) * 0.1
residual = np.zeros((n_nodes, 2))

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(solve_newton_direction(residual, hessian))",
            "gold_call": "pin(_oracle_solve_newton_direction(residual, hessian))",
        },
        # --- Edge: a singular tangent carrying an exactly empty node ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(85)
n_nodes = 6
raw = rng.standard_normal((2 * n_nodes, 2 * n_nodes))
hessian = 0.04 * (raw @ raw.T) + np.eye(2 * n_nodes) * 0.05
hessian[4, :] = 0.0
hessian[:, 4] = 0.0
hessian[5, :] = 0.0
hessian[:, 5] = 0.0
residual = 0.02 * rng.standard_normal((n_nodes, 2))
residual[2] = 0.0

def pin(values):
    flat = 1.0e6 * np.asarray(values, dtype=float).ravel()
    weight = np.arange(1.0, flat.size + 1.0) / (0.5 * flat.size * (flat.size + 1.0))
    moments = np.sqrt(np.sum(weight * flat ** 2)) + 0.5 * np.sum(weight * flat)
    return float(flat.size + moments)
""",
            "call": "pin(solve_newton_direction(residual, hessian))",
            "gold_call": "pin(_oracle_solve_newton_direction(residual, hessian))",
        },
        # --- Invalid: a tangent whose size does not match the residual ---
        {
            "setup": """import numpy as np
residual = np.zeros((4, 2))
hessian = np.eye(6)
def run_model():
    try:
        solve_newton_direction(residual, hessian)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_newton_direction(residual, hessian)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a residual carrying a non-finite entry ---
        {
            "setup": """import numpy as np
residual = np.zeros((4, 2))
residual[1, 0] = np.inf
hessian = np.eye(8)
def run_model():
    try:
        solve_newton_direction(residual, hessian)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_newton_direction(residual, hessian)
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
