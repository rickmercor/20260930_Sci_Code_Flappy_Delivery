"""
Build the strength-of-connection graph of a level operator and partition its unknowns into aggregates by a deterministic greedy sweep.

Algebraic multigrid has no access to a mesh, so it must infer from the matrix alone which unknowns are tied together strongly enough that a smoother cannot resolve the error between them and a coarse representative must therefore be introduced. The device that encodes this inference is the strength-of-connection graph. An off-diagonal entry is declared a strong connection when its magnitude is at least a fixed fraction of the largest off-diagonal magnitude in its own row, the fraction being the strength threshold. Row-wise normalisation is what makes the criterion scale-free: it is insensitive to a rescaling of the whole matrix, and, more importantly for a multibody tangent operator whose joint stiffnesses span many orders of magnitude, it lets a compliant row keep its own strong neighbours instead of being swamped by the stiffest rows in the system. A low threshold keeps almost every connection and yields a conservative, expensive hierarchy with many levels and a high operator complexity; a high threshold coarsens aggressively, cheapens the setup, and degrades the convergence rate.

The strength relation as defined is not symmetric, because the row maxima of the two partners differ, so it is symmetrised before it is used to group unknowns: a pair is tied if either direction is strong. The unknowns are then partitioned into disjoint aggregates by a single greedy pass in ascending index. The first unassigned unknown encountered becomes the root of a new aggregate, and every still-unassigned unknown tied to it joins that aggregate; unknowns already claimed by an earlier aggregate are never reassigned. Scanning in a fixed index order is what makes the partition reproducible, which matters because every downstream quantity, including the coarse operator and ultimately the spectrum of the preconditioned system, inherits that choice. Isolated unknowns, having no strong ties, become singleton aggregates rather than being dropped, so the aggregation always covers the whole index set exactly once.

The aggregates are not yet a coarse grid. They define only the piecewise-constant tentative prolongator, one basis vector per aggregate, which the next step smooths into the actual transfer operator.

Returns
-------
np.ndarray of shape (n,), int: the aggregate index of every unknown, consecutively numbered from zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def build_strength_aggregates(matrix: np.ndarray, theta: float) -> np.ndarray:
    """Partition the unknowns of a level operator into aggregates.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric level operator of shape (n, n) with n >= 1.
    theta : float
        Strength-of-connection threshold, 0 < theta <= 1.

    Returns
    -------
    aggregates : np.ndarray
        Integer array of shape (n,) giving the aggregate index of every
        unknown. Aggregate indices are consecutive and start at zero, in the
        order the aggregates are created.
    """
    return aggregates  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_strength_aggregates(matrix: np.ndarray, theta: float) -> np.ndarray:
    # Local imports and nested helpers keep the oracle self-contained when the
    # harness executes it in isolation.
    import numpy as np

    def _strength_graph(matrix, theta):
        """Return the boolean strength-of-connection graph of a level operator."""
        magnitude = np.abs(matrix).copy()
        np.fill_diagonal(magnitude, 0.0)
        row_max = magnitude.max(axis=1)
        row_max = np.where(row_max > 0.0, row_max, 1.0)
        return (magnitude >= theta * row_max[:, None]) & (magnitude > 0.0)

    if not (isinstance(theta, (int, float)) and np.isfinite(theta)
            and 0.0 < float(theta) <= 1.0):
        raise ValueError("theta must be a finite number in the half-open interval (0, 1]")
    level = np.asarray(matrix, dtype=float)
    if level.ndim != 2 or level.shape[0] != level.shape[1] or level.shape[0] < 1:
        raise ValueError("matrix must be a square 2D array of order n >= 1")
    if not np.all(np.isfinite(level)):
        raise ValueError("matrix must be finite")

    n_dof = level.shape[0]
    strong = _strength_graph(level, float(theta))
    # Symmetrise: a pair is tied when either direction is strong.
    strong = strong | strong.T

    aggregates = -np.ones(n_dof, dtype=int)
    next_id = 0
    for i in range(n_dof):
        if aggregates[i] >= 0:
            continue
        neighbours = np.flatnonzero(strong[i] & (aggregates < 0))
        aggregates[i] = next_id
        aggregates[neighbours] = next_id
        next_id += 1

    return aggregates.astype(int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark level operator (normal scenario) ---
        {
            "setup": """import numpy as np
n = 60
i = np.arange(n)
base = np.exp(-0.4 * np.abs(i[:, None] - i[None, :]))
base = -0.5 * (base + base.T)
np.fill_diagonal(base, 0.0)
matrix = base + np.diag(2.0 + np.abs(base).sum(axis=1))
theta = 0.25
""",
            "call": "build_strength_aggregates(matrix, theta)",
            "gold_call": "_oracle_build_strength_aggregates(matrix, theta)",
        },
        # --- Valid: same operator at an aggressive threshold ---
        {
            "setup": """import numpy as np
n = 60
i = np.arange(n)
base = np.exp(-0.4 * np.abs(i[:, None] - i[None, :]))
base = -0.5 * (base + base.T)
np.fill_diagonal(base, 0.0)
matrix = base + np.diag(2.0 + np.abs(base).sum(axis=1))
theta = 0.75
""",
            "call": "build_strength_aggregates(matrix, theta)",
            "gold_call": "_oracle_build_strength_aggregates(matrix, theta)",
        },
        # --- Boundary: threshold of one keeps only the strongest connection per row ---
        {
            "setup": """import numpy as np
matrix = np.array([[4.0, -1.0, -0.2, 0.0],
                   [-1.0, 4.0, -1.0, -0.2],
                   [-0.2, -1.0, 4.0, -1.0],
                   [0.0, -0.2, -1.0, 4.0]])
theta = 1.0
""",
            "call": "build_strength_aggregates(matrix, theta)",
            "gold_call": "_oracle_build_strength_aggregates(matrix, theta)",
        },
        # --- Edge: purely diagonal operator, every unknown a singleton aggregate ---
        {
            "setup": """import numpy as np
matrix = np.diag(np.linspace(1.0, 5.0, 7))
theta = 0.25
""",
            "call": "build_strength_aggregates(matrix, theta)",
            "gold_call": "_oracle_build_strength_aggregates(matrix, theta)",
        },
        # --- Invalid: threshold outside its admissible interval ---
        {
            "setup": """import numpy as np
matrix = np.eye(5) * 3.0 - np.eye(5, k=1) - np.eye(5, k=-1)
def run_model():
    try:
        build_strength_aggregates(matrix, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_strength_aggregates(matrix, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-square operator ---
        {
            "setup": """import numpy as np
matrix = np.zeros((3, 5))
def run_model():
    try:
        build_strength_aggregates(matrix, 0.25)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_strength_aggregates(matrix, 0.25)
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
