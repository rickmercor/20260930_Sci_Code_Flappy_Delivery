"""
Assemble the block-diagonal Schur complement approximation, add the dense low-rank correction on the cut constraints, and regularise it against constraint redundancy.

The exact Schur complement of the saddle-point system is the constraint Jacobian times the inverse of the mechanical block times the Jacobian transpose. The preconditioner replaces the inverse by the V-cycle operator and then sparsifies the result. Because a cluster's Jacobian rows are supported only on the degrees of freedom of the bodies that cluster touches, restricting the V-cycle to those degrees of freedom and contracting it with the cluster's rows gives exactly the small dense local Schur complement of that cluster: the cluster's rows times the V-cycle operator times their transpose. Collecting these blocks along the diagonal, one per cluster, gives the sparsified approximation. Its blocks are at most a few constraints across, so each is factorised once during setup and applied in constant time thereafter.

The block-diagonal form discards the couplings that run around kinematic loops, and those are restored explicitly rather than ignored. The constraints on the cut joints identified previously carry a dense correction, their rows times the V-cycle operator times their transpose, added on top of the block-diagonal entries they already have in the cut part of the matrix. Its rank equals the number of cut constraints, which for a connected mechanism is set by the number of independent kinematic cycles, so the correction stays small however large the mechanism grows: the resulting object is block diagonal plus low rank, and inverting it costs one small dense solve on top of the per-cluster solves.

A last ingredient is needed because real models are over-constrained. Redundant constraints make the Jacobian rank-deficient and the exact Schur complement singular, and the sparsified approximation inherits that singularity, so the preconditioner solve would break down. A Tikhonov shift proportional to the identity is added, with the coefficient set relative to the matrix one-norm so that it scales with the problem rather than being an absolute number; the framework uses a shift eight orders of magnitude below the one-norm. The shift perturbs only the search direction of the Newton iteration, not the nonlinear equations being solved, so at convergence the multiplier increment tends to zero and with it the perturbation, which is why a device this crude does not damage long-term physical accuracy. It is not free at the level of the linear algebra, though: it moves the spectrum of the preconditioned system, which is one of the effects the final measurement exposes.

Returns
-------
np.ndarray of shape (m, m), float: the regularised block-diagonal-plus-low-rank Schur complement approximation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_schur_preconditioner(jacobian: np.ndarray, vcycle_operator: np.ndarray,
                                  partition: np.ndarray,
                                  eps_rel: float) -> np.ndarray:
    """Assemble the regularised Schur complement approximation.

    Parameters
    ----------
    jacobian : np.ndarray
        Constraint Jacobian of shape (m, n).
    vcycle_operator : np.ndarray
        V-cycle approximate inverse of the mechanical block, shape (n, n).
    partition : np.ndarray
        Integer array of shape (m, 2) holding cluster indices in column zero
        and cut flags in column one.
    eps_rel : float
        Tikhonov shift relative to the matrix one-norm, eps_rel >= 0.

    Returns
    -------
    schur_prec : np.ndarray
        Symmetric matrix of shape (m, m): block diagonal over the clusters,
        plus the dense correction on the cut constraints, plus the shift.
    """
    return schur_prec  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_assemble_schur_preconditioner(jacobian: np.ndarray, vcycle_operator: np.ndarray,
                                          partition: np.ndarray,
                                          eps_rel: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(eps_rel, (int, float)) and np.isfinite(eps_rel)
            and float(eps_rel) >= 0.0):
        raise ValueError("eps_rel must be a finite number >= 0")
    matrix = np.asarray(jacobian, dtype=float)
    cycle = np.asarray(vcycle_operator, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("jacobian must be a non-empty 2D array")
    if cycle.ndim != 2 or cycle.shape[0] != cycle.shape[1]:
        raise ValueError("vcycle_operator must be a square 2D array")
    if cycle.shape[0] != matrix.shape[1]:
        raise ValueError("vcycle_operator order must match the column count of jacobian")
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(cycle))):
        raise ValueError("jacobian and vcycle_operator must be finite")

    table = np.asarray(partition)
    if table.ndim != 2 or table.shape != (matrix.shape[0], 2):
        raise ValueError("partition must have shape (m, 2) with m rows of jacobian")
    if not np.issubdtype(table.dtype, np.integer):
        raise ValueError("partition must be an integer array")
    labels = table[:, 0]
    cut = table[:, 1]
    if labels.min() != 0 or not np.array_equal(np.unique(labels),
                                               np.arange(labels.max() + 1)):
        raise ValueError("cluster indices must be consecutive integers starting at zero")
    if not np.all(np.isin(cut, (0, 1))):
        raise ValueError("cut flags must be zero or one")

    n_rows = matrix.shape[0]
    schur_prec = np.zeros((n_rows, n_rows), dtype=float)

    # Block-diagonal part: one local Schur complement per cluster.
    for cluster in range(int(labels.max()) + 1):
        rows = np.flatnonzero(labels == cluster)
        block = matrix[rows]
        schur_prec[np.ix_(rows, rows)] = block @ cycle @ block.T

    # Dense low-rank correction reinstating the couplings across kinematic loops.
    cut_rows = np.flatnonzero(cut == 1)
    if cut_rows.size > 0:
        block = matrix[cut_rows]
        schur_prec[np.ix_(cut_rows, cut_rows)] += block @ cycle @ block.T

    # Tikhonov shift set relative to the one-norm of the assembled matrix.
    shift = float(eps_rel) * float(np.linalg.norm(schur_prec, 1))
    schur_prec = schur_prec + shift * np.eye(n_rows)

    return schur_prec

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: clustered constraints with a cut set (normal scenario) ---
        {
            "setup": """import numpy as np
n_bodies, n_dof = 8, 48
edges = [tuple(sorted((b, (b + 1) % n_bodies))) for b in range(n_bodies)]
rows = []
for k, (i, j) in enumerate(edges):
    for d in range(3):
        r = np.zeros(n_dof)
        r[6 * i + d] = 1.0
        r[6 * j + d] = -1.0
        r[6 * i + 3 + (d + 1) % 3] = np.sin(k + d)
        rows.append(r)
jacobian = np.array(rows)
m = jacobian.shape[0]
base = np.diag(np.linspace(1.0, 4.0, n_dof))
vcycle_operator = np.linalg.inv(base + 0.05 * np.ones((n_dof, n_dof)))
labels = np.arange(m) // 6
cut = np.zeros(m, dtype=int)
cut[-3:] = 1
partition = np.column_stack([labels, cut]).astype(int)
eps_rel = 1.0e-8
""",
            "call": "assemble_schur_preconditioner(jacobian, vcycle_operator, partition, eps_rel)",
            "gold_call": "_oracle_assemble_schur_preconditioner(jacobian, vcycle_operator, partition, eps_rel)",
        },
        # --- Valid: no cut constraints, so the correction is empty ---
        {
            "setup": """import numpy as np
n_dof = 24
jacobian = np.zeros((9, n_dof))
for r in range(9):
    jacobian[r, r] = 1.0
    jacobian[r, r + 12] = -1.0
vcycle_operator = np.diag(np.linspace(0.5, 2.0, n_dof))
labels = np.arange(9) // 3
cut = np.zeros(9, dtype=int)
partition = np.column_stack([labels, cut]).astype(int)
eps_rel = 1.0e-8
""",
            "call": "assemble_schur_preconditioner(jacobian, vcycle_operator, partition, eps_rel)",
            "gold_call": "_oracle_assemble_schur_preconditioner(jacobian, vcycle_operator, partition, eps_rel)",
        },
        # --- Boundary: zero shift, so no regularisation is applied ---
        {
            "setup": """import numpy as np
n_dof = 12
jacobian = np.zeros((4, n_dof))
for r in range(4):
    jacobian[r, r] = 1.0
    jacobian[r, r + 6] = -0.5
vcycle_operator = np.eye(n_dof) * 0.25
partition = np.column_stack([np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])]).astype(int)
eps_rel = 0.0
""",
            "call": "assemble_schur_preconditioner(jacobian, vcycle_operator, partition, eps_rel)",
            "gold_call": "_oracle_assemble_schur_preconditioner(jacobian, vcycle_operator, partition, eps_rel)",
        },
        # --- Edge: redundant duplicated constraints and a heavy shift ---
        {
            "setup": """import numpy as np
n_dof = 12
row = np.zeros(n_dof)
row[0] = 1.0
row[6] = -1.0
jacobian = np.vstack([row, row, row * 2.0])
vcycle_operator = np.eye(n_dof) * 0.5
partition = np.column_stack([np.array([0, 0, 1]), np.array([0, 0, 1])]).astype(int)
eps_rel = 1.0e-2
""",
            "call": "assemble_schur_preconditioner(jacobian, vcycle_operator, partition, eps_rel)",
            "gold_call": "_oracle_assemble_schur_preconditioner(jacobian, vcycle_operator, partition, eps_rel)",
        },
        # --- Invalid: negative Tikhonov shift ---
        {
            "setup": """import numpy as np
jacobian = np.eye(3, 6)
vcycle_operator = np.eye(6)
partition = np.column_stack([np.arange(3), np.zeros(3, dtype=int)]).astype(int)
def run_model():
    try:
        assemble_schur_preconditioner(jacobian, vcycle_operator, partition, -1.0e-8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_schur_preconditioner(jacobian, vcycle_operator, partition, -1.0e-8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: cluster indices that are not consecutive from zero ---
        {
            "setup": """import numpy as np
jacobian = np.eye(3, 6)
vcycle_operator = np.eye(6)
partition = np.column_stack([np.array([0, 2, 3]), np.zeros(3, dtype=int)]).astype(int)
def run_model():
    try:
        assemble_schur_preconditioner(jacobian, vcycle_operator, partition, 1.0e-8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_schur_preconditioner(jacobian, vcycle_operator, partition, 1.0e-8)
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
