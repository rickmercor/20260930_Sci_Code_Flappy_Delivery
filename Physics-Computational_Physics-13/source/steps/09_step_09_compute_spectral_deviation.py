"""
Form the block-triangular preconditioned saddle-point operator and measure how far its nonzero spectrum spreads away from unity.

The convergence rate of a Krylov method on the preconditioned system is governed by how tightly the spectrum clusters, and the target of the design is a cluster around one. The preconditioner is block lower triangular: the mechanical block is the operator the V-cycle inverts, the block below it is the constraint Jacobian, and the constraint block carries the negated Schur approximation. The negation is not a convention that can be flipped at will. The exact Schur complement of the saddle-point system is the negative of the contraction of the inverse mechanical block with the Jacobian, so a constraint block built as an approximation to the positive contraction must carry the minus sign in order to reproduce it; that sign is what places the preconditioned spectrum around one, and taking the block positive would preserve the eigenvalue magnitudes of interest but move the cluster away from the value the theory predicts. The preconditioner itself is neither symmetric nor definite: it is nonsymmetric whenever the Jacobian is nonzero, and it is indefinite because a multiplier-only direction returns the negative of a positive definite quadratic form. It therefore calls for a nonsymmetric Krylov method; it is the block-diagonal variant, carrying the mechanical block and the positive Schur approximation on its diagonal, that is symmetric positive definite and pairs with a short-recurrence symmetric solver.

Two features make the measurement cheap and unambiguous. First, the inverse of a block lower triangular matrix is available in closed form, and it involves the inverse of the mechanical block only through the V-cycle operator, which is already the approximate inverse rather than something requiring a solve; only the Schur approximation, small and already factorised in practice, has to be inverted. Second, the preconditioned operator is not symmetric, so its eigenvalues need not be real: the analysis of this preconditioner produces a quadratic characteristic relation per invariant direction whose discriminant can be negative, and the resulting conjugate pairs are genuine, not numerical artefacts. The deviation from unity is therefore a complex modulus, and taking real parts would silently understate the spread.

Directions in the null space of the preconditioned operator are excluded. They arise only when the constraint Jacobian is row-rank deficient, they lie outside the Krylov subspace generated from any consistent right-hand side, and they do not influence the convergence rate; the theory bounds the nonzero eigenvalues only. Screening them by magnitude relative to the largest eigenvalue keeps the criterion scale-free. The number returned is the worst-case observed deviation, the quantity the spectral theory of the framework claims to bound.

Returns
-------
float: the largest modulus of the deviation from unity over the nonzero spectrum of the preconditioned saddle-point operator, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_spectral_deviation(stiffness: np.ndarray, jacobian: np.ndarray,
                               vcycle_operator: np.ndarray,
                               schur_prec: np.ndarray) -> float:
    """Measure the worst deviation from unity of the preconditioned spectrum.

    Parameters
    ----------
    stiffness : np.ndarray
        Mechanical block of the saddle-point system, shape (n, n).
    jacobian : np.ndarray
        Constraint Jacobian, shape (m, n).
    vcycle_operator : np.ndarray
        V-cycle approximate inverse of the mechanical block, shape (n, n).
    schur_prec : np.ndarray
        Nonsingular Schur complement approximation, shape (m, m).

    Returns
    -------
    deviation : float
        Largest modulus of one minus an eigenvalue, over the eigenvalues of
        the preconditioned operator that are not numerically zero, as a native
        Python float.
    """
    return deviation  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_spectral_deviation(stiffness: np.ndarray, jacobian: np.ndarray,
                                       vcycle_operator: np.ndarray,
                                       schur_prec: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    mechanical = np.asarray(stiffness, dtype=float)
    constraint = np.asarray(jacobian, dtype=float)
    cycle = np.asarray(vcycle_operator, dtype=float)
    schur = np.asarray(schur_prec, dtype=float)

    if mechanical.ndim != 2 or mechanical.shape[0] != mechanical.shape[1]:
        raise ValueError("stiffness must be a square 2D array")
    if constraint.ndim != 2 or constraint.shape[1] != mechanical.shape[0]:
        raise ValueError("jacobian must have as many columns as the order of stiffness")
    if cycle.shape != mechanical.shape:
        raise ValueError("vcycle_operator must have the same shape as stiffness")
    if schur.ndim != 2 or schur.shape != (constraint.shape[0], constraint.shape[0]):
        raise ValueError("schur_prec must be square of order the row count of jacobian")
    for name, value in (("stiffness", mechanical), ("jacobian", constraint),
                        ("vcycle_operator", cycle), ("schur_prec", schur)):
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite")

    n_dof, n_con = mechanical.shape[0], constraint.shape[0]

    system = np.zeros((n_dof + n_con, n_dof + n_con), dtype=float)
    system[:n_dof, :n_dof] = mechanical
    system[:n_dof, n_dof:] = constraint.T
    system[n_dof:, :n_dof] = constraint

    # Closed-form inverse of the block lower triangular preconditioner.
    try:
        schur_inverse = np.linalg.inv(schur)
    except np.linalg.LinAlgError as exc:
        raise ValueError("schur_prec must be nonsingular") from exc

    preconditioner_inverse = np.zeros_like(system)
    preconditioner_inverse[:n_dof, :n_dof] = cycle
    preconditioner_inverse[n_dof:, :n_dof] = schur_inverse @ constraint @ cycle
    preconditioner_inverse[n_dof:, n_dof:] = -schur_inverse

    eigenvalues = np.linalg.eigvals(preconditioner_inverse @ system)
    magnitude = np.abs(eigenvalues)
    largest = float(np.max(magnitude))
    if largest == 0.0:
        raise ValueError("the preconditioned operator vanishes identically")
    nonzero = eigenvalues[magnitude > 1.0e-10 * largest]
    if nonzero.size == 0:
        raise ValueError("the preconditioned operator has no nonzero eigenvalue")

    return float(np.max(np.abs(nonzero - 1.0)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: inexact V-cycle with the matching exact Schur complement ---
        {
            "setup": """import numpy as np
n, m = 12, 4
i = np.arange(n)
stiffness = np.diag(2.0 + i) + 0.1 * np.exp(-np.abs(i[:, None] - i[None, :]))
stiffness = 0.5 * (stiffness + stiffness.T)
jacobian = np.zeros((m, n))
for r in range(m):
    jacobian[r, r] = 1.0
    jacobian[r, r + 4] = -0.5
    jacobian[r, r + 8] = 0.25
    jacobian[r, (r + 1) % 4] += 0.4
vcycle_operator = np.diag(1.0 / np.diag(stiffness))
schur_prec = jacobian @ vcycle_operator @ jacobian.T
""",
            "call": "compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)",
            "gold_call": "_oracle_compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)",
        },
        # --- Valid: block-diagonal Schur surrogate discarding the coupling ---
        {
            "setup": """import numpy as np
n, m = 12, 4
i = np.arange(n)
stiffness = np.diag(2.0 + i) + 0.1 * np.exp(-np.abs(i[:, None] - i[None, :]))
stiffness = 0.5 * (stiffness + stiffness.T)
jacobian = np.zeros((m, n))
for r in range(m):
    jacobian[r, r] = 1.0
    jacobian[r, r + 4] = -0.5
    jacobian[r, r + 8] = 0.25
    jacobian[r, (r + 1) % 4] += 0.4
vcycle_operator = np.diag(1.0 / np.diag(stiffness))
full = jacobian @ vcycle_operator @ jacobian.T
schur_prec = np.diag(np.diag(full))
""",
            "call": "compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)",
            "gold_call": "_oracle_compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)",
        },
        # --- Boundary: heavy Tikhonov shift dominating the constraint block ---
        {
            "setup": """import numpy as np
n, m = 10, 4
stiffness = np.diag(np.linspace(1.0, 5.0, n))
jacobian = np.zeros((m, n))
for r in range(m):
    jacobian[r, r] = 1.0
    jacobian[r, r + 5] = -1.0
    jacobian[r, (r + 2) % n] += 0.3
vcycle_operator = np.diag(1.0 / np.diag(stiffness)) * 0.8
full = jacobian @ vcycle_operator @ jacobian.T
schur_prec = full + 0.5 * np.linalg.norm(full, 1) * np.eye(m)
""",
            "call": "compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)",
            "gold_call": "_oracle_compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)",
        },
        # --- Edge: rank-deficient Jacobian producing null directions to screen out ---
        {
            "setup": """import numpy as np
n, m = 8, 3
stiffness = np.diag(np.linspace(2.0, 6.0, n)) + 0.4 * np.eye(n, k=1) + 0.4 * np.eye(n, k=-1)
jacobian = np.zeros((m, n))
jacobian[0, 0] = 1.0
jacobian[0, 4] = -1.0
jacobian[1] = jacobian[0]
jacobian[2, 1] = 1.0
jacobian[2, 5] = -1.0
vcycle_operator = np.diag(1.0 / np.diag(stiffness))
schur_prec = jacobian @ vcycle_operator @ jacobian.T + 1.0e-3 * np.eye(m)
""",
            "call": "compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)",
            "gold_call": "_oracle_compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)",
        },
        # --- Invalid: singular Schur approximation ---
        {
            "setup": """import numpy as np
n, m = 6, 2
stiffness = np.eye(n) * 2.0
jacobian = np.zeros((m, n))
jacobian[0, 0] = 1.0
jacobian[1, 0] = 1.0
vcycle_operator = np.eye(n) * 0.5
schur_prec = jacobian @ vcycle_operator @ jacobian.T
def run_model():
    try:
        compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: Jacobian column count inconsistent with the mechanical block ---
        {
            "setup": """import numpy as np
stiffness = np.eye(6)
jacobian = np.ones((2, 5))
vcycle_operator = np.eye(6)
schur_prec = np.eye(2)
def run_model():
    try:
        compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_spectral_deviation(stiffness, jacobian, vcycle_operator, schur_prec)
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
