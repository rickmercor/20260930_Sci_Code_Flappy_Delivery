"""
Measure the sharpest spectral-equivalence constant between the multigrid V-cycle operator and the effective stiffness it approximates.

The convergence theory of the preconditioner rests on a single scalar that quantifies how faithfully the V-cycle reproduces the inverse of the effective stiffness. The statement is a two-sided spectral containment: the effective stiffness is bounded below by one minus that scalar times the operator the V-cycle inverts, and above by one plus the same scalar times it. Writing the V-cycle operator as the inverse of an implicit operator, the containment is equivalent to saying that every eigenvalue of the V-cycle operator composed with the effective stiffness lies within that distance of one. The sharpest admissible constant is therefore the largest deviation from unity over the spectrum of that composed operator. A value near zero means the cycle is nearly an exact solve; a value approaching one means some mode is barely reduced at all, and the theory that consumes this constant degenerates.

Two properties make the measurement well posed. First, both the effective stiffness and the V-cycle operator are symmetric, so their product, while not symmetric itself, is similar to a symmetric matrix and has a real spectrum. Second, positive definiteness of the stiffness lets that similarity be realised explicitly through a Cholesky factor: conjugating the product by the transpose of the factor produces the symmetric matrix whose eigenvalues are the ones sought, and a symmetric eigensolver returns them without the spurious imaginary parts a general eigensolver would introduce at this conditioning. Using a general nonsymmetric eigensolver on the raw product is the common way to obtain a constant that is polluted at the fourth or fifth significant digit.

The constant measured here is a property of the multigrid component alone. It knows nothing about the constraint block, the clustering of the constraints, or the treatment of kinematic loops, which is precisely why the comparison made at the end of the pipeline is informative: it asks how much of the spectral spread of the fully preconditioned saddle-point system is explained by the multigrid quality alone.

Returns
-------
float: the sharpest spectral-equivalence constant between the V-cycle operator and the stiffness, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_spectral_equivalence(stiffness: np.ndarray,
                                 vcycle_operator: np.ndarray) -> float:
    """Measure the sharpest spectral-equivalence constant of the V-cycle.

    Parameters
    ----------
    stiffness : np.ndarray
        Symmetric positive-definite operator of shape (n, n).
    vcycle_operator : np.ndarray
        Symmetric approximate inverse of the stiffness, shape (n, n).

    Returns
    -------
    gamma : float
        Largest deviation from unity over the spectrum of the V-cycle operator
        composed with the stiffness, as a native Python float.
    """
    return gamma  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_spectral_equivalence(stiffness: np.ndarray,
                                         vcycle_operator: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    matrix = np.asarray(stiffness, dtype=float)
    cycle = np.asarray(vcycle_operator, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("stiffness must be a square 2D array of order n >= 1")
    if cycle.shape != matrix.shape:
        raise ValueError("vcycle_operator must have the same shape as stiffness")
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(cycle))):
        raise ValueError("stiffness and vcycle_operator must be finite")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=0.0):
        raise ValueError("stiffness must be symmetric")

    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("stiffness must be positive definite") from exc

    # L^T M L is symmetric and similar to M K, so its spectrum is the one sought.
    similar = factor.T @ (0.5 * (cycle + cycle.T)) @ factor
    eigenvalues = np.linalg.eigvalsh(0.5 * (similar + similar.T))

    return float(np.max(np.abs(eigenvalues - 1.0)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: damped Jacobi approximate inverse of a tridiagonal operator ---
        {
            "setup": """import numpy as np
n = 40
stiffness = np.diag(np.full(n, 4.0)) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
vcycle_operator = (2.0 / 3.0) * np.diag(1.0 / np.diag(stiffness))
""",
            "call": "compute_spectral_equivalence(stiffness, vcycle_operator)",
            "gold_call": "_oracle_compute_spectral_equivalence(stiffness, vcycle_operator)",
        },
        # --- Valid: strongly heterogeneous operator with a diagonal approximate inverse ---
        {
            "setup": """import numpy as np
n = 25
w = 10.0 ** np.linspace(-2.0, 2.0, n - 1)
stiffness = np.zeros((n, n))
for k in range(n - 1):
    stiffness[k, k] += w[k]
    stiffness[k + 1, k + 1] += w[k]
    stiffness[k, k + 1] -= w[k]
    stiffness[k + 1, k] -= w[k]
stiffness += np.eye(n) * 1.0e-1
vcycle_operator = np.diag(1.0 / np.diag(stiffness))
""",
            "call": "compute_spectral_equivalence(stiffness, vcycle_operator)",
            "gold_call": "_oracle_compute_spectral_equivalence(stiffness, vcycle_operator)",
        },
        # --- Boundary: exact inverse, so the constant collapses to zero ---
        {
            "setup": """import numpy as np
n = 12
stiffness = np.diag(np.full(n, 3.0)) - np.diag(0.5 * np.ones(n - 1), 1) - np.diag(0.5 * np.ones(n - 1), -1)
vcycle_operator = np.linalg.inv(stiffness)
""",
            "call": "compute_spectral_equivalence(stiffness, vcycle_operator)",
            "gold_call": "_oracle_compute_spectral_equivalence(stiffness, vcycle_operator)",
        },
        # --- Edge: over-scaled approximate inverse driving the constant above one ---
        {
            "setup": """import numpy as np
n = 6
stiffness = np.diag(np.linspace(1.0, 6.0, n))
vcycle_operator = 2.5 * np.linalg.inv(stiffness)
""",
            "call": "compute_spectral_equivalence(stiffness, vcycle_operator)",
            "gold_call": "_oracle_compute_spectral_equivalence(stiffness, vcycle_operator)",
        },
        # --- Invalid: indefinite stiffness has no Cholesky factor ---
        {
            "setup": """import numpy as np
stiffness = np.diag([1.0, -1.0, 2.0])
vcycle_operator = np.eye(3)
def run_model():
    try:
        compute_spectral_equivalence(stiffness, vcycle_operator)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_spectral_equivalence(stiffness, vcycle_operator)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: mismatched operator shapes ---
        {
            "setup": """import numpy as np
stiffness = np.eye(4) * 2.0
vcycle_operator = np.eye(3)
def run_model():
    try:
        compute_spectral_equivalence(stiffness, vcycle_operator)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_spectral_equivalence(stiffness, vcycle_operator)
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
