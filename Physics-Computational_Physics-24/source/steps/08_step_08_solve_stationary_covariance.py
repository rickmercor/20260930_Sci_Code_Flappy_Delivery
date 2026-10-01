"""
Obtain the stationary covariance matrix of a linear stochastic system from its drift matrix and its diffusion matrix, without sampling any trajectory.

When the drift is linear and the diffusion matrix is evaluated at the stationary mean, the second-moment equations close and the stationary covariance is fixed by a single linear matrix equation rather than by an ensemble of realisations. The equation has a unique solution precisely when every mode of the drift decays.

Returns
-------
np.ndarray of shape (n, n), float: the symmetric stationary covariance matrix in squared counts.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_stationary_covariance(drift_matrix: np.ndarray,
                                diffusion_matrix: np.ndarray) -> np.ndarray:
    """Solve for the stationary covariance of a linear stochastic system.

    Parameters
    ----------
    drift_matrix : np.ndarray
        Square array of shape (n, n) in inverse seconds; the linear part of the
        deterministic drift.
    diffusion_matrix : np.ndarray
        Symmetric positive semi-definite array of the same shape in inverse
        seconds; the instantaneous covariance rate of the stochastic forcing.

    Returns
    -------
    covariance : np.ndarray
        Symmetric array of shape (n, n) holding the stationary covariance of the
        state, in squared counts.

    Raises
    ------
    ValueError
        If either input is not a finite two-dimensional square array, if the two
        do not have the same shape, if ``diffusion_matrix`` is not symmetric
        within 1e-9 of its largest absolute entry, if ``diffusion_matrix`` has an
        eigenvalue below -1e-9 times its largest absolute eigenvalue and so is
        not positive semi-definite, if any eigenvalue of ``drift_matrix`` has a
        non-negative real part, or if the resulting covariance fails the same
        positive semi-definiteness test.
    """
    return covariance  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_stationary_covariance(drift_matrix: np.ndarray,
                                        diffusion_matrix: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    drift = np.asarray(drift_matrix, dtype=float)
    diffusion = np.asarray(diffusion_matrix, dtype=float)
    for name, array in (("drift_matrix", drift), ("diffusion_matrix", diffusion)):
        if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
            raise ValueError(f"{name} must be a non-empty two-dimensional square array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
    if drift.shape != diffusion.shape:
        raise ValueError("drift_matrix and diffusion_matrix must have the same shape")

    scale = float(np.abs(diffusion).max())
    tolerance = 1e-9 * scale if scale > 0.0 else 1e-12
    if float(np.abs(diffusion - diffusion.T).max()) > tolerance:
        raise ValueError("diffusion_matrix must be symmetric")

    # A diffusion matrix is a covariance rate, so it cannot have a materially
    # negative eigenvalue; without this check an indefinite input is accepted
    # and returns something that is not a covariance at all.
    def _reject_indefinite(matrix, name):
        symmetric = 0.5 * (matrix + matrix.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        largest = float(np.abs(eigenvalues).max())
        floor = -1e-9 * largest if largest > 0.0 else -1e-12
        if float(eigenvalues.min()) < floor:
            raise ValueError(f"{name} must be positive semi-definite")

    _reject_indefinite(diffusion, "diffusion_matrix")

    # A unique stationary covariance exists only if the drift has no
    # non-decaying mode.
    spectrum = np.linalg.eigvals(drift)
    if float(np.max(spectrum.real)) >= 0.0:
        raise ValueError("drift_matrix has a non-decaying mode, so no stationary covariance exists")

    # The stationary balance is the linear matrix equation in which the drift
    # acting on the covariance from the left and from the right offsets the
    # diffusion; vectorising it turns the equation into an ordinary linear
    # system whose operator is the Kronecker sum of the drift with itself.
    size = drift.shape[0]
    identity = np.eye(size)
    operator = np.kron(identity, drift) + np.kron(drift, identity)
    try:
        solution = np.linalg.solve(operator, -np.asarray(diffusion).reshape(-1))
    except np.linalg.LinAlgError as exc:
        raise ValueError("the stationary covariance equation is singular") from exc

    covariance = solution.reshape(size, size)
    # Round-off makes the raw solution very slightly asymmetric; the exact
    # solution is symmetric because the diffusion matrix is.
    covariance = 0.5 * (covariance + covariance.T)
    _reject_indefinite(covariance, "the stationary covariance")

    return covariance

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a scalar relaxation, whose stationary variance is the
        #     diffusion divided by twice the decay rate (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
drift = np.array([[-2.5]])
diffusion = np.array([[7.0]])
""",
            "call": "sig(solve_stationary_covariance(drift, diffusion), 1.0e0)",
            "gold_call": "sig(_oracle_solve_stationary_covariance(drift, diffusion), 1.0e0)",
        },
        # --- Valid: a two-region single-group loop driven by neutron noise alone,
        #     so the precursor variances are entirely inherited ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
beta, gen, tc, te = 0.0065, 1.0e-3, 7.5, 12.5
lam = np.array([0.08])
alpha = np.array([1.0])
rho = -0.03
n = lam.size
drift = np.zeros((1 + 2 * n, 1 + 2 * n))
drift[0, 0] = (rho - beta) / gen
drift[0, 1:1 + n] = lam
drift[1:1 + n, 0] = beta * alpha / gen
for j in range(n):
    drift[1 + j, 1 + j] = -(lam[j] + 1.0 / tc)
    drift[1 + j, 1 + n + j] = 1.0 / te
    drift[1 + n + j, 1 + n + j] = -(lam[j] + 1.0 / te)
    drift[1 + n + j, 1 + j] = 1.0 / tc
diffusion = np.zeros((3, 3))
diffusion[0, 0] = 1983.378055917 * 352.0
""",
            "call": "sig(solve_stationary_covariance(drift, diffusion), 1.0e6)",
            "gold_call": "sig(_oracle_solve_stationary_covariance(drift, diffusion), 1.0e6)",
        },
        # --- Boundary: a diffusion matrix of zeros, whose stationary covariance
        #     vanishes identically ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
drift = np.array([[-3.0, 1.0, 0.0], [0.5, -4.0, 2.0], [0.0, 1.5, -5.0]])
diffusion = np.zeros((3, 3))
""",
            "call": "sig(solve_stationary_covariance(drift, diffusion), 1.0e0)",
            "gold_call": "sig(_oracle_solve_stationary_covariance(drift, diffusion), 1.0e0)",
        },
        # --- Edge: a stiff, strongly coupled drift with a dense diffusion matrix
        #     carrying negative off-diagonal correlations ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
drift = np.array([[-1.0e3, 4.0, 0.0],
                  [2.0, -3.0, 0.7],
                  [0.0, 0.9, -1.1]])
root = np.array([[3.0, -1.0, 0.5], [-1.0, 2.0, -0.25], [0.5, -0.25, 1.5]])
diffusion = root @ root.T
""",
            "call": "sig(solve_stationary_covariance(drift, diffusion), 1.0e0)",
            "gold_call": "sig(_oracle_solve_stationary_covariance(drift, diffusion), 1.0e0)",
        },
        # --- Invalid: a drift with a growing mode, for which no stationary
        #     covariance exists ---
        {
            "setup": """import numpy as np
drift = np.array([[0.5, 0.0], [0.0, -2.0]])
diffusion = np.array([[1.0, 0.0], [0.0, 1.0]])
def run_model():
    try:
        solve_stationary_covariance(drift, diffusion)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_stationary_covariance(drift, diffusion)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a symmetric but indefinite diffusion matrix, which no
        #     stochastic forcing can have; accepting it returns a matrix with a
        #     negative eigenvalue that is not a covariance at all ---
        {
            "setup": """import numpy as np
drift = -np.eye(2)
diffusion = np.array([[1.0, 2.0], [2.0, 1.0]])
def run_model():
    try:
        solve_stationary_covariance(drift, diffusion)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_stationary_covariance(drift, diffusion)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: an asymmetric diffusion matrix, which is not a covariance
        #     rate of any stochastic forcing ---
        {
            "setup": """import numpy as np
drift = np.array([[-1.0, 0.0], [0.0, -2.0]])
diffusion = np.array([[1.0, 0.3], [-0.3, 1.0]])
def run_model():
    try:
        solve_stationary_covariance(drift, diffusion)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_stationary_covariance(drift, diffusion)
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
