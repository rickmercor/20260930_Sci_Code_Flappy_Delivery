"""
Solve the source-driven stationary state of the two-region point-kinetic model for the neutron population and for the in-core and ex-core precursor inventory of every delayed group.

A subcritical assembly fed by a constant external source settles onto a stationary state fixed by the balance of the drift matrix against the source vector, and that state is where the noise of both stochastic descriptions has to be evaluated. Existence of the state requires every mode of the drift matrix to be decaying.

Returns
-------
np.ndarray of shape (1 + 2 * n_groups,), float: the stationary neutron and precursor populations.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_steady_state_populations(kinetics_matrix: np.ndarray,
                                   source_rate: float) -> np.ndarray:
    """Solve for the source-driven stationary populations of the two-region model.

    Parameters
    ----------
    kinetics_matrix : np.ndarray
        Square drift matrix of shape (1 + 2 * n_groups, 1 + 2 * n_groups) with
        the state ordering of sub-problem 03.
    source_rate : float
        Rate at which the external source emits neutrons into the core, in
        neutrons per second (source_rate > 0).

    Returns
    -------
    state : np.ndarray
        Array of shape (1 + 2 * n_groups,) holding the stationary neutron
        population followed by the in-core and then the ex-core precursor
        populations, all dimensionless counts.

    Raises
    ------
    ValueError
        If ``kinetics_matrix`` is not a finite two-dimensional square array of
        odd size at least three, if ``source_rate`` is not a finite number
        greater than zero, if any eigenvalue of ``kinetics_matrix`` has a
        non-negative real part so that no stationary state exists, or if the
        resulting state is not strictly positive in every component.
    """
    return state  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_steady_state_populations(kinetics_matrix: np.ndarray,
                                           source_rate: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    if not (isinstance(source_rate, (int, float, np.floating, np.integer))
            and not isinstance(source_rate, bool) and np.isfinite(source_rate)):
        raise ValueError("source_rate must be a finite number")
    source_rate = float(source_rate)
    if source_rate <= 0.0:
        raise ValueError("source_rate must be greater than zero")

    matrix = np.asarray(kinetics_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("kinetics_matrix must be a two-dimensional square array")
    size = matrix.shape[0]
    if size < 3 or size % 2 == 0:
        raise ValueError("kinetics_matrix must have odd size of at least three")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("kinetics_matrix must contain only finite entries")

    # A stationary state exists only if every mode of the drift decays; a
    # non-decaying mode means the configuration is at or above critical and the
    # source-driven population grows without bound.
    spectrum = np.linalg.eigvals(matrix)
    if float(np.max(spectrum.real)) >= 0.0:
        raise ValueError("kinetics_matrix has a non-decaying mode, so no stationary state exists")

    # The external source enters the neutron balance only; the precursor rows
    # are driven purely by fission.
    source = np.zeros(size, dtype=float)
    source[0] = source_rate

    try:
        state = np.linalg.solve(matrix, -source)
    except np.linalg.LinAlgError as exc:
        raise ValueError("kinetics_matrix is singular") from exc

    if not np.all(np.isfinite(state)) or np.any(state <= 0.0):
        raise ValueError("the stationary state must be strictly positive in every component")

    return state

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the testbed at its operating point (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
def kmat(rho, beta, alpha, lam, gen, tc, te):
    n = lam.size
    A = np.zeros((1 + 2 * n, 1 + 2 * n))
    A[0, 0] = (rho - beta) / gen
    A[0, 1:1 + n] = lam
    A[1:1 + n, 0] = beta * alpha / gen
    for j in range(n):
        A[1 + j, 1 + j] = -(lam[j] + 1.0 / tc)
        A[1 + j, 1 + n + j] = 1.0 / te
        A[1 + n + j, 1 + n + j] = -(lam[j] + 1.0 / te)
        A[1 + n + j, 1 + j] = 1.0 / tc
    return A
alpha = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
lam = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
A = kmat(-0.022820635778950265, 0.0065, alpha, lam, 1.0e-3, 7.5, 12.5)
source_rate = 8800.0
""",
            "call": "sig(solve_steady_state_populations(A, source_rate), 1.0e4)",
            "gold_call": "sig(_oracle_solve_steady_state_populations(A, source_rate), 1.0e4)",
        },
        # --- Valid: a much deeper subcriticality, where the population collapses ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
def kmat(rho, beta, alpha, lam, gen, tc, te):
    n = lam.size
    A = np.zeros((1 + 2 * n, 1 + 2 * n))
    A[0, 0] = (rho - beta) / gen
    A[0, 1:1 + n] = lam
    A[1:1 + n, 0] = beta * alpha / gen
    for j in range(n):
        A[1 + j, 1 + j] = -(lam[j] + 1.0 / tc)
        A[1 + j, 1 + n + j] = 1.0 / te
        A[1 + n + j, 1 + n + j] = -(lam[j] + 1.0 / te)
        A[1 + n + j, 1 + j] = 1.0 / tc
    return A
alpha = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
lam = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
A = kmat(-0.2, 0.0065, alpha, lam, 1.0e-3, 7.5, 12.5)
source_rate = 8800.0
""",
            "call": "sig(solve_steady_state_populations(A, source_rate), 1.0e3)",
            "gold_call": "sig(_oracle_solve_steady_state_populations(A, source_rate), 1.0e3)",
        },
        # --- Boundary: a two-group loop only marginally below its circulating
        #     critical point, where the state is large and stiff ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
def kmat(rho, beta, alpha, lam, gen, tc, te):
    n = lam.size
    A = np.zeros((1 + 2 * n, 1 + 2 * n))
    A[0, 0] = (rho - beta) / gen
    A[0, 1:1 + n] = lam
    A[1:1 + n, 0] = beta * alpha / gen
    for j in range(n):
        A[1 + j, 1 + j] = -(lam[j] + 1.0 / tc)
        A[1 + j, 1 + n + j] = 1.0 / te
        A[1 + n + j, 1 + n + j] = -(lam[j] + 1.0 / te)
        A[1 + n + j, 1 + j] = 1.0 / tc
    return A
alpha = np.array([0.4, 0.6])
lam = np.array([0.02, 0.5])
A = kmat(0.0010, 0.0065, alpha, lam, 1.0e-3, 20.0, 30.0)
source_rate = 500.0
""",
            "call": "sig(solve_steady_state_populations(A, source_rate), 1.0e5)",
            "gold_call": "sig(_oracle_solve_steady_state_populations(A, source_rate), 1.0e5)",
        },
        # --- Edge: a single group with a very weak source ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
def kmat(rho, beta, alpha, lam, gen, tc, te):
    n = lam.size
    A = np.zeros((1 + 2 * n, 1 + 2 * n))
    A[0, 0] = (rho - beta) / gen
    A[0, 1:1 + n] = lam
    A[1:1 + n, 0] = beta * alpha / gen
    for j in range(n):
        A[1 + j, 1 + j] = -(lam[j] + 1.0 / tc)
        A[1 + j, 1 + n + j] = 1.0 / te
        A[1 + n + j, 1 + n + j] = -(lam[j] + 1.0 / te)
        A[1 + n + j, 1 + j] = 1.0 / tc
    return A
alpha = np.array([1.0])
lam = np.array([0.08])
A = kmat(-0.05, 0.0075, alpha, lam, 2.0e-4, 4.0, 6.0)
source_rate = 1.0
""",
            "call": "sig(solve_steady_state_populations(A, source_rate), 1.0e0)",
            "gold_call": "sig(_oracle_solve_steady_state_populations(A, source_rate), 1.0e0)",
        },
        # --- Invalid: a supercritical configuration, which has no stationary state ---
        {
            "setup": """import numpy as np
def kmat(rho, beta, alpha, lam, gen, tc, te):
    n = lam.size
    A = np.zeros((1 + 2 * n, 1 + 2 * n))
    A[0, 0] = (rho - beta) / gen
    A[0, 1:1 + n] = lam
    A[1:1 + n, 0] = beta * alpha / gen
    for j in range(n):
        A[1 + j, 1 + j] = -(lam[j] + 1.0 / tc)
        A[1 + j, 1 + n + j] = 1.0 / te
        A[1 + n + j, 1 + n + j] = -(lam[j] + 1.0 / te)
        A[1 + n + j, 1 + j] = 1.0 / tc
    return A
alpha = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
lam = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
A = kmat(0.004, 0.0065, alpha, lam, 1.0e-3, 7.5, 12.5)
source_rate = 8800.0
def run_model():
    try:
        solve_steady_state_populations(A, source_rate)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_steady_state_populations(A, source_rate)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a source rate of zero, which leaves no stationary state to
        #     evaluate the noise on ---
        {
            "setup": """import numpy as np
def kmat(rho, beta, alpha, lam, gen, tc, te):
    n = lam.size
    A = np.zeros((1 + 2 * n, 1 + 2 * n))
    A[0, 0] = (rho - beta) / gen
    A[0, 1:1 + n] = lam
    A[1:1 + n, 0] = beta * alpha / gen
    for j in range(n):
        A[1 + j, 1 + j] = -(lam[j] + 1.0 / tc)
        A[1 + j, 1 + n + j] = 1.0 / te
        A[1 + n + j, 1 + n + j] = -(lam[j] + 1.0 / te)
        A[1 + n + j, 1 + j] = 1.0 / tc
    return A
alpha = np.array([0.4, 0.6])
lam = np.array([0.02, 0.5])
A = kmat(-0.02, 0.0065, alpha, lam, 1.0e-3, 7.5, 12.5)
def run_model():
    try:
        solve_steady_state_populations(A, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_steady_state_populations(A, 0.0)
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
