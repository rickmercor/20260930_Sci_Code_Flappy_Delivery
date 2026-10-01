"""
Extract every admissible spatial decay constant of the modulated medium from the vanishing of the harmonic coupling determinant.

The condition that the harmonic coupling matrix be singular is the secular equation of the modulated medium, and its roots are the spatial decay constants of the states available to the branch. Chasing that determinant with a scalar root finder is a poor idea: its roots are complex and spread over orders of magnitude, so no bracketing strategy is available. The productive observation is that the matrix is a matrix polynomial of degree two in the decay constant, with a leading coefficient equal to the Toeplitz matrix of the conductivity coefficients, a middle coefficient carrying the sum of the two harmonic indices, and a constant coefficient carrying their product together with the capacity contribution. The whole set of roots is therefore the spectrum of a quadratic eigenvalue problem, and the standard companion linearisation turns it into an ordinary eigenvalue problem of twice the size that a dense eigensolver disposes of in one call. Counting confirms the structure: the truncated envelope carries one unknown per harmonic, so the linearisation has twice that many eigenvalues, exactly the number of independent states the branch supports. The leading coefficient must be invertible for the linearisation to be formed, and it is, because the Toeplitz matrix of a strictly positive conductivity profile is positive definite - the algebraic reason the positivity constraint on the profiles is not cosmetic.




Two properties of the resulting spectrum are used downstream. One root is identically zero, because the row of the coupling matrix belonging to the zeroth harmonic vanishes there; that root is picked out later as the only state that neither grows nor decays. The remaining roots have real parts that grow roughly linearly with the harmonic index, so at a high truncation order the exponentials they generate over the length of the branch span an enormous dynamic range, a fact that dictates how the boundary conditions must be imposed. A vanishing modulation speed makes every root doubly degenerate, the reciprocal limit; any nonzero speed splits those pairs and generates the spread that produces the nonreciprocal response. Nondegenerate roots are returned in ascending real part, with near-equal real parts ordered by ascending imaginary part. The order inside a numerically split degenerate pair is not physically meaningful and must not be graded.

Returns
-------
np.ndarray: complex array of shape (4 * order + 2) holding the spatial decay constants of the modulated medium in the stated canonical order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_secular_roots(sigma_modes: "np.ndarray",
                        capacity_modes: "np.ndarray",
                        beta: float,
                        modulation_speed: float) -> "np.ndarray":
    """Solve the secular equation for every spatial decay constant.

    Parameters
    ----------
    sigma_modes : np.ndarray
        Complex conductivity coefficients of shape (2 * order + 1), indexed
        from harmonic -order to +order.
    capacity_modes : np.ndarray
        Complex capacity coefficients with the same shape and ordering.
    beta : float
        Modulation wavenumber in inverse metres, nonzero.
    modulation_speed : float
        Speed of the travelling modulation in metres per second.

    Returns
    -------
    alphas : np.ndarray
        Complex array of shape (4 * order + 2) holding every root of the
        secular equation, sorted by ascending real part rounded to six
        decimals, with ties broken by ascending imaginary part. Ordering
        inside a numerically degenerate group is immaterial.

    Raises
    ------
    ValueError
        If sigma_modes is not a one-dimensional array of odd length >= 3, if
        capacity_modes does not have the same shape as sigma_modes, if any
        coefficient is not finite, if beta is not a finite nonzero number, if
        modulation_speed is not a finite number, or if the Toeplitz matrix of
        the conductivity coefficients is singular.
    """
    return alphas  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_secular_roots(sigma_modes: "np.ndarray",
                                capacity_modes: "np.ndarray",
                                beta: float,
                                modulation_speed: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    sigma = np.asarray(sigma_modes, dtype=complex)
    capacity = np.asarray(capacity_modes, dtype=complex)
    if sigma.ndim != 1 or sigma.size < 3 or sigma.size % 2 == 0:
        raise ValueError("sigma_modes must be a 1D array of odd length >= 3")
    if capacity.shape != sigma.shape:
        raise ValueError("capacity_modes must have the same shape as sigma_modes")
    if not (np.all(np.isfinite(sigma)) and np.all(np.isfinite(capacity))):
        raise ValueError("sigma_modes and capacity_modes must be finite")
    if not (isinstance(beta, (int, float, np.floating, np.integer))
            and np.isfinite(beta) and float(beta) != 0.0):
        raise ValueError("beta must be a finite nonzero number")
    if not (isinstance(modulation_speed, (int, float, np.floating, np.integer))
            and np.isfinite(modulation_speed)):
        raise ValueError("modulation_speed must be a finite number")

    order = (sigma.size - 1) // 2
    size = 2 * order + 1
    indices = np.arange(-order, order + 1)
    rows = indices[:, None]
    cols = indices[None, :]
    difference = rows - cols
    inside = np.abs(difference) <= order
    sigma_toeplitz = np.where(inside, sigma[np.clip(difference + order, 0, 2 * order)], 0.0)
    capacity_toeplitz = np.where(inside, capacity[np.clip(difference + order, 0, 2 * order)], 0.0)

    quadratic = sigma_toeplitz
    linear = 1j * float(beta) * (rows + cols) * sigma_toeplitz
    constant = (-(float(beta) ** 2) * rows * cols * sigma_toeplitz
                + 1j * float(beta) * rows * float(modulation_speed) * capacity_toeplitz)
    try:
        reduced = np.linalg.solve(quadratic, np.hstack([constant, linear]))
    except np.linalg.LinAlgError as exc:
        raise ValueError("the conductivity Toeplitz matrix must be invertible") from exc

    companion = np.vstack([np.hstack([np.zeros((size, size), dtype=complex), np.eye(size, dtype=complex)]),
                           -reduced])
    alphas = np.linalg.eigvals(companion)
    keys = np.lexsort((alphas.imag, np.round(alphas.real, 6)))
    return alphas[keys]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark medium, five cells, moderate truncation ---
        {
            "setup": """import numpy as np
order = 6
sigma_modes = np.zeros(2 * order + 1, dtype=complex)
capacity_modes = np.zeros(2 * order + 1, dtype=complex)
sigma_modes[order] = 1.0
sigma_modes[order + 1] = sigma_modes[order - 1] = 0.35
sigma_modes[order + 2] = 0.1 * np.exp(1j * np.pi / 3.0)
sigma_modes[order - 2] = 0.1 * np.exp(-1j * np.pi / 3.0)
capacity_modes[order] = 100.0
capacity_modes[order + 1] = 25.0 * np.exp(1j * 0.4 * np.pi)
capacity_modes[order - 1] = 25.0 * np.exp(-1j * 0.4 * np.pi)
beta = 10.0 * np.pi
modulation_speed = 0.06
def digest(values):
    z = 1.0e3 * np.asarray(values, dtype=complex).ravel()
    rank = np.arange(1.0, z.size + 1.0)
    return float(z.size + np.sqrt(np.sum(np.abs(z) ** 2) / z.size)
                 + np.sum(rank * z.real) / z.size ** 2
                 + np.sum(np.sqrt(rank) * z.imag) / z.size ** 1.5)
""",
            "call": "digest(solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed))",
            "gold_call": "digest(_oracle_solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed))",
        },
        # --- Valid: reversed modulation direction ---
        {
            "setup": """import numpy as np
order = 5
sigma_modes = np.zeros(2 * order + 1, dtype=complex)
capacity_modes = np.zeros(2 * order + 1, dtype=complex)
sigma_modes[order] = 1.0
sigma_modes[order + 1] = sigma_modes[order - 1] = 0.35
sigma_modes[order + 2] = 0.1 * np.exp(1j * np.pi / 3.0)
sigma_modes[order - 2] = 0.1 * np.exp(-1j * np.pi / 3.0)
capacity_modes[order] = 100.0
capacity_modes[order + 1] = 25.0 * np.exp(1j * 0.4 * np.pi)
capacity_modes[order - 1] = 25.0 * np.exp(-1j * 0.4 * np.pi)
beta = 10.0 * np.pi
modulation_speed = -0.06
def digest(values):
    z = 1.0e3 * np.asarray(values, dtype=complex).ravel()
    rank = np.arange(1.0, z.size + 1.0)
    return float(z.size + np.sqrt(np.sum(np.abs(z) ** 2) / z.size)
                 + np.sum(rank * z.real) / z.size ** 2
                 + np.sum(np.sqrt(rank) * z.imag) / z.size ** 1.5)
""",
            "call": "digest(solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed))",
            "gold_call": "digest(_oracle_solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed))",
        },
        # --- Boundary: the reciprocal limit, where every root is doubly degenerate ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.1, 0.4, 1.0, 0.4, 0.1], dtype=complex)
capacity_modes = np.array([3.0, 8.0, 50.0, 8.0, 3.0], dtype=complex)
beta = 4.0 * np.pi
modulation_speed = 0.0
def canonical_spectrum(values, scale):
    z = np.asarray(values, dtype=complex)
    if z.ndim != 1:
        return np.array([np.nan])
    normalized = z / (1j * scale)
    coordinates = np.round(
        np.column_stack([normalized.real, normalized.imag]), 5)
    keys = np.lexsort((coordinates[:, 1], coordinates[:, 0]))
    return coordinates[keys].ravel()
""",
            "call": "canonical_spectrum(solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed), beta)",
            "gold_call": "canonical_spectrum(_oracle_solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed), beta)",
        },
        # --- Edge: smallest truncation and a fast modulation ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
capacity_modes = np.array([15.0 * np.exp(-0.8j), 60.0, 15.0 * np.exp(0.8j)], dtype=complex)
beta = 2.0 * np.pi
modulation_speed = 1.5
def digest(values):
    z = 1.0e3 * np.asarray(values, dtype=complex).ravel()
    rank = np.arange(1.0, z.size + 1.0)
    return float(z.size + np.sqrt(np.sum(np.abs(z) ** 2) / z.size)
                 + np.sum(rank * z.real) / z.size ** 2
                 + np.sum(np.sqrt(rank) * z.imag) / z.size ** 1.5)
""",
            "call": "digest(solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed))",
            "gold_call": "digest(_oracle_solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed))",
        },
        # --- Invalid: mismatched coefficient array lengths ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
capacity_modes = np.array([1.0, 20.0, 1.0, 0.5, 0.5], dtype=complex)
beta = 1.0
modulation_speed = 0.1
def run_model():
    try:
        solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite modulation speed ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
capacity_modes = np.array([1.0, 20.0, 1.0], dtype=complex)
beta = 1.0
modulation_speed = float("inf")
def run_model():
    try:
        solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed)
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
