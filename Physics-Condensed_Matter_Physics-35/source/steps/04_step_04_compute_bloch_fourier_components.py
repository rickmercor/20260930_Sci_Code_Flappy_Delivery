"""
Recover the periodic envelope belonging to one spatial decay constant as the null vector of the harmonic coupling matrix under a fixed normalisation.

Each root of the secular equation makes the harmonic coupling matrix singular, so the envelope belonging to that root is the corresponding null vector, determined only up to a multiplicative constant. Fixing that constant is a modelling decision, not a numerical convenience: the envelope of a Bloch state of this kind is conventionally normalised so that its average over the modulation cell equals unity, which means its zeroth Fourier coefficient is set to one and the plane-wave prefactor carries all the amplitude information. With that normalisation the remaining coefficients solve an inhomogeneous linear system rather than a homogeneous one - the column of the coupling matrix belonging to the zeroth harmonic multiplies a known quantity and therefore moves to the right-hand side, leaving the submatrix obtained by deleting the zeroth row and the zeroth column to be inverted against it. Deleting the zeroth row is the right choice because that row is the one made redundant by the singularity: physically it expresses conservation of the cell-averaged flux, which the state satisfies identically, and numerically it is the row that vanishes outright at the zero root. The surviving submatrix is generically well conditioned even where the full matrix is singular by construction, so the system can be solved directly rather than by a rank-revealing factorisation.




The zero root deserves separate attention because it is the one that ends up carrying the entire time-averaged flux. There the two conductivity wavenumber factors collapse to pure harmonic wavenumbers and the right-hand side reduces to the capacity contribution alone, so the envelope of that state is generated purely by the modulation of the capacity: if the capacity were uniform, the state would be strictly constant, its envelope would have no structure, and the medium would pass no net flux under the mechanism at work here. That is the sharpest statement of why both coefficients have to be modulated, and it appears in this step as the observation that the inhomogeneity vanishes when the capacity harmonics do.

Returns
-------
np.ndarray: complex array of shape (2 * order + 1) holding the Fourier coefficients of the normalised periodic envelope of the given Bloch state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_bloch_fourier_components(alpha: complex,
                                     sigma_modes: "np.ndarray",
                                     capacity_modes: "np.ndarray",
                                     beta: float,
                                     modulation_speed: float) -> "np.ndarray":
    """Recover the normalised periodic envelope of one Bloch state.

    Parameters
    ----------
    alpha : complex
        Spatial decay constant of the state, a root of the secular equation.
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
    envelope : np.ndarray
        Complex array of shape (2 * order + 1) holding the Fourier
        coefficients of the envelope from harmonic -order to +order, with the
        zeroth coefficient equal to one.

    Raises
    ------
    ValueError
        If sigma_modes is not a one-dimensional array of odd length >= 3, if
        capacity_modes does not have the same shape as sigma_modes, if any
        coefficient is not finite, if beta is not a finite nonzero number, if
        modulation_speed is not a finite number, if alpha is not finite, or if
        the reduced coupling matrix is singular.
    """
    return envelope  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_bloch_fourier_components(alpha: complex,
                                             sigma_modes: "np.ndarray",
                                             capacity_modes: "np.ndarray",
                                             beta: float,
                                             modulation_speed: float) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.  The coupling matrix itself comes from the
    # earlier step's own reference implementation, which the executing namespace
    # supplies, so the two steps cannot drift apart and every input check that
    # step declares is applied here too.
    import numpy as np

    matrix = np.asarray(_oracle_assemble_secular_matrix(
        alpha, sigma_modes, capacity_modes, beta, modulation_speed))
    order = (matrix.shape[0] - 1) // 2

    # Deleting the zeroth row and column leaves the system that the
    # normalisation of the zeroth coefficient to one makes inhomogeneous.
    keep = [i for i in range(2 * order + 1) if i != order]
    envelope = np.zeros(2 * order + 1, dtype=complex)
    envelope[order] = 1.0
    try:
        envelope[keep] = -np.linalg.solve(matrix[np.ix_(keep, keep)], matrix[keep, order])
    except np.linalg.LinAlgError as exc:
        raise ValueError("the reduced coupling matrix must be invertible") from exc
    return envelope

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the flux-carrying state of the benchmark medium ---
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
alpha = 0.0 + 0.0j

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
            "gold_call": "digest(_oracle_compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
        },
        # --- Valid: a decaying state at a generic complex decay constant ---
        {
            "setup": """import numpy as np
order = 4
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
alpha = -11.579034922479 - 20.399036342114j

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
            "gold_call": "digest(_oracle_compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
        },
        # --- Boundary: uniform capacity, where the envelope collapses to a constant ---
        {
            "setup": """import numpy as np
order = 3
sigma_modes = np.zeros(2 * order + 1, dtype=complex)
capacity_modes = np.zeros(2 * order + 1, dtype=complex)
sigma_modes[order] = 1.0
sigma_modes[order + 1] = sigma_modes[order - 1] = 0.4
capacity_modes[order] = 100.0
beta = 6.0 * np.pi
modulation_speed = 0.2
alpha = 0.0 + 0.0j

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
            "gold_call": "digest(_oracle_compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
        },
        # --- Edge: smallest truncation and the exact non-decaying root ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
capacity_modes = np.array([30.0 * np.exp(-1.4j), 60.0, 30.0 * np.exp(1.4j)], dtype=complex)
beta = 2.0 * np.pi
modulation_speed = 0.9
alpha = 0.0 + 0.0j

def digest(values):
    grid = np.asarray(values, dtype=complex)
    flat = 1.0e6 * grid.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    weight = rank / rank.sum()
    root = np.sqrt(rank) / np.sqrt(rank).sum()
    return float(flat.size + 3.0 * grid.shape[0]
                 + np.sqrt(np.sum(weight * np.abs(flat) ** 2))
                 + 0.5 * np.sum(weight * flat.real) + 0.25 * np.sum(root * flat.imag))
""",
            "call": "digest(compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
            "gold_call": "digest(_oracle_compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
        },
        # --- Invalid: non-finite decay constant ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.3, 1.0, 0.3], dtype=complex)
capacity_modes = np.array([5.0, 40.0, 5.0], dtype=complex)
beta = 2.0
modulation_speed = 0.1
alpha = complex(float("nan"), 0.0)
def run_model():
    try:
        compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: even-length coefficient array has no zeroth harmonic ---
        {
            "setup": """import numpy as np
sigma_modes = np.ones(6, dtype=complex)
capacity_modes = np.ones(6, dtype=complex)
beta = 2.0
modulation_speed = 0.1
alpha = 1.0 + 0.0j
def run_model():
    try:
        compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_bloch_fourier_components(alpha, sigma_modes, capacity_modes, beta, modulation_speed)
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
