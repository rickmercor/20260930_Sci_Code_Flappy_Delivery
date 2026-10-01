"""
Assemble the harmonic coupling matrix whose determinant vanishes on the admissible spatial decay constants of the modulated medium.

Rewriting the modulated diffusion equation in the co-moving coordinate leaves coefficients that depend on one variable only, so Bloch's theorem applies and the solutions are a plane-wave factor times a periodic envelope. Restricting attention to states whose spatial profile is stationary in the laboratory frame ties the Bloch frequency to the Bloch wavenumber through the modulation speed. Writing $K=-i\alpha$ gives the spatial factor $e^{\alpha x}$ with generally complex $\alpha$: its real part sets growth or decay, while its imaginary part supplies spatial phase, and the envelope repeats with the modulation period. Projecting that ansatz on harmonics converts the differential equation into a linear system whose unknowns are the envelope's Fourier coefficients and whose matrix mixes a harmonic of the potential with a harmonic of the material through the difference of their indices.




The matrix has two contributions of quite different character. One comes from the divergence of the conducting flux and is quadratic in the decay constant, carrying a wavenumber factor on each side of the conductivity coefficient. The other comes from the time derivative of the stored quantity and is linear in the harmonic index alone, because in the co-moving frame that derivative becomes a convection at the modulation speed acting on the product of capacity and potential; proportional to the modulation speed and to the capacity coefficients, it couples the two modulations and is the entire origin of the nonreciprocity. Setting the modulation speed to zero removes it and leaves a matrix that factorises into diagonal wavenumber factors times a Toeplitz conductivity matrix - the reciprocal limit of an ordinary heterogeneous conductor. One structural feature is worth noticing before any determinant is taken: the row belonging to the zeroth harmonic carries the harmonic index as a factor in both contributions, so that entire row vanishes when the decay constant is set to zero. A zero decay constant is therefore always a root of the secular equation, whatever the medium, and the state it labels is the only one that neither grows nor decays across the branch.

Returns
-------
np.ndarray: complex array of shape (2 * order + 1, 2 * order + 1) holding the harmonic coupling matrix evaluated at the given decay constant.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_secular_matrix(alpha: complex,
                            sigma_modes: "np.ndarray",
                            capacity_modes: "np.ndarray",
                            beta: float,
                            modulation_speed: float) -> "np.ndarray":
    """Assemble the harmonic coupling matrix at one trial decay constant.

    Parameters
    ----------
    alpha : complex
        Trial spatial decay constant in inverse metres.
    sigma_modes : np.ndarray
        Complex conductivity coefficients of shape (2 * order + 1), indexed
        from harmonic -order to +order. The profile carries no harmonics
        beyond that range, so any coefficient whose harmonic index falls
        outside it is zero.
    capacity_modes : np.ndarray
        Complex capacity coefficients with the same shape and ordering, under
        the same convention.
    beta : float
        Modulation wavenumber in inverse metres, nonzero.
    modulation_speed : float
        Speed of the travelling modulation in metres per second.

    Returns
    -------
    matrix : np.ndarray
        Complex array of shape (2 * order + 1, 2 * order + 1) whose row index
        runs over the projected harmonic and whose column index runs over the
        envelope harmonic, both from -order to +order.

    Raises
    ------
    ValueError
        If sigma_modes is not a one-dimensional array of odd length >= 3, if
        capacity_modes does not have the same shape as sigma_modes, if any
        coefficient is not finite, if beta is not a finite nonzero number, if
        modulation_speed is not a finite number, or if alpha is not finite.
    """
    return matrix  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_assemble_secular_matrix(alpha: complex,
                                    sigma_modes: "np.ndarray",
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
    alpha = complex(alpha)
    if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag):
        raise ValueError("alpha must be finite")

    order = (sigma.size - 1) // 2
    indices = np.arange(-order, order + 1)
    rows = indices[:, None]
    cols = indices[None, :]

    # Toeplitz pickers: entry (n, m) reads the coefficient of harmonic n - m.
    difference = rows - cols
    inside = np.abs(difference) <= order
    sigma_toeplitz = np.where(inside, sigma[np.clip(difference + order, 0, 2 * order)], 0.0)
    capacity_toeplitz = np.where(inside, capacity[np.clip(difference + order, 0, 2 * order)], 0.0)

    conduction = (alpha + 1j * float(beta) * rows) * (alpha + 1j * float(beta) * cols) * sigma_toeplitz
    storage = 1j * float(beta) * rows * float(modulation_speed) * capacity_toeplitz
    return conduction + storage

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark medium at a generic complex decay constant ---
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
alpha = 0.7 - 3.2j

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
            "call": "digest(assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
            "gold_call": "digest(_oracle_assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
        },
        # --- Boundary: zero decay constant, where the zeroth-harmonic row vanishes ---
        {
            "setup": """import numpy as np
order = 3
sigma_modes = np.zeros(2 * order + 1, dtype=complex)
capacity_modes = np.zeros(2 * order + 1, dtype=complex)
sigma_modes[order] = 1.0
sigma_modes[order + 1] = sigma_modes[order - 1] = 0.45
capacity_modes[order] = 80.0
capacity_modes[order + 1] = 20.0 * np.exp(1j * 1.1)
capacity_modes[order - 1] = 20.0 * np.exp(-1j * 1.1)
beta = 6.0 * np.pi
modulation_speed = -0.12
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
            "call": "digest(assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
            "gold_call": "digest(_oracle_assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
        },
        # --- Boundary: static medium, where the storage contribution disappears ---
        {
            "setup": """import numpy as np
order = 2
sigma_modes = np.array([0.05 * np.exp(-0.7j), 0.3 * np.exp(-0.4j), 1.0,
                        0.3 * np.exp(0.4j), 0.05 * np.exp(0.7j)], dtype=complex)
capacity_modes = np.array([2.0, 5.0, 40.0, 5.0, 2.0], dtype=complex)
beta = 2.0 * np.pi
modulation_speed = 0.0
alpha = 1.5 + 0.0j

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
            "call": "digest(assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
            "gold_call": "digest(_oracle_assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
        },
        # --- Edge: smallest admissible truncation, a single harmonic each side ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.25, 1.0, 0.25], dtype=complex)
capacity_modes = np.array([10.0j, 60.0, -10.0j], dtype=complex)
beta = 1.0
modulation_speed = 3.0
alpha = -0.4 + 0.9j

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
            "call": "digest(assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
            "gold_call": "digest(_oracle_assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed))",
        },
        # --- Invalid: even-length coefficient array has no zeroth harmonic ---
        {
            "setup": """import numpy as np
sigma_modes = np.ones(4, dtype=complex)
capacity_modes = np.ones(4, dtype=complex)
beta = 1.0
modulation_speed = 0.1
alpha = 1.0 + 0.0j
def run_model():
    try:
        assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: vanishing modulation wavenumber ---
        {
            "setup": """import numpy as np
sigma_modes = np.array([0.2, 1.0, 0.2], dtype=complex)
capacity_modes = np.array([1.0, 20.0, 1.0], dtype=complex)
beta = 0.0
modulation_speed = 0.1
alpha = 1.0 + 0.0j
def run_model():
    try:
        assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_assemble_secular_matrix(alpha, sigma_modes, capacity_modes, beta, modulation_speed)
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
