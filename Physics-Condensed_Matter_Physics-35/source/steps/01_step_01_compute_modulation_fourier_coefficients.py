"""
Reduce the two travelling-wave material profiles to the complex Fourier coefficients through which they enter the modulated diffusion problem.

A medium whose conductivity and capacity are driven as co-propagating travelling waves depends on space and time only through the co-moving coordinate, and both profiles are periodic in it with the modulation period. Every later stage of the analysis touches the medium exclusively through the complex Fourier coefficients of those two functions, never through the profiles themselves, which is what makes the framework indifferent to the shape of the modulation. The harmonic index runs symmetrically about zero because the matrices assembled downstream couple a harmonic of the potential to a harmonic of the material through the difference of their indices, so both signs are needed even for a purely real profile.




Each profile here is a real cosine series about a positive mean, with an amplitude and a phase offset for every harmonic present. Realness forces the coefficients at opposite indices to be complex conjugates, the phase offset of a harmonic appears in them as a pure phase whose sign follows the sign of the index, and the zeroth coefficient is the mean of the profile. The relative phase between the two series is not decorative: the effective drift arises from the interference of the two modulations, so shifting one relative to the other changes the magnitude of the nonreciprocity and can reverse its direction, while shifting both together does nothing. Two physical constraints must also survive the reduction. Both quantities are strictly positive, so amplitudes that drive either profile to zero or below somewhere in the cell describe no material at all and cost the downstream Toeplitz matrix of the conductivity coefficients the invertibility the whole construction relies on; and the truncation order has to be at least as large as the highest harmonic actually present, or the reduction silently discards part of the medium it was given.

Returns
-------
np.ndarray: complex array of shape (2, 2 * order + 1) holding the conductivity coefficients in row 0 and the capacity coefficients in row 1, indexed from harmonic -order to +order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_modulation_fourier_coefficients(sigma_mean: float,
                                            sigma_amplitudes: "np.ndarray",
                                            sigma_phases: "np.ndarray",
                                            capacity_mean: float,
                                            capacity_amplitudes: "np.ndarray",
                                            capacity_phases: "np.ndarray",
                                            order: int) -> "np.ndarray":
    """Reduce the two modulation profiles to their complex Fourier coefficients.

    Parameters
    ----------
    sigma_mean : float
        Mean conductivity of the profile, strictly positive.
    sigma_amplitudes : np.ndarray
        Relative amplitudes of harmonics 1, 2, ... of the conductivity.
    sigma_phases : np.ndarray
        Phase offsets in radians of the same conductivity harmonics.
    capacity_mean : float
        Mean capacity of the profile, strictly positive.
    capacity_amplitudes : np.ndarray
        Relative amplitudes of harmonics 1, 2, ... of the capacity.
    capacity_phases : np.ndarray
        Phase offsets in radians of the same capacity harmonics.
    order : int
        Fourier truncation order, at least 1 and at least the highest
        harmonic present in either profile.

    Returns
    -------
    modes : np.ndarray
        Complex array of shape (2, 2 * order + 1). Row 0 holds the
        conductivity coefficients and row 1 the capacity coefficients, both
        ordered by harmonic index from -order to +order.

    Raises
    ------
    ValueError
        If order is not an integer >= 1, if either profile mean is not a
        finite number > 0, if an amplitude array and its phase array have
        different lengths, if any amplitude or phase is not finite, if order
        is smaller than the highest harmonic present in either profile, or if
        the amplitudes drive either profile to zero or below anywhere in the
        cell.
    """
    return modes  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_modulation_fourier_coefficients(sigma_mean: float,
                                                    sigma_amplitudes: "np.ndarray",
                                                    sigma_phases: "np.ndarray",
                                                    capacity_mean: float,
                                                    capacity_amplitudes: "np.ndarray",
                                                    capacity_phases: "np.ndarray",
                                                    order: int) -> "np.ndarray":
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(order, (int, np.integer)) and not isinstance(order, bool)
            and int(order) >= 1):
        raise ValueError("order must be an integer >= 1")
    order = int(order)
    grid = np.linspace(0.0, 2.0 * np.pi, 4097)[:-1]
    modes = np.zeros((2, 2 * order + 1), dtype=complex)

    for row, (mean, amplitudes, phases) in enumerate(
            ((sigma_mean, sigma_amplitudes, sigma_phases),
             (capacity_mean, capacity_amplitudes, capacity_phases))):
        if not (isinstance(mean, (int, float, np.floating, np.integer))
                and np.isfinite(mean) and float(mean) > 0.0):
            raise ValueError("each profile mean must be a finite number > 0")
        amp = np.atleast_1d(np.asarray(amplitudes, dtype=float)).ravel()
        pha = np.atleast_1d(np.asarray(phases, dtype=float)).ravel()
        if amp.shape != pha.shape:
            raise ValueError("amplitudes and phases must have the same length")
        if not (np.all(np.isfinite(amp)) and np.all(np.isfinite(pha))):
            raise ValueError("amplitudes and phases must be finite")
        if amp.size > order:
            raise ValueError("order must be at least the highest harmonic present")

        profile = np.full(grid.shape, float(mean))
        for harmonic, (a, p) in enumerate(zip(amp, pha), start=1):
            profile = profile + float(mean) * a * np.cos(harmonic * grid + p)
            # Realness of the profile pairs opposite indices as conjugates.
            modes[row, order + harmonic] = 0.5 * float(mean) * a * np.exp(1j * p)
            modes[row, order - harmonic] = 0.5 * float(mean) * a * np.exp(-1j * p)
        if np.min(profile) <= 0.0:
            raise ValueError("conductivity and capacity profiles must stay positive")
        modes[row, order] = float(mean)

    return modes

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: two-harmonic conductivity and one-harmonic capacity ---
        {
            "setup": """import numpy as np
sigma_mean = 1.0
sigma_amplitudes = np.array([0.7, 0.2])
sigma_phases = np.array([0.0, np.pi / 3.0])
capacity_mean = 100.0
capacity_amplitudes = np.array([0.5])
capacity_phases = np.array([0.4 * np.pi])
order = 16

def digest(values):
    modes = np.asarray(values, dtype=complex)
    if modes.ndim != 2 or modes.shape[0] != 2 or modes.shape[1] % 2 == 0:
        return -1.0
    rows, columns = modes.shape
    order = (columns - 1) // 2
    flat = modes.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    # Realness of both profiles makes opposite harmonics conjugate and the
    # zeroth coefficient real, so these two residuals vanish for a correct
    # return and any single wrong coefficient shows up in one of them.
    mirror = float(np.sum(np.abs(modes[:, order + 1:] - np.conj(modes[:, order - 1::-1]))))
    centre = float(np.sum(np.abs(modes[:, order].imag)))
    return float(rows + 7.0 * columns + 13.0 * flat.size
                 + np.sum(np.abs(flat) * np.cos(rank))
                 + np.sum(flat.real * np.sin(rank))
                 + np.sum(flat.imag * np.cos(rank))
                 + 1.0e6 * (mirror + centre))
""",
            "call": "digest(compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order))",
            "gold_call": "digest(_oracle_compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order))",
        },
        # --- Boundary: truncation order equal to the highest harmonic present ---
        {
            "setup": """import numpy as np
sigma_mean = 2.5
sigma_amplitudes = np.array([0.4, 0.3, 0.1])
sigma_phases = np.array([0.25, -1.1, 2.0])
capacity_mean = 7.0
capacity_amplitudes = np.array([0.6, 0.2, 0.05])
capacity_phases = np.array([-0.5, 0.9, 0.0])
order = 3

def digest(values):
    modes = np.asarray(values, dtype=complex)
    if modes.ndim != 2 or modes.shape[0] != 2 or modes.shape[1] % 2 == 0:
        return -1.0
    rows, columns = modes.shape
    order = (columns - 1) // 2
    flat = modes.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    # Realness of both profiles makes opposite harmonics conjugate and the
    # zeroth coefficient real, so these two residuals vanish for a correct
    # return and any single wrong coefficient shows up in one of them.
    mirror = float(np.sum(np.abs(modes[:, order + 1:] - np.conj(modes[:, order - 1::-1]))))
    centre = float(np.sum(np.abs(modes[:, order].imag)))
    return float(rows + 7.0 * columns + 13.0 * flat.size
                 + np.sum(np.abs(flat) * np.cos(rank))
                 + np.sum(flat.real * np.sin(rank))
                 + np.sum(flat.imag * np.cos(rank))
                 + 1.0e6 * (mirror + centre))
""",
            "call": "digest(compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order))",
            "gold_call": "digest(_oracle_compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order))",
        },
        # --- Boundary: unmodulated capacity, so only the mean survives in row 1 ---
        {
            "setup": """import numpy as np
sigma_mean = 1.0
sigma_amplitudes = np.array([0.9])
sigma_phases = np.array([0.6])
capacity_mean = 50.0
capacity_amplitudes = np.array([0.0])
capacity_phases = np.array([0.0])
order = 5

def digest(values):
    modes = np.asarray(values, dtype=complex)
    if modes.ndim != 2 or modes.shape[0] != 2 or modes.shape[1] % 2 == 0:
        return -1.0
    rows, columns = modes.shape
    order = (columns - 1) // 2
    flat = modes.reshape(-1)
    rank = np.arange(1.0, flat.size + 1.0)
    # Realness of both profiles makes opposite harmonics conjugate and the
    # zeroth coefficient real, so these two residuals vanish for a correct
    # return and any single wrong coefficient shows up in one of them.
    mirror = float(np.sum(np.abs(modes[:, order + 1:] - np.conj(modes[:, order - 1::-1]))))
    centre = float(np.sum(np.abs(modes[:, order].imag)))
    return float(rows + 7.0 * columns + 13.0 * flat.size
                 + np.sum(np.abs(flat) * np.cos(rank))
                 + np.sum(flat.real * np.sin(rank))
                 + np.sum(flat.imag * np.cos(rank))
                 + 1.0e6 * (mirror + centre))
""",
            "call": "digest(compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order))",
            "gold_call": "digest(_oracle_compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order))",
        },
        # --- Edge: amplitudes that drive the conductivity profile through zero ---
        # The two harmonics are in antiphase, so they reinforce each other at the
        # far side of the cell: the profile reaches -0.3 and stays negative over
        # about a sixth of the cell, well away from a grazing touch of zero.
        {
            "setup": """import numpy as np
sigma_mean = 1.0
sigma_amplitudes = np.array([0.9, 0.4])
sigma_phases = np.array([0.0, np.pi])
capacity_mean = 100.0
capacity_amplitudes = np.array([0.5])
capacity_phases = np.array([0.0])
order = 4
def run_model():
    try:
        compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: truncation order below the highest harmonic present ---
        {
            "setup": """import numpy as np
sigma_mean = 1.0
sigma_amplitudes = np.array([0.3, 0.2, 0.1])
sigma_phases = np.array([0.0, 0.0, 0.0])
capacity_mean = 10.0
capacity_amplitudes = np.array([0.2])
capacity_phases = np.array([0.0])
order = 2
def run_model():
    try:
        compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: amplitude and phase lists of different lengths ---
        {
            "setup": """import numpy as np
sigma_mean = 1.0
sigma_amplitudes = np.array([0.3, 0.2])
sigma_phases = np.array([0.0])
capacity_mean = 10.0
capacity_amplitudes = np.array([0.2])
capacity_phases = np.array([0.0])
order = 4
def run_model():
    try:
        compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_modulation_fourier_coefficients(sigma_mean, sigma_amplitudes, sigma_phases, capacity_mean, capacity_amplitudes, capacity_phases, order)
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
