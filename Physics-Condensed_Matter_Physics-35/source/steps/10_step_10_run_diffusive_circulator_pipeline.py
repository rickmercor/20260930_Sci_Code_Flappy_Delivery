"""
Final orchestrator. Chain every step to compare the rectification a modulated branch actually delivers with the rectification its many-cell homogenised description predicts.

The final quantity is a verdict on an approximation rather than a property of a material. The many-cell limit is the description a designer would reach for, because it collapses the entire boundary-value problem into two numbers that depend on the cell alone; the exact treatment keeps the finite number of cells, the finite branch length and the terminals where they actually are. Dividing the rectification ratio of the real branch by the rectification ratio the limiting description assigns to the same terminals measures, in one number, how much of the device behaviour the convenient description captures. A value near one means the branch is already in the asymptotic regime; a value far from one means its finite-cell effective parameters have not reached their many-cell limits; a negative value means the two descriptions do not even agree on which way the flux circulates.




Chaining the steps also exposes a consistency check that no individual step can see. The rectification ratio of the branch may be assembled in two independent ways: directly from the two exchanged time-averaged fluxes, each obtained from the amplitude of the non-decaying state and the cell average of the conductivity acting on its envelope, or from the closed form in the intrinsic drift-to-diffusion ratio extracted by the equivalence. The two routes share almost no arithmetic - one goes through the flux, the other through a logarithm of a ratio of boundary coefficients - so their agreement pins down the sign conventions, the identification of the non-decaying state, and the pairing of boundary coefficients to terminals all at once. The spectrum contains two branch-scale bulk factors, while the remaining fast-decaying states supply localized terminal corrections; their decay lengths must be distinguished when judging how far terminal layers reach. The orchestrator is also where the truncation order has to be chosen honestly: the potential profile converges quickly, but the effective parameters and the homogenised limit converge more slowly because they depend on the reciprocal of a strongly modulated conductivity, and a ratio of two such quantities is only meaningful once both have settled.

Returns
-------
float: the rectification ratio of the modulated branch divided by the rectification ratio of its many-cell homogenised description, as a native # Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_diffusive_circulator_pipeline(n_cells: int = 5,
                                      modulation_speed: float = 0.06,
                                      phi_high: float = 30.0,
                                      phi_low: float = 10.0,
                                      length: float = 1.0,
                                      sigma_mean: float = 1.0,
                                      capacity_mean: float = 100.0,
                                      sigma_amplitudes: tuple = (0.7, 0.2),
                                      sigma_phases: tuple = (0.0, 1.0471975511965976),
                                      capacity_amplitudes: tuple = (0.5,),
                                      capacity_phases: tuple = (1.2566370614359172,),
                                      order: int = 16) -> float:
    """Compare the rectification of a modulated branch with its many-cell limit.

    Parameters
    ----------
    n_cells : int
        Number of modulation cells inside the branch, at least 1.
    modulation_speed : float
        Speed of the travelling modulation in metres per second, nonzero.
    phi_high : float
        Potential imposed at the terminal at position zero, in volts.
    phi_low : float
        Potential imposed at the terminal at the far end, in volts.
    length : float
        Length of the branch in metres, strictly positive.
    sigma_mean : float
        Mean conductivity of the profile, strictly positive.
    capacity_mean : float
        Mean capacity of the profile, strictly positive.
    sigma_amplitudes : tuple
        Relative amplitudes of harmonics 1, 2, ... of the conductivity.
    sigma_phases : tuple
        Phase offsets in radians of the same conductivity harmonics.
    capacity_amplitudes : tuple
        Relative amplitudes of harmonics 1, 2, ... of the capacity.
    capacity_phases : tuple
        Phase offsets in radians of the same capacity harmonics.
    order : int
        Fourier truncation order used throughout.

    Returns
    -------
    ratio : float
        Rectification ratio of the branch divided by the rectification ratio of
        its many-cell homogenised description, as a native Python float.

    Raises
    ------
    ValueError
        If n_cells is not an integer >= 1, if length is not a finite number
        > 0, if modulation_speed is not a finite nonzero number, if the profile
        means, amplitudes, phases or truncation order do not define finite
        strictly positive modulation profiles, if a derived secular or boundary
        system is singular or inconsistent, if the branch passes no flux, if
        the flux and effective-medium routes disagree, or if the homogenised
        rectification vanishes so that the requested ratio is undefined.
    """
    return ratio  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_diffusive_circulator_pipeline(
        n_cells: int = 5, modulation_speed: float = 0.06, phi_high: float = 30.0,
        phi_low: float = 10.0, length: float = 1.0, sigma_mean: float = 1.0,
        capacity_mean: float = 100.0, sigma_amplitudes: tuple = (0.7, 0.2),
        sigma_phases: tuple = (0.0, 1.0471975511965976),
        capacity_amplitudes: tuple = (0.5,), capacity_phases: tuple = (1.2566370614359172,),
        order: int = 16) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.  Every earlier step is called through its own
    # _oracle_ twin, which the executing namespace supplies, so the reference
    # chain never runs a submitted implementation.
    import numpy as np

    def _number(value, positive=False, nonzero=False):
        return (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value) and (float(value) > 0.0 or not positive)
                and (float(value) != 0.0 or not nonzero))

    if not (isinstance(n_cells, (int, np.integer)) and not isinstance(n_cells, bool)
            and int(n_cells) >= 1):
        raise ValueError("n_cells must be an integer >= 1")
    if not _number(length, positive=True):
        raise ValueError("length must be a finite number > 0")
    if not _number(modulation_speed, nonzero=True):
        raise ValueError("modulation_speed must be a finite nonzero number")

    beta = 2.0 * np.pi * int(n_cells) / float(length)
    modes = _oracle_compute_modulation_fourier_coefficients(
        sigma_mean, np.asarray(sigma_amplitudes, dtype=float),
        np.asarray(sigma_phases, dtype=float), capacity_mean,
        np.asarray(capacity_amplitudes, dtype=float),
        np.asarray(capacity_phases, dtype=float), order)
    sigma_modes, capacity_modes = modes[0], modes[1]
    middle = (np.asarray(sigma_modes).size - 1) // 2
    alphas = _oracle_solve_secular_roots(sigma_modes, capacity_modes, beta, modulation_speed)
    envelopes = np.column_stack([_oracle_compute_bloch_fourier_components(
        a, sigma_modes, capacity_modes, beta, modulation_speed) for a in alphas])
    star = int(np.argmin(np.abs(alphas)))
    star_envelope = envelopes[:, star]

    # Each envelope must be annihilated by the coupling matrix at its own
    # decay constant - tightly for the non-decaying state, loosely for the
    # fastest decaying one - and the matrix must carry the conductivity
    # quadratically far from every root.
    for index, tolerance in ((star, 1.0e-12), (int(np.argmax(np.abs(alphas))), 1.0e-6)):
        coupling = np.asarray(_oracle_assemble_secular_matrix(
            alphas[index], sigma_modes, capacity_modes, beta, modulation_speed))
        column = envelopes[:, index]
        if np.linalg.norm(coupling @ column) > tolerance * np.linalg.norm(
                coupling) * np.linalg.norm(column):
            raise ValueError("an envelope does not annihilate the coupling matrix")
    far = np.asarray(_oracle_assemble_secular_matrix(
        1.0e6, sigma_modes, capacity_modes, beta, modulation_speed))
    if abs(far[middle, middle] / 1.0e12 - sigma_modes[middle]) > 1.0e-6 * abs(sigma_modes[middle]):
        raise ValueError("the coupling matrix does not carry the conductivity at leading order")

    coefficients = _oracle_compute_boundary_coefficients(alphas, envelopes, 0.0, float(length))
    forward = _oracle_compute_time_averaged_flux(sigma_modes, star_envelope, beta, coefficients, phi_high, phi_low)
    backward = _oracle_compute_time_averaged_flux(sigma_modes, star_envelope, beta, coefficients, phi_low, phi_high)
    largest = max(abs(forward), abs(backward))
    if largest == 0.0:
        raise ValueError("the branch passes no flux in either direction")
    branch = (forward + backward) / largest
    parameters = _oracle_compute_effective_parameters(sigma_modes, star_envelope, beta, coefficients, length)
    closed_form = _oracle_compute_rectification_ratio(float(parameters[2]), length, phi_high, phi_low)
    if abs(closed_form - branch) > 1.0e-6 * max(1.0, abs(branch)):
        raise ValueError("the flux and equivalence routes disagree on the rectification")

    limit = _oracle_compute_homogenized_limit(sigma_modes, capacity_modes, modulation_speed)
    limiting = _oracle_compute_rectification_ratio(float(limit[2]), length, phi_high, phi_low)
    if limiting == 0.0:
        raise ValueError("the homogenised description shows no rectification to compare with")
    return float(branch / limiting)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark configuration whose result is the final answer ---
        {
            "setup": """import numpy as np
n_cells = 5
""",
            "call": "run_diffusive_circulator_pipeline(n_cells)",
            "gold_call": "_oracle_run_diffusive_circulator_pipeline(n_cells)",
        },
        # --- Valid: many cells, where the branch approaches its homogenised limit ---
        {
            "setup": """import numpy as np
n_cells = 24
order = 10
""",
            "call": "run_diffusive_circulator_pipeline(n_cells, order=order)",
            "gold_call": "_oracle_run_diffusive_circulator_pipeline(n_cells, order=order)",
        },
        # --- Boundary: a single modulation cell in the branch ---
        {
            "setup": """import numpy as np
n_cells = 1
order = 8
""",
            "call": "run_diffusive_circulator_pipeline(n_cells, order=order)",
            "gold_call": "_oracle_run_diffusive_circulator_pipeline(n_cells, order=order)",
        },
        # --- Edge: reversed travelling wave and a much faster modulation ---
        {
            "setup": """import numpy as np
n_cells = 5
modulation_speed = -0.2
order = 8
""",
            "call": "run_diffusive_circulator_pipeline(n_cells, modulation_speed, order=order)",
            "gold_call": "_oracle_run_diffusive_circulator_pipeline(n_cells, modulation_speed, order=order)",
        },
        # --- Invalid: a static medium, which cannot rectify at all ---
        {
            "setup": """import numpy as np
n_cells = 5
modulation_speed = 0.0
def run_model():
    try:
        run_diffusive_circulator_pipeline(n_cells, modulation_speed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_diffusive_circulator_pipeline(n_cells, modulation_speed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a branch with no cells in it ---
        {
            "setup": """import numpy as np
n_cells = 0
def run_model():
    try:
        run_diffusive_circulator_pipeline(n_cells)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_diffusive_circulator_pipeline(n_cells)
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
