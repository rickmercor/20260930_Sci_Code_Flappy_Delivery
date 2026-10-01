"""
Reduce a recorded moment history to the history of the perturbed electrostatic field amplitude that the Poisson equation ties to it.

With immobile ions the electrostatic field of a single Fourier mode is fixed entirely by the electron density perturbation through the Poisson equation, so the field carries no information the density does not, only a different weighting with wave number. It is nevertheless the field, and its energy, that a kinetic benchmark reports.

Returns
-------
np.ndarray of shape (n_levels,), complex: the normalised perturbed electrostatic field amplitude at each level.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_field_amplitude_history(moment_history: np.ndarray,
                                    wavenumber: float) -> np.ndarray:
    """Reduce a moment history to the perturbed electrostatic field amplitude.

    The field is normalised by E0 = T0*k_p/e, where T0 is the equilibrium
    temperature in energy units, k_p is the Debye wave number, and e is the
    elementary charge. This agrees with the potential normalisation T0/e in
    sub-problem 06. The wave vector points along the positive axis.

    Parameters
    ----------
    moment_history : np.ndarray
        Complex array of shape (n_levels, 3) as returned by sub-problem 07,
        holding the normalised density, velocity and pressure perturbations at
        each time level.
    wavenumber : float
        Perturbation wave number divided by the Debye wave number of the
        equilibrium, strictly positive.

    Returns
    -------
    field_history : np.ndarray
        Complex array of shape (n_levels,) holding the normalised perturbed
        electrostatic field amplitude at each time level.

    Raises
    ------
    ValueError
        If ``moment_history`` is not a finite two-dimensional array with three
        columns and at least one row, or if ``wavenumber`` is not a finite
        number greater than zero.
    """
    return field_history  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_field_amplitude_history(moment_history: np.ndarray,
                                            wavenumber: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    history = np.asarray(moment_history, dtype=complex)
    if history.ndim != 2 or history.shape[1] != 3 or history.shape[0] < 1:
        raise ValueError("moment_history must be two-dimensional with three columns")
    if not np.all(np.isfinite(history)):
        raise ValueError("moment_history must contain only finite entries")

    if not (isinstance(wavenumber, (int, float, np.floating, np.integer))
            and not isinstance(wavenumber, bool) and np.isfinite(wavenumber)
            and float(wavenumber) > 0.0):
        raise ValueError("wavenumber must be a finite number greater than zero")
    wavenumber = float(wavenumber)

    # Poisson ties the normalised potential to minus the density divided by the
    # squared wave number; the field is minus the gradient of the potential, so
    # in Fourier space it is the density divided by the wave number, rotated by
    # a quarter turn.
    return 1j * history[:, 0] / wavenumber

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a short recorded history at the benchmark wave number
        #     (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
levels = np.arange(6, dtype=float)
moment_history = np.stack([
    0.02 * np.exp(-0.05 * levels) * np.exp(-1j * 0.4 * levels),
    0.01 * np.exp(-1j * 0.4 * levels),
    0.02 * np.exp(-0.02 * levels)], axis=1)
wavenumber = 0.4
""",
            "call": "sig(compute_field_amplitude_history(moment_history, wavenumber), 1.0)",
            "gold_call": "sig(_oracle_compute_field_amplitude_history(moment_history, wavenumber), 1.0)",
        },
        # --- Valid: a long-wavelength history, where the field is amplified
        #     relative to the density ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
levels = np.arange(9, dtype=float)
moment_history = np.stack([
    0.02 * np.cos(0.3 * levels) + 0.0j,
    -0.005 * np.sin(0.3 * levels) + 0.0j,
    0.02 * np.cos(0.3 * levels) + 0.0j], axis=1)
wavenumber = 0.05
""",
            "call": "sig(compute_field_amplitude_history(moment_history, wavenumber), 1.0)",
            "gold_call": "sig(_oracle_compute_field_amplitude_history(moment_history, wavenumber), 1.0)",
        },
        # --- Boundary: a single recorded level ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
moment_history = np.array([[0.02 + 0.0j, 0.0 + 0.0j, 0.02 + 0.0j]])
wavenumber = 0.4
""",
            "call": "sig(compute_field_amplitude_history(moment_history, wavenumber), 1.0)",
            "gold_call": "sig(_oracle_compute_field_amplitude_history(moment_history, wavenumber), 1.0)",
        },
        # --- Edge: a history whose density column has been fully damped away,
        #     leaving the other moments alive but the field zero ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k) + 1.0) / s), 9)
moment_history = np.zeros((5, 3), dtype=complex)
moment_history[:, 1] = 0.3
moment_history[:, 2] = -0.7
wavenumber = 0.6
""",
            "call": "sig(compute_field_amplitude_history(moment_history, wavenumber), 1.0)",
            "gold_call": "sig(_oracle_compute_field_amplitude_history(moment_history, wavenumber), 1.0)",
        },
        # --- Invalid: a history with the wrong number of moment columns ---
        {
            "setup": """import numpy as np
moment_history = np.zeros((4, 2), dtype=complex)
def run_model():
    try:
        compute_field_amplitude_history(moment_history, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_field_amplitude_history(moment_history, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a vanishing wave number, at which the field normalisation
        #     diverges ---
        {
            "setup": """import numpy as np
moment_history = np.zeros((4, 3), dtype=complex)
def run_model():
    try:
        compute_field_amplitude_history(moment_history, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_field_amplitude_history(moment_history, 0.0)
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
