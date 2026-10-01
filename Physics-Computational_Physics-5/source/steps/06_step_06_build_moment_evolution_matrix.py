"""
Assemble the linear operator that advances the three normalised Fourier moments of an electrostatic electron plasma once a heat-flux closure has been supplied.

Linearising the density, momentum and pressure equations about a uniform Maxwellian and eliminating the potential with the Poisson equation leaves a closed three-by-three system in Fourier space, provided the heat flux is expressed through the other moments. The closure parameters enter that system only through its pressure row, so all of the closure physics sits in three matrix entries.

Returns
-------
np.ndarray of shape (3, 3), complex: the linear evolution operator of the closed three-moment system.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_moment_evolution_matrix(wavenumber: float,
                                  closure_parameters: np.ndarray) -> np.ndarray:
    """Assemble the three-moment evolution operator at one wave number.

    The state is ordered as the normalised density, velocity and pressure
    perturbations. Define v_t = sqrt(T0/m), where T0 is the equilibrium
    temperature in energy units and m is the electron mass. Density is
    normalised by the equilibrium density, velocity by sqrt(2)*v_t, pressure
    by the equilibrium pressure P0, heat flux by P0*sqrt(2)*v_t, and
    electrostatic potential by T0/e, where e is the elementary charge. Time
    is measured in inverse plasma frequencies and the wave vector points
    along the positive axis. The returned operator ``M`` is the one for
    which the state derivative equals ``M`` acting on the state.

    Parameters
    ----------
    wavenumber : float
        Perturbation wave number divided by the Debye wave number of the
        equilibrium, strictly positive.
    closure_parameters : np.ndarray
        Array of shape (3,) of real closure parameters as returned by
        sub-problem 05.

    Returns
    -------
    evolution_matrix : np.ndarray
        Complex array of shape (3, 3), the linear operator of the closed
        three-moment system in the state ordering above.

    Raises
    ------
    ValueError
        If ``wavenumber`` is not a finite number greater than zero, or if
        ``closure_parameters`` is not a finite real array of shape (3,).
    """
    return evolution_matrix  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_moment_evolution_matrix(wavenumber: float,
                                          closure_parameters: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    if not (isinstance(wavenumber, (int, float, np.floating, np.integer))
            and not isinstance(wavenumber, bool) and np.isfinite(wavenumber)
            and float(wavenumber) > 0.0):
        raise ValueError("wavenumber must be a finite number greater than zero")
    wavenumber = float(wavenumber)

    parameters = np.asarray(closure_parameters)
    if np.iscomplexobj(parameters):
        raise ValueError("closure_parameters must be a real array")
    parameters = parameters.astype(float)
    if parameters.shape != (3,):
        raise ValueError("closure_parameters must be an array of shape (3,)")
    if not np.all(np.isfinite(parameters)):
        raise ValueError("closure_parameters must contain only finite entries")

    velocity, potential, temperature = (float(value) for value in parameters)
    root_two = np.sqrt(2.0)
    matrix = np.zeros((3, 3), dtype=complex)

    # Continuity: the density is driven by the compression of the velocity.
    matrix[0, 1] = -1j * root_two * wavenumber
    # Momentum: the pressure gradient and, through the Poisson equation, the
    # space-charge restoring force, which is the term that carries the inverse
    # wave number.
    matrix[1, 0] = -1j / (root_two * wavenumber)
    matrix[1, 2] = -1j * wavenumber / root_two
    # Pressure: adiabatic compression plus the divergence of the closed heat
    # flux. The potential is eliminated in favour of the density there too, so
    # the potential-corrected pressure coefficient also reaches the density
    # column, divided by the wave number.
    matrix[2, 0] = root_two * (potential / wavenumber + wavenumber * temperature)
    matrix[2, 1] = -1j * root_two * wavenumber * (3.0 + velocity)
    matrix[2, 2] = root_two * wavenumber * (potential - temperature)
    return matrix

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the wave-number-dependent closure at the benchmark wave
        #     number (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a)
    shape = float(a.ndim) + sum((j + 2.0) * n for j, n in enumerate(a.shape))
    a = a.ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s + shape), 9)
wavenumber = 0.4
closure_parameters = np.array([1.354207870, 0.0, 0.780864445])
""",
            "call": "sig(build_moment_evolution_matrix(wavenumber, closure_parameters), 1.0)",
            "gold_call": "sig(_oracle_build_moment_evolution_matrix(wavenumber, closure_parameters), 1.0)",
        },
        # --- Valid: the Hammett-Perkins closure at the same wave number, which
        #     leaves the velocity and potential entries bare ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a)
    shape = float(a.ndim) + sum((j + 2.0) * n for j, n in enumerate(a.shape))
    a = a.ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s + shape), 9)
wavenumber = 0.4
closure_parameters = np.array([0.0, 0.0, 2.0 / np.sqrt(np.pi)])
""",
            "call": "sig(build_moment_evolution_matrix(wavenumber, closure_parameters), 1.0)",
            "gold_call": "sig(_oracle_build_moment_evolution_matrix(wavenumber, closure_parameters), 1.0)",
        },
        # --- Valid: a closure with a non-vanishing potential coefficient at a long
        #     wavelength, where the space-charge entry is largest ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a)
    shape = float(a.ndim) + sum((j + 2.0) * n for j, n in enumerate(a.shape))
    a = a.ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s + shape), 9)
wavenumber = 0.1
closure_parameters = np.array([4.062513306, 0.581824262, 3.420406857])
""",
            "call": "sig(build_moment_evolution_matrix(wavenumber, closure_parameters), 10.0)",
            "gold_call": "sig(_oracle_build_moment_evolution_matrix(wavenumber, closure_parameters), 10.0)",
        },
        # --- Boundary: a vanishing closure, which reduces the system to the
        #     adiabatic three-moment fluid with no heat flux at all ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a)
    shape = float(a.ndim) + sum((j + 2.0) * n for j, n in enumerate(a.shape))
    a = a.ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s + shape), 9)
wavenumber = 0.6
closure_parameters = np.zeros(3)
""",
            "call": "sig(build_moment_evolution_matrix(wavenumber, closure_parameters), 1.0)",
            "gold_call": "sig(_oracle_build_moment_evolution_matrix(wavenumber, closure_parameters), 1.0)",
        },
        # --- Edge: a short wavelength with a strongly damping closure ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a)
    shape = float(a.ndim) + sum((j + 2.0) * n for j, n in enumerate(a.shape))
    a = a.ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s + shape), 9)
wavenumber = 1.5
closure_parameters = np.array([2.9, -0.4, 2.2])
""",
            "call": "sig(build_moment_evolution_matrix(wavenumber, closure_parameters), 1.0)",
            "gold_call": "sig(_oracle_build_moment_evolution_matrix(wavenumber, closure_parameters), 1.0)",
        },
        # --- Invalid: a vanishing wave number, at which the space-charge entry
        #     diverges ---
        {
            "setup": """import numpy as np
closure_parameters = np.array([1.0, 0.0, 1.0])
def run_model():
    try:
        build_moment_evolution_matrix(0.0, closure_parameters)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_moment_evolution_matrix(0.0, closure_parameters)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a wrongly sized closure parameter list ---
        {
            "setup": """import numpy as np
closure_parameters = np.array([1.0, 0.0])
def run_model():
    try:
        build_moment_evolution_matrix(0.4, closure_parameters)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_moment_evolution_matrix(0.4, closure_parameters)
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
