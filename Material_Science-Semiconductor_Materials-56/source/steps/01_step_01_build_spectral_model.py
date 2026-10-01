"""
Sample the isotropic phonon spectrum of the testbed material and return, for every sampled shell, its angular frequency, its group velocity and the volumetric heat capacity it contributes at the reference temperature.

A non-gray Boltzmann solver needs the phonon spectrum resolved rather than collapsed into a single grey mode, because in silicon at room temperature the modes that carry most of the heat capacity and the modes that carry most of the conductivity are not the same modes: the zone-edge modes are numerous but slow and short-lived, while the long-wavelength modes are few but travel hundreds of nanometres between collisions. Any statement about size effects is a statement about that separation, so the spectral sampling has to keep it.




The testbed replaces the true Brillouin zone by the isotropic Debye construction, a sphere in wave-vector space whose radius is fixed by requiring that it hold exactly three modes per atom, which for a number density of atoms gives a radius equal to the cube root of six times pi squared times that density. Three degenerate acoustic polarisations share one dispersion relation of Born-von Karman type, sinusoidal in the wave-vector magnitude, which reproduces both the linear low-frequency behaviour of an elastic continuum and the flattening of the branch at the zone edge; the group velocity is the analytic derivative of that relation and therefore falls to zero at the edge, which is precisely the feature a Debye model with a constant sound speed misses.




Sampling is by shells of equal width in the wave-vector magnitude, each represented by its midpoint. The number of modes per unit volume in a shell follows from the density of states of an isotropic sphere, three polarisations times the surface of the sphere over the volume of a reciprocal-space cell, and the volumetric heat capacity a shell contributes is that mode count times the temperature derivative of the Bose-Einstein energy of one mode. Written with the dimensionless ratio of the phonon energy to the thermal energy, that derivative is Boltzmann's constant times the Einstein function, which tends to Boltzmann's constant in the classical limit and is exponentially suppressed for modes far above the thermal energy. At three hundred kelvin and a maximum frequency of a few terahertz the whole spectrum sits near but not at the classical limit, so the heat capacity is close to, and slightly below, the Dulong-Petit value of three times Boltzmann's constant per atom.

Returns
-------
np.ndarray of shape (n_shells, 3), float: angular frequency (rad/s), group velocity (m/s) and volumetric heat capacity (J m^-3 K^-1) of every sampled shell, in SI units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def build_spectral_model(n_shells: int, omega_max: float, number_density: float,
                         temperature: float) -> np.ndarray:
    """Sample the isotropic phonon spectrum of the testbed material.

    Parameters
    ----------
    n_shells : int
        Number of equal-width shells in wave-vector magnitude (n_shells >= 1).
    omega_max : float
        Maximum angular frequency of the dispersion in rad/s (omega_max > 0).
    number_density : float
        Atomic number density in m^-3 (number_density > 0).
    temperature : float
        Reference temperature in K (temperature > 0).

    Returns
    -------
    spectral : np.ndarray
        Array of shape (n_shells, 3) whose columns are the angular frequency
        in rad/s, the group velocity in m/s and the volumetric heat capacity
        in J/(m^3 K) of each sampled shell.

    Raises
    ------
    ValueError
        If ``n_shells`` is not an integer at least 1, or if ``omega_max``,
        ``number_density`` or ``temperature`` is not finite and strictly
        positive.
    """
    return spectral  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_spectral_model(n_shells: int, omega_max: float,
                                 number_density: float,
                                 temperature: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    hbar = 1.054571817e-34
    k_boltzmann = 1.380649e-23

    if not (isinstance(n_shells, (int, np.integer)) and not isinstance(n_shells, bool)
            and int(n_shells) >= 1):
        raise ValueError("n_shells must be an integer >= 1")
    for name, value in (("omega_max", omega_max),
                        ("number_density", number_density),
                        ("temperature", temperature)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    n_shells = int(n_shells)
    omega_max = float(omega_max)
    number_density = float(number_density)
    temperature = float(temperature)

    # Debye sphere holding exactly three modes per atom.
    k_debye = (6.0 * np.pi ** 2 * number_density) ** (1.0 / 3.0)
    dk = k_debye / n_shells
    wavevector = (np.arange(n_shells, dtype=float) + 0.5) * dk

    # Born-von Karman dispersion and its analytic derivative.
    phase = 0.5 * np.pi * wavevector / k_debye
    omega = omega_max * np.sin(phase)
    velocity = omega_max * (0.5 * np.pi / k_debye) * np.cos(phase)

    # Mode density of the isotropic sphere times the Einstein heat capacity.
    x = hbar * omega / (k_boltzmann * temperature)
    ex = np.exp(x)
    mode_density = 3.0 * wavevector ** 2 * dk / (2.0 * np.pi ** 2)
    capacity = mode_density * k_boltzmann * x ** 2 * ex / (ex - 1.0) ** 2

    return np.column_stack([omega, velocity, capacity])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark spectral sampling (normal scenario) ---
        {
            "setup": """import numpy as np
n_shells = 400
omega_max = 2.0 * np.pi * 9.0e12
number_density = 5.0e28
temperature = 300.0
""",
            "call": "build_spectral_model(n_shells, omega_max, number_density, temperature)",
            "gold_call": "_oracle_build_spectral_model(n_shells, omega_max, number_density, temperature)",
        },
        # --- Valid: coarser sampling of a stiffer, denser lattice ---
        {
            "setup": """import numpy as np
n_shells = 64
omega_max = 2.0 * np.pi * 2.0e13
number_density = 1.76e29
temperature = 300.0
""",
            "call": "build_spectral_model(n_shells, omega_max, number_density, temperature)",
            "gold_call": "_oracle_build_spectral_model(n_shells, omega_max, number_density, temperature)",
        },
        # --- Boundary: a single shell, so the midpoint sits at half the Debye radius ---
        {
            "setup": """import numpy as np
n_shells = 1
omega_max = 2.0 * np.pi * 9.0e12
number_density = 5.0e28
temperature = 300.0
""",
            "call": "build_spectral_model(n_shells, omega_max, number_density, temperature)",
            "gold_call": "_oracle_build_spectral_model(n_shells, omega_max, number_density, temperature)",
        },
        # --- Edge: cryogenic reference temperature freezes out most of the spectrum ---
        {
            "setup": """import numpy as np
n_shells = 120
omega_max = 2.0 * np.pi * 9.0e12
number_density = 5.0e28
temperature = 10.0
""",
            "call": "build_spectral_model(n_shells, omega_max, number_density, temperature)",
            "gold_call": "_oracle_build_spectral_model(n_shells, omega_max, number_density, temperature)",
        },
        # --- Invalid: non-positive number of shells ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_spectral_model(0, 2.0 * np.pi * 9.0e12, 5.0e28, 300.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_spectral_model(0, 2.0 * np.pi * 9.0e12, 5.0e28, 300.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: absolute zero has no defined Bose-Einstein heat capacity ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_spectral_model(50, 2.0 * np.pi * 9.0e12, 5.0e28, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_spectral_model(50, 2.0 * np.pi * 9.0e12, 5.0e28, 0.0)
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
