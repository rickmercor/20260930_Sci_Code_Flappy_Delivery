"""
Fix the strength of the anharmonic scattering term of the lifetime model by requiring the bulk relaxation-time-approximation conductivity of the sampled spectrum to reproduce a prescribed value.

The lifetime model of the testbed is the standard two-channel form used since Callaway and Holland: a Rayleigh term whose rate grows as the fourth power of the frequency, representing elastic scattering off isotopes and point defects, and an anharmonic term whose rate grows as the square of the frequency, representing the umklapp three-phonon processes that alone can degrade a heat current. The two channels are added as rates, so the shorter lifetime dominates, and their competition is what produces the wide spread of mean free paths that makes the size effect non-trivial: the anharmonic term saturates the long-wavelength lifetimes at a value that grows without bound as the frequency falls, while the Rayleigh term truncates the high-frequency end far more aggressively.




Only the shape of that competition is fixed a priori. The Rayleigh coefficient is a material constant that can be computed from the isotopic composition, but the anharmonic coefficient carries the temperature dependence and the anharmonicity of the crystal and is in practice fitted. The calibration used here is the standard one: choose it so that the kinetic-theory conductivity of the sampled spectrum, one third of the sum over shells of heat capacity times squared group velocity times lifetime, matches the measured bulk value at the reference temperature. This is not cosmetic. The quantity the whole calculation reports is a ratio of a kinetic prediction to a Fourier prediction that uses that same bulk conductivity, so if the spectral model did not reproduce it the ratio would be contaminated by a mismatch between numerator and denominator that has nothing to do with size effects.




The calibration is a one-dimensional root problem with useful structure: increasing the anharmonic coefficient shortens every lifetime, so the conductivity is strictly decreasing in it and the root is unique. The coefficient itself spans many orders of magnitude across materials and temperatures, so the search is naturally carried out on its logarithm, over a bracket wide enough that the conductivity is above the target at one end and below it at the other; a solver that brackets and bisects converges to machine precision in a fixed number of steps and needs no derivative.

Returns
-------
float: the anharmonic scattering coefficient in seconds that reproduces the target bulk conductivity, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def calibrate_umklapp_coefficient(spectral: np.ndarray, impurity: float,
                                  kappa_bulk: float) -> float:
    """Fix the anharmonic scattering coefficient from the bulk conductivity.

    Parameters
    ----------
    spectral : np.ndarray
        Array of shape (n_shells, 3) holding the angular frequency, group
        velocity and volumetric heat capacity of every sampled shell.
    impurity : float
        Coefficient of the fourth-power scattering channel in s^3
        (impurity >= 0).
    kappa_bulk : float
        Target bulk thermal conductivity in W/(m K) (kappa_bulk > 0).

    Returns
    -------
    umklapp : float
        Coefficient of the second-power scattering channel in s, as a native
        Python float, for which the relaxation-time-approximation
        conductivity of the sampled spectrum equals kappa_bulk.

    Raises
    ------
    ValueError
        If ``spectral`` is not a finite ``(n_shells, 3)`` array with positive
        frequencies and non-negative heat capacities, if ``impurity`` is not
        finite and non-negative, if ``kappa_bulk`` is not finite and strictly
        positive, or if no admissible coefficient brackets the target.
    """
    return umklapp  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_calibrate_umklapp_coefficient(spectral: np.ndarray, impurity: float,
                                          kappa_bulk: float) -> float:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    table = np.asarray(spectral, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 1:
        raise ValueError("spectral must be a 2D array of shape (n_shells, 3)")
    if not np.all(np.isfinite(table)):
        raise ValueError("spectral must be finite")
    if np.any(table[:, 0] <= 0.0) or np.any(table[:, 2] < 0.0):
        raise ValueError("spectral frequencies must be > 0 and heat capacities >= 0")
    for name, value, floor in (("impurity", impurity, 0.0),
                               ("kappa_bulk", kappa_bulk, None)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if float(impurity) < 0.0:
        raise ValueError("impurity must be a finite number >= 0")
    if float(kappa_bulk) <= 0.0:
        raise ValueError("kappa_bulk must be a finite number > 0")

    omega, velocity, capacity = table[:, 0], table[:, 1], table[:, 2]
    impurity = float(impurity)
    kappa_bulk = float(kappa_bulk)

    def _residual(exponent):
        # Kinetic-theory conductivity of the sampled spectrum, minus the target.
        lifetime = 1.0 / (impurity * omega ** 4 + (10.0 ** exponent) * omega ** 2)
        return float(np.sum(capacity * velocity ** 2 * lifetime) / 3.0) - kappa_bulk

    low, high = -40.0, 20.0
    if _residual(low) < 0.0 or _residual(high) > 0.0:
        raise ValueError("no admissible umklapp coefficient in the search bracket")

    # The conductivity decreases monotonically with the coefficient, so plain
    # bisection on the exponent is unconditionally convergent.
    for _ in range(200):
        mid = 0.5 * (low + high)
        if _residual(mid) > 0.0:
            low = mid
        else:
            high = mid

    return float(10.0 ** (0.5 * (low + high)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark silicon-like calibration (normal scenario) ---
        {
            "setup": """import numpy as np
hbar, kb = 1.054571817e-34, 1.380649e-23
n_shells, omega_max, density, temperature = 400, 2.0 * np.pi * 9.0e12, 5.0e28, 300.0
kd = (6.0 * np.pi ** 2 * density) ** (1.0 / 3.0)
dk = kd / n_shells
k = (np.arange(n_shells) + 0.5) * dk
phase = 0.5 * np.pi * k / kd
omega = omega_max * np.sin(phase)
velocity = omega_max * (0.5 * np.pi / kd) * np.cos(phase)
x = hbar * omega / (kb * temperature)
capacity = (3.0 * k ** 2 * dk / (2.0 * np.pi ** 2)) * kb * x ** 2 * np.exp(x) / (np.exp(x) - 1.0) ** 2
spectral = np.column_stack([omega, velocity, capacity])
impurity = 1.32e-45
kappa_bulk = 148.0
""",
            "call": "1.0e17 * calibrate_umklapp_coefficient(spectral, impurity, kappa_bulk)",
            "gold_call": "1.0e17 * _oracle_calibrate_umklapp_coefficient(spectral, impurity, kappa_bulk)",
        },
        # --- Valid: much lower target conductivity on the same spectrum ---
        {
            "setup": """import numpy as np
omega = np.linspace(1.0e12, 5.0e13, 200)
velocity = np.full(200, 6000.0)
capacity = np.full(200, 8.0e3)
spectral = np.column_stack([omega, velocity, capacity])
impurity = 1.32e-45
kappa_bulk = 5.0
""",
            "call": "1.0e17 * calibrate_umklapp_coefficient(spectral, impurity, kappa_bulk)",
            "gold_call": "1.0e17 * _oracle_calibrate_umklapp_coefficient(spectral, impurity, kappa_bulk)",
        },
        # --- Boundary: no point-defect channel at all ---
        {
            "setup": """import numpy as np
omega = np.linspace(2.0e12, 4.0e13, 128)
velocity = np.linspace(6000.0, 1000.0, 128)
capacity = np.linspace(1.0e3, 2.0e4, 128)
spectral = np.column_stack([omega, velocity, capacity])
impurity = 0.0
kappa_bulk = 90.0
""",
            "call": "1.0e17 * calibrate_umklapp_coefficient(spectral, impurity, kappa_bulk)",
            "gold_call": "1.0e17 * _oracle_calibrate_umklapp_coefficient(spectral, impurity, kappa_bulk)",
        },
        # --- Edge: single shell, so the calibration has a closed-form root ---
        {
            "setup": """import numpy as np
spectral = np.array([[1.0e13, 5000.0, 1.0e6]])
impurity = 1.0e-45
kappa_bulk = 100.0
""",
            "call": "1.0e17 * calibrate_umklapp_coefficient(spectral, impurity, kappa_bulk)",
            "gold_call": "1.0e17 * _oracle_calibrate_umklapp_coefficient(spectral, impurity, kappa_bulk)",
        },
        # --- Invalid: target unreachable because point-defect scattering alone caps the conductivity ---
        {
            "setup": """import numpy as np
omega = np.linspace(1.0e13, 5.0e13, 64)
velocity = np.full(64, 4000.0)
capacity = np.full(64, 1.0e3)
spectral = np.column_stack([omega, velocity, capacity])
def run_model():
    try:
        calibrate_umklapp_coefficient(spectral, 1.0e-44, 1.0e6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_calibrate_umklapp_coefficient(spectral, 1.0e-44, 1.0e6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: spectral table with the wrong number of columns ---
        {
            "setup": """import numpy as np
spectral = np.ones((10, 2))
def run_model():
    try:
        calibrate_umklapp_coefficient(spectral, 1.32e-45, 148.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_calibrate_umklapp_coefficient(spectral, 1.32e-45, 148.0)
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
