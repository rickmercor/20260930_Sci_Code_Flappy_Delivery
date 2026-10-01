"""
Return the trap depth and the infinite-temperature capture cross section of a level from its emission rates measured across a set of temperatures.

The emission rate carries the trap depth in an exponential and the capture cross section in a prefactor, and the prefactor also carries the temperature dependence of the thermal velocity and of the effective density of states. Dividing the rate by the square of the temperature cancels that dependence exactly, so the logarithm of the reduced rate is strictly linear in reciprocal thermal energy. In those coordinates a straight-line regression separates the two unknowns cleanly: the slope is the negative trap depth, and the intercept is the logarithm of the cross section multiplied by the temperature-independent constant that the effective mass alone determines.




Recovering the cross section therefore requires dividing out that constant, and getting it wrong rescales the cross section without touching the trap depth, which is why cross sections quoted in the literature vary while activation energies agree. The constant is the product of the two prefactor coefficients: the square root of three times the Boltzmann constant over the effective mass, and twice the three-halves power of two pi times the effective mass times the Boltzmann constant over Planck's constant squared.




Because the intercept is an extrapolation to infinite temperature from a window of a few hundred kelvin, the cross section is exponentially sensitive to any error in the slope: a shift of ten millielectronvolts in the fitted depth moves the extrapolated cross section by tens of percent. A level whose rates are recovered slightly imprecisely - typically the weaker of two overlapping levels - therefore returns an activation energy that is only mildly wrong and a cross section that can be wrong by a large factor, and the smoothed rates that this regression supplies to the amplitude stage inherit that error.

Returns
-------
np.ndarray of shape (2,), float: trap depth (eV) and infinite-temperature capture cross section (cm^2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fit_arrhenius_parameters(temperature: np.ndarray, emission_rate: np.ndarray,
                             mass_ratio: float = 0.063) -> np.ndarray:
    """Return the trap depth and infinite-temperature cross section of a level.

    Parameters
    ----------
    temperature : np.ndarray
        Temperatures in K at which the emission rate was determined, all
        strictly positive and at least two distinct values.
    emission_rate : np.ndarray
        Emission rate in 1/s at each temperature, all strictly positive, same
        length as temperature.
    mass_ratio : float
        Carrier effective mass of the receiving band in units of the free
        electron mass (mass_ratio > 0).

    Returns
    -------
    parameters : np.ndarray
        Array of shape (2,) holding the trap depth in eV and the
        infinite-temperature capture cross section in cm^2.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return parameters  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fit_arrhenius_parameters(temperature: np.ndarray, emission_rate: np.ndarray,
                                     mass_ratio: float = 0.063) -> np.ndarray:
    import numpy as np

    kb_ev = 8.617333262e-5
    kb_j = 1.380649e-23
    h_planck = 6.62607015e-34
    m_e = 9.1093837015e-31

    temperature = np.asarray(temperature, dtype=float)
    emission_rate = np.asarray(emission_rate, dtype=float)

    if (isinstance(mass_ratio, bool)
            or not isinstance(mass_ratio, (int, float, np.floating, np.integer))
            or not np.isfinite(mass_ratio) or float(mass_ratio) <= 0.0):
        raise ValueError("mass_ratio must be a finite number > 0")
    if temperature.ndim != 1 or temperature.shape != emission_rate.shape:
        raise ValueError("temperature and emission_rate must be one-dimensional and equal length")
    if temperature.size < 2 or np.unique(temperature).size < 2:
        raise ValueError("at least two distinct temperatures are required")
    if np.any(temperature <= 0.0) or np.any(emission_rate <= 0.0):
        raise ValueError("temperature and emission_rate must be strictly positive")

    mass = float(mass_ratio) * m_e

    # Temperature-independent constant left in the prefactor once the rate has
    # been divided by the square of the temperature.
    prefactor = (np.sqrt(3.0 * kb_j / mass) * 100.0
                 * 2.0 * (2.0 * np.pi * mass * kb_j / h_planck ** 2) ** 1.5 * 1e-6)

    abscissa = 1.0 / (kb_ev * temperature)
    ordinate = np.log(emission_rate / temperature ** 2)

    slope, intercept = np.polyfit(abscissa, ordinate, 1)

    depth = -float(slope)
    cross_section = float(np.exp(intercept) / prefactor)

    return np.array([depth, cross_section], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: noiseless rates of the dominant level recover their input ---
        {
            "setup": """import numpy as np
kb = 8.617333262e-5
kj = 1.380649e-23
h = 6.62607015e-34
m = 0.063 * 9.1093837015e-31
T = np.arange(372.0, 450.0 + 1e-9, 3.0)
v = np.sqrt(3.0 * kj * T / m) * 100.0
nc = 2.0 * (2.0 * np.pi * m * kj * T / h ** 2) ** 1.5 * 1e-6
e = 1.8e-15 * v * nc * np.exp(-0.711 / (kb * T))
""",
            "call": "fit_arrhenius_parameters(T, e) * np.array([1.0e3, 1.0e18]) + 1000.0",
            "gold_call": "_oracle_fit_arrhenius_parameters(T, e) * np.array([1.0e3, 1.0e18]) + 1000.0",
        },
        # --- Valid: the cross section of the same level ---
        {
            "setup": """import numpy as np
kb = 8.617333262e-5
kj = 1.380649e-23
h = 6.62607015e-34
m = 0.063 * 9.1093837015e-31
T = np.arange(372.0, 450.0 + 1e-9, 3.0)
v = np.sqrt(3.0 * kj * T / m) * 100.0
nc = 2.0 * (2.0 * np.pi * m * kj * T / h ** 2) ** 1.5 * 1e-6
e = 9.1e-15 * v * nc * np.exp(-0.658 / (kb * T))
""",
            "call": "fit_arrhenius_parameters(T, e) * np.array([1.0e3, 1.0e18]) + 1000.0",
            "gold_call": "_oracle_fit_arrhenius_parameters(T, e) * np.array([1.0e3, 1.0e18]) + 1000.0",
        },
        # --- Valid: scattered rates give a least-squares compromise ---
        {
            "setup": """import numpy as np
T = np.array([372.0, 390.0, 408.0, 426.0, 444.0])
e = np.array([12.4, 39.5, 118.0, 331.0, 880.0])
""",
            "call": "fit_arrhenius_parameters(T, e) * np.array([1.0e3, 1.0e18]) + 1000.0",
            "gold_call": "_oracle_fit_arrhenius_parameters(T, e) * np.array([1.0e3, 1.0e18]) + 1000.0",
        },
        # --- Boundary: two temperatures determine the line exactly ---
        {
            "setup": """import numpy as np
T = np.array([372.0, 450.0])
e = np.array([12.4, 880.0])
""",
            "call": "fit_arrhenius_parameters(T, e) * np.array([1.0e3, 1.0e18]) + 1000.0",
            "gold_call": "_oracle_fit_arrhenius_parameters(T, e) * np.array([1.0e3, 1.0e18]) + 1000.0",
        },
        # --- Edge: a heavier effective mass rescales only the cross section ---
        {
            "setup": """import numpy as np
T = np.array([372.0, 390.0, 408.0, 426.0, 444.0])
e = np.array([12.4, 39.5, 118.0, 331.0, 880.0])
""",
            "call": "fit_arrhenius_parameters(T, e, 0.28) * np.array([1.0e3, 1.0e18]) + 1000.0",
            "gold_call": "_oracle_fit_arrhenius_parameters(T, e, 0.28) * np.array([1.0e3, 1.0e18]) + 1000.0",
        },
        # --- Invalid: a non-positive emission rate cannot be logged ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        fit_arrhenius_parameters(np.array([372.0, 450.0]), np.array([0.0, 880.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_fit_arrhenius_parameters(np.array([372.0, 450.0]), np.array([0.0, 880.0]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a single distinct temperature cannot fix a slope ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        fit_arrhenius_parameters(np.array([400.0, 400.0]), np.array([12.4, 12.5]))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_fit_arrhenius_parameters(np.array([400.0, 400.0]), np.array([12.4, 12.5]))
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
