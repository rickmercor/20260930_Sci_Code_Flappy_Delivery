"""
Return the thermal emission rate of a deep level with a given activation energy and high-temperature capture cross section, at each of a set of sample temperatures.

A deep level that has captured a majority carrier releases it thermally at a rate set by three factors: how often a carrier confined to the defect attempts to escape, how many band states are available to receive it, and the Boltzmann penalty for surmounting the trap depth. Detailed balance between capture and emission turns those factors into a product of the capture cross section extrapolated to infinite temperature, the thermal velocity of the free carrier, and the effective density of states of the receiving band, multiplied by the exponential of the negative trap depth over the thermal energy.




Two conventions for the thermal velocity are in circulation and they differ by a fixed factor of about 0.92, so the one in force has to be named rather than described. Deep-level spectroscopy uses the root-mean-square speed of a three-dimensional Maxwellian, v = sqrt(3 k_B T / m*), not the Maxwell mean speed sqrt(8 k_B T / (pi m*)); the effective density of states is the standard N_C = 2 (2 pi m* k_B T / h^2)^(3/2). Both are expressed per cubic centimetre and per second here, to match a capture cross section quoted in square centimetres.




That velocity grows as the square root of temperature and the effective density of states as the three-halves power, so the whole prefactor scales as the square of the temperature. That is why deep-level spectroscopies plot the logarithm of the emission rate divided by the square of the temperature against reciprocal temperature: in those coordinates the temperature dependence of the prefactor is removed exactly, the slope returns the trap depth alone, and the intercept returns the capture cross section alone. The temperature-independent constant that remains in the prefactor is fixed entirely by the effective mass of the band the carrier is emitted into.




Because the emission rate depends exponentially on the trap depth, two levels separated by only a few tens of millielectronvolts can differ in emission rate by more than an order of magnitude at a fixed temperature while their spectroscopic peaks still overlap when the temperature is scanned. Establishing the emission rate as a function of temperature is therefore the first step of any quantitative analysis of a transient that contains more than one level.

Returns
-------
np.ndarray of shape (n_temperatures,), float: thermal emission rate in 1/s.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_emission_rate(temperature: np.ndarray, activation_energy: float,
                          sigma_inf: float, mass_ratio: float = 0.063) -> np.ndarray:
    """Return the thermal emission rate of a deep level at each temperature.

    Parameters
    ----------
    temperature : np.ndarray
        Sample temperatures in K, all strictly positive.
    activation_energy : float
        Trap depth below the receiving band edge in eV (activation_energy > 0).
    sigma_inf : float
        Capture cross section extrapolated to infinite temperature in cm^2
        (sigma_inf > 0).
    mass_ratio : float
        Carrier effective mass of the receiving band in units of the free
        electron mass (mass_ratio > 0).

    Returns
    -------
    rate : np.ndarray
        Emission rate in 1/s, one entry per input temperature, equal to
        sigma_inf * v_th * N_C * exp(-activation_energy / (k_B T)) with the
        root-mean-square thermal velocity v_th = sqrt(3 k_B T / m*) in cm/s and
        the effective density of states N_C = 2 (2 pi m* k_B T / h^2)^(3/2) in
        1/cm^3.

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.

    Notes
    -----
    The CODATA 2018 constants are used throughout, as k_B = 8.617333262e-5 eV/K
    and 1.380649e-23 J/K, h = 6.62607015e-34 J s and m_e = 9.1093837015e-31 kg.
    """
    return rate  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_emission_rate(temperature: np.ndarray, activation_energy: float,
                                  sigma_inf: float, mass_ratio: float = 0.063) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes
    # it in isolation.
    import numpy as np

    kb_ev = 8.617333262e-5
    kb_j = 1.380649e-23
    h_planck = 6.62607015e-34
    m_e = 9.1093837015e-31

    for name, value in (("activation_energy", activation_energy),
                        ("sigma_inf", sigma_inf), ("mass_ratio", mass_ratio)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float, np.floating, np.integer))
                or not np.isfinite(value) or float(value) <= 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    temperature = np.asarray(temperature, dtype=float)
    if temperature.size == 0 or not np.all(np.isfinite(temperature)):
        raise ValueError("temperature must be non-empty and finite")
    if np.any(temperature <= 0.0):
        raise ValueError("temperature must be strictly positive")

    mass = float(mass_ratio) * m_e

    # Root-mean-square thermal velocity of the free carrier, m/s to cm/s.
    v_thermal = np.sqrt(3.0 * kb_j * temperature / mass) * 100.0

    # Effective density of states of the receiving band, converted to 1/cm^3.
    n_states = 2.0 * (2.0 * np.pi * mass * kb_j * temperature
                      / h_planck ** 2) ** 1.5 * 1e-6

    boltzmann = np.exp(-float(activation_energy) / (kb_ev * temperature))
    return float(sigma_inf) * v_thermal * n_states * boltzmann

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: dominant level over the benchmark temperature span ---
        {
            "setup": """import numpy as np
temperature = np.arange(300.0, 480.0 + 1e-9, 3.0)
""",
            "call": "np.log10(compute_emission_rate(temperature, 0.711, 1.8e-15)) + 1000.0",
            "gold_call": "np.log10(_oracle_compute_emission_rate(temperature, 0.711, 1.8e-15)) + 1000.0",
        },
        # --- Valid: shallower level with a larger cross section ---
        {
            "setup": """import numpy as np
temperature = np.array([351.0, 399.0, 450.0])
""",
            "call": "np.log10(compute_emission_rate(temperature, 0.658, 9.1e-15)) + 1000.0",
            "gold_call": "np.log10(_oracle_compute_emission_rate(temperature, 0.658, 9.1e-15)) + 1000.0",
        },
        # --- Boundary: a single temperature and a heavier effective mass ---
        {
            "setup": """import numpy as np
temperature = np.array([399.0])
""",
            "call": "np.log10(compute_emission_rate(temperature, 0.711, 1.8e-15, 0.28)) + 1000.0",
            "gold_call": "np.log10(_oracle_compute_emission_rate(temperature, 0.711, 1.8e-15, 0.28)) + 1000.0",
        },
        # --- Edge: very deep level, emission rate far below the measurable range ---
        {
            "setup": """import numpy as np
temperature = np.array([300.0, 480.0])
""",
            "call": "np.log10(compute_emission_rate(temperature, 1.35, 5.0e-16)) + 1000.0",
            "gold_call": "np.log10(_oracle_compute_emission_rate(temperature, 1.35, 5.0e-16)) + 1000.0",
        },
        # --- Invalid: non-positive activation energy ---
        {
            "setup": """import numpy as np
temperature = np.array([400.0])
def run_model():
    try:
        compute_emission_rate(temperature, 0.0, 1.8e-15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_emission_rate(temperature, 0.0, 1.8e-15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive temperature ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_emission_rate(np.array([0.0, 400.0]), 0.711, 1.8e-15)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_emission_rate(np.array([0.0, 400.0]), 0.711, 1.8e-15)
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
