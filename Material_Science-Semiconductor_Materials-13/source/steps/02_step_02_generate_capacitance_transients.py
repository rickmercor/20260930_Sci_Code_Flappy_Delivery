"""
Return the noisy capacitance transient recorded at each sample temperature for a junction containing several deep levels, given the levels' emission parameters and capacitance deflections.

After the filling pulse is removed the depletion region widens and every occupied deep level begins to release its carriers independently of the others. Each level therefore contributes an exponential relaxation of the junction capacitance whose decay constant is that level's emission rate and whose amplitude is the capacitance deflection produced by the charge it had captured. Because the levels do not interact, the measured transient is the plain linear superposition of those exponentials on top of the quiescent capacitance at the reverse bias, and the sign of every amplitude is negative when majority carriers are captured, since compensating charge inside the space charge region widens it.




The measurement is a repeated-average acquisition on a capacitance bridge, so the residual scatter is white on the sampling interval and Gaussian to a good approximation. Its magnitude is set by Johnson-Nyquist noise in the bridge, whose voltage spectral density scales as the square root of absolute temperature; a transient recorded at a higher temperature is therefore noisier by the square root of the temperature ratio even though nothing about the defects has changed. Reproducing that scaling matters because the levels of interest are separated in temperature, so a fixed noise amplitude would misrepresent the relative quality of the fast and slow parts of the data set.




The uniform sampling grid runs from one sampling interval to the total acquisition time. That grid fixes the two ends of the accessible emission-rate range: a level whose time constant exceeds the acquisition time has not measurably decayed by the last sample, and a level whose time constant is shorter than a few sampling intervals has already vanished by the first, so in either case the transient carries almost no information about its amplitude.

Returns
-------
np.ndarray of shape (n_temperatures, n_samples), float: capacitance in pF.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def generate_capacitance_transients(temperature: np.ndarray, activation_energy: np.ndarray,
                                    sigma_inf: np.ndarray, deflection: np.ndarray,
                                    base_capacitance: float, sampling_rate: float,
                                    n_samples: int, noise_ref: float,
                                    temperature_ref: float, seed: int,
                                    mass_ratio: float = 0.063) -> np.ndarray:
    """Return the noisy capacitance transients of a multi-level junction.

    Parameters
    ----------
    temperature : np.ndarray
        Sample temperatures in K, in the order they are acquired.
    activation_energy : np.ndarray
        Trap depth of each deep level in eV.
    sigma_inf : np.ndarray
        Infinite-temperature capture cross section of each level in cm^2.
    deflection : np.ndarray
        Capacitance deflection of each level in pF, negative for majority
        carrier capture.
    base_capacitance : float
        Quiescent capacitance at the reverse bias in pF.
    sampling_rate : float
        Sampling rate in Hz (sampling_rate > 0).
    n_samples : int
        Number of samples per transient (n_samples >= 1).
    noise_ref : float
        Standard deviation in pF of the additive noise at the reference
        temperature (noise_ref >= 0).
    temperature_ref : float
        Reference temperature in K at which noise_ref applies.
    seed : int
        Seed of the NumPy default random generator; the noise of each
        temperature is drawn in the order the temperatures are given.
    mass_ratio : float
        Carrier effective mass of the receiving band in units of the free
        electron mass.

    Returns
    -------
    transients : np.ndarray
        Array of shape (n_temperatures, n_samples) holding the capacitance in
        pF sampled at times sampling_rate**-1, 2/sampling_rate, ... .

    Raises
    ------
    ValueError
        If any argument is outside the stated domain.
    """
    return transients  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_generate_capacitance_transients(temperature: np.ndarray, activation_energy: np.ndarray,
                                            sigma_inf: np.ndarray, deflection: np.ndarray,
                                            base_capacitance: float, sampling_rate: float,
                                            n_samples: int, noise_ref: float,
                                            temperature_ref: float, seed: int,
                                            mass_ratio: float = 0.063) -> np.ndarray:
    import numpy as np

    kb_ev = 8.617333262e-5
    kb_j = 1.380649e-23
    h_planck = 6.62607015e-34
    m_e = 9.1093837015e-31

    temperature = np.asarray(temperature, dtype=float)
    activation_energy = np.atleast_1d(np.asarray(activation_energy, dtype=float))
    sigma_inf = np.atleast_1d(np.asarray(sigma_inf, dtype=float))
    deflection = np.atleast_1d(np.asarray(deflection, dtype=float))

    if not (isinstance(n_samples, (int, np.integer)) and not isinstance(n_samples, bool)
            and int(n_samples) >= 1):
        raise ValueError("n_samples must be an integer >= 1")
    if not np.isfinite(sampling_rate) or float(sampling_rate) <= 0.0:
        raise ValueError("sampling_rate must be a finite number > 0")
    if float(noise_ref) < 0.0:
        raise ValueError("noise_ref must be non-negative")
    if activation_energy.size != sigma_inf.size or activation_energy.size != deflection.size:
        raise ValueError("level parameter arrays must have matching lengths")
    if temperature.size == 0 or np.any(temperature <= 0.0):
        raise ValueError("temperature must be non-empty and strictly positive")

    mass = float(mass_ratio) * m_e
    n_samples = int(n_samples)
    times = np.arange(1, n_samples + 1) / float(sampling_rate)
    rng = np.random.default_rng(seed)

    transients = np.empty((temperature.size, n_samples), dtype=float)
    for i, temp in enumerate(temperature):
        v_thermal = np.sqrt(3.0 * kb_j * temp / mass) * 100.0
        n_states = 2.0 * (2.0 * np.pi * mass * kb_j * temp / h_planck ** 2) ** 1.5 * 1e-6
        signal = np.full(n_samples, float(base_capacitance))
        for k in range(activation_energy.size):
            rate = (sigma_inf[k] * v_thermal * n_states
                    * np.exp(-activation_energy[k] / (kb_ev * temp)))
            signal = signal + deflection[k] * np.exp(-rate * times)
        scale = float(noise_ref) * np.sqrt(temp / float(temperature_ref))
        transients[i] = signal + rng.standard_normal(n_samples) * scale

    return transients

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: two overlapping levels, short acquisition (normal scenario) ---
        {
            "setup": """import numpy as np
temperature = np.arange(360.0, 450.0 + 1e-9, 30.0)
ea = np.array([0.711, 0.658])
sg = np.array([1.8e-15, 9.1e-15])
dc = np.array([-0.75, -0.05])
""",
            "call": "generate_capacitance_transients(temperature, ea, sg, dc, 204.5, 1.0e5, 2000, 0.0032, 350.0, 20260722) + 1000.0",
            "gold_call": "_oracle_generate_capacitance_transients(temperature, ea, sg, dc, 204.5, 1.0e5, 2000, 0.0032, 350.0, 20260722) + 1000.0",
        },
        # --- Valid: noise-free single level reproduces the analytic exponential ---
        {
            "setup": """import numpy as np
temperature = np.array([399.0])
ea = np.array([0.711])
sg = np.array([1.8e-15])
dc = np.array([-0.75])
""",
            "call": "generate_capacitance_transients(temperature, ea, sg, dc, 204.5, 1.0e5, 500, 0.0, 350.0, 1) + 1000.0",
            "gold_call": "_oracle_generate_capacitance_transients(temperature, ea, sg, dc, 204.5, 1.0e5, 500, 0.0, 350.0, 1) + 1000.0",
        },
        # --- Boundary: a single sample per transient ---
        {
            "setup": """import numpy as np
temperature = np.array([372.0, 450.0])
ea = np.array([0.711, 0.658])
sg = np.array([1.8e-15, 9.1e-15])
dc = np.array([-0.75, -0.05])
""",
            "call": "generate_capacitance_transients(temperature, ea, sg, dc, 204.5, 1.0e5, 1, 0.0032, 350.0, 7) + 1000.0",
            "gold_call": "_oracle_generate_capacitance_transients(temperature, ea, sg, dc, 204.5, 1.0e5, 1, 0.0032, 350.0, 7) + 1000.0",
        },
        # --- Edge: three levels including a minority-carrier (positive) deflection ---
        {
            "setup": """import numpy as np
temperature = np.array([400.0, 420.0])
ea = np.array([0.711, 0.658, 0.540])
sg = np.array([1.8e-15, 9.1e-15, 3.0e-16])
dc = np.array([-0.75, -0.05, 0.02])
""",
            "call": "generate_capacitance_transients(temperature, ea, sg, dc, 180.0, 5.0e4, 1000, 0.001, 350.0, 99) + 1000.0",
            "gold_call": "_oracle_generate_capacitance_transients(temperature, ea, sg, dc, 180.0, 5.0e4, 1000, 0.001, 350.0, 99) + 1000.0",
        },
        # --- Invalid: mismatched level parameter arrays ---
        {
            "setup": """import numpy as np
temperature = np.array([400.0])
def run_model():
    try:
        generate_capacitance_transients(temperature, np.array([0.711, 0.658]), np.array([1.8e-15]), np.array([-0.75]), 204.5, 1.0e5, 100, 0.0, 350.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_generate_capacitance_transients(temperature, np.array([0.711, 0.658]), np.array([1.8e-15]), np.array([-0.75]), 204.5, 1.0e5, 100, 0.0, 350.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: zero samples requested ---
        {
            "setup": """import numpy as np
temperature = np.array([400.0])
ea = np.array([0.711])
sg = np.array([1.8e-15])
dc = np.array([-0.75])
def run_model():
    try:
        generate_capacitance_transients(temperature, ea, sg, dc, 204.5, 1.0e5, 0, 0.0, 350.0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_generate_capacitance_transients(temperature, ea, sg, dc, 204.5, 1.0e5, 0, 0.0, 350.0, 1)
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
