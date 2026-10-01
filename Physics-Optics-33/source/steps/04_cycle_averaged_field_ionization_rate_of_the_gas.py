"""
Compute the cycle-averaged field-ionization rate of the gas at each supplied instantaneous intensity.

The medium is a single-species gas whose ionization from the neutral ground state is treated through one channel, with magnetic quantum number zero and residual charge one, and whose effective principal quantum number is the residual charge divided by the square root of twice the ionization potential. The power-law prefactor of the rate carries unit amplitude in atomic units and an exponent of twice the effective principal quantum number minus one. Atomic units are used internally: the field amplitude follows from the cycle-averaged intensity through the atomic unit of intensity, 3.5094452e16 W/cm^2, and the atomic unit of time is 2.4188843265e-17 s.



Intensities are supplied in watts per square centimetre and may have any shape; the returned rates have the same shape and are expressed per second. Wherever the supplied intensity is not positive the returned rate is exactly zero, which keeps the fringe nodes well defined against the rounding of the interference term. The ionization potential is supplied in electronvolts and the driving wavelength, which sets the carrier frequency entering the rate, in metres.

Returns
-------
Return rate_per_s, a NumPy array of dtype float64 with the same shape as the supplied intensity, holding the ionization rate in inverse seconds. Entries corresponding to a non-positive supplied intensity are exactly zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_ionization_rate(intensity_wcm2: "np.ndarray", ionization_potential_ev: float, pump_wavelength_m: float) -> "np.ndarray":
    '''Return the field-ionization rate for each supplied intensity.

    Parameters
    ----------
    intensity_wcm2 : np.ndarray
        Instantaneous cycle-averaged intensity in watts per square centimetre. Any
        shape is accepted. All entries must be finite.
    ionization_potential_ev : float
        Ionization potential of the gas, in electronvolts. Must be positive.
    pump_wavelength_m : float
        Central wavelength of the driving field, in metres. Must be positive.

    Returns
    -------
    rate_per_s : np.ndarray
        Ionization rate in inverse seconds, with the same shape as the supplied
        intensity.

    Raises
    ------
    ValueError
        If any supplied intensity is not finite, or if the ionization potential or
        the wavelength is not positive.
    '''
    return rate_per_s

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_ionization_rate(intensity_wcm2: "np.ndarray", ionization_potential_ev: float, pump_wavelength_m: float) -> "np.ndarray":
    intensity = np.asarray(intensity_wcm2, dtype=float)
    if not np.all(np.isfinite(intensity)):
        raise ValueError("intensity_wcm2 entries must be finite")
    if not np.isfinite(ionization_potential_ev) or ionization_potential_ev <= 0.0:
        raise ValueError("ionization_potential_ev must be a positive finite number")
    if not np.isfinite(pump_wavelength_m) or pump_wavelength_m <= 0.0:
        raise ValueError("pump_wavelength_m must be a positive finite number")

    speed_of_light_ms = 2.99792458e8
    atomic_intensity_wcm2 = 3.5094452e16
    atomic_time_s = 2.4188843265e-17
    hartree_ev = 27.211386245988
    residual_charge = 1.0

    ip_au = float(ionization_potential_ev) / hartree_ev
    kappa = np.sqrt(2.0 * ip_au)
    n_star = residual_charge / kappa
    omega_au = (2.0 * np.pi * speed_of_light_ms / float(pump_wavelength_m)) * atomic_time_s

    positive = intensity > 0.0
    safe = np.where(positive, intensity, 1.0)
    field_au = np.sqrt(safe / atomic_intensity_wcm2)

    gamma = omega_au * kappa / field_au
    keldysh_factor = (3.0 / (2.0 * gamma)) * (
        (1.0 + 1.0 / (2.0 * gamma ** 2)) * np.arcsinh(gamma)
        - np.sqrt(1.0 + gamma ** 2) / (2.0 * gamma)
    )

    log_rate_au = (2.0 * n_star - 1.0) * np.log(2.0 * kappa ** 3 / field_au) - (
        2.0 * kappa ** 3 / (3.0 * field_au)
    ) * keldysh_factor

    with np.errstate(over="ignore", under="ignore"):
        rate_au = np.where(positive, np.exp(log_rate_au), 0.0)
    return rate_au / atomic_time_s

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
I_c1 = np.logspace(12.0, 16.0, 25)
IP_c1 = 15.7596
LAM_c1 = 800.0e-9

def log_rates_c1(fn):
    rates = np.asarray(fn(I_c1.copy(), IP_c1, LAM_c1), dtype=float)
    return np.log(np.where(rates > 0.0, rates, np.finfo(float).tiny))
""",
            "call": 'log_rates_c1(compute_ionization_rate)',
            "gold_call": 'log_rates_c1(_oracle_compute_ionization_rate)',
        },
        {
            "setup": """import numpy as np
I_c2 = np.linspace(1.0e14, 6.0e14, 11)
IP_c2 = 15.7596
LAM_c2 = 800.0e-9
""",
            "call": 'compute_ionization_rate(I_c2.copy(), IP_c2, LAM_c2)',
            "gold_call": '_oracle_compute_ionization_rate(I_c2.copy(), IP_c2, LAM_c2)',
        },
        {
            "setup": """import numpy as np
I_c3 = np.array([0.0, 0.0, 0.0])
IP_c3 = 15.7596
LAM_c3 = 800.0e-9
""",
            "call": 'compute_ionization_rate(I_c3.copy(), IP_c3, LAM_c3)',
            "gold_call": '_oracle_compute_ionization_rate(I_c3.copy(), IP_c3, LAM_c3)',
        },
        {
            "setup": """import numpy as np
I_c4 = np.array([1.0e8, 1.0e10, 1.0e12])
IP_c4 = 15.7596
LAM_c4 = 800.0e-9

def log_rates_c4(fn):
    rates = np.asarray(fn(I_c4.copy(), IP_c4, LAM_c4), dtype=float)
    return np.log(np.where(rates > 0.0, rates, np.finfo(float).tiny))
""",
            "call": 'log_rates_c4(compute_ionization_rate)',
            "gold_call": 'log_rates_c4(_oracle_compute_ionization_rate)',
        },
        {
            "setup": """import numpy as np
I_c5 = np.logspace(12.0, 15.0, 13)
IP_c5 = 12.0697
LAM_c5 = 800.0e-9
""",
            "call": 'compute_ionization_rate(I_c5.copy(), IP_c5, LAM_c5)',
            "gold_call": '_oracle_compute_ionization_rate(I_c5.copy(), IP_c5, LAM_c5)',
        },
        {
            "setup": """import numpy as np
I_c6 = np.logspace(13.0, 16.0, 13)
IP_c6 = 24.5874
LAM_c6 = 800.0e-9
""",
            "call": 'compute_ionization_rate(I_c6.copy(), IP_c6, LAM_c6)',
            "gold_call": '_oracle_compute_ionization_rate(I_c6.copy(), IP_c6, LAM_c6)',
        },
        {
            "setup": """import numpy as np
I_c7 = np.logspace(12.0, 15.0, 13).reshape(13, 1) * np.ones((1, 3))
IP_c7 = 15.7596
LAM_c7 = 400.0e-9
""",
            "call": 'compute_ionization_rate(I_c7.copy(), IP_c7, LAM_c7)',
            "gold_call": '_oracle_compute_ionization_rate(I_c7.copy(), IP_c7, LAM_c7)',
        },
        {
            "setup": """import numpy as np
I_c8 = np.array([1.0e14, np.nan])
IP_c8 = 15.7596
LAM_c8 = 800.0e-9

def run_c8(fn):
    try:
        fn(I_c8.copy(), IP_c8, LAM_c8)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c8(compute_ionization_rate)',
            "gold_call": 'run_c8(_oracle_compute_ionization_rate)',
        },
        {
            "setup": """import numpy as np
I_c9 = np.array([1.0e14])
IP_c9 = 0.0
LAM_c9 = 800.0e-9

def run_c9(fn):
    try:
        fn(I_c9.copy(), IP_c9, LAM_c9)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c9(compute_ionization_rate)',
            "gold_call": 'run_c9(_oracle_compute_ionization_rate)',
        },
        {
            "setup": """import numpy as np
I_c10 = np.array([1.0e14])
IP_c10 = 15.7596
LAM_c10 = -800.0e-9

def run_c10(fn):
    try:
        fn(I_c10.copy(), IP_c10, LAM_c10)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": 'run_c10(compute_ionization_rate)',
            "gold_call": 'run_c10(_oracle_compute_ionization_rate)',
        },
    ]
