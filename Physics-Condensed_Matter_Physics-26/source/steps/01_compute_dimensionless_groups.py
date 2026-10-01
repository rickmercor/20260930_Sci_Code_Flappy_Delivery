"""
Derive the four dimensionless groups and the dimensionless elapsed time that control the transverse heat problem and the thermomechanical coupling of a laser-heated photoresponsive beam.

The beam is nondimensionalized using the transverse thermal diffusion time t_diff = rho0*c_theta*h^2/k as the time unit (h the beam's square cross-section side, k its thermal conductivity, c_theta its specific heat capacity, rho0 its density), so a physical elapsed time t_phys corresponds to a dimensionless elapsed time t = t_phys / t_diff. The Biot number Bi = H*h/k measures convective surface heat loss relative to conduction (H the heat transfer coefficient), the dimensionless optical attenuation coefficient beta = beta_dim*h measures how many attenuation lengths fit across the beam's thickness (beta_dim the dimensional attenuation coefficient), and the dimensionless laser radius w = w_dim/h compares the laser spot size to the beam's thickness (w_dim the laser spot radius).

A fourth dimensionless group, Gamma1, sets the overall strength of the coupling between the transverse thermal problem and the resulting bending: Gamma1 = gamma1 * eta_th * beta_dim * h * L * [P/(pi*w_dim^2)] / k, where gamma1 is the beam's longitudinal coefficient of thermal expansion, eta_th is the fraction of absorbed optical power converted to heat, L is the beam's length, and P/(pi*w_dim^2) is the laser's incident intensity prefactor (power divided by the area factor of its Gaussian spot). This specific combination is the one that the underlying asymptotic beam model derives; it is not a generic thermal-stress coupling parameter and cannot be guessed from dimensional analysis alone, since several different combinations of these nine physical quantities would also be dimensionless.

Returns
-------
np.ndarray of shape (5,), float: [Bi, beta, w, Gamma1, t], in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_dimensionless_groups(
    length: float,
    thickness: float,
    thermal_conductivity: float,
    specific_heat: float,
    density: float,
    thermal_expansion: float,
    attenuation_coefficient: float,
    heat_conversion_fraction: float,
    heat_transfer_coefficient: float,
    laser_radius: float,
    incident_intensity_prefactor: float,
    elapsed_time: float,
) -> np.ndarray:
    """Dimensionless groups and elapsed time for the beam heating problem.

    Parameters
    ----------
    length : float
        Beam length L (m), > 0.
    thickness : float
        Beam (square cross-section) side length h (m), > 0.
    thermal_conductivity : float
        Isotropic thermal conductivity k (W/m/K), > 0.
    specific_heat : float
        Specific heat capacity c_theta (J/kg/K), > 0.
    density : float
        Initial density rho0 (kg/m^3), > 0.
    thermal_expansion : float
        Longitudinal coefficient of thermal expansion gamma1 (1/K), finite
        and nonzero.
    attenuation_coefficient : float
        Dimensional optical attenuation coefficient beta_dim (1/m), > 0.
    heat_conversion_fraction : float
        Fraction eta_th of absorbed optical power converted to heat,
        0 < eta_th <= 1.
    heat_transfer_coefficient : float
        Convective heat transfer coefficient H (W/m^2/K), > 0.
    laser_radius : float
        Laser beam radius w_dim (m), > 0.
    incident_intensity_prefactor : float
        The ratio P/(pi*w_dim^2), with units W/m^2, > 0.
    elapsed_time : float
        Physical elapsed time since the laser was switched on (s), >= 0.

    Returns
    -------
    groups : np.ndarray
        Array of shape (5,): [Bi, beta, w, Gamma1, t], the Biot number, the
        dimensionless attenuation coefficient, the dimensionless laser
        radius, the thermomechanical coupling constant, and the
        dimensionless elapsed time, in that order.

    Raises
    ------
    ValueError
        If length, thickness, thermal_conductivity, specific_heat, density,
        attenuation_coefficient, heat_transfer_coefficient, laser_radius or
        incident_intensity_prefactor is not a finite number > 0; if
        thermal_expansion is not finite and nonzero; if
        heat_conversion_fraction is not in (0, 1]; or if elapsed_time is
        not a finite number >= 0.
    """
    return groups  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_dimensionless_groups(
    length: float,
    thickness: float,
    thermal_conductivity: float,
    specific_heat: float,
    density: float,
    thermal_expansion: float,
    attenuation_coefficient: float,
    heat_conversion_fraction: float,
    heat_transfer_coefficient: float,
    laser_radius: float,
    incident_intensity_prefactor: float,
    elapsed_time: float,
) -> np.ndarray:
    import numpy as np

    for name, value in (("length", length), ("thickness", thickness),
                        ("thermal_conductivity", thermal_conductivity),
                        ("specific_heat", specific_heat), ("density", density),
                        ("attenuation_coefficient", attenuation_coefficient),
                        ("heat_transfer_coefficient", heat_transfer_coefficient),
                        ("laser_radius", laser_radius),
                        ("incident_intensity_prefactor", incident_intensity_prefactor)):
        if not (isinstance(value, (int, float, np.floating)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not (isinstance(thermal_expansion, (int, float, np.floating)) and np.isfinite(thermal_expansion)
            and float(thermal_expansion) != 0.0):
        raise ValueError("thermal_expansion must be finite and nonzero")
    if not (isinstance(heat_conversion_fraction, (int, float, np.floating))
            and np.isfinite(heat_conversion_fraction) and 0.0 < float(heat_conversion_fraction) <= 1.0):
        raise ValueError("heat_conversion_fraction must be in (0, 1]")
    if not (isinstance(elapsed_time, (int, float, np.floating)) and np.isfinite(elapsed_time)
            and float(elapsed_time) >= 0.0):
        raise ValueError("elapsed_time must be a finite number >= 0")

    L = float(length)
    h = float(thickness)
    k = float(thermal_conductivity)
    c_theta = float(specific_heat)
    rho0 = float(density)
    gamma1 = float(thermal_expansion)
    beta_dim = float(attenuation_coefficient)
    eta_th = float(heat_conversion_fraction)
    H = float(heat_transfer_coefficient)
    w_dim = float(laser_radius)
    i0_pref = float(incident_intensity_prefactor)
    t_phys = float(elapsed_time)

    bi = H * h / k
    beta = beta_dim * h
    w = w_dim / h
    gamma1_dimless = gamma1 * eta_th * beta_dim * h * L * i0_pref / k
    t_diff = rho0 * c_theta * h ** 2 / k
    t = t_phys / t_diff

    return np.array([bi, beta, w, gamma1_dimless, t], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark scenario (normal scenario) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
args = dict(length=0.025, thickness=0.0025, thermal_conductivity=0.55,
            specific_heat=4.0e3, density=1000.0, thermal_expansion=-0.015,
            attenuation_coefficient=220.0, heat_conversion_fraction=0.8,
            heat_transfer_coefficient=150.0, laser_radius=0.005,
            incident_intensity_prefactor=2.5e4, elapsed_time=3.0)
""",
            "call": "compute_dimensionless_groups(**args)",
            "gold_call": "_oracle_compute_dimensionless_groups(**args)",
        },
        # --- Valid: a materially different beam and laser configuration ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
args = dict(length=0.04, thickness=0.004, thermal_conductivity=0.3,
            specific_heat=3.5e3, density=1100.0, thermal_expansion=-0.008,
            attenuation_coefficient=60.0, heat_conversion_fraction=0.6,
            heat_transfer_coefficient=80.0, laser_radius=0.006,
            incident_intensity_prefactor=1.8e4, elapsed_time=6.0)
""",
            "call": "compute_dimensionless_groups(**args)",
            "gold_call": "_oracle_compute_dimensionless_groups(**args)",
        },
        # --- Boundary: elapsed_time = 0 and a positive thermal-expansion coefficient ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
args = dict(length=0.025, thickness=0.0025, thermal_conductivity=0.55,
            specific_heat=4.0e3, density=1000.0, thermal_expansion=0.02,
            attenuation_coefficient=90.0, heat_conversion_fraction=1.0,
            heat_transfer_coefficient=150.0, laser_radius=0.005,
            incident_intensity_prefactor=2.5e4, elapsed_time=0.0)
""",
            "call": "compute_dimensionless_groups(**args)",
            "gold_call": "_oracle_compute_dimensionless_groups(**args)",
        },
        # --- Invalid: non-positive thickness ---
        {
            "setup": """import numpy as np
args = dict(length=0.025, thickness=0.0, thermal_conductivity=0.55,
            specific_heat=4.0e3, density=1000.0, thermal_expansion=-0.015,
            attenuation_coefficient=220.0, heat_conversion_fraction=0.8,
            heat_transfer_coefficient=150.0, laser_radius=0.005,
            incident_intensity_prefactor=2.5e4, elapsed_time=3.0)
def run_model():
    try:
        compute_dimensionless_groups(**args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_dimensionless_groups(**args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: heat_conversion_fraction out of (0, 1] ---
        {
            "setup": """import numpy as np
args = dict(length=0.025, thickness=0.0025, thermal_conductivity=0.55,
            specific_heat=4.0e3, density=1000.0, thermal_expansion=-0.015,
            attenuation_coefficient=220.0, heat_conversion_fraction=1.5,
            heat_transfer_coefficient=150.0, laser_radius=0.005,
            incident_intensity_prefactor=2.5e4, elapsed_time=3.0)
def run_model():
    try:
        compute_dimensionless_groups(**args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_dimensionless_groups(**args)
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
