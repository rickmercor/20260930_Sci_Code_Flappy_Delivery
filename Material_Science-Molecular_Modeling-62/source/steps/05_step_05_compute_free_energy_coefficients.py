"""
Reduce the available-energy increment of the nucleus to the three parameters that define its dependence on the nucleus radius.

The available-energy increment of a nucleus is the vapour density multiplied by the specific Gibbs free-energy increment, integrated over the nucleus volume. That increment has two parts. The first prices the phase change against the local superheat and is the latent heat divided by the saturation temperature, multiplied by the saturation temperature minus the local temperature; integrated, it splits into a plain volume term carrying the latent heat and a temperature-weighted volume term carrying the latent heat divided by the saturation temperature. The second prices the vapour against the surrounding liquid pressure and is the specific gas constant multiplied by the local temperature and by the logarithm of the pressure ratio; integrated, it carries the temperature-weighted volume as well.




Both volume terms scale as the squared radius, so the whole increment is the squared radius multiplied by a bracket that depends on the radius only through the pressure ratio. Mechanical equilibrium across the curved interface fixes the vapour overpressure by the Young-Laplace relation as twice the surface tension divided by the curvature radius of the nucleus, so the pressure ratio is one plus a capillary length divided by the radius, that capillary length being twice the surface tension divided by the liquid pressure. The state is therefore fully described by three numbers: a balance coefficient, which is the difference between the latent-heat cost of vaporising the mass and the superheat gain of doing so in hot liquid; a Laplace coefficient, which is strictly positive and multiplies the logarithm; and the capillary length itself. A barrier exists only when the balance coefficient is negative, that is when the superheat gain outruns the vaporisation cost, which for a fixed shape is a condition on how far the wall-adjacent liquid sits above saturation. If it is not met, the increment grows without bound and the nucleus can never become supercritical.




Keeping the logarithm rather than expanding it matters here. Expanding to first order replaces the bracket by a term inversely proportional to the radius and collapses the increment into a quadratic, which is convenient but is only valid while the overpressure is small compared with the liquid pressure. At the nucleus sizes this model produces the overpressure is several times the liquid pressure, so the expansion overstates the pressure term by a large factor and the convenience is bought at the cost of the answer.

Returns
-------
np.ndarray of shape (3,), float: the balance coefficient in J/m^2, the Laplace coefficient in J/m^2 and the capillary length in metres.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_free_energy_coefficients(volume_factor: float, moment_factor: float,
                                     t_liquid: float, vapour_state: np.ndarray,
                                     t_sat: float = 373.15,
                                     p_liquid: float = 101325.0,
                                     gamma_lv: float = 67.70e-3,
                                     depth: float = 31.30e-10) -> np.ndarray:
    """Reduce the available-energy increment to its three defining parameters.

    The increment at nucleus radius r is the squared radius multiplied by the
    balance coefficient plus the Laplace coefficient multiplied by the natural
    logarithm of one plus the capillary length divided by r.

    Parameters
    ----------
    volume_factor : float
        Dimensionless volume factor of the segment (> 0).
    moment_factor : float
        Dimensionless temperature-weighted volume factor of the segment (> 0).
    t_liquid : float
        Wall-adjacent liquid temperature in kelvin (> 0).
    vapour_state : np.ndarray
        Array of shape (3,) holding the specific gas constant in J/(kg K), the
        saturated vapour density in kg/m^3 and the latent heat in J/kg.
    t_sat : float
        Saturation temperature of the liquid in kelvin (> 0).
    p_liquid : float
        Liquid pressure in pascal (> 0).
    gamma_lv : float
        Liquid-vapour surface tension in N/m (> 0).
    depth : float
        Depth of the cylindrical nucleus in metres (> 0).

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (3,) holding the balance coefficient in J/m^2, the
        Laplace coefficient in J/m^2 and the capillary length in metres.
    """
    return coefficients  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_free_energy_coefficients(volume_factor: float, moment_factor: float,
                                             t_liquid: float, vapour_state: np.ndarray,
                                             t_sat: float = 373.15,
                                             p_liquid: float = 101325.0,
                                             gamma_lv: float = 67.70e-3,
                                             depth: float = 31.30e-10) -> np.ndarray:
    for name, value in (("volume_factor", volume_factor), ("moment_factor", moment_factor),
                        ("t_liquid", t_liquid), ("t_sat", t_sat), ("p_liquid", p_liquid),
                        ("gamma_lv", gamma_lv), ("depth", depth)):
        if not (isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    state = np.asarray(vapour_state, dtype=float).ravel()
    if state.size != 3 or not np.all(np.isfinite(state)) or np.any(state <= 0.0):
        raise ValueError("vapour_state must hold three finite positive entries")
    r_specific, rho_vapour, h_fg = state

    prefactor = float(depth) * rho_vapour

    # Latent-heat cost of the vaporised mass, less the superheat gained by
    # forming it inside the wall-normal temperature profile.
    balance = prefactor * h_fg * (float(volume_factor)
                                  - float(t_liquid) * float(moment_factor) / float(t_sat))

    # Multiplies the logarithm of the pressure ratio, which is retained exactly.
    laplace = prefactor * float(t_liquid) * float(moment_factor) * r_specific

    # Young-Laplace overpressure divided by the liquid pressure gives this
    # length divided by the nucleus radius.
    capillary = 2.0 * float(gamma_lv) / float(p_liquid)

    return np.array([balance, laplace, capillary], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np 


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: governing state of the testbed (normal scenario) ---
        {
            "setup": """import numpy as np
theta = np.deg2rad(78.7)
volume_factor = np.pi - theta + 0.5 * np.sin(2.0 * theta)
moment_factor = (3.0 * volume_factor - 2.0 * np.sin(theta) ** 3) / (3.0 * (1.0 + np.cos(theta)))
t_liquid = 731.5876975182241
vapour_state = np.array([461.5231157260608, 0.5883553522770233, 2227187.9287662064])
""",
            "call": "compute_free_energy_coefficients(volume_factor, moment_factor, t_liquid, vapour_state)",
            "gold_call": "_oracle_compute_free_energy_coefficients(volume_factor, moment_factor, t_liquid, vapour_state)",
        },
        # --- Valid: hydrophobic state at the strongest driving force ---
        {
            "setup": """import numpy as np
theta = np.deg2rad(130.8)
volume_factor = np.pi - theta + 0.5 * np.sin(2.0 * theta)
moment_factor = (3.0 * volume_factor - 2.0 * np.sin(theta) ** 3) / (3.0 * (1.0 + np.cos(theta)))
t_liquid = 675.6701664977229
vapour_state = np.array([461.5231157260608, 0.5883553522770233, 2227187.9287662064])
""",
            "call": "compute_free_energy_coefficients(volume_factor, moment_factor, t_liquid, vapour_state)",
            "gold_call": "_oracle_compute_free_energy_coefficients(volume_factor, moment_factor, t_liquid, vapour_state)",
        },
        # --- Boundary: liquid barely above saturation, so the balance coefficient is positive ---
        {
            "setup": """import numpy as np
theta = np.deg2rad(62.5)
volume_factor = np.pi - theta + 0.5 * np.sin(2.0 * theta)
moment_factor = (3.0 * volume_factor - 2.0 * np.sin(theta) ** 3) / (3.0 * (1.0 + np.cos(theta)))
t_liquid = 400.0
vapour_state = np.array([461.5231157260608, 0.5883553522770233, 2227187.9287662064])
""",
            "call": "compute_free_energy_coefficients(volume_factor, moment_factor, t_liquid, vapour_state)",
            "gold_call": "_oracle_compute_free_energy_coefficients(volume_factor, moment_factor, t_liquid, vapour_state)",
        },
        # --- Edge: raised system pressure and a thicker liquid slab ---
        {
            "setup": """import numpy as np
theta = np.deg2rad(90.0)
volume_factor = np.pi
moment_factor = (3.0 * volume_factor - 2.0) / 3.0
t_liquid = 700.0
vapour_state = np.array([461.5231157260608, 1.1554824570743413, 2136459.0])
""",
            "call": "compute_free_energy_coefficients(volume_factor, moment_factor, t_liquid, vapour_state, 393.35, 202650.0, 0.0589, 5.0e-9)",
            "gold_call": "_oracle_compute_free_energy_coefficients(volume_factor, moment_factor, t_liquid, vapour_state, 393.35, 202650.0, 0.0589, 5.0e-9)",
        },
        # --- Invalid: vapour state of the wrong length ---
        {
            "setup": """import numpy as np
bad_state = np.array([461.5231157260608, 0.5883553522770233])
def run_model():
    try:
        compute_free_energy_coefficients(1.96, 1.11, 731.59, bad_state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_free_energy_coefficients(1.96, 1.11, 731.59, bad_state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive liquid temperature ---
        {
            "setup": """import numpy as np
vapour_state = np.array([461.5231157260608, 0.5883553522770233, 2227187.9287662064])
def run_model():
    try:
        compute_free_energy_coefficients(1.96, 1.11, 0.0, vapour_state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_free_energy_coefficients(1.96, 1.11, 0.0, vapour_state)
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
