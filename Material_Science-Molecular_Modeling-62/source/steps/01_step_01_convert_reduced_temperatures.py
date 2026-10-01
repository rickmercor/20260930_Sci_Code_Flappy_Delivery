"""
Convert the substrate temperature and the interfacial temperature jump from Lennard-Jones reduced units to kelvin and return the wall-adjacent liquid temperature that drives the nucleation model.

Molecular dynamics of water on a metal substrate is reported in reduced Lennard-Jones units built from the oxygen-oxygen pair of the water model, because those units are what make the simulation independent of the particular parameterization used. A temperature is reduced as the ratio of the thermal energy to the well depth of that pair, so the reduced value multiplied by the well depth expressed as an energy and divided by the Boltzmann constant recovers Kelvin. For the TIP4P/2005 water model, the oxygen-oxygen well depth is 0.008031 eV, which corresponds to 93.196 K, and the substrate temperatures used in reflood-relevant simulations of zirconium cladding, roughly 720 K to 1080 K, map to reduced values between about 7.8 and 11.6. The conversion is not cosmetic here, because every term of the nucleation model that follows is dimensional and a reduced temperature substituted directly into it would be wrong by two orders of magnitude.

The second quantity converted the same way is the interfacial temperature jump. A solid-liquid contact carries a finite thermal resistance, so the liquid touching the metal is colder than the metal itself, by an amount that grows with the heat flux crossing the interface and shrinks as the solid-liquid interaction strengthens. That jump is what makes wettability enter a nucleation calculation through the temperature field rather than only through the contact angle, and at the fluxes reached under a hot cladding surface it is a large fraction of the wall superheat rather than a small correction. Subtracting it from the substrate temperature gives the temperature of the liquid at the foot of the nucleus, which is the anchor of the linear temperature profile assumed across the nucleus and therefore the temperature that sets both the superheat available for the phase change and the thermal energy against which the resulting barrier is measured.

Returns
-------
np.ndarray of shape (3,), float: the substrate temperature, the interfacial temperature jump and the wall-adjacent liquid temperature in kelvin.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def convert_reduced_temperatures(t_substrate_red: float, t_jump_red: float,
                                 epsilon_ev: float = 0.008031) -> np.ndarray:
    """Convert reduced substrate and jump temperatures to kelvin.

    Parameters
    ----------
    t_substrate_red : float
        Substrate temperature in reduced Lennard-Jones units (> 0).
    t_jump_red : float
        Interfacial temperature jump in reduced Lennard-Jones units (>= 0).
    epsilon_ev : float
        Lennard-Jones well depth of the reference pair in electronvolts (> 0).

    Returns
    -------
    temperatures : np.ndarray
        Array of shape (3,) holding the substrate temperature, the interfacial
        temperature jump and the wall-adjacent liquid temperature, all in
        kelvin.
    """
    return temperatures  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_convert_reduced_temperatures(t_substrate_red: float, t_jump_red: float,
                                         epsilon_ev: float = 0.008031) -> np.ndarray:
    boltzmann = 1.380649e-23
    elementary_charge = 1.602176634e-19

    if not (isinstance(t_substrate_red, (int, float)) and not isinstance(t_substrate_red, bool)
            and np.isfinite(t_substrate_red) and float(t_substrate_red) > 0.0):
        raise ValueError("t_substrate_red must be a finite number > 0")
    if not (isinstance(t_jump_red, (int, float)) and not isinstance(t_jump_red, bool)
            and np.isfinite(t_jump_red) and float(t_jump_red) >= 0.0):
        raise ValueError("t_jump_red must be a finite number >= 0")
    if not (isinstance(epsilon_ev, (int, float)) and not isinstance(epsilon_ev, bool)
            and np.isfinite(epsilon_ev) and float(epsilon_ev) > 0.0):
        raise ValueError("epsilon_ev must be a finite number > 0")

    # One reduced temperature unit is the well depth expressed as an energy
    # divided by the Boltzmann constant.
    scale = float(epsilon_ev) * elementary_charge / boltzmann

    t_substrate = float(t_substrate_red) * scale
    t_jump = float(t_jump_red) * scale
    t_liquid = t_substrate - t_jump

    if t_liquid <= 0.0:
        raise ValueError("the interfacial jump cannot exceed the substrate temperature")

    return np.array([t_substrate, t_jump, t_liquid], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark state of the testbed (normal scenario) ---
        {
            "setup": """import numpy as np
t_substrate_red = 9.95
t_jump_red = 1.83
""",
            "call": "convert_reduced_temperatures(t_substrate_red, t_jump_red)",
            "gold_call": "_oracle_convert_reduced_temperatures(t_substrate_red, t_jump_red)",
        },
        # --- Valid: coldest substrate of the reflood-relevant sweep ---
        {
            "setup": """import numpy as np
t_substrate_red = 7.79
t_jump_red = 1.83
""",
            "call": "convert_reduced_temperatures(t_substrate_red, t_jump_red)",
            "gold_call": "_oracle_convert_reduced_temperatures(t_substrate_red, t_jump_red)",
        },
        # --- Boundary: no interfacial resistance, so the liquid sits at the wall temperature ---
        {
            "setup": """import numpy as np
t_substrate_red = 11.57
t_jump_red = 0.0
""",
            "call": "convert_reduced_temperatures(t_substrate_red, t_jump_red)",
            "gold_call": "_oracle_convert_reduced_temperatures(t_substrate_red, t_jump_red)",
        },
        # --- Edge: a different well depth rescales both temperatures ---
        {
            "setup": """import numpy as np
t_substrate_red = 9.95
t_jump_red = 1.83
epsilon_ev = 0.0104
""",
            "call": "convert_reduced_temperatures(t_substrate_red, t_jump_red, epsilon_ev)",
            "gold_call": "_oracle_convert_reduced_temperatures(t_substrate_red, t_jump_red, epsilon_ev)",
        },
        # --- Invalid: the jump swallows the whole substrate temperature ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        convert_reduced_temperatures(1.83, 9.95)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_convert_reduced_temperatures(1.83, 9.95)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive well depth ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        convert_reduced_temperatures(9.95, 1.83, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_convert_reduced_temperatures(9.95, 1.83, 0.0)
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
