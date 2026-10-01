"""
Express the critical activation energy in units of the thermal energy of the wall-adjacent liquid.

An activation energy in joules says nothing on its own about whether the transition it guards will happen. What decides that is its size relative to the energy a thermal fluctuation can supply, which for a liquid at temperature T is of order the Boltzmann constant multiplied by T. Dividing the critical activation energy by that thermal energy gives the dimensionless barrier height, the quantity that appears in the exponent of every nucleation-rate expression from classical theory onwards. The convention adopted here is to refer the barrier to the wall-adjacent liquid temperature rather than to the substrate temperature, because it is the liquid at the foot of the nucleus that has to fluctuate, and the interfacial jump can put those two temperatures hundreds of kelvin apart.




Reading the resulting number is a matter of orders of magnitude rather than of digits. A dimensionless barrier of a few tens corresponds to nucleation rates that are observable on laboratory timescales, and one of a few hundred already places the event beyond any accessible observation time; a barrier of thousands or more says the model predicts no nucleation at all under the conditions supplied. When a molecular dynamics simulation of the same interface is nevertheless observed to nucleate within a few hundred picoseconds, a very large dimensionless barrier is not evidence that the simulation is wrong but evidence about the domain of the continuum model, which prices the nucleus with bulk surface tension, bulk latent heat and a continuum density even when the nucleus it predicts is far larger than the liquid film available to contain it. The ratio is therefore the natural place to compare the two descriptions, and the natural place to see them part company.

Returns
-------
float: the dimensionless ratio of the critical activation energy to the thermal energy of the wall-adjacent liquid, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_thermal_barrier_ratio(energy_critical: float, t_liquid: float) -> float:
    """Express the critical activation energy in units of the thermal energy.

    Parameters
    ----------
    energy_critical : float
        Critical activation energy in joules (> 0).
    t_liquid : float
        Wall-adjacent liquid temperature in kelvin (> 0).

    Returns
    -------
    barrier_ratio : float
        Critical activation energy divided by the product of the Boltzmann
        constant and the wall-adjacent liquid temperature.
    """
    return barrier_ratio  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_thermal_barrier_ratio(energy_critical: float, t_liquid: float) -> float:
    boltzmann = 1.380649e-23

    for name, value in (("energy_critical", energy_critical), ("t_liquid", t_liquid)):
        if not (isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    return float(float(energy_critical) / (boltzmann * float(t_liquid)))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark hydrophilic state (normal scenario) ---
        {
            "setup": """import numpy as np
energy_critical = 2.7267800527176885e-16
t_liquid = 756.7505864774497
""",
            "call": "compute_thermal_barrier_ratio(energy_critical, t_liquid)",
            "gold_call": "_oracle_compute_thermal_barrier_ratio(energy_critical, t_liquid)",
        },
        # --- Valid: benchmark hydrophobic state at the strongest driving force ---
        {
            "setup": """import numpy as np
energy_critical = 2.8377363250000000e-17
t_liquid = 756.7505864774497
""",
            "call": "compute_thermal_barrier_ratio(energy_critical, t_liquid)",
            "gold_call": "_oracle_compute_thermal_barrier_ratio(energy_critical, t_liquid)",
        },
        # --- Boundary: barrier of exactly one thermal unit at room temperature ---
        {
            "setup": """import numpy as np
t_liquid = 300.0
energy_critical = 1.380649e-23 * t_liquid
""",
            "call": "compute_thermal_barrier_ratio(energy_critical, t_liquid)",
            "gold_call": "_oracle_compute_thermal_barrier_ratio(energy_critical, t_liquid)",
        },
        # --- Edge: barrier small enough to make nucleation effectively barrierless ---
        {
            "setup": """import numpy as np
energy_critical = 1.0e-24
t_liquid = 1000.0
""",
            "call": "compute_thermal_barrier_ratio(energy_critical, t_liquid)",
            "gold_call": "_oracle_compute_thermal_barrier_ratio(energy_critical, t_liquid)",
        },
        # --- Invalid: non-positive activation energy ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_thermal_barrier_ratio(-1.0e-18, 756.75)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_thermal_barrier_ratio(-1.0e-18, 756.75)
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
def run_model():
    try:
        compute_thermal_barrier_ratio(1.0e-18, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_thermal_barrier_ratio(1.0e-18, 0.0)
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
