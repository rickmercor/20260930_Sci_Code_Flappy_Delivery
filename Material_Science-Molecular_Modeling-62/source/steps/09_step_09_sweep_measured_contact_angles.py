"""
Turn a table of measured quasi-steady bubble contact angles and their wall-adjacent liquid temperatures into the dimensionless nucleation barrier of each state.

The measurable that carries both the wettability of the substrate and the effect of the imposed flow into the nucleation model is the quasi-steady contact angle of the bubble at the three-phase line. Wettability sets its value at rest, since a strongly interacting surface holds the liquid against the wall and pushes the vapour into a compact shape, while the imposed driving force shears the growing bubble and lowers the angle further as it is raised. The model, however, is written in terms of the contact angle of the liquid, which is the supplement of the bubble angle, so every measured angle must be reflected about a straight angle before it enters the geometry. Skipping that reflection does not produce a small error; it exchanges the hydrophilic states for the hydrophobic ones and reverses the trend of the result with driving force.




The thermal state is not shared across the table. Each measured state carries its own interfacial temperature jump, because the jump is set by the solid-liquid thermal resistance and therefore by both the wettability of the surface and the flow being driven over it, so each state has its own wall-adjacent liquid temperature. The barrier of a state consequently depends on two things that vary independently: the pair of shape factors, which fall as the liquid-side angle rises, and the superheat, which enters through the balance coefficient and grows sharply in influence as a state approaches the threshold where a barrier ceases to exist. Because the two dependences act in opposite directions across a driving-force sweep, no ordering can be read off either variable alone and every state has to be evaluated in full before the hardest one can be identified.




Sweeping a whole measurement table through the model is therefore a stronger statement than comparing two states in isolation. It exposes whether the wettability effect and the driving-force effect are separable, and it is the only way to see when a state on one surface overtakes a state on the other. Each state requires its own numerical solution of the stationarity condition, since the critical radius depends on the balance and Laplace coefficients in a way that admits no closed form.

Returns
-------
np.ndarray with the same shape as theta_d_deg, float: the dimensionless nucleation barrier of each measured state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sweep_measured_contact_angles(theta_d_deg: np.ndarray, t_liquid: np.ndarray,
                                  vapour_state: np.ndarray,
                                  t_sat: float = 373.15,
                                  p_liquid: float = 101325.0,
                                  gamma_lv: float = 67.70e-3,
                                  depth: float = 31.30e-10) -> np.ndarray:
    """Compute the dimensionless nucleation barrier of every measured state.

    Parameters
    ----------
    theta_d_deg : np.ndarray
        One-dimensional array of measured quasi-steady bubble contact angles in
        degrees, each strictly between 0 and 180.
    t_liquid : np.ndarray
        One-dimensional array of wall-adjacent liquid temperatures in kelvin,
        one per measured state and of the same shape as theta_d_deg.
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
    barrier_ratios : np.ndarray
        Array with the same shape as theta_d_deg holding the critical
        activation energy of each state in units of the thermal energy of its
        own wall-adjacent liquid.
    """
    return barrier_ratios  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import glob
import importlib.util
import os
import sys

import numpy as np


def _oracle_sweep_measured_contact_angles(theta_d_deg: np.ndarray, t_liquid: np.ndarray,
                                          vapour_state: np.ndarray,
                                          t_sat: float = 373.15,
                                          p_liquid: float = 101325.0,
                                          gamma_lv: float = 67.70e-3,
                                          depth: float = 31.30e-10) -> np.ndarray:
    # -- Resolve the oracle functions of the earlier sub-problems. Preference
    #    order: (1) already present in the executing namespace (shared-namespace
    #    harness), (2) loaded from a sibling sub-problem file matched by name
    #    pattern (standalone execution; file prefixes may vary), (3) the public
    #    function of the same step if the harness injected it.
    def _resolve_step(oracle_name, pattern):
        namespace = globals()
        candidate = namespace.get(oracle_name)
        if callable(candidate):
            return candidate
        bases = []
        if "__file__" in namespace:
            bases.append(os.path.dirname(os.path.abspath(namespace["__file__"])))
        bases.append(os.getcwd())
        if sys.argv and sys.argv[0]:
            bases.append(os.path.dirname(os.path.abspath(sys.argv[0])))
        search_dirs = []
        for base in bases:
            if not base:
                continue
            parent = os.path.dirname(base)
            search_dirs += [base, os.path.join(base, "sub_problems"),
                            parent, os.path.join(parent, "sub_problems")]
        seen = set()
        search_dirs = [d for d in search_dirs if not (d in seen or seen.add(d))]
        for directory in search_dirs:
            for path in sorted(glob.glob(os.path.join(directory, pattern))):
                spec = importlib.util.spec_from_file_location(
                    os.path.basename(path)[:-3], path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, oracle_name):
                    return getattr(module, oracle_name)
        public = namespace.get(oracle_name.replace("_oracle_", "", 1))
        if callable(public):
            return public
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    segment_volume = _resolve_step(
        "_oracle_compute_segment_volume_factor", "*compute_segment_volume_factor*.py")
    thermal_moment = _resolve_step(
        "_oracle_compute_thermal_volume_moment", "*compute_thermal_volume_moment*.py")
    free_energy = _resolve_step(
        "_oracle_compute_free_energy_coefficients", "*compute_free_energy_coefficients*.py")
    critical_nucleus = _resolve_step(
        "_oracle_locate_critical_nucleus", "*locate_critical_nucleus*.py")
    barrier_ratio = _resolve_step(
        "_oracle_compute_thermal_barrier_ratio", "*compute_thermal_barrier_ratio*.py")

    angles = np.asarray(theta_d_deg, dtype=float)
    if angles.ndim != 1 or angles.size < 1:
        raise ValueError("theta_d_deg must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(angles)):
        raise ValueError("theta_d_deg must be finite")
    if np.any(angles <= 0.0) or np.any(angles >= 180.0):
        raise ValueError("theta_d_deg entries must lie strictly between 0 and 180 degrees")

    temperatures = np.asarray(t_liquid, dtype=float)
    if temperatures.shape != angles.shape:
        raise ValueError("t_liquid must have the same shape as theta_d_deg")
    if not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0.0):
        raise ValueError("t_liquid entries must be finite and strictly positive")

    # The model is written in the liquid-side contact angle, the supplement of
    # the measured bubble contact angle.
    theta = np.deg2rad(180.0 - angles)

    # Normalise the vapour state once, so no earlier step is ever handed a list,
    # an object array or a non-native float.
    vapour = np.asarray(vapour_state, dtype=float).ravel()
    if vapour.size != 3 or not np.all(np.isfinite(vapour)) or np.any(vapour <= 0.0):
        raise ValueError("vapour_state must hold three finite positive entries")

    ratios = np.empty(angles.shape, dtype=float)
    for index, angle in enumerate(theta):
        state_temperature = float(temperatures[index])
        volume_factor = float(segment_volume(float(angle)))
        moment_factor = float(thermal_moment(float(angle), volume_factor))

        # Sub-problem 05 supplies the triple [balance, laplace, capillary],
        # which is forwarded as it comes back. Casting to a native float array
        # only spares sub-problem 07 a list or an object array.
        coefficients = np.asarray(
            free_energy(volume_factor, moment_factor, state_temperature, vapour,
                        float(t_sat), float(p_liquid), float(gamma_lv), float(depth)),
            dtype=float).ravel()
        if coefficients.size != 3 or not np.all(np.isfinite(coefficients)):
            raise ValueError("the free-energy coefficients must hold three finite entries")

        critical_state = np.asarray(critical_nucleus(coefficients), dtype=float).ravel()
        if critical_state.size != 2 or not np.all(np.isfinite(critical_state)):
            raise ValueError("the critical state must hold a finite radius and energy")
        energy_critical = float(critical_state[1])
        ratios[index] = float(barrier_ratio(energy_critical, state_temperature))

    return ratios

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the eight measured states of the testbed (normal scenario) ---
        {
            "setup": """import numpy as np
theta_d_deg = np.array([117.5, 112.4, 106.8, 101.3, 61.4, 57.9, 53.6, 49.2])
t_liquid = np.array([792.1650227904338, 768.8660515318916, 745.5670802733495,
                     731.5876975182241, 736.2474917699326, 712.9485205113904,
                     689.6495492528483, 675.6701664977229])
vapour_state = np.array([461.5231157260608, 0.5883553522770233, 2227187.9287662064])
""",
            "call": "sweep_measured_contact_angles(theta_d_deg, t_liquid, vapour_state)",
            "gold_call": "_oracle_sweep_measured_contact_angles(theta_d_deg, t_liquid, vapour_state)",
        },
        # --- Valid: the hydrophilic branch alone ---
        {
            "setup": """import numpy as np
theta_d_deg = np.array([117.5, 112.4, 106.8, 101.3])
t_liquid = np.array([792.1650227904338, 768.8660515318916, 745.5670802733495,
                     731.5876975182241])
vapour_state = np.array([461.5231157260608, 0.5883553522770233, 2227187.9287662064])
""",
            "call": "sweep_measured_contact_angles(theta_d_deg, t_liquid, vapour_state)",
            "gold_call": "_oracle_sweep_measured_contact_angles(theta_d_deg, t_liquid, vapour_state)",
        },
        # --- Boundary: a single state and a hotter wall-adjacent liquid ---
        {
            "setup": """import numpy as np
theta_d_deg = np.array([90.0])
t_liquid = np.array([900.0])
vapour_state = np.array([461.5231157260608, 0.5883553522770233, 2227187.9287662064])
""",
            "call": "sweep_measured_contact_angles(theta_d_deg, t_liquid, vapour_state)",
            "gold_call": "_oracle_sweep_measured_contact_angles(theta_d_deg, t_liquid, vapour_state)",
        },
        # --- Edge: raised system pressure, thicker slab and a different surface tension ---
        {
            "setup": """import numpy as np
theta_d_deg = np.array([117.5, 61.4])
t_liquid = np.array([800.0, 820.0])
vapour_state = np.array([461.5231157260608, 1.1554824570743413, 2136459.0])
""",
            "call": "sweep_measured_contact_angles(theta_d_deg, t_liquid, vapour_state, 393.35, 202650.0, 0.0589, 5.0e-9)",
            "gold_call": "_oracle_sweep_measured_contact_angles(theta_d_deg, t_liquid, vapour_state, 393.35, 202650.0, 0.0589, 5.0e-9)",
        },
        # --- Invalid: a temperature table of the wrong length ---
        {
            "setup": """import numpy as np
vapour_state = np.array([461.5231157260608, 0.5883553522770233, 2227187.9287662064])
def run_model():
    try:
        sweep_measured_contact_angles(np.array([117.5, 61.4]), np.array([792.165]), vapour_state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sweep_measured_contact_angles(np.array([117.5, 61.4]), np.array([792.165]), vapour_state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: liquid too close to saturation for any state to carry a barrier ---
        {
            "setup": """import numpy as np
vapour_state = np.array([461.5231157260608, 0.5883553522770233, 2227187.9287662064])
def run_model():
    try:
        sweep_measured_contact_angles(np.array([117.5, 61.4]), np.array([400.0, 400.0]), vapour_state)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sweep_measured_contact_angles(np.array([117.5, 61.4]), np.array([400.0, 400.0]), vapour_state)
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
