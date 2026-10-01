"""
Chain the sub-problem functions 01-09 end-to-end on the measured flow-boiling states and return the largest dimensionless nucleation barrier.

Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (convert_reduced_temperatures, compute_vapour_state, compute_segment_volume_factor, compute_thermal_volume_moment, compute_free_energy_coefficients, evaluate_available_energy_profile, locate_critical_nucleus, compute_thermal_barrier_ratio, sweep_measured_contact_angles) rather than reimplementing them.

This step runs the whole measurement end to end. It (i) converts the reduced substrate temperature and each state's own reduced interfacial temperature jump into kelvin with sub-problem 01, building one wall-adjacent liquid temperature per measured state, (ii) evaluates the mass-based vapour properties at the saturation state with sub-problem 02, (iii) reflects every measured bubble contact angle into a liquid-side angle and pushes it through the geometry of sub-problems 03 and 04, the free-energy reduction of sub-problem 05, the numerical extremum of sub-problem 07 and the thermal normalisation of sub-problem 08, all inside the sweep of sub-problem 09, and (iv) selects the state with the largest barrier and confirms, with the profile of sub-problem 06, that its critical radius really is a maximum of the available-energy increment rather than an inflection or a minimum.




Because each state carries its own interfacial jump, the sweep is not a pure geometry scan and the largest barrier need not fall at either end of the measured table. The shape factors fall as the liquid-side angle rises, while the superheat entering the balance coefficient falls as the jump grows, and the two act against each other along a driving-force sweep. The state that governs is therefore whichever compromise between the two happens to sit closest to the threshold where a barrier ceases to exist, and it can belong to either surface; nothing short of evaluating every state establishes which one it is.

The returned scalar answers a question the simulation cannot answer by itself: whether the thermodynamic model that is normally invoked to rationalise molecular dynamics observations of flow boiling assigns a nucleation barrier of a size compatible with those observations. If the largest barrier across the measured table were of order tens, the model would be a quantitative account of the same events the simulation resolves. If it is orders of magnitude larger, the model retains only comparative value, and its ordering of the states must be defended on the structure of the free-energy balance rather than on any agreement of magnitudes, while its critical radius should be checked against the thickness of liquid actually available to hold a nucleus.

Returns
-------
float: the largest dimensionless nucleation barrier across the measured states, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_nucleation_barrier_pipeline(t_substrate_red: float = 9.95,
                                    t_jump_red: tuple = (1.45, 1.70, 1.95, 2.10,
                                                         2.05, 2.30, 2.55, 2.70),
                                    theta_d_deg: tuple = (117.5, 112.4, 106.8, 101.3,
                                                          61.4, 57.9, 53.6, 49.2),
                                    epsilon_ev: float = 0.008031,
                                    t_sat: float = 373.15,
                                    p_liquid: float = 101325.0,
                                    gamma_lv: float = 67.70e-3,
                                    depth: float = 31.30e-10) -> float:
    """Run the full nucleation-barrier measurement on the flow-boiling testbed.

    Parameters
    ----------
    t_substrate_red : float
        Substrate temperature in reduced Lennard-Jones units (> 0), shared by
        every measured state.
    t_jump_red : tuple
        Interfacial temperature jump of each measured state in reduced
        Lennard-Jones units (>= 0), of the same length as theta_d_deg.
    theta_d_deg : tuple
        Measured quasi-steady bubble contact angles in degrees, each strictly
        between 0 and 180.
    epsilon_ev : float
        Lennard-Jones well depth of the reference pair in electronvolts (> 0).
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
    barrier_max : float
        Largest critical activation energy across the measured states, in units
        of the thermal energy of its own wall-adjacent liquid, as a native
        Python float.
    """
    return barrier_max  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import glob
import importlib.util
import os
import sys

import numpy as np


def _oracle_run_nucleation_barrier_pipeline(t_substrate_red: float = 9.95,
                                            t_jump_red: tuple = (1.45, 1.70, 1.95, 2.10,
                                                                 2.05, 2.30, 2.55, 2.70),
                                            theta_d_deg: tuple = (117.5, 112.4, 106.8, 101.3,
                                                                  61.4, 57.9, 53.6, 49.2),
                                            epsilon_ev: float = 0.008031,
                                            t_sat: float = 373.15,
                                            p_liquid: float = 101325.0,
                                            gamma_lv: float = 67.70e-3,
                                            depth: float = 31.30e-10) -> float:
    # -- Resolve the oracle functions of sub-problems 01-09. Preference order:
    #    (1) already present in the executing namespace (shared-namespace
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

    convert_temperatures = _resolve_step(
        "_oracle_convert_reduced_temperatures", "*convert_reduced_temperatures*.py")
    vapour_properties = _resolve_step(
        "_oracle_compute_vapour_state", "*compute_vapour_state*.py")
    segment_volume = _resolve_step(
        "_oracle_compute_segment_volume_factor", "*compute_segment_volume_factor*.py")
    thermal_moment = _resolve_step(
        "_oracle_compute_thermal_volume_moment", "*compute_thermal_volume_moment*.py")
    free_energy = _resolve_step(
        "_oracle_compute_free_energy_coefficients", "*compute_free_energy_coefficients*.py")
    energy_profile = _resolve_step(
        "_oracle_evaluate_available_energy_profile", "*evaluate_available_energy_profile*.py")
    critical_nucleus = _resolve_step(
        "_oracle_locate_critical_nucleus", "*locate_critical_nucleus*.py")
    sweep_states = _resolve_step(
        "_oracle_sweep_measured_contact_angles", "*sweep_measured_contact_angles*.py")
    barrier_ratio = _resolve_step(
        "_oracle_compute_thermal_barrier_ratio", "*compute_thermal_barrier_ratio*.py")

    # -- Validate the orchestrator inputs.
    for name, value in (("t_sat", t_sat), ("p_liquid", p_liquid),
                        ("gamma_lv", gamma_lv), ("depth", depth)):
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    angles = np.asarray(theta_d_deg, dtype=float).ravel()
    if angles.size < 1:
        raise ValueError("theta_d_deg must hold at least one measured angle")
    jumps = np.asarray(t_jump_red, dtype=float).ravel()
    if jumps.shape != angles.shape:
        raise ValueError("t_jump_red must hold one jump per measured angle")

    # -- Sub-problem 01: reduced temperatures to kelvin, one state at a time,
    #    because each state carries its own interfacial jump.
    t_liquid = np.array(
        [float(convert_temperatures(float(t_substrate_red), float(jump),
                                    float(epsilon_ev))[2]) for jump in jumps],
        dtype=float)

    # -- Sub-problem 02: mass-based vapour properties at saturation.
    vapour_state = vapour_properties(float(t_sat), float(p_liquid))

    # -- Sub-problems 03-05, 07-08 through the sweep of sub-problem 09.
    ratios = sweep_states(angles, t_liquid, vapour_state, float(t_sat),
                          float(p_liquid), float(gamma_lv), float(depth))

    # -- Sub-problems 03-07: confirm that the governing state's critical radius
    #    really is a maximum of the available-energy increment.
    governing = int(np.argmax(ratios))
    theta = float(np.deg2rad(180.0 - angles[governing]))
    volume_factor = float(segment_volume(theta))
    moment_factor = float(thermal_moment(theta, volume_factor))
    # Cast to a native float array so that no earlier step is ever handed a
    # list or an object array.
    vapour = np.asarray(vapour_state, dtype=float).ravel()
    if vapour.size != 3 or not np.all(np.isfinite(vapour)) or np.any(vapour <= 0.0):
        raise ValueError("vapour_state must hold three finite positive entries")
    t_governing = float(t_liquid[governing])

    # Sub-problem 05 supplies the triple [balance, laplace, capillary], which is
    # forwarded as it comes back.
    coefficients = np.asarray(
        free_energy(volume_factor, moment_factor, t_governing, vapour,
                    float(t_sat), float(p_liquid), float(gamma_lv),
                    float(depth)), dtype=float).ravel()
    if coefficients.size != 3 or not np.all(np.isfinite(coefficients)):
        raise ValueError("the free-energy coefficients must hold three finite entries")
    critical_state = np.asarray(critical_nucleus(coefficients), dtype=float).ravel()
    if critical_state.size != 2 or not np.all(np.isfinite(critical_state)):
        raise ValueError("the critical state must hold a finite radius and energy")
    radius_critical = float(critical_state[0])
    probe = energy_profile(coefficients,
                           np.array([0.5 * radius_critical, radius_critical,
                                     2.0 * radius_critical], dtype=float))
    if not (probe[1] > probe[0] and probe[1] > probe[2]):
        raise ValueError("the located critical radius is not a maximum of the profile")

    # -- Sub-problem 08: the governing ratio is re-derived here from the barrier
    #    this branch located, independently of the sweep, and the two readings
    #    must agree to floating-point tolerance.
    answer = float(np.max(ratios))
    direct = float(barrier_ratio(float(critical_state[1]), t_governing))
    if not np.isclose(direct, answer, rtol=1.0e-9, atol=0.0):
        raise ValueError("the governing barrier ratio disagrees with the sweep maximum")

    return answer

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: final-answer configuration (normal scenario) ---
        {
            "setup": """import numpy as np
t_substrate_red = 9.95
t_jump_red = (1.45, 1.70, 1.95, 2.10, 2.05, 2.30, 2.55, 2.70)
""",
            "call": "run_nucleation_barrier_pipeline(t_substrate_red, t_jump_red)",
            "gold_call": "_oracle_run_nucleation_barrier_pipeline(t_substrate_red, t_jump_red)",
        },
        # --- Integration: hydrophobic branch alone at a hotter substrate ---
        {
            "setup": """import numpy as np
theta_d_deg = (61.4, 57.9, 53.6, 49.2)
t_jump_red = (2.05, 2.30, 2.55, 2.70)
""",
            "call": "run_nucleation_barrier_pipeline(10.49, t_jump_red, theta_d_deg)",
            "gold_call": "_oracle_run_nucleation_barrier_pipeline(10.49, t_jump_red, theta_d_deg)",
        },
        # --- Integration (boundary): no interfacial jump, so the liquid sits at the wall ---
        {
            "setup": """import numpy as np
theta_d_deg = (117.5, 61.4)
t_jump_red = (0.0, 0.0)
""",
            "call": "run_nucleation_barrier_pipeline(9.95, t_jump_red, theta_d_deg)",
            "gold_call": "_oracle_run_nucleation_barrier_pipeline(9.95, t_jump_red, theta_d_deg)",
        },
        # --- Integration (edge): raised system pressure and a thicker liquid slab ---
        {
            "setup": """import numpy as np
theta_d_deg = (117.5, 106.8, 61.4, 53.6)
t_jump_red = (1.45, 1.95, 2.05, 2.55)
""",
            "call": "run_nucleation_barrier_pipeline(10.49, t_jump_red, theta_d_deg, 0.008031, 393.35, 202650.0, 0.0589, 5.0e-9)",
            "gold_call": "_oracle_run_nucleation_barrier_pipeline(10.49, t_jump_red, theta_d_deg, 0.008031, 393.35, 202650.0, 0.0589, 5.0e-9)",
        },
        # --- Invalid: substrate too cold for any measured state to carry a barrier ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_nucleation_barrier_pipeline(6.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_nucleation_barrier_pipeline(6.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: one jump per state is required ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_nucleation_barrier_pipeline(9.95, (1.45, 1.70), (117.5, 112.4, 106.8))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_nucleation_barrier_pipeline(9.95, (1.45, 1.70), (117.5, 112.4, 106.8))
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
