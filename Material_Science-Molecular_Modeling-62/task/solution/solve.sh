#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def convert_reduced_temperatures(t_substrate_red: float, t_jump_red: float,
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

import numpy as np


def compute_vapour_state(t_sat: float, p_liquid: float,
                                 h_reference: float = 2441.7e3,
                                 t_reference: float = 298.15,
                                 t_critical: float = 647.096,
                                 watson_exponent: float = 0.38) -> np.ndarray:
    universal_gas_constant = 8.314462618
    molar_mass_water = 0.018015268

    for name, value in (("t_sat", t_sat), ("p_liquid", p_liquid),
                        ("h_reference", h_reference), ("t_reference", t_reference),
                        ("t_critical", t_critical),
                        ("watson_exponent", watson_exponent)):
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    if not float(t_sat) < float(t_critical):
        raise ValueError("t_sat must lie below the critical temperature")
    if not float(t_reference) < float(t_critical):
        raise ValueError("t_reference must lie below the critical temperature")

    # The increment being integrated is specific, so the gas constant must be
    # the mass-based one rather than the universal constant.
    r_specific = universal_gas_constant / molar_mass_water

    # Saturated steam at these pressures is well described as an ideal gas.
    rho_vapour = float(p_liquid) / (r_specific * float(t_sat))

    # Watson correlation for the decay of the latent heat towards the critical point.
    reduced = (float(t_critical) - float(t_sat)) / (float(t_critical) - float(t_reference))
    h_fg = float(h_reference) * reduced ** float(watson_exponent)

    return np.array([r_specific, rho_vapour, h_fg], dtype=float)

import numpy as np


def compute_segment_volume_factor(theta: float) -> float:
    if not (isinstance(theta, (int, float, np.floating)) and not isinstance(theta, bool)
            and np.isfinite(theta)):
        raise ValueError("theta must be a finite number")
    theta = float(theta)
    if not (0.0 < theta < np.pi):
        raise ValueError("theta must lie strictly between 0 and pi radians")

    # Area of the circular segment above the wall, divided by the squared radius.
    return float(np.pi - theta + 0.5 * np.sin(2.0 * theta))

import numpy as np


def compute_thermal_volume_moment(theta: float, volume_factor: float) -> float:
    if not (isinstance(theta, (int, float, np.floating)) and not isinstance(theta, bool)
            and np.isfinite(theta)):
        raise ValueError("theta must be a finite number")
    if not (isinstance(volume_factor, (int, float, np.floating))
            and not isinstance(volume_factor, bool)
            and np.isfinite(volume_factor) and float(volume_factor) > 0.0):
        raise ValueError("volume_factor must be a finite number > 0")
    theta = float(theta)
    if not (0.0 < theta < np.pi):
        raise ValueError("theta must lie strictly between 0 and pi radians")

    apex_factor = 1.0 + np.cos(theta)
    if apex_factor <= 0.0:
        raise ValueError("the apex height factor must be positive")

    # Plain volume less the first moment about the wall, divided by the apex height.
    numerator = 3.0 * float(volume_factor) - 2.0 * np.sin(theta) ** 3
    return float(numerator / (3.0 * apex_factor))

import numpy as np


def compute_free_energy_coefficients(volume_factor: float, moment_factor: float,
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

import numpy as np


def evaluate_available_energy_profile(coefficients: np.ndarray,
                                              radii: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=float).ravel()
    if coeffs.size != 3 or not np.all(np.isfinite(coeffs)):
        raise ValueError("coefficients must hold three finite entries")

    grid = np.asarray(radii, dtype=float)
    if grid.ndim != 1 or grid.size < 1:
        raise ValueError("radii must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("radii must be finite and strictly positive")

    balance, laplace, capillary = coeffs
    if capillary <= 0.0:
        raise ValueError("the capillary length must be strictly positive")

    return grid ** 2 * (balance + laplace * np.log1p(capillary / grid))

import numpy as np


def locate_critical_nucleus(coefficients: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(coefficients, dtype=float).ravel()
    if coeffs.size != 3 or not np.all(np.isfinite(coeffs)):
        raise ValueError("coefficients must hold three finite entries")

    balance, laplace, capillary = coeffs
    if balance >= 0.0:
        raise ValueError("no barrier exists unless the balance coefficient is negative")
    if laplace <= 0.0:
        raise ValueError("the Laplace coefficient must be strictly positive")
    if capillary <= 0.0:
        raise ValueError("the capillary length must be strictly positive")

    def _stationarity(radius):
        # Derivative of the increment with respect to the radius, divided by the
        # radius. Strictly decreasing, unbounded above as the radius tends to
        # zero and tending to twice the balance coefficient as it grows.
        return (2.0 * balance + 2.0 * laplace * np.log1p(capillary / radius)
                - laplace * capillary / (radius + capillary))

    # Bracket the sign change, then bisect. No starting guess is needed because
    # the stationarity condition is monotone.
    upper = capillary
    for _ in range(4000):
        if _stationarity(upper) < 0.0:
            break
        upper *= 2.0
    else:
        raise ValueError("failed to bracket the stationary radius from above")

    lower = upper
    for _ in range(4000):
        lower *= 0.5
        if _stationarity(lower) > 0.0:
            break
    else:
        raise ValueError("failed to bracket the stationary radius from below")

    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if _stationarity(middle) > 0.0:
            lower = middle
        else:
            upper = middle

    radius_critical = 0.5 * (lower + upper)
    energy_critical = radius_critical ** 2 * (
        balance + laplace * np.log1p(capillary / radius_critical))

    return np.array([radius_critical, energy_critical], dtype=float)

import numpy as np


def compute_thermal_barrier_ratio(energy_critical: float, t_liquid: float) -> float:
    boltzmann = 1.380649e-23

    for name, value in (("energy_critical", energy_critical), ("t_liquid", t_liquid)):
        if not (isinstance(value, (int, float, np.floating)) and not isinstance(value, bool)
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")

    return float(float(energy_critical) / (boltzmann * float(t_liquid)))

import glob
import importlib.util
import os
import sys

import numpy as np


def sweep_measured_contact_angles(theta_d_deg: np.ndarray, t_liquid: np.ndarray,
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
        public = namespace.get(oracle_name.replace("", "", 1))
        if callable(public):
            return public
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    segment_volume = _resolve_step(
        "compute_segment_volume_factor", "*compute_segment_volume_factor*.py")
    thermal_moment = _resolve_step(
        "compute_thermal_volume_moment", "*compute_thermal_volume_moment*.py")
    free_energy = _resolve_step(
        "compute_free_energy_coefficients", "*compute_free_energy_coefficients*.py")
    critical_nucleus = _resolve_step(
        "locate_critical_nucleus", "*locate_critical_nucleus*.py")
    barrier_ratio = _resolve_step(
        "compute_thermal_barrier_ratio", "*compute_thermal_barrier_ratio*.py")

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

import glob
import importlib.util
import os
import sys

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
        public = namespace.get(oracle_name.replace("", "", 1))
        if callable(public):
            return public
        raise RuntimeError(f"cannot resolve required step function {oracle_name}")

    convert_temperatures = _resolve_step(
        "convert_reduced_temperatures", "*convert_reduced_temperatures*.py")
    vapour_properties = _resolve_step(
        "compute_vapour_state", "*compute_vapour_state*.py")
    segment_volume = _resolve_step(
        "compute_segment_volume_factor", "*compute_segment_volume_factor*.py")
    thermal_moment = _resolve_step(
        "compute_thermal_volume_moment", "*compute_thermal_volume_moment*.py")
    free_energy = _resolve_step(
        "compute_free_energy_coefficients", "*compute_free_energy_coefficients*.py")
    energy_profile = _resolve_step(
        "evaluate_available_energy_profile", "*evaluate_available_energy_profile*.py")
    critical_nucleus = _resolve_step(
        "locate_critical_nucleus", "*locate_critical_nucleus*.py")
    sweep_states = _resolve_step(
        "sweep_measured_contact_angles", "*sweep_measured_contact_angles*.py")
    barrier_ratio = _resolve_step(
        "compute_thermal_barrier_ratio", "*compute_thermal_barrier_ratio*.py")

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
SCICODE_GOLD_EOF
