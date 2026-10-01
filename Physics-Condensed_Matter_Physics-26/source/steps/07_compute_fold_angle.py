"""
Orchestrator: compute the light-induced fold angle of a free-ended photoresponsive thermoelastic beam from its dimensional material, geometric and laser parameters.

The fold angle follows from chaining Steps 1 through 6: derive the dimensionless groups and elapsed time (Step 1), solve the merged transverse eigenbasis (Step 2), project the laser's dimensionless intensity profile onto every mode (Step 3), compute the cross-sectional moment-arm weight for every mode (Step 4), evaluate the transient modal temperature amplitudes at the requested elapsed time for every odd/even mode-pair combination (Step 5), and assemble the total transient thermal moment M(t) from the weights and the modal temperatures (Step 6). The quantity to return is the physical fold angle of the resulting V: the angle through which one arm is rotated relative to the other. The mechanical response is quasi-static and the ends are free, so each arm is straight, and its rotation is small, of the order of the aspect ratio h/L. Each arm's rotation is therefore taken equal to its physical slope, dv/dx1 (dimensional transverse deflection per unit dimensional length along the beam), and the fold angle is the magnitude of the jump in this physical slope across the illuminated point. The pipeline's quantities are dimensionless (Gamma1 from Step 1, M2 from Step 6), so they must be converted to this physical slope jump; that conversion is not given here.

This step is the final orchestrator: called with no arguments, it reproduces the exact benchmark scenario and returns the fold angle in radians.

Returns
-------
float, the fold angle phi(t) in radians: the magnitude of the jump in the physical slope dv/dx1 across the illuminated point (small-rotation approximation), >= 0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_fold_angle(
    length: float = 0.025,
    thickness: float = 0.0025,
    thermal_conductivity: float = 0.55,
    specific_heat: float = 4.0e3,
    density: float = 1000.0,
    thermal_expansion: float = -0.015,
    attenuation_coefficient: float = 220.0,
    heat_conversion_fraction: float = 0.8,
    heat_transfer_coefficient: float = 150.0,
    laser_radius: float = 0.005,
    incident_intensity_prefactor: float = 2.5e4,
    elapsed_time: float = 3.0,
    n_modes: int = 4,
) -> float:
    """Light-induced fold angle of a free-ended photoresponsive beam.

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
        and nonzero (may be negative, as for a deswelling hydrogel).
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
        The ratio P/(pi*w_dim^2), where P is the laser power (W), giving
        units of W/m^2, > 0.
    elapsed_time : float
        Physical elapsed time since the laser was switched on (s), >= 0.
    n_modes : int
        Number of odd and even transverse modes to retain (each), an
        integer >= 1.

    Returns
    -------
    phi : float
        The fold angle in radians, >= 0: the relative rotation of the two
        arms, taken as the magnitude of the jump in the physical slope
        dv/dx1 across the illuminated point (small-rotation approximation).

    Raises
    ------
    ValueError
        If any underlying step raises ValueError on its own inputs, or if
        length, thickness, thermal_conductivity, specific_heat, density,
        attenuation_coefficient, heat_transfer_coefficient, laser_radius or
        incident_intensity_prefactor is not a finite number > 0; if
        thermal_expansion is not finite and nonzero; if
        heat_conversion_fraction is not in (0, 1]; if elapsed_time is not a
        finite number >= 0; or if n_modes is not an integer >= 1.
    """
    return phi  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_fold_angle(
    length: float = 0.025,
    thickness: float = 0.0025,
    thermal_conductivity: float = 0.55,
    specific_heat: float = 4.0e3,
    density: float = 1000.0,
    thermal_expansion: float = -0.015,
    attenuation_coefficient: float = 220.0,
    heat_conversion_fraction: float = 0.8,
    heat_transfer_coefficient: float = 150.0,
    laser_radius: float = 0.005,
    incident_intensity_prefactor: float = 2.5e4,
    elapsed_time: float = 3.0,
    n_modes: int = 4,
) -> float:
    import numpy as np

    if not (isinstance(n_modes, (int, np.integer)) and not isinstance(n_modes, bool) and int(n_modes) >= 1):
        raise ValueError("n_modes must be an integer >= 1")
    n_modes = int(n_modes)

    # Step 1: dimensionless groups.
    groups = _oracle_compute_dimensionless_groups(
        length, thickness, thermal_conductivity, specific_heat, density,
        thermal_expansion, attenuation_coefficient, heat_conversion_fraction,
        heat_transfer_coefficient, laser_radius, incident_intensity_prefactor, elapsed_time,
    )
    bi, beta, w, gamma1_dimless, t = groups

    # Step 2: merged transverse eigenbasis.
    modes = _oracle_solve_transverse_eigenbasis(bi, n_modes)

    # Step 3: per-mode heat-source projection.
    proj = _oracle_project_heat_source_onto_modes(modes, beta, w)

    # Step 4: per-mode cross-sectional moment-arm weights.
    weights = _oracle_compute_cross_sectional_moment_weights(modes)

    # Step 5: transient modal temperature amplitudes.
    theta = _oracle_compute_transient_modal_temperatures(modes, proj, w, t)

    # Step 6: assemble the total thermal moment M2(t) (first moment of the inner temperature field).
    m_t = _oracle_assemble_thermal_moment(modes, weights, theta)

    # Step 7 (this orchestrator): the physical fold angle.
    # The reduced model's hinge condition, in its scaled variables X1 = x1/L and u2 = v/h, is
    # [d u2 / d X1] = -12 * Gamma1 * M2 (the factor 12 = h^4 / I, I = h^4/12 the second moment of area).
    # The physical slope is dv/dx1 = (h/L) * d u2/d X1, so the physical slope jump is
    # -12 * (h/L) * Gamma1 * M2 = -gamma1 * M*, with M* = 12 * Delta_theta * M2 the thermal moment in kelvin
    # (gamma1 * Delta_theta = (h/L) * Gamma1). With free ends each arm is straight, and in the small-rotation
    # approximation each arm's rotation equals its slope, so the fold angle is the magnitude of the slope jump.
    # (The reduced form with v~ = sqrt(12) u2 and M~ = 12 sqrt(12) M2 gives the same: (h/L) |Gamma1 M~| / sqrt(12).)
    aspect = float(thickness) / float(length)
    phi = float(12.0 * aspect * abs(gamma1_dimless * m_t))

    return phi

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the exact benchmark scenario, all default arguments ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_fold_angle()",
            "gold_call": "_oracle_compute_fold_angle()",
        },
        # --- Valid: a stronger laser power at the same elapsed time ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_fold_angle(incident_intensity_prefactor=4.0e4)",
            "gold_call": "_oracle_compute_fold_angle(incident_intensity_prefactor=4.0e4)",
        },
        # --- Valid: a positive thermal-expansion coefficient (swelling rather than deswelling) ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_fold_angle(thermal_expansion=0.02, attenuation_coefficient=90.0)",
            "gold_call": "_oracle_compute_fold_angle(thermal_expansion=0.02, attenuation_coefficient=90.0)",
        },
        # --- Boundary: elapsed_time = 0 must give exactly zero fold angle ---
        {
            "setup": """import numpy as np
def check(fn):
    return int(float(fn(elapsed_time=0.0)) == 0.0)
""",
            "call": "check(compute_fold_angle)",
            "gold_call": "check(_oracle_compute_fold_angle)",
        },
        # --- Valid: a late elapsed time (40 s), close to the fully equilibrated fold angle,
        # compared directly with the reference (one call on each side) ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_fold_angle(elapsed_time=40.0)",
            "gold_call": "_oracle_compute_fold_angle(elapsed_time=40.0)",
        },
        # --- Valid: eight modes of each parity instead of four, compared directly with the
        # reference (one call on each side) ---
        {
            "setup": """import numpy as np
""",
            "call": "compute_fold_angle(n_modes=8)",
            "gold_call": "_oracle_compute_fold_angle(n_modes=8)",
        },
        # --- Invalid: non-positive thickness ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_fold_angle(thickness=0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_fold_angle(thickness=0.0)
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
def run_model():
    try:
        compute_fold_angle(heat_conversion_fraction=1.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_fold_angle(heat_conversion_fraction=1.4)
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
