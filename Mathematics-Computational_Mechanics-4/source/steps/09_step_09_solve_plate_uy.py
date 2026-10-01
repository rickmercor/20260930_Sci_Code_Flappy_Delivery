"""
Orchestrate the full pipeline and return the target scalar.



This step builds the material-point lattice, enumerates the directed bonds with the pre-crack removed, calibrates the micro-modulus for every horizon, computes each bond's squared critical stretch, estimates each region's critical time step, verifies the reference configuration, integrates the body to the final time, and reports the vertical displacement at the probe point.

Region identifiers are assigned from the point coordinates: a point belongs to the last region whose lower x-bound it exceeds, with region 0 as the default. The prescribed step of each region must be strictly below that region's critical time step.

Two properties of the reference configuration are checked before integrating. The internal force density of the undeformed, fully intact body must vanish identically, and no bond may fail at zero deformation. The bond states surviving the second check are the states from which the integration starts.

The external loading is a constant body-force density applied to the single outermost row of points along the top and bottom edges, positive on the top and negative on the bottom, of magnitude applied_stress divided by the local spacing. Points within no_fail_width of either edge are exempt from failure.

The probe point must coincide with a material point of the lattice, within an absolute tolerance of 1e-9 times the larger of height and one.

The returned value is the vertical displacement at the probe point at the final time, expressed in micrometres.

Raises ValueError if: region_steps does not have shape (R,) matching spacings, or is not finite and strictly positive; probe_point does not have shape (2,) or is not finite; no_fail_width is not finite or is negative; applied_stress is not finite; probe_point does not coincide with a material point; any region step is not strictly below that region's critical time step; the reference configuration carries non-zero internal force; any bond fails in the reference configuration. Conditions raised by the steps it calls propagate unchanged.

The quantities assembled by the preceding steps are not independent. The micro-modulus follows from the horizon field, the failure thresholds follow from the micro-modulus and the horizon field together, the admissible time steps follow from the micro-modulus and the connectivity, and the integrator consumes all of them. An error introduced anywhere upstream propagates to the final displacement without any intermediate quantity looking obviously wrong, which is why the pipeline is worth assembling explicitly rather than in a single monolithic routine.

Two properties of the reference configuration are worth verifying before committing to a long integration. An undeformed body must carry no internal force, since every bond stretch vanishes; a non-zero result indicates that the reference lengths and the deformed lengths have been formed inconsistently. And no bond should fail before any load is applied; a failure at zero deformation would mean the calibrated thresholds are not positive quantities of the expected magnitude. Both checks cost one force evaluation and one state update, and both fail loudly rather than producing a plausible wrong trajectory.

Reporting a displacement at a single interior material point, rather than an aggregate such as total damage or dissipated energy, keeps the target sensitive to the whole computation while remaining a directly observable quantity. The point is chosen away from the loaded boundaries, away from the pre-crack, and away from any symmetry line of the discretisation, so that it responds to the wave field, to the bonds that break, and to the coupling across the refinement interface rather than to any one of these alone.

Returns
-------
float, the vertical displacement at the probe point at final_time, in micrometres, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_plate_uy(youngs_modulus: float, density: float, fracture_energy: float,
                   influence_exponent: float, x_bounds: np.ndarray, spacings: np.ndarray,
                   height: float, horizon_ratio: float, crack_y: float, crack_x_max: float,
                   no_fail_width: float, applied_stress: float, region_steps: np.ndarray,
                   final_time: float, probe_point: np.ndarray) -> float:
    '''Run the full pipeline and return the probe displacement in micrometres.

    Parameters
    ----------
    youngs_modulus : float
        Young's modulus E of the material.
    density : float
        Mass density.
    fracture_energy : float
        Critical energy release rate G_c.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.
    x_bounds : np.ndarray
        (R, 2) float array of region x-intervals, contiguous and increasing.
    spacings : np.ndarray
        (R,) float array of grid spacings, one per region.
    height : float
        Extent of the plate in y, spanning [0, height].
    horizon_ratio : float
        Ratio m in delta = m * spacing.
    crack_y : float
        Ordinate of the horizontal pre-crack line.
    crack_x_max : float
        Largest abscissa at which the pre-crack severs a bond.
    no_fail_width : float
        Width of the failure-exempt layer at the top and bottom edges.
    applied_stress : float
        Magnitude of the applied edge traction.
    region_steps : np.ndarray
        (R,) float array of local time steps, one per region.
    final_time : float
        Time at which the integration stops.
    probe_point : np.ndarray
        (2,) float array giving the reference coordinates of the probe point.

    Returns
    -------
    u_y : float
        Vertical displacement at the probe point at final_time, in micrometres.
    '''
    u_y = 0.0
    return u_y  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_plate_uy(youngs_modulus: float, density: float, fracture_energy: float,
                           influence_exponent: float, x_bounds: np.ndarray, spacings: np.ndarray,
                           height: float, horizon_ratio: float, crack_y: float, crack_x_max: float,
                           no_fail_width: float, applied_stress: float,
                           region_steps: np.ndarray, final_time: float,
                           probe_point: np.ndarray) -> float:
    """Reference implementation."""
    xb = np.asarray(x_bounds, dtype=float)
    sp = np.asarray(spacings, dtype=float)
    rs = np.asarray(region_steps, dtype=float)
    pp = np.asarray(probe_point, dtype=float)
    if (rs.ndim != 1 or rs.shape[0] != sp.shape[0] or not np.all(np.isfinite(rs))
            or np.any(rs <= 0.0)):
        raise ValueError("region_steps must have shape (R,) matching spacings and be finite and > 0")
    if pp.shape != (2,) or not np.all(np.isfinite(pp)):
        raise ValueError("probe_point must have shape (2,) and be finite")
    if not (np.isfinite(no_fail_width) and no_fail_width >= 0.0):
        raise ValueError("no_fail_width must be finite and >= 0")
    if not np.isfinite(applied_stress):
        raise ValueError("applied_stress must be finite")

    lattice = _oracle_build_point_lattice(xb, sp, height, horizon_ratio)
    d2 = (lattice[:, 0] - pp[0]) ** 2 + (lattice[:, 1] - pp[1]) ** 2
    probe = int(np.argmin(d2))
    if d2[probe] > (1e-9 * max(height, 1.0)) ** 2:
        raise ValueError("probe_point does not coincide with a material point")

    bonds = _oracle_build_directed_bonds(lattice, crack_y, crack_x_max)
    micromodulus = _oracle_calibrate_micromodulus(youngs_modulus, lattice[:, 3],
                                                  influence_exponent)
    sc2 = _oracle_bond_critical_stretch_squared(lattice, bonds, micromodulus,
                                                fracture_energy, influence_exponent)
    N = lattice.shape[0]
    region_ids = np.zeros(N, dtype=np.int64)
    for r in range(1, xb.shape[0]):
        region_ids[lattice[:, 0] > xb[r, 0]] = r
    hc = _oracle_regional_critical_time_step(lattice, bonds, micromodulus, region_ids,
                                             density, influence_exponent)
    if np.any(rs >= hc):
        raise ValueError("each region step must be below that region's critical time step")

    zero_u = np.zeros((N, 2), dtype=float)
    intact = np.ones(bonds.shape[0], dtype=float)
    f0 = _oracle_internal_force_density(lattice, bonds, micromodulus, intact,
                                        zero_u, influence_exponent)
    if np.any(f0 != 0.0):
        raise ValueError("the reference configuration must carry zero internal force")

    no_fail = ((lattice[:, 1] < no_fail_width) |
               (lattice[:, 1] > height - no_fail_width)).astype(float)
    mu0 = _oracle_update_bond_states(lattice, bonds, intact, sc2, zero_u, no_fail)
    if np.any(mu0 != 1.0):
        raise ValueError("no bond may fail in the reference configuration")

    body_force = np.zeros((N, 2), dtype=float)
    dx = lattice[:, 2]
    top = lattice[:, 1] > height - dx
    bot = lattice[:, 1] < dx
    body_force[top, 1] = applied_stress / dx[top]
    body_force[bot, 1] = -applied_stress / dx[bot]

    local_steps = rs[region_ids]
    u = _oracle_integrate_avv(lattice, bonds, micromodulus, sc2, no_fail, body_force,
                              local_steps, final_time, density, influence_exponent, mu0)
    return float(u[probe, 1] * 1e6)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the full production configuration ---
        {
            "setup": """import numpy as np
youngs_modulus = 72e9
density = 2440.0
fracture_energy = 135.0
influence_exponent = 2.0
x_bounds = np.array([[0.0, 0.02], [0.02, 0.04]], dtype=float)
spacings = np.array([0.002, 0.001], dtype=float)
height = 0.02
horizon_ratio = 3.015
crack_y = 0.0101
crack_x_max = 0.0151
no_fail_width = 0.0021
applied_stress = 14e6
region_steps = np.array([4e-8, 2e-8], dtype=float)
final_time = 2e-5
probe_point = np.array([0.0255, 0.0155], dtype=float)
""",
            "call": "solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)",
            "gold_call": "_oracle_solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)",
        },
        # --- Boundary: single uniform region, one local grid ---
        {
            "setup": """import numpy as np
youngs_modulus = 72e9
density = 2440.0
fracture_energy = 135.0
influence_exponent = 2.0
x_bounds = np.array([[0.0, 0.012]], dtype=float)
spacings = np.array([0.002], dtype=float)
height = 0.008
horizon_ratio = 3.015
crack_y = 0.0041
crack_x_max = 0.005
no_fail_width = 0.0021
applied_stress = 30e6
region_steps = np.array([4e-8], dtype=float)
final_time = 4e-6
probe_point = np.array([0.007, 0.005], dtype=float)
""",
            "call": "solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)",
            "gold_call": "_oracle_solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)",
        },
        # --- Edge: three regions with a constant influence function ---
        {
            "setup": """import numpy as np
youngs_modulus = 72e9
density = 2440.0
fracture_energy = 135.0
influence_exponent = 0.0
x_bounds = np.array([[0.0, 0.008], [0.008, 0.016], [0.016, 0.024]], dtype=float)
spacings = np.array([0.002, 0.001, 0.002], dtype=float)
height = 0.008
horizon_ratio = 3.015
crack_y = 0.0041
crack_x_max = 0.005
no_fail_width = 0.0021
applied_stress = 30e6
region_steps = np.array([4e-8, 2e-8, 4e-8], dtype=float)
final_time = 4e-6
probe_point = np.array([0.019, 0.005], dtype=float)
""",
            "call": "solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)",
            "gold_call": "_oracle_solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)",
        },
        # --- Invalid: a region step at or above its critical time step ---
        {
            "setup": """import numpy as np
youngs_modulus = 72e9
density = 2440.0
fracture_energy = 135.0
influence_exponent = 2.0
x_bounds = np.array([[0.0, 0.02], [0.02, 0.04]], dtype=float)
spacings = np.array([0.002, 0.001], dtype=float)
height = 0.02
horizon_ratio = 3.015
crack_y = 0.0101
crack_x_max = 0.0151
no_fail_width = 0.0021
applied_stress = 14e6
region_steps = np.array([4e-7, 2e-8], dtype=float)
final_time = 2e-5
probe_point = np.array([0.0255, 0.0155], dtype=float)
def run_model():
    try:
        solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: probe point not on the lattice ---
        {
            "setup": """import numpy as np
youngs_modulus = 72e9
density = 2440.0
fracture_energy = 135.0
influence_exponent = 2.0
x_bounds = np.array([[0.0, 0.02], [0.02, 0.04]], dtype=float)
spacings = np.array([0.002, 0.001], dtype=float)
height = 0.02
horizon_ratio = 3.015
crack_y = 0.0101
crack_x_max = 0.0151
no_fail_width = 0.0021
applied_stress = 14e6
region_steps = np.array([4e-8, 2e-8], dtype=float)
final_time = 2e-5
probe_point = np.array([0.0256, 0.0155], dtype=float)
def run_model():
    try:
        solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: region_steps length does not match the number of regions ---
        {
            "setup": """import numpy as np
youngs_modulus = 72e9
density = 2440.0
fracture_energy = 135.0
influence_exponent = 2.0
x_bounds = np.array([[0.0, 0.02], [0.02, 0.04]], dtype=float)
spacings = np.array([0.002, 0.001], dtype=float)
height = 0.02
horizon_ratio = 3.015
crack_y = 0.0101
crack_x_max = 0.0151
no_fail_width = 0.0021
applied_stress = 14e6
region_steps = np.array([4e-8], dtype=float)
final_time = 2e-5
probe_point = np.array([0.0255, 0.0155], dtype=float)
def run_model():
    try:
        solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_plate_uy(youngs_modulus, density, fracture_energy, influence_exponent, x_bounds, spacings, height, horizon_ratio, crack_y, crack_x_max, no_fail_width, applied_stress, region_steps, final_time, probe_point)
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
