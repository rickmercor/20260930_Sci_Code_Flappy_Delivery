"""
This stage runs the whole measurement, from the raw configuration to the number the task grades. Stage 1 turns the orientation list into axial surrogates and collapses the queue into the uniform bar a homogeneous reading would assume. Stage 2 lays the specimen out inside the periodic cell twice over, once with the padding the harmonic solver needs, very compliant and massless, and once with the padding the march needs, stress free and carrying the specimen specimen_density, and gives every link its modulus. Stage 3 solves the harmonic pencil and fixes the baseline frequencies. Stage 4 supplies the implicit operator and its preconditioned inverse, which stage 5 uses to march the struck bar forward and write down the receptor displacement. Stage 6 collapses that record into a spectrum and sharpens the line belonging to each baseline frequency below the ceiling.

What is graded is the mean, over every axial resonance of the specimen below the ceiling, of the signed difference between the sharpened line and the baseline frequency. The two padding constructions differ in exactly one respect that the specimen can feel: the padding used by the march carries mass, and the link joining it to the specimen carries the specimen's own stiffness, so one voxel of fictitious mass hangs on each free face of the bar while the march runs. That mass is absent from the harmonic baseline. The mean difference is therefore a measurement of the fictitious inertia the embedding adds at the boundary, and averaging it over the whole set of resonances below the ceiling, rather than reading one mode, is what makes it a property of the specimen rather than of a mode.

The pulse strikes the first specimen voxel along the bar axis and the receptor sits on the last specimen voxel, so the two are the voxels adjacent to the two end faces. The Newmark parameters are one quarter and one half. Padding for the march is built with a stiffness factor of zero and a specimen_density factor of one; padding for the harmonic solver with a stiffness factor of one part in ten million and a specimen_density factor of zero.

Alongside the graded mean the stage returns what verifies it: how many axial resonances the ceiling admits, how many resonances of both families lie below it, the mean relative shift against twice the ratio of voxel size to specimen bar_length, the fictitious mass per face in units of one voxel of specimen mass, the sharpened offset and the log curvature of the lowest line, the lowest transverse natural frequency, the homogenised constants and the analytic ladder they imply.

Returns
-------
dict, the mean boundary-inertia gap with every reported intermediate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def report_boundary_inertia_gap(
    euler_angles: np.ndarray,
    crystal_constants: dict,
    grain_voxels: np.ndarray,
    specimen_density: float,
    bar_length: float,
    n_pad: int,
    time_step: float,
    n_steps: int,
    traction_amplitude: float,
    gaussian_width: float,
    gaussian_centre: float,
    traction_axis: np.ndarray,
    ceiling: float,
    search_fraction: float,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Measure the fictitious inertia that the periodic embedding hangs on the ends of the bar.

    Parameters
    ----------
    euler_angles : np.ndarray
        Orientation angles in degrees, shape (n_grains, 3).
    crystal_constants : dict
        The five hexagonal entries in pascal, under the keys c11, c33, c12, c13 and c44.
    grain_voxels : np.ndarray
        Voxels held by each grain, an integer array of shape (n_grains,).
    specimen_density : float
        Density of the specimen in kilogram per cubic metre.
    bar_length : float
        Length of the specimen in metre.
    n_pad : int
        How many padding voxels follow the specimen.
    time_step : float
        Length of one time step in second.
    n_steps : int
        How many time steps to take.
    traction_amplitude : float
        Height of the traction pulse in pascal.
    gaussian_width : float
        Width of one Gaussian in second.
    gaussian_centre : float
        Where the earlier Gaussian is centred, in second.
    traction_axis : np.ndarray
        Aim of the traction, a non-zero vector of three entries.
    ceiling : float
        Frequency in hertz below which the axial resonances are collected.
    search_fraction : float
        Half-width of the line search window as a fraction of the target frequency.
    residual_tolerance : float
        Relative residual demanded of the linear solver.
    iteration_ceiling : int
        Ceiling on the linear solver iteration count.

    Returns
    -------
    dict
        Under the keys mean_gap, mode_count, resonance_count, eigen_longitudinal, refined_line, gap, mean_relative_shift, voxel_mass_fraction, face_mass_ratio, first_offset, first_curvature, bin_spacing, first_transverse, analytic_first_longitudinal, young_hom, poisson_hom, longitudinal_speed and mean_cg_iterations.

    Raises
    ------
    ValueError
        When ceiling fails to sit above zero, when no axial resonance falls below it, when the voxels total an even number, or when an argument breaks a condition of one of the earlier stages, whose rejections travel through unchanged.
    RuntimeError
        When an implicit solve of the march misses the requested relative residual.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _newmark_pair():
    """The average-acceleration parameters used throughout this configuration."""
    return 0.25, 0.5


def _padding_factors():
    """Stiffness and specimen_density factors of the harmonic padding, then of the padding used by the march."""
    return (1.0e-7, 0.0), (0.0, 1.0)


def _oracle_report_boundary_inertia_gap(
    euler_angles: np.ndarray,
    crystal_constants: dict,
    grain_voxels: np.ndarray,
    specimen_density: float,
    bar_length: float,
    n_pad: int,
    time_step: float,
    n_steps: int,
    traction_amplitude: float,
    gaussian_width: float,
    gaussian_centre: float,
    traction_axis: np.ndarray,
    ceiling: float,
    search_fraction: float,
    residual_tolerance: float,
    iteration_ceiling: int,
) -> dict:
    """Reference implementation chaining every earlier stage."""
    cap_frequency = float(ceiling)
    if not np.isfinite(cap_frequency) or cap_frequency <= 0.0:
        raise ValueError("ceiling wants a finite value above zero")
    newmark_beta, newmark_gamma = _newmark_pair()
    harmonic_padding, marching_padding = _padding_factors()

    material = _oracle_reduce_oriented_grains(  # noqa: F821
        euler_angles, crystal_constants, grain_voxels, specimen_density, bar_length, 3
    )
    axial, lateral = material["longitudinal_modulus"], material["shear_modulus"]
    harmonic_cell = _oracle_assemble_face_stiffness_cell(  # noqa: F821
        grain_voxels, axial, lateral, specimen_density, n_pad, harmonic_padding[0], harmonic_padding[1],
    )
    n_specimen = harmonic_cell["n_specimen"]
    edge = float(bar_length) / n_specimen

    # the baseline is established before the bar is struck, because the graded quantity is a
    # departure from it rather than a reading in its own right
    wanted = min(n_specimen - 1, 24)
    baseline = _oracle_solve_padded_eigenproblem(  # noqa: F821
        harmonic_cell["face_modulus"], harmonic_cell["density"], edge, n_specimen, wanted
    )
    targets = baseline["longitudinal"][baseline["longitudinal"] < cap_frequency]
    if targets.size < 1:
        raise ValueError("no axial resonance of the specimen falls below the ceiling")
    both = np.concatenate([baseline["longitudinal"], baseline["transverse"]])

    marching_cell = _oracle_assemble_face_stiffness_cell(  # noqa: F821
        grain_voxels, axial, lateral, specimen_density, n_pad, marching_padding[0], marching_padding[1],
    )
    march = _oracle_march_pulsed_bar(  # noqa: F821
        marching_cell["face_modulus"], marching_cell["density"], edge, time_step, n_steps,
        newmark_beta, newmark_gamma, traction_amplitude, gaussian_width, gaussian_centre,
        traction_axis, 0, n_specimen - 1, residual_tolerance, iteration_ceiling,
    )
    lines = _oracle_fit_subbin_resonance_lines(  # noqa: F821
        march["record"], time_step, targets, search_fraction
    )

    gap = lines["refined_frequency"] - targets
    mean_gap = float(gap.mean())
    relative = mean_gap / float(targets.mean())
    voxel_fraction = 2.0 / n_specimen
    return {
        "mean_gap": mean_gap,
        "mode_count": int(targets.size),
        "resonance_count": int((both < cap_frequency).sum()),
        "eigen_longitudinal": targets,
        "refined_line": lines["refined_frequency"],
        "gap": gap,
        "mean_relative_shift": relative,
        "voxel_mass_fraction": voxel_fraction,
        "face_mass_ratio": -relative / voxel_fraction,
        "first_offset": float(lines["line_offset"][0]),
        "first_curvature": float(lines["line_curvature"][0]),
        "bin_spacing": lines["bin_spacing"],
        "first_transverse": float(baseline["transverse"][0]),
        "analytic_first_longitudinal": float(material["analytic_longitudinal"][0]),
        "young_hom": material["young_hom"],
        "poisson_hom": material["poisson_hom"],
        "longitudinal_speed": material["longitudinal_speed"],
        "mean_cg_iterations": march["mean_iterations"],
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
HEX = {"c11": 162.4e9, "c33": 180.7e9, "c12": 92.0e9,
       "c13": 69.0e9, "c44": 46.7e9}
TRIPLES = np.array([[30.0, 60.0, 45.0], [10.0, 80.0, 5.0]])
SHARE = np.array([20, 20])
AIM = np.array([1.0, 0.0, 0.0])
def verdict(fn, pad=4, ceil=2.0e6):
    try:
        fn(TRIPLES, HEX, SHARE, 4506.3, 5.0e-3, pad, 4.0e-9, 200, 1.0e6, 2.0e-8, 8.0e-8, AIM,
           ceil, 0.05, 1e-10, 500)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": ("(verdict(report_boundary_inertia_gap), "
                     "verdict(report_boundary_inertia_gap, pad=5, ceil=-1.0), "
                     "verdict(report_boundary_inertia_gap, pad=5, ceil=1.0e3), "
                     "verdict(report_boundary_inertia_gap, pad=5))"),
            "gold_call": ("(verdict(_oracle_report_boundary_inertia_gap), "
                          "verdict(_oracle_report_boundary_inertia_gap, pad=5, ceil=-1.0), "
                          "verdict(_oracle_report_boundary_inertia_gap, pad=5, ceil=1.0e3), "
                          "verdict(_oracle_report_boundary_inertia_gap, pad=5))"),
        },
        {
            "setup": """import numpy as np
HEX = {"c11": 162.4e9, "c33": 180.7e9, "c12": 92.0e9,
       "c13": 69.0e9, "c44": 46.7e9}
# a single grain, so the bar is uniform and the analytic ladder is exact up to discretisation
TRIPLES = np.array([[30.0, 60.0, 45.0]])
SHARE = np.array([65])
AIM = np.array([1.0, 0.0, 0.0])
def digest(out):
    # every mode must be dragged down by nearly the same relative amount, that amount being twice
    # the ratio of voxel size to bar bar_length, since one voxel of padding mass hangs on each face
    spread = float(np.abs(out["gap"] / out["eigen_longitudinal"] - out["mean_relative_shift"]).max())
    return (out["mode_count"], out["resonance_count"], int(out["mean_gap"] < 0.0),
            int(spread < 0.4 * abs(out["mean_relative_shift"])),
            round(out["mean_gap"], 4), round(out["mean_relative_shift"], 8),
            round(out["voxel_mass_fraction"], 10), round(out["face_mass_ratio"], 6),
            round(out["first_offset"], 8), round(out["first_curvature"], 8),
            round(out["bin_spacing"], 6), round(out["first_transverse"], 4),
            round(out["analytic_first_longitudinal"], 4), round(out["young_hom"] / 1e9, 6),
            round(out["poisson_hom"], 8), int(out["mean_cg_iterations"] < 30))
""",
            "call": ("digest(report_boundary_inertia_gap(TRIPLES, HEX, SHARE, 4506.3, 5.0e-3, 6, 4.0e-9, 6000, "
                     "1.0e6, 2.0e-8, 8.0e-8, AIM, 3.0e6, 0.05, 1e-10, 500))"),
            "gold_call": ("digest(_oracle_report_boundary_inertia_gap(TRIPLES, HEX, SHARE, 4506.3, 5.0e-3, 6, 4.0e-9, 6000, "
                          "1.0e6, 2.0e-8, 8.0e-8, AIM, 3.0e6, 0.05, 1e-10, 500))"),
        },
        {
            "setup": """import numpy as np
HEX = {"c11": 162.4e9, "c33": 180.7e9, "c12": 92.0e9,
       "c13": 69.0e9, "c44": 46.7e9}
TRIPLES = np.array([[38.5, 37.9, 1.9], [311.8, 106.0, 249.3], [50.3, 112.1, 287.9],
                    [158.3, 40.3, 350.5], [211.3, 150.3, 54.2], [114.9, 42.8, 90.7]])
SHARE = np.array([11, 9, 12, 7, 13, 13])
AIM = np.array([1.0, 0.0, 0.0])
def digest(out):
    return (out["mode_count"], out["resonance_count"],
            round(out["mean_gap"], 4), round(out["mean_relative_shift"], 8),
            round(out["face_mass_ratio"], 6), tuple(np.round(out["gap"], 3)),
            tuple(np.round(out["refined_line"], 3)), round(out["first_curvature"], 6),
            round(out["first_transverse"], 4), round(out["longitudinal_speed"], 5))
""",
            "call": ("digest(report_boundary_inertia_gap(TRIPLES, HEX, SHARE, 4506.3, 2.5e-3, 6, 2.0e-9, 8000, "
                     "1.0e6, 2.0e-8, 8.0e-8, AIM, 5.0e6, 0.05, 1e-10, 500))"),
            "gold_call": ("digest(_oracle_report_boundary_inertia_gap(TRIPLES, HEX, SHARE, 4506.3, 2.5e-3, 6, 2.0e-9, 8000, "
                          "1.0e6, 2.0e-8, 8.0e-8, AIM, 5.0e6, 0.05, 1e-10, 500))"),
        },
    ]
