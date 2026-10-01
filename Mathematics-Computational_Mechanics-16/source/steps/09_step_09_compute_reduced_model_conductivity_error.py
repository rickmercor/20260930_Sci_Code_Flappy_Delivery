"""
This final step runs the whole spectral reduced-model workflow from the raw configuration and reports the relative error of the reduced effective conductivity together with every check quantity. The chain is: step 1 samples the periodic disc microstructure on the odd grid and returns the phase map and the conductivity; step 2 provides the Fourier-discretised cell operator, applied inside step 3 to an orthonormal mean-free basis to produce the matrices of the cell operator and of the reference operator, the reference conductivity being the mean of the smallest and largest phase conductivities; step 4 solves the generalised eigenvalue problem of the preconditioned operator with energetic normalisation, returning the ascending spectrum, its bounds, the accumulation values and the counts on and off them; step 5 expands the load on the eigenstates, yielding the components and the spectral coupling vectors; step 6 reconstructs the exact discrete solution, verifies equilibrium and forms the effective conductivity by the flux average and by the spectral sum; step 7 retains the eigenstates of largest component magnitude until the energetic-norm error of the truncated fluctuation drops below the threshold and evaluates the reduced flux; step 8 merges every inclusion phase into one, solves the geometric eigenvalue problem of the resulting two-phase cell and evaluates its contrast-independent representation at the requested contrast against a direct solution. The reported final quantity is the relative error of the first component of the reduced effective flux against the full one for the given loading.

Returns
-------
dict with native float relative_error, tuple of native ints pixel_counts, native float reference_conductivity, tuple of native floats bounds, tuple of native floats plateau_values, tuple of native ints plateau_counts, native int transition_count, float array effective_tensor of shape (2, 2), native floats dominant_magnitude and dominant_eigenvalue, native int n_retained, native floats energy_error and reduced_conductivity, tuple of native ints two_phase_limit_counts, native floats two_phase_max_projection and two_phase_conductivity, and native floats identity_residual, equilibrium_residual and two_phase_residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_reduced_model_conductivity_error(
    n_pixels: int,
    cell_lengths: tuple,
    discs: list,
    phase_conductivities: np.ndarray,
    macroscopic_gradient: np.ndarray,
    plateau_tolerance: float,
    energy_error_threshold: float,
    two_phase_contrast: float,
) -> dict:
    """Run the spectral reduced-model workflow end to end and report its relative conductivity error.

    Raises
    ------
    ValueError
        If n_pixels is not an odd integer of at least three, if a cell length or a phase conductivity is not strictly positive, if fewer than two phases are given, if macroscopic_gradient does not have two components, if plateau_tolerance is not strictly positive, if energy_error_threshold is not strictly between zero and one, or if two_phase_contrast is not strictly positive or equals one.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_reduced_model_conductivity_error(
    n_pixels: int,
    cell_lengths: tuple,
    discs: list,
    phase_conductivities: np.ndarray,
    macroscopic_gradient: np.ndarray,
    plateau_tolerance: float,
    energy_error_threshold: float,
    two_phase_contrast: float,
) -> dict:
    """Reference implementation chaining every earlier step."""
    if not isinstance(n_pixels, (int, np.integer)) or int(n_pixels) < 3 or int(n_pixels) % 2 == 0:
        raise ValueError("n_pixels must be an odd integer of at least three")
    n_pixels = int(n_pixels)
    lengths = tuple(float(v) for v in cell_lengths)
    if len(lengths) != 2 or min(lengths) <= 0.0:
        raise ValueError("cell_lengths must be two strictly positive values")
    conductivities = np.asarray(phase_conductivities, dtype=float).ravel()
    if conductivities.size < 2 or np.any(conductivities <= 0.0):
        raise ValueError("at least two strictly positive phase conductivities are required")
    gradient = np.asarray(macroscopic_gradient, dtype=float).ravel()
    if gradient.shape != (2,):
        raise ValueError("macroscopic_gradient must have two components")
    plateau_tolerance = float(plateau_tolerance)
    if plateau_tolerance <= 0.0:
        raise ValueError("plateau_tolerance must be strictly positive")
    threshold = float(energy_error_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("energy_error_threshold must lie strictly between zero and one")
    contrast = float(two_phase_contrast)
    if contrast <= 0.0 or contrast == 1.0:
        raise ValueError("two_phase_contrast must be strictly positive and different from one")
    reference_conductivity = 0.5 * (float(conductivities.min()) + float(conductivities.max()))

    microstructure = _oracle_rasterise_periodic_disc_microstructure(  # noqa: F821
        n_pixels, lengths, discs, conductivities
    )
    conductivity = microstructure["conductivity"]
    matrices = _oracle_assemble_mean_free_operator_matrices(  # noqa: F821
        conductivity, lengths, reference_conductivity
    )
    spectrum = _oracle_solve_preconditioned_cell_operator_spectrum(  # noqa: F821
        matrices["operator"], matrices["reference_operator"], conductivities,
        reference_conductivity, plateau_tolerance,
    )
    projection = _oracle_project_cell_load_onto_eigenstates(  # noqa: F821
        matrices["basis"], spectrum["eigenvalues"], spectrum["eigenvectors"],
        conductivity, lengths, gradient,
    )
    solution = _oracle_reconstruct_solution_and_effective_tensor(  # noqa: F821
        matrices["basis"], spectrum["eigenvalues"], spectrum["eigenvectors"],
        projection["components"], projection["spectral_vectors"], conductivity, lengths, gradient,
    )
    reduced = _oracle_build_truncated_reduced_model(  # noqa: F821
        spectrum["eigenvalues"], projection["components"], projection["spectral_vectors"],
        solution["mean_conductivity"], solution["effective_column"], gradient, threshold,
    )
    two_phase = _oracle_compute_contrast_independent_representation(  # noqa: F821
        microstructure["phase"], lengths, 1, float(conductivities[0]), contrast, gradient, plateau_tolerance,
    )
    return {
        "relative_error": float(reduced["relative_errors"][0]),
        "pixel_counts": microstructure["pixel_counts"],
        "reference_conductivity": float(reference_conductivity),
        "bounds": spectrum["bounds"],
        "plateau_values": spectrum["plateau_values"],
        "plateau_counts": spectrum["plateau_counts"],
        "transition_count": spectrum["transition_count"],
        "effective_tensor": solution["effective_tensor"],
        "dominant_magnitude": projection["dominant_magnitude"],
        "dominant_eigenvalue": projection["dominant_eigenvalue"],
        "n_retained": reduced["n_retained"],
        "energy_error": reduced["energy_error"],
        "reduced_conductivity": float(reduced["truncated_column"][0]),
        "two_phase_limit_counts": two_phase["limit_counts"],
        "two_phase_max_projection": two_phase["max_projection"],
        "two_phase_conductivity": float(two_phase["effective_column"][0]),
        "identity_residual": projection["identity_residual"],
        "equilibrium_residual": solution["equilibrium_residual"],
        "two_phase_residual": two_phase["consistency_residual"],
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
DISCS = [(0.27, 0.31, 0.24, 3), (0.76, 0.72, 0.21, 3), (0.78, 0.22, 0.155, 2), (0.24, 0.79, 0.135, 2)]
def summarize(out):
    return (
        round(float(out["relative_error"]), 10), tuple(int(v) for v in out["pixel_counts"]),
        round(float(out["reference_conductivity"]), 12),
        tuple(int(v) for v in out["plateau_counts"]), int(out["transition_count"]),
        round(float(out["effective_tensor"][0, 0]), 9), round(float(out["dominant_magnitude"]), 9),
        int(out["n_retained"]), round(float(out["energy_error"]), 9),
        round(float(out["reduced_conductivity"]), 9),
        tuple(int(v) for v in out["two_phase_limit_counts"]),
        round(float(out["two_phase_conductivity"]), 9),
        int(out["identity_residual"] < 1e-10), int(out["equilibrium_residual"] < 1e-9), int(out["two_phase_residual"] < 1e-9),
    )
""",
            "call": "summarize(compute_reduced_model_conductivity_error(21, (1.0, 1.0), DISCS, np.array([1.0, 3.0, 10.0]), np.array([1.0, 0.0]), 1e-4, 0.1, 10.0))",
            "gold_call": "summarize(_oracle_compute_reduced_model_conductivity_error(21, (1.0, 1.0), DISCS, np.array([1.0, 3.0, 10.0]), np.array([1.0, 0.0]), 1e-4, 0.1, 10.0))",
        },
        {
            "setup": """import numpy as np
DISCS = [(0.37, 0.53, 0.29, 2), (0.83, 1.17, 0.21, 2)]
def summarize(out):
    return (
        round(float(out["relative_error"]), 10), tuple(int(v) for v in out["pixel_counts"]),
        tuple(round(float(v), 9) for v in out["bounds"]),
        tuple(int(v) for v in out["plateau_counts"]), int(out["transition_count"]),
        round(float(out["effective_tensor"][0, 0]), 9), round(float(out["dominant_eigenvalue"]), 9),
        int(out["n_retained"]), round(float(out["reduced_conductivity"]), 9),
        round(float(out["two_phase_max_projection"]), 9), round(float(out["two_phase_conductivity"]), 9),
        int(abs(out["two_phase_conductivity"] - out["effective_tensor"][0, 0]) < 1e-9),
    )
""",
            "call": "summarize(compute_reduced_model_conductivity_error(15, (1.0, 1.4), DISCS, np.array([1.0, 6.0]), np.array([1.0, 0.0]), 1e-3, 0.3, 6.0))",
            "gold_call": "summarize(_oracle_compute_reduced_model_conductivity_error(15, (1.0, 1.4), DISCS, np.array([1.0, 6.0]), np.array([1.0, 0.0]), 1e-3, 0.3, 6.0))",
        },
        {
            "setup": """import numpy as np
def run(fn):
    try:
        fn(15, (1.0, 1.0), [(0.5, 0.5, 0.3, 2)], np.array([1.0, 6.0]), np.array([1.0, 0.0]), 1e-4, 1.0, 6.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(compute_reduced_model_conductivity_error)",
            "gold_call": "run(_oracle_compute_reduced_model_conductivity_error)",
        },
    ]
