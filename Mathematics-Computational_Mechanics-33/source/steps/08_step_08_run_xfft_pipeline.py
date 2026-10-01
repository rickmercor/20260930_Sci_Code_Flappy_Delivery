"""
Complete deterministic X-FFT homogenization pipeline.




The orchestrator builds the periodic spherical interface, resolves every cut tetrahedron by subcell quadrature, evaluates the modified absolute enrichment, applies componentwise internal scaling during element assembly, constructs the Fourier Green operator, solves the scaled system by matrix-free preconditioned linear CG, and extracts the effective axial stress. This is the final step in Studio.




Inputs

------

n_voxels : int

radius : float

center : array-like of shape (3,)

matrix_lame : array-like of shape (2,)

inclusion_lame : array-like of shape (2,)

tol : float

max_iter : int




Returns

-------

result : dict

    Keys mesh, quadrature, first_cut_element, first_cut_enrichment, elements, green, solution, effective, and final_answer, with nested schemas specified below.

Returns
-------
dict with mesh from build_periodic_sphere_mesh; quadrature containing integer arrays element_ptr (e + 1,) and phases (q,) plus float64 arrays barycentric (q, 4) and weights (q,); native int first_cut_element; first_cut_enrichment from evaluate_modified_abs_enrichment; elements from scale_and_assemble_elements; complex128 array green (n_voxels, n_voxels, n_voxels, 3, 3); solution from solve_scaled_xfft_system; effective from compute_effective_axial_stress; and native float final_answer, where e and q count elements and total quadrature points.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_xfft_pipeline(
    n_voxels: int = 3,
    radius: float = 0.31,
    center: tuple = (0.43, 0.37, 0.52),
    matrix_lame: tuple = (3.0, 2.0),
    inclusion_lame: tuple = (30.0, 20.0),
    tol: float = 1e-10,
    max_iter: int = 500,
) -> dict:
    """Run the full X-FFT toy homogenization and return its numeric state.

    Parameters
    ----------
    n_voxels : int
        Number of periodic voxels per cell edge.
    radius : float
        Radius of the periodic spherical inclusion.
    center : tuple
        Three coordinates of the inclusion center.
    matrix_lame : tuple
        Lamé pair (lambda, mu) of the matrix.
    inclusion_lame : tuple
        Lamé pair (lambda, mu) of the inclusion.
    tol : float
        Relative preconditioned residual tolerance.
    max_iter : int
        Maximum number of linear CG iterations.

    Returns
    -------
    result : dict
        Keys mesh, quadrature, first_cut_element, first_cut_enrichment, elements, green, solution, effective, and final_answer, with nested schemas specified below.
    Raises
    ------
    ValueError
        Propagated from the earlier steps for invalid inputs, in particular
        n_voxels below two, radius outside (0, 0.5), or a center on or outside
        the open unit cell.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_run_xfft_pipeline(
    n_voxels: int = 3,
    radius: float = 0.31,
    center: tuple = (0.43, 0.37, 0.52),
    matrix_lame: tuple = (3.0, 2.0),
    inclusion_lame: tuple = (30.0, 20.0),
    tol: float = 1e-10,
    max_iter: int = 500,
) -> dict:
    """Reference implementation chaining every earlier step."""
    center_array = np.asarray(center, dtype=float)
    matrix_lame_array = np.asarray(matrix_lame, dtype=float)
    inclusion_lame_array = np.asarray(inclusion_lame, dtype=float)
    macrostrain = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mesh = _oracle_build_periodic_sphere_mesh(  # noqa: F821
        n_voxels, radius, center_array
    )

    pointer = [0]
    barycentric = []
    weights = []
    phases = []
    for vertices, levels in zip(mesh["vertices"], mesh["levels"]):
        local = _oracle_construct_subcell_quadrature(vertices, levels)  # noqa: F821
        barycentric.extend(local["barycentric"])
        weights.extend(local["weights"])
        phases.extend(local["phases"])
        pointer.append(len(weights))
    quadrature = {
        "element_ptr": np.asarray(pointer, dtype=int),
        "barycentric": np.asarray(barycentric, dtype=float),
        "weights": np.asarray(weights, dtype=float),
        "phases": np.asarray(phases, dtype=int),
    }

    first_cut = int(np.flatnonzero(mesh["cut"])[0])
    start, stop = quadrature["element_ptr"][first_cut : first_cut + 2]
    first_vertices = mesh["vertices"][first_cut]
    first_interpolation = np.column_stack([np.ones(4), first_vertices])
    first_gradients = np.linalg.inv(first_interpolation)[1:, :].T
    enrichment = _oracle_evaluate_modified_abs_enrichment(  # noqa: F821
        quadrature["barycentric"][start:stop],
        mesh["levels"][first_cut],
        first_gradients,
    )
    elements = _oracle_scale_and_assemble_elements(  # noqa: F821
        mesh,
        quadrature,
        matrix_lame_array,
        inclusion_lame_array,
        macrostrain,
    )
    green = _oracle_build_fourier_green_operator(n_voxels)  # noqa: F821
    solution = _oracle_solve_scaled_xfft_system(  # noqa: F821
        elements, green, tol, max_iter
    )
    effective = _oracle_compute_effective_axial_stress(  # noqa: F821
        elements["element_stress"],
        elements["element_dofs"],
        elements["stress_macro"],
        solution["displacement"],
    )
    return {
        "mesh": mesh,
        "quadrature": quadrature,
        "first_cut_element": first_cut,
        "first_cut_enrichment": enrichment,
        "elements": elements,
        "green": green,
        "solution": solution,
        "effective": effective,
        "final_answer": float(effective["effective_axial_stiffness"]),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of integration test specifications for the whole pipeline."""
    return [
        {
            "setup": """import numpy as np
def summarize(result):
    return (
        len(result["mesh"]["vertices"]),
        int(np.sum(result["mesh"]["cut"])),
        len(result["quadrature"]["weights"]),
        round(float(np.sum(result["quadrature"]["weights"][result["quadrature"]["phases"] < 0])), 14),
        round(float(np.min(result["elements"]["scaling_energy"])), 14),
        round(float(np.max(result["elements"]["scaling_energy"])), 14),
        result["solution"]["iterations"],
        result["solution"]["converged"],
        np.round(result["effective"]["effective_stress"], 12).tolist(),
        round(result["final_answer"], 12),
    )
""",
            "call": "summarize(run_xfft_pipeline())",
            "gold_call": "summarize(_oracle_run_xfft_pipeline())",
        },
        {
            "setup": """import numpy as np
def summarize(result):
    return (
        result["solution"]["iterations"],
        np.round(result["effective"]["effective_stress"], 12).tolist(),
        round(result["final_answer"], 12),
    )
""",
            "call": "summarize(run_xfft_pipeline(inclusion_lame=(12.0, 8.0), tol=1e-9))",
            "gold_call": "summarize(_oracle_run_xfft_pipeline(inclusion_lame=(12.0, 8.0), tol=1e-9))",
        },
        {
            "setup": """def run_model():
    try:
        run_xfft_pipeline(n_voxels=1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_run_xfft_pipeline(n_voxels=1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
