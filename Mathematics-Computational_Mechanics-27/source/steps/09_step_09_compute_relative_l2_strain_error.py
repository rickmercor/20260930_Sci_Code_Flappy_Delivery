"""
This step is the orchestrator. It runs the whole chain in sequence: step 1 supplies the inclusion radius from the neutrality condition, step 2 the exact shell coefficients, step 3 the periodic three-phase mesh and its level set, steps 4 and 5 the interface quadrature and the modified enrichment that step 6 uses to assemble the internally scaled element operators, step 7 the Fourier multiplier of the preconditioner and step 8 the conjugate-gradient solution of the equilibrium system. It then compares the discrete and exact strain fields at every quadrature point of the assembled rule and forms the relative error.

It returns the error together with the intermediate quantities a reader needs to judge it: the simulated effective bulk modulus, which neutrality requires to return the matrix value, the derived inclusion radius, the shell coefficients and the residual of the interface condition that was not used to determine them, the mesh and quadrature counts, the quadrature-weighted volume of each phase and the iteration count of the solve.

The discrete strain at a quadrature point is $\epsilon_h = \overline{\epsilon} + B_e u_e$ with the scaled strain operator of the owning element. The exact strain is evaluated from the shell law of the phase that the quadrature point already carries for the stiffness, that is the phase implied by the linearised interface, and not from the phase implied by the true spherical radius. The two differ only on slivers of thickness $\mathcal{O}(h^2)$ between the sphere and its chordal reconstruction, but the strain jumps by $\mathcal{O}(1)$ across a material interface, so evaluating the exact field by true radius would make the functional depend on whether individual quadrature points happen to fall inside those slivers. Using the phase already assigned keeps the integrand smooth on every sub-tetrahedron and makes the functional stable.

Both norms use the same points and weights, $\| a \|_{L^2} = [\sum_q w_q a(q) \cdot a(q)]^{1/2}$ with the sum running over the quadrature points of the assembled rule, and $e = \| \epsilon_* - \epsilon_h \|_{L^2} / \| \epsilon_* \|_{L^2}$, so any common volume normalisation cancels.

Returns
-------
dict, the relative L2 strain error with the effective modulus and the supporting counts.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_relative_l2_strain_error(
    n_voxels: int,
    cell_size: float,
    centre: tuple,
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_coating: float,
    macroscopic_strain,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Run the enriched solver and return the relative local strain error.

    Parameters
    ----------
    n_voxels : int
        Number of voxels along each cell edge.
    cell_size : float
        Edge length of the cubic periodic cell.
    centre : tuple
        Centre of the coated sphere.
    bulk_matrix : float
        Bulk modulus of the matrix.
    bulk_coating : float
        Bulk modulus of the coating.
    bulk_inclusion : float
        Bulk modulus of the inclusion.
    poisson_ratio : float
        Poisson ratio shared by all three phases.
    radius_coating : float
        Outer radius of the coating.
    macroscopic_strain : array_like
        Prescribed macroscopic strain of shape (6,).
    tolerance : float
        Relative tolerance of the stopping test.
    max_iterations : int
        Maximum number of conjugate-gradient iterations.

    Returns
    -------
    dict
        Keys relative_error, effective_bulk_modulus, radius_inclusion, coefficients, traction_residual, n_cut, n_enriched, n_quadrature, phase_volumes, iterations, error_norm_squared and exact_norm_squared.

    Raises
    ------
    ValueError
        If macroscopic_strain does not have shape (6,), or if n_voxels is below two or cell_size is not strictly positive.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

R2 = np.sqrt(2.0)
MATRIX, COATING, INCLUSION = 0, 1, 2

def _isotropic_stiffness(bulk, poisson_ratio):
    """Mandel stiffness matrix of an isotropic phase given its bulk modulus."""
    lame = 3.0 * bulk * poisson_ratio / (1.0 + poisson_ratio)
    shear = 3.0 * bulk * (1.0 - 2.0 * poisson_ratio) / (2.0 * (1.0 + poisson_ratio))
    stiffness = lame * np.ones((6, 6))
    stiffness[3:, :] = 0.0
    stiffness[:, 3:] = 0.0
    return stiffness + 2.0 * shear * np.eye(6), shear

def _mandel(tensor):
    """Orthonormal Mandel vector of a symmetric second-order tensor field."""
    return np.stack([tensor[..., 0, 0], tensor[..., 1, 1], tensor[..., 2, 2],
                     R2 * tensor[..., 1, 2], R2 * tensor[..., 0, 2],
                     R2 * tensor[..., 0, 1]], axis=-1)


def _exact_strain_by_phase(points, centre, coefficients, phase):
    """Exact Mandel strain at points, each evaluated with the shell law of its own phase."""
    coefficient_a, coefficient_b, coefficient_c = coefficients
    offset = np.asarray(points, dtype=float) - np.asarray(centre, dtype=float)
    radius = np.linalg.norm(offset, axis=-1)
    normal = offset / np.maximum(radius, 1e-300)[:, None]
    dyad = normal[:, :, None] * normal[:, None, :]
    identity = np.eye(3)
    tensor = np.zeros((offset.shape[0], 3, 3))
    tensor[phase == MATRIX] = identity
    coating = phase == COATING
    tensor[coating] = (coefficient_a * identity
                       + (coefficient_b / radius[coating] ** 3)[:, None, None]
                       * (identity - 3.0 * dyad[coating]))
    tensor[phase == INCLUSION] = coefficient_c * identity
    return _mandel(tensor)


def _oracle_compute_relative_l2_strain_error(
    n_voxels: int,
    cell_size: float,
    centre: tuple,
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_coating: float,
    macroscopic_strain,
    tolerance: float,
    max_iterations: int,
) -> dict:
    """Reference implementation, chaining the earlier steps in sequence."""
    n_voxels = int(n_voxels)
    cell_size = float(cell_size)
    macroscopic_strain = np.asarray(macroscopic_strain, dtype=float)
    if macroscopic_strain.shape != (6,):
        raise ValueError("macroscopic_strain must have shape (6,)")
    if n_voxels < 2 or cell_size <= 0.0:
        raise ValueError("n_voxels must be at least two and cell_size strictly positive")

    neutral = _oracle_derive_neutral_inclusion_radius(
        bulk_matrix, bulk_coating, bulk_inclusion, poisson_ratio, radius_coating)
    radius_inclusion = neutral["radius_inclusion"]
    field = _oracle_solve_coated_sphere_exact_field(
        bulk_matrix, bulk_coating, bulk_inclusion, poisson_ratio,
        radius_inclusion, radius_coating)
    coefficients = (field["coefficient_a"], field["coefficient_b"], field["coefficient_c"])

    mesh = _oracle_build_three_phase_periodic_mesh(
        n_voxels, cell_size, centre, radius_inclusion, radius_coating)
    stiffness = np.array([_isotropic_stiffness(bulk, poisson_ratio)[0]
                          for bulk in (bulk_matrix, bulk_coating, bulk_inclusion)])
    operators = _oracle_assemble_scaled_three_phase_system(mesh, stiffness)
    green = _oracle_build_fourier_green_operator(n_voxels, cell_size)
    solution = _oracle_solve_scaled_xfft_system(
        operators, green, macroscopic_strain, cell_size ** 3,
        float(tolerance), int(max_iterations))
    if not solution["converged"]:
        raise RuntimeError("the conjugate-gradient iteration did not reach the tolerance")

    padded = np.concatenate([solution["displacement"], [0.0]])
    local = padded[operators["element_dofs"]][operators["quadrature_element"]]
    discrete = macroscopic_strain[None, :] + np.einsum(
        "qij,qj->qi", operators["strain_operator"], local)
    exact = _exact_strain_by_phase(operators["quadrature_points"], centre, coefficients,
                                   operators["quadrature_phase"])
    weights = operators["quadrature_weights"]
    error_norm = float(np.sum(weights * np.sum((discrete - exact) ** 2, axis=1)))
    exact_norm = float(np.sum(weights * np.sum(exact ** 2, axis=1)))
    phase = operators["quadrature_phase"]
    return {
        "relative_error": float(np.sqrt(error_norm / exact_norm)),
        "effective_bulk_modulus": float(np.sum(solution["effective_stress"][:3]) / 9.0),
        "radius_inclusion": float(radius_inclusion),
        "coefficients": tuple(float(value) for value in coefficients),
        "traction_residual": float(field["traction_residual"]),
        "n_cut": int(mesh["is_cut"].sum()),
        "n_enriched": int(mesh["n_enriched"]),
        "n_quadrature": int(operators["n_quadrature"]),
        "phase_volumes": tuple(float(weights[phase == label].sum())
                               for label in (MATRIX, COATING, INCLUSION)),
        "iterations": int(solution["iterations"]),
        "error_norm_squared": error_norm,
        "exact_norm_squared": exact_norm,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
STRAIN = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
def summarize(data):
    return (round(data["relative_error"], 10),
            round(data["effective_bulk_modulus"], 12),
            round(data["radius_inclusion"], 11),
            data["iterations"], data["n_quadrature"])
""",
            "call": "summarize(compute_relative_l2_strain_error(8, 16.0, (8.0, 8.0, 8.0), 1.0, 0.808024, 8.080240, 0.25, 2*np.pi, STRAIN, 1e-7, 200))",
            "gold_call": "summarize(_oracle_compute_relative_l2_strain_error(8, 16.0, (8.0, 8.0, 8.0), 1.0, 0.808024, 8.080240, 0.25, 2*np.pi, STRAIN, 1e-7, 200))",
        },
        {
            "setup": """import numpy as np
STRAIN = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
def invariance(fn):
    # the relative error and the normalised effective modulus are invariant under a uniform
    # rescaling of the cell together with the radii
    a = fn(8, 16.0, (8.0, 8.0, 8.0), 1.0, 0.808024, 8.080240, 0.25, 2*np.pi, STRAIN, 1e-8, 200)
    b = fn(8, 8.0, (4.0, 4.0, 4.0), 1.0, 0.808024, 8.080240, 0.25, np.pi, STRAIN, 1e-8, 200)
    return (round(abs(a["relative_error"] - b["relative_error"]), 11),
            round(abs(a["effective_bulk_modulus"] - b["effective_bulk_modulus"]), 11),
            round(abs(2.0*b["radius_inclusion"] - a["radius_inclusion"]), 11))
""",
            "call": "invariance(compute_relative_l2_strain_error)",
            "gold_call": "invariance(_oracle_compute_relative_l2_strain_error)",
        },
        {
            "setup": """import numpy as np
STRAIN = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
def run(fn):
    codes = []
    for args in [(8, 16.0, (8.0, 8.0, 8.0), 1.0, 0.808024, 8.080240, 0.25, 2*np.pi, np.zeros(5), 1e-7, 200),
                 (1, 16.0, (8.0, 8.0, 8.0), 1.0, 0.808024, 8.080240, 0.25, 2*np.pi, STRAIN, 1e-7, 200)]:
        try:
            fn(*args)
            codes.append(0)
        except ValueError:
            codes.append(1)
        except Exception:
            codes.append(2)
    return tuple(codes)
""",
            "call": "run(compute_relative_l2_strain_error)",
            "gold_call": "run(_oracle_compute_relative_l2_strain_error)",
        },
    ]
