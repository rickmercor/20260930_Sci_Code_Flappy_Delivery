"""
Chain the sub-problem functions 01 to 09 end-to-end on the distorted PN junction and return the terminal current at the requested contact. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (build_ddfv_mesh, build_ddfv_node_table, evaluate_bernoulli, build_junction_state, assemble_ddfv_laplacian, assemble_harmonic_flux_matrix, assemble_coupled_residual, solve_drift_diffusion, compute_terminal_current) rather than reimplementing them.

This step runs the whole measurement end to end. It (i) builds the diamond geometry of the deliberately distorted triangulation with sub-problem 01 and the node table carrying the control volume measures, the equation classes and the contact tags with sub-problem 02, (ii) drives the coupled nonlinear solve with sub-problem 08, which calls the doping and Ohmic contact values of sub-problem 04, the discrete duality Laplacian of sub-problem 05, the harmonic-average flux matrices of sub-problem 06, themselves built on the Bernoulli function of sub-problem 03, and the coupled residual of sub-problem 07 at every Newton iteration, and (iii) sums the converged fluxes over the requested contact with sub-problem 09.




The scalar returned answers a question that neither the classical exponentially fitted method nor the plain discrete duality method can answer on its own. The classical finite volume Scharfetter-Gummel scheme is unavailable on this mesh, because the interior vertices have been displaced until the triangulation is no longer Delaunay and the Voronoi dual it depends on no longer exists in usable form. The plain discrete duality scheme is available but centred, and on a junction whose potential falls by more than twenty thermal voltages across a fraction of the domain a centred flux oscillates.




Two comparisons make the number interpretable and both are cheap to run through the same entry point. Setting the scalar product of the two unit normals to zero in the diamond table removes the non-orthogonal coupling everywhere, including the Poisson operator and both carrier-flux operators, so the change in terminal current measures the global effect of that coupling rather than the carrier-flux cross term alone. Running the same calculation at zero distortion answers a regular mesh, so the change between the two measures how far the computed current drifts when the mesh is ruined. A large first number together with a small second one is the outcome that justifies the extra unknowns; either one alone would be inconclusive.

Returns
-------
float, the terminal current at the requested contact in scaled units, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def run_ddfv_ha_pipeline(nx: int = 6, ny: int = 12, distortion: float = 0.40,
                         anode_voltage: float = 0.4, n_steps: int = 4,
                         tolerance: float = 1.0e-10, max_iterations: int = 50,
                         contact_tag: int = 1, drop_cross_term: bool = False) -> float:
    """Run the full harmonic-average discrete duality measurement.

    Parameters
    ----------
    nx : int
        Number of rectangle columns across the unit square (nx >= 2).
    ny : int
        Number of rectangle rows up the unit square (ny >= 2).
    distortion : float
        Interior-vertex displacement as a fraction of the rectangle side
        lengths, 0 <= distortion < 0.5.
    anode_voltage : float
        Voltage applied at the anode contact, in volts.
    n_steps : int
        Number of equal voltage continuation steps (n_steps >= 1).
    tolerance : float
        Convergence threshold on the maximum absolute scaled residual
        (tolerance > 0).
    max_iterations : int
        Maximum Newton iterations per continuation step (max_iterations >= 1).
    contact_tag : int
        Contact to report: 1 for the grounded cathode, 2 for the anode.
    drop_cross_term : bool
        When True the scalar product of the two diamond normals is zeroed
        before solving, which removes the non-orthogonal coupling from the
        Poisson operator and both carrier-flux operators.

    Returns
    -------
    current : float
        Terminal current at the requested contact in scaled units, as a
        native Python float, positive when conventional current leaves the
        device through that contact.

    Raises
    ------
    ValueError
        If an integer or scalar bound is invalid, contact_tag is not 1 or 2,
        or drop_cross_term is not boolean.
    """
    return current  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_ddfv_ha_pipeline(
    nx: int = 6,
    ny: int = 12,
    distortion: float = 0.40,
    anode_voltage: float = 0.4,
    n_steps: int = 4,
    tolerance: float = 1.0e-10,
    max_iterations: int = 50,
    contact_tag: int = 1,
    drop_cross_term: bool = False,
) -> float:
    import numpy as np

    for name, value, floor in (
        ("nx", nx, 2),
        ("ny", ny, 2),
        ("n_steps", n_steps, 1),
        ("max_iterations", max_iterations, 1),
    ):
        if not (
            isinstance(value, (int, np.integer))
            and not isinstance(value, bool)
            and int(value) >= floor
        ):
            raise ValueError(
                f"{name} must be an integer >= {floor}"
            )

    if not (
        isinstance(distortion, (int, float))
        and np.isfinite(distortion)
        and 0.0 <= float(distortion) < 0.5
    ):
        raise ValueError(
            "distortion must be a finite number in [0, 0.5)"
        )

    if not (
        isinstance(anode_voltage, (int, float))
        and np.isfinite(anode_voltage)
    ):
        raise ValueError(
            "anode_voltage must be a finite number"
        )

    if not (
        isinstance(tolerance, (int, float))
        and np.isfinite(tolerance)
        and float(tolerance) > 0.0
    ):
        raise ValueError(
            "tolerance must be a finite number > 0"
        )

    if not (
        isinstance(contact_tag, (int, np.integer))
        and not isinstance(contact_tag, bool)
        and int(contact_tag) in (1, 2)
    ):
        raise ValueError(
            "contact_tag must be the integer 1 or 2"
        )

    if not isinstance(
        drop_cross_term,
        (bool, np.bool_),
    ):
        raise ValueError(
            "drop_cross_term must be a boolean"
        )

    # Earlier steps are available in Studio's concatenated namespace.
    build_mesh = _oracle_build_ddfv_mesh
    build_nodes = _oracle_build_ddfv_node_table
    _evaluate_bernoulli = _oracle_evaluate_bernoulli
    junction_state = _oracle_build_junction_state
    assemble_laplacian = _oracle_assemble_ddfv_laplacian
    assemble_flux = _oracle_assemble_harmonic_flux_matrix
    assemble_residual = _oracle_assemble_coupled_residual
    solve_system = _oracle_solve_drift_diffusion
    terminal_current = _oracle_compute_terminal_current

    # Steps 01 and 02: construct the distorted mesh.
    diamonds = build_mesh(
        int(nx),
        int(ny),
        float(distortion),
    )

    nodes = build_nodes(
        int(nx),
        int(ny),
        float(distortion),
    )

    if bool(drop_cross_term):
        diamonds = np.array(
            diamonds,
            dtype=float,
            copy=True,
        )
        diamonds[:, 7] = 0.0

    # Steps 03 to 07: construct and verify one complete residual.
    bernoulli_probe = _evaluate_bernoulli(
        np.array([
            0.0,
            1.0,
            -1.0,
        ])
    )

    state = junction_state(
        nodes,
        float(anode_voltage),
    )

    potential = state[:, 3]

    laplacian = assemble_laplacian(
        diamonds,
        nodes,
    )

    electron_matrix = assemble_flux(
        diamonds,
        nodes,
        potential,
        1.0,
        False,
    )

    hole_matrix = assemble_flux(
        diamonds,
        nodes,
        potential,
        12.16336 / 36.63227,
        True,
    )

    initial_unknowns = np.column_stack([
        potential,
        state[:, 1],
        state[:, 2],
    ])

    residual = assemble_residual(
        nodes,
        state,
        laplacian,
        electron_matrix,
        hole_matrix,
        initial_unknowns,
    )

    dependency_outputs = (
        bernoulli_probe,
        state,
        laplacian,
        electron_matrix,
        hole_matrix,
        residual,
    )

    if not all(
        np.all(np.isfinite(value))
        for value in dependency_outputs
    ):
        raise ValueError(
            "an earlier pipeline step produced a non-finite output"
        )

    solution = solve_system(
        diamonds,
        nodes,
        float(anode_voltage),
        int(n_steps),
        float(tolerance),
        int(max_iterations),
    )

    return float(
        terminal_current(
            diamonds,
            nodes,
            solution,
            int(contact_tag),
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES (integration tests -- whole pipeline)
# =============================================================================


import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: small distorted mesh, whole pipeline (normal scenario) ---
        {
            "setup": """import numpy as np
nx, ny = 3, 4
""",
            "call": "run_ddfv_ha_pipeline(nx, ny)",
            "gold_call": "_oracle_run_ddfv_ha_pipeline(nx, ny)",
        },
        # --- Integration: the benchmark configuration whose value is the final answer ---
        {
            "setup": """import numpy as np
nx, ny, distortion = 6, 12, 0.40
""",
            "call": "run_ddfv_ha_pipeline(nx, ny, distortion)",
            "gold_call": "_oracle_run_ddfv_ha_pipeline(nx, ny, distortion)",
        },
        # --- Integration: the anode contact, which must mirror the cathode ---
        {
            "setup": """import numpy as np
nx, ny = 3, 4
""",
            "call": "run_ddfv_ha_pipeline(nx, ny, contact_tag=2)",
            "gold_call": "_oracle_run_ddfv_ha_pipeline(nx, ny, contact_tag=2)",
        },
        # --- Integration (boundary): undistorted mesh at zero bias ---
        {
            "setup": """import numpy as np
nx, ny = 2, 2
""",
            "call": "run_ddfv_ha_pipeline(nx, ny, 0.0, 0.0, 1)",
            "gold_call": "_oracle_run_ddfv_ha_pipeline(nx, ny, 0.0, 0.0, 1)",
        },
        # --- Integration (control): the cross term removed on the benchmark mesh ---
        {
            "setup": """import numpy as np
nx, ny, distortion = 3, 4, 0.40
""",
            "call": "run_ddfv_ha_pipeline(nx, ny, distortion, drop_cross_term=True)",
            "gold_call": "_oracle_run_ddfv_ha_pipeline(nx, ny, distortion, drop_cross_term=True)",
        },
        # --- Invalid: distortion at the tangling threshold ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_ddfv_ha_pipeline(3, 4, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_ddfv_ha_pipeline(3, 4, 0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: contact tag outside the allowed pair ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_ddfv_ha_pipeline(3, 4, contact_tag=3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_ddfv_ha_pipeline(3, 4, contact_tag=3)
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
