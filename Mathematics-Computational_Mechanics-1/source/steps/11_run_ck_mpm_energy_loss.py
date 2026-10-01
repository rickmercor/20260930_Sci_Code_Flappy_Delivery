"""
Chain the sub-problem functions 01-10 over the whole implicit march of the released pre-strained block and return the fraction of its mechanical energy the scheme has lost.

Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (evaluate_kernel_weights, transfer_particles_to_grid, interpolate_grid_velocity, evaluate_corotated_response, evaluate_corotated_tangent, assemble_potential_gradient, assemble_potential_hessian, solve_newton_direction, advance_particle_state, measure_mechanical_energy) rather than reimplementing them.

The whole march is a repetition of one cycle: deposit the material points onto the staggered grids, solve the implicit momentum balance there by Newton iteration on the incremental potential, read the converged field back and advect. The body carries no external loading, so the exact motion would hold its mechanical energy fixed, and whatever the march removes is the numerical dissipation of the scheme.

Returns
-------
float: the fraction of the initial total mechanical energy removed by the implicit march, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_ck_mpm_energy_loss(spacing: float = 0.02, cells: int = 10,
                           particles_per_cell: int = 4, pre_stretch: float = 1.20,
                           density: float = 1200.0, shear_speed: float = 6.0,
                           dilatational_speed: float = 14.0, step: float = 1.0e-3,
                           n_steps: int = 20, tolerance: float = 1.0e-10,
                           max_iterations: int = 8, kernel_name: str = "compact",
                           grid_offsets: tuple = (0.0, 0.5)) -> float:
    """Run the implicit march of the released block and return the energy loss.

    The body is a square of side ``cells * spacing`` whose lower left corner
    sits on a node of the unshifted grid. It is filled with
    ``particles_per_cell`` material points per cell, laid out as a square
    sub-lattice at the cell fractions ``(j + 1/2) / side`` for
    ``j = 0, ..., side - 1`` in each direction, with ``side`` the square root
    of the number of particles per cell. Every particle carries the reference
    volume ``spacing^2 / particles_per_cell``, the mass that volume and the
    density give, zero velocity, a zero affine state and the isochoric
    deformation gradient with ``pre_stretch`` and its reciprocal on the
    diagonal.

    The Lame constants follow from the density and the two plane-strain wave
    speeds. Every step deposits the points onto the grids, drives the residual
    of the incremental potential below the tolerance by Newton iteration
    started from the deposited velocities and stopped after at most
    ``max_iterations`` updates, then reads the converged field back and
    advects. No external loading acts at any point.

    The returned number is one minus the ratio of the total mechanical energy
    after the march to the total mechanical energy before it.

    Parameters
    ----------
    spacing : float
        Background grid spacing in metres (spacing > 0).
    cells : int
        Number of cells along each side of the block (cells >= 1).
    particles_per_cell : int
        Number of material points per cell; must be a perfect square >= 1.
    pre_stretch : float
        Stretch carried by the first diagonal entry of the initial deformation
        gradient (pre_stretch > 0).
    density : float
        Mass density in kilograms per cubic metre (density > 0).
    shear_speed : float
        Shear wave speed in m s^-1 (shear_speed > 0).
    dilatational_speed : float
        Plane-strain dilatational wave speed in m s^-1; it must exceed the
        shear speed times the square root of two.
    step : float
        Time step in seconds (step > 0).
    n_steps : int
        Number of steps to march (n_steps >= 0).
    tolerance : float
        Largest absolute entry of the residual accepted as converged, in
        kilogram metre per second (tolerance > 0).
    max_iterations : int
        Largest number of Newton updates per step (max_iterations >= 1).
    kernel_name : str
        Either "compact" or "quadratic".
    grid_offsets : tuple
        Offsets of the grids of the family, in units of the spacing.

    Returns
    -------
    energy_loss : float
        Fraction of the initial total mechanical energy the march has removed,
        as a native Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above or if the initial
        state carries no mechanical energy, or if a Newton solve fails to meet
        the requested residual tolerance within ``max_iterations`` updates.
    """
    return energy_loss  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# -- The end-to-end march. Every earlier step is called through its own oracle,
#    so the value returned here never depends on a candidate's steps 01 to 10.
def _oracle_run_ck_mpm_energy_loss(spacing: float = 0.02, cells: int = 10,
                                   particles_per_cell: int = 4, pre_stretch: float = 1.20,
                                   density: float = 1200.0, shear_speed: float = 6.0,
                                   dilatational_speed: float = 14.0, step: float = 1.0e-3,
                                   n_steps: int = 20, tolerance: float = 1.0e-10,
                                   max_iterations: int = 8, kernel_name: str = "compact",
                                   grid_offsets: tuple = (0.0, 0.5)) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    for name, value, floor in (("cells", cells, 1), ("particles_per_cell", particles_per_cell, 1),
                               ("n_steps", n_steps, 0), ("max_iterations", max_iterations, 1)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    for name, value in (("spacing", spacing), ("pre_stretch", pre_stretch),
                        ("density", density), ("shear_speed", shear_speed),
                        ("dilatational_speed", dilatational_speed), ("step", step),
                        ("tolerance", tolerance)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    side = int(np.round(np.sqrt(float(particles_per_cell))))
    if side * side != int(particles_per_cell):
        raise ValueError("particles_per_cell must be a perfect square")
    if float(dilatational_speed) <= np.sqrt(2.0) * float(shear_speed):
        raise ValueError("dilatational_speed must exceed the shear speed times the square root of two")
    if kernel_name not in ("compact", "quadratic"):
        raise ValueError("kernel_name must be 'compact' or 'quadratic'")

    spacing = float(spacing)
    cells = int(cells)
    n_steps = int(n_steps)
    max_iterations = int(max_iterations)
    tolerance = float(tolerance)
    shear = float(density) * float(shear_speed) ** 2
    lame = float(density) * (float(dilatational_speed) ** 2 - 2.0 * float(shear_speed) ** 2)

    # -- Initial particle state: a square block at rest carrying an isochoric
    #    pre-strain, so all of its mechanical energy is strain energy.
    fractions = (np.arange(side) + 0.5) / side
    line = (np.arange(cells)[:, None] + fractions[None, :]).ravel() * spacing
    grid_x, grid_y = np.meshgrid(line, line, indexing='ij')
    positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    count = positions.shape[0]
    volumes = np.full(count, spacing * spacing / float(particles_per_cell))
    masses = float(density) * volumes
    gradients = np.zeros((count, 2, 2))
    gradients[:, 0, 0] = float(pre_stretch)
    gradients[:, 1, 1] = 1.0 / float(pre_stretch)
    velocities = np.zeros((count, 2))
    affine = np.zeros((count, 2, 2))
    identity = np.eye(2)[None]

    # -- Sub-problems 04 and 10: the mechanical energy before the march.
    density_before = _oracle_evaluate_corotated_response(gradients, shear, lame)[0]
    initial = _oracle_measure_mechanical_energy(masses, velocities, volumes,
                                                density_before, positions)
    if float(initial[2]) <= 0.0:
        raise ValueError("the initial state carries no mechanical energy")

    for _ in range(n_steps):
        # -- Sub-problems 01 and 02: the stencil and the deposit.
        stencil = _oracle_evaluate_kernel_weights(positions, spacing, grid_offsets, kernel_name)
        node_indices, weights, weight_gradients, node_offsets = stencil
        deposited = _oracle_transfer_particles_to_grid(masses, velocities, affine, node_indices,
                                                       weights, node_offsets)
        node_slots, node_masses, node_velocities = deposited[1], deposited[2], deposited[3]

        # -- Sub-problems 03 to 08: Newton on the incremental potential.
        trial = np.array(node_velocities, dtype=float, copy=True)
        newton_converged = False
        # Check the transferred field first, then permit exactly
        # max_iterations Newton updates and check the final update as well.
        for iteration in range(max_iterations + 1):
            rate = _oracle_interpolate_grid_velocity(trial, node_slots, weights,
                                                     weight_gradients, node_offsets)[1]
            trial_gradients = np.einsum('pab,pbc->pac', identity + step * rate, gradients)
            response = _oracle_evaluate_corotated_response(trial_gradients, shear, lame)
            residual = _oracle_assemble_potential_gradient(trial, node_velocities, node_masses,
                                                           node_slots, weight_gradients,
                                                           gradients, response[0], response[1],
                                                           volumes, step)[1]
            if float(np.max(np.abs(residual))) < tolerance:
                newton_converged = True
                break
            if iteration == max_iterations:
                break
            tangents = _oracle_evaluate_corotated_tangent(trial_gradients, shear, lame)
            hessian = _oracle_assemble_potential_hessian(tangents, gradients, weight_gradients,
                                                         node_slots, node_masses, volumes, step)
            trial = trial + _oracle_solve_newton_direction(residual, hessian)

        if not newton_converged:
            raise ValueError(
                "Newton iteration did not reach the requested residual "
                "tolerance within max_iterations updates"
            )

        # -- Sub-problems 03 and 09: read the converged field back and advect.
        converged = _oracle_interpolate_grid_velocity(trial, node_slots, weights,
                                                      weight_gradients, node_offsets)
        velocities, affine = converged[0], converged[2]
        advanced = _oracle_advance_particle_state(positions, gradients, velocities,
                                                  converged[1], step)
        positions, gradients = advanced[0], advanced[1]

    # -- Sub-problems 04 and 10: the mechanical energy after the march.
    density_after = _oracle_evaluate_corotated_response(gradients, shear, lame)[0]
    final = _oracle_measure_mechanical_energy(masses, velocities, volumes,
                                              density_after, positions)
    return float(1.0 - float(final[2]) / float(initial[2]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    import numpy as np
    """Return list of test case specifications."""
    # Every case returns a plain float: the pipeline itself returns one, and the
    # invalid cases return a status code.
    return [
        # --- Integration: small block over a short march (normal scenario) ---
        {
            "setup": """import numpy as np
cells = 4
n_steps = 4
""",
            "call": "run_ck_mpm_energy_loss(cells=cells, n_steps=n_steps)",
            "gold_call": "_oracle_run_ck_mpm_energy_loss(cells=cells, n_steps=n_steps)",
        },
        # --- Integration (boundary): no steps at all, so nothing is lost ---
        # The bare result is zero here, so it is lifted and scaled to keep the
        # comparison relative rather than vacuous.
        {
            "setup": """import numpy as np
""",
            "call": "float(1.0 + 1.0e6 * run_ck_mpm_energy_loss(cells=3, n_steps=0))",
            "gold_call": "float(1.0 + 1.0e6 * _oracle_run_ck_mpm_energy_loss(cells=3, n_steps=0))",
        },
        # --- Integration (edge): the wider quadratic kernel on a single grid ---
        {
            "setup": """import numpy as np
""",
            "call": "run_ck_mpm_energy_loss(cells=4, n_steps=4, kernel_name='quadratic', grid_offsets=(0.0,))",
            "gold_call": "_oracle_run_ck_mpm_energy_loss(cells=4, n_steps=4, kernel_name='quadratic', grid_offsets=(0.0,))",
        },
        # --- Integration (edge): a halved step over twice as many steps ---
        {
            "setup": """import numpy as np
""",
            "call": "run_ck_mpm_energy_loss(cells=4, n_steps=8, step=5.0e-4)",
            "gold_call": "_oracle_run_ck_mpm_energy_loss(cells=4, n_steps=8, step=5.0e-4)",
        },
        # --- Boundary: one particle per cell lies on the half-shifted lattice ---
        {
            "setup": """import numpy as np
""",
            "call": "run_ck_mpm_energy_loss(cells=3, n_steps=1, particles_per_cell=1)",
            "gold_call": "_oracle_run_ck_mpm_energy_loss(cells=3, n_steps=1, particles_per_cell=1)",
        },
        # --- Invalid: a particle count per cell that is not a perfect square ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_ck_mpm_energy_loss(cells=3, n_steps=1, particles_per_cell=3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_ck_mpm_energy_loss(cells=3, n_steps=1, particles_per_cell=3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: wave speeds that give a negative first Lame constant ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_ck_mpm_energy_loss(cells=3, n_steps=1, shear_speed=6.0, dilatational_speed=7.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_ck_mpm_energy_loss(cells=3, n_steps=1, shear_speed=6.0, dilatational_speed=7.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: the nonlinear solve must not advance unconverged state ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_ck_mpm_energy_loss(cells=2, n_steps=1, tolerance=1.0e-30,
                               max_iterations=1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_ck_mpm_energy_loss(cells=2, n_steps=1, tolerance=1.0e-30,
                                       max_iterations=1)
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
