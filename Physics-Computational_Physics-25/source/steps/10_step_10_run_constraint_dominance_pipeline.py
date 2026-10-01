"""
Chain the sub-problem functions 01-09 end-to-end over the three stratification orders and return the constraint dominance index of the layered half-space. Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (assemble_near_field_operators, assemble_patch_load_vector, assemble_scaling_surface_coefficients, solve_far_field_conduction_matrix, compute_precise_propagator, integrate_temperature_field, solve_thermoelastic_displacement, compute_peak_stress_and_heave, compute_surface_gradient) rather than reimplementing them.

This step runs the whole measurement end to end for each of the three stratification orders. It (i) assembles the layered near field's conductance and capacity operators with sub-problem 01 and the convective load vector of the heated patch with sub-problem 02, (ii) builds the radial coefficient matrices of the unbounded exterior with sub-problem 03 and condenses them into the steady far-field conduction matrix with sub-problem 04, (iii) adds that matrix to the interface degrees of freedom and inverts the capacity operator against the total conductance to form the state matrix of the semi-discrete system, (iv) builds with sub-problem 05 the propagators the march repeats at every step, one over the whole step and one over the tail of each quadrature abscissa, and marches the temperature field to the end of the analysis with sub-problem 06, (v) solves the one-way coupled restrained elastic problem with sub-problem 07, (vi) reduces the result to a peak equivalent stress and a peak surface heave with sub-problem 08, and (vii) measures the steepness of the thermal front in the uppermost stratum with sub-problem 09.




The returned scalar compares two spreads across the three orders: how far the peak stress moves, divided by how far the near-surface thermal gradient moves. A value near one would mean the stress simply tracks the thermal picture, so that reading severity off a temperature contour plot is safe and the stratification enters mechanically only through the temperature field it produces. A value well above one means the two orderings are governed by different material properties - the thermal properties of the surface stratum for the gradient, the product of stiffness and expansion coefficient under restraint for the stress - and that the thermal picture is therefore not a proxy for the mechanical one. The index is dimensionless and built from ratios, so it is insensitive to the absolute level of the thermal loading and reports only the relative sensitivity of the two fields to the layering.

Returns
-------
float: the constraint dominance index of the layered half-space for this configuration, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def run_constraint_dominance_pipeline(width: float = 120.0, depth: float = 40.0,
                                      patch_width: float = 60.0, n_lateral: int = 4,
                                      n_depth: int = 8, film_coefficient: float = 25.0,
                                      ambient_amplitude: float = 80.0,
                                      ramp_time: float = 1.0e5, step: float = 2500.0,
                                      n_steps: int = 80, n_gauss: int = 6,
                                      n_levels: int = 20, contraction: float = 0.5,
                                      radial_coordinate: float = 1.0) -> float:
    """Run the constraint dominance measurement over the three stratification orders.

    Parameters
    ----------
    width : float
        Lateral extent of the near-field box in metres (width > 0).
    depth : float
        Depth of the near-field box in metres (depth > 0).
    patch_width : float
        Side length of the centred square heated patch in metres
        (patch_width >= 0). The patch is resolved by the mesh, so a
        top-surface element face is either wholly heated or not heated.
    n_lateral : int
        Number of elements along each lateral direction (n_lateral >= 1).
    n_depth : int
        Number of elements through the depth, a multiple of four
        (n_depth >= 1).
    film_coefficient : float
        Convective film coefficient over the patch in W m^-2 K^-1
        (film_coefficient >= 0).
    ambient_amplitude : float
        Final ambient temperature of the saturating history, in degrees
        Celsius.
    ramp_time : float
        Time constant of the ambient history in seconds (ramp_time > 0).
    step : float
        Time step in seconds (step > 0).
    n_steps : int
        Number of steps to march (n_steps >= 0).
    n_gauss : int
        Number of Gauss-Legendre points used for the load integral over each
        step (n_gauss >= 1).
    n_levels : int
        Number of subdivision levels used to build each propagator
        (n_levels >= 0).
    contraction : float
        Ratio by which the boundary square is contracted to form the scaling
        surface, 0 <= contraction < 1.
    radial_coordinate : float
        Radial coordinate at which the far-field coefficient matrices are
        frozen (radial_coordinate > 0).

    Returns
    -------
    index : float
        The spread of the peak von Mises stress across the three
        stratification orders divided by the spread of the near-surface
        thermal gradient, as a native Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above.
    """
    return index  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================


def _oracle_run_constraint_dominance_pipeline(width: float = 120.0, depth: float = 40.0,
                                              patch_width: float = 60.0, n_lateral: int = 4,
                                              n_depth: int = 8, film_coefficient: float = 25.0,
                                              ambient_amplitude: float = 80.0,
                                              ramp_time: float = 1.0e5, step: float = 2500.0,
                                              n_steps: int = 80, n_gauss: int = 6,
                                              n_levels: int = 20, contraction: float = 0.5,
                                              radial_coordinate: float = 1.0) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    # (thermal conductivity, density, Young's modulus, Poisson's ratio,
    #  thermal expansion coefficient, specific heat capacity) of materials 1 to 4.
    _MATERIALS = (
        (50.0, 1000.0, 10.0e6, 0.30, 1.0e-5, 10.0),
        (66.0, 1500.0, 15.0e6, 0.35, 5.0e-5, 15.0),
        (89.0, 2000.0, 20.0e6, 0.40, 1.0e-6, 20.0),
        (100.0, 2500.0, 25.0e6, 0.45, 5.0e-6, 25.0),
    )

    # The three stratifications compared, listed from the free surface downwards.
    _STRATIFICATIONS = ((1, 2, 3, 4), (2, 3, 1, 4), (4, 3, 2, 1))

    def _interface_nodes(n_lateral, n_depth):
        """Return the global indices of the base-face nodes, in surface-local order."""
        return [i + (n_lateral + 1) * (j + (n_lateral + 1) * n_depth)
                for j in range(n_lateral + 1) for i in range(n_lateral + 1)]

    # -- The earlier steps are called by their reference names, so that this
    #    value is produced by the reference chain alone and never by whatever
    #    implementation of an earlier step happens to be in scope.

    # -- Validate the orchestrator inputs.
    for name, value, floor in (("n_lateral", n_lateral, 1), ("n_depth", n_depth, 1),
                               ("n_steps", n_steps, 0), ("n_gauss", n_gauss, 1),
                               ("n_levels", n_levels, 0)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    for name, value in (("width", width), ("depth", depth), ("ramp_time", ramp_time),
                        ("step", step), ("radial_coordinate", radial_coordinate)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    for name, value in (("patch_width", patch_width),
                        ("film_coefficient", film_coefficient)):
        if not (isinstance(value, (int, float)) and np.isfinite(value) and float(value) >= 0.0):
            raise ValueError(f"{name} must be a finite number >= 0")
    if not (isinstance(contraction, (int, float)) and np.isfinite(contraction)
            and 0.0 <= float(contraction) < 1.0):
        raise ValueError("contraction must be a finite number in the interval [0, 1)")
    if int(n_depth) % 4 != 0:
        raise ValueError("n_depth must be a multiple of 4 so that every element lies in one stratum")

    n_lateral = int(n_lateral)
    n_depth = int(n_depth)
    n_gauss = int(n_gauss)
    n_levels = int(n_levels)
    interface = _interface_nodes(n_lateral, n_depth)

    peak_stress = []
    peak_gradient = []
    for order in _STRATIFICATIONS:
        # -- Sub-problems 01-02: the near-field operators and the patch load.
        operators = _oracle_assemble_near_field_operators(list(order), width, depth, patch_width,
                                                  n_lateral, n_depth, film_coefficient)
        conductance = np.array(operators[0], dtype=float)
        capacity = np.array(operators[1], dtype=float)
        load = _oracle_assemble_patch_load_vector(width, depth, patch_width, n_lateral, n_depth,
                                          film_coefficient)

        # -- Sub-problems 03-04: the unbounded exterior, condensed onto the base.
        coefficients = _oracle_assemble_scaling_surface_coefficients(width, depth, n_lateral,
                                                             _MATERIALS[order[3] - 1][0],
                                                             contraction, radial_coordinate)
        far_field = _oracle_solve_far_field_conduction_matrix(coefficients[0], coefficients[1],
                                                      coefficients[2])
        conductance[np.ix_(interface, interface)] += far_field

        # -- Sub-problems 05-06: the semi-discrete system, the propagators the
        #    march repeats at every step, and the march itself. The propagator
        #    over a whole step and the one over the tail of each quadrature
        #    abscissa depend only on the state matrix, the step and the number
        #    of subdivision levels, so they are built here once per order and
        #    handed to the march.
        state = -np.linalg.solve(capacity, conductance)
        influence = np.linalg.solve(capacity, load)
        abscissae = np.polynomial.legendre.leggauss(n_gauss)[0]
        step_propagator = _oracle_compute_precise_propagator(state, step, n_levels)
        tail_propagators = [_oracle_compute_precise_propagator(state, 0.5 * step * (1.0 - node), n_levels)
                            for node in abscissae]
        temperature = _oracle_integrate_temperature_field(step_propagator, tail_propagators, influence,
                                                  ambient_amplitude, ramp_time, step, n_steps)

        # -- Sub-problems 07-09: the mechanical response and the thermal front.
        displacement = _oracle_solve_thermoelastic_displacement(list(order), width, depth,
                                                        n_lateral, n_depth, temperature)
        measures = _oracle_compute_peak_stress_and_heave(list(order), width, depth, n_lateral,
                                                 n_depth, temperature, displacement)
        peak_stress.append(float(measures[0]))
        peak_gradient.append(float(_oracle_compute_surface_gradient(width, depth, n_lateral,
                                                            n_depth, temperature)))

    stress = np.array(peak_stress, dtype=float)
    gradient = np.array(peak_gradient, dtype=float)
    if np.min(stress) <= 0.0 or np.min(gradient) <= 0.0:
        raise ValueError("the configuration produced a vanishing stress or gradient spread")
    return float((stress.max() / stress.min()) / (gradient.max() / gradient.min()))

# =============================================================================
# TEST CASES
# =============================================================================

# =============================================================================
# TEST CASES (integration tests -- whole pipeline)
# =============================================================================



def test_cases():
    """Return list of test case specifications."""
    # Every case returns a plain float: the pipeline itself returns one, and the
    # invalid cases return a status code.
    return [
        # --- Integration: coarse mesh, whole pipeline (normal scenario) ---
        {
            "setup": """import numpy as np
n_depth = 4
n_steps = 20
""",
            "call": "run_constraint_dominance_pipeline(n_depth=n_depth, n_steps=n_steps, step=1.0e4)",
            "gold_call": "_oracle_run_constraint_dominance_pipeline(n_depth=n_depth, n_steps=n_steps, step=1.0e4)",
        },
        # --- Integration: final-answer configuration ---
        {
            "setup": """import numpy as np
""",
            "call": "run_constraint_dominance_pipeline()",
            "gold_call": "_oracle_run_constraint_dominance_pipeline()",
        },
        # --- Integration (boundary): the whole free surface heated ---
        {
            "setup": """import numpy as np
patch_width = 120.0
""",
            "call": "run_constraint_dominance_pipeline(patch_width=patch_width, n_depth=4, n_steps=20, step=1.0e4)",
            "gold_call": "_oracle_run_constraint_dominance_pipeline(patch_width=patch_width, n_depth=4, n_steps=20, step=1.0e4)",
        },
        # --- Integration (edge): scaling surface collapsed to a point, coarse quadrature ---
        {
            "setup": """import numpy as np
""",
            "call": "run_constraint_dominance_pipeline(n_depth=4, n_steps=20, step=1.0e4, n_gauss=3, contraction=0.0)",
            "gold_call": "_oracle_run_constraint_dominance_pipeline(n_depth=4, n_steps=20, step=1.0e4, n_gauss=3, contraction=0.0)",
        },
        # --- Invalid: depth discretisation incompatible with four strata ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_constraint_dominance_pipeline(n_depth=5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_constraint_dominance_pipeline(n_depth=5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: contraction outside its admissible interval ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_constraint_dominance_pipeline(contraction=1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_constraint_dominance_pipeline(contraction=1.5)
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
