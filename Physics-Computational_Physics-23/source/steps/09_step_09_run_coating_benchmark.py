"""
Chain the sub-problem functions 01-08 over the coating and the substrate and return the factor by which the near-singular treatment improves the accuracy of the reconstructed temperature.

Each layer is reconstructed independently from its own boundary data and its own domain density, and the accuracy of the reconstruction is measured against the benchmark field at the interior sample points of both layers pooled together over the Gauss node times of one step. Running the whole reconstruction twice, once with the near-singular treatment active on the element quadrature and once without it, isolates the contribution of that treatment to the accuracy of an ultra-thin coating model.

Returns
-------
float: the ratio of the pooled relative error without the near-singular treatment to the pooled relative error with it, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_coating_benchmark(length: float, thickness: float, substrate_depth: float,
                          conductivities: tuple, coating_mesh: tuple, substrate_mesh: tuple,
                          num_nodes: int, time_step: float, num_gauss: int,
                          num_radial: int) -> float:
    """Run the whole two-layer benchmark and return the accuracy gain factor.

    Parameters
    ----------
    length : float
        Extent of both layers along x (``length > 0``).
    thickness : float
        Thickness of the coating, which occupies ``0 <= y <= thickness``
        (``thickness > 0``).
    substrate_depth : float
        Depth of the substrate, which occupies ``-substrate_depth <= y <= 0``
        (``substrate_depth > 0``).
    conductivities : tuple
        Two positive floats, the thermal conductivity of the coating and of the
        substrate.
    coating_mesh : tuple
        Four positive integers for the coating: the number of boundary elements
        on each long face, the number on each short face, the number of
        interior sample columns and the number of interior sample rows.
    substrate_mesh : tuple
        The same four positive integers for the substrate.
    num_nodes : int
        Number of Gauss nodes of the time step (``num_nodes >= 2``).
    time_step : float
        Length of the time step, which starts at time zero (``time_step > 0``).
    num_gauss : int
        Number of Gauss points used on every boundary element, both for the
        boundary integrals and for the circumferential direction of the domain
        integrals (``num_gauss >= 1``).
    num_radial : int
        Number of Gauss points used in the radial direction of the domain
        integrals (``num_radial >= 1``).

    Returns
    -------
    gain : float
        Ratio of the pooled relative error obtained without the near-singular
        treatment to the pooled relative error obtained with it, as a native
        Python float.

    Raises
    ------
    ValueError
        If ``length``, ``thickness``, ``substrate_depth`` or ``time_step`` is
        not a finite real number > 0, if ``conductivities`` is not a pair of
        finite real numbers > 0, if ``coating_mesh`` or ``substrate_mesh`` is
        not a quadruple of integers >= 1, if ``num_nodes`` is not an integer
        >= 2, or if ``num_gauss`` or ``num_radial`` is not an integer >= 1.

    Notes
    -----
    This is the final, orchestrating step: call the public functions of
    sub-problems 01-08 (``time_spectral_operator``, ``rectangular_layer_mesh``,
    ``layer_reference_fields``, ``near_singular_rule``,
    ``element_boundary_terms``, ``sct_domain_terms``,
    ``domain_integrand_values``, ``reconstruct_interior_value``) and feed each
    returned value into the next, rather than reimplementing them. The same
    element quadrature rule is used for the boundary integrals and for the
    circumferential direction of the domain integrals of that element. The
    coefficient of y in the benchmark field of each layer is whatever
    continuity of temperature and of normal heat flux across the interface
    at y = 0 demands, normalised so that the substrate carries unity. The
    pooled relative error is the Euclidean norm of the difference between
    the reconstructed and the benchmark temperatures
    over the interior sample points of both layers and all Gauss node times,
    divided by the Euclidean norm of the benchmark temperatures over the same
    set. Include every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_run_coating_benchmark(length: float, thickness: float, substrate_depth: float,
                                  conductivities: tuple, coating_mesh: tuple, substrate_mesh: tuple,
                                  num_nodes: int, time_step: float, num_gauss: int,
                                  num_radial: int) -> float:
    import numpy as np

    # -- Validate the orchestrator inputs.
    for name, value in (("length", length), ("thickness", thickness),
                        ("substrate_depth", substrate_depth), ("time_step", time_step)):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite number > 0")
    if not isinstance(conductivities, (tuple, list)) or len(conductivities) != 2:
        raise ValueError("conductivities must be a sequence of exactly two numbers")
    for value in conductivities:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("conductivities must contain real numbers")
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("conductivities must be finite numbers > 0")
    for name, spec in (("coating_mesh", coating_mesh), ("substrate_mesh", substrate_mesh)):
        if not isinstance(spec, (tuple, list)) or len(spec) != 4:
            raise ValueError(f"{name} must be a sequence of exactly four integers")
        for value in spec:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must contain integers only")
            if int(value) < 1:
                raise ValueError(f"{name} entries must be >= 1")
    for name, value in (("num_nodes", num_nodes), ("num_gauss", num_gauss),
                        ("num_radial", num_radial)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if int(num_nodes) < 2:
        raise ValueError("num_nodes must be an integer >= 2")
    if int(num_gauss) < 1 or int(num_radial) < 1:
        raise ValueError("num_gauss and num_radial must be integers >= 1")

    k_coating = float(conductivities[0])
    k_substrate = float(conductivities[1])

    # -- Sub-problem 01 reference: the Gauss node times of the step and the operator that
    # turns nodal temperatures into nodal time derivatives.
    times, operator = _oracle_time_spectral_operator(int(num_nodes), float(time_step))
    times = np.asarray(times, dtype=float)

    # Interface continuity of temperature and of normal flux fixes the tilt of
    # the benchmark field in each layer.
    layers = (
        (0.0, float(thickness), tuple(coating_mesh), k_substrate / k_coating, k_coating),
        (-float(substrate_depth), 0.0, tuple(substrate_mesh), 1.0, k_substrate),
    )

    errors = {}
    for use_sinh in (True, False):
        computed = []
        reference = []
        for y_low, y_high, spec, tilt, conductivity in layers:
            # -- Sub-problem 02 reference: the outline and the interior sample points.
            elements, interior = _oracle_rectangular_layer_mesh(
                float(length), y_low, y_high,
                int(spec[0]), int(spec[1]), int(spec[2]), int(spec[3]))
            elements = np.asarray(elements, dtype=float)
            interior = np.asarray(interior, dtype=float)

            for index in range(interior.shape[0]):
                source = interior[index]
                boundary_blocks = []
                domain_blocks = []
                for element in range(elements.shape[0]):
                    nodes = elements[element]
                    # -- Sub-problem 04 reference: the element quadrature rule.
                    rule = _oracle_near_singular_rule(source, nodes, int(num_gauss), bool(use_sinh))
                    # -- Sub-problem 05 reference: the boundary quadrature of that element.
                    boundary_blocks.append(_oracle_element_boundary_terms(source, nodes, rule))
                    # -- Sub-problem 06 reference: the domain quadrature of its sector.
                    domain_blocks.append(_oracle_sct_domain_terms(source, nodes, rule, int(num_radial)))
                boundary = np.vstack([np.asarray(block, dtype=float) for block in boundary_blocks])
                domain = np.vstack([np.asarray(block, dtype=float) for block in domain_blocks])

                # -- Sub-problem 03 reference: the benchmark data the two quadratures need.
                domain_points = domain[:, :2]
                nodal = np.vstack([
                    np.asarray(_oracle_layer_reference_fields(
                        domain_points, float(instant), tilt, conductivity),
                        dtype=float)[:, 0] for instant in times])
                sources = np.vstack([
                    np.asarray(_oracle_layer_reference_fields(
                        domain_points, float(instant), tilt, conductivity),
                        dtype=float)[:, 3] for instant in times])
                initial = np.asarray(_oracle_layer_reference_fields(
                    domain_points, 0.0, tilt, conductivity), dtype=float)[:, 0]

                # -- Sub-problem 07 reference: the density of the domain integral.
                density = np.asarray(_oracle_domain_integrand_values(
                    operator, nodal, initial, sources, conductivity), dtype=float)

                for step in range(times.size):
                    fields = np.asarray(_oracle_layer_reference_fields(
                        boundary[:, :2], float(times[step]), tilt, conductivity),
                        dtype=float)
                    # -- Sub-problem 08 reference: close the representation formula.
                    computed.append(_oracle_reconstruct_interior_value(
                        boundary, fields[:, :3], domain, density[step]))
                    exact = np.asarray(_oracle_layer_reference_fields(
                        source[None, :], float(times[step]), tilt, conductivity),
                        dtype=float)
                    reference.append(float(exact[0, 0]))

        computed = np.asarray(computed, dtype=float)
        reference = np.asarray(reference, dtype=float)
        errors[use_sinh] = float(np.linalg.norm(computed - reference)
                                 / np.linalg.norm(reference))

    if errors[True] <= 0.0:
        raise ValueError("the regularised reconstruction is exact, so no gain is defined")
    return float(errors[False] / errors[True])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- A coarse two-layer configuration, cheap enough to run in a test ---
        {
            "setup": """import numpy as np
""",
            "call": ("run_coating_benchmark(4.0, 1.0e-3, 1.0, (2.0, 15.0), (4, 1, 3, 1),"
                     " (4, 1, 3, 1), 3, 1.0, 12, 6)"),
            "gold_call": ("_oracle_run_coating_benchmark(4.0, 1.0e-3, 1.0, (2.0, 15.0), (4, 1, 3, 1),"
                          " (4, 1, 3, 1), 3, 1.0, 12, 6)"),
        },
        # --- A thicker coating, where the plain rule still resolves the kernels
        # and the gain is far smaller ---
        {
            "setup": """import numpy as np
""",
            "call": ("run_coating_benchmark(2.0, 0.25, 0.5, (1.0, 4.0), (3, 1, 2, 1),"
                     " (3, 1, 2, 1), 3, 0.5, 10, 5)"),
            "gold_call": ("_oracle_run_coating_benchmark(2.0, 0.25, 0.5, (1.0, 4.0), (3, 1, 2, 1),"
                          " (3, 1, 2, 1), 3, 0.5, 10, 5)"),
        },
        # --- A square coating on a square substrate with equal conductivities,
        # where the benchmark field has the same tilt in both layers ---
        {
            "setup": """import numpy as np
""",
            "call": ("run_coating_benchmark(1.0, 1.0, 1.0, (3.0, 3.0), (2, 2, 2, 2),"
                     " (2, 2, 2, 2), 4, 2.0, 10, 6)"),
            "gold_call": ("_oracle_run_coating_benchmark(1.0, 1.0, 1.0, (3.0, 3.0), (2, 2, 2, 2),"
                          " (2, 2, 2, 2), 4, 2.0, 10, 6)"),
        },
        # --- An extremely thin coating on a coarse mesh, where the plain rule
        # loses the boundary integral altogether ---
        {
            "setup": """import numpy as np
""",
            "call": ("run_coating_benchmark(4.0, 1.0e-7, 1.0, (2.0, 15.0), (4, 1, 2, 1),"
                     " (4, 1, 2, 1), 3, 1.0, 16, 6)"),
            "gold_call": ("_oracle_run_coating_benchmark(4.0, 1.0e-7, 1.0, (2.0, 15.0), (4, 1, 2, 1),"
                          " (4, 1, 2, 1), 3, 1.0, 16, 6)"),
            "tol": 1.0e-5,
        },
        # --- Invalid: a non-positive coating thickness ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_coating_benchmark(4.0, 0.0, 1.0, (2.0, 15.0), (2, 1, 2, 1), (2, 1, 2, 1), 3, 1.0, 8, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_coating_benchmark(4.0, 0.0, 1.0, (2.0, 15.0), (2, 1, 2, 1), (2, 1, 2, 1), 3, 1.0, 8, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a mesh specification with only three entries ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_coating_benchmark(4.0, 1.0e-3, 1.0, (2.0, 15.0), (2, 1, 2), (2, 1, 2, 1), 3, 1.0, 8, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_coating_benchmark(4.0, 1.0e-3, 1.0, (2.0, 15.0), (2, 1, 2), (2, 1, 2, 1), 3, 1.0, 8, 4)
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
