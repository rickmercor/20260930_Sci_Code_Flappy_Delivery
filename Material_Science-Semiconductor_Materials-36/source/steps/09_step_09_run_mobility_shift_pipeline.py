"""
Chain the sub-problem functions 01-08 end to end on the quenched through-silicon via and return the piezoresistive carrier mobility change rate at the requested point and instant.

Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (compute_axial_eigenmodes, propagate_radial_eigenfunction, compute_radial_eigenvalues, compute_modal_coefficients, compute_temperature_rise, compute_potential_stress_terms, compute_love_mode_coefficients, assemble_mode_stress) rather than reimplementing them.

This step runs the whole measurement end to end. It (i) builds the axial eigenvalues and norms of the insulated-to-sink via with sub-problem 01, (ii) for each axial mode finds the thermal decay rates of the layered stack with sub-problem 03, which internally repeats the outward propagation of sub-problem 02, (iii) stores the layer amplitudes of every mode with sub-problem 02, (iv) projects the uniform initial temperature rise onto that basis with sub-problem 04, (v) evaluates the thermoelastic-potential amplitudes at the two boundaries of every layer with sub-problem 06, (vi) solves the bonded-interface and traction-free system for the Love coefficients of every axial mode with sub-problem 07, (vii) assembles the total stress amplitudes at the query radius with sub-problem 08, and (viii) sums the modes against cos(eta z) for the normal components and sin(eta z) for the shear component. Sub-problem 05 supplies the temperature at the same point, and the assembled potential stresses are checked against it before the shift is returned: a relative disagreement above 1e-6 raises a ValueError.

The returned scalar is the change in hole mobility, in percent, of a PMOS transistor at the query point whose channel is aligned with the radial direction. With Delta rho / rho = piezo_coefficient * orientation_factor * sigma_rr the piezoresistive change of the channel resistivity (sigma_rr in Pa, tension positive), the returned change is Delta mu / mu = -Delta rho / rho, positive when the hole mobility increases; this minus sign is stated as a correction, because the relation is sometimes printed without it. The locus where the change falls below the tolerance of a circuit defines the keep-out zone around the via, and because the stress relaxes as the via cools, that zone changes with time.

Expected return: the piezoresistive carrier mobility change rate at the requested point and instant, expressed in percent.

Returns
-------
float: the piezoresistive carrier mobility change rate at the query point and instant, in percent, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_mobility_shift_pipeline(radius: float = 20.0e-6,
                                depth: float = 200.0e-6 / 3.0,
                                time: float = 5.0e-5,
                                n_axial: int = 8, n_radial: int = 8,
                                radii=(15.0e-6, 16.0e-6, 30.0e-6),
                                height: float = 200.0e-6,
                                conductivity=(400.0, 1.4, 130.0),
                                density=(8960.0, 2200.0, 2329.0),
                                heat_capacity=(385.0, 730.0, 700.0),
                                expansion=(17.0e-6, 0.5e-6, 2.6e-6),
                                young=(110.0e9, 70.0e9, 170.0e9),
                                poisson=(0.35, 0.17, 0.28),
                                initial_rise: float = 100.0,
                                piezo_coefficient: float = 71.8e-11,
                                orientation_factor: float = 1.0) -> float:
    """Run the full transient thermal stress and mobility measurement.

    Parameters
    ----------
    radius : float
        Radial coordinate of the evaluation point in metres, 0 < radius <= radii[-1].
    depth : float
        Axial coordinate of the evaluation point in metres, measured from the
        insulated base (>= 0).
    time : float
        Elapsed time since the quench in seconds (>= 0).
    n_axial : int
        Number of axial modes retained (n_axial >= 1).
    n_radial : int
        Number of radial modes retained per axial mode (n_radial >= 1).
    radii : sequence of float
        Outer radius of each layer in metres, strictly increasing, l >= 2.
    height : float
        Total via height in metres (> 0).
    conductivity : sequence of float
        Thermal conductivity of each layer in W/(m K), all > 0.
    density : sequence of float
        Density of each layer in kg/m^3, all > 0.
    heat_capacity : sequence of float
        Specific heat capacity of each layer in J/(kg K), all > 0.
    expansion : sequence of float
        Coefficient of thermal expansion of each layer in 1/K, all > 0.
    young : sequence of float
        Young modulus of each layer in Pa, all > 0.
    poisson : sequence of float
        Poisson ratio of each layer, each in (0, 0.5).
    initial_rise : float
        Uniform initial temperature rise above ambient in kelvin.
    piezo_coefficient : float
        Piezoresistive coefficient in 1/Pa.
    orientation_factor : float
        Orientation factor between the stress and the transistor channel.

    Returns
    -------
    shift : float
        Piezoresistive carrier mobility change rate in percent, positive when
        the hole mobility increases, as a native Python float.

    Raises
    ------
    ValueError
        If radii holds fewer than two layers; if conductivity, density,
        heat_capacity, expansion, young or poisson does not have one
        entry per layer or carries an entry that is not finite and
        greater than 0; if radii is not finite, positive and strictly
        increasing; if any poisson entry is not below 0.5; if n_axial or
        n_radial is not an integer greater than or equal to 1; if
        height, initial_rise or piezo_coefficient is not a finite number
        greater than 0; if radius, depth, time or orientation_factor is
        not a finite number; if radius lies outside the half-open
        interval (0, radii[-1]]; or if depth or time is negative.
    """
    return shift  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_mobility_shift_pipeline(radius: float = 20.0e-6,
                                        depth: float = 200.0e-6 / 3.0,
                                        time: float = 5.0e-5,
                                        n_axial: int = 8, n_radial: int = 8,
                                        radii=(15.0e-6, 16.0e-6, 30.0e-6),
                                        height: float = 200.0e-6,
                                        conductivity=(400.0, 1.4, 130.0),
                                        density=(8960.0, 2200.0, 2329.0),
                                        heat_capacity=(385.0, 730.0, 700.0),
                                        expansion=(17.0e-6, 0.5e-6, 2.6e-6),
                                        young=(110.0e9, 70.0e9, 170.0e9),
                                        poisson=(0.35, 0.17, 0.28),
                                        initial_rise: float = 100.0,
                                        piezo_coefficient: float = 71.8e-11,
                                        orientation_factor: float = 1.0) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    # -- The oracles of sub-problems 01-08 share this namespace.
    axial_modes = _oracle_compute_axial_eigenmodes
    propagate = _oracle_propagate_radial_eigenfunction
    radial_modes = _oracle_compute_radial_eigenvalues
    modal_coefficients = _oracle_compute_modal_coefficients
    temperature_rise = _oracle_compute_temperature_rise
    potential_terms = _oracle_compute_potential_stress_terms
    love_coefficients = _oracle_compute_love_mode_coefficients
    mode_stress = _oracle_assemble_mode_stress

    # -- Validate the orchestrator inputs.
    radii = np.asarray(radii, dtype=float).ravel()
    conductivity = np.asarray(conductivity, dtype=float).ravel()
    density = np.asarray(density, dtype=float).ravel()
    heat_capacity = np.asarray(heat_capacity, dtype=float).ravel()
    expansion = np.asarray(expansion, dtype=float).ravel()
    young = np.asarray(young, dtype=float).ravel()
    poisson = np.asarray(poisson, dtype=float).ravel()
    n_layers = radii.size
    if n_layers < 2:
        raise ValueError("the via must have at least two layers")
    for name, table in (("conductivity", conductivity), ("density", density),
                        ("heat_capacity", heat_capacity), ("expansion", expansion),
                        ("young", young), ("poisson", poisson)):
        if table.size != n_layers:
            raise ValueError(f"{name} must have one entry per layer")
        if not np.all(np.isfinite(table)) or np.any(table <= 0.0):
            raise ValueError(f"{name} entries must be finite and > 0")
    if not np.all(np.isfinite(radii)) or radii[0] <= 0.0 or np.any(np.diff(radii) <= 0.0):
        raise ValueError("radii must be finite, positive and strictly increasing")
    if np.any(poisson >= 0.5):
        raise ValueError("poisson entries must be below 0.5")
    for name, value, floor in (("n_axial", n_axial, 1), ("n_radial", n_radial, 1)):
        if not (isinstance(value, (int, np.integer)) and not isinstance(value, bool)
                and int(value) >= floor):
            raise ValueError(f"{name} must be an integer >= {floor}")
    for name, value in (("height", height), ("initial_rise", initial_rise),
                        ("piezo_coefficient", piezo_coefficient)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value) and float(value) > 0.0):
            raise ValueError(f"{name} must be a finite number > 0")
    for name, value in (("radius", radius), ("depth", depth), ("time", time),
                        ("orientation_factor", orientation_factor)):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
    if float(radius) <= 0.0 or float(radius) > radii[-1]:
        raise ValueError("radius must lie in the half-open interval (0, radii[-1]]")
    if float(depth) < 0.0 or float(time) < 0.0:
        raise ValueError("depth and time must be >= 0")

    radius = float(radius)
    depth = float(depth)
    time = float(time)
    n_axial = int(n_axial)
    n_radial = int(n_radial)

    diffusivity = conductivity / (density * heat_capacity)
    shear = young / (2.0 * (1.0 + poisson))
    inner_radii = np.concatenate(([0.0], radii[:-1]))
    layer_of_query = int(min(np.searchsorted(radii, radius, side="left"), n_layers - 1))

    # -- Sub-problem 01: axial eigenvalues and their squared norms.
    axial = np.asarray(axial_modes(float(height), n_axial), dtype=float)

    # The temperature evaluation of sub-problem 05 takes the whole basis at
    # once, so the per-mode tables are collected while the modes are built.
    decay_table = np.zeros((n_axial, n_radial), dtype=float)
    coefficient_table = np.zeros((n_axial, n_radial), dtype=float)
    eigen_table = np.zeros((n_axial, n_radial, n_layers, 3), dtype=float)
    potential_trace = 0.0

    stress = np.zeros(4, dtype=float)
    for m in range(n_axial):
        eta = float(axial[m, 0])
        axial_norm = float(axial[m, 1])

        # -- Sub-problem 03: thermal decay rates of this axial mode.
        rates = np.asarray(radial_modes(radii, conductivity, diffusivity, eta,
                                        n_radial), dtype=float)
        # -- Sub-problem 02: layer amplitudes of each radial eigenfunction.
        eigen = np.array([np.asarray(propagate(radii, conductivity, diffusivity,
                                               eta, float(rate)), dtype=float)
                          for rate in rates])
        # -- Sub-problem 04: projection of the uniform initial rise.
        coefficients = np.asarray(modal_coefficients(radii, conductivity, diffusivity,
                                                     float(height), eta, axial_norm,
                                                     eigen, float(initial_rise)),
                                  dtype=float)
        decay_table[m] = rates
        coefficient_table[m] = coefficients
        eigen_table[m] = eigen

        # -- Sub-problem 06: potential amplitudes at both faces of every layer.
        boundary_terms = np.zeros((n_layers, 2, 6), dtype=float)
        for i in range(n_layers):
            boundary_terms[i, 0] = potential_terms(
                float(radii[i]), eta, rates, coefficients, eigen[:, i, :],
                float(expansion[i]), float(poisson[i]), float(shear[i]), time)
            if i > 0:
                boundary_terms[i, 1] = potential_terms(
                    float(inner_radii[i]), eta, rates, coefficients, eigen[:, i, :],
                    float(expansion[i]), float(poisson[i]), float(shear[i]), time)

        # -- Sub-problem 07: Love coefficients enforcing the interfaces and surface.
        love = np.asarray(love_coefficients(radii, poisson, shear, eta,
                                            boundary_terms), dtype=float)

        # -- Sub-problem 06 again, now at the query radius, then sub-problem 08.
        query_terms = potential_terms(
            radius, eta, rates, coefficients, eigen[:, layer_of_query, :],
            float(expansion[layer_of_query]), float(poisson[layer_of_query]),
            float(shear[layer_of_query]), time)
        amplitudes = np.asarray(mode_stress(radius, radii, poisson, shear, eta,
                                            love, query_terms), dtype=float)
        potential_trace += float(query_terms[2] + query_terms[3]
                                 + query_terms[4]) * np.cos(eta * depth)

        # The three normal components ride on cos(eta z), the shear on sin(eta z).
        stress += amplitudes * np.array([np.cos(eta * depth), np.cos(eta * depth),
                                         np.cos(eta * depth), np.sin(eta * depth)])

    # -- Sub-problem 05: the temperature the stress state is read against. The
    #    potential is built so that its Laplacian returns the temperature, so the
    #    trace of the potential part of the stress at the query point is exactly
    #    -4 G alpha (1 + nu) / (1 - nu) theta there. Checking the two against one
    #    another ties the assembled mechanics to the temperature series.
    rise = float(temperature_rise(radii, axial[:, 0], decay_table,
                                  coefficient_table, eigen_table, radius, depth,
                                  time))
    expected_trace = (-4.0 * float(shear[layer_of_query])
                      * float(expansion[layer_of_query])
                      * (1.0 + float(poisson[layer_of_query]))
                      / (1.0 - float(poisson[layer_of_query])) * rise)
    scale = max(abs(potential_trace), abs(expected_trace))
    if abs(potential_trace - expected_trace) > 1.0e-6 * scale:
        raise ValueError("the temperature series and the thermoelastic potential "
                         "disagree at the query point")

    # -- Piezoresistive conversion: dmu/mu = -drho/rho, drho/rho = Pi * beta * sigma_rr.
    return float(-100.0 * float(piezo_coefficient) * float(orientation_factor) * stress[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: final-answer configuration (normal scenario) ---
        {
            "setup": """import numpy as np
""",
            "call": "run_mobility_shift_pipeline()",
            "gold_call": "_oracle_run_mobility_shift_pipeline()",
        },
        # --- Integration: same instant, closer to the liner where the stress is larger ---
        {
            "setup": """import numpy as np
radius = 17.0e-6
""",
            "call": "run_mobility_shift_pipeline(radius)",
            "gold_call": "_oracle_run_mobility_shift_pipeline(radius)",
        },
        # --- Integration: earlier instant, near the base plane ---
        {
            "setup": """import numpy as np
radius = 22.0e-6
depth = 200.0e-6 / 6.0
time = 1.0e-5
""",
            "call": "run_mobility_shift_pipeline(radius, depth, time)",
            "gold_call": "_oracle_run_mobility_shift_pipeline(radius, depth, time)",
        },
        # --- Integration (boundary): a single retained mode in each direction ---
        {
            "setup": """import numpy as np
""",
            "call": "run_mobility_shift_pipeline(20.0e-6, 200.0e-6 / 3.0, 5.0e-5, 1, 1)",
            "gold_call": "_oracle_run_mobility_shift_pipeline(20.0e-6, 200.0e-6 / 3.0, 5.0e-5, 1, 1)",
        },
        # --- Integration (edge): tungsten via with a nitride liner in a thin die ---
        {
            "setup": """import numpy as np
radii = (2.5e-6, 2.9e-6, 5.0e-6)
height = 50.0e-6
conductivity = (173.0, 20.0, 130.0)
density = (19250.0, 3100.0, 2329.0)
heat_capacity = (132.0, 700.0, 700.0)
expansion = (4.5e-6, 2.3e-6, 2.6e-6)
young = (411.0e9, 250.0e9, 170.0e9)
poisson = (0.28, 0.23, 0.28)
""",
            "call": "run_mobility_shift_pipeline(4.0e-6, 50.0e-6 / 3.0, 2.0e-6, 6, 6, radii, height, conductivity, density, heat_capacity, expansion, young, poisson, 200.0)",
            "gold_call": "_oracle_run_mobility_shift_pipeline(4.0e-6, 50.0e-6 / 3.0, 2.0e-6, 6, 6, radii, height, conductivity, density, heat_capacity, expansion, young, poisson, 200.0)",
        },
        # --- Invalid: query radius outside the unit cell ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_mobility_shift_pipeline(45.0e-6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_mobility_shift_pipeline(45.0e-6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: no modes retained ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_mobility_shift_pipeline(20.0e-6, 200.0e-6 / 3.0, 5.0e-5, 0, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_mobility_shift_pipeline(20.0e-6, 200.0e-6 / 3.0, 5.0e-5, 0, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative elapsed time ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_mobility_shift_pipeline(20.0e-6, 200.0e-6 / 3.0, -1.0e-5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_mobility_shift_pipeline(20.0e-6, 200.0e-6 / 3.0, -1.0e-5)
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
