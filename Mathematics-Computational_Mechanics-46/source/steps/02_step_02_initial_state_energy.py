"""
Builds the initial free-expansion velocity field and its kinetic energy, the setup-time interface stiffness cap, and the full-row Gershgorin stability bound that fixes the march's fixed time step and step count.

Initial state, interface stiffness cap, and the setup time step.




The bar starts undeformed with a linear velocity field proportional to position, so every discrete degree of freedom carries a nonzero initial velocity while the initial displacement and stress are both zero; the discrete kinetic energy formed from the lumped masses very slightly overestimates the continuum kinetic-energy integral of the same field because lumping concentrates each element's mass at its two end nodes, and the initial momentum vanishes because the velocity field is antisymmetric about the bar's centre. Every cohesive interface is assigned, once at setup, a capped spring constant far stiffer than the bulk element spring; because interface damage only ever grows and the damage-secant spring can only get softer than that cap, the capped value is a valid upper bound on every interface's stiffness contribution for the entire simulation, which lets a single stability estimate taken at this cap remain valid throughout the march instead of needing to be recomputed at every step. That estimate is a full-row Gershgorin bound on the squared highest natural frequency of the assembled system: the worst-case row belongs to a duplicated interface degree of freedom, which couples to its bulk neighbour through the ordinary bulk spring and to its own twin face through the capped interface spring, so its row-sum of stiffness magnitudes is the largest in the whole assembled matrix relative to its lumped mass. A safety-factored fraction of the resulting critical step is adopted as the setup step, the total number of steps to the requested horizon is fixed by rounding the horizon divided by that setup step, and the step actually used in the march is the horizon divided by that whole number of steps so that the march lands on the horizon exactly; the convexity margin confirms the actually-used step still respects the same stability bound with room to spare.

Returns
-------
result : dict -- E0_discrete, E0_continuum, momentum, k_tilde, k_tilde_A, k_bulk_element, omega2_max, omega_max, dt_setup, N_steps, dt_prime, convexity_margin, bar_period, wave_speed, t_final (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def initial_state_energy(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, t_star_over_tb: float = 0.42, safety: float = 0.99) -> dict:
    """Initial kinetic energy, interface cap, and the Gershgorin setup step.

    Parameters
    ----------
    n_e : int
        Number of uniform two-node elements (must be an even integer >= 2).
    edot : float
        Nominal applied strain rate, in 1/s (must be > 0).
    alpha : float
        Interface stiffness cap ratio, dimensionless (must be > 0); the
        capped interface spring per unit area is alpha * youngs_modulus /
        h_e, with h_e = length / n_e.
    length : float
        Total bar length, in metres (must be > 0).
    youngs_modulus : float
        Bulk Young's modulus, in pascals (must be > 0).
    density : float
        Bulk mass density, in kg/m^3 (must be > 0).
    area : float
        Cross-sectional area, in m^2 (must be > 0).
    t_star_over_tb : float
        Requested horizon as a fraction of the bar period 2*length/wave_speed
        (must be > 0).
    safety : float
        Safety factor applied to the Gershgorin critical step, in (0, 1].

    Returns
    -------
    result : dict
        E0_discrete : float, initial kinetic energy from the lumped masses.
        E0_continuum : float, the continuum kinetic-energy estimate for the
            same velocity field.
        momentum : float, total initial momentum (nominally zero).
        k_tilde : float, capped interface spring per unit area.
        k_tilde_A : float, capped interface spring, k_tilde * area.
        k_bulk_element : float, bulk element spring youngs_modulus*area/h_e.
        omega2_max : float, full-row Gershgorin bound on the squared highest
            natural frequency.
        omega_max : float, sqrt(omega2_max).
        dt_setup : float, safety * 2 / omega_max.
        N_steps : int, round(t_final / dt_setup) with t_final =
            t_star_over_tb * bar_period.
        dt_prime : float, t_final / N_steps.
        convexity_margin : float, dt_prime * omega_max / 2 (must stay < 1
            for the contact solve of later steps to remain a convex QP).
        bar_period : float, 2 * length / wave_speed.
        wave_speed : float, sqrt(youngs_modulus / density).
        t_final : float, t_star_over_tb * bar_period.

    Raises
    ------
    ValueError
        If n_e is not an even integer >= 2, or if edot, alpha, length,
        youngs_modulus, density, area, t_star_over_tb, or safety is not a
        positive real number.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _build_positions_and_mass(n_e, length, density, area):
    """Duplicated-node positions and lumped masses (self-contained; mirrors
    the DOF map of 01_bar_model_setup without importing it)."""
    n_nodes = n_e + 1
    h_e = length / n_e
    node_dofs = [None] * n_nodes
    dof = 0
    for k in range(n_nodes):
        if k % 2 == 1 and k != n_e:
            node_dofs[k] = (dof, dof + 1)
            dof += 2
        else:
            node_dofs[k] = (dof,)
            dof += 1
    n_dof = dof
    mass_regular = density * area * h_e
    mass_face = mass_regular / 2.0
    x = np.zeros(n_dof)
    mass = np.zeros(n_dof)
    for k in range(n_nodes):
        dofs = node_dofs[k]
        for d in dofs:
            x[d] = k * h_e - length / 2.0
        if len(dofs) == 2:
            mass[list(dofs)] = mass_face
        elif k == 0 or k == n_e:
            mass[dofs[0]] = mass_face
        else:
            mass[dofs[0]] = mass_regular
    return x, mass, h_e, mass_regular, mass_face


def _oracle_initial_state_energy(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, t_star_over_tb: float = 0.42, safety: float = 0.99) -> dict:
    """Reference implementation of initial_state_energy."""
    if not isinstance(n_e, (int, np.integer)) or int(n_e) < 2 or int(n_e) % 2 != 0:
        raise ValueError("n_e must be an even integer >= 2")
    if not (edot > 0.0 and alpha > 0.0 and length > 0.0 and youngs_modulus > 0.0
            and density > 0.0 and area > 0.0 and t_star_over_tb > 0.0 and 0.0 < safety <= 1.0):
        raise ValueError("edot, alpha, length, youngs_modulus, density, area, t_star_over_tb, and safety must be positive")
    n_e = int(n_e)
    x, mass, h_e, mass_regular, mass_face = _build_positions_and_mass(n_e, length, density, area)
    v = edot * x
    E0_discrete = float(np.sum(mass * v * v)) * 0.5
    E0_continuum = 0.5 * density * area * edot ** 2 * length ** 3 / 12.0
    momentum = float(np.sum(mass * v))

    k_e = youngs_modulus * area / h_e
    k_tilde = alpha * youngs_modulus / h_e
    k_tilde_A = k_tilde * area

    ratio_regular = 4.0 * k_e / mass_regular
    ratio_end = 2.0 * k_e / mass_face
    ratio_interface = (2.0 * k_e + 2.0 * k_tilde_A) / mass_face
    omega2_max = max(ratio_regular, ratio_end, ratio_interface)
    omega_max = float(np.sqrt(omega2_max))

    wave_speed = float(np.sqrt(youngs_modulus / density))
    bar_period = 2.0 * length / wave_speed
    t_final = t_star_over_tb * bar_period
    dt_setup = safety * 2.0 / omega_max
    N_steps = int(round(t_final / dt_setup))
    dt_prime = t_final / N_steps
    convexity_margin = dt_prime * omega_max / 2.0

    return {
        "E0_discrete": E0_discrete,
        "E0_continuum": E0_continuum,
        "momentum": momentum,
        "k_tilde": k_tilde,
        "k_tilde_A": k_tilde_A,
        "k_bulk_element": k_e,
        "omega2_max": omega2_max,
        "omega_max": omega_max,
        "dt_setup": dt_setup,
        "N_steps": N_steps,
        "dt_prime": dt_prime,
        "convexity_margin": convexity_margin,
        "bar_period": bar_period,
        "wave_speed": wave_speed,
        "t_final": t_final,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 3 cases: standalone benchmark checks for both initial-energy
    estimates, the Gershgorin bound, the setup and used steps, the step
    count, and the convexity margin; a boundary case at a minimal mesh; and
    two edge cases (invalid n_e, invalid safety) that raise ValueError.
    """
    return [
        {
            # benchmark config: discrete initial kinetic energy.
            "setup": "import numpy as np",
            "call": "float(initial_state_energy(500)['E0_discrete'])",
            "gold_call": "float(_oracle_initial_state_energy(500)['E0_discrete'])",
        },
        {
            # benchmark config: continuum initial kinetic-energy estimate.
            "setup": "import numpy as np",
            "call": "float(initial_state_energy(500)['E0_continuum'])",
            "gold_call": "float(_oracle_initial_state_energy(500)['E0_continuum'])",
        },
        {
            # benchmark config: full-row Gershgorin frequency bound.
            "setup": "import numpy as np",
            "call": "float(initial_state_energy(500)['omega2_max'])",
            "gold_call": "float(_oracle_initial_state_energy(500)['omega2_max'])",
        },
        {
            # benchmark config: stability-limited setup step.
            "setup": "import numpy as np",
            "call": "float(initial_state_energy(500)['dt_setup'] * 1e12)",
            "gold_call": "float(_oracle_initial_state_energy(500)['dt_setup'] * 1e12)",
        },
        {
            # benchmark config: rounded number of physical time steps.
            "setup": "import numpy as np",
            "call": "float(initial_state_energy(500)['N_steps'])",
            "gold_call": "float(_oracle_initial_state_energy(500)['N_steps'])",
        },
        {
            # benchmark config: time step used by the physical march.
            "setup": "import numpy as np",
            "call": "float(initial_state_energy(500)['dt_prime'] * 1e12)",
            "gold_call": "float(_oracle_initial_state_energy(500)['dt_prime'] * 1e12)",
        },
        {
            # benchmark config: convexity margin of the used time step.
            "setup": "import numpy as np",
            "call": "float(initial_state_energy(500)['convexity_margin'])",
            "gold_call": "float(_oracle_initial_state_energy(500)['convexity_margin'])",
        },
        {
            # boundary: minimal mesh, check the interface cap ratio and that
            # momentum stays at machine-precision zero.
            "setup": "import numpy as np",
            "call": (
                "float(initial_state_energy(8)['k_tilde'] / initial_state_energy(8)['k_bulk_element']) "
                "+ float(abs(initial_state_energy(8)['momentum']) < 1e-8)"
            ),
            "gold_call": (
                "float(_oracle_initial_state_energy(8)['k_tilde'] / _oracle_initial_state_energy(8)['k_bulk_element']) "
                "+ float(abs(_oracle_initial_state_energy(8)['momentum']) < 1e-8)"
            ),
        },
        {
            # edge: odd n_e is invalid
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: initial_state_energy(9))",
            "gold_call": "_guard(lambda: _oracle_initial_state_energy(9))",
        },
        {
            # edge: a non-positive safety factor is invalid
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: initial_state_energy(8, safety=0.0))",
            "gold_call": "_guard(lambda: _oracle_initial_state_energy(8, safety=0.0))",
        },
    ]
