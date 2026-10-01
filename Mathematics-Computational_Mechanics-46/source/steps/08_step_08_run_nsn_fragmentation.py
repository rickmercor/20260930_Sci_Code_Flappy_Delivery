"""
Chains the geometry, initial state, cohesive law, predictor, contact solve, and corrector physics of the preceding steps, step by step, into the complete free-expansion fragmentation run at the requested configuration, cross-checks the result against the independently-expressed full march, and extracts the single requested observable: the right-end displacement at the horizon, normalized by the bar length.

Result extraction and verification.




Running the geometry, initial-state, cohesive-law, predictor, contact-solve, and corrector steps in sequence for every step of the march, updating the displacement, velocity, and acceleration each time, is exactly the same computation as the fixed time-march loop, expressed one physics piece at a time instead of as a single fused loop; agreement between the two expressions to machine precision is itself part of the run's verification, the same kind of cross-implementation check the golden verification block relies on. The requested observable is the axial displacement of the bar's right-end node at the requested horizon, normalized by the bar length; because the march is fully deterministic given its configuration, a single run at the pinned configuration produces this value with no additional post-processing beyond reading off the last degree of freedom and dividing by the bar length. Running the whole pipeline with no arguments reproduces the benchmark configuration exactly, which is what makes the normalized end displacement a fixed, reproducible number rather than one more input a caller has to supply.

Returns
-------
u_end_over_L : float -- the right-end displacement at the horizon, normalized by the bar length.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_nsn_fragmentation(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, d0: float = 1e-6, sigma_c: float = 262e6, Gc: float = 50.0, t_star_over_tb: float = 0.42, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, safety: float = 0.99) -> float:
    """Run the full NSN fragmentation pipeline and return the normalized
    end displacement.

    Parameters
    ----------
    n_e : int
        Number of uniform two-node elements (must be an even integer >= 2).
    edot : float
        Nominal applied strain rate, in 1/s (must be > 0).
    alpha : float
        Interface stiffness cap ratio, dimensionless (must be > 0).
    d0 : float
        Initial damage of every interface (must lie in (0, 1)).
    sigma_c : float
        Cohesive strength, in pascals (must be > 0).
    Gc : float
        Specific fracture energy, in J/m^2 (must be > 0).
    t_star_over_tb : float
        Requested horizon as a fraction of the bar period (must be > 0).
    length : float
        Total bar length, in metres (must be > 0).
    youngs_modulus : float
        Bulk Young's modulus, in pascals (must be > 0).
    density : float
        Bulk mass density, in kg/m^3 (must be > 0).
    area : float
        Cross-sectional area, in m^2 (must be > 0).
    safety : float
        Safety factor applied to the Gershgorin critical step, in (0, 1].

    Returns
    -------
    u_end_over_L : float
        The right-end displacement at the horizon, normalized by length.

    Raises
    ------
    ValueError
        If n_e is not an even integer >= 2, if edot, alpha, sigma_c, Gc,
        t_star_over_tb, length, youngs_modulus, density, area, or safety is
        not a positive real number, if d0 does not lie in (0, 1), or if the
        step-by-step chain disagrees with the fused march beyond machine
        tolerance.
    """
    return u_end_over_L

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _build_positions_and_mass(n_e, length, density, area):
    """Duplicated-node positions and lumped masses (mirrors the DOF map of
    _oracle_bar_model_setup, adding the node positions it does not return)."""
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


def _oracle_run_nsn_fragmentation(n_e: int = 500, edot: float = 5e4, alpha: float = 10.0, d0: float = 1e-6, sigma_c: float = 262e6, Gc: float = 50.0, t_star_over_tb: float = 0.42, length: float = 1e-3, youngs_modulus: float = 370e9, density: float = 3900.0, area: float = 6.45e-4, safety: float = 0.99) -> float:
    """Reference implementation of run_nsn_fragmentation: chains
    _oracle_bar_model_setup, _oracle_initial_state_energy,
    _oracle_cohesive_law_state, _oracle_predictor_forces,
    _oracle_contact_qp_solve, and _oracle_nonsmooth_corrector step by step,
    then cross-checks the result against _oracle_nsn_time_march's fused
    march before returning it."""
    if not isinstance(n_e, (int, np.integer)) or int(n_e) < 2 or int(n_e) % 2 != 0:
        raise ValueError("n_e must be an even integer >= 2")
    if not (edot > 0.0 and alpha > 0.0 and sigma_c > 0.0 and Gc > 0.0 and t_star_over_tb > 0.0
            and length > 0.0 and youngs_modulus > 0.0 and density > 0.0 and area > 0.0
            and 0.0 < safety <= 1.0):
        raise ValueError("edot, alpha, sigma_c, Gc, t_star_over_tb, length, youngs_modulus, density, area, and safety must be positive")
    if not (0.0 < d0 < 1.0):
        raise ValueError("d0 must lie in (0, 1)")

    n_e = int(n_e)
    setup = _oracle_bar_model_setup(n_e, length, youngs_modulus, density, area)
    state0 = _oracle_initial_state_energy(n_e, edot, alpha, length, youngs_modulus, density, area, t_star_over_tb, safety)

    n_dof = setup["n_dof"]
    dof_minus, dof_plus = setup["dof_minus"], setup["dof_plus"]
    element_dofs, mass = setup["element_dofs"], setup["mass"]
    k_e = youngs_modulus * area / setup["h_e"]
    n_if = dof_minus.shape[0]

    N = state0["N_steps"]
    dtp = state0["dt_prime"]
    delta_c = 2.0 * Gc / sigma_c

    dmax_open = np.full(n_if, d0 * delta_c)
    u = np.zeros(n_dof)
    x, _mass_check, _h_e_check, _mr, _mf = _build_positions_and_mass(n_e, length, density, area)
    v = edot * x
    a = np.zeros(n_dof)

    for _ in range(N):
        dt = dtp
        ut_probe = u + dt * v + 0.5 * dt * dt * a
        delta_pred = ut_probe[dof_plus] - ut_probe[dof_minus]
        law = _oracle_cohesive_law_state(delta_pred, dmax_open, n_e, length, youngs_modulus, alpha, sigma_c, Gc)
        damage = law["damage"]
        dmax_open = law["dmax_open"]
        d_tilde = law["d_tilde"]

        r4 = _oracle_predictor_forces(u, v, a, damage, dof_minus, dof_plus, element_dofs, mass, k_e, d_tilde, dt, alpha, sigma_c, Gc, area)
        r5 = _oracle_contact_qp_solve(dof_minus, dof_plus, mass, r4["k_diag"], r4["k_up"], r4["u_tilde"], r4["v_free"], v, dt, n_dof)
        r6 = _oracle_nonsmooth_corrector(r4["u_tilde"], v, a, r5["p"], dof_minus, dof_plus, mass, r4["k_diag"], r4["k_up"], r4["fcap"], dt)
        u, v, a = r6["u_next"], r6["v_next"], r6["a_next"]

    u_end_over_L = float(u[-1] / length)

    fused = _oracle_nsn_time_march(n_e=n_e, edot=edot, alpha=alpha, d0=d0, sigma_c=sigma_c, Gc=Gc, t_star_over_tb=t_star_over_tb, length=length, youngs_modulus=youngs_modulus, density=density, area=area, safety=safety)
    fused_value = fused["u_end_over_L"]
    if not np.allclose(u_end_over_L, fused_value, rtol=1e-9, atol=1e-12):
        raise ValueError(f"step-by-step chain ({u_end_over_L}) disagrees with the fused march ({fused_value}) beyond tolerance")

    return u_end_over_L

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    Tiny meshes for every case except the last, which is the benchmark
    configuration needed for the no-args harvest of the final answer.
    >= 3 cases: a normal tiny-mesh scenario, a boundary scenario at a
    different alpha, the benchmark-config zero-args case, and two invalid-
    input edge cases.
    """
    shared_setup = (
        "import importlib\n"
        "import sys\n"
        "import numpy as np\n"
        "try:\n"
        "    _t46 = importlib.import_module('08_run_nsn_fragmentation')\n"
        "except ImportError:\n"
        "    _t46 = next((sys.modules[k] for k in sys.modules if '08_run_nsn_fragmentation' in k), None)\n"
        "if _t46 is None:\n"
        "    pass  # runner injected the module members directly into this namespace\n"
        "else:\n"
        "    run_nsn_fragmentation = _t46.run_nsn_fragmentation\n"
    )
    guard_def = (
        "def _guard(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 2\n"
        "    except Exception:\n"
        "        return 1"
    )
    return [
        {
            # normal: tiny mesh, non-default strain rate
            "setup": shared_setup,
            "call": "float(run_nsn_fragmentation(n_e=8, edot=2e4))",
            "gold_call": "float(_oracle_run_nsn_fragmentation(n_e=8, edot=2e4))",
        },
        {
            # boundary: tiny mesh, a much smaller cap ratio (the answer is
            # known to be sensitive to alpha; exercises the branch where
            # more interfaces reach the secant regime sooner)
            "setup": shared_setup,
            "call": "float(run_nsn_fragmentation(n_e=8, alpha=1.0))",
            "gold_call": "float(_oracle_run_nsn_fragmentation(n_e=8, alpha=1.0))",
        },
        {
            # benchmark config, zero required args: this is what
            # task_numbers.py's no-args harvest and the final-answer check
            # both read from.
            "setup": shared_setup,
            "call": "float(run_nsn_fragmentation())",
            "gold_call": "float(_oracle_run_nsn_fragmentation())",
        },
        {
            # edge: an odd n_e is invalid
            "setup": shared_setup + "\n" + guard_def,
            "call": "_guard(lambda: run_nsn_fragmentation(n_e=9))",
            "gold_call": "_guard(lambda: _oracle_run_nsn_fragmentation(n_e=9))",
        },
        {
            # edge: a non-positive cohesive strength is invalid
            "setup": shared_setup + "\n" + guard_def,
            "call": "_guard(lambda: run_nsn_fragmentation(n_e=8, sigma_c=0.0))",
            "gold_call": "_guard(lambda: _oracle_run_nsn_fragmentation(n_e=8, sigma_c=0.0))",
        },
    ]
