"""
Step 04: Transition state and rise of the mixing angle along a constrained path. Highest point of the correlated pi energy along a symmetry-constrained electrocyclic path and the change of the frontier mixing angle.

Following a ring closure under a conserved twofold axis or mirror plane keeps the electronic states in fixed symmetry blocks, so a thermally forbidden pathway cannot avoid the crossing of a bonding and an antibonding frontier orbital. In a correlated description the crossing shows up as a smooth but strong change of the ground state, from a closed-shell reactant through an open-shell singlet diradicaloid region towards the product. The top of the energy profile along such a constrained path serves as the transition state of the model, and comparing the frontier mixing angle there with its value in the open reactant measures how much bonding character is lost on the way. On an allowed path the angle barely changes, on a forbidden path it rises by tens of degrees, and the change has been proposed as a quantitative, geometric measure of forbiddenness.

Returns
-------
numpy.ndarray [s_ts, energy rise in eV, Theta at s = 0, Theta at s_ts, rise of Theta] along the constrained path
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def path_transition_state(n_sites: int, mode: str, U: float, params: dict) -> "np.ndarray":
    '''Location, energy rise and frontier mixing angles of the energy maximum along a constrained ring-closure path.

    Parameters
    ----------
    n_sites : int
        Number of pi sites, even integer >= 4 and <= 10.
    mode : str
        'con' or 'dis', as in pi_hamiltonian.
    U : float
        On-site repulsion in eV, finite and > 0.
    params : dict
        Model parameters as in pi_hamiltonian.

    Returns
    -------
    ts : np.ndarray
        Real array (s_ts, dE, Theta_0, Theta_ts, dTheta). With E(s) the lowest singlet energy of
        pi_hamiltonian(n_sites, s, mode, params) and U (singlet_ground_state), s_ts is the progress in (0, 1) at which E
        attains its maximum over [0, 1], dE = E(s_ts) - E(0) in eV, Theta_0 and Theta_ts are the frontier mixing angles in
        degrees (frontier_mixing_descriptors) at s = 0 and s = s_ts, and dTheta = Theta_ts - Theta_0. Inputs are such that
        the maximum over [0, 1] is attained at a single interior point, where E is smooth, and any other local maximum
        lies at least 0.05 eV lower. s_ts is returned within 1e-8 and the angles within 1e-6 degrees.

    Raises
    ------
    ValueError
        If n_sites or mode violates the conditions of pi_hamiltonian, n_sites > 10, U is not finite or U <= 0, or the
        maximum of E over [0, 1] lies at s = 0 or s = 1.
    '''
    return ts

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _path_energy(n_sites, s, mode, U, params):
    return _oracle_singlet_ground_state(_oracle_pi_hamiltonian(n_sites, s, mode, params), U)[0]


def _oracle_path_transition_state(n_sites: int, mode: str, U: float, params: dict) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import minimize_scalar
    if not np.isfinite(U) or U <= 0.0:
        raise ValueError("U must be finite and > 0")
    if isinstance(n_sites, bool) or int(n_sites) != n_sites or n_sites > 10:
        raise ValueError("n_sites must be an even integer between 4 and 10")
    grid = np.linspace(0.0, 1.0, 41)
    E = np.array([_path_energy(n_sites, s, mode, U, params) for s in grid])
    k = int(np.argmax(E))
    if k == 0 or k == len(grid) - 1:
        raise ValueError("the energy profile has no interior maximum")
    res = minimize_scalar(lambda s: -_path_energy(n_sites, s, mode, U, params),
                          bounds=(grid[k - 1], grid[k + 1]), method="bounded", options={"xatol": 1e-11})
    s_ts = float(res.x)
    e_ts = _path_energy(n_sites, s_ts, mode, U, params)
    th_0 = _oracle_frontier_mixing_descriptors(_oracle_pi_hamiltonian(n_sites, 0.0, mode, params), U)[0]
    th_ts = _oracle_frontier_mixing_descriptors(_oracle_pi_hamiltonian(n_sites, s_ts, mode, params), U)[0]
    return np.array([s_ts, e_ts - E[0], th_0, th_ts, th_ts - th_0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "P = {'beta_double': -2.8, 'beta_single': -2.2, 'r_open': 3.0, 'r_closed': 1.54,"
            " 'tau_pi': -2.0, 'zeta_pi': 1.4, 'tau_sigma': 3.2, 'zeta_sigma': 1.2}\n")
    return [
        # --- Normal: hexatriene, forbidden conrotatory path ---
        {"setup": base, "call": "np.asarray(path_transition_state(6, 'con', 3.6, dict(P)), dtype=float)",
         "gold_call": "np.asarray(_oracle_path_transition_state(6, 'con', 3.6, dict(P)), dtype=float)", "tol": 1e-5},
        # --- Normal: hexatriene, allowed disrotatory path ---
        {"setup": base, "call": "np.asarray(path_transition_state(6, 'dis', 3.6, dict(P)), dtype=float)",
         "gold_call": "np.asarray(_oracle_path_transition_state(6, 'dis', 3.6, dict(P)), dtype=float)", "tol": 1e-5},
        # --- Normal: butadiene, forbidden disrotatory path with different couplings ---
        {"setup": base + "Q = dict(P, beta_double=-2.6, tau_sigma=2.9, zeta_pi=1.1)\n",
         "call": "np.asarray(path_transition_state(4, 'dis', 2.5, dict(Q)), dtype=float)",
         "gold_call": "np.asarray(_oracle_path_transition_state(4, 'dis', 2.5, dict(Q)), dtype=float)", "tol": 1e-5},
        # --- Boundary: butadiene, allowed conrotatory path, weak repulsion ---
        {"setup": base, "call": "np.asarray(path_transition_state(4, 'con', 0.8, dict(P)), dtype=float)",
         "gold_call": "np.asarray(_oracle_path_transition_state(4, 'con', 0.8, dict(P)), dtype=float)", "tol": 1e-5},
        # --- Boundary: hexatriene, forbidden conrotatory path at strong repulsion with slower terminal decay ---
        {"setup": base + "Q = dict(P, zeta_sigma=0.9, r_open=3.3)\n",
         "call": "np.asarray(path_transition_state(6, 'con', 7.0, dict(Q)), dtype=float)",
         "gold_call": "np.asarray(_oracle_path_transition_state(6, 'con', 7.0, dict(Q)), dtype=float)", "tol": 1e-5},
        # --- Error: a strong sigma coupling makes the energy fall all the way, no interior maximum ---
        {"setup": base + "Q = dict(P, tau_sigma=40.0, zeta_sigma=0.1, tau_pi=-8.0, zeta_pi=0.1)\n"
                  "def _probe(fn):\n    try:\n        fn(6, 'dis', 3.0, dict(Q))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(path_transition_state)", "gold_call": "_probe(_oracle_path_transition_state)"},
    ]
