"""
Step 07: Spin-projected angle at the calibrated transition state (orchestrator). Mean-field diradical angle at the forbidden transition state of a pi model calibrated on two reported angle changes.

The orbital mixing angle of a forbidden transition state can be obtained in two ways: from the natural occupations of a correlated wavefunction, or from the much cheaper spin-projected broken-symmetry mean-field solution through Yamaguchi's diradical index. Comparing the two on a model whose repulsion and single-bond resonance integral have both been matched to a multiconfigurational study of the same molecule, on its forbidden and on its allowed pathway, shows how far the cheap estimate can be trusted once the correlated description is pinned down from both sides. The workflow chains every piece: the constrained pi Hamiltonian along each path, exact singlet energies and natural occupations, the transition state as the top of each profile, the two-dimensional fit of the model to the two reported angle changes, and finally the unrestricted mean-field solution at the forbidden transition state and its projected angle.

Returns
-------
float, spin-projected mixing angle in degrees at the transition state of the calibrated model
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def calibrated_puhf_angle(n_sites: int, mode: str, target_con: float, target_dis: float, params: dict,
                          U_bounds: tuple, beta_bounds: tuple) -> float:
    '''Spin-projected mean-field mixing angle at the transition state of a model fitted to two angle changes.

    Parameters
    ----------
    n_sites : int
        Number of pi sites, as in path_transition_state.
    mode : str
        'con' or 'dis', the path whose transition state is used.
    target_con : float
        Target rise of the frontier mixing angle in degrees on the conrotatory path, as in calibrate_model.
    target_dis : float
        Target rise of the frontier mixing angle in degrees on the disrotatory path, as in calibrate_model.
    params : dict
        Model parameters as in pi_hamiltonian; 'beta_single' is replaced by the fitted value.
    U_bounds : tuple
        (U_low, U_high) in eV, as in calibrate_model.
    beta_bounds : tuple
        (beta_low, beta_high) in eV, as in calibrate_model.

    Returns
    -------
    theta_puhf : float
        With (U_star, beta_star) = calibrate_model(n_sites, target_con, target_dis, params, U_bounds, beta_bounds),
        p the parameters with beta_single = beta_star, and s_ts the first entry of
        path_transition_state(n_sites, mode, U_star, p), the value Theta_puhf in degrees of
        yamaguchi_mixing_angle(pi_hamiltonian(n_sites, s_ts, mode, p), U_star), returned within 1e-4 degrees.

    Raises
    ------
    ValueError
        If mode is not 'con' or 'dis', or any input violates the conditions of calibrate_model.
    '''
    return theta_puhf

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_calibrated_puhf_angle(n_sites: int, mode: str, target_con: float, target_dis: float, params: dict,
                                  U_bounds: tuple, beta_bounds: tuple) -> float:
    if mode not in ("con", "dis"):
        raise ValueError("mode must be 'con' or 'dis'")
    U_star, beta_star = _oracle_calibrate_model(n_sites, target_con, target_dis, params, U_bounds, beta_bounds)
    fitted = dict(params, beta_single=beta_star)
    s_ts = _oracle_path_transition_state(n_sites, mode, U_star, fitted)[0]
    return float(_oracle_yamaguchi_mixing_angle(_oracle_pi_hamiltonian(n_sites, s_ts, mode, fitted), U_star)[3])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "P = {'beta_double': -2.8, 'beta_single': -2.2, 'r_open': 3.0, 'r_closed': 1.54,"
            " 'tau_pi': -2.0, 'zeta_pi': 1.4, 'tau_sigma': 3.2, 'zeta_sigma': 1.2}\n")
    return [
        # --- Normal: hexatriene, forbidden conrotatory transition state of the fitted model ---
        {"setup": base, "call": "float(calibrated_puhf_angle(6, 'con', 27.0, 5.0, dict(P), (3.0, 8.0), (-2.4, -1.2)))",
         "gold_call": "float(_oracle_calibrated_puhf_angle(6, 'con', 27.0, 5.0, dict(P), (3.0, 8.0), (-2.4, -1.2)))",
         "tol": 1e-4},
        # --- Normal: butadiene, allowed conrotatory transition state of the fitted model ---
        {"setup": base, "call": "float(calibrated_puhf_angle(4, 'con', 3.0, 24.0, dict(P), (2.0, 7.0), (-2.6, -1.4)))",
         "gold_call": "float(_oracle_calibrated_puhf_angle(4, 'con', 3.0, 24.0, dict(P), (2.0, 7.0), (-2.6, -1.4)))",
         "tol": 1e-4},
        # --- Boundary: butadiene, whose forbidden path is the disrotatory one ---
        {"setup": base, "call": "float(calibrated_puhf_angle(4, 'dis', 3.0, 24.0, dict(P), (2.0, 7.0), (-2.6, -1.4)))",
         "gold_call": "float(_oracle_calibrated_puhf_angle(4, 'dis', 3.0, 24.0, dict(P), (2.0, 7.0), (-2.6, -1.4)))",
         "tol": 1e-4},
        # --- Error: unknown rotation mode ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(6, 'mix', 27.0, 5.0, dict(P), (3.0, 8.0), (-2.4, -1.2))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(calibrated_puhf_angle)", "gold_call": "_probe(_oracle_calibrated_puhf_angle)"},
    ]
