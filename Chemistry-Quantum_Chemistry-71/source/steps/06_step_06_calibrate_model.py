"""
Step 06: Two-parameter calibration on the rises of the mixing angle. Two-parameter calibration of a correlated pi model against the mixing-angle changes of a forbidden and an allowed pathway.

A minimal correlated model of a ring closure has two quantities that control how much bonding character is lost on the way to the transition state: the strength of the on-site repulsion, which decides how much static correlation the open reactant already carries, and the resonance integral of the formal single bonds, which sets how strongly the two halves of the pi system talk to each other while the terminal orbitals rotate. Neither can be fixed by one reference number alone, because raising the repulsion and weakening the single bonds pull the change of the mixing angle in different directions on the two stereochemical pathways: on the forbidden path the change first grows and then falls as the single bonds weaken, while on the allowed path it rises steadily. Matching the changes reported for both pathways of the same molecule therefore determines the pair, and it is a genuinely two-dimensional root problem in which every residual evaluation needs two transition states, each located on an exactly diagonalised path.

Returns
-------
numpy.ndarray [U_star in eV, beta_single_star in eV], the pair reproducing the conrotatory and disrotatory target rises of the frontier mixing angle
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def calibrate_model(n_sites: int, target_con: float, target_dis: float, params: dict, U_bounds: tuple,
                    beta_bounds: tuple) -> "np.ndarray":
    '''On-site repulsion and single-bond resonance integral that reproduce the mixing-angle changes of both pathways.

    Parameters
    ----------
    n_sites : int
        Number of pi sites, as in path_transition_state.
    target_con : float
        Target rise of the frontier mixing angle in degrees on the conrotatory path, finite.
    target_dis : float
        Target rise of the frontier mixing angle in degrees on the disrotatory path, finite.
    params : dict
        Model parameters as in pi_hamiltonian; the value stored under 'beta_single' is ignored and replaced by the
        trial value.
    U_bounds : tuple
        (U_low, U_high) in eV, finite with 0 < U_low < U_high.
    beta_bounds : tuple
        (beta_low, beta_high) in eV for the single-bond resonance integral, finite with beta_low < beta_high < 0.

    Returns
    -------
    fit : np.ndarray
        Real array (U_star, beta_single_star). At these values dTheta of path_transition_state(n_sites, 'con', U_star,
        params with beta_single = beta_single_star) equals target_con and dTheta of the same call with 'dis' equals
        target_dis. Inputs are such that exactly one such pair exists in the open rectangle U_bounds x beta_bounds, the
        conditions of path_transition_state hold there, and the pair is returned within 1e-6 of its exact value.

    Raises
    ------
    ValueError
        If target_con or target_dis is not finite, the bounds are not finite, U_low <= 0, U_high <= U_low,
        beta_high <= beta_low, beta_high >= 0, or the search leaves the rectangle without converging.
    '''
    return fit

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _fit_box_map(z, bounds):
    import numpy as np
    (u_low, u_high), (b_low, b_high) = bounds
    return (u_low + (u_high - u_low) / (1.0 + np.exp(-z[0])), b_low + (b_high - b_low) / (1.0 + np.exp(-z[1])))


def _fit_residual(z, n_sites, targets, params, bounds):
    U, b = _fit_box_map(z, bounds)
    p = dict(params, beta_single=b)
    return [_oracle_path_transition_state(n_sites, "con", U, p)[4] - targets[0],
            _oracle_path_transition_state(n_sites, "dis", U, p)[4] - targets[1]]


def _oracle_calibrate_model(n_sites: int, target_con: float, target_dis: float, params: dict, U_bounds: tuple,
                            beta_bounds: tuple) -> "np.ndarray":
    import numpy as np
    from scipy.optimize import fsolve
    U_low, U_high = float(U_bounds[0]), float(U_bounds[1])
    b_low, b_high = float(beta_bounds[0]), float(beta_bounds[1])
    if not all(np.isfinite([target_con, target_dis, U_low, U_high, b_low, b_high])):
        raise ValueError("targets and bounds must be finite")
    if U_low <= 0.0 or U_high <= U_low or b_high <= b_low or b_high >= 0.0:
        raise ValueError("need 0 < U_low < U_high and beta_low < beta_high < 0")
    bounds = ((U_low, U_high), (b_low, b_high))
    sol, info, ier, msg = fsolve(_fit_residual, [0.0, 0.0],
                                 args=(n_sites, (target_con, target_dis), params, bounds),
                                 full_output=True, xtol=1e-13, epsfcn=1e-4)
    U, b = _fit_box_map(sol, bounds)
    if max(abs(v) for v in info["fvec"]) > 1e-5 or not (U_low < U < U_high and b_low < b < b_high):
        raise ValueError("the search does not converge inside the rectangle")
    return np.array([U, b])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "P = {'beta_double': -2.8, 'beta_single': -2.2, 'r_open': 3.0, 'r_closed': 1.54,"
            " 'tau_pi': -2.0, 'zeta_pi': 1.4, 'tau_sigma': 3.2, 'zeta_sigma': 1.2}\n")
    return [
        # --- Normal: hexatriene matched to a 27 degree forbidden and 5 degree allowed rise ---
        {"setup": base, "call": "np.asarray(calibrate_model(6, 27.0, 5.0, dict(P), (3.0, 8.0), (-2.4, -1.2)), dtype=float)",
         "gold_call": "np.asarray(_oracle_calibrate_model(6, 27.0, 5.0, dict(P), (3.0, 8.0), (-2.4, -1.2)), dtype=float)",
         "tol": 1e-5},
        # --- Normal: butadiene, where the forbidden path is the disrotatory one ---
        {"setup": base, "call": "np.asarray(calibrate_model(4, 3.0, 24.0, dict(P), (2.0, 7.0), (-2.6, -1.4)), dtype=float)",
         "gold_call": "np.asarray(_oracle_calibrate_model(4, 3.0, 24.0, dict(P), (2.0, 7.0), (-2.6, -1.4)), dtype=float)",
         "tol": 1e-5},
        # --- Boundary: butadiene with a slower sigma decay and a wider open chain ---
        {"setup": base + "Q = dict(P, zeta_sigma=0.9, r_open=3.3)\n",
         "call": "np.asarray(calibrate_model(4, 3.5, 22.0, dict(Q), (2.0, 7.0), (-2.6, -1.4)), dtype=float)",
         "gold_call": "np.asarray(_oracle_calibrate_model(4, 3.5, 22.0, dict(Q), (2.0, 7.0), (-2.6, -1.4)), dtype=float)",
         "tol": 1e-5},
        # --- Error: positive upper bound on the single-bond resonance integral ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(6, 27.0, 5.0, dict(P), (3.0, 8.0), (-2.4, 0.5))\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(calibrate_model)", "gold_call": "_probe(_oracle_calibrate_model)"},
    ]
