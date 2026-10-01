"""
Step 07: Predicted partner level after refitting the scale (orchestrator). Energy of a partner bound state predicted after the potential has been scaled to reproduce one observed state.

In spectroscopic work on weakly bound complexes a model interaction is often adjusted so that one measured level comes out right, and the adjusted model is then used to predict levels that have not been assigned. The prediction is only as good as the labelling: fitting the wrong member of a close pair of states produces a different scaling factor and moves the predicted partner by far more than any propagation error, so the label each state carries has to survive the whole search. The partner can lie below or above the fitted state, and no energy window for it is known in advance, so a bracket has to be established from the physics of the rescaled model before the state can be converged.

Returns
-------
float, energy in cm^-1 of state m_pred after scaling the interaction so that state m_fit lies at E_obs
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def predicted_partner_energy(params: dict, m_fit: int, E_obs: float, m_pred: int, lam_low: float, lam_high: float,
                             R_min: float, R_max: float) -> float:
    '''Converged energy of state m_pred after scaling the interaction so that state m_fit lies at E_obs.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix; 'scale' is ignored.
    m_fit : int
        Label of the fitted state, integer >= 1.
    E_obs : float
        Observed energy of the fitted state in cm^-1, finite.
    m_pred : int
        Label of the predicted state, integer >= 1 and different from m_fit.
    lam_low : float
        Lower end of the search interval for the scaling factor, finite and > 0.
    lam_high : float
        Upper end of the search interval, finite and > lam_low.
    R_min : float
        Inner wall in angstrom, finite, > 0 and < 4.0; every radial channel function vanishes there.
    R_max : float
        Outer wall in angstrom, finite and > 4.0; every radial channel function vanishes there.

    Returns
    -------
    E_pred : float
        With lam_star the exact scaling factor of scale_for_level(params, m_fit, E_obs, lam_low, lam_high, R_min, R_max,
        tol), the exact energy in cm^-1 of state m_pred (labels as in bound_state_energies) of the walled problem for
        params with 'scale' set to lam_star, returned within 5e-9 cm^-1. Propagations are matched at 4.0 angstrom,
        the matching distance this problem fixes, so the walls must enclose it.

    Raises
    ------
    ValueError
        If m_fit or m_pred is not an integer >= 1, m_pred equals m_fit, E_obs, lam_low, lam_high, R_min or R_max is not
        finite, lam_low <= 0, lam_high <= lam_low, R_min <= 0, R_max <= R_min, 4.0 angstrom does not lie strictly
        between R_min and R_max, or the interval does not bracket state m_fit at E_obs.
    '''
    return E_pred

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_predicted_partner_energy(params: dict, m_fit: int, E_obs: float, m_pred: int, lam_low: float,
                                     lam_high: float, R_min: float, R_max: float) -> float:
    import numpy as np
    if isinstance(m_pred, bool) or int(m_pred) != m_pred or m_pred < 1 or m_pred == m_fit:
        raise ValueError("m_pred must be an integer >= 1 different from m_fit")
    m_pred = int(m_pred)
    lam_star = _oracle_scale_for_level(params, m_fit, E_obs, lam_low, lam_high, R_min, R_max, 1e-10)
    scaled = dict(params, scale=lam_star)
    setting = (scaled, 0.0, float(R_min), float(R_max), 0.005, "E")
    # no state lies below the lowest eigenvalue of the effective potential matrix anywhere on the range
    floor = min(np.linalg.eigvalsh(_oracle_channel_matrix(R, scaled))[0]
                for R in np.linspace(R_min, R_max, 4001)) * (16.8576292 / params["mu"])
    E_low = floor - 0.05 * abs(floor) - 1.0
    n_low = _node_count(setting, E_low)
    if n_low >= m_pred:
        raise ValueError("nodes appear below the bottom of the potential")
    E_high = float(E_obs) + 1.0
    while _node_count(setting, E_high) < m_pred:
        E_high = E_high + max(10.0, abs(E_high))
    # narrow the window on the node count alone until it holds the predicted state, then hand it to step 05
    lo, hi, n_lo = E_low, E_high, n_low
    for _ in range(200):
        if _node_count(setting, hi) - n_lo == 1:
            break
        mid = 0.5 * (lo + hi)
        n_mid = _node_count(setting, mid)
        if n_mid >= m_pred:
            hi = mid
        else:
            lo, n_lo = mid, n_mid
    energies = _oracle_bound_state_energies(scaled, lo, hi, R_min, R_max, 1e-9)
    return float(energies[m_pred - n_lo - 1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "P = {'mu': 19.2, 'B': 10.4, 'eps': 180.0, 'Rm': 3.8, 'a': (1.0, 0.25, 0.35), 'b': (1.0, 0.15, 0.30),"
            " 'jmax': 6, 'J': 0, 'parity': 1, 'scale': 1.0}\n")
    return [
        # --- Normal: benchmark, state 17 fitted to -30 cm^-1 and its lower partner predicted ---
        {"setup": base, "call": "float(predicted_partner_energy(dict(P), 17, -30.0, 16, 1.0, 1.4, 2.5, 15.0))",
         "gold_call": "float(_oracle_predicted_partner_energy(dict(P), 17, -30.0, 16, 1.0, 1.4, 2.5, 15.0))", "tol": 1e-9},
        # --- Normal: the other member of the pair fitted instead, upper partner predicted ---
        {"setup": base, "call": "float(predicted_partner_energy(dict(P), 16, -30.0, 17, 1.0, 1.4, 2.8, 15.0))",
         "gold_call": "float(_oracle_predicted_partner_energy(dict(P), 16, -30.0, 17, 1.0, 1.4, 2.8, 15.0))", "tol": 1e-9},
        # --- Normal: J = 3, deep fitted state and a partner far above it ---
        {"setup": base, "call": "float(predicted_partner_energy(dict(P, J=3, parity=1), 3, -100.0, 12, 1.0, 1.3, 2.8, 15.0))",
         "gold_call": "float(_oracle_predicted_partner_energy(dict(P, J=3, parity=1), 3, -100.0, 12, 1.0, 1.3, 2.8, 15.0))",
         "tol": 1e-9},
        # --- Boundary: heavy J = 5 odd-parity complex, its lowest state predicted from state 130 ---
        {"setup": base + "Q = dict(P, mu=40.0, B=1.1, jmax=5, J=5, parity=-1)\n",
         "call": "float(predicted_partner_energy(dict(Q), 130, -45.0, 1, 1.0, 1.1, 2.9, 13.0))",
         "gold_call": "float(_oracle_predicted_partner_energy(dict(Q), 130, -45.0, 1, 1.0, 1.1, 2.9, 13.0))", "tol": 1e-9},
        # --- Boundary: near-degenerate pair at an avoided crossing ---
        {"setup": base + "Q = dict(P, a=(1.0, 4e-4, 6e-4), b=(1.0, 2e-4, 4e-4), jmax=4, B=10.0331)\n",
         "call": "float(predicted_partner_energy(dict(Q), 11, -52.5, 10, 0.98, 1.02, 2.6, 12.0))",
         "gold_call": "float(_oracle_predicted_partner_energy(dict(Q), 11, -52.5, 10, 0.98, 1.02, 2.6, 12.0))", "tol": 1e-9},
        # --- Edge: J = 1 odd parity for a larger complex whose well sits near 5.2 angstrom, partner below the fit ---
        {"setup": base + "Q = dict(P, Rm=5.2, eps=140.0, mu=24.0, B=6.5, J=1, parity=-1, jmax=5)\n",
         "call": "float(predicted_partner_energy(dict(Q), 12, -85.0, 2, 1.0, 1.1, 3.6, 16.0))",
         "gold_call": "float(_oracle_predicted_partner_energy(dict(Q), 12, -85.0, 2, 1.0, 1.1, 3.6, 16.0))", "tol": 1e-9},
        # --- Error: the walls do not enclose the matching distance ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(dict(P), 17, -30.0, 16, 1.0, 1.4, 4.1, 15.0)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(predicted_partner_energy)", "gold_call": "_probe(_oracle_predicted_partner_energy)"},
        # --- Error: the predicted label equals the fitted label ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(dict(P), 17, -30.0, 17, 1.0, 1.4, 2.5, 15.0)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(predicted_partner_energy)", "gold_call": "_probe(_oracle_predicted_partner_energy)"},
    ]
