"""
Return the room-temperature dopant activation ratio, the hole density divided by the total dopant concentration, of an acceptor-doped polycrystalline film after linear cooling with species-by-species defect freeze-in.

The activation ratio is the figure of merit for p-type doping: it measures how much of the incorporated dopant ends up as free holes once compensating native defects, dopant-related donors and complexes have been accounted for. Under finite-rate cooling it depends on the defect energetics, the migration barriers, the cooling rate and the grain size together, and it is the quantity compared with Hall measurements on doped films and crystals.

Returns
-------
float, the hole density at t_min divided by the total dopant concentration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dopant_activation_ratio(process: dict, host: dict, defects: dict) -> float:
    '''Hole density at the end of cooling divided by the total dopant concentration.

    Parameters
    ----------
    process : dict
        Cooling and doping conditions as documented for sequential_freeze_in_totals; the hole
        density is evaluated at "t_min".
    host : dict
        Host description as documented for solve_partial_equilibrium.
    defects : dict
        Defect table in the form the problem gives it, with per-defect entries of length n_def:
        "charges" (charge states, strictly decreasing in unit steps and containing 0),
        "e_neutral" (neutral formation energy, eV), "levels" (transition levels between
        consecutive charge states, eV), and "site", "n_added", "n_dopant" and "em" as documented
        for solve_partial_equilibrium.

    Returns
    -------
    ratio : float
        Dimensionless activation ratio p / dopant_total at t_min, where the defect totals are
        those retained by sequential freeze-in, their charge states follow the Fermi level at
        t_min, and the state is electrically neutral.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If the process parameters are invalid (see freeze_in_temperature).
    '''
    return ratio

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_dopant_activation_ratio(process: dict, host: dict, defects: dict) -> float:
    cs_def, cs_q, cs_e = [], [], []
    for j, (ch, e0, lv) in enumerate(zip(defects["charges"], defects["e_neutral"], defects["levels"])):
        e = _oracle_charge_state_formation_energies(e0, np.asarray(ch), np.asarray(lv, dtype=float))
        cs_def += [j] * len(ch)
        cs_q += list(ch)
        cs_e += list(e)
    full = {k: np.asarray(defects[k]) for k in ("site", "n_added", "n_dopant", "em")}
    full.update(cs_def=np.array(cs_def), cs_q=np.array(cs_q), cs_e=np.array(cs_e))
    totals = _oracle_sequential_freeze_in_totals(process, host, full)
    state = _oracle_solve_partial_equilibrium(process["t_min"], totals, process["dopant_total"], host, full)
    return float(state[2] / float(process["dopant_total"]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
host = {"eg0": 1.583, "varshni_alpha": 3.4e-4, "varshni_beta": 128.0, "f_cb": 0.78,
        "nc_ref": 7.9e17, "nv_ref": 1.69e19, "t_ref": 296.0, "hw0": 0.0119}
import copy
defects = {
    "charges": [[2, 1, 0], [0, -1, -2], [0, -1], [1, 0], [1, 0]],
    "e_neutral": np.array([3.15, 2.61, 1.21, 1.30, 1.84]),
    "levels": [[1.37, 1.49], [0.31, 0.74], [0.13], [1.29], [1.18]],
    "site": np.array([1.484e22, 1.484e22, 1.484e22, 1.484e22, 5.936e22]),
    "n_added": np.array([1, -1, 0, 0, 1]),
    "n_dopant": np.array([0, 0, 1, 1, 1]),
    "em": np.array([0.97, 1.43, 1.62, 1.77, 1.86]),
}
def fresh():
    return copy.deepcopy(defects)
"""
    return [
        # --- Valid: few-micron grains at a faster cooling rate ---
        {
            "setup": base + """
process = {"t_max": 1123.0, "t_min": 296.0, "gamma": 2.9, "grain_size": 3.8e-4, "d0": 0.25, "dopant_total": 4.3e16}
""",
            "call": "dopant_activation_ratio(dict(process), dict(host), fresh())",
            "gold_call": "_oracle_dopant_activation_ratio(dict(process), dict(host), fresh())",
            "tol": 1e-6,
        },
        # --- Valid: finer grains and lighter doping ---
        {
            "setup": base + """
process = {"t_max": 1123.0, "t_min": 296.0, "gamma": 0.73, "grain_size": 1.1e-4, "d0": 0.25, "dopant_total": 8.8e15}
""",
            "call": "dopant_activation_ratio(dict(process), dict(host), fresh())",
            "gold_call": "_oracle_dopant_activation_ratio(dict(process), dict(host), fresh())",
            "tol": 1e-6,
        },
        # --- Boundary: bulk-like grains (every species frozen at the start of cooling) ---
        {
            "setup": base + """
process = {"t_max": 1123.0, "t_min": 296.0, "gamma": 0.73, "grain_size": 0.62, "d0": 0.25, "dopant_total": 4.3e16}
""",
            "call": "np.log10(dopant_activation_ratio(dict(process), dict(host), fresh()))",
            "gold_call": "np.log10(_oracle_dopant_activation_ratio(dict(process), dict(host), fresh()))",
            "tol": 1e-6,
        },
        # --- Edge: no compensating interstitial or complex (acceptor, antisite and vacancy only) ---
        {
            "setup": base + """
keep = [1, 2, 3]
reduced = {k: ([copy.deepcopy(defects[k][i]) for i in keep] if isinstance(defects[k], list) else defects[k][keep].copy()) for k in defects}
process = {"t_max": 1123.0, "t_min": 296.0, "gamma": 0.73, "grain_size": 3.8e-4, "d0": 0.25, "dopant_total": 4.3e16}
""",
            "call": "dopant_activation_ratio(dict(process), dict(host), copy.deepcopy(reduced))",
            "gold_call": "_oracle_dopant_activation_ratio(dict(process), dict(host), copy.deepcopy(reduced))",
            "tol": 1e-6,
        },
    ]
