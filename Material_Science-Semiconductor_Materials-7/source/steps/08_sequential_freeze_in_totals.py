"""
Return the total concentration of every defect species retained after a doped polycrystalline film is cooled at a constant rate, when each species freezes in at its own temperature.

Neither of the two textbook limits of defect chemistry describes a sample cooled at a finite rate: full equilibrium assumes every defect keeps adjusting down to room temperature, and a full quench assumes every defect is locked in at the highest temperature. In reality slow-diffusing species stop adjusting early, fast interstitials keep adjusting to much lower temperatures, and because all charged species are coupled through charge neutrality, the order in which species freeze in shapes the populations the remaining species reach.

Returns
-------
np.ndarray of float, the total concentration (cm^-3) of each defect retained at the end of cooling, in table order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sequential_freeze_in_totals(process: dict, host: dict, defects: dict) -> "np.ndarray":
    '''Frozen-in defect totals after linear cooling with species-by-species freeze-in.

    Parameters
    ----------
    process : dict
        Cooling and doping conditions with keys
        "t_max" : starting temperature (K), where every species that has not frozen is in
        equilibrium;
        "t_min" : final temperature of the cooling path (K), 0 < t_min < t_max;
        "gamma" : constant cooling rate (K/s);
        "grain_size" : grain size of the film (cm);
        "d0" : Arrhenius diffusion prefactor (cm^2/s), shared by all defects;
        "dopant_total" : total dopant concentration (cm^-3), fixed throughout.
    host : dict
        Host description as documented for solve_partial_equilibrium.
    defects : dict
        Defect table as documented for solve_partial_equilibrium, including the per-defect
        migration energies "em" (eV).

    Returns
    -------
    totals : np.ndarray
        Float array of length n_def: the total concentration (cm^-3) of each defect, in table
        order, retained at the end of the cooling path under the source framework's sequential
        freeze-in model, in which whole defects (all charge states together) freeze in.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If the process parameters are invalid (see freeze_in_temperature).
    '''
    return totals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sequential_freeze_in_totals(process: dict, host: dict, defects: dict) -> "np.ndarray":
    em = np.asarray(defects["em"], dtype=float).ravel()
    t_freeze = np.array([
        _oracle_freeze_in_temperature(e, process["d0"], process["gamma"], process["t_max"], process["t_min"], process["grain_size"])
        for e in em
    ])
    frozen = np.full(em.size, np.nan)
    # Highest freeze-in temperature first; each species keeps the total it has when it freezes.
    for idx in np.argsort(-t_freeze, kind="stable"):
        state = _oracle_solve_partial_equilibrium(t_freeze[idx], frozen, process["dopant_total"], host, defects)
        frozen[idx] = state[3 + idx]
    return frozen

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
host = {"eg0": 1.583, "varshni_alpha": 3.4e-4, "varshni_beta": 128.0, "f_cb": 0.78,
        "nc_ref": 7.9e17, "nv_ref": 1.69e19, "t_ref": 296.0, "hw0": 0.0119}
defects = {
    "site": np.array([1.484e22, 1.484e22, 1.484e22, 1.484e22, 5.936e22]),
    "n_added": np.array([1, -1, 0, 0, 1]),
    "n_dopant": np.array([0, 0, 1, 1, 1]),
    "em": np.array([0.97, 1.43, 1.62, 1.77, 1.86]),
    "cs_def": np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4]),
    "cs_q": np.array([2, 1, 0, 0, -1, -2, 0, -1, 1, 0, 1, 0]),
    "cs_e": np.array([0.29, 1.66, 3.15, 2.61, 2.92, 3.66, 1.21, 1.34, 0.01, 1.30, 0.66, 1.84]),
}
def fresh():
    return {k: v.copy() for k, v in defects.items()}
"""
    return [
        # --- Valid: few-micron grains, moderate cooling rate ---
        {
            "setup": base + """
process = {"t_max": 1123.0, "t_min": 296.0, "gamma": 0.73, "grain_size": 3.8e-4, "d0": 0.25, "dopant_total": 4.3e16}
""",
            "call": "np.log10(sequential_freeze_in_totals(dict(process), dict(host), fresh()))",
            "gold_call": "np.log10(_oracle_sequential_freeze_in_totals(dict(process), dict(host), fresh()))",
            "tol": 1e-6,
        },
        # --- Valid: slow cooling of a coarse-grained sample with heavier doping ---
        {
            "setup": base + """
process = {"t_max": 1123.0, "t_min": 296.0, "gamma": 0.0081, "grain_size": 5.2e-2, "d0": 0.25, "dopant_total": 1.9e17}
""",
            "call": "np.log10(sequential_freeze_in_totals(dict(process), dict(host), fresh()))",
            "gold_call": "np.log10(_oracle_sequential_freeze_in_totals(dict(process), dict(host), fresh()))",
            "tol": 1e-6,
        },
        # --- Boundary: bulk-like grains, so every species freezes at the start of cooling ---
        {
            "setup": base + """
process = {"t_max": 1123.0, "t_min": 296.0, "gamma": 0.73, "grain_size": 0.62, "d0": 0.25, "dopant_total": 4.3e16}
""",
            "call": "np.log10(sequential_freeze_in_totals(dict(process), dict(host), fresh()))",
            "gold_call": "np.log10(_oracle_sequential_freeze_in_totals(dict(process), dict(host), fresh()))",
            "tol": 1e-6,
        },
        # --- Edge: acceptor-plus-interstitial system (two species, one dopant-containing) ---
        {
            "setup": base + """
pair = {"site": np.array([1.484e22, 1.484e22]), "n_added": np.array([1, 0]), "n_dopant": np.array([0, 1]),
        "em": np.array([0.97, 1.62]), "cs_def": np.array([0, 0, 0, 1, 1]), "cs_q": np.array([2, 1, 0, 0, -1]),
        "cs_e": np.array([0.29, 1.66, 3.15, 1.21, 1.34])}
process = {"t_max": 1123.0, "t_min": 296.0, "gamma": 0.73, "grain_size": 3.8e-4, "d0": 0.25, "dopant_total": 4.3e16}
""",
            "call": "np.log10(sequential_freeze_in_totals(dict(process), dict(host), {k: v.copy() for k, v in pair.items()}))",
            "gold_call": "np.log10(_oracle_sequential_freeze_in_totals(dict(process), dict(host), {k: v.copy() for k, v in pair.items()}))",
            "tol": 1e-6,
        },
    ]
