"""
Solve for the Fermi level, free-carrier densities and defect totals of a doped semiconductor in partial equilibrium at one temperature, where some defect species are frozen at fixed total concentrations and the rest are still free to equilibrate.

During cooling, defect species lose the ability to change their numbers one after another, but electrons and holes, and the charge states of every defect, keep equilibrating with the Fermi level. Charge neutrality couples all species, so a frozen species still influences how the open species respond at lower temperatures. When the total dopant content is fixed rather than set by a reservoir, the dopant atoms that are not locked in frozen species are shared among the open dopant-containing defects.

Returns
-------
np.ndarray [E_F (eV), n (cm^-3), p (cm^-3), then the total (cm^-3) of each defect in table order]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_partial_equilibrium(temperature: float, frozen_totals: "np.ndarray", dopant_total: float, host: dict, defects: dict) -> "np.ndarray":
    '''Charge-neutral partial-equilibrium state at one temperature.

    Parameters
    ----------
    temperature : float
        Absolute temperature (K), strictly positive.
    frozen_totals : np.ndarray
        Float array of length n_def. A finite entry is the fixed total concentration (cm^-3,
        summed over charge states) of a frozen defect; NaN marks a defect that is open and
        equilibrates at this temperature. The array is not modified.
    dopant_total : float
        Total dopant concentration (cm^-3) summed over every dopant-containing defect, open or
        frozen.
    host : dict
        Host description with the keys used by band_edges_and_densities ("eg0",
        "varshni_alpha", "varshni_beta", "f_cb", "nc_ref", "nv_ref", "t_ref") and "hw0", the
        representative vibrational quantum of the lattice (eV).
    defects : dict
        Defect table with per-defect arrays "site", "n_added", "n_dopant" (0 or 1) and "em",
        and per-charge-state arrays "cs_def", "cs_q", "cs_e", as documented for
        open_charge_state_concentrations. The tabulated energies are referenced to a dopant
        chemical potential of zero.

    Returns
    -------
    state : np.ndarray
        Float array of length 3 + n_def: [Fermi level (eV) on the fixed energy scale whose
        zero is the 0 K valence-band maximum, electron density (cm^-3), hole density (cm^-3),
        then the total concentration (cm^-3) of each defect in table order]. Free carriers are
        non-degenerate. Open defects follow the dilute-limit equilibrium populations, with the
        dopant chemical potential set so that the dopant content of all defects equals
        dopant_total whenever an open dopant-containing defect exists. A frozen defect keeps
        its total while its charge-state populations follow the Fermi level. The state is
        electrically neutral.
        Temperatures convert to energies with k_B = 8.617333262e-5 eV/K (CODATA 2018).

    Raises
    ------
    ValueError
        If temperature is not positive, frozen_totals has the wrong length or a negative
        entry, or an open dopant-containing defect exists while frozen species already hold
        all of dopant_total.
    '''
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _frozen_fractions(g, cs_def, idx, kt):
    """Charge-state fractions of defect idx from relative formation energies g."""
    sel = cs_def == idx
    gs = g[sel]
    w = np.exp(-(gs - gs.min()) / kt)
    return sel, w / w.sum()

def _oracle_solve_partial_equilibrium(temperature: float, frozen_totals: "np.ndarray", dopant_total: float, host: dict, defects: dict) -> "np.ndarray":
    kb_ev = 8.617333262e-5  # Boltzmann constant (eV/K)
    t = float(temperature)
    if not t > 0.0:
        raise ValueError("temperature must be positive")
    site = np.asarray(defects["site"], dtype=float).ravel()
    n_def = site.size
    frozen = np.array(frozen_totals, dtype=float).ravel()
    if frozen.size != n_def:
        raise ValueError("frozen_totals must have one entry per defect")
    is_frozen = np.isfinite(frozen)
    if np.any(frozen[is_frozen] < 0.0):
        raise ValueError("frozen totals must be non-negative")
    cs_def = np.asarray(defects["cs_def"], dtype=int).ravel()
    cs_q = np.asarray(defects["cs_q"], dtype=float).ravel()
    cs_e = np.asarray(defects["cs_e"], dtype=float).ravel()
    n_dop = np.asarray(defects["n_dopant"], dtype=int).ravel()

    ec, ev, nc, nv = _oracle_band_edges_and_densities(t, host)
    kt = kb_ev * t
    open_dopant = (~is_frozen) & (n_dop == 1)
    dopant_left = float(dopant_total) - float(np.sum(frozen[is_frozen] * n_dop[is_frozen]))
    if np.any(open_dopant) and not dopant_left > 0.0:
        raise ValueError("no dopant left for the open dopant-containing defects")
    open_dopant_cs = open_dopant[cs_def]

    def _populations(ef):
        conc = _oracle_open_charge_state_concentrations(t, ef, 0.0, host, defects)
        if np.any(open_dopant_cs):
            conc[open_dopant_cs] *= dopant_left / conc[open_dopant_cs].sum()
        g_rel = cs_e + cs_q * ef
        for idx in np.flatnonzero(is_frozen):
            sel, frac = _frozen_fractions(g_rel, cs_def, idx, kt)
            conc[sel] = frozen[idx] * frac
        n = nc * np.exp((ef - ec) / kt)
        p = nv * np.exp((ev - ef) / kt)
        return conc, n, p

    def _net_charge(ef):
        conc, n, p = _populations(ef)
        return float(np.sum(cs_q * conc) + p - n)

    ef = brentq(_net_charge, ev - 1.0, ec + 1.0, xtol=1e-14, rtol=1e-15, maxiter=1000)
    conc, n, p = _populations(ef)
    totals = np.bincount(cs_def, weights=conc, minlength=n_def)
    return np.concatenate(([ef, n, p], totals))

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
def view(state):
    s = np.asarray(state, dtype=float)
    return np.concatenate(([s[0]], np.log10(s[1:])))
"""
    raise_setup = base + """
def run(fn, *args):
    try:
        fn(*args)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Valid: every defect open at the start of cooling, fixed dopant total ---
        {
            "setup": base + """
frozen = np.full(5, np.nan)
""",
            "call": "view(solve_partial_equilibrium(1123.0, frozen.copy(), 4.3e16, dict(host), fresh()))",
            "gold_call": "view(_oracle_solve_partial_equilibrium(1123.0, frozen.copy(), 4.3e16, dict(host), fresh()))",
            "tol": 1e-6,
        },
        # --- Valid: dopant-containing species frozen, interstitial and vacancy open ---
        {
            "setup": base + """
frozen = np.array([np.nan, np.nan, 3.1e16, 6.0e14, 1.13e16])
""",
            "call": "view(solve_partial_equilibrium(702.0, frozen.copy(), 4.3e16, dict(host), fresh()))",
            "gold_call": "view(_oracle_solve_partial_equilibrium(702.0, frozen.copy(), 4.3e16, dict(host), fresh()))",
            "tol": 1e-6,
        },
        # --- Valid: complex frozen, remaining dopant shared by the open acceptor and antisite ---
        {
            "setup": base + """
frozen = np.array([np.nan, np.nan, np.nan, np.nan, 8.7e15])
""",
            "call": "view(solve_partial_equilibrium(1004.0, frozen.copy(), 4.3e16, dict(host), fresh()))",
            "gold_call": "view(_oracle_solve_partial_equilibrium(1004.0, frozen.copy(), 4.3e16, dict(host), fresh()))",
            "tol": 1e-6,
        },
        # --- Boundary: everything frozen at room temperature (only charge states respond) ---
        {
            "setup": base + """
frozen = np.array([2.2e15, 3.0e5, 3.35e16, 4.1e14, 9.1e15])
""",
            "call": "view(solve_partial_equilibrium(296.0, frozen.copy(), 4.3e16, dict(host), fresh()))",
            "gold_call": "view(_oracle_solve_partial_equilibrium(296.0, frozen.copy(), 4.3e16, dict(host), fresh()))",
            "tol": 1e-6,
        },
        # --- Edge: undoped crystal with only native defects open (no dopant closure) ---
        {
            "setup": """import numpy as np
host = {"eg0": 1.583, "varshni_alpha": 3.4e-4, "varshni_beta": 128.0, "f_cb": 0.78,
        "nc_ref": 7.9e17, "nv_ref": 1.69e19, "t_ref": 296.0, "hw0": 0.0119}
native = {"site": np.array([1.484e22, 1.484e22]), "n_added": np.array([1, -1]),
          "n_dopant": np.array([0, 0]), "em": np.array([0.97, 1.43]),
          "cs_def": np.array([0, 0, 0, 1, 1, 1]), "cs_q": np.array([2, 1, 0, 0, -1, -2]),
          "cs_e": np.array([0.29, 1.66, 3.15, 2.61, 2.92, 3.66])}
def view(state):
    s = np.asarray(state, dtype=float)
    return np.concatenate(([s[0]], np.log10(s[1:])))
""",
            "call": "view(solve_partial_equilibrium(850.0, np.full(2, np.nan), 0.0, dict(host), {k: v.copy() for k, v in native.items()}))",
            "gold_call": "view(_oracle_solve_partial_equilibrium(850.0, np.full(2, np.nan), 0.0, dict(host), {k: v.copy() for k, v in native.items()}))",
            "tol": 1e-6,
        },
        # --- Valid: the frozen_totals argument is left unchanged by the call ---
        {
            "setup": base + """
frozen = np.array([np.nan, np.nan, 3.1e16, 6.0e14, 1.13e16])
def keep(fn):
    f = frozen.copy()
    s = fn(702.0, f, 4.3e16, dict(host), fresh())
    return np.concatenate((view(s), [1.0 if np.array_equal(f, frozen, equal_nan=True) else 0.0]))
""",
            "call": "keep(solve_partial_equilibrium)",
            "gold_call": "keep(_oracle_solve_partial_equilibrium)",
            "tol": 1e-6,
        },
        # --- Invalid: frozen species already hold all the dopant while the acceptor is open ---
        {
            "setup": raise_setup + """
frozen = np.array([np.nan, np.nan, np.nan, 2.0e16, 2.4e16])
""",
            "call": "run(solve_partial_equilibrium, 900.0, frozen.copy(), 4.3e16, dict(host), fresh())",
            "gold_call": "run(_oracle_solve_partial_equilibrium, 900.0, frozen.copy(), 4.3e16, dict(host), fresh())",
            "tol": 1e-6,
        },
    ]
