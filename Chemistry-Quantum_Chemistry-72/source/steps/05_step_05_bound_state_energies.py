"""
Step 05: All bound states in an energy window. All coupled-channel bound states in an energy window, converged on individual eigenvalues of the matching matrix.

The node count at the two ends of an energy window says how many bound states lie inside it and which labels they carry, with a state labelled by the value the node count takes just above it. Converging on a state from the node count alone gains one bit of its energy per propagation, and propagations are the whole cost of the calculation, so an accuracy of 1e-9 cm^-1 has to be bought some other way. What the matching matrix offers is a smooth function of the energy for each state, with the awkward property that which member of the ordered spectrum belongs to a given state changes as the energy moves. Locally closed channels at the matching distance add slowly varying eigenvalues that thread through avoided crossings; a matching distance in a classically forbidden region, or close to a node of a state, makes the useful energy range of its eigenvalue very narrow; and two states can lie closer together than the step-size error of a single grid. Every energy already propagated narrows the brackets of all the states at once.

Returns
-------
numpy.ndarray: ascending energies in cm^-1 of all bound states with labels n(E_bottom) + 1 to n(E_top)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bound_state_energies(params: dict, E_bottom: float, E_top: float, R_min: float, R_max: float,
                         tol: float) -> "np.ndarray":
    '''Converged energies of every bound state of the walled coupled-channel problem between E_bottom and E_top.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix.
    E_bottom : float
        Lower end of the energy window in cm^-1, finite, not within 1e-6 cm^-1 of a bound state.
    E_top : float
        Upper end of the energy window in cm^-1, finite, > E_bottom and not within 1e-6 cm^-1 of a bound state.
    R_min : float
        Inner wall in angstrom, finite, > 0 and < 4.0; every radial channel function vanishes there.
    R_max : float
        Outer wall in angstrom, finite and > 4.0; every radial channel function vanishes there.
    tol : float
        Required absolute accuracy of each energy in cm^-1, finite and >= 1e-9.

    Returns
    -------
    energies : np.ndarray
        One-dimensional float array in ascending order holding every eigenvalue E of the exact coupled radial
        equations d^2 psi / dR^2 = [W(R) - (mu / C) E I] psi (W from channel_matrix) with psi(R_min) = psi(R_max) = 0
        that lies in (E_bottom, E_top), each within tol of its exact value. Degenerate eigenvalues are repeated. The
        exact multichannel node count n(E), the number of eigenvalues below E, labels the states: the returned
        energies are those of states n(E_bottom) + 1, ..., n(E_top). Finite-grid results from matching_spectrum carry a
        step-size error that must be removed to reach tol. Wherever a propagation is needed the outward and inward
        matrices are matched at 4.0 angstrom, the matching distance this problem fixes, so the walls must enclose it.

    Raises
    ------
    ValueError
        If E_bottom, E_top, R_min, R_max or tol is not finite, E_top <= E_bottom, R_min <= 0, R_max <= R_min,
        tol < 1e-9, or 4.0 angstrom does not lie strictly between R_min and R_max.
    '''
    return energies

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _problem_at(setting, x):
    """The walled problem at variable value x: params, energy and the grid matched at 4.0 angstrom."""
    params, E, R_min, R_max, h, variable = setting
    if variable == "E":
        E = float(x)
    else:
        params = dict(params, scale=float(x))
    return params, float(E), {"R_min": R_min, "R_match": 4.0, "R_max": R_max, "h": h}


def _node_count(setting, x):
    spec = _oracle_matching_spectrum(*_problem_at(setting, x))
    return int(round(spec[0] + spec[1]))


def _tracked(setting, x, m):
    params, E, grid = _problem_at(setting, x)
    entry = _oracle_tracked_eigenvalue(params, m, E, grid)
    return int(round(entry[0])), float(entry[1])


def _converge_levels(setting, labels, x_low, x_high, tol):
    """Bracket bookkeeping of the paper: for each label keep the tightest points with node count below m and at or
    above m, read the eigenvalue of state m from tracked_eigenvalue, bisect until one position brackets a sign change
    of it and then run Brent on that position. A Brent result is accepted only if the node counts it produced bracket
    it within tol."""
    import numpy as np
    from scipy.optimize import brentq
    counts = {float(x_low): _node_count(setting, x_low), float(x_high): _node_count(setting, x_high)}

    def bracket(m):
        return (max(x for x, n in counts.items() if n < m), min(x for x, n in counts.items() if n >= m))

    def signed(m, x):
        x = float(x)
        if x not in counts:
            counts[x] = _node_count(setting, x)
        i, e = _tracked(setting, x, m)
        return e if i else (1.0 if counts[x] < m else -1.0)

    roots = []
    for m in labels:
        brent_allowed = True
        while True:
            lo, hi = bracket(m)
            if hi - lo <= tol:
                roots.append(0.5 * (lo + hi))
                break
            i_lo, e_lo = _tracked(setting, lo, m)
            i_hi, e_hi = _tracked(setting, hi, m)
            if brent_allowed and i_lo and i_lo == i_hi and e_lo > 0.0 > e_hi:
                root = brentq(lambda x: signed(m, x), lo, hi, xtol=0.25 * tol, rtol=4.0 * np.finfo(float).eps)
                lo, hi = bracket(m)
                if hi - lo <= tol and lo - tol <= root <= hi + tol:
                    roots.append(root)
                    break
                brent_allowed = False
                continue
            mid = 0.5 * (lo + hi)
            counts[mid] = _node_count(setting, mid)
    return np.array(roots, dtype=float)


def _extrapolated_roots(setting_of_h, labels, x_low, x_high, tol):
    """Roots on grids h = 0.005 and 0.0025 angstrom, combined by Richardson extrapolation of the h^4 error."""
    import numpy as np
    coarse = setting_of_h(0.005)
    r1 = _converge_levels(coarse, labels, x_low, x_high, 0.05 * tol)
    fine = setting_of_h(0.0025)
    r2 = []
    for m, r in zip(labels, r1):
        width = max(1e-6 * (1.0 + abs(r)), 1e3 * tol)
        while True:
            lo, hi = max(x_low, r - width), min(x_high, r + width)
            if _node_count(fine, lo) < m <= _node_count(fine, hi) or (lo == x_low and hi == x_high):
                break
            width *= 10.0
        r2.append(_converge_levels(fine, [m], lo, hi, 0.05 * tol)[0])
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    return r2 + (r2 - r1) / 15.0


def _oracle_bound_state_energies(params: dict, E_bottom: float, E_top: float, R_min: float, R_max: float,
                                 tol: float) -> "np.ndarray":
    import numpy as np
    if not all(np.isfinite([E_bottom, E_top, R_min, R_max, tol])) or E_top <= E_bottom:
        raise ValueError("need finite E_bottom < E_top")
    if R_min <= 0.0 or R_max <= R_min or tol < 1e-9:
        raise ValueError("need 0 < R_min < R_max and tol >= 1e-9")
    setting_of_h = lambda h: (dict(params), 0.0, float(R_min), float(R_max), h, "E")
    coarse = setting_of_h(0.005)
    n_bottom = _node_count(coarse, E_bottom)
    labels = list(range(n_bottom + 1, _node_count(coarse, E_top) + 1))
    return _extrapolated_roots(setting_of_h, labels, float(E_bottom), float(E_top), tol)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "P = {'mu': 19.2, 'B': 10.4, 'eps': 180.0, 'Rm': 3.8, 'a': (1.0, 0.25, 0.35), 'b': (1.0, 0.15, 0.30),"
            " 'jmax': 6, 'J': 0, 'parity': 1, 'scale': 1.0}\n")
    return [
        # --- Normal: the five states of the benchmark complex around the observed energy, independent literals ---
        {"setup": base, "call": "np.asarray(bound_state_energies(dict(P), -45.0, -20.0, 2.5, 15.0, 2e-9), dtype=float)",
         "gold_call": "np.asarray(_oracle_bound_state_energies(dict(P), -45.0, -20.0, 2.5, 15.0, 2e-9), dtype=float)",
         "tol": 1e-9},
        # --- Normal: heavy J = 4 complex with 25 channels and a dense spectrum ---
        {"setup": base + "Q = dict(P, mu=40.0, B=1.1, jmax=6, J=4, parity=1)\n",
         "call": "np.asarray(bound_state_energies(dict(Q), -62.0, -60.0, 2.9, 13.0, 2e-9), dtype=float)",
         "gold_call": "np.asarray(_oracle_bound_state_energies(dict(Q), -62.0, -60.0, 2.9, 13.0, 2e-9), dtype=float)",
         "tol": 4.5e-11},
        # --- Boundary: weak anisotropy tuned to an avoided crossing, two states 0.0115 cm^-1 apart ---
        {"setup": base + "Q = dict(P, a=(1.0, 4e-4, 6e-4), b=(1.0, 2e-4, 4e-4), jmax=4, B=10.0331)\n",
         "call": "np.asarray(bound_state_energies(dict(Q), -54.0, -51.0, 2.6, 12.0, 2e-9), dtype=float)",
         "gold_call": "np.asarray(_oracle_bound_state_energies(dict(Q), -54.0, -51.0, 2.6, 12.0, 2e-9), dtype=float)",
         "tol": 1e-9},
        # --- Boundary: J = 2 odd parity with a deepened well and four states over a wide window ---
        {"setup": base, "call": "np.asarray(bound_state_energies(dict(P, scale=1.15, J=2, parity=-1), -75.0, -55.0, 2.8, 12.0, 2e-9), dtype=float)",
         "gold_call": "np.asarray(_oracle_bound_state_energies(dict(P, scale=1.15, J=2, parity=-1), -75.0, -55.0, 2.8, 12.0, 2e-9), dtype=float)",
         "tol": 1e-9},
        # --- Edge: J = 1 odd parity for a larger complex whose well sits near 5.2 angstrom ---
        {"setup": base + "Q = dict(P, Rm=5.2, eps=140.0, mu=24.0, B=6.5, J=1, parity=-1, jmax=5)\n",
         "call": "np.asarray(bound_state_energies(dict(Q), -100.0, -80.0, 3.6, 16.0, 2e-9), dtype=float)",
         "gold_call": "np.asarray(_oracle_bound_state_energies(dict(Q), -100.0, -80.0, 3.6, 16.0, 2e-9), dtype=float)",
         "tol": 1e-9},
        # --- Edge: a single channel, every bound state from the bottom of the well upwards ---
        {"setup": base, "call": "np.asarray(bound_state_energies(dict(P, jmax=0), -200.0, -30.0, 2.5, 15.0, 2e-9), dtype=float)",
         "gold_call": "np.asarray(_oracle_bound_state_energies(dict(P, jmax=0), -200.0, -30.0, 2.5, 15.0, 2e-9), dtype=float)",
         "tol": 1e-9},
        # --- Error: the walls do not enclose the matching distance ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(dict(P), -40.0, -24.0, 4.2, 15.0, 2e-9)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(bound_state_energies)", "gold_call": "_probe(_oracle_bound_state_energies)"},
        # --- Error: empty energy window ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(dict(P), -30.0, -30.0, 2.5, 15.0, 1e-8)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(bound_state_energies)", "gold_call": "_probe(_oracle_bound_state_energies)"},
    ]
