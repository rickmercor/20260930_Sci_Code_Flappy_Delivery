"""
Step 06: Potential scaling factor for a level at fixed energy. Potential scaling factor that places a chosen coupled-channel bound state at a fixed energy.

Spectroscopic fitting and studies of near-threshold states often need the value of a model parameter, rather than the energy, at which a bound state appears at a given energy. A common case is a factor that multiplies the whole interaction potential: making the well deeper pulls states down, so at fixed energy the node count rises by one each time the factor carries a state through that energy. Whether that count is monotonic in the parameter is not guaranteed in general, and where it is not, states can be missed or found twice. Nor does the parameter behave like the energy in every respect. The derivative of the coupled-channel matrix with respect to the scaling factor is the interaction itself, which is repulsive at short range and attractive in the well, so the sign pattern that the matching eigenvalues show as a function of energy need not survive the change of variable.

Returns
-------
float, interaction scaling factor at which state m lies at energy E
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scale_for_level(params: dict, m: int, E: float, lam_low: float, lam_high: float, R_min: float, R_max: float,
                    tol: float) -> float:
    '''Converged interaction scaling factor at which bound state m of the walled coupled-channel problem lies at energy E.

    Parameters
    ----------
    params : dict
        Model parameters as in channel_matrix; the value stored under 'scale' is ignored and replaced by the trial factor.
    m : int
        Bound-state label, the exact node count immediately above the state; integer >= 1.
    E : float
        Energy in cm^-1 at which state m is required, finite.
    lam_low : float
        Lower end of the search interval for the scaling factor, finite and > 0.
    lam_high : float
        Upper end of the search interval, finite and > lam_low.
    R_min : float
        Inner wall in angstrom, finite, > 0 and < 4.0; every radial channel function vanishes there.
    R_max : float
        Outer wall in angstrom, finite and > 4.0; every radial channel function vanishes there.
    tol : float
        Required absolute accuracy of the scaling factor, finite and >= 1e-10.

    Returns
    -------
    lam_star : float
        With n(lam) the exact multichannel node count at energy E of the coupled radial equations for params with
        'scale' set to lam and psi(R_min) = psi(R_max) = 0 (the number of eigenvalues below E), and n(lam)
        non-decreasing on [lam_low, lam_high], lam_star is the infimum of lam in that interval with n(lam) >= m,
        returned within tol of its exact value. The interval ends are not within 1e-6 of a crossing of E by any state.
        Propagations are matched at 4.0 angstrom, the matching distance this problem fixes, so the walls must enclose it.

    Raises
    ------
    ValueError
        If m is not an integer >= 1, E, lam_low, lam_high, R_min, R_max or tol is not finite, lam_low <= 0,
        lam_high <= lam_low, R_min <= 0, R_max <= R_min, tol < 1e-10, 4.0 angstrom does not lie strictly between R_min
        and R_max, or the interval does not bracket the state, that is unless n(lam_low) < m <= n(lam_high).
    '''
    return lam_star

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_scale_for_level(params: dict, m: int, E: float, lam_low: float, lam_high: float, R_min: float,
                            R_max: float, tol: float) -> float:
    import numpy as np
    if isinstance(m, bool) or int(m) != m or m < 1:
        raise ValueError("m must be an integer >= 1")
    if not all(np.isfinite([E, lam_low, lam_high, R_min, R_max, tol])) or lam_low <= 0.0 or lam_high <= lam_low:
        raise ValueError("invalid energy or interval")
    if R_min <= 0.0 or R_max <= R_min or tol < 1e-10:
        raise ValueError("need 0 < R_min < R_max and tol >= 1e-10")
    m = int(m)
    setting_of_h = lambda h: (dict(params), float(E), float(R_min), float(R_max), h, "scale")
    coarse = setting_of_h(0.005)
    if not (_node_count(coarse, lam_low) < m <= _node_count(coarse, lam_high)):
        raise ValueError("the interval does not bracket state m at this energy")
    return float(_extrapolated_roots(setting_of_h, [m], float(lam_low), float(lam_high), tol)[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = ("import numpy as np\n"
            "P = {'mu': 19.2, 'B': 10.4, 'eps': 180.0, 'Rm': 3.8, 'a': (1.0, 0.25, 0.35), 'b': (1.0, 0.15, 0.30),"
            " 'jmax': 6, 'J': 0, 'parity': 1, 'scale': 1.0}\n")
    return [
        # --- Normal: benchmark complex, state 17 moved down to -30 cm^-1 ---
        {"setup": base, "call": "float(scale_for_level(dict(P), 17, -30.0, 1.0, 1.4, 2.5, 15.0, 3e-10))",
         "gold_call": "float(_oracle_scale_for_level(dict(P), 17, -30.0, 1.0, 1.4, 2.5, 15.0, 3e-10))", "tol": 1e-9},
        # --- Normal: shallower well, factor below one ---
        {"setup": base, "call": "float(scale_for_level(dict(P), 9, -60.0, 0.6, 1.0, 2.8, 15.0, 3e-10))",
         "gold_call": "float(_oracle_scale_for_level(dict(P), 9, -60.0, 0.6, 1.0, 2.8, 15.0, 3e-10))", "tol": 1e-9},
        # --- Boundary: heavy J = 5 odd-parity complex, state 130 in a dense manifold ---
        {"setup": base + "Q = dict(P, mu=40.0, B=1.1, jmax=5, J=5, parity=-1)\n",
         "call": "float(scale_for_level(dict(Q), 130, -45.0, 1.0, 1.1, 2.9, 13.0, 3e-10))",
         "gold_call": "float(_oracle_scale_for_level(dict(Q), 130, -45.0, 1.0, 1.1, 2.9, 13.0, 3e-10))", "tol": 1e-9},
        # --- Boundary: near-degenerate pair at an avoided crossing, the lower-labelled state of the two ---
        {"setup": base + "Q = dict(P, a=(1.0, 4e-4, 6e-4), b=(1.0, 2e-4, 4e-4), jmax=4, B=10.0331)\n",
         "call": "float(scale_for_level(dict(Q), 10, -52.5, 0.98, 1.02, 2.6, 12.0, 3e-10))",
         "gold_call": "float(_oracle_scale_for_level(dict(Q), 10, -52.5, 0.98, 1.02, 2.6, 12.0, 3e-10))", "tol": 1e-9},
        # --- Edge: J = 1 odd parity for a larger complex whose well sits near 5.2 angstrom ---
        {"setup": base + "Q = dict(P, Rm=5.2, eps=140.0, mu=24.0, B=6.5, J=1, parity=-1, jmax=5)\n",
         "call": "float(scale_for_level(dict(Q), 12, -85.0, 1.0, 1.1, 3.6, 16.0, 3e-10))",
         "gold_call": "float(_oracle_scale_for_level(dict(Q), 12, -85.0, 1.0, 1.1, 3.6, 16.0, 3e-10))", "tol": 1e-9},
        # --- Edge: single channel over a wide interval ---
        {"setup": base, "call": "float(scale_for_level(dict(P, jmax=0), 5, -20.0, 0.5, 3.0, 2.5, 15.0, 3e-10))",
         "gold_call": "float(_oracle_scale_for_level(dict(P, jmax=0), 5, -20.0, 0.5, 3.0, 2.5, 15.0, 3e-10))", "tol": 1e-9},
        # --- Error: the walls do not enclose the matching distance ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(dict(P), 17, -30.0, 1.0, 1.4, 2.5, 3.9, 3e-10)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(scale_for_level)", "gold_call": "_probe(_oracle_scale_for_level)"},
        # --- Error: the interval does not bracket the state ---
        {"setup": base + "def _probe(fn):\n    try:\n        fn(dict(P), 30, -30.0, 1.0, 1.1, 2.5, 15.0, 1e-8)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(scale_for_level)", "gold_call": "_probe(_oracle_scale_for_level)"},
    ]
