"""
Assemble the full pipeline and return the price of the European call at the configuration supplied in cfg.



Compute the shape integrals; verify that the model's mean-subcriticality condition holds, so that the activity process has a finite stationary mean; determine the finite-horizon exponent boundary; fix the damping level and its objective; cross-check the scalar driver over the range [0, 20] at the selected damping level against that boundary, requiring the driver to change sign when the boundary equals the positive tempering rate; evaluate the transform on the non-negative half of the frequency grid used by the coefficient rule; form the payoff and density coefficients over the translation range; and combine them into the discounted price.



The boundary is treated as equal to the positive tempering rate when the two differ by less than one part in a thousand.



Return the price as a native float.



Raises ValueError if the mean-subcriticality condition fails; if the driver is strictly positive over [0, 20] at the selected damping level while the boundary equals the positive tempering rate; or if any stage of the chain rejects the configuration, for example a positive tempering rate that does not exceed one, or an even node count.

A transform-based pricing calculation is a chain in which each stage constrains the next, and the constraints run in an order that is not obvious from the pricing formula alone. The jump-size law fixes a set of constants; one of those constants fixes the drift under the pricing measure; the drift enters the coefficient system that defines the transform; the behaviour of that system at purely imaginary arguments fixes which damping levels are usable; the damping level fixes both families of expansion coefficients; and only then does the price follow as a single inner product. Reversing any two of these stages produces a calculation that appears to run but answers a different question.




Because the chain is long, an assembled pipeline benefits from checks that use quantities already computed rather than adding new ones. Two are natural here. The first is a stability check: the feedback strength multiplied by the mean excitation must fall below the mean-reversion rate, or the activity process has no finite stationary mean and the model is not well posed at any horizon. The second is a consistency check between two independent routes to the same fact. The exponent boundary is obtained by locating where the coefficient trajectory first escapes in the available time, while the sign of the driver at a given exponent says directly whether escape is possible at all. The implication runs one way. If the boundary coincides with the level permitted by the jump-size law then escape never occurs at any admissible exponent, so the driver must change sign at the selected one, and a disagreement there means one of the two stages is wrong. The converse does not hold: the driver loses its root at a threshold that lies strictly below the finite-horizon boundary, so a selected exponent below that boundary may still sit below the threshold, keep its root, and be admissible at every horizon.




The final combination is an inner product of the two coefficient families over the retained translation indices, discounted at the risk-free rate. Neither family decays on both sides of the range on its own, and it is only their product that does, which is why the truncation range is properly a property of the pair rather than of the density alone.

Returns
-------
float, the discounted price of the European call as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def damped_swift_price(cfg: dict) -> float:
    '''Assemble the pipeline and return the European call price.

    Parameters
    ----------
    cfg : dict
        Configuration with keys p, M, G, a_ts, kappa, lam_bar, eta, a_exc,
        lam0, r, sigma, S0, K, T, m, NQ, lam_lo, lam_hi, frac.

    Returns
    -------
    result : float
        The discounted price of the European call.

    Raises
    ------
    ValueError
        If the mean-subcriticality condition fails; if the driver is
        strictly positive over [0, 20] at the selected damping level while
        the boundary equals the positive tempering rate; or if any stage of
        the chain rejects the configuration, for example a positive
        tempering rate that does not exceed one, or an even node count.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_damped_swift_price(cfg: dict) -> float:
    """Reference implementation."""
    c = cfg

    constants = _oracle_levy_shape_integrals(c["p"], c["M"], c["G"],
                                             c["a_ts"], c["a_exc"])
    chi_j = float(constants[1])
    if not c["eta"] * float(constants[0]) < c["kappa"]:
        raise ValueError("mean-subcriticality condition violated")

    alpha_expl = _oracle_transform_strip_boundary(
        c["T"], chi_j, c["p"], c["M"], c["G"], c["a_ts"], c["a_exc"],
        c["kappa"], c["eta"])

    selection = _oracle_damping_parameter(
        alpha_expl, c["frac"], c["K"], c["T"], chi_j, c["p"], c["M"], c["G"],
        c["a_ts"], c["a_exc"], c["kappa"], c["lam_bar"], c["eta"], c["lam0"],
        c["r"], c["sigma"], c["S0"])
    alpha = float(selection[0])

    driver = _oracle_moment_growth_driver(
        np.linspace(0.0, 20.0, 201), alpha, chi_j, c["p"], c["M"], c["G"],
        c["a_ts"], c["a_exc"], c["kappa"], c["eta"])
    explosive = alpha_expl < c["M"] - 1e-3
    if (not explosive) and float(driver.min()) > 0.0:
        raise ValueError(
            "driver is strictly positive at the selected damping level "
            "although the boundary equals the tempering rate")

    half_grid = np.linspace(0.0, 2.0 ** c["m"] * np.pi, (c["NQ"] + 1) // 2)
    packed = _oracle_activity_riccati_transform(
        half_grid, alpha, c["T"], chi_j, c["p"], c["M"], c["G"], c["a_ts"],
        c["a_exc"], c["kappa"], c["lam_bar"], c["eta"], c["lam0"], c["r"],
        c["sigma"], c["S0"])

    payoff = _oracle_payoff_coefficients(alpha, c["m"], c["lam_lo"],
                                         c["lam_hi"], c["NQ"], c["K"])
    density = _oracle_density_coefficients(packed, alpha, c["m"], c["lam_lo"],
                                           c["lam_hi"], c["NQ"])

    return float(math.exp(-c["r"] * c["T"]) * np.dot(density, payoff))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _setup = """import numpy as np
cfg = dict(p=0.55, M=6.5, G=3.0, a_ts=1.1, kappa=3.5, lam_bar=0.015,
           eta=2.0, a_exc=0.8, lam0=0.02, r=0.03, sigma=0.05, S0=95.0,
           K=98.0, T=0.25, m=3, NQ=513, lam_lo=20, lam_hi=101, frac=0.85)
"""
    return [
        # --- normal: the locked configuration ---
        {
            "setup": _setup,
            "call": "damped_swift_price(cfg)",
            "gold_call": "_oracle_damped_swift_price(cfg)",
        },
        # --- boundary: one resolution level finer, with the node count and
        #     translation range scaled to match ---
        {
            "setup": _setup + """
fine = dict(cfg)
fine['m'] = 4
fine['NQ'] = 1025
fine['lam_lo'] = 40
fine['lam_hi'] = 202
""",
            "call": "damped_swift_price(fine)",
            "gold_call": "_oracle_damped_swift_price(fine)",
        },
        # --- edge: no feedback, so the boundary equals the tempering rate and
        #     the driver must change sign at the selected damping level ---
        {
            "setup": _setup + "flat = dict(cfg)\nflat['eta'] = 0.0\n",
            "call": "damped_swift_price(flat)",
            "gold_call": "_oracle_damped_swift_price(flat)",
        },
        # --- edge: an in-the-money strike, where the price must exceed the
        #     discounted intrinsic value ---
        {
            "setup": _setup + "itm = dict(cfg)\nitm['K'] = 88.0\n",
            "call": "damped_swift_price(itm)",
            "gold_call": "_oracle_damped_swift_price(itm)",
        },
        # --- structural probe: the price is positive, below the spot, above
        #     the undiscounted intrinsic value, increases with the horizon, and
        #     respects the no-arbitrage lower bound in the money; returned as
        #     exact integers ---
        {
            "setup": _setup + """
def probe():
    base = damped_swift_price(cfg)
    longer = dict(cfg)
    longer['T'] = 0.5
    later = damped_swift_price(longer)
    itm = damped_swift_price({**cfg, 'K': 88.0})
    bound = cfg['S0'] - 88.0 * np.exp(-cfg['r'] * cfg['T'])
    return [int(base > 0.0), int(base < cfg['S0']),
            int(base > max(cfg['S0'] - cfg['K'], 0.0)),
            int(later > base), int(itm > bound), int(round(1e6 * base))]
def probe_gold():
    base = _oracle_damped_swift_price(cfg)
    longer = dict(cfg)
    longer['T'] = 0.5
    later = _oracle_damped_swift_price(longer)
    itm = _oracle_damped_swift_price({**cfg, 'K': 88.0})
    bound = cfg['S0'] - 88.0 * np.exp(-cfg['r'] * cfg['T'])
    return [int(base > 0.0), int(base < cfg['S0']),
            int(base > max(cfg['S0'] - cfg['K'], 0.0)),
            int(later > base), int(itm > bound), int(round(1e6 * base))]
""",
            "call": "probe()",
            "gold_call": "probe_gold()",
        },
        # --- invalid: feedback strong enough to break mean subcriticality ---
        {
            "setup": _setup + """
def run_model():
    try:
        damped_swift_price({**cfg, 'eta': 5.0})
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_damped_swift_price({**cfg, 'eta': 5.0})
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- edge: a fraction low enough to place the damping level below the
        #     unbounded-horizon threshold, where the driver keeps its root and
        #     the escape time is infinite. The level is still inside the
        #     admissible set, so the pipeline must price rather than reject ---
        {
            "setup": _setup + "low = dict(cfg)\nlow['frac'] = 0.4\n",
            "call": "damped_swift_price(low)",
            "gold_call": "_oracle_damped_swift_price(low)",
        },
        # --- invalid: an even node count, rejected by the coefficient steps ---
        {
            "setup": _setup + """
def run_model():
    try:
        damped_swift_price({**cfg, 'NQ': 512})
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_damped_swift_price({**cfg, 'NQ': 512})
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: a tempering rate at the positive-tail bound, rejected by
        #     the shape-integral step ---
        {
            "setup": _setup + """
def run_model():
    try:
        damped_swift_price({**cfg, 'M': 1.0})
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_damped_swift_price({**cfg, 'M': 1.0})
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
