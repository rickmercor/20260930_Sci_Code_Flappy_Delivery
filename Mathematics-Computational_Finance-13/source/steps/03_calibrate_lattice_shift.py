"""
Calibrate the deterministic shift of a shifted square-root short-rate model directly on its discrete lattice. The shift is a sequence of per-step increments, one for each numerical time step, chosen so that the discount factor implied by the lattice matches the given market discount factor at every grid maturity. Propagate state prices forward across the rate lattice, using at every node the transition stencil selected by cir_transition_stencil (including its tie rule) and discounting each transition with the rate factor at its two endpoints, and determine each increment from the requirement at that maturity before advancing. Return the increments in time order.

A square-root short-rate factor cannot by itself reproduce an observed initial term structure. The shifted extension writes the short rate as r_t = X_t + phi(t) with X a square-root factor and phi deterministic, so that the model curve carries an extra multiplicative factor exp(-integral of phi) which can be chosen to match the market exactly while leaving X non-negative.



In a lattice implementation the object needed is not phi itself but its integral over each numerical step,



    s_n = integral of phi(u) du over [t_n, t_{n+1}]



which enters the branchwise integrated short rate together with the factor contribution. Over a step from a node carrying factor level x_k to a successor carrying x_kb, the integrated rate is approximated from the factor at both endpoints of the step rather than at the start alone, so that the discount applied along a branch is symmetric in the two levels.

Calibrating s_n on the lattice rather than in continuous time matters because the lattice discount factor is not the continuous-time one: a shift obtained by differencing the closed-form zero-coupon price of the unshifted factor leaves a residual mismatch at every grid maturity. The lattice condition instead determines each increment uniquely. It is one scalar requirement per step in one unknown per step, and because s_n enters the step discount multiplicatively and identically along every branch, it factors out of the forward sum: the total discounted state price at the next maturity is the undiscounted forward mass times exp(-s_n), so s_n is available in closed form at each step with no iteration.

The forward recursion carries state prices, or Arrow-Debreu masses, across the lattice. Starting from unit mass at the root, each step distributes mass to successors weighted by the transition probabilities and the branch discount, the increment is read off from the maturity condition, and the masses are rescaled by exp(-s_n) before the next step. Only the rate lattice is involved: when a second factor is coupled to the rate factor by a table preserving both marginals, the rate marginal evolves exactly as it would alone, and the branch discount depends on no other factor.

Returns
-------
np.ndarray of shape (N,), the integral of the deterministic shift over each numerical step in time order, as native floats
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def calibrate_lattice_shift(x_root: float, kappa: float, theta: float, sigma: float,
                            lam: float, h: float, market_disc: np.ndarray) -> np.ndarray:
    '''Calibrate the per-step deterministic shift increments on the rate lattice.

    Parameters
    ----------
    x_root : float
        Strictly positive initial level of the square-root rate factor.
    kappa : float
        Strictly positive mean-reversion speed.
    theta : float
        Strictly positive long-run level.
    sigma : float
        Strictly positive diffusion coefficient.
    lam : float
        Lattice scale parameter, strictly between 0 and 2.
    h : float
        Strictly positive time step.
    market_disc : np.ndarray
        Shape (N,), strictly positive market discount factors at the grid
        maturities t_1, ..., t_N, where t_n = n*h.

    Returns
    -------
    shift : np.ndarray
        Shape (N,), the integral of the deterministic shift over each step, in
        time order.

    Raises
    ------
    ValueError
        If the arguments do not describe a valid rate lattice, or if the supplied
        curve is not a usable set of discount factors.
    '''
    return shift

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _node(z_root, sigma, lam, h, j):
    chi = 2.0 * np.sqrt(z_root) / sigma + j * lam * np.sqrt(h)
    return (sigma / 2.0 * max(chi, 0.0)) ** 2


def _moments(z, kappa, theta, sigma, h):
    e = np.exp(-kappa * h)
    return (theta + (z - theta) * e,
            sigma ** 2 * z / kappa * e * (1.0 - e) + theta * sigma ** 2 / (2.0 * kappa) * (1.0 - e) ** 2)


def _probs(Zs, m, v):
    d1, d2, d3 = (np.asarray(Zs, float) - m)
    den = np.array([(d1 - d2) * (d1 - d3), (d2 - d1) * (d2 - d3), (d3 - d1) * (d3 - d2)])
    if np.min(np.abs(den)) < 1e-300:
        return None
    return np.array([v + d2 * d3, v + d1 * d3, v + d1 * d2]) / den


def _stencil(x_root, kappa, theta, sigma, lam, h, j, j_min):
    z = _node(x_root, sigma, lam, h, j)
    m, v = _moments(z, kappa, theta, sigma, h)
    for span in range(2, 8):
        cands = [(lo, mid, lo + span) for lo in range(j - span, j + 1)
                 for mid in range(lo + 1, lo + span)]
        cands.sort(key=lambda t: abs((t[0] + t[2]) / 2.0 - j))
        for idx in cands:
            Zs = [_node(x_root, sigma, lam, h, i) for i in idx]
            if len(set(np.round(Zs, 15))) < 3:
                continue
            p = _probs(Zs, m, v)
            if p is None or p.min() < -1e-12:
                continue
            p = np.clip(p, 0.0, None)
            return np.array([max(i, j_min) for i in idx]), p / p.sum()
    raise ValueError("no admissible stencil exists on the rate lattice")


def _oracle_calibrate_lattice_shift(x_root: float, kappa: float, theta: float, sigma: float,
                                    lam: float, h: float, market_disc: np.ndarray) -> np.ndarray:
    for name, val in (("x_root", x_root), ("kappa", kappa), ("theta", theta),
                      ("sigma", sigma), ("h", h)):
        if not isinstance(val, (int, float, np.integer, np.floating)) or not float(val) > 0.0:
            raise ValueError(f"{name} must be a positive real number")
    if not isinstance(lam, (int, float, np.integer, np.floating)) or not 0.0 < float(lam) < 2.0:
        raise ValueError("lam must satisfy 0 < lam < 2")
    P = np.asarray(market_disc, dtype=float)
    if P.ndim != 1 or P.size < 1 or not np.all(np.isfinite(P)) or P.min() <= 0.0:
        raise ValueError("market_disc must be a non-empty 1-D array of positive discount factors")

    x_root, kappa, theta = float(x_root), float(kappa), float(theta)
    sigma, lam, h = float(sigma), float(lam), float(h)
    j_min = int(np.ceil(-2.0 * np.sqrt(x_root) / (sigma * lam * np.sqrt(h))))

    shift = np.zeros(P.size, dtype=float)
    ad = {0: 1.0}
    for n in range(P.size):
        raw, total = {}, 0.0
        for k, mass in ad.items():
            xk = _node(x_root, sigma, lam, h, k)
            child, prob = _stencil(x_root, kappa, theta, sigma, lam, h, k, j_min)
            for c, p in zip(child, prob):
                w = mass * p * np.exp(-0.5 * h * (xk + _node(x_root, sigma, lam, h, int(c))))
                raw[int(c)] = raw.get(int(c), 0.0) + w
                total += w
        s = float(np.log(total / P[n]))
        shift[n] = s
        ad = {c: w * np.exp(-s) for c, w in raw.items()}
    return shift

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pre = """import numpy as np
x_root, kappa, theta, sigma, lam, h = 0.018, 0.45, 0.025, 0.10, 1.4, 0.25
def _curve(u):
    return np.exp(-(0.03 * u - 0.02 * (1.0 - np.exp(-0.5 * u))))
tg = np.array([_curve(0.25 * (n + 1)) for n in range(12)])
tg4 = tg[:4]
flat = np.array([np.exp(-0.025 * 0.25 * (n + 1)) for n in range(12)])
steep = np.array([np.exp(-(0.01 + 0.02 * 0.25 * (n + 1)) * 0.25 * (n + 1)) for n in range(12)])
"""
    err = pre + """
def run_model(**kw):
    a = dict(x_root=0.018, kappa=0.45, theta=0.025, sigma=0.10, lam=1.4, h=0.25, market_disc=tg)
    a.update(kw)
    try:
        calibrate_lattice_shift(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold(**kw):
    a = dict(x_root=0.018, kappa=0.45, theta=0.025, sigma=0.10, lam=1.4, h=0.25, market_disc=tg)
    a.update(kw)
    try:
        _oracle_calibrate_lattice_shift(**a); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
"""
    return [
        {"setup": pre, "call": "calibrate_lattice_shift(x_root, kappa, theta, sigma, lam, h, tg)",
         "gold_call": "_oracle_calibrate_lattice_shift(x_root, kappa, theta, sigma, lam, h, tg)"},
        {"setup": pre, "call": "calibrate_lattice_shift(x_root, kappa, theta, sigma, lam, h, tg4)",
         "gold_call": "_oracle_calibrate_lattice_shift(x_root, kappa, theta, sigma, lam, h, tg4)"},
        {"setup": pre, "call": "calibrate_lattice_shift(x_root, kappa, theta, sigma, lam, h, flat)",
         "gold_call": "_oracle_calibrate_lattice_shift(x_root, kappa, theta, sigma, lam, h, flat)"},
        {"setup": pre, "call": "calibrate_lattice_shift(x_root, kappa, theta, sigma, lam, h, steep)",
         "gold_call": "_oracle_calibrate_lattice_shift(x_root, kappa, theta, sigma, lam, h, steep)"},
        {"setup": pre, "call": "calibrate_lattice_shift(x_root, kappa, theta, sigma, lam, h, tg[:1])",
         "gold_call": "_oracle_calibrate_lattice_shift(x_root, kappa, theta, sigma, lam, h, tg[:1])"},
        {"setup": pre, "call": "calibrate_lattice_shift(0.004, kappa, theta, sigma, 1.0, h, tg)",
         "gold_call": "_oracle_calibrate_lattice_shift(0.004, kappa, theta, sigma, 1.0, h, tg)"},
        {"setup": err, "call": "run_model(market_disc=np.array([1.0, 0.0, -0.5]))",
         "gold_call": "run_gold(market_disc=np.array([1.0, 0.0, -0.5]))"},
        {"setup": err, "call": "run_model(market_disc=np.array([]))",
         "gold_call": "run_gold(market_disc=np.array([]))"},
        {"setup": err, "call": "run_model(lam=0.0)", "gold_call": "run_gold(lam=0.0)"},
    ]
