"""
Determine the largest exponent at which the terminal stock price of this model has a finite exponential moment at the horizon T.

This boundary is what makes a damping level admissible for the pricing method of the following steps. It is a property of the model at a finite horizon, and it is at most the positive tempering rate M, which is the boundary of the jump-size shape's own exponential moment. It is also at least the corresponding boundary for an unbounded horizon.

Evaluate every Levy integral with the split protocol of the shape-integral step: reflection onto the positive half-line, truncation at y_cut = 20, a 128-node Gauss-Jacobi rule on (0, 1) carrying the endpoint weight induced by the near-origin behaviour of the compensated integrands, and a 224-node Gauss-Legendre rule on the tail. This step truncates at 20 rather than at the 10 used by the shape-integral step, because the feedback factor exp(eta * x * g(y)) multiplies the dropped tail by the same amount as the part that is kept, so y_cut = 10 does not reach the absolute accuracy of 1e-9 that this step requires. In this step's own integrand, as in the drift-restriction integral supplied to it, only jumps of size below one are compensated. The excitation function is g(y) = 1 - exp(-a_exc * y**2), with a_exc > 0.

Determine the boundary to an absolute accuracy of at least 1e-9 and return it. Search only over exponents strictly between 1 and M. If the terminal price has a finite exponential moment at every exponent below M, return M - 1e-6, that is M minus the absolute amount 10^-6, which is the largest exponent the search considers.

Raises ValueError if T <= 0; if M <= 1; if kappa <= 0; or if eta < 0.

Whether a random variable has a finite exponential moment at a given exponent is a question about the tail of its distribution, and for a process built from an exponentially tempered jump measure there are two separate reasons the moment can fail to exist. The first is intrinsic to the jump-size law: the tempering rate on the positive tail sets a threshold beyond which even a single jump has an infinite exponential moment, and no exponent above that threshold can work whatever the rest of the model does. The second reason is dynamic and arises only when the intensity of jumps responds to the jumps themselves. Feedback of that kind can make large moves beget further large moves quickly enough that the aggregate over a finite horizon has a heavier tail than any single jump, so the moment fails at an exponent the jump-size law alone would permit.

The dynamic mechanism is visible in the coefficient equation of the preceding step. Because the exponent enters that equation both directly and inside the exponential of the feedback term, the driver can remain strictly positive for exponents at which the corresponding driver without feedback would have had a root. Once the driver has no root the coefficient escapes to infinity, and it does so in a finite time that shrinks as the exponent grows. The moment at a given horizon is finite precisely when that escape time exceeds the horizon, so the boundary separating finite from infinite moments depends on the horizon and decreases as the horizon lengthens.

Two limits bound the result. As the horizon shrinks the escape time constraint becomes vacuous and the boundary rises toward whatever the jump-size law alone permits. As the horizon grows the boundary falls toward the exponent at which the driver first loses its root, below which the coefficient never escapes at all and the moment is finite for every horizon. In the special case of no feedback the driver is affine and decreasing, so it always has a root, the escape never occurs, and the boundary is set entirely by the jump-size law.

Returns
-------
float, the finite-horizon exponent boundary as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transform_strip_boundary(T: float, chiJ: float, p: float, M: float,
                             G: float, a_ts: float, a_exc: float,
                             kappa: float, eta: float) -> float:
    '''Largest exponent with a finite terminal exponential moment at T.

    Parameters
    ----------
    T : float
        Horizon, positive.
    chiJ : float
        Drift-restriction integral from the shape-integral step.
    p, M, G, a_ts, a_exc : float
        Jump-size shape and excitation parameters.
    kappa : float
        Activity mean-reversion rate, positive.
    eta : float
        Feedback strength, non-negative.

    Returns
    -------
    result : float
        The finite-horizon exponent boundary.

    Raises
    ------
    ValueError
        If T <= 0; if M <= 1; if kappa <= 0; or if eta < 0.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np
from scipy.special import roots_jacobi, roots_legendre, gamma as _gamma_fn


def _h_nodes(a_ts, n_small=128, n_tail=128, y_cut=10.0):
    """Quadrature nodes and weights for the split protocol."""
    jac_a, jac_b = 0.0, 1.0 - a_ts
    xj, wj = roots_jacobi(n_small, jac_a, jac_b)
    ys = (xj + 1.0) / 2.0
    ws = wj / 2.0 ** (jac_a + jac_b + 1.0)
    xl, wl = roots_legendre(n_tail)
    yt = 0.5 * (y_cut - 1.0) * xl + 0.5 * (y_cut + 1.0)
    wt = 0.5 * (y_cut - 1.0) * wl
    return ys, ws, yt, wt


def _h_density(p, M, G, a_ts, a_exc, ys, yt):
    """Shape constants, density factors and the excitation on both node sets."""
    c_pos = p * M ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    c_neg = (1.0 - p) * G ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    return {
        "sp": c_pos * np.exp(-M * ys),
        "sm": c_neg * np.exp(-G * ys),
        "tp": c_pos * np.exp(-M * yt) * yt ** (-1.0 - a_ts),
        "tm": c_neg * np.exp(-G * yt) * yt ** (-1.0 - a_ts),
        "gs": 1.0 - np.exp(-a_exc * ys ** 2),
        "gt": 1.0 - np.exp(-a_exc * yt ** 2),
    }


def _h_make_driver(chiJ, kappa, eta, ys, yt, dens, w_sp, w_tp, w_sm, w_tm):
    """Return the scalar driver as a function of (x, alpha), vectorised in x."""

    def _driver(x, a):
        xa = np.atleast_1d(np.asarray(x, dtype=float))
        exc_s = eta * np.multiply.outer(xa, dens["gs"])
        exc_t = eta * np.multiply.outer(xa, dens["gt"])
        jump = ((np.exp(a * ys + exc_s) - 1.0 - a * ys) @ w_sp
                + (np.exp(a * yt + exc_t) - 1.0) @ w_tp
                + (np.exp(-a * ys + exc_s) - 1.0 + a * ys) @ w_sm
                + (np.exp(-a * yt + exc_t) - 1.0) @ w_tm)
        out = -kappa * xa - a * chiJ + jump
        return out if np.ndim(x) else float(out[0])

    return _driver


def _h_blowup_time(driver, a, eta, x_cap=18.0, n_scan=4001):
    """Time at which the activity coefficient escapes, infinite if it cannot."""
    xs = np.linspace(0.0, x_cap, n_scan)
    fv = driver(xs, a)
    if fv.min() <= 0.0:
        return np.inf
    inv = 1.0 / fv
    h = xs[1] - xs[0]
    lower = h / 3.0 * (inv[0] + inv[-1]
                       + 4.0 * inv[1:-1:2].sum() + 2.0 * inv[2:-2:2].sum())
    xg, wg = roots_legendre(64)
    t_max = math.exp(-eta * x_cap)
    xt = 0.5 * t_max * xg + 0.5 * t_max
    wt_ = 0.5 * t_max * wg
    upper = float(np.sum(wt_ / (eta * xt * driver(-np.log(xt) / eta, a))))
    return lower + upper


def _oracle_transform_strip_boundary(T: float, chiJ: float, p: float,
                                     M: float, G: float, a_ts: float,
                                     a_exc: float, kappa: float,
                                     eta: float) -> float:
    """Reference implementation."""
    if float(T) <= 0.0:
        raise ValueError("T must be positive")
    if float(M) <= 1.0:
        raise ValueError("M must exceed 1")
    if float(kappa) <= 0.0:
        raise ValueError("kappa must be positive")
    if float(eta) < 0.0:
        raise ValueError("eta must be non-negative")

    ys, ws, yt, wt = _h_nodes(float(a_ts), n_tail=224, y_cut=20.0)
    dens = _h_density(float(p), float(M), float(G), float(a_ts),
                      float(a_exc), ys, yt)
    driver = _h_make_driver(float(chiJ), float(kappa), float(eta), ys, yt,
                            dens,
                            ws * dens["sp"] / ys ** 2, wt * dens["tp"],
                            ws * dens["sm"] / ys ** 2, wt * dens["tm"])

    lo = 1.0 + 1e-6
    hi = float(M) - 1e-6

    if float(eta) == 0.0:
        # the driver is affine and decreasing, so it always has a root and the
        # moment is finite at every exponent the search considers
        return float(hi)
    if _h_blowup_time(driver, hi, float(eta)) == np.inf:
        return float(hi)

    while hi - lo > 1e-9:
        mid = 0.5 * (lo + hi)
        if _h_blowup_time(driver, mid, float(eta)) > float(T):
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _setup = """import numpy as np
p, M, G, a_ts, a_exc = 0.55, 6.5, 3.0, 1.1, 0.8
kappa, eta = 3.5, 2.0
chiJ = 0.4803150971
T = 0.25
"""
    return [
        # --- normal: the locked horizon ---
        {
            "setup": _setup,
            "call": ("transform_strip_boundary(T, chiJ, p, M, G, a_ts, a_exc, "
                     "kappa, eta)"),
            "gold_call": ("_oracle_transform_strip_boundary(T, chiJ, p, M, G, "
                          "a_ts, a_exc, kappa, eta)"),
        },
        # --- boundary: a very short horizon pushes the exponent boundary up
        #     toward the positive tempering rate ---
        {
            "setup": _setup,
            "call": ("transform_strip_boundary(0.02, chiJ, p, M, G, a_ts, "
                     "a_exc, kappa, eta)"),
            "gold_call": ("_oracle_transform_strip_boundary(0.02, chiJ, p, M, "
                          "G, a_ts, a_exc, kappa, eta)"),
        },
        # --- boundary: a long horizon pushes it down toward the unbounded
        #     horizon value ---
        {
            "setup": _setup,
            "call": ("transform_strip_boundary(25.0, chiJ, p, M, G, a_ts, "
                     "a_exc, kappa, eta)"),
            "gold_call": ("_oracle_transform_strip_boundary(25.0, chiJ, p, M, "
                          "G, a_ts, a_exc, kappa, eta)"),
        },
        # --- edge: no feedback, so the moment is finite at every exponent
        #     below the positive tempering rate and the horizon is irrelevant ---
        {
            "setup": _setup,
            "call": ("transform_strip_boundary(T, chiJ, p, M, G, a_ts, a_exc, "
                     "kappa, 0.0)"),
            "gold_call": ("_oracle_transform_strip_boundary(T, chiJ, p, M, G, "
                          "a_ts, a_exc, kappa, 0.0)"),
        },
        # --- edge: strong feedback with fast mean reversion ---
        {
            "setup": _setup,
            "call": ("transform_strip_boundary(T, chiJ, p, M, G, a_ts, a_exc, "
                     "9.0, 6.0)"),
            "gold_call": ("_oracle_transform_strip_boundary(T, chiJ, p, M, G, "
                          "a_ts, a_exc, 9.0, 6.0)"),
        },
        # --- edge: feedback present but mean reversion fast enough that the
        #     coefficient never escapes, so the boundary sits at the top of
        #     the range the jump-size law allows ---
        {
            "setup": _setup,
            "call": ("transform_strip_boundary(T, chiJ, p, M, G, a_ts, a_exc, "
                     "3.5, 0.1)"),
            "gold_call": ("_oracle_transform_strip_boundary(T, chiJ, p, M, G, "
                          "a_ts, a_exc, 3.5, 0.1)"),
        },
        # --- structural probe: the boundary is strictly below the positive
        #     tempering rate, decreases with the horizon, and is horizon
        #     independent without feedback; returned as exact integers ---
        {
            "setup": _setup + """
def probe():
    b_short = transform_strip_boundary(0.05, chiJ, p, M, G, a_ts, a_exc,
                                       kappa, eta)
    b_long = transform_strip_boundary(2.00, chiJ, p, M, G, a_ts, a_exc,
                                      kappa, eta)
    n_a = transform_strip_boundary(0.25, chiJ, p, M, G, a_ts, a_exc,
                                   kappa, 0.0)
    n_b = transform_strip_boundary(4.00, chiJ, p, M, G, a_ts, a_exc,
                                   kappa, 0.0)
    return [int(b_short < M), int(b_long < b_short), int(b_long > 1.0),
            int(round(1e6 * abs(n_a - n_b))),
            int(round(100 * transform_strip_boundary(0.25, chiJ, p, M, G,
                                                     a_ts, a_exc, kappa,
                                                     eta)))]
def probe_gold():
    b_short = _oracle_transform_strip_boundary(0.05, chiJ, p, M, G, a_ts,
                                               a_exc, kappa, eta)
    b_long = _oracle_transform_strip_boundary(2.00, chiJ, p, M, G, a_ts,
                                              a_exc, kappa, eta)
    n_a = _oracle_transform_strip_boundary(0.25, chiJ, p, M, G, a_ts, a_exc,
                                           kappa, 0.0)
    n_b = _oracle_transform_strip_boundary(4.00, chiJ, p, M, G, a_ts, a_exc,
                                           kappa, 0.0)
    return [int(b_short < M), int(b_long < b_short), int(b_long > 1.0),
            int(round(1e6 * abs(n_a - n_b))),
            int(round(100 * _oracle_transform_strip_boundary(
                0.25, chiJ, p, M, G, a_ts, a_exc, kappa, eta)))]
""",
            "call": "probe()",
            "gold_call": "probe_gold()",
        },
        # --- invalid: non-positive horizon ---
        {
            "setup": _setup + """
def run_model():
    try:
        transform_strip_boundary(0.0, chiJ, p, M, G, a_ts, a_exc, kappa, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_transform_strip_boundary(0.0, chiJ, p, M, G, a_ts, a_exc,
                                         kappa, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: tempering rate at or below the positive-tail bound ---
        {
            "setup": _setup + """
def run_model():
    try:
        transform_strip_boundary(T, chiJ, p, 1.0, G, a_ts, a_exc, kappa, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_transform_strip_boundary(T, chiJ, p, 1.0, G, a_ts, a_exc,
                                         kappa, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-positive mean-reversion rate ---
        {
            "setup": _setup + """
def run_model():
    try:
        transform_strip_boundary(T, chiJ, p, M, G, a_ts, a_exc, 0.0, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_transform_strip_boundary(T, chiJ, p, M, G, a_ts, a_exc, 0.0,
                                         eta)
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
