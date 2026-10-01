"""
Evaluate the scalar autonomous driver that governs the activity coefficient function when the transform of the previous step is evaluated at a purely imaginary frequency argument.



At such an argument the coefficient system collapses to a single real autonomous ordinary differential equation in the activity coefficient alone. Its driver is the generator-induced expression already used in the full system, specialised to a real exponent, and its sign structure decides whether the exponential moment of the terminal stock price at that exponent stays finite over a given horizon.



Evaluate every Levy integral with the split protocol of the shape-integral step: reflection onto the positive half-line, truncation at y_cut = 20, a 128-node Gauss-Jacobi rule on (0, 1) carrying the endpoint weight induced by the near-origin behaviour of the compensated integrands, and a 224-node Gauss-Legendre rule on the tail. This step truncates at 20 rather than at the 10 used by the shape-integral step, because the feedback factor exp(eta * x * g(y)) multiplies the dropped tail by the same amount as the part that is kept, so y_cut = 10 does not reach the relative accuracy of 1e-9 that this step requires. In this step's own integrand, as in the drift-restriction integral supplied to it, only jumps of size below one are compensated. The excitation function is g(y) = 1 - exp(-a_exc * y**2), with a_exc > 0.



Return the driver evaluated at each point of psi_grid, in the order supplied.



Raises ValueError if psi_grid is not a non-empty one-dimensional array; if any entry of psi_grid is negative; if alpha <= 0; if kappa <= 0; or if eta < 0.

When a system of ordinary differential equations has a distinguished invariant structure, specialising its argument can reduce it to something far simpler than the general case. For an exponential-affine transform the equations couple a constant coefficient to one coefficient per state variable, and the constant coefficient is obtained by quadrature once the others are known. If in addition the frequency argument is chosen so that every complex quantity in the remaining equation becomes real, what is left is a single scalar equation with no explicit time dependence.

Scalar autonomous equations are completely understood by inspecting the sign of their driver. A solution started at the origin increases as long as the driver is positive there, and its fate is decided by whether the driver has a zero above the starting point. If it does, the solution rises to that zero and stops, remaining bounded for all time. If the driver stays strictly positive, the solution increases without bound, and whether it reaches infinity in finite time is governed by how fast the driver grows, since the time to blow up is the integral of the reciprocal of the driver along the trajectory.

For an affine model with jumps this dichotomy has a direct probabilistic meaning. The transform evaluated at a purely imaginary argument is an exponential moment of the terminal price, so boundedness of the coefficient over a given horizon is exactly finiteness of that moment. The driver in this setting has a linear mean-reverting part pulling the coefficient down, a constant contribution from the drift restriction, and a Levy integral in which the exponent contains both the exponential moment argument and the feedback term. Because the feedback enters inside an exponential, the driver is convex and can lose its root as the exponent grows, and the exponent at which the root is lost separates a regime of finite moments at all horizons from one where the moment is finite only up to a horizon that depends on the exponent.

Returns
-------
np.ndarray of shape (n,), the scalar driver evaluated at each grid point as native float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def moment_growth_driver(psi_grid: 'np.ndarray', alpha: float, chiJ: float,
                         p: float, M: float, G: float, a_ts: float,
                         a_exc: float, kappa: float,
                         eta: float) -> 'np.ndarray':
    '''Evaluate the scalar autonomous driver of the activity coefficient.

    Parameters
    ----------
    psi_grid : np.ndarray
        Shape (n,), non-negative activity-coefficient values.
    alpha : float
        Real exponent at which the driver is specialised, positive.
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
    result : np.ndarray
        Shape (n,), the driver at each grid point, in the order supplied.

    Raises
    ------
    ValueError
        If psi_grid is not a non-empty one-dimensional array; if any entry of
        psi_grid is negative; if alpha <= 0; if kappa <= 0; or if eta < 0.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_moment_growth_driver(psi_grid: 'np.ndarray', alpha: float,
                                 chiJ: float, p: float, M: float, G: float,
                                 a_ts: float, a_exc: float, kappa: float,
                                 eta: float) -> 'np.ndarray':
    """Reference implementation."""
    xs = np.asarray(psi_grid, dtype=float)
    if xs.ndim != 1 or xs.size < 1:
        raise ValueError("psi_grid must be a non-empty one-dimensional array")
    if np.any(xs < 0.0):
        raise ValueError("psi_grid entries must be non-negative")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive")
    if float(kappa) <= 0.0:
        raise ValueError("kappa must be positive")
    if float(eta) < 0.0:
        raise ValueError("eta must be non-negative")

    a = float(alpha)
    ys, ws, yt, wt = _h_nodes(float(a_ts), n_tail=224, y_cut=20.0)
    dens = _h_density(float(p), float(M), float(G), float(a_ts),
                      float(a_exc), ys, yt)
    w_sp = ws * dens["sp"] / ys ** 2
    w_tp = wt * dens["tp"]
    w_sm = ws * dens["sm"] / ys ** 2
    w_tm = wt * dens["tm"]

    exc_s = float(eta) * np.multiply.outer(xs, dens["gs"])
    exc_t = float(eta) * np.multiply.outer(xs, dens["gt"])
    jump = ((np.exp(a * ys + exc_s) - 1.0 - a * ys) @ w_sp
            + (np.exp(a * yt + exc_t) - 1.0) @ w_tp
            + (np.exp(-a * ys + exc_s) - 1.0 + a * ys) @ w_sm
            + (np.exp(-a * yt + exc_t) - 1.0) @ w_tm)
    return -float(kappa) * xs - a * float(chiJ) + jump

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _setup = """import numpy as np
p, M, G, a_ts, a_exc = 0.55, 6.5, 3.0, 1.1, 0.8
kappa, eta = 3.5, 2.0
chiJ = 0.4803150971
"""
    return [
        # --- normal: the locked damping level over a moderate grid ---
        {
            "setup": _setup + "grid = np.linspace(0.0, 6.0, 25)\nalpha = 4.3317046485\n",
            "call": ("moment_growth_driver(grid, alpha, chiJ, p, M, G, a_ts, "
                     "a_exc, kappa, eta)"),
            "gold_call": ("_oracle_moment_growth_driver(grid, alpha, chiJ, p, "
                          "M, G, a_ts, a_exc, kappa, eta)"),
        },
        # --- boundary: exponent near the value at which the driver loses its
        #     positive root, over a wide grid ---
        {
            "setup": _setup + "grid = np.linspace(0.0, 20.0, 41)\nalpha = 2.9156319\n",
            "call": ("moment_growth_driver(grid, alpha, chiJ, p, M, G, a_ts, "
                     "a_exc, kappa, eta)"),
            "gold_call": ("_oracle_moment_growth_driver(grid, alpha, chiJ, p, "
                          "M, G, a_ts, a_exc, kappa, eta)"),
        },
        # --- edge: zero feedback makes the driver affine in its argument ---
        {
            "setup": _setup + "grid = np.array([0.0, 1.0, 5.0])\n",
            "call": ("moment_growth_driver(grid, 2.0, chiJ, p, M, G, a_ts, "
                     "a_exc, kappa, 0.0)"),
            "gold_call": ("_oracle_moment_growth_driver(grid, 2.0, chiJ, p, M, "
                          "G, a_ts, a_exc, kappa, 0.0)"),
        },
        # --- edge: single-point grid at the origin, where the drift
        #     restriction forces the driver to vanish at exponent one ---
        {
            "setup": _setup + "grid = np.array([0.0])\n",
            "call": ("moment_growth_driver(grid, 1.0, chiJ, p, M, G, a_ts, "
                     "a_exc, kappa, eta)"),
            "gold_call": ("_oracle_moment_growth_driver(grid, 1.0, chiJ, p, M, "
                          "G, a_ts, a_exc, kappa, eta)"),
        },
        # --- structural probe: whether the driver stays strictly positive
        #     across a range of exponents, and whether it is affine when the
        #     feedback is switched off, as exact integers ---
        {
            "setup": _setup + """
def probe():
    g = np.linspace(0.0, 20.0, 201)
    signs = [int(min(moment_growth_driver(g, a, chiJ, p, M, G, a_ts, a_exc,
                                          kappa, eta)) > 0.0)
             for a in [2.0, 2.5, 3.2, 4.3317046485]]
    lin = moment_growth_driver(np.array([0.0, 1.0, 2.0]), 2.0, chiJ, p, M, G,
                               a_ts, a_exc, kappa, 0.0)
    affine = int(round(1e9 * abs(float(lin[0] + lin[2] - 2.0 * lin[1]))))
    return signs + [affine]
def probe_gold():
    g = np.linspace(0.0, 20.0, 201)
    signs = [int(min(_oracle_moment_growth_driver(g, a, chiJ, p, M, G, a_ts,
                                                  a_exc, kappa, eta)) > 0.0)
             for a in [2.0, 2.5, 3.2, 4.3317046485]]
    lin = _oracle_moment_growth_driver(np.array([0.0, 1.0, 2.0]), 2.0, chiJ,
                                       p, M, G, a_ts, a_exc, kappa, 0.0)
    affine = int(round(1e9 * abs(float(lin[0] + lin[2] - 2.0 * lin[1]))))
    return signs + [affine]
""",
            "call": "probe()",
            "gold_call": "probe_gold()",
        },
        # --- invalid: negative entry in the grid ---
        {
            "setup": _setup + """
def run_model():
    try:
        moment_growth_driver(np.array([-1.0, 0.0]), 2.0, chiJ, p, M, G, a_ts,
                             a_exc, kappa, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_moment_growth_driver(np.array([-1.0, 0.0]), 2.0, chiJ, p, M,
                                     G, a_ts, a_exc, kappa, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-positive exponent ---
        {
            "setup": _setup + """
def run_model():
    try:
        moment_growth_driver(np.array([0.0, 1.0]), 0.0, chiJ, p, M, G, a_ts,
                             a_exc, kappa, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_moment_growth_driver(np.array([0.0, 1.0]), 0.0, chiJ, p, M, G,
                                     a_ts, a_exc, kappa, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: negative feedback strength ---
        {
            "setup": _setup + """
def run_model():
    try:
        moment_growth_driver(np.array([0.0, 1.0]), 2.0, chiJ, p, M, G, a_ts,
                             a_exc, kappa, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_moment_growth_driver(np.array([0.0, 1.0]), 2.0, chiJ, p, M, G,
                                     a_ts, a_exc, kappa, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: two-dimensional grid ---
        {
            "setup": _setup + """
def run_model():
    try:
        moment_growth_driver(np.zeros((2, 2)), 2.0, chiJ, p, M, G, a_ts,
                             a_exc, kappa, eta)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_moment_growth_driver(np.zeros((2, 2)), 2.0, chiJ, p, M, G,
                                     a_ts, a_exc, kappa, eta)
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
