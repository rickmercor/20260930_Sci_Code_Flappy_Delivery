"""
Evaluate the conditional characteristic function of the terminal log-price of the endogenous-activity jump model, on a horizontal line in the complex frequency plane.

The model couples a log-price carrying a Brownian component and an infinite-activity jump component whose predictable activity scale is itself driven by the asset's own realized price jumps, through a bounded excitation of the realized jump variation. The joint state of log-price and activity is affine, so its conditional Fourier-Laplace transform is exponential-affine in the state, with coefficient functions solving a system of ordinary differential equations obtained from the infinitesimal generator of the joint process. The log-price drift is not free: it is fixed by the requirement that the discounted stock price be a local martingale.

The argument u_line is a strictly increasing array of n non-negative real frequencies. Write the transform in the convention in which it is the expectation of the exponential of i times the argument times the terminal log-price. In that convention this step evaluates it at the argument whose real part is the negative of the entry of u_line and whose imaginary part is the negative of the damping level alpha. Both signs are part of the contract: they are the ones under which these values feed the density coefficients of a later step without further conjugation or reflection.

Evaluate every Levy integral with the split protocol of the shape-integral step: reflection onto the positive half-line, truncation at y_cut = 10, a 128-node Gauss-Jacobi rule on (0, 1) carrying the endpoint weight induced by the near-origin behaviour of the compensated integrands, and a 128-node Gauss-Legendre rule on the tail. In this step's own integrand, as in the drift-restriction integral supplied to it, only jumps of size below one are compensated. The excitation function is g(y) = 1 - exp(-a_exc * y**2), with a_exc > 0.

Integrate the coefficient system over the interval [0, T] to a relative accuracy of at least 1e-11. The integration scheme is not constrained.

Return the transform values packed as a two-row real array: row 0 the real parts, row 1 the imaginary parts, in the order of u_line.

Raises ValueError if u_line is not a one-dimensional array of non-negative, strictly increasing values; if T <= 0; if lam0 < 0; if sigma < 0; if S0 <= 0; or if the coefficient system fails to remain bounded on [0, T], which indicates that the requested damping level lies outside the admissible strip.

A stochastic process is called affine when the logarithm of its conditional Fourier-Laplace transform is an affine function of the current state. For such processes the transform is available without knowing the density: the coefficients multiplying each state variable satisfy a coupled system of ordinary differential equations, obtained by applying the infinitesimal generator to an exponential-affine test function and matching the result against the backward equation. The system is of generalized Riccati type, meaning the equations are quadratic or, for models with jumps, carry an integral of an exponential of the coefficients against the jump measure.

Self-exciting jump models have traditionally been built from point processes, where a discrete event raises the intensity of future events and the resulting intensity dynamics remain affine. That construction does not transfer directly to infinite-activity returns, where there are no countable events. Replacing event counts by a bounded function of realized jump size preserves the affine structure while allowing the driving jump component to have infinitely many jumps on every interval. The price paid is that the equation for the activity coefficient now contains a Levy integral of an exponential in that coefficient, so it is genuinely nonlinear and has no closed-form solution.

Two structural features of such a system matter numerically. First, the drift of the log-price is not a free parameter under a pricing measure: it must offset the compensated exponential jump integral, and this restriction appears explicitly in the coefficient equations. Second, well-posedness is not automatic. For real frequency arguments one can show that the activity coefficient stays in a half-plane where the exponential inside the Levy integral remains bounded, which gives global existence. Once the argument acquires an imaginary part, that argument no longer applies: the exponential of the jump size can grow faster than the tempering damps it, the coefficient can escape the half-plane, and the solution can blow up in finite time. A solver evaluating the transform off the real axis therefore has to treat unboundedness as a possible outcome rather than an impossibility.

Returns
-------
np.ndarray of shape (2, n), the real and imaginary parts of the transform on the damped line as native float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def activity_riccati_transform(u_line: 'np.ndarray', alpha: float, T: float,
                               chiJ: float, p: float, M: float, G: float,
                               a_ts: float, a_exc: float, kappa: float,
                               lam_bar: float, eta: float, lam0: float,
                               r: float, sigma: float,
                               S0: float) -> 'np.ndarray':
    '''Evaluate the conditional characteristic function on a damped line.

    Parameters
    ----------
    u_line : np.ndarray
        Shape (n,), strictly increasing non-negative real frequencies.
    alpha : float
        Damping level fixing the imaginary part of the transform argument.
    T : float
        Horizon, positive.
    chiJ : float
        Drift-restriction integral from the shape-integral step.
    p, M, G, a_ts, a_exc : float
        Jump-size shape and excitation parameters.
    kappa, lam_bar, eta, lam0 : float
        Activity mean-reversion rate, long-run level, feedback strength and
        initial level.
    r, sigma, S0 : float
        Risk-free rate, Brownian volatility and initial spot price.

    Returns
    -------
    result : np.ndarray
        Shape (2, n): row 0 real parts, row 1 imaginary parts.

    Raises
    ------
    ValueError
        If u_line is not a non-empty one-dimensional array of non-negative,
        strictly increasing values; if T <= 0; if lam0 < 0; if sigma < 0; if
        S0 <= 0; or if the coefficient system fails to remain bounded on
        [0, T], which indicates that the requested damping level lies outside
        the admissible strip.
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


def _h_rhs_activity(u_arg, psi, chiJ, kappa, eta, ys, yt, dens,
                    w_sp, w_tp, w_sm, w_tm):
    """Right-hand side of the activity coefficient equation, vectorised."""
    iu = 1j * u_arg
    e_sp = np.exp(np.multiply.outer(iu, ys)
                  + eta * np.multiply.outer(psi, dens["gs"]))
    e_tp = np.exp(np.multiply.outer(iu, yt)
                  + eta * np.multiply.outer(psi, dens["gt"]))
    e_sm = np.exp(np.multiply.outer(-iu, ys)
                  + eta * np.multiply.outer(psi, dens["gs"]))
    e_tm = np.exp(np.multiply.outer(-iu, yt)
                  + eta * np.multiply.outer(psi, dens["gt"]))
    jump = (np.einsum("ij,j->i", e_sp - 1.0 - np.multiply.outer(iu, ys), w_sp)
            + np.einsum("ij,j->i", e_tp - 1.0, w_tp)
            + np.einsum("ij,j->i", e_sm - 1.0 + np.multiply.outer(iu, ys), w_sm)
            + np.einsum("ij,j->i", e_tm - 1.0, w_tm))
    return -kappa * psi - iu * chiJ + jump


def _h_rhs_constant(u_arg, psi, kappa, lam_bar, r, sigma):
    """Right-hand side of the constant coefficient equation."""
    return (1j * u_arg * (r - 0.5 * sigma ** 2)
            - 0.5 * sigma ** 2 * u_arg * u_arg
            + kappa * lam_bar * psi)


def _oracle_activity_riccati_transform(u_line: 'np.ndarray', alpha: float,
                                       T: float, chiJ: float, p: float,
                                       M: float, G: float, a_ts: float,
                                       a_exc: float, kappa: float,
                                       lam_bar: float, eta: float,
                                       lam0: float, r: float, sigma: float,
                                       S0: float) -> 'np.ndarray':
    """Reference implementation."""
    u = np.asarray(u_line, dtype=float)
    if u.ndim != 1 or u.size < 1:
        raise ValueError("u_line must be a non-empty one-dimensional array")
    if np.any(u < 0.0):
        raise ValueError("u_line entries must be non-negative")
    if u.size > 1 and np.any(np.diff(u) <= 0.0):
        raise ValueError("u_line must be strictly increasing")
    if float(T) <= 0.0:
        raise ValueError("T must be positive")
    if float(lam0) < 0.0:
        raise ValueError("lam0 must be non-negative")
    if float(sigma) < 0.0:
        raise ValueError("sigma must be non-negative")
    if float(S0) <= 0.0:
        raise ValueError("S0 must be positive")

    ys, ws, yt, wt = _h_nodes(float(a_ts))
    dens = _h_density(float(p), float(M), float(G), float(a_ts),
                      float(a_exc), ys, yt)
    w_sp = ws * dens["sp"] / ys ** 2
    w_tp = wt * dens["tp"]
    w_sm = ws * dens["sm"] / ys ** 2
    w_tm = wt * dens["tm"]

    u_arg = -u - 1j * float(alpha)
    n_steps = 200
    h = float(T) / n_steps
    psi = np.zeros_like(u_arg)
    phi = np.zeros_like(u_arg)

    for _ in range(n_steps):
        k1p = _h_rhs_activity(u_arg, psi, chiJ, kappa, eta, ys, yt, dens,
                              w_sp, w_tp, w_sm, w_tm)
        k1c = _h_rhs_constant(u_arg, psi, kappa, lam_bar, r, sigma)
        k2p = _h_rhs_activity(u_arg, psi + 0.5 * h * k1p, chiJ, kappa, eta,
                              ys, yt, dens, w_sp, w_tp, w_sm, w_tm)
        k2c = _h_rhs_constant(u_arg, psi + 0.5 * h * k1p, kappa, lam_bar,
                              r, sigma)
        k3p = _h_rhs_activity(u_arg, psi + 0.5 * h * k2p, chiJ, kappa, eta,
                              ys, yt, dens, w_sp, w_tp, w_sm, w_tm)
        k3c = _h_rhs_constant(u_arg, psi + 0.5 * h * k2p, kappa, lam_bar,
                              r, sigma)
        k4p = _h_rhs_activity(u_arg, psi + h * k3p, chiJ, kappa, eta,
                              ys, yt, dens, w_sp, w_tp, w_sm, w_tm)
        k4c = _h_rhs_constant(u_arg, psi + h * k3p, kappa, lam_bar, r, sigma)
        phi = phi + h / 6.0 * (k1c + 2.0 * k2c + 2.0 * k3c + k4c)
        psi = psi + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        if not np.all(np.isfinite(psi)) or np.max(psi.real) > 50.0:
            raise ValueError(
                "coefficient system unbounded: damping level outside the "
                "admissible strip")

    value = np.exp(1j * u_arg * math.log(float(S0)) + phi + psi * float(lam0))
    if not np.all(np.isfinite(value)):
        raise ValueError(
            "coefficient system unbounded: damping level outside the "
            "admissible strip")
    return np.vstack([value.real, value.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _setup = """import numpy as np
p, M, G, a_ts, a_exc = 0.55, 6.5, 3.0, 1.1, 0.8
kappa, lam_bar, eta, lam0 = 3.5, 0.015, 2.0, 0.02
r, sigma, S0, T = 0.03, 0.05, 95.0, 0.25
chiJ = 0.4803150971
"""
    return [
        # --- normal: the locked damping level on a short frequency line ---
        {
            "setup": _setup + "u = np.linspace(0.0, 12.0, 7)\nalpha = 4.3317046485\n",
            "call": ("activity_riccati_transform(u, alpha, T, chiJ, p, M, G, "
                     "a_ts, a_exc, kappa, lam_bar, eta, lam0, r, sigma, S0)"),
            "gold_call": ("_oracle_activity_riccati_transform(u, alpha, T, "
                          "chiJ, p, M, G, a_ts, a_exc, kappa, lam_bar, eta, "
                          "lam0, r, sigma, S0)"),
        },
        # --- boundary: zero damping, transform on the real axis ---
        {
            "setup": _setup + "u = np.array([0.0, 1.0, 25.0])\nalpha = 0.0\n",
            "call": ("activity_riccati_transform(u, alpha, T, chiJ, p, M, G, "
                     "a_ts, a_exc, kappa, lam_bar, eta, lam0, r, sigma, S0)"),
            "gold_call": ("_oracle_activity_riccati_transform(u, alpha, T, "
                          "chiJ, p, M, G, a_ts, a_exc, kappa, lam_bar, eta, "
                          "lam0, r, sigma, S0)"),
        },
        # --- edge: no feedback, so the activity coefficient equation is
        #     linear and the transform reduces to the exogenous-activity case ---
        {
            "setup": _setup + "u = np.array([0.0, 3.5, 40.0])\nalpha = 2.0\n",
            "call": ("activity_riccati_transform(u, alpha, T, chiJ, p, M, G, "
                     "a_ts, a_exc, kappa, lam_bar, 0.0, lam0, r, sigma, S0)"),
            "gold_call": ("_oracle_activity_riccati_transform(u, alpha, T, "
                          "chiJ, p, M, G, a_ts, a_exc, kappa, lam_bar, 0.0, "
                          "lam0, r, sigma, S0)"),
        },
        # --- edge: zero initial activity, so only the constant coefficient
        #     contributes to the exponent ---
        {
            "setup": _setup + "u = np.array([0.0, 8.0])\nalpha = 1.5\n",
            "call": ("activity_riccati_transform(u, alpha, T, chiJ, p, M, G, "
                     "a_ts, a_exc, kappa, lam_bar, eta, 0.0, r, sigma, S0)"),
            "gold_call": ("_oracle_activity_riccati_transform(u, alpha, T, "
                          "chiJ, p, M, G, a_ts, a_exc, kappa, lam_bar, eta, "
                          "0.0, r, sigma, S0)"),
        },
        # --- structural probe: at damping level one and zero frequency the
        #     drift restriction forces the value to the forward price ---
        {
            "setup": _setup + """
def probe():
    v = activity_riccati_transform(np.array([0.0]), 1.0, T, chiJ, p, M, G,
                                   a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                   r, sigma, S0)
    fwd = S0 * np.exp(r * T)
    return [int(round(1e9 * abs(float(v[0, 0]) - fwd))),
            int(round(1e9 * abs(float(v[1, 0]))))]
def probe_gold():
    v = _oracle_activity_riccati_transform(np.array([0.0]), 1.0, T, chiJ, p,
                                           M, G, a_ts, a_exc, kappa, lam_bar,
                                           eta, lam0, r, sigma, S0)
    fwd = S0 * np.exp(r * T)
    return [int(round(1e9 * abs(float(v[0, 0]) - fwd))),
            int(round(1e9 * abs(float(v[1, 0]))))]
""",
            "call": "probe()",
            "gold_call": "probe_gold()",
        },
        # --- invalid: damping level beyond the admissible strip ---
        {
            "setup": _setup + """
def run_model():
    try:
        activity_riccati_transform(np.array([0.0]), 5.7, T, chiJ, p, M, G,
                                   a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                   r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_activity_riccati_transform(np.array([0.0]), 5.7, T, chiJ, p,
                                           M, G, a_ts, a_exc, kappa, lam_bar,
                                           eta, lam0, r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: frequency line not strictly increasing ---
        {
            "setup": _setup + """
def run_model():
    try:
        activity_riccati_transform(np.array([1.0, 0.5]), 2.0, T, chiJ, p, M,
                                   G, a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                   r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_activity_riccati_transform(np.array([1.0, 0.5]), 2.0, T, chiJ,
                                           p, M, G, a_ts, a_exc, kappa,
                                           lam_bar, eta, lam0, r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: negative frequency entry ---
        {
            "setup": _setup + """
def run_model():
    try:
        activity_riccati_transform(np.array([-1.0, 2.0]), 2.0, T, chiJ, p, M,
                                   G, a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                   r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_activity_riccati_transform(np.array([-1.0, 2.0]), 2.0, T,
                                           chiJ, p, M, G, a_ts, a_exc, kappa,
                                           lam_bar, eta, lam0, r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-positive horizon ---
        {
            "setup": _setup + """
def run_model():
    try:
        activity_riccati_transform(np.array([0.0]), 2.0, 0.0, chiJ, p, M, G,
                                   a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                   r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_activity_riccati_transform(np.array([0.0]), 2.0, 0.0, chiJ, p,
                                           M, G, a_ts, a_exc, kappa, lam_bar,
                                           eta, lam0, r, sigma, S0)
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
