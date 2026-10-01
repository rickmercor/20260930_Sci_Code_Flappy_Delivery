"""
Fix the damping level for the pricing method and report the value of the parameter-selection objective there.

The damping level is placed at the fraction frac of the way from the lower edge of the admissible set, which for a European call payoff is the exponent one, to the finite-horizon boundary supplied as alpha_expl. The objective reported alongside it is the objective of the damped wavelet method's own parameter-selection algorithm, evaluated at that damping level; it is diagnostic only and does not influence the returned damping level.

That objective is the product of two masses. One is the mass of the damped payoff, that is the integral of the damped call payoff over the real line. The other is the mass of the damped density, that is the exponential moment of the terminal stock price at the damping level, which this step obtains by evaluating the transform at zero frequency using the same coefficient system, Levy quadrature protocol and accuracy requirement as the transform step. In this step's own integrand, as in the drift-restriction integral supplied to it, only jumps of size below one are compensated. The excitation function is g(y) = 1 - exp(-a_exc * y**2), with a_exc > 0.

Return the two values in the order: damping level, objective value.

Raises ValueError if frac is not in (0,1); if alpha_expl <= 1; if K <= 0; or if the resulting damping level lies outside the admissible strip.

Damping a payoff by a decaying exponential and the density by the reciprocal weight leaves the price unchanged, since the two factors cancel in the valuation integral. What it does change is the numerical behaviour of the frequency-domain representation. The damped payoff and the damped density each have a Fourier transform, and the coefficients of the wavelet expansion are finite-interval integrals of those transforms. A poorly chosen damping level makes one of the two integrands very large and highly peaked near the origin, which degrades the conditioning of the quadrature even though the exact answer is unaffected.

This motivates choosing the damping level by minimising a measure of that peak size. A convenient proxy is the product of the two masses, each being the corresponding transform evaluated at zero frequency, which for non-negative factors is just the product of their integrals. Both masses are finite only inside the admissible set. Near the lower edge the damped call payoff decays too slowly at large log-prices and its mass diverges; near the upper edge the damped density fails to be integrable and its mass diverges. The objective therefore blows up at both ends of the admissible set and attains an interior minimum somewhere between them.

Where that minimum falls is a property of the particular model and contract, not a universal one. For contracts whose payoff mass diverges sharply at the lower edge while the density mass stays moderate over most of the range, the objective can fall steeply away from the lower edge and remain nearly flat over the bulk of the admissible set, turning upward only in a narrow region close to the upper edge. In such cases the minimiser sits very near the boundary, and locating it accurately requires resolving the objective in a region where the density mass is already growing rapidly. Fixing the damping level by a prescribed rule instead, and reporting the objective only as a diagnostic, avoids making the final answer depend on the conditioning of that search.

Returns
-------
np.ndarray of shape (2,), the damping level and the selection objective value as native float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def damping_parameter(alpha_expl: float, frac: float, K: float, T: float,
                      chiJ: float, p: float, M: float, G: float, a_ts: float,
                      a_exc: float, kappa: float, lam_bar: float, eta: float,
                      lam0: float, r: float, sigma: float,
                      S0: float) -> 'np.ndarray':
    '''Fix the damping level and report the selection objective there.

    Parameters
    ----------
    alpha_expl : float
        Finite-horizon exponent boundary, greater than 1.
    frac : float
        Fraction of the way from the lower edge to the boundary, in (0,1).
    K : float
        Strike, positive.
    T : float
        Horizon, positive.
    chiJ : float
        Drift-restriction integral from the shape-integral step.
    p, M, G, a_ts, a_exc, kappa, lam_bar, eta, lam0, r, sigma, S0 : float
        Model and market parameters.

    Returns
    -------
    result : np.ndarray
        Shape (2,): damping level, objective value, in that order.

    Raises
    ------
    ValueError
        If frac is not in (0,1); if alpha_expl <= 1; if K <= 0; or if the
        resulting damping level lies outside the admissible strip.
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


def _h_exponential_moment(alpha, T, chiJ, p, M, G, a_ts, a_exc, kappa,
                          lam_bar, eta, lam0, r, sigma, S0):
    """Exponential moment of the terminal price at the given exponent."""
    ys, ws, yt, wt = _h_nodes(a_ts)
    dens = _h_density(p, M, G, a_ts, a_exc, ys, yt)
    w_sp = ws * dens["sp"] / ys ** 2
    w_tp = wt * dens["tp"]
    w_sm = ws * dens["sm"] / ys ** 2
    w_tm = wt * dens["tm"]
    u_arg = np.array([-1j * float(alpha)])

    def _rhs_activity(psi):
        iu = 1j * u_arg
        e_sp = np.exp(np.multiply.outer(iu, ys)
                      + eta * np.multiply.outer(psi, dens["gs"]))
        e_tp = np.exp(np.multiply.outer(iu, yt)
                      + eta * np.multiply.outer(psi, dens["gt"]))
        e_sm = np.exp(np.multiply.outer(-iu, ys)
                      + eta * np.multiply.outer(psi, dens["gs"]))
        e_tm = np.exp(np.multiply.outer(-iu, yt)
                      + eta * np.multiply.outer(psi, dens["gt"]))
        jump = ((e_sp - 1.0 - np.multiply.outer(iu, ys)) @ w_sp
                + (e_tp - 1.0) @ w_tp
                + (e_sm - 1.0 + np.multiply.outer(iu, ys)) @ w_sm
                + (e_tm - 1.0) @ w_tm)
        return -kappa * psi - iu * chiJ + jump

    def _rhs_constant(psi):
        return (1j * u_arg * (r - 0.5 * sigma ** 2)
                - 0.5 * sigma ** 2 * u_arg * u_arg + kappa * lam_bar * psi)

    n_steps = 4000
    h = float(T) / n_steps
    psi = np.zeros_like(u_arg)
    phi = np.zeros_like(u_arg)
    for _ in range(n_steps):
        k1p, k1c = _rhs_activity(psi), _rhs_constant(psi)
        k2p = _rhs_activity(psi + 0.5 * h * k1p)
        k2c = _rhs_constant(psi + 0.5 * h * k1p)
        k3p = _rhs_activity(psi + 0.5 * h * k2p)
        k3c = _rhs_constant(psi + 0.5 * h * k2p)
        k4p = _rhs_activity(psi + h * k3p)
        k4c = _rhs_constant(psi + h * k3p)
        phi = phi + h / 6.0 * (k1c + 2.0 * k2c + 2.0 * k3c + k4c)
        psi = psi + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
        if not np.all(np.isfinite(psi)) or np.max(psi.real) > 50.0:
            raise ValueError(
                "damping level lies outside the admissible strip")
    value = np.exp(1j * u_arg * math.log(float(S0)) + phi + psi * float(lam0))
    if not np.all(np.isfinite(value)):
        raise ValueError("damping level lies outside the admissible strip")
    return float(abs(value[0]))


def _oracle_damping_parameter(alpha_expl: float, frac: float, K: float,
                              T: float, chiJ: float, p: float, M: float,
                              G: float, a_ts: float, a_exc: float,
                              kappa: float, lam_bar: float, eta: float,
                              lam0: float, r: float, sigma: float,
                              S0: float) -> 'np.ndarray':
    """Reference implementation."""
    if not (0.0 < float(frac) < 1.0):
        raise ValueError("frac must lie in (0,1)")
    if float(alpha_expl) <= 1.0:
        raise ValueError("alpha_expl must exceed 1")
    if float(K) <= 0.0:
        raise ValueError("K must be positive")

    alpha = 1.0 + float(frac) * (float(alpha_expl) - 1.0)
    payoff_mass = abs(float(K) ** (1.0 - alpha) / (alpha * (alpha - 1.0)))
    density_mass = _h_exponential_moment(
        alpha, float(T), float(chiJ), float(p), float(M), float(G),
        float(a_ts), float(a_exc), float(kappa), float(lam_bar), float(eta),
        float(lam0), float(r), float(sigma), float(S0))
    return np.array([alpha, payoff_mass * density_mass], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _setup = """import numpy as np
p, M, G, a_ts, a_exc = 0.55, 6.5, 3.0, 1.1, 0.8
kappa, lam_bar, eta, lam0 = 3.5, 0.015, 2.0, 0.02
r, sigma, S0, K, T = 0.03, 0.05, 95.0, 98.0, 0.25
chiJ = 0.4803150971
alpha_expl = 4.9196525277
"""
    return [
        # --- normal: the locked fraction ---
        {
            "setup": _setup,
            "call": ("damping_parameter(alpha_expl, 0.85, K, T, chiJ, p, M, G, "
                     "a_ts, a_exc, kappa, lam_bar, eta, lam0, r, sigma, S0)"),
            "gold_call": ("_oracle_damping_parameter(alpha_expl, 0.85, K, T, "
                          "chiJ, p, M, G, a_ts, a_exc, kappa, lam_bar, eta, "
                          "lam0, r, sigma, S0)"),
        },
        # --- boundary: a fraction close to zero puts the damping level just
        #     above the lower edge, where the payoff mass grows without bound ---
        {
            "setup": _setup,
            "call": ("damping_parameter(alpha_expl, 0.005, K, T, chiJ, p, M, "
                     "G, a_ts, a_exc, kappa, lam_bar, eta, lam0, r, sigma, "
                     "S0)"),
            "gold_call": ("_oracle_damping_parameter(alpha_expl, 0.005, K, T, "
                          "chiJ, p, M, G, a_ts, a_exc, kappa, lam_bar, eta, "
                          "lam0, r, sigma, S0)"),
        },
        # --- boundary: a fraction close to one puts it just inside the upper
        #     edge, where the density mass begins to grow ---
        {
            "setup": _setup,
            "call": ("damping_parameter(alpha_expl, 0.995, K, T, chiJ, p, M, "
                     "G, a_ts, a_exc, kappa, lam_bar, eta, lam0, r, sigma, "
                     "S0)"),
            "gold_call": ("_oracle_damping_parameter(alpha_expl, 0.995, K, T, "
                          "chiJ, p, M, G, a_ts, a_exc, kappa, lam_bar, eta, "
                          "lam0, r, sigma, S0)"),
        },
        # --- edge: no feedback and a narrow admissible set ---
        {
            "setup": _setup,
            "call": ("damping_parameter(2.6, 0.5, 1.0, T, chiJ, p, M, G, "
                     "a_ts, a_exc, kappa, lam_bar, 0.0, lam0, r, sigma, S0)"),
            "gold_call": ("_oracle_damping_parameter(2.6, 0.5, 1.0, T, chiJ, "
                          "p, M, G, a_ts, a_exc, kappa, lam_bar, 0.0, lam0, "
                          "r, sigma, S0)"),
        },
        # --- structural probe: the damping level is affine in the fraction,
        #     and the objective falls steeply away from the lower edge before
        #     turning up again only very close to the upper edge; returned as
        #     exact integers ---
        {
            "setup": _setup + """
def _j(f):
    return float(damping_parameter(alpha_expl, f, K, T, chiJ, p, M, G, a_ts,
                                   a_exc, kappa, lam_bar, eta, lam0, r,
                                   sigma, S0)[1])
def _a(f):
    return float(damping_parameter(alpha_expl, f, K, T, chiJ, p, M, G, a_ts,
                                   a_exc, kappa, lam_bar, eta, lam0, r,
                                   sigma, S0)[0])
def probe():
    affine = int(round(1e9 * abs(_a(0.02) + _a(0.98) - 2.0 * _a(0.50))))
    return [int(_j(0.02) > _j(0.50)), int(_j(0.50) > _j(0.95)),
            int(_j(0.999999) > _j(0.995)), affine,
            int(round(1e7 * _a(0.85)))]
def probe_gold():
    g02 = _oracle_damping_parameter(alpha_expl, 0.02, K, T, chiJ, p, M, G,
                                    a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                    r, sigma, S0)
    g50 = _oracle_damping_parameter(alpha_expl, 0.50, K, T, chiJ, p, M, G,
                                    a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                    r, sigma, S0)
    g85 = _oracle_damping_parameter(alpha_expl, 0.85, K, T, chiJ, p, M, G,
                                    a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                    r, sigma, S0)
    g95 = _oracle_damping_parameter(alpha_expl, 0.95, K, T, chiJ, p, M, G,
                                    a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                    r, sigma, S0)
    g98 = _oracle_damping_parameter(alpha_expl, 0.98, K, T, chiJ, p, M, G,
                                    a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                    r, sigma, S0)
    g995 = _oracle_damping_parameter(alpha_expl, 0.995, K, T, chiJ, p, M, G,
                                     a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                     r, sigma, S0)
    g999 = _oracle_damping_parameter(alpha_expl, 0.999999, K, T, chiJ, p, M, G,
                                     a_ts, a_exc, kappa, lam_bar, eta, lam0,
                                     r, sigma, S0)
    affine = int(round(1e9 * abs(float(g02[0]) + float(g98[0])
                                 - 2.0 * float(g50[0]))))
    return [int(float(g02[1]) > float(g50[1])),
            int(float(g50[1]) > float(g95[1])),
            int(float(g999[1]) > float(g995[1])), affine,
            int(round(1e7 * float(g85[0])))]
""",
            "call": "probe()",
            "gold_call": "probe_gold()",
        },
        # --- invalid: fraction at the upper end of its range ---
        {
            "setup": _setup + """
def run_model():
    try:
        damping_parameter(alpha_expl, 1.0, K, T, chiJ, p, M, G, a_ts, a_exc,
                          kappa, lam_bar, eta, lam0, r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_damping_parameter(alpha_expl, 1.0, K, T, chiJ, p, M, G, a_ts,
                                  a_exc, kappa, lam_bar, eta, lam0, r, sigma,
                                  S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: boundary at or below the lower edge ---
        {
            "setup": _setup + """
def run_model():
    try:
        damping_parameter(1.0, 0.85, K, T, chiJ, p, M, G, a_ts, a_exc, kappa,
                          lam_bar, eta, lam0, r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_damping_parameter(1.0, 0.85, K, T, chiJ, p, M, G, a_ts, a_exc,
                                  kappa, lam_bar, eta, lam0, r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: a boundary above the admissible strip drives the
        #     coefficient system out of the half-plane ---
        {
            "setup": _setup + """
def run_model():
    try:
        damping_parameter(6.4, 0.99, K, T, chiJ, p, M, G, a_ts, a_exc, kappa,
                          lam_bar, eta, lam0, r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_damping_parameter(6.4, 0.99, K, T, chiJ, p, M, G, a_ts, a_exc,
                                  kappa, lam_bar, eta, lam0, r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-positive strike ---
        {
            "setup": _setup + """
def run_model():
    try:
        damping_parameter(alpha_expl, 0.85, 0.0, T, chiJ, p, M, G, a_ts,
                          a_exc, kappa, lam_bar, eta, lam0, r, sigma, S0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_damping_parameter(alpha_expl, 0.85, 0.0, T, chiJ, p, M, G,
                                  a_ts, a_exc, kappa, lam_bar, eta, lam0, r,
                                  sigma, S0)
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
