"""
Chain the eight earlier steps into the whole calculation and return the value the method produces. Fix the scaling factor, compute the diagnostic level difference, select the level, allocate the points, then evaluate the level-zero term and each level difference with its own point count and add them. This is the orchestrator step: it chains the eight earlier public functions rather than reproducing any of them inline.

What the method returns is not the exact price and is not meant to be. It is the output of an error-controlled procedure at a prescribed tolerance, and every part of that procedure leaves a fingerprint on the number: which contour the integrand sits on, how the weight is scaled, how many levels the hierarchy has and how the points are shared among them. Asking for a converged price would erase all of that, since every reasonable variant converges to the same limit. Asking for the procedure's own output at a stated tolerance keeps the variants apart.

The arrangement is deliberately free of stopping rules and adaptivity. The tolerance is given, the split is the default one, the diagnostic point count is fixed, and the constants of the error models are supplied rather than fitted, so nothing depends on a convergence test that could terminate differently on a different machine. The only integers the procedure decides for itself are the level and the point counts, and both come from closed-form rules.

One evaluation of a level difference costs two solves, and the two must use the same quadrature nodes; evaluating the two levels on different rules would leave a remainder that does not telescope and the hierarchy would stop saving anything.

Returns
-------
float. For the task configuration the value is 94.7377956301. The pipeline is positively homogeneous of degree one when the contract, the tolerance and the quadrature constants are scaled together, and the value stays between zero and the spot price.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_multilevel_pricer(model, contract, numerics):
    """Chain the eight earlier steps and return the multilevel option value.

    Fixes the Laguerre scaling factor, computes the first level difference
    with the diagnostic number of quadrature points to obtain D1, selects the
    finest level L from it, allocates the quadrature points across the
    hierarchy, and evaluates the telescoping multilevel estimator

        V = Q_{N_0}[g_0] + sum_{l=1..L} Q_{N_l}[g_l - g_{l-1}],

    where Q_N is the scaled Gauss-Laguerre rule applied to the transformed
    integrand and already carries the factor two of the even-integrand
    reduction.  The same scaling factor is used at every level; only the
    number of points changes.  The per-node solver cost model is
    W_l = dt_l**(-beta), the level-zero algebraic constant and smoothness
    index are supplied, and the level-difference constants follow
    A_l = CA * dt_l**p with the common correction index s.

    This is the orchestrator step: chain the eight earlier public functions rather than reproducing any of them inline.

    Args:
        model: (6,) array [alpha, gam, nu, rho, V0, theta].
        contract: (5,) array [S0, K, T, r, R].
        numerics: (8,) array [dt0, eps, nbar, A0, s0, CA, s, beta].

    Expected return:
        float: the multilevel Fourier price V.
        Raises:
        ValueError: if model, contract and numerics do not have
        lengths 6, 5 and 8, or if S0 <= 0.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_multilevel_pricer(model, contract, numerics):
    def _level_integrand(u, level, alpha, dt0, T, gam, nu, rho, theta, V0,
                         X0, r, R):
        dt = dt0 * 2.0 ** (-level)
        M = int(round(T / dt))
        w = _oracle_adams_convolution_weights(alpha, dt, M)
        gr = np.empty(u.size)
        gi = np.empty(u.size)
        for i in range(u.size):
            hp = _oracle_solve_riccati_nodal(w, float(u[i]), R, alpha, dt, M,
                                             gam, nu, rho)
            G = _oracle_characteristic_exponent(hp, float(u[i]), R, dt, X0, r,
                                                gam, nu, rho, theta, V0)
            gr[i] = G[0]
            gi[i] = G[1]
        return gr, gi

    model = np.asarray(model, dtype=float).reshape(-1)
    contract = np.asarray(contract, dtype=float).reshape(-1)
    numerics = np.asarray(numerics, dtype=float).reshape(-1)
    if model.size != 6 or contract.size != 5 or numerics.size != 8:
        raise ValueError("model, contract and numerics must have sizes 6, 5, 8")

    alpha, gam, nu, rho, V0, theta = model
    S0, K, T, r, R = contract
    dt0, eps, nbar, A0, s0, CA, s, beta = numerics
    nbar = int(nbar)
    if S0 <= 0.0:
        raise ValueError("S0 must be positive")
    X0 = float(np.log(S0))
    p = 1.0 + alpha
    eps_disc = eps_quad = 0.5 * eps

    sigma = _oracle_laguerre_scaling(T, alpha, gam, nu, rho, theta, V0)

    def _value(N, level):
        rule = _oracle_scaled_laguerre_rule(int(N), sigma)
        n = int(N)
        u, wq = rule[:n], rule[n:]
        gr, gi = _level_integrand(u, level, alpha, dt0, T, gam, nu, rho,
                                  theta, V0, X0, r, R)
        g = _oracle_fourier_integrand(u, gr, gi, R, K, r, T)
        return 2.0 * float(np.sum(wq * np.exp(sigma * u) * g))

    D1 = abs(_value(nbar, 1) - _value(nbar, 0))
    L = int(_oracle_select_discretization_level(D1, eps_disc, p))

    dt = np.array([dt0 * 2.0 ** (-l) for l in range(L + 1)])
    W = dt ** (-beta)
    A = np.concatenate(([A0], CA * dt[1:] ** p))
    N = _oracle_allocate_quadrature_points(A, s0, s, eps_quad, W)

    total = _value(int(N[0]), 0)
    for l in range(1, L + 1):
        n = int(N[l])
        rule = _oracle_scaled_laguerre_rule(n, sigma)
        u, wq = rule[:n], rule[n:]
        gr_hi, gi_hi = _level_integrand(u, l, alpha, dt0, T, gam, nu, rho,
                                        theta, V0, X0, r, R)
        gr_lo, gi_lo = _level_integrand(u, l - 1, alpha, dt0, T, gam, nu, rho,
                                        theta, V0, X0, r, R)
        diff = (_oracle_fourier_integrand(u, gr_hi, gi_hi, R, K, r, T)
                - _oracle_fourier_integrand(u, gr_lo, gi_lo, R, K, r, T))
        total += 2.0 * float(np.sum(wq * np.exp(sigma * u) * diff))
    return float(total)

# =============================================================================

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

MODEL = "np.array([0.62, 0.1, 0.331, -0.681, 0.0392, 0.3156])"
CONTRACT = "np.array([1000.0, 1000.0, 1.0, 0.0, -6.817])"
NUMERICS = "np.array([1.0 / 32, 2e-3, 16, 0.43, 6.0, 4.0, 4.0, 2.0])"


def test_cases():
    return [
        {
            "setup": (
                "m = " + MODEL + "\n"
                "c = " + CONTRACT + "\n"
                "q = " + NUMERICS
            ),
            "call": "round(run_multilevel_pricer(m, c, q), 9)",
            "gold_call": "94.73779563",
            "tol": 1e-7,
        },
        {
            "setup": (
                "m = " + MODEL + "\n"
                "c = " + CONTRACT + "\n"
                "q = np.array([1.0 / 32, 2e-3, 16, 0.43, 6.0, 4.0, 4.0, 2.0])\n"
                "q[1] = 8e-3"
            ),
            "call": "round(run_multilevel_pricer(m, c, q), 9)",
            "gold_call": "94.888553819",
            "tol": 1e-7,
        },
        {
            "setup": (
                "m = " + MODEL + "\n"
                "c = np.array([1000.0, 1200.0, 1.0, 0.0, -6.817])\n"
                "q = " + NUMERICS + "\n"
                "atm = run_multilevel_pricer(m, " + CONTRACT + ", q)\n"
                "otm = run_multilevel_pricer(m, c, q)"
            ),
            "call": "bool(0.0 < otm < atm)",
            "gold_call": "True",
        },
        {
            "setup": (
                "m = " + MODEL + "\n"
                "c = " + CONTRACT + "\n"
                "q = " + NUMERICS + "\n"
                "v = run_multilevel_pricer(m, c, q)\n"
                "lo = max(0.0, 1000.0 - 1000.0)"
            ),
            "call": "bool(np.isfinite(v) and lo <= v <= 1000.0)",
            "gold_call": "True",
        },
        {
            "setup": (
                "m = " + MODEL + "\n"
                "a = run_multilevel_pricer(m, " + CONTRACT + ", " + NUMERICS + ")\n"
                "c2 = np.array([2000.0, 2000.0, 1.0, 0.0, -6.817])\n"
                "q2 = np.array([1.0 / 32, 4e-3, 16, 0.86, 6.0, 8.0, 4.0, 2.0])\n"
                "b = run_multilevel_pricer(m, c2, q2)"
            ),
            "call": "round(float(0.5 * b - a), 9)",
            "gold_call": "0.0",
            "tol": 1e-8,
        },
        {
            "setup": (
                "m = " + MODEL + "\n"
                "c = " + CONTRACT + "\n"
                "def run_model():\n"
                "    try:\n"
                "        run_multilevel_pricer(m, c, np.zeros(3))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_run_multilevel_pricer(m, c, np.zeros(3))\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": (
                "m = " + MODEL + "\n"
                "q = " + NUMERICS + "\n"
                "c = np.array([-1000.0, 1000.0, 1.0, 0.0, -6.817])\n"
                "def run_model():\n"
                "    try:\n"
                "        run_multilevel_pricer(m, c, q)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_run_multilevel_pricer(m, c, q)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
