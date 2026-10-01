"""
Chain the eight earlier steps over the whole premium schedule and return the par spread the semi-analytical method produces. For every premium date the grid over the average point is rebuilt, the variational frequency and width are solved at each node, the reduced integrals, displacements, correction, normalisation and endpoint coefficients are assembled, and the two expectations are summed; the survival values build the premium annuity and the intensity-weighted values build the protection leg. This is the orchestrator step: it chains the eight earlier reference implementations rather than reproducing any of them inline.

What the method returns is the output of an approximation at a stated configuration, not a converged price, and that is the point. Every convention in the construction leaves a fingerprint on the number: whether the fluctuation width carries the constraint correction, whether the level of the trial potential is smeared, whether the action correction is present, whether the drift-removal factor is folded into the endpoint coefficients, and whether the payoff is smeared before summing. A converged price would erase all of these, because every reasonable variant converges to the same limit; the procedure's own output at a fixed grid keeps them apart.

Nothing here adapts. The schedule, the node count, the window width, the tolerance of the scalar fixed point and the two leg conventions are all prescribed, so no convergence test can terminate differently on a different machine. The only quantities the procedure decides for itself are the per-node frequencies, and those come from a contraction with a unique limit that does not depend on the starting value.

Both legs are discounted on the same deterministic curve and share the same grid of dates, so the spread is a ratio of two sums over the same schedule and inherits the exact proportionality to one minus the recovery rate.

Returns
-------
float, the par spread in basis points. For the task configuration the value is 212.0732334903. The spread is exactly proportional to one minus the recovery rate, is positive, and falls as the mean-reversion level falls.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def par_cds_spread(model, contract, numerics):
    """Chain the eight earlier steps and return the par credit spread.

    For each premium date T_i = i*dtau, i = 1..nper, builds the average-point
    grid, solves the variational frequency at every node and recovers the
    fluctuation width that belongs to it, assembles
    the reduced integrals, the displacements, the action correction, the
    normalisation and the endpoint coefficients, and sums the two smeared
    expectations.  The survival values Q(T_i) form the premium annuity

        annuity = sum_i dtau * D(T_i) * Q(T_i),

    and the intensity-weighted values G(T_i) form the protection leg by the
    composite trapezoidal rule on the same dates, with G(0) = exp(x0) and
    Q(0) = 1,

        protection = (1 - rec) * trapezoid over [0, T_nper] of D(u) * G(u),

    where D(u) = exp(-rate*u).  The spread is 1e4 * protection / annuity.

    Args:
        model: (4,) array [k, sigma, theta, x0].
        contract: (5,) array [dtau, nper, rec, rate, lam].
        numerics: (3,) array [nq, span, reserved].

    Expected return:
        float: the par credit spread in basis points.

    Raises:
        ValueError: if model, contract and numerics do not have lengths
        4, 5 and 3, or if dtau <= 0, or if nper < 1.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_par_cds_spread(model, contract, numerics):
    def _node(xbar, x0, T, k, sigma, theta, lam):
        omega = float(_oracle_variational_frequency(xbar, T, k, sigma, lam))
        alpha = float(_oracle_harmonic_fluctuation_width(omega, T, sigma))
        red = _oracle_linear_coefficient_integrals(xbar, omega, T, k, sigma,
                                                   theta)
        dsp = _oracle_displacement_and_correction(red, omega, T, sigma)
        lgn = _oracle_log_trial_normalisation(xbar, omega, alpha, float(dsp[1]),
                                              T, k, sigma, lam, theta)
        return _oracle_endpoint_coefficients(xbar, x0, float(dsp[0]), red,
                                             alpha, omega, lgn, T, k, sigma,
                                             theta)

    m = np.asarray(model, dtype=float).reshape(-1)
    c = np.asarray(contract, dtype=float).reshape(-1)
    q = np.asarray(numerics, dtype=float).reshape(-1)
    if m.size != 4 or c.size != 5 or q.size != 3:
        raise ValueError("model, contract and numerics must have sizes 4, 5, 3")
    k, sigma, theta, x0 = float(m[0]), float(m[1]), float(m[2]), float(m[3])
    dtau, rec, rate, lam = float(c[0]), float(c[2]), float(c[3]), float(c[4])
    nper = int(round(float(c[1])))
    nq, span = int(round(float(q[0]))), float(q[1])
    if dtau <= 0.0:
        raise ValueError("dtau must be positive")
    if nper < 1:
        raise ValueError("nper must be at least 1")

    taus = dtau * np.arange(nper + 1)
    surv = np.empty(nper + 1)
    inten = np.empty(nper + 1)
    surv[0], inten[0] = 1.0, np.exp(x0)
    for i in range(1, nper + 1):
        T = float(taus[i])
        grid = _oracle_average_point_grid(T, x0, k, sigma, theta, nq, span)
        u, w = grid[:nq], grid[nq:]
        lw = np.empty(nq)
        ce = np.empty(nq)
        aa = np.empty(nq)
        for j in range(nq):
            e = _node(float(u[j]), x0, T, k, sigma, theta, lam)
            aa[j], ce[j], lw[j] = float(e[0]), float(e[3]), float(e[4])
        val = _oracle_assemble_expectation(lw, ce, aa, w)
        surv[i], inten[i] = float(val[0]), float(val[1])

    disc = np.exp(-rate * taus)
    annuity = float(np.sum(dtau * disc[1:] * surv[1:]))
    y = disc * inten
    prot = (1.0 - rec) * float(dtau * (0.5 * y[0] + y[1:-1].sum()
                                       + 0.5 * y[-1]))
    return float(1.0e4 * prot / annuity)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

MODEL = "np.array([0.35, 0.62, -3.688879454114, -3.352407217493])"
CONTRACT = "np.array([0.25, 20.0, 0.4, 0.025, 1.0])"
NUMERICS = "np.array([81.0, 9.0, 0.0])"


def test_cases():
    return [
        {
            "setup": ("m = " + MODEL + "\nc = " + CONTRACT + "\nq = " + NUMERICS),
            "call": "round(par_cds_spread(m, c, q), 8)",
            "gold_call": "212.07323349",
            "tol": 1e-7,
        },
        {
            "setup": (
                "m = " + MODEL + "\nq = " + NUMERICS + "\n"
                "base = par_cds_spread(m, " + CONTRACT + ", q)\n"
                "zero = par_cds_spread(m, np.array([0.25, 20.0, 0.0, 0.025, 1.0]), q)"
            ),
            "call": "round(float(zero * 0.6 - base), 8)",
            "gold_call": "0.0",
            "tol": 1e-8,
        },
        {
            "setup": (
                "m = " + MODEL + "\nq = " + NUMERICS + "\n"
                "short = par_cds_spread(m, np.array([0.25, 4.0, 0.4, 0.025, 1.0]), q)"
            ),
            "call": "round(float(short), 8)",
            "gold_call": "216.34070632",
            "tol": 1e-7,
        },
        {
            "setup": (
                "m = " + MODEL + "\nc = " + CONTRACT + "\nq = " + NUMERICS + "\n"
                "a = par_cds_spread(m, c, q)\n"
                "b = par_cds_spread(m, c, np.array([101.0, 10.0, 0.0]))"
            ),
            "call": "bool(abs(float(a - b)) < 1e-6 and 0.0 < float(a) < 1.0e4)",
            "gold_call": "True",
        },
        {
            "setup": (
                "c = " + CONTRACT + "\nq = " + NUMERICS + "\n"
                "lo = par_cds_spread(np.array([0.35, 0.62, -4.3, -3.352407217493]), c, q)\n"
                "hi = par_cds_spread(np.array([0.35, 0.62, -3.1, -3.352407217493]), c, q)"
            ),
            "call": "bool(0.0 < float(lo) < float(hi))",
            "gold_call": "True",
        },
        {
            "setup": (
                "m = " + MODEL + "\nq = " + NUMERICS + "\n"
                "def run_model():\n"
                "    try:\n"
                "        par_cds_spread(m, np.zeros(3), q)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_par_cds_spread(m, np.zeros(3), q)\n"
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
                "m = " + MODEL + "\nq = " + NUMERICS + "\n"
                "c = np.array([-0.25, 20.0, 0.4, 0.025, 1.0])\n"
                "def run_model():\n"
                "    try:\n"
                "        par_cds_spread(m, c, q)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_par_cds_spread(m, c, q)\n"
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
