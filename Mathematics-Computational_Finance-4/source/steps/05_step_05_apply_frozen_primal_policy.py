"""
Apply the frozen training-sample stopping rule to an independent pricing sample and return each path's stopping index, its discounted cashflow and their mean, the primal lower-bound estimate. Given the pricing-sample payoffs and basis, the coefficients fitted on the training sample, the maturity and the risk-free rate, stop every path at its first eligible exercise date.

A stopping rule estimated on one sample and evaluated on independent paths is a feasible, generally suboptimal exercise policy, so the average of its discounted payoffs estimates a lower bound on the option value. The coefficients are used exactly as supplied; nothing is refitted on the pricing sample.

The rule stops a path at the first eligible date before maturity where its payoff is strictly positive and not below the frozen continuation estimate (that date's features combined with that date's coefficients); otherwise the path stops at maturity whatever its payoff there. Time zero is not an exercise date.

Returns
-------
tuple (stopping_indices (n_paths,) int64 ndarray, discounted_cashflows (n_paths,) float ndarray, primal_lower_bound float): stopping date index per path, time-zero discounted cashflow per path, and their mean
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_frozen_primal_policy(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    coefficients: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray, float]":
    """Evaluate a frozen regression stopping rule on an independent sample.

    Parameters
    ----------
    payoffs : np.ndarray
        Pricing-sample payoffs of shape (n_paths, n_times), finite and
        non-negative, n_times >= 2.
    basis : np.ndarray
        Pricing-sample features of shape (n_paths, n_times, p), finite.
    coefficients : np.ndarray
        Frozen training coefficients of shape (n_times, p), finite.
    maturity : float
        Maturity T, strictly positive; dates are t_j = j T / n_steps.
    rate : float
        Continuously compounded risk-free rate.

    Returns
    -------
    stopping_indices : np.ndarray
        Int64 array of shape (n_paths,): the stopping date index of each
        path, in 1, ..., n_steps.
    discounted_cashflows : np.ndarray
        Shape (n_paths,): each path's payoff at its stopping date,
        discounted to time zero.
    primal_lower_bound : float
        Mean of discounted_cashflows, as a Python float.

    Raises
    ------
    ValueError
        If payoffs is not a finite, non-negative two-dimensional array with at
        least two dates, basis is not a finite array of shape (n_paths,
        n_times, p) with p >= 1, coefficients is not a finite array of shape
        (n_times, p), maturity is not finite and positive, or rate is not
        finite.
    """
    return stopping_indices, discounted_cashflows, primal_lower_bound

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_apply_frozen_primal_policy(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    coefficients: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray, float]":
    h = np.asarray(payoffs, dtype=float)
    b = np.asarray(basis, dtype=float)
    c = np.asarray(coefficients, dtype=float)
    if h.ndim != 2 or h.shape[0] < 1 or h.shape[1] < 2:
        raise ValueError("payoffs must have shape (n_paths, n_times) with n_times >= 2.")
    if not np.all(np.isfinite(h)) or np.any(h < 0.0):
        raise ValueError("payoffs must be finite and non-negative.")
    if b.ndim != 3 or b.shape[:2] != h.shape or b.shape[2] < 1 or not np.all(np.isfinite(b)):
        raise ValueError("basis must be a finite array of shape (n_paths, n_times, p) with p >= 1.")
    if c.shape != (h.shape[1], b.shape[2]) or not np.all(np.isfinite(c)):
        raise ValueError("coefficients must be a finite array of shape (n_times, p).")
    maturity, rate = float(maturity), float(rate)
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if not np.isfinite(rate):
        raise ValueError("rate must be finite.")

    n_paths, n_times = h.shape
    n_steps = n_times - 1
    dt = maturity / n_steps
    stops = np.full(n_paths, n_steps, dtype=np.int64)
    alive = np.ones(n_paths, dtype=bool)
    for j in range(1, n_steps):
        continuation = b[:, j, :] @ c[j]
        exercise = alive & (h[:, j] > 0.0) & (h[:, j] >= continuation)
        stops[exercise] = j
        alive[exercise] = False
    cashflows = h[np.arange(n_paths), stops] * np.exp(-rate * stops * dt)
    return stops, cashflows, float(np.mean(cashflows))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pack = (
        "import numpy as np\n"
        "def pack(out):\n"
        "    s, c, p = out\n"
        "    return np.concatenate((np.asarray(s, dtype=np.float64), np.asarray(c, dtype=np.float64), [float(p)]))\n"
    )
    scenario = (
        "payoffs = np.array([[0.0, 8.0, 12.0, 20.0],\n"
        "                    [0.0, 5.0, 9.0, 1.0],\n"
        "                    [0.0, 0.0, 7.0, 0.0],\n"
        "                    [999.0, 0.0, 0.0, 4.0],\n"
        "                    [0.0, 2.0, 3.0, 11.0]])\n"
        "basis = np.ones((5, 4, 1))\n"
        "coefficients = np.array([[0.0], [5.0], [7.0], [0.0]])\n"
    )
    codes = (
        "import numpy as np\n"
        "h = np.zeros((4, 3))\n"
        "b = np.ones((4, 3, 2))\n"
        "c = np.zeros((3, 2))\n"
        "def code(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    invalid_args = [
        "h, b, np.zeros((2, 2)), 1.0, 0.05",
        "h, b, np.zeros((3, 3)), 1.0, 0.05",
        "-np.ones((4, 3)), b, c, 1.0, 0.05",
        "h, np.ones((4, 2, 2)), c, 1.0, 0.05",
        "h, b, c, -1.0, 0.05",
    ]

    def _maxcall_sample(name, n_paths, seed):
        # Three independent GBM assets (S0 = K = 100, r = 0.05, q = 0.10, sigma = 0.20, T = 3, nine steps),
        # their max-call payoffs and a polynomial basis in the two largest prices divided by the strike.
        return (
            "g = np.random.default_rng(%d)\n" % seed
            + "z = g.standard_normal((%d, 9, 3))\n" % n_paths
            + "dt = 3.0 / 9\n"
            + "s = 100.0 * np.exp(np.concatenate((np.zeros((%d, 1, 3)), np.cumsum((0.05 - 0.10 - 0.02) * dt + 0.2 * np.sqrt(dt) * z, axis=1)), axis=1))\n" % n_paths
            + "x = np.sort(s, axis=2)[:, :, ::-1] / 100.0\n"
            + "%s_h = np.maximum(s.max(axis=2) - 100.0, 0.0)\n" % name
            + "%s_b = np.stack([np.ones_like(x[..., 0]), x[..., 0], x[..., 0] ** 2, x[..., 1], x[..., 1] ** 2, x[..., 0] * x[..., 1]], axis=2)\n" % name
        )

    # Frozen coefficients: at each exercise date, regress the discounted maturity payoff of the training sample on its basis.
    frozen = (
        "coefficients = np.zeros((10, 6))\n"
        "for j in range(1, 9):\n"
        "    coefficients[j] = np.linalg.lstsq(tr_b[:, j, :], np.exp(-0.05 * (9 - j) * dt) * tr_h[:, 9], rcond=None)[0]\n"
    )
    return [
        # Hand scenario: first eligible date, equality exercises, zero payoff never exercises,
        # time-zero payoff ignored, maturity fallback; zero rate.
        {
            "setup": pack + scenario,
            "call": "pack(apply_frozen_primal_policy(payoffs, basis, coefficients, 3.0, 0.0))",
            "gold_call": "pack(_oracle_apply_frozen_primal_policy(payoffs.copy(), basis.copy(), coefficients.copy(), 3.0, 0.0))",
            "tol": 1e-9,
        },
        # Same scenario with discounting to the stopping date.
        {
            "setup": pack + scenario,
            "call": "pack(apply_frozen_primal_policy(payoffs, basis, coefficients, 1.5, 0.10))",
            "gold_call": "pack(_oracle_apply_frozen_primal_policy(payoffs.copy(), basis.copy(), coefficients.copy(), 1.5, 0.10))",
            "tol": 1e-9,
        },
        # Max-call: regression coefficients fitted on one simulated sample, applied to an independent one.
        {
            "setup": pack + _maxcall_sample("tr", 800, 21) + _maxcall_sample("pr", 700, 22) + frozen + "payoffs, basis = pr_h, pr_b\n",
            "call": "pack(apply_frozen_primal_policy(payoffs, basis, coefficients, 3.0, 0.05))",
            "gold_call": "pack(_oracle_apply_frozen_primal_policy(payoffs.copy(), basis.copy(), coefficients.copy(), 3.0, 0.05))",
            "tol": 1e-9,
        },
        # Two dates only: every path stops at maturity.
        {
            "setup": pack + "payoffs = np.array([[5.0, 2.0], [0.0, 0.0], [1.0, 7.5]])\n"
                            "basis = np.ones((3, 2, 2))\ncoefficients = np.zeros((2, 2))\n",
            "call": "pack(apply_frozen_primal_policy(payoffs, basis, coefficients, 2.0, 0.03))",
            "gold_call": "pack(_oracle_apply_frozen_primal_policy(payoffs.copy(), basis.copy(), coefficients.copy(), 2.0, 0.03))",
            "tol": 1e-9,
        },
    ] + [
        # Invalid inputs raise ValueError, one invalid condition per case.
        {
            "setup": codes,
            "call": "code(lambda: apply_frozen_primal_policy(%s))" % args,
            "gold_call": "code(lambda: _oracle_apply_frozen_primal_policy(%s))" % args,
        }
        for args in invalid_args
    ]
