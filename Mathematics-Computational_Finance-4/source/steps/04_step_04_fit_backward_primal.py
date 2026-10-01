"""
Fit the backward least-squares continuation regressions on the training sample and return the fitted coefficients, the training exercise decisions and the backward value process. Given the exercise payoffs, the regression basis at every path and date, the maturity and the risk-free rate, work backward from maturity through the eligible exercise dates.

The continuation value at an eligible date is the conditional expectation of the option's value one date later, discounted over one step. It is estimated by ordinary least squares over all training paths, with the realised value of the stopping rule at the next date as the regressand and the current date's basis as the regressors. A path exercises when its payoff is strictly positive and not below this estimate. The value carried backward is the realised one, the payoff on exercise and otherwise the discounted next-date value, so regression errors do not accumulate through the recursion.

Time zero is not an exercise date and maturity needs no regression.

Returns
-------
tuple (coefficients (n_times, p) float ndarray, exercise_policy (n_paths, n_times) bool ndarray, backward_values (n_paths, n_times) float ndarray): fitted continuation coefficients, training exercise decisions and the backward value process
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_backward_primal(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Backward least-squares continuation regressions on a training sample.

    Parameters
    ----------
    payoffs : np.ndarray
        Exercise payoffs of shape (n_paths, n_times), finite and
        non-negative, with n_times = n_steps + 1 >= 2.
    basis : np.ndarray
        Regression features of shape (n_paths, n_times, p), finite, p >= 1.
    maturity : float
        Maturity T, strictly positive; dates are t_j = j T / n_steps.
    rate : float
        Continuously compounded risk-free rate.

    Returns
    -------
    coefficients : np.ndarray
        Shape (n_times, p): the least-squares coefficients of the
        continuation regression at each eligible date j = 1, ...,
        n_steps - 1; rows 0 and n_steps are zero.
    exercise_policy : np.ndarray
        Boolean shape (n_paths, n_times): the rule's exercise decisions at
        the eligible dates, exercise at maturity exactly when the payoff is
        positive, and no exercise at time zero.
    backward_values : np.ndarray
        Shape (n_paths, n_times): the realised value of the rule at every
        date, undiscounted; the time-zero column is the date-one value
        discounted over one step.

    Raises
    ------
    ValueError
        If payoffs is not a finite, non-negative two-dimensional array with at
        least two dates, basis is not a finite array of shape (n_paths,
        n_times, p) with p >= 1, maturity is not finite and positive, or rate
        is not finite.
    """
    return coefficients, exercise_policy, backward_values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_fit_backward_primal(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    h = np.asarray(payoffs, dtype=float)
    b = np.asarray(basis, dtype=float)
    if h.ndim != 2 or h.shape[0] < 1 or h.shape[1] < 2:
        raise ValueError("payoffs must have shape (n_paths, n_times) with n_times >= 2.")
    if not np.all(np.isfinite(h)) or np.any(h < 0.0):
        raise ValueError("payoffs must be finite and non-negative.")
    if b.ndim != 3 or b.shape[:2] != h.shape or b.shape[2] < 1 or not np.all(np.isfinite(b)):
        raise ValueError("basis must be a finite array of shape (n_paths, n_times, p) with p >= 1.")
    maturity, rate = float(maturity), float(rate)
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive.")
    if not np.isfinite(rate):
        raise ValueError("rate must be finite.")

    n_paths, n_times, p = b.shape
    n_steps = n_times - 1
    one_step = np.exp(-rate * maturity / n_steps)
    coefficients = np.zeros((n_times, p))
    policy = np.zeros((n_paths, n_times), dtype=bool)
    values = np.zeros((n_paths, n_times))
    values[:, n_steps] = h[:, n_steps]
    policy[:, n_steps] = h[:, n_steps] > 0.0
    tail = h[:, n_steps].copy()
    for j in range(n_steps - 1, 0, -1):
        target = one_step * tail
        beta = np.linalg.lstsq(b[:, j, :], target, rcond=None)[0]
        coefficients[j] = beta
        continuation = b[:, j, :] @ beta
        exercise = (h[:, j] > 0.0) & (h[:, j] >= continuation)
        tail = np.where(exercise, h[:, j], target)
        values[:, j] = tail
        policy[:, j] = exercise
    values[:, 0] = one_step * values[:, 1]
    return coefficients, policy, values

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pack = (
        "import numpy as np\n"
        "def pack(out):\n"
        "    c, e, v = out\n"
        "    b = basis\n"
        "    continuation = np.einsum('ntk,tk->nt', np.asarray(b, dtype=np.float64), np.asarray(c, dtype=np.float64))\n"
        "    return np.concatenate((continuation.ravel(), np.asarray(e, dtype=np.float64).ravel(), np.asarray(v, dtype=np.float64).ravel()))\n"
    )
    synthetic = (
        "g = np.random.default_rng(11)\n"
        "n_paths, n_times = 400, 6\n"
        "s = 100.0 * np.exp(np.cumsum(np.concatenate((np.zeros((n_paths, 1)), 0.1 * g.standard_normal((n_paths, n_times - 1))), axis=1), axis=1))\n"
        "payoffs = np.maximum(s - 100.0, 0.0)\n"
        "x = s / 100.0\n"
        "basis = np.stack((np.ones_like(x), x, x ** 2, np.maximum(x - 1.0, 0.0)), axis=2)\n"
    )
    codes = (
        "import numpy as np\n"
        "good_h = np.zeros((5, 3))\n"
        "good_b = np.ones((5, 3, 2))\n"
        "def code(thunk):\n"
        "    try:\n"
        "        thunk()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    invalid_args = [
        "np.zeros((5, 1)), np.ones((5, 1, 2)), 1.0, 0.05",
        "-np.ones((5, 3)), good_b, 1.0, 0.05",
        "good_h, np.ones((5, 3)), 1.0, 0.05",
        "good_h, np.ones((4, 3, 2)), 1.0, 0.05",
        "good_h, good_b, 0.0, 0.05",
        "good_h, good_b, 1.0, np.nan",
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

    return [
        # Synthetic call-like problem with a four-function basis; fitted continuation values, decisions and values.
        {
            "setup": pack + synthetic,
            "call": "pack(fit_backward_primal(payoffs, basis, 1.5, 0.04))",
            "gold_call": "pack(_oracle_fit_backward_primal(payoffs.copy(), basis.copy(), 1.5, 0.04))",
            "tol": 1e-9,
        },
        # Max-call payoffs on 600 simulated three-asset paths with a sorted-price polynomial basis.
        {
            "setup": pack + _maxcall_sample("tr", 600, 3) + "payoffs, basis = tr_h, tr_b\n",
            "call": "pack(fit_backward_primal(payoffs, basis, 3.0, 0.05))",
            "gold_call": "pack(_oracle_fit_backward_primal(payoffs.copy(), basis.copy(), 3.0, 0.05))",
            "tol": 1e-8,
        },
        # Only one step: no regression date, maturity decision and time-zero discounting only.
        {
            "setup": pack + "payoffs = np.array([[0.0, 3.0], [0.0, 0.0], [0.0, 1.5]])\nbasis = np.ones((3, 2, 1))\n",
            "call": "pack(fit_backward_primal(payoffs, basis, 0.5, 0.08))",
            "gold_call": "pack(_oracle_fit_backward_primal(payoffs.copy(), basis.copy(), 0.5, 0.08))",
            "tol": 1e-9,
        },
        # All payoffs zero before maturity: no early exercise anywhere.
        {
            "setup": pack + synthetic + "payoffs[:, 1:-1] = 0.0\n",
            "call": "pack(fit_backward_primal(payoffs, basis, 1.5, 0.04))",
            "gold_call": "pack(_oracle_fit_backward_primal(payoffs.copy(), basis.copy(), 1.5, 0.04))",
            "tol": 1e-9,
        },
    ] + [
        # Invalid inputs raise ValueError, one invalid condition per case.
        {
            "setup": codes,
            "call": "code(lambda: fit_backward_primal(%s))" % args,
            "gold_call": "code(lambda: _oracle_fit_backward_primal(%s))" % args,
        }
        for args in invalid_args
    ]
