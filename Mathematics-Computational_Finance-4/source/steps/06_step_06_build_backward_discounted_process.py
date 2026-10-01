"""
Construct the backward value process of the frozen stopping rule on the pricing sample, in undiscounted and in time-zero-discounted units. Given the pricing-sample payoffs and basis, the frozen training coefficients, the maturity and the risk-free rate, roll the realized value back from maturity to time zero.

The regression-based dual construction needs the realised value of the frozen policy at every date of every pricing path, not only the cashflow at the first stopping time: the payoff where the rule would exercise at that date, and otherwise the value one date later discounted over one step. The rule is the one used for the primal bound, and time zero is not an exercise date.

The discounted process expresses every date's value in time-zero money; the dual projections are taken in these units.

Returns
-------
tuple (backward_values (n_paths, n_times) float ndarray, discounted_backward (n_paths, n_times) float ndarray): backward value process of the frozen policy and the same process multiplied by exp(-rate t_j)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_backward_discounted_process(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    coefficients: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray]":
    """Backward value process of the frozen policy on the pricing sample.

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
    backward_values : np.ndarray
        Shape (n_paths, n_times): the realised value of the frozen rule at
        every date, undiscounted, with the time-zero column equal to the
        date-one value discounted over one step.
    discounted_backward : np.ndarray
        Shape (n_paths, n_times): backward_values discounted from each date
        t_j to time zero.

    Raises
    ------
    ValueError
        If payoffs is not a finite, non-negative two-dimensional array with at
        least two dates, basis is not a finite array of shape (n_paths,
        n_times, p) with p >= 1, coefficients is not a finite array of shape
        (n_times, p), maturity is not finite and positive, or rate is not
        finite.
    """
    return backward_values, discounted_backward

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_backward_discounted_process(
    payoffs: "np.ndarray",
    basis: "np.ndarray",
    coefficients: "np.ndarray",
    maturity: float,
    rate: float,
) -> "tuple[np.ndarray, np.ndarray]":
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
    one_step = np.exp(-rate * dt)
    values = np.empty((n_paths, n_times))
    values[:, n_steps] = h[:, n_steps]
    for j in range(n_steps - 1, 0, -1):
        continuation = b[:, j, :] @ c[j]
        exercise = (h[:, j] > 0.0) & (h[:, j] >= continuation)
        values[:, j] = np.where(exercise, h[:, j], one_step * values[:, j + 1])
    values[:, 0] = one_step * values[:, 1]
    discounted = values * np.exp(-rate * dt * np.arange(n_times))[None, :]
    return values, discounted

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    pack = (
        "import numpy as np\n"
        "def pack(out):\n"
        "    v, d = out\n"
        "    return np.concatenate((np.asarray(v, dtype=np.float64).ravel(), np.asarray(d, dtype=np.float64).ravel()))\n"
    )
    scenario = (
        "payoffs = np.array([[0.0, 8.0, 12.0, 20.0],\n"
        "                    [0.0, 5.0, 9.0, 1.0],\n"
        "                    [0.0, 0.0, 7.0, 0.0],\n"
        "                    [0.0, 2.0, 3.0, 11.0]])\n"
        "basis = np.ones((4, 4, 1))\n"
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
        "h, b, np.zeros((3, 1)), 1.0, 0.05",
        "np.full((4, 3), np.inf), b, c, 1.0, 0.05",
        "h, b[:, :, :0], c[:, :0], 1.0, 0.05",
        "h, b, c, 0.0, 0.05",
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
        # Hand scenario with discounting: exercise, carry-back and time-zero values.
        {
            "setup": pack + scenario,
            "call": "pack(build_backward_discounted_process(payoffs, basis, coefficients, 1.5, 0.10))",
            "gold_call": "pack(_oracle_build_backward_discounted_process(payoffs.copy(), basis.copy(), coefficients.copy(), 1.5, 0.10))",
            "tol": 1e-9,
        },
        # Zero rate: the discounted process equals the undiscounted one.
        {
            "setup": pack + scenario,
            "call": "pack(build_backward_discounted_process(payoffs, basis, coefficients, 3.0, 0.0))",
            "gold_call": "pack(_oracle_build_backward_discounted_process(payoffs.copy(), basis.copy(), coefficients.copy(), 3.0, 0.0))",
            "tol": 1e-9,
        },
        # Max-call pricing sample with coefficients frozen from an independent training sample.
        {
            "setup": pack + _maxcall_sample("tr", 800, 31) + _maxcall_sample("pr", 600, 32) + frozen + "payoffs, basis = pr_h, pr_b\n",
            "call": "pack(build_backward_discounted_process(payoffs, basis, coefficients, 3.0, 0.05))",
            "gold_call": "pack(_oracle_build_backward_discounted_process(payoffs.copy(), basis.copy(), coefficients.copy(), 3.0, 0.05))",
            "tol": 1e-9,
        },
        # Two dates only: maturity payoff and one discounting step back to time zero.
        {
            "setup": pack + "payoffs = np.array([[4.0, 2.0], [0.0, 6.0]])\nbasis = np.ones((2, 2, 1))\ncoefficients = np.zeros((2, 1))\n",
            "call": "pack(build_backward_discounted_process(payoffs, basis, coefficients, 0.75, 0.04))",
            "gold_call": "pack(_oracle_build_backward_discounted_process(payoffs.copy(), basis.copy(), coefficients.copy(), 0.75, 0.04))",
            "tol": 1e-9,
        },
    ] + [
        # Invalid inputs raise ValueError, one invalid condition per case.
        {
            "setup": codes,
            "call": "code(lambda: build_backward_discounted_process(%s))" % args,
            "gold_call": "code(lambda: _oracle_build_backward_discounted_process(%s))" % args,
        }
        for args in invalid_args
    ]
