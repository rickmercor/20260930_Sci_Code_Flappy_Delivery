"""
Evaluate the signed likelihood-ratio root of the background-only test for arrays of on/off counts.

For the on/off likelihood $L(s, b)$ of a signal-region count $n$ with mean $s + b$ and a control count $m$ with mean $\tau b$, the profile likelihood ratio of the hypothesis $s = 0$ is $\lambda(0) = L(0, \hat{\hat{b}}_0) / L(\hat{s}, \hat{b})$ with the unconstrained estimators $\hat{s} = n - m/\tau$, $\hat{b} = m/\tau$ and the profiled background $\hat{\hat{b}}_0 = (n + m)/(1 + \tau)$. The signed likelihood-ratio root is the signed square root of the likelihood-ratio statistic,

$$r(0) = \operatorname{sign}(\hat{s}) \sqrt{-2 \ln \lambda(0)} = \operatorname{sign}\!\left(n - \frac{m}{\tau}\right) \left\{ -2 \left[ n \ln\!\left(\frac{n + m}{(1 + \tau)\, n}\right) + m \ln\!\left(\frac{\tau (n + m)}{(1 + \tau)\, m}\right) \right] \right\}^{1/2} .$$

It is positive when the signal region shows an excess over the background inferred from the control region and negative when it shows a deficit. In the large-sample limit $r(0)$ follows a standard Gaussian, and the discovery statistic is $q_0 = [\max\{0, r(0)\}]^2$, so that the first-order significance is $Z = \max\{0, r(0)\}$.

At the boundaries of the sample space the terms are interpreted through their continuous limits, $x \ln x \to 0$ as $x \to 0$, so an empty region contributes nothing to the sum, and $r(0) = 0$ when both regions are empty. The counts may be real-valued, because the same statistic is evaluated on the Asimov data set. The function is vectorised so that a whole grid of hypothetical outcomes can be evaluated at once when a tail probability is summed. At large counts the two logarithmic terms are individually large and nearly cancel, while the statistic itself stays of order one, so each term has to be evaluated in a form that keeps the root accurate at counts of order $10^8$ and above.

Returns
-------
numpy.ndarray of float, the signed background-only likelihood-ratio root with the broadcast shape of n and m, including the specified zero-count limits.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def signed_likelihood_root(n: "np.ndarray", m: "np.ndarray", tau: float) -> "np.ndarray":
    r"""Return the signed root r(0) of the background-only profile likelihood ratio.

    Parameters
    ----------
    n : np.ndarray
        Signal-region counts, an array of finite values of at least zero.
        Real values are allowed for Asimov data.
    m : np.ndarray
        Control-region counts, an array of finite values of at least zero,
        broadcastable against n.
    tau : float
        Scale factor between the control and signal regions, finite and
        above zero.

    Returns
    -------
    r : np.ndarray
        Float array with the broadcast shape of n and m. Entry-wise it is the
        signed likelihood-ratio root of the hypothesis s = 0, positive for an
        excess n > m / tau, negative for a deficit and zero when n equals
        m / tau or both counts are zero. Zero counts contribute nothing to
        the log-likelihood difference, and a round-off negative value of the
        likelihood-ratio statistic is treated as zero. Every entry is accurate
        to a relative error of 1e-9 for counts up to 1e9.

    Raises
    ------
    ValueError
        If tau is not finite or not above zero, or if any count is negative
        or not finite.
    """
    return r

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_signed_likelihood_root(n: "np.ndarray", m: "np.ndarray", tau: float) -> "np.ndarray":
    n = np.asarray(n, dtype=float)
    m = np.asarray(m, dtype=float)
    tau = float(tau)
    if not (math.isfinite(tau) and tau > 0.0):
        raise ValueError("tau must be a finite scale factor above zero")
    for name, counts in (("n", n), ("m", m)):
        if not np.all(np.isfinite(counts)):
            raise ValueError(f"{name} must be finite")
        if np.any(counts < 0.0):
            raise ValueError(f"{name} must be at least zero")

    n, m = np.broadcast_arrays(n, m)
    # Each term is x ln(ratio) with the continuous limit 0 ln(0) = 0. The ratios
    # are written as 1 + delta and evaluated with log1p, because at large counts
    # the rounding of a ratio near one would be amplified by the count in front.
    term_n = np.zeros(n.shape, dtype=float)
    term_m = np.zeros(m.shape, dtype=float)
    on = n > 0.0
    off = m > 0.0
    term_n[on] = n[on] * np.log1p((m[on] - tau * n[on]) / ((1.0 + tau) * n[on]))
    term_m[off] = m[off] * np.log1p((tau * n[off] - m[off]) / ((1.0 + tau) * m[off]))
    statistic = np.maximum(-2.0 * (term_n + term_m), 0.0)
    return np.sign(n - m / tau) * np.sqrt(statistic)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    err = ""
    for name, function in (("run_model", "signed_likelihood_root"),
                           ("run_gold", "_oracle_signed_likelihood_root")):
        err += (f"def {name}():\n    try:\n        {function}(*args)\n"
                "        return 0\n    except ValueError:\n        return 1\n")
    cases = [
        # observed pairs with excesses and deficits, so both signs appear
        {
            "setup": """import numpy as np
n = np.array([8, 3, 12, 5, 1])
m = np.array([2, 6, 4, 5, 9])
tau = 1.25
""",
            "call": "signed_likelihood_root(n, m, tau)",
            "gold_call": "_oracle_signed_likelihood_root(n, m, tau)",
        },
        # a two-dimensional grid of hypothetical outcomes formed by broadcasting
        {
            "setup": """import numpy as np
n = np.arange(0, 7)[:, None]
m = np.arange(0, 5)[None, :]
tau = 2.0
""",
            "call": "signed_likelihood_root(n, m, tau)",
            "gold_call": "_oracle_signed_likelihood_root(n, m, tau)",
        },
        # the benchmark Asimov data set and its neighbours, real-valued counts
        {
            "setup": """import numpy as np
n = np.array([5.8, 5.3, 6.3])
m = np.array([1.0, 1.0, 1.0])
tau = 1.25
""",
            "call": "signed_likelihood_root(n, m, tau)",
            "gold_call": "_oracle_signed_likelihood_root(n, m, tau)",
        },
        # boundary: an empty control region, an empty signal region and both empty
        {
            "setup": """import numpy as np
n = np.array([4, 0, 0])
m = np.array([0, 3, 0])
tau = 1.5
""",
            "call": "signed_likelihood_root(n, m, tau)",
            "gold_call": "_oracle_signed_likelihood_root(n, m, tau)",
        },
        # boundary: counts exactly at the background expectation, where the root is zero
        {
            "setup": """import numpy as np
n = np.array([4, 6, 10])
m = np.array([8, 12, 20])
tau = 2.0
""",
            "call": "signed_likelihood_root(n, m, tau)",
            "gold_call": "_oracle_signed_likelihood_root(n, m, tau)",
        },
        # edge: a small scale factor, where the control region constrains the background weakly
        {
            "setup": """import numpy as np
n = np.array([30, 40, 50])
m = np.array([3, 3, 3])
tau = 0.1
""",
            "call": "signed_likelihood_root(n, m, tau)",
            "gold_call": "_oracle_signed_likelihood_root(n, m, tau)",
        },
        # edge: counts of order 1e8, where the two logarithmic terms nearly cancel
        {
            "setup": """import numpy as np
n = np.array([100010000, 99990000, 100000000])
m = np.array([100000000, 100000000, 100000000])
tau = 1.0
""",
            "call": "signed_likelihood_root(n, m, tau)",
            "gold_call": "_oracle_signed_likelihood_root(n, m, tau)",
        },
        # edge: real-valued counts of order 1e8 with a scale factor above one, mixed with small counts
        {
            "setup": """import numpy as np
n = np.array([2.5e8, 3.0e8, 12.0])
m = np.array([3.1253e8, 3.75e8, 4.0])
tau = 1.25
""",
            "call": "signed_likelihood_root(n, m, tau)",
            "gold_call": "_oracle_signed_likelihood_root(n, m, tau)",
        },
        # edge: real-valued counts of order 1e8 with a control region half the signal region
        {
            "setup": """import numpy as np
n = np.array([4.0e8, 4.0004e8, 3.9996e8])
m = np.array([2.0e8, 2.0e8, 2.0e8])
tau = 0.5
""",
            "call": "signed_likelihood_root(n, m, tau)",
            "gold_call": "_oracle_signed_likelihood_root(n, m, tau)",
        },
        # edge: counts of order 1e9 with a control region three times the signal region
        {
            "setup": """import numpy as np
n = np.array([1.0e9, 1.00003e9, 9.9997e8])
m = np.array([3.0e9, 3.0e9, 3.0e9])
tau = 3.0
""",
            "call": "signed_likelihood_root(n, m, tau)",
            "gold_call": "_oracle_signed_likelihood_root(n, m, tau)",
        },
        # edge: a two-dimensional grid of real-valued counts of order 1e8 with a scale factor of three quarters
        {
            "setup": """import numpy as np
n = np.array([8.0e8, 8.0002e8, 7.9998e8])[:, None]
m = np.array([6.0e8, 6.0003e8])[None, :]
tau = 0.75
""",
            "call": "signed_likelihood_root(n, m, tau)",
            "gold_call": "_oracle_signed_likelihood_root(n, m, tau)",
        },
        {
            "setup": """import numpy as np
# invalid: a scale factor of zero
args = (np.array([3, 4]), np.array([1, 2]), 0.0)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a negative count in the control region
args = (np.array([3, 4]), np.array([1, -2]), 1.0)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a negative scale factor
args = (np.array([3, 4]), np.array([1, 2]), -1.0)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
# invalid: a signal-region count that is not finite
args = (np.array([3.0, np.inf]), np.array([1, 2]), 1.0)
""" + err,
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
    # Both sides receive their own deep copy of every argument, so an in-place
    # implementation on either side cannot contaminate the other's inputs.
    isolate = (
        "\nimport copy as _test_copy\n\n"
        "def _independent_inputs(_function):\n"
        "    def _invoke(*_args, **_kwargs):\n"
        "        _args, _kwargs = _test_copy.deepcopy((_args, _kwargs))\n"
        "        return _function(*_args, **_kwargs)\n"
        "    return _invoke\n\n"
        "signed_likelihood_root = _independent_inputs(signed_likelihood_root)\n"
        "_oracle_signed_likelihood_root = _independent_inputs(_oracle_signed_likelihood_root)\n"
    )
    for case in cases:
        case["setup"] += isolate
    return cases
