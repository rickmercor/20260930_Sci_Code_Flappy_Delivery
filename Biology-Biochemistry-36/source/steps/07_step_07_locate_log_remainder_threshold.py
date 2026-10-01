"""
Locate a mutation rate at which omitted log-partition substitution orders reach a prescribed share.

A bracketed solve makes the reported design threshold deterministic and avoids extrapolating a locally truncated nonlinear observable beyond a controlled interval.

Returns
-------
float: mutation rate at the unique bracketed remainder-share crossing.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_log_remainder_threshold(
    normalized_coefficients: "np.ndarray",
    log_coefficients: "np.ndarray",
    target_share: float,
    lower_rate: float,
    upper_rate: float,
    rate_tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    """Return the bracketed crossing of a relative log-remainder target.

    Use ``evaluate_log_remainder_share`` to define
    ``f(mu) = remainder_share(mu) - target_share``.  The caller guarantees
    that the closed interval contains exactly one crossing; its endpoint
    values must have opposite signs or one must be exactly zero.  Use a
    bracketing method and stop once the bracket width is no larger than
    ``rate_tolerance``.  Return the midpoint of the final bracket, except
    that an exact endpoint crossing is returned directly.

    Parameters
    ----------
    normalized_coefficients, log_coefficients : np.ndarray
        Inputs accepted by ``evaluate_log_remainder_share``.
    target_share : float
        Finite nonnegative target.
    lower_rate, upper_rate : float
        Finite rates satisfying ``0 < lower_rate < upper_rate <= 1``.
    rate_tolerance : float
        Finite positive absolute bracket-width tolerance.
    max_iterations : int
        Positive integer iteration cap; booleans are rejected.

    Returns
    -------
    float
        Mutation rate of the unique bracketed crossing.

    Raises
    ------
    ValueError
        If scalar controls are invalid, endpoint evaluation fails, the target
        is not bracketed, or the iteration cap is reached first.
    """
    return mutation_rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _root_scalar(value, name, *, positive=False, nonnegative=False):
    """Validate and return a finite non-boolean scalar."""
    import numpy as np

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite scalar") from None
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _oracle_locate_log_remainder_threshold(
    normalized_coefficients: "np.ndarray",
    log_coefficients: "np.ndarray",
    target_share: float,
    lower_rate: float,
    upper_rate: float,
    rate_tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    """Reference bisection of the relative log-remainder crossing."""
    target = _root_scalar(target_share, "target_share", nonnegative=True)
    lower = _root_scalar(lower_rate, "lower_rate")
    upper = _root_scalar(upper_rate, "upper_rate")
    tolerance = _root_scalar(rate_tolerance, "rate_tolerance", positive=True)
    if not (0.0 < lower < upper <= 1.0):
        raise ValueError("rates must satisfy 0 < lower_rate < upper_rate <= 1")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)):
        raise ValueError("max_iterations must be a positive integer")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")

    f_lower = _oracle_evaluate_log_remainder_share(
        normalized_coefficients, log_coefficients, lower
    ) - target
    f_upper = _oracle_evaluate_log_remainder_share(
        normalized_coefficients, log_coefficients, upper
    ) - target
    if f_lower == 0.0:
        return float(lower)
    if f_upper == 0.0:
        return float(upper)
    if f_lower * f_upper > 0.0:
        raise ValueError("the target is not bracketed")

    for _ in range(max_iterations):
        if upper - lower <= tolerance:
            return float(0.5 * (lower + upper))
        middle = 0.5 * (lower + upper)
        f_middle = _oracle_evaluate_log_remainder_share(
            normalized_coefficients, log_coefficients, middle
        ) - target
        if f_middle == 0.0:
            return float(middle)
        if f_lower * f_middle < 0.0:
            upper = middle
            f_upper = f_middle
        else:
            lower = middle
            f_lower = f_middle
    raise ValueError("max_iterations reached before rate_tolerance")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only normal, boundary, edge and validation cases."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": "import numpy as np\nC = np.array([1.0, 2.0])\nG = np.array([0.0, 2.0])\n",
            "call": "float(locate_log_remainder_threshold(C.copy(), G.copy(), .2, .05, .8))",
            "gold_call": "float(_oracle_locate_log_remainder_threshold(C.copy(), G.copy(), .2, .05, .8))",
            "tol": 2e-12,
        },
        {
            "setup": "import numpy as np\nC = np.array([1.0, 2.0])\nG = np.array([0.0, 2.0])\n",
            "call": "float(locate_log_remainder_threshold(C.copy(), G.copy(), .4, .2, .8, 1e-10, 100))",
            "gold_call": "float(_oracle_locate_log_remainder_threshold(C.copy(), G.copy(), .4, .2, .8, 1e-10, 100))",
            "tol": 2e-10,
        },
        {
            "setup": "import numpy as np\nC = np.array([1.0, .5, .2])\nG = np.array([0.0, .5, .075])\n",
            "call": "float(locate_log_remainder_threshold(C.copy(), G.copy(), .05, .05, 1.0))",
            "gold_call": "float(_oracle_locate_log_remainder_threshold(C.copy(), G.copy(), .05, .05, 1.0))",
            "tol": 2e-12,
        },
        {
            "setup": "import numpy as np\nC = np.array([1.0, 2.0])\nG = np.array([0.0, 2.0])\nT = abs(np.log(1.2) - .2) / abs(np.log(1.2))\n",
            "call": "float(locate_log_remainder_threshold(C.copy(), G.copy(), T, .1, .5))",
            "gold_call": "float(_oracle_locate_log_remainder_threshold(C.copy(), G.copy(), T, .1, .5))",
            "tol": 2e-12,
        },
        {
            "setup": "import numpy as np\nC = np.array([1.0, 2.0])\nG = np.array([0.0, 2.0])\nT = abs(np.log(2.0) - 1.0) / abs(np.log(2.0))\n",
            "call": "float(locate_log_remainder_threshold(C.copy(), G.copy(), T, .1, .5))",
            "gold_call": "float(_oracle_locate_log_remainder_threshold(C.copy(), G.copy(), T, .1, .5))",
            "tol": 2e-12,
        },
        {
            "setup": "import numpy as np\nC = np.array([1.0, 2.0])\nG = np.array([0.0, 2.0])\nT = abs(np.log(1.6) - .6) / abs(np.log(1.6))\n",
            "call": "float(locate_log_remainder_threshold(C.copy(), G.copy(), T, .1, .5))",
            "gold_call": "float(_oracle_locate_log_remainder_threshold(C.copy(), G.copy(), T, .1, .5))",
            "tol": 2e-12,
        },
        {
            "setup": status + "C = np.array([1.0, 2.0])\nG = np.array([0.0, 2.0])\n",
            "call": "_status(lambda: locate_log_remainder_threshold(C, G, .9, .05, .2))",
            "gold_call": "_status(lambda: _oracle_locate_log_remainder_threshold(C, G, .9, .05, .2))",
        },
        {
            "setup": status + "C = np.array([1.0, 2.0])\nG = np.array([0.0, 2.0])\n",
            "call": "_status(lambda: locate_log_remainder_threshold(C, G, .2, 0.0, .8))",
            "gold_call": "_status(lambda: _oracle_locate_log_remainder_threshold(C, G, .2, 0.0, .8))",
        },
        {
            "setup": status + "C = np.array([1.0, 2.0])\nG = np.array([0.0, 2.0])\n",
            "call": "_status(lambda: locate_log_remainder_threshold(C, G, .2, .05, .8, 1e-30, 2))",
            "gold_call": "_status(lambda: _oracle_locate_log_remainder_threshold(C, G, .2, .05, .8, 1e-30, 2))",
        },
    ]
