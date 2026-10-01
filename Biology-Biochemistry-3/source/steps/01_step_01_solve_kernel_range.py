"""
Fix the range parameter of the adhesion distance kernel so that the kernel is minimal at a prescribed centre-to-centre distance.

In polarity-dependent adhesion models the pair potential is a polarity factor times one distance kernel, so the kernel minimum sets the spacing of adhering cells whatever their polarities.

Returns
-------
float: the kernel range parameter beta.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def solve_kernel_range(rest_distance: float) -> float:
    """Return the kernel range parameter that places the kernel minimum at a given distance.

    The distance kernel is ``U(r) = exp(-r) - exp(-r / beta)``. Return the
    value ``beta != 1`` for which ``U`` has a strict local minimum at
    ``r = rest_distance``, with relative accuracy ``1e-13``.

    Parameters
    ----------
    rest_distance : float
        Distance at which the kernel must be minimal.

    Returns
    -------
    float
        The range parameter ``beta``.

    Raises
    ------
    ValueError
        If ``rest_distance`` is not a finite real number (booleans are
        rejected), or if no ``beta != 1`` gives ``U`` a strict local minimum
        at ``rest_distance``.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_kernel_range(rest_distance: float) -> float:
    """Reference implementation (bisection on the non-trivial stationarity root)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not _is_number(rest_distance):
        raise ValueError("rest_distance must be a finite real number")
    distance = float(rest_distance)
    # U'(r) = 0 reads beta * exp(-r) = exp(-r / beta). With x = r / beta this is
    # x - ln(x) = r - ln(r): one root is x = r (beta = 1, U identically zero),
    # the other lies on the far side of x = 1. U''(r) has the sign of
    # 1 - 1 / beta there, so a strict minimum needs beta > 1, i.e. x < 1,
    # which exists only when r > 1.
    if distance <= 1.0:
        raise ValueError("no kernel range gives a strict minimum at this distance")
    level = distance - np.log(distance)

    def _excess(x):
        return x - np.log(x) - level

    low = 0.5 * np.exp(-level)
    high = 1.0
    # _excess is strictly decreasing on (0, 1), positive at low, negative at 1.
    for _ in range(400):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _excess(middle) > 0.0:
            low = middle
        else:
            high = middle
    root = 0.5 * (low + high)
    for _ in range(3):
        slope = 1.0 - 1.0 / root
        if slope == 0.0:
            break
        step = _excess(root) / slope
        if not np.isfinite(step) or not (0.0 < root - step < 1.0):
            break
        root -= step
    beta = distance / root
    if not (np.isfinite(beta) and beta > 1.0):
        raise ValueError("no kernel range gives a strict minimum at this distance")
    return float(beta)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    stationary = (
        "import numpy as np\n"
        "def _minimum(r, b):\n"
        "    slope = -np.exp(-r) + np.exp(-r / b) / b\n"
        "    curvature = np.exp(-r) - np.exp(-r / b) / b ** 2\n"
        "    return float(1.0e6 * slope + (1.0 if curvature > 1e-6 else 0.0))\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "solve_kernel_range(1.6)",
            "gold_call": "_oracle_solve_kernel_range(1.6)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "solve_kernel_range(2.7) / 10.0",
            "gold_call": "_oracle_solve_kernel_range(2.7) / 10.0",
        },
        {
            "setup": "import numpy as np\n",
            "call": "solve_kernel_range(1.05)",
            "gold_call": "_oracle_solve_kernel_range(1.05)",
        },
        {
            "setup": stationary,
            "call": "_minimum(3.1, solve_kernel_range(3.1))",
            "gold_call": "_minimum(3.1, _oracle_solve_kernel_range(3.1))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(np.exp(-2.35) - np.exp(-2.35 / solve_kernel_range(2.35)))",
            "gold_call": "float(np.exp(-2.35) - np.exp(-2.35 / _oracle_solve_kernel_range(2.35)))",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_kernel_range(0.8))",
            "gold_call": "_status(lambda: _oracle_solve_kernel_range(0.8))",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_kernel_range(float('nan')))",
            "gold_call": "_status(lambda: _oracle_solve_kernel_range(float('nan')))",
        },
    ]
