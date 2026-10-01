"""
Calibration compares active reaction estimates with their physiological variability intervals. A scenario is acceptable only when its physiological state space exists and its largest active interval excess meets the stated limit.

Calibration compares active reaction estimates with their physiological variability intervals. A scenario is acceptable only when its physiological state space exists and its largest active interval excess meets the stated limit.

Returns
-------
A finite ndarray of shape (C,P,2): compatibility flag and energy excess
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compatibility_panel(
    energies: "np.ndarray",
    directions: "np.ndarray",
    ranges: "np.ndarray",
    allowed_excess: float,
) -> "np.ndarray":
    """
    energies and directions have shape (C,P,N); directions contains -1,0,+1. ranges has
    shape (C,P,1+2N) in the preceding range convention. allowed_excess is finite and
    nonnegative in kJ mol^-1. Return (C,P,2) rows [compatible flag, largest active
    distance outside a range]. Compatibility uses an inclusive excess bound and the
    feasibility flag. The excess is 0 for infeasible rows or rows with no active
    coordinates; the feasibility flag still controls compatibility. Invalid data raise
    ValueError.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compatibility_panel(
    energies: "np.ndarray",
    directions: "np.ndarray",
    ranges: "np.ndarray",
    allowed_excess: float,
) -> "np.ndarray":
    import numpy as np

    e = np.asarray(energies, dtype=float)
    z = np.asarray(directions, dtype=float)
    t = np.asarray(ranges, dtype=float)
    a = float(allowed_excess)
    if (
        e.ndim != 3
        or min(e.shape) == 0
        or z.shape != e.shape
        or (t.shape != e.shape[:-1] + (1 + 2 * e.shape[-1],))
        or any((not np.isfinite(x).all() for x in (e, z, t)))
        or (not np.isin(z, [-1, 0, 1]).all())
        or (not np.isin(t[..., 0], [0, 1]).all())
        or (not np.isfinite(a))
        or (a < 0)
    ):
        raise ValueError("Invalid compatibility inputs")
    n = e.shape[-1]
    lower = t[..., 1 : 1 + n]
    upper = t[..., 1 + n :]
    if np.any((lower > upper) & (t[..., 0, None] == 1)):
        raise ValueError("Reversed feasible ranges")
    distance = np.maximum(np.maximum(lower - e, e - upper), 0.0)
    distance = np.where(z != 0, distance, 0.0)
    excess = np.max(distance, axis=-1)
    excess = np.where(t[..., 0] == 1, excess, 0.0)
    passed = (t[..., 0] == 1) & (excess <= a)
    return np.stack([passed.astype(float), excess], axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\n"
            "e = np.array([[[-2.0, 0.5]]])\n"
            "z = np.array([[[1.0, 0.0]]])\n"
            "p = np.array([[[1.0, -3.0, 2.0, -1.0, 3.0]]])",
            "call": "compatibility_panel(e,z,p,.1)",
            "gold_call": "_oracle_compatibility_panel(e,z,p,.1)",
        },
        {
            "setup": "import numpy as np\n"
            "e = np.array([[[-1.25]]])\n"
            "z = np.ones((1, 1, 1))\n"
            "p = np.array([[[1.0, -1.0, -0.1]]])",
            "call": "compatibility_panel(e,z,p,.25)",
            "gold_call": "_oracle_compatibility_panel(e,z,p,.25)",
        },
        {
            "setup": "import numpy as np\n"
            "e = np.array([[[-2.0]], [[2.0]]])\n"
            "z = np.ones((2, 1, 1))\n"
            "p = np.array([[[1.0, -1.5, -0.1]], [[0.0, 0.0, 0.0]]])",
            "call": "compatibility_panel(e,z,p,.1)",
            "gold_call": "_oracle_compatibility_panel(e,z,p,.1)",
        },
        {
            "setup": "import numpy as np\n"
            "e = np.array([[[1.0]]])\n"
            "z = np.array([[[2.0]]])\n"
            "p = np.array([[[1.0, -2.0, 2.0]]])\n"
            "\n"
            "def _error_check(fn, args):\n"
            "    try:\n"
            "        fn(*args)\n"
            "    except ValueError:\n"
            "        return 1.0\n"
            "    return 0.0",
            "call": "_error_check(compatibility_panel, (e,z,p,0.1,))",
            "gold_call": "_error_check(_oracle_compatibility_panel, (e,z,p,0.1,))",
            "tol": 0.0,
        },
    ]
