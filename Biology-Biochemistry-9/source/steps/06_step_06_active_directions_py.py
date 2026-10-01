"""
Identify active reaction directions.

The physiological calculation uses directions of significantly active net reactions. Kinetically inactive coordinates carry no prescribed driving-force sign.

Returns
-------
A same-shaped float ndarray containing only -1, 0 and +1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def active_directions(
    nets: "np.ndarray", activity_threshold: float
) -> "np.ndarray":
    """
    nets is any nonempty finite array. activity_threshold is positive and finite in net-
    flux units. Return a same-shaped numeric array with +1 or -1 for positive or
    negative active coordinates, and 0 when the absolute net value is at most the
    threshold. Invalid inputs raise ValueError.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_active_directions(
    nets: "np.ndarray", activity_threshold: float
) -> "np.ndarray":
    import numpy as np

    v = np.asarray(nets, dtype=float)
    t = float(activity_threshold)
    if v.size == 0 or not np.isfinite(v).all() or (not np.isfinite(t)) or (t <= 0):
        raise ValueError("Finite nets and positive activity threshold required")
    return np.where(np.abs(v) > t, np.sign(v), 0.0).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nv = np.array([-3.0, 0.0, 0.2, 4.0])",
            "call": "active_directions(v,.1)",
            "gold_call": "_oracle_active_directions(v,.1)",
        },
        {
            "setup": "import numpy as np\nv = np.array([-1e-07, 1e-07, 1.00001e-07])",
            "call": "active_directions(v,1e-7)",
            "gold_call": "_oracle_active_directions(v,1e-7)",
        },
        {
            "setup": "import numpy as np\nv = np.array([[1e-12, -1e-12]])",
            "call": "active_directions(v,1e-15)",
            "gold_call": "_oracle_active_directions(v,1e-15)",
        },
        {
            "setup": "import numpy as np\n"
            "v = np.array([1.0])\n"
            "\n"
            "def _error_check(fn, args):\n"
            "    try:\n"
            "        fn(*args)\n"
            "    except ValueError:\n"
            "        return 1.0\n"
            "    return 0.0",
            "call": "_error_check(active_directions, (v,-1.0,))",
            "gold_call": "_error_check(_oracle_active_directions, (v,-1.0,))",
            "tol": 0.0,
        },
    ]
