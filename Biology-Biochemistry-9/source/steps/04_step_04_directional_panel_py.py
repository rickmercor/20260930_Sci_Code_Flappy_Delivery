"""
Recover the directional flux components.

The directional representation records two opposing microscopic contributions for each net reaction. Its stationary normalization is fixed by the matching transcriptomic treatment.

Returns
-------
A positive finite ndarray of shape nets.shape + (2,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def directional_panel(nets: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """
    nets and weights are same-shaped nonempty finite arrays; weights is strictly
    positive. Recover the source-matched directional pair for each net value and weight.
    The added last axis is [forward, reverse]; all earlier axes are unchanged. Raise
    ValueError for invalid inputs or unrepresentable positive directional values.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_directional_panel(
    nets: "np.ndarray", weights: "np.ndarray"
) -> "np.ndarray":
    import numpy as np

    v = np.asarray(nets, dtype=float)
    g = np.asarray(weights, dtype=float)
    if (
        v.shape != g.shape
        or v.size == 0
        or (not np.isfinite(v).all())
        or (not np.isfinite(g).all())
        or np.any(g <= 0)
    ):
        raise ValueError("Aligned finite nets and positive weights required")
    a = g / np.e
    large = (np.hypot(v, 2 * a) + np.abs(v)) / 2
    small = a * (a / large)
    f = np.where(v >= 0, large, small)
    r = np.where(v >= 0, small, large)
    result = np.stack([f, r], axis=-1)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("Directional values outside representable range")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\n"
            "v = np.array([0.0, 2.0, -3.0])\n"
            "g = np.array([1.0, 2.0, 4.0])",
            "call": "directional_panel(v,g)",
            "gold_call": "_oracle_directional_panel(v,g)",
        },
        {
            "setup": "import numpy as np\nv = np.array([0.0])\ng = np.array([7.0])",
            "call": "directional_panel(v,g)",
            "gold_call": "_oracle_directional_panel(v,g)",
        },
        {
            "setup": "import numpy as np\n"
            "v = np.array([100000000.0, -100000000.0])\n"
            "g = np.array([0.0001, 0.0001])",
            "call": "directional_panel(v,g)",
            "gold_call": "_oracle_directional_panel(v,g)",
        },
        {
            "setup": "import numpy as np\n"
            "v = np.array([1.0, 2.0])\n"
            "g = np.array([1.0])\n"
            "\n"
            "def _error_check(fn, args):\n"
            "    try:\n"
            "        fn(*args)\n"
            "    except ValueError:\n"
            "        return 1.0\n"
            "    return 0.0",
            "call": "_error_check(directional_panel, (v,g,))",
            "gold_call": "_error_check(_oracle_directional_panel, (v,g,))",
            "tol": 0.0,
        },
    ]
