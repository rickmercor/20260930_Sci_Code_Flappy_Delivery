"""
Find the lateral stretch and the along-load nominal stress of the honeycomb stripe pulled along the load with free lateral boundaries.

A stripe stretched between clamps with free side edges carries no transverse load, so its width contracts until the across-load stress vanishes.

Returns
-------
np.ndarray: [along-load nominal stress, lateral stretch] with zero across-load stress.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_uniaxial_state(
    stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> "np.ndarray":
    """Return the along-load nominal stress and lateral stretch under free lateral boundaries.

    The honeycomb of ``compute_honeycomb_stresses`` is stretched by
    ``stretch`` along the load while its lateral boundaries carry no load:
    the lateral stretch is the positive value at which the across-load
    nominal stress vanishes. Return that lateral stretch and the along-load
    nominal stress of the resulting state, both with absolute accuracy
    ``1e-12``.

    Parameters
    ----------
    stretch : float
        Positive stretch along the load.
    edge_scale : float
        Positive edge-length scale of the reference regular hexagon.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.

    Returns
    -------
    np.ndarray
        Float array ``[stress, lateral_stretch]``.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if no positive lateral stretch makes the across-load
        nominal stress vanish, or if ``compute_honeycomb_stresses`` rejects
        the resulting state.
    """
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_uniaxial_state(
    stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> "np.ndarray":
    """Reference implementation (the across-load stress is affine in the lateral stretch)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("stretch", stretch), ("edge_scale", edge_scale),
                 ("kappa", kappa), ("chi", chi))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    lam = float(stretch)
    k = float(edge_scale)
    chi_star = np.sqrt(8.0 * np.sqrt(3.0))
    edge = k * chi_star / 6.0
    # The relaxed perimeter is 3 * edge * (stretch + lateral), so these two trial
    # lateral stretches keep the edges in tension.
    first = max(float(chi) / (3.0 * edge) - lam, 0.0) + 1.0
    second = first + 1.0
    across_first = _oracle_compute_honeycomb_stresses(lam, first, k, kappa, chi)[1]
    across_second = _oracle_compute_honeycomb_stresses(lam, second, k, kappa, chi)[1]
    slope = (across_second - across_first) / (second - first)
    if not (np.isfinite(slope) and slope > 0.0):
        raise ValueError("the across-load stress does not rise with the lateral stretch")
    lateral = first - across_first / slope
    if not (np.isfinite(lateral) and lateral > 0.0):
        raise ValueError("no positive lateral stretch unloads the lateral boundaries")
    along = _oracle_compute_honeycomb_stresses(lam, lateral, k, kappa, chi)[0]
    return np.array([float(along), float(lateral)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    pair = (
        "import numpy as np\n"
        "chi_star = float(np.sqrt(8.0 * np.sqrt(3.0)))\n"
        "def _pair(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (2,):\n"
        "        return -1.0\n"
        "    return float(a[0] + 3.0 * a[1])\n"
    )
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
            "setup": pair,
            "call": "_pair(solve_uniaxial_state(1.4, 0.9, 0.3, 2.9))",
            "gold_call": "_pair(_oracle_solve_uniaxial_state(1.4, 0.9, 0.3, 2.9))",
        },
        {
            "setup": pair,
            "call": "_pair(solve_uniaxial_state(1.55, 0.99784776, 0.16, 3.7))",
            "gold_call": "_pair(_oracle_solve_uniaxial_state(1.55, 0.99784776, 0.16, 3.7))",
        },
        {
            "setup": pair,
            "call": "_pair(solve_uniaxial_state(1.0, 1.0, 0.5, chi_star))",
            "gold_call": "_pair(_oracle_solve_uniaxial_state(1.0, 1.0, 0.5, chi_star))",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_uniaxial_state(2.4, 0.9, 0.2, 3.0))",
            "gold_call": "_status(lambda: _oracle_solve_uniaxial_state(2.4, 0.9, 0.2, 3.0))",
        },
        {
            "setup": pair,
            "call": "float(solve_uniaxial_state(0.9, 0.85, 0.25, 2.6)[0])",
            "gold_call": "float(_oracle_solve_uniaxial_state(0.9, 0.85, 0.25, 2.6)[0])",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_uniaxial_state(5.0, 0.6, 2.0, 0.5))",
            "gold_call": "_status(lambda: _oracle_solve_uniaxial_state(5.0, 0.6, 2.0, 0.5))",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_uniaxial_state(0.0, 0.9, 0.3, 2.9))",
            "gold_call": "_status(lambda: _oracle_solve_uniaxial_state(0.0, 0.9, 0.3, 2.9))",
        },
    ]
