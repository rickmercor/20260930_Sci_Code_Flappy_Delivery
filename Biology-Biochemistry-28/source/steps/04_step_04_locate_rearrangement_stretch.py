"""
Find the stretch at which the load-perpendicular cell edges of the freely contracting stripe shrink to the rearrangement threshold.

Stretching an ordered epithelium shortens the junctions perpendicular to the load, and a junction that becomes short enough is resolved by a T1 neighbour exchange.

Returns
-------
float: the smallest stretch above 1 at which the load-perpendicular edge equals the threshold.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_rearrangement_stretch(
    threshold: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Return the stretch at which the load-perpendicular edges reach the rearrangement threshold.

    Under the free-lateral stretching of ``solve_uniaxial_state``, the cell
    edges perpendicular to the load shorten as the stretch grows. Return the
    smallest stretch in ``(1, 10]`` at which their length equals
    ``threshold`` (in the units of the cell energy, the square root of the
    preferred cell area), with absolute accuracy ``1e-12``. Beyond that
    first crossing the edge keeps shrinking and can turn negative, where
    ``solve_uniaxial_state`` rejects the stretch, so locate the crossing
    from below (for example by stepping up from stretch 1) rather than by
    evaluating the edge at stretch 10.


    Parameters
    ----------
    threshold : float
        Positive edge length at which the edge rearranges.
    edge_scale : float
        Positive edge-length scale of the reference regular hexagon.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.

    Returns
    -------
    float
        The rearrangement stretch.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if the load-perpendicular edge at stretch 1 is not longer
        than ``threshold``, if that edge does not shrink to ``threshold`` for
        any stretch up to 10, or if ``solve_uniaxial_state`` rejects a
        stretch before the edge reaches ``threshold``.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_locate_rearrangement_stretch(
    threshold: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Reference implementation (march then bisection on the edge length)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("threshold", threshold), ("edge_scale", edge_scale),
                 ("kappa", kappa), ("chi", chi))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    level = float(threshold)
    k = float(edge_scale)
    edge = k * np.sqrt(8.0 * np.sqrt(3.0)) / 6.0

    def _excess(lam):
        # At 120-degree junctions the slanted edges have length edge * stretch and
        # rise by half of it, so the load-perpendicular edge is the row spacing
        # 1.5 * edge * lateral minus that rise.
        lateral = float(_oracle_solve_uniaxial_state(lam, k, kappa, chi)[1])
        return edge * (1.5 * lateral - 0.5 * lam) - level

    if not _excess(1.0) > 0.0:
        raise ValueError("the edge at stretch 1 is not longer than the threshold")
    low = 1.0
    high = None
    for lam in 1.0 + 0.01 * np.arange(1, 901):
        if _excess(float(lam)) <= 0.0:
            high = float(lam)
            break
        low = float(lam)
    if high is None:
        raise ValueError("the edge does not reach the threshold up to stretch 10")
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _excess(middle) > 0.0:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    star = (
        "import numpy as np\n"
        "chi_star = float(np.sqrt(8.0 * np.sqrt(3.0)))\n"
    )
    status = (
        "import numpy as np\n"
        "chi_star = float(np.sqrt(8.0 * np.sqrt(3.0)))\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": star,
            "call": "locate_rearrangement_stretch(0.1, 0.99784776, 0.16, 3.7)",
            "gold_call": "_oracle_locate_rearrangement_stretch(0.1, 0.99784776, 0.16, 3.7)",
        },
        {
            "setup": star,
            "call": "locate_rearrangement_stretch(0.05, 0.9, 0.3, 2.9)",
            "gold_call": "_oracle_locate_rearrangement_stretch(0.05, 0.9, 0.3, 2.9)",
        },
        {
            "setup": star,
            "call": "locate_rearrangement_stretch(0.6, 1.0, 0.5, chi_star)",
            "gold_call": "_oracle_locate_rearrangement_stretch(0.6, 1.0, 0.5, chi_star)",
        },
        {
            "setup": star,
            "call": "locate_rearrangement_stretch(0.004, 0.62, 0.9, 1.6)",
            "gold_call": "_oracle_locate_rearrangement_stretch(0.004, 0.62, 0.9, 1.6)",
        },
        {
            "setup": status,
            "call": "_status(lambda: locate_rearrangement_stretch(0.7, 1.0, 0.5, chi_star))",
            "gold_call": "_status(lambda: _oracle_locate_rearrangement_stretch(0.7, 1.0, 0.5, chi_star))",
        },
        {
            "setup": status,
            "call": "_status(lambda: locate_rearrangement_stretch(-0.1, 0.9, 0.3, 2.9))",
            "gold_call": "_status(lambda: _oracle_locate_rearrangement_stretch(-0.1, 0.9, 0.3, 2.9))",
        },
    ]
