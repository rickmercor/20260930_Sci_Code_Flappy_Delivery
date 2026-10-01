"""
Find the stretch at which the shortest load-perpendicular junction of the relaxed stripe first shrinks to the neighbour-exchange threshold.

Stretching an ordered epithelium shortens the junctions perpendicular to the load, and a junction that becomes short enough is resolved by a T1 neighbour exchange.

Returns
-------
float: the stretch at which the shortest relaxed load-perpendicular junction first equals the threshold.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_bifurcation_stretch(
    n_rows: int,
    n_columns: int,
    kappa: float,
    chi: float,
    line_tension: float,
    threshold: float,
    tolerance: float,
) -> float:
    """Return the stretch at which a load-perpendicular junction reaches ``threshold``.

    Build the stripe of ``build_stress_free_stripe``. At stretch ``lam`` its
    vertex ``x`` coordinates and its period are multiplied by ``lam``, and
    ``relax_stripe`` (gradient tolerance ``1e-12``) is applied from that
    configuration with the given ``line_tension``. The load-perpendicular
    junctions are local edge 0 of every cell in ``compute_cell_geometry``.
    Let ``g(lam)`` be the length of the shortest of them minus
    ``threshold``. Step ``lam`` upward from 1 in increments of 0.05 until
    ``g(lam) <= 0``, then bisect the last increment, keeping ``g > 0`` at
    the lower end and ``g <= 0`` at the upper end, until the bracket is no
    wider than ``tolerance``, and return its midpoint.

    Parameters
    ----------
    n_rows, n_columns, kappa, chi
        Stripe as in ``build_stress_free_stripe``.
    line_tension : float
        Non-negative line tension of the free edges.
    threshold : float
        Positive junction length at which a neighbour exchange occurs.
    tolerance : float
        Positive bracket width, at most ``1e-6``.

    Returns
    -------
    float
        The stretch at the first threshold crossing.

    Raises
    ------
    ValueError
        If ``line_tension`` is not a finite non-negative real number,
        ``threshold`` is not a finite positive real number, ``tolerance`` is
        not a finite positive real number at most ``1e-6`` (booleans are
        rejected), ``g(1) <= 0``, ``g`` stays positive up to stretch 4, or an
        earlier step rejects a state it visits.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_locate_bifurcation_stretch(
    n_rows: int,
    n_columns: int,
    kappa: float,
    chi: float,
    line_tension: float,
    threshold: float,
    tolerance: float,
) -> float:
    """Reference implementation (march in 0.05 increments, then bisection on relaxed states)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not (_is_number(line_tension) and line_tension >= 0.0):
        raise ValueError("line_tension must be a finite non-negative number")
    if not (_is_number(threshold) and threshold > 0.0):
        raise ValueError("threshold must be a finite positive number")
    if not (_is_number(tolerance) and 0.0 < tolerance <= 1e-6):
        raise ValueError("tolerance must be a finite number in (0, 1e-6]")
    vertices, cells, shifts, period = _oracle_build_stress_free_stripe(n_rows, n_columns, kappa, chi)

    def _gap(stretch):
        start = vertices.copy()
        start[:, 0] *= stretch
        relaxed = _oracle_relax_stripe(start, cells, shifts, stretch * period, kappa, chi,
                                       line_tension, 1e-12)
        lengths = _oracle_compute_cell_geometry(relaxed, cells, shifts, stretch * period)[2]
        return float(np.min(lengths[:, 0])) - float(threshold)

    if not _gap(1.0) > 0.0:
        raise ValueError("the load-perpendicular junctions are not longer than the threshold at stretch 1")
    low, high = 1.0, None
    for step in range(1, 61):
        stretch = 1.0 + 0.05 * step
        if _gap(stretch) <= 0.0:
            high = stretch
            break
        low = stretch
    if high is None:
        raise ValueError("no junction reaches the threshold up to stretch 4")
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _gap(middle) > 0.0:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))

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
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "locate_bifurcation_stretch(3, 2, 0.16, 3.5, 0.03, 0.1, 1e-10)",
            "gold_call": "_oracle_locate_bifurcation_stretch(3, 2, 0.16, 3.5, 0.03, 0.1, 1e-10)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "locate_bifurcation_stretch(2, 2, 0.35, 2.45, 0.05, 0.08, 1e-10)",
            "gold_call": "_oracle_locate_bifurcation_stretch(2, 2, 0.35, 2.45, 0.05, 0.08, 1e-10)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "locate_bifurcation_stretch(4, 2, 0.6, 1.8, 0.0, 0.12, 1e-10)",
            "gold_call": "_oracle_locate_bifurcation_stretch(4, 2, 0.6, 1.8, 0.0, 0.12, 1e-10)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "locate_bifurcation_stretch(1, 3, 0.25, 3.3, 0.02, 0.1, 1e-10)",
            "gold_call": "_oracle_locate_bifurcation_stretch(1, 3, 0.25, 3.3, 0.02, 0.1, 1e-10)",
        },
        {
            "setup": status,
            "call": "_status(lambda: locate_bifurcation_stretch(3, 2, 0.16, 3.5, 0.03, 0.7, 1e-10))",
            "gold_call": "_status(lambda: _oracle_locate_bifurcation_stretch(3, 2, 0.16, 3.5, 0.03, 0.7, 1e-10))",
        },
        {
            "setup": status,
            "call": "_status(lambda: locate_bifurcation_stretch(3, 2, 0.16, 3.5, 0.03, 0.1, 1e-3))",
            "gold_call": "_status(lambda: _oracle_locate_bifurcation_stretch(3, 2, 0.16, 3.5, 0.03, 0.1, 1e-3))",
        },
    ]
