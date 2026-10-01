"""
Integrate the stripe's nominal stress over a stretch interval that may straddle the rearrangement.

The work done per unit reference area in stretching a homogeneously deforming tissue is the area under its nominal stress-stretch curve, which here includes the stress drop at rearrangement.

Returns
-------
float: the integral of the nominal stress over the stretch interval.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_stretching_work(
    stretch_start: float,
    stretch_end: float,
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Return the work per unit reference area between two stretches.

    Return the integral of ``compute_nominal_stress`` (with the given
    ``transition_stretch``, ``edge_scale``, ``kappa`` and ``chi``) with
    respect to the stretch from ``stretch_start`` to ``stretch_end``, with
    absolute accuracy ``1e-12``. The integral is negative when
    ``stretch_end < stretch_start``.

    Parameters
    ----------
    stretch_start : float
        Positive lower limit of integration.
    stretch_end : float
        Positive upper limit of integration.
    transition_stretch : float
        Stretch of the rearrangement; must exceed 1.
    edge_scale : float
        Positive edge-length scale of the reference regular hexagon.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.

    Returns
    -------
    float
        The work per unit reference area.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if ``transition_stretch`` does not exceed 1, or if
        ``solve_uniaxial_state`` rejects a cell state on the integration path.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_stretching_work(
    stretch_start: float,
    stretch_end: float,
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Reference implementation (exact difference of stored energy on each branch)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("stretch_start", stretch_start), ("stretch_end", stretch_end),
                 ("transition_stretch", transition_stretch), ("edge_scale", edge_scale),
                 ("kappa", kappa), ("chi", chi))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    lam_t = float(transition_stretch)
    k = float(edge_scale)
    if not lam_t > 1.0:
        raise ValueError("transition_stretch must exceed 1")
    chi_star = np.sqrt(8.0 * np.sqrt(3.0))
    jump = 2.0 * np.sqrt(3.0) / 4.5

    def _energy(cellular):
        # Stored energy per unit reference area of the free-sided honeycomb; its
        # derivative is the along-load nominal stress because the across-load
        # stress vanishes on this path.
        lateral = float(_oracle_solve_uniaxial_state(cellular, k, kappa, chi)[1])
        area = cellular * lateral * k * k
        perimeter = 0.5 * k * chi_star * (cellular + lateral)
        value = 0.5 * (area - 1.0) ** 2 + 0.5 * float(kappa) * (perimeter - float(chi)) ** 2
        return value / (k * k)

    def _validate_cellular_path(start, end):
        """Raise if the free-lateral state is invalid anywhere on this interval."""
        low, high = sorted((float(start), float(end)))
        # Positivity and geometric feasibility reduce to endpoint checks along
        # this free-lateral branch. The sign of perimeter - chi is a convex
        # quadratic in cellular stretch whose vertex is chi / (k * chi_star).
        _energy(low)
        if high > low:
            _energy(high)
            critical = float(chi) / (k * chi_star)
            if low < critical < high:
                _energy(critical)

    start = float(stretch_start)
    end = float(stretch_end)
    direction = 1.0 if end >= start else -1.0
    low, high = (start, end) if direction > 0.0 else (end, start)
    work = 0.0

    if low <= lam_t:
        pre_end = min(high, lam_t)
        _validate_cellular_path(low, pre_end)
        work += _energy(pre_end) - _energy(low)

    if high > lam_t:
        post_start = max(low, lam_t)
        cellular_start = jump * post_start
        cellular_end = jump * high
        _validate_cellular_path(cellular_start, cellular_end)
        work += _energy(cellular_end) - _energy(cellular_start)

    return float(direction * work)

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
            "call": "compute_stretching_work(1.2, 1.9, 1.45, 0.9, 0.3, 2.9)",
            "gold_call": "_oracle_compute_stretching_work(1.2, 1.9, 1.45, 0.9, 0.3, 2.9)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_stretching_work(1.0, 1.4, 1.45, 0.9, 0.3, 2.9)",
            "gold_call": "_oracle_compute_stretching_work(1.0, 1.4, 1.45, 0.9, 0.3, 2.9)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_stretching_work(1.6, 2.1, 1.45, 0.9, 0.3, 2.9)",
            "gold_call": "_oracle_compute_stretching_work(1.6, 2.1, 1.45, 0.9, 0.3, 2.9)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_stretching_work(1.85, 1.1, 1.53, 0.99784776, 0.16, 3.7) * 10.0",
            "gold_call": "_oracle_compute_stretching_work(1.85, 1.1, 1.53, 0.99784776, 0.16, 3.7) * 10.0",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_stretching_work(1.45, 1.45, 1.45, 0.9, 0.3, 2.9)",
            "gold_call": "_oracle_compute_stretching_work(1.45, 1.45, 1.45, 0.9, 0.3, 2.9)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_stretching_work(1.1, 1.4, 1.2, 0.62, 0.9, 1.6)",
            "gold_call": "_oracle_compute_stretching_work(1.1, 1.4, 1.2, 0.62, 0.9, 1.6)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_stretching_work(2.05, 2.15, 1.1, 1.0, 0.3, 4.0)",
            "gold_call": "_oracle_compute_stretching_work(2.05, 2.15, 1.1, 1.0, 0.3, 4.0)",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_stretching_work(0.5, 1.6, 2.0, 1.0, 0.3, 4.0))",
            "gold_call": "_status(lambda: _oracle_compute_stretching_work(0.5, 1.6, 2.0, 1.0, 0.3, 4.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_stretching_work(1.1, 1.6, 0.95, 0.9, 0.3, 2.9))",
            "gold_call": "_status(lambda: _oracle_compute_stretching_work(1.1, 1.6, 0.95, 0.9, 0.3, 2.9))",
        },
    ]
