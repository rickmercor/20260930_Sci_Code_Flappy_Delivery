"""
Evaluate the two nominal stresses of an ordered honeycomb whose cells follow a homogeneous tissue stretch while their remaining vertex freedom relaxes.

In a mean-field reading of the vertex model the tissue stretches set each cell's extents along and across the load, and the cell shape otherwise relaxes to its minimum energy.

Returns
-------
np.ndarray: [nominal stress along the load, nominal stress across the load].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_honeycomb_stresses(
    stretch: float,
    lateral_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> "np.ndarray":
    """Return the two nominal stresses of a homogeneously stretched honeycomb.

    The reference tissue is the honeycomb of regular hexagons whose edge
    length is ``edge_scale`` times that of the unit-area regular hexagon,
    oriented so that one pair of edges of every cell is perpendicular to the
    load. The tissue is stretched by ``stretch`` along the load and by
    ``lateral_stretch`` across it: every cell's extent along the load (the
    lattice period in that direction) and across it (the spacing of the
    cell rows) scale with these stretches, and the remaining vertex degree
    of freedom of each cell relaxes to minimise the cell energy
    ``e = (a - 1)**2 / 2 + kappa * (p - chi)**2 / 2`` of its dimensionless
    area ``a`` and perimeter ``p``.

    Return the nominal stresses conjugate to the two stretches: the partial
    derivatives of the elastic energy per unit reference area with respect
    to ``stretch`` and to ``lateral_stretch``.

    Parameters
    ----------
    stretch : float
        Positive stretch along the load.
    lateral_stretch : float
        Positive stretch across the load.
    edge_scale : float
        Positive edge-length scale of the reference regular hexagon.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.

    Returns
    -------
    np.ndarray
        Float array ``[along, across]`` of the two nominal stresses.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), or if the smallest perimeter a cell with these extents can
        have is below ``chi`` by more than a relative ``1e-12`` (edges not
        under tension, a regime this model does not cover), or if the relaxed
        120-degree shape would have a negative load-perpendicular edge by more
        than a relative ``1e-12``.
    """
    return stresses

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_honeycomb_stresses(
    stretch: float,
    lateral_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> "np.ndarray":
    """Reference implementation (relaxed hexagon at fixed extents)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("stretch", stretch), ("lateral_stretch", lateral_stretch),
                 ("edge_scale", edge_scale), ("kappa", kappa), ("chi", chi))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    lam = float(stretch)
    lat = float(lateral_stretch)
    k = float(edge_scale)
    chi_star = np.sqrt(8.0 * np.sqrt(3.0))
    edge = k * chi_star / 6.0
    period = np.sqrt(3.0) * edge * lam
    spacing = 1.5 * edge * lat
    # With both extents fixed the area is period * spacing whatever the shape,
    # and the perimeter 2 L / cos(phi) + 2 H - L tan(phi) of a hexagon whose
    # four slanted edges make the angle phi with the load is smallest at
    # phi = 30 degrees (every interior angle 120 degrees). Under tension that
    # shape minimises e.
    phi = np.pi / 6.0
    area = period * spacing
    perimeter = 2.0 * period / np.cos(phi) + 2.0 * spacing - period * np.tan(phi)
    perpendicular_edge = spacing - 0.5 * period * np.tan(phi)
    geometry_tolerance = 1e-12 * max(period, spacing)
    if perpendicular_edge < -geometry_tolerance:
        raise ValueError("the relaxed honeycomb has a negative load-perpendicular edge")
    if perimeter < float(chi) * (1.0 - 1e-12):
        raise ValueError("cell edges are not under tension at these stretches")
    pressure = area - 1.0
    tension = float(kappa) * (perimeter - float(chi))
    d_area = np.array([area / lam, area / lat])
    d_perimeter = np.array([(2.0 / np.cos(phi) - np.tan(phi)) * period / lam,
                            2.0 * spacing / lat])
    reference_area = np.sqrt(3.0) * 1.5 * edge * edge
    return (pressure * d_area + tension * d_perimeter) / reference_area

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
            "call": "_pair(compute_honeycomb_stresses(1.3, 0.75, 0.9, 0.3, 2.9))",
            "gold_call": "_pair(_oracle_compute_honeycomb_stresses(1.3, 0.75, 0.9, 0.3, 2.9))",
        },
        {
            "setup": pair,
            "call": "float(compute_honeycomb_stresses(1.7, 0.6, 0.62, 0.9, 1.6)[0])",
            "gold_call": "float(_oracle_compute_honeycomb_stresses(1.7, 0.6, 0.62, 0.9, 1.6)[0])",
        },
        {
            "setup": pair,
            "call": "_pair(compute_honeycomb_stresses(1.0, 1.0, 1.0, 0.5, chi_star))",
            "gold_call": "_pair(_oracle_compute_honeycomb_stresses(1.0, 1.0, 1.0, 0.5, chi_star))",
        },
        {
            "setup": pair,
            "call": "_pair(compute_honeycomb_stresses(1.15, 1.15, 0.95, 0.2, 3.0))",
            "gold_call": "_pair(_oracle_compute_honeycomb_stresses(1.15, 1.15, 0.95, 0.2, 3.0))",
        },
        {
            "setup": pair,
            "call": "_pair(compute_honeycomb_stresses(1.0, 1.4, 0.7, 1.2, 1.9))",
            "gold_call": "_pair(_oracle_compute_honeycomb_stresses(1.0, 1.4, 0.7, 1.2, 1.9))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_honeycomb_stresses(0.8, 0.8, 1.0, 0.3, 3.6))",
            "gold_call": "_status(lambda: _oracle_compute_honeycomb_stresses(0.8, 0.8, 1.0, 0.3, 3.6))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_honeycomb_stresses(-1.2, 0.8, 1.0, 0.3, 2.0))",
            "gold_call": "_status(lambda: _oracle_compute_honeycomb_stresses(-1.2, 0.8, 1.0, 0.3, 2.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_honeycomb_stresses(1.2, 0.8, 1.0, True, 2.0))",
            "gold_call": "_status(lambda: _oracle_compute_honeycomb_stresses(1.2, 0.8, 1.0, True, 2.0))",
        },
    ]
