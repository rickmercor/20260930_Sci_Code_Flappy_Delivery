"""
Evaluate the homogeneous-deformation nominal stress of the stripe on both sides of the collective T1 rearrangement.

In the mean-field picture a T1 exchange lets more cells share the tissue elongation along the load, so the cellular stretch, and with it the stress, drops abruptly at the rearrangement stretch.

Returns
-------
float: the nominal stress of the stripe at the given tissue stretch.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_nominal_stress(
    stretch: float,
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Return the nominal stress of the stripe across the rearrangement.

    For ``stretch <= transition_stretch`` the stripe is the honeycomb of
    ``solve_uniaxial_state`` stretched by ``stretch``. At
    ``transition_stretch`` every four-cell unit built around a
    load-perpendicular edge exchanges neighbours: the unit keeps its length
    along the load but now holds three cells along the load where it held
    two, and the stripe becomes the same honeycomb turned by 90 degrees (one
    pair of edges of every cell parallel to the load). For
    ``stretch > transition_stretch`` the turned honeycomb is stretched along
    the load in proportion to the tissue stretch, with free lateral
    boundaries. Return the nominal stress, the derivative with respect to
    ``stretch`` of the elastic energy per unit reference area, with absolute
    accuracy ``1e-12``.

    Parameters
    ----------
    stretch : float
        Positive tissue stretch along the load.
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
        The nominal stress along the load.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if ``transition_stretch`` does not exceed 1, or if
        ``solve_uniaxial_state`` rejects the cell state reached.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_nominal_stress(
    stretch: float,
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
) -> float:
    """Reference implementation (cellular stretch jump plus energy conjugacy)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("stretch", stretch), ("transition_stretch", transition_stretch),
                 ("edge_scale", edge_scale), ("kappa", kappa), ("chi", chi))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    lam = float(stretch)
    lam_t = float(transition_stretch)
    k = float(edge_scale)
    if not lam_t > 1.0:
        raise ValueError("transition_stretch must exceed 1")
    if lam <= lam_t:
        return float(_oracle_solve_uniaxial_state(lam, k, kappa, chi)[0])
    edge = k * np.sqrt(8.0 * np.sqrt(3.0)) / 6.0
    period = np.sqrt(3.0) * edge
    spacing = 1.5 * edge
    # Two cells of extent lam_t * period become three cells whose extent along
    # the load is measured on the turned cell, i.e. on the row spacing.
    jump = 2.0 * period / (3.0 * spacing)
    cellular = jump * lam
    # The relaxed cell energy is symmetric in its two extents, so the turned
    # honeycomb with free sides is the uniaxial state at the cellular stretch;
    # the energy per reference area depends on stretch only through
    # cellular = jump * stretch, hence the factor jump.
    return float(jump * _oracle_solve_uniaxial_state(cellular, k, kappa, chi)[0])

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
            "call": "compute_nominal_stress(1.3, 1.53, 0.99784776, 0.16, 3.7)",
            "gold_call": "_oracle_compute_nominal_stress(1.3, 1.53, 0.99784776, 0.16, 3.7)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_nominal_stress(1.53, 1.53, 0.99784776, 0.16, 3.7) * 10.0",
            "gold_call": "_oracle_compute_nominal_stress(1.53, 1.53, 0.99784776, 0.16, 3.7) * 10.0",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_nominal_stress(1.9, 1.53, 0.99784776, 0.16, 3.7) * 10.0",
            "gold_call": "_oracle_compute_nominal_stress(1.9, 1.53, 0.99784776, 0.16, 3.7) * 10.0",
        },
        {
            "setup": "import numpy as np\nabove = float(np.nextafter(1.45, 2.0))\n",
            "call": "compute_nominal_stress(above, 1.45, 0.9, 0.3, 2.9)",
            "gold_call": "_oracle_compute_nominal_stress(above, 1.45, 0.9, 0.3, 2.9)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_nominal_stress(1.9, 1.45, 0.9, 0.3, 2.9)",
            "gold_call": "_oracle_compute_nominal_stress(1.9, 1.45, 0.9, 0.3, 2.9)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_nominal_stress(0.95, 1.45, 0.9, 0.3, 2.9)",
            "gold_call": "_oracle_compute_nominal_stress(0.95, 1.45, 0.9, 0.3, 2.9)",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_nominal_stress(1.2, 1.0, 0.9, 0.3, 2.9))",
            "gold_call": "_status(lambda: _oracle_compute_nominal_stress(1.2, 1.0, 0.9, 0.3, 2.9))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_nominal_stress(float('inf'), 1.4, 0.9, 0.3, 2.9))",
            "gold_call": "_status(lambda: _oracle_compute_nominal_stress(float('inf'), 1.4, 0.9, 0.3, 2.9))",
        },
    ]
