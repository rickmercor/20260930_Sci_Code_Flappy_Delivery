"""
Find the edge-length scale of the regular hexagonal cell that is stress-free under the area-perimeter cell energy.

An ordered vertex-model sheet relaxes to regular hexagons whose size balances area elasticity against perimeter tension, and every later stretch is measured from that stress-free state.

Returns
-------
float: the edge-length scale k of the stress-free regular hexagon.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_stress_free_scale(kappa: float, chi: float) -> float:
    """Return the edge-length scale of the stress-free regular hexagonal cell.

    A cell of dimensionless area ``a`` and perimeter ``p`` carries the energy
    ``e = (a - 1)**2 / 2 + kappa * (p - chi)**2 / 2``. Among regular
    hexagons, return the ratio ``k`` of the edge length that minimises ``e``
    to the edge length of the regular hexagon of unit area, with relative
    accuracy ``1e-13``.

    Parameters
    ----------
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter (shape index).

    Returns
    -------
    float
        The edge-length scale ``k``.

    Raises
    ------
    ValueError
        If ``kappa`` or ``chi`` is not a finite positive real number
        (booleans are rejected).
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_stress_free_scale(kappa: float, chi: float) -> float:
    """Reference implementation (bracketed bisection with a Newton polish)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    if not (_is_number(kappa) and kappa > 0.0):
        raise ValueError("kappa must be a finite positive number")
    if not (_is_number(chi) and chi > 0.0):
        raise ValueError("chi must be a finite positive number")
    kappa = float(kappa)
    chi = float(chi)
    # Perimeter of the unit-area regular hexagon; a hexagon of edge scale k has
    # area k**2 and perimeter k * chi_star.
    chi_star = np.sqrt(8.0 * np.sqrt(3.0))

    def _slope(k):
        return 2.0 * k * (k * k - 1.0) + kappa * chi_star * (k * chi_star - chi)

    # 2 k^3 + (kappa chi*^2 - 2) k - kappa chi* chi has one sign change in its
    # coefficients, hence one positive root; the slope is negative at k = 0 and
    # non-negative at max(1, chi / chi*).
    low = 0.0
    high = max(1.0, chi / chi_star)
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            break
        if _slope(middle) < 0.0:
            low = middle
        else:
            high = middle
    root = 0.5 * (low + high)
    for _ in range(2):
        curvature = 6.0 * root * root - 2.0 + kappa * chi_star * chi_star
        if curvature <= 0.0:
            break
        step = _slope(root) / curvature
        if not (np.isfinite(step) and low <= root - step <= high):
            break
        root -= step
    return float(root)

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
    star = (
        "import numpy as np\n"
        "chi_star = float(np.sqrt(8.0 * np.sqrt(3.0)))\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "solve_stress_free_scale(0.16, 3.7)",
            "gold_call": "_oracle_solve_stress_free_scale(0.16, 3.7)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "solve_stress_free_scale(0.6, 1.8)",
            "gold_call": "_oracle_solve_stress_free_scale(0.6, 1.8)",
        },
        {
            "setup": star,
            "call": "solve_stress_free_scale(0.8, chi_star)",
            "gold_call": "_oracle_solve_stress_free_scale(0.8, chi_star)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "solve_stress_free_scale(40.0, 1.5)",
            "gold_call": "_oracle_solve_stress_free_scale(40.0, 1.5)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "solve_stress_free_scale(0.05, 4.6)",
            "gold_call": "_oracle_solve_stress_free_scale(0.05, 4.6)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(solve_stress_free_scale(2e-4, 3.1) ** 2)",
            "gold_call": "float(_oracle_solve_stress_free_scale(2e-4, 3.1) ** 2)",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_stress_free_scale(0.0, 3.0))",
            "gold_call": "_status(lambda: _oracle_solve_stress_free_scale(0.0, 3.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_stress_free_scale(0.3, float('nan')))",
            "gold_call": "_status(lambda: _oracle_solve_stress_free_scale(0.3, float('nan')))",
        },
    ]
