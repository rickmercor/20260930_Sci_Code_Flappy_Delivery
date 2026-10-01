"""
Evaluate distinct thickness averages for symmetric Q2D screening.

The response average concerns an orbital at a specified height and one uniform

coordinate across the slab. The interaction average concerns two independent

uniform coordinates. Their distinction is part of the source construction.

Returns
-------
a real array (G, A+1) of one- and two-coordinate averages.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def thickness_averages(
    magnitudes: "np.ndarray", heights: "np.ndarray", thickness: float
) -> "np.ndarray":
    """Return dimensionless averages of the off-plane Coulomb exponential.

    Parameters
    ----------
    magnitudes : "np.ndarray"
        Finite nonnegative real array (G,), G >= 1, of |q+G| in inverse
        angstroms.
    heights : "np.ndarray"
        Finite real array (A,), A >= 1, of orbital heights in angstroms.
        Each lies in [-thickness/2, thickness/2].
    thickness : float
        Finite nonnegative slab thickness in angstroms. At zero thickness,
        all heights must be zero and all returned averages equal one.

    Returns
    -------
    averages : "np.ndarray"
        Real array (G, A+1). Column a < A is the uniform slab average of
        exp(-p*abs(z-heights[a])) over z. The last column is the average of
        exp(-p*abs(z-z_prime)) over two independent uniform slab positions.
        At p=0 both kinds of average are exactly one. Results must remain
        accurate for small p*thickness, including values below 1e-8.

    Raises
    ------
    ValueError
        If inputs violate the shapes, finiteness, real-valuedness,
        nonnegativity, or allowed height interval specified above.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_thickness_averages(
    magnitudes: "np.ndarray", heights: "np.ndarray", thickness: float
) -> "np.ndarray":
    """Stable closed forms of the two independently defined averages."""
    p = _finite_array(magnitudes, float, "magnitudes")
    z = _finite_array(heights, float, "heights")
    d = _finite_scalar(thickness, "thickness")
    if p.ndim != 1 or len(p) < 1 or np.any(p < 0):
        raise ValueError("magnitudes must be a nonempty nonnegative vector")
    if z.ndim != 1 or len(z) < 1 or np.any(np.abs(z) > d / 2):
        raise ValueError("heights must lie inside the slab")
    result = np.ones((len(p), len(z) + 1))
    if d > 0:
        x = p * d
        nonzero = p > 0
        result[nonzero, :-1] = (
            -np.expm1(-p[nonzero, None] * (d / 2 - z))
            - np.expm1(-p[nonzero, None] * (d / 2 + z))
        ) / x[nonzero, None]
        small = nonzero & (x < 1e-3)
        xs = x[small]
        result[small, -1] = (
            1 - xs / 3 + xs**2 / 12 - xs**3 / 60 + xs**4 / 360 - xs**5 / 2520
        )
        large = nonzero & ~small
        result[large, -1] = (
            2 * (np.expm1(-x[large]) + x[large]) / x[large] ** 2
        )
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """
    Ordinary, zero-thickness, small-argument, surface and short-wave cases.
    """
    return [
        {
            "setup": """import numpy as np

p = np.array([0.0, 0.12, 0.8, 2.4])
z = np.array([-1.3, 0.9])
d = 5.5
""",
            "call": "thickness_averages(p.copy(), z.copy(), d)",
            "gold_call": (
                "_oracle_thickness_averages(p.copy(), z.copy(), " "d)"
            ),
        },
        {
            "setup": """import numpy as np

p = np.array([0.0, 0.7, 8.0])
z = np.zeros(2)
d = 0.0
""",
            "call": "thickness_averages(p.copy(), z.copy(), d)",
            "gold_call": (
                "_oracle_thickness_averages(p.copy(), z.copy(), " "d)"
            ),
        },
        {
            "setup": """import numpy as np

p = np.array([1e-12, 1e-8, 1e-5, 0.003])
z = np.array([-0.3, 0.49])
d = 1.0
""",
            "call": "thickness_averages(p.copy(), z.copy(), d)",
            "gold_call": (
                "_oracle_thickness_averages(p.copy(), z.copy(), " "d)"
            ),
        },
        {
            "setup": """import numpy as np

p = np.array([0.01, 2.0, 400.0])
z = np.array([-2.0, 0.0, 2.0])
d = 4.0
""",
            "call": "thickness_averages(p.copy(), z.copy(), d)",
            "gold_call": (
                "_oracle_thickness_averages(p.copy(), z.copy(), " "d)"
            ),
        },
        {
            "setup": """import numpy as np


def rejected(fn):
    try:
        fn(np.array([0.5]), np.array([0.6]), 1.0)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "rejected(thickness_averages)",
            "gold_call": "rejected(_oracle_thickness_averages)",
        },
    ]
