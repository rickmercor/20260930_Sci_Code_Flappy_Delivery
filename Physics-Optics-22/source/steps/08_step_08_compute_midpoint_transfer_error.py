"""
Find the leading transfer error of a midpoint-sliced thermal lens.

The continuously varying ray system has an entrance curvature $g$ and an axially decaying curvature $g e^{-\alpha z}$. A finite stack of constant curvature slices approximates its full position-slope transfer. For $N$ equal slices, each curvature is sampled at its axial midpoint and later slices multiply on the left. The returned matrix is the converged leading coefficient of the difference between this stack and the continuous transfer:



$$E_M=\lim_{N\to\infty}N^2[M_N(L)-M(L)].$$



The limit is zero for a uniform or absent lens. It is tangent to the determinant-one transfer constraint.

Returns
-------
A real shape $(2,2)$ array is the $N^{-2}$ transfer-error coefficient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_midpoint_transfer_error(
    g: float, alpha: float, length: float, order: int = 96
) -> "np.ndarray":
    r"""Return the leading midpoint-stack transfer error.

    Parameters
    ----------
    g : float
        Finite entrance ray curvature $g\geq0$, in $\mathrm{m^{-2}}$.
    alpha : float
        Finite absorption coefficient $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    length : float
        Finite propagation length $L\geq0$, in $\mathrm{m}$.
    order : int, optional
        Integer quadrature order at least $16$, default $96$.

    Returns
    -------
    error : np.ndarray
        Finite real shape $(2,2)$ coefficient $E_M$, with position-slope
        transfer entry units $(1,\mathrm m;\mathrm{m^{-1}},1)$.

    Raises
    ------
    ValueError
        If an input is nonfinite or negative, order is invalid, or the
        evaluated matrix coefficient is nonfinite.

    Notes
    -----
    No input is modified and no random state is used.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss


def _oracle_compute_midpoint_transfer_error(
    g: float, alpha: float, length: float, order: int = 96
) -> "np.ndarray":
    g = _finite_scalar(g, "g")
    alpha = _finite_scalar(alpha, "alpha")
    length = _finite_scalar(length, "length")
    nodes, weights = leggauss(_quadrature_order(order))
    if g == 0.0 or alpha == 0.0 or length == 0.0:
        return np.zeros((2, 2))

    full = _oracle_compute_ray_matrix(g, alpha, length)
    integral = np.zeros((2, 2))
    for node, weight in zip(nodes, weights):
        z = length * (node + 1.0) / 2.0
        curvature = g * np.exp(-alpha * z)
        generator = np.array(
            [
                [alpha * curvature / 12.0, 0.0],
                [alpha**2 * curvature / 24.0, -alpha * curvature / 12.0],
            ]
        )
        entrance_to_z = _oracle_compute_ray_matrix(g, alpha, z)
        z_to_exit = np.linalg.solve(entrance_to_z.T, full.T).T
        integral += weight * (z_to_exit @ generator @ entrance_to_z)
    error = 0.5 * length**3 * integral
    if not np.all(np.isfinite(error)):
        raise ValueError("midpoint transfer error is nonfinite")
    return error

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return varying, uniform, absent, zero-length, and invalid cases."""
    return [
        {
            "setup": "import numpy as np",
            "call": "compute_midpoint_transfer_error(65000.0, 180.0, .01)",
            "gold_call": "_oracle_compute_midpoint_transfer_error(65000.0, 180.0, .01)",
            "tol": 1e-8,
        },
        {
            "setup": "import numpy as np",
            "call": "compute_midpoint_transfer_error(80000.0, 250.0, .008)",
            "gold_call": "_oracle_compute_midpoint_transfer_error(80000.0, 250.0, .008)",
            "tol": 1e-8,
        },
        {
            "setup": "import numpy as np",
            "call": "compute_midpoint_transfer_error(65000.0, 0.0, .01)",
            "gold_call": "_oracle_compute_midpoint_transfer_error(65000.0, 0.0, .01)",
        },
        {
            "setup": "import numpy as np",
            "call": "compute_midpoint_transfer_error(0.0, 180.0, .01)",
            "gold_call": "_oracle_compute_midpoint_transfer_error(0.0, 180.0, .01)",
        },
        {
            "setup": "import numpy as np",
            "call": "compute_midpoint_transfer_error(65000.0, 180.0, 0.0)",
            "gold_call": "_oracle_compute_midpoint_transfer_error(65000.0, 180.0, 0.0)",
        },
        {
            "setup": """import numpy as np
def _raises(function):
    try:
        function(65000.0, 180.0, .01, 8)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_raises(compute_midpoint_transfer_error)",
            "gold_call": "_raises(_oracle_compute_midpoint_transfer_error)",
        },
    ]
