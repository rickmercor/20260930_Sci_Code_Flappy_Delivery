"""
Compute a directional derivative of the continuous optical transfer matrix.

The directional inputs change both thermal controls at fixed propagation distance.



$$g(\varepsilon)=g+\varepsilon\dot g,\qquad \alpha(\varepsilon)=\alpha+\varepsilon\dot\alpha.$$



The result is the derivative of the position-slope transfer with respect to $\varepsilon$ at zero. An absorption perturbation changes the axial profile as well as its entrance value. The transfer at zero distance is the identity for every direction, so its directional response is zero.

Returns
-------
A real array of shape $(2,2)$ contains the directional derivative of the ray transfer.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_ray_response(
    g: float,
    alpha: float,
    length: float,
    g_direction: float,
    alpha_direction: float,
    order: int = 64,
) -> "np.ndarray":
    r"""Return a transfer derivative along the supplied parameter direction.

    Parameters
    ----------
    g : float
        Finite curvature $g\geq0$, in $\mathrm{m^{-2}}$.
    alpha : float
        Finite absorption $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    length : float
        Finite propagation length $L\geq0$, in $\mathrm{m}$.
    g_direction : float
        Finite $\dot g=dg/d\varepsilon$.
    alpha_direction : float
        Finite $\dot\alpha=d\alpha/d\varepsilon$.
    order : int, optional
        Quadrature order, an integer at least $16$; default $64$.

    Returns
    -------
    response : np.ndarray
        Shape $(2,2)$ real array $\dot M(L)$. Units are those of each
        transfer entry divided by the units of $\varepsilon$.

    Raises
    ------
    ValueError
        If a ray input violates its nonnegative finite contract, either
        direction is nonfinite, or the quadrature order is invalid.

    Notes
    -----
    Boundary derivatives are continuous one-sided extensions where needed.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss


def _quadrature_order(order: int) -> int:
    if (
        isinstance(order, (bool, np.bool_))
        or not isinstance(order, (int, np.integer))
        or order < 16
    ):
        raise ValueError("order must be an integer at least 16")
    return int(order)


def _oracle_compute_ray_response(
    g: float,
    alpha: float,
    length: float,
    g_direction: float,
    alpha_direction: float,
    order: int = 64,
) -> "np.ndarray":
    g = _finite_scalar(g, "g")
    alpha = _finite_scalar(alpha, "alpha")
    length = _finite_scalar(length, "length")
    if not np.isfinite([g_direction, alpha_direction]).all():
        raise ValueError("parameter directions must be finite")
    nodes, weights = leggauss(_quadrature_order(order))
    response_integral = np.zeros((2, 2))
    for node, weight in zip(nodes, weights):
        z = length * (node + 1.0) / 2.0
        matrix = _oracle_compute_ray_matrix(g, alpha, z)
        perturbation = np.zeros((2, 2))
        perturbation[1, 0] = -np.exp(-alpha * z) * (
            g_direction - g * z * alpha_direction
        )
        response_integral += weight * np.linalg.solve(matrix, perturbation @ matrix)
    final = _oracle_compute_ray_matrix(g, alpha, length)
    return final @ response_integral * (length / 2.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    return [
        {
            "setup": """import numpy as np
""",
            "call": "compute_ray_response(65000.0, 180.0, .01, 230.0, 1.0)",
            "gold_call": "_oracle_compute_ray_response(65000.0, 180.0, .01, 230.0, 1.0)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
""",
            "call": "compute_ray_response(0.0, 0.0, .02, 1.0, 0.0)",
            "gold_call": "_oracle_compute_ray_response(0.0, 0.0, .02, 1.0, 0.0)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
""",
            "call": "compute_ray_response(65000.0, 180.0, .01, 0.0, 0.0)",
            "gold_call": "_oracle_compute_ray_response(65000.0, 180.0, .01, 0.0, 0.0)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
""",
            "call": "compute_ray_response(40000.0, 0.0, 0.0, 1.0, 1.0)",
            "gold_call": "_oracle_compute_ray_response(40000.0, 0.0, 0.0, 1.0, 1.0)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np

def _raises(function):
    try:
        function(40000.0, 180.0, .01, 1.0, 0.0, order=4)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_raises(compute_ray_response)",
            "gold_call": "_raises(_oracle_compute_ray_response)",
        },
    ]
