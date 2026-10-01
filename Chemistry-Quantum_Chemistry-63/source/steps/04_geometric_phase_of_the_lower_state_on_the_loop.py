"""
Integrate the Berry connection of the lower state anticlockwise around the circle of radius Q on an equally spaced grid and return the geometric phase.

The geometric phase is the line integral of the Berry connection of the lower state around the circular contour of radius Q, taken anticlockwise. In the orthonormal polar frame the line element contributes Q d theta against the angular component of the connection, so the integrand is Q times that component. The integrand is a smooth periodic function of theta, so the equally spaced trapezoidal rule converges geometrically and a few hundred points already saturate double precision; the grid is theta_k = 2 pi k / n_theta for k = 0 .. n_theta - 1. Without the gap term this integral would take the quantised value pi at every radius, which is the topological result; the gap term makes it depend on the radius instead.

Returns
-------
float: the geometric phase of the lower adiabatic state around the circular loop of radius Q, in radians
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bo_geometric_phase(Q: float, g: float, K: float, Delta: float,
                       n_theta: int) -> float:
    '''Geometric phase of the lower adiabatic state on the loop of radius Q.

    Parameters
    ----------
    Q : float
        Radius of the circular nuclear contour, strictly positive.
    g : float
        Linear vibronic coupling strength, strictly positive.
    K : float
        Harmonic force constant of the two normal modes.
    Delta : float
        Strength of the constant term that opens the gap, strictly positive.
    n_theta : int
        Number of equally spaced quadrature points on the loop, at least 8.
        The grid is theta_k = 2 pi k / n_theta for k = 0 .. n_theta - 1.

    Returns
    -------
    phase : float
        The geometric phase, in radians.

    Raises
    ------
    ValueError
        If any of Q, g, K or Delta is not a finite scalar, if Q, g or Delta is
        not strictly positive, or if n_theta is not an integer of at least 8.
    '''
    return phase

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_point(Q, theta, g, K, Delta):
    vals = [Q, theta, g, K, Delta]

    if any(np.ndim(v) != 0 for v in vals):
        raise ValueError("Q, theta, g, K and Delta must all be scalars")

    Q, theta, g, K, Delta = (float(v) for v in vals)

    if not np.all(np.isfinite([Q, theta, g, K, Delta])):
        raise ValueError("every model parameter must be finite")

    if Q <= 0.0:
        raise ValueError(
            "Q must be strictly positive: the loop may not pass through the origin"
        )

    if g <= 0.0:
        raise ValueError("g must be strictly positive")

    if Delta <= 0.0:
        raise ValueError(
            "Delta must be strictly positive: at Delta = 0 the lower state is "
            "antiperiodic and the phase convention below cannot be imposed"
        )

    return Q, theta, g, K, Delta


def _grad_alpha(Q, theta, g, Delta):
    """(d alpha / dQ, (1/Q) d alpha / d theta) of the azimuth of the field direction."""
    S2 = g * g * Q * Q * np.sin(theta) ** 2 + Delta * Delta

    return np.array([
        Delta * g * np.sin(theta) / S2,
        Delta * g * np.cos(theta) / S2,
    ])


def _oracle_bo_geometric_phase(
    Q: float,
    g: float,
    K: float,
    Delta: float,
    n_theta: int,
) -> float:
    _check_point(Q, 0.0, g, K, Delta)

    if (
        np.ndim(n_theta) != 0
        or int(n_theta) != n_theta
        or int(n_theta) < 8
    ):
        raise ValueError("n_theta must be an integer of at least 8")

    n = int(n_theta)
    grid = 2.0 * np.pi * np.arange(n) / n
    total = 0.0

    for th in grid:
        total += _oracle_berry_connections(
            Q,
            th,
            g,
            K,
            Delta,
        )[0, 1]

    return float(total * Q * 2.0 * np.pi / n)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\n",
         "call": "round(bo_geometric_phase(0.20, 0.60, 1.00, 0.12, 512), 10)",
         "gold_call": "round(_oracle_bo_geometric_phase(0.20, 0.60, 1.00, 0.12, 512), 10)"},
        {"setup": "import numpy as np\n",
         "call": "round(bo_geometric_phase(0.20, 0.60, 1.00, 0.12, 2048), 10)",
         "gold_call": "round(_oracle_bo_geometric_phase(0.20, 0.60, 1.00, 0.12, 2048), 10)"},
        {"setup": "import numpy as np\n",
         "call": "round(bo_geometric_phase(0.02, 0.60, 1.00, 0.12, 1024), 10)",
         "gold_call": "round(_oracle_bo_geometric_phase(0.02, 0.60, 1.00, 0.12, 1024), 10)"},
        {"setup": "import numpy as np\n",
         "call": "round(bo_geometric_phase(4.00, 0.60, 1.00, 0.12, 1024), 10)",
         "gold_call": "round(_oracle_bo_geometric_phase(4.00, 0.60, 1.00, 0.12, 1024), 10)"},
        {"setup": "import numpy as np\n",
         "call": "round(bo_geometric_phase(0.75, 0.30, 0.50, 0.45, 256), 10)",
         "gold_call": "round(_oracle_bo_geometric_phase(0.75, 0.30, 0.50, 0.45, 256), 10)"},
    ]
