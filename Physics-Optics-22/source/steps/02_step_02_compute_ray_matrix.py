"""
Compute continuous paraxial ray transport through exponentially decreasing focusing.

The ray state is $(r,r')^{\mathsf T}$ with $r'=dr/dz$, and its continuously varying focusing profile satisfies the stated ray equation.



$$r''+g e^{-\alpha z}r=0,\qquad M(0)=I.$$



The matrix maps entrance position and physical axial slope to their values at distance $z$. Its determinant is one; the diagonal entries are dimensionless, while $M_{12}$ has units of metres and $M_{21}$ inverse metres. Zero curvature gives free drift, and zero absorption means a constant longitudinal curvature.

Returns
-------
A real array of shape $(2,2)$ gives the position-slope transfer matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_ray_matrix(g: float, alpha: float, distance: float) -> "np.ndarray":
    r"""Return the transfer matrix for the position-slope state.

    Parameters
    ----------
    g : float
        Entrance curvature $g\geq0$, in $\mathrm{m^{-2}}$.
    alpha : float
        Absorption coefficient $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    distance : float
        Propagation distance $z\geq0$, in $\mathrm{m}$.

    Returns
    -------
    matrix : np.ndarray
        Real shape $(2,2)$ array $M$ mapping entrance $(r,r')$ to exit
        $(r,r')$. Its diagonal entries are dimensionless, $B$ has units
        $\mathrm{m}$, and $C$ has units $\mathrm{m^{-1}}$.

    Raises
    ------
    ValueError
        If an input is negative or nonfinite, or finite transport cannot
        be computed at the supplied numerical scale.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, yv


def _oracle_compute_ray_matrix(g: float, alpha: float, distance: float) -> "np.ndarray":
    g = _finite_scalar(g, "g")
    alpha = _finite_scalar(alpha, "alpha")
    distance = _finite_scalar(distance, "distance")
    if g == 0.0 or distance == 0.0:
        return np.array([[1.0, distance], [0.0, 1.0]])
    frequency = np.sqrt(g)
    if alpha == 0.0:
        phase = frequency * distance
        cosine, sine = np.cos(phase), np.sin(phase)
        return np.array([[cosine, sine / frequency], [-frequency * sine, cosine]])
    if alpha * distance < 1e-7:
        # Integrate the same continuous equation when the basis loses precision.
        def _rhs(z, state):
            matrix = state.reshape(2, 2)
            return (
                np.array([[0.0, 1.0], [-g * np.exp(-alpha * z), 0.0]]) @ matrix
            ).ravel()

        solution = solve_ivp(
            _rhs,
            (0.0, distance),
            np.eye(2).ravel(),
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        if not solution.success:
            raise ValueError("continuous ray integration did not converge")
        return solution.y[:, -1].reshape(2, 2)
    entrance = 2.0 * frequency / alpha
    attenuation = np.exp(-alpha * distance / 2.0)
    exit_value = entrance * attenuation
    initial = np.array(
        [
            [jv(0, entrance), yv(0, entrance)],
            [frequency * jv(1, entrance), frequency * yv(1, entrance)],
        ]
    )
    final = np.array(
        [
            [jv(0, exit_value), yv(0, exit_value)],
            [
                frequency * attenuation * jv(1, exit_value),
                frequency * attenuation * yv(1, exit_value),
            ],
        ]
    )
    matrix = np.linalg.solve(initial.T, final.T).T
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite continuous ray transfer")
    return matrix

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    return [
        {
            "setup": """import numpy as np
""",
            "call": "compute_ray_matrix(65000.0, 180.0, .01)",
            "gold_call": "_oracle_compute_ray_matrix(65000.0, 180.0, .01)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
""",
            "call": "compute_ray_matrix(0.0, 180.0, .02)",
            "gold_call": "_oracle_compute_ray_matrix(0.0, 180.0, .02)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
""",
            "call": "compute_ray_matrix(40000.0, 0.0, .01)",
            "gold_call": "_oracle_compute_ray_matrix(40000.0, 0.0, .01)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
""",
            "call": "compute_ray_matrix(40000.0, 1e-7, .01)",
            "gold_call": "_oracle_compute_ray_matrix(40000.0, 1e-7, .01)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
""",
            "call": "compute_ray_matrix(40000.0, 180.0, 0.0)",
            "gold_call": "_oracle_compute_ray_matrix(40000.0, 180.0, 0.0)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np

def _raises(function):
    try:
        function(-1.0, 180.0, .01)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_raises(compute_ray_matrix)",
            "gold_call": "_raises(_oracle_compute_ray_matrix)",
        },
    ]
