"""
Return the integrated carrier population from the complete discrete transport solve.

Solve for the continuous quadratic field and integrate it over the physical rectangle.

The exact local integral for a tensor-product quadratic nodal element is

$$

\int_Eu_h\,dA

=*\frac*{h_xh_y}{36}*\sum_*{i,j=0}^2w_iw_j u_{ij},

\qquad (w_0,w_1,w_2)=(1,4,1).

$$

Sum element integrals once, including shared nodes with their element weights.

This final step returns the dimensionless total population, not a sum of nodal values or a control-volume lumped approximation.

Returns
-------
A finite float equal to the integrated dimensionless carrier population.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_carrier_population(
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    diffusion: float,
    drift: np.ndarray,
    source: np.ndarray,
    boundary: np.ndarray,
    partition: float = 0.27,
    trace_fraction: float = 0.18,
) -> float:
    r"""Return the integrated carrier population from the complete discrete transport solve.

    Parameters
    ----------
    x_knots, y_knots : np.ndarray
        Strictly increasing finite primary coordinate vectors, each of length at least two.
    diffusion : float
        Positive dimensionless diffusion coefficient.
    drift : np.ndarray
        Shape $(2,)$; finite constant horizontal and vertical drift.
    source : np.ndarray
        Shape $(4,)$; coefficients $[f_0,f_x,f_y,f_{xy}]$.
    boundary : np.ndarray
        Shape $(3,)$; Dirichlet coefficients $[g_0,g_x,g_y]$.
    partition : float, optional
        Dual cut fraction $p$ in $(0,1/2)$.
    trace_fraction : float, optional
        Positive half-width fraction $r<p$; all face Peclet magnitudes must be at most 1000.

    Returns
    -------
    float
        Dimensionless integrated carrier population over the full rectangular domain.

    Raises
    ------
    ValueError
        If array shapes or finite-value requirements fail, knots are not increasing,
        diffusion is non-positive, $0<r<p<1/2$ fails, a face Peclet magnitude
        exceeds 1000, or the discrete system is singular or has no finite solution,
        or the integrated population is not finite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_carrier_population(
    x_knots: np.ndarray,
    y_knots: np.ndarray,
    diffusion: float,
    drift: np.ndarray,
    source: np.ndarray,
    boundary: np.ndarray,
    partition: float = 0.27,
    trace_fraction: float = 0.18,
) -> float:
    """Evaluate the reference numerical operation."""
    field = _oracle_solve_carrier_balance(
        x_knots, y_knots, diffusion, drift, source, boundary, partition, trace_fraction
    )
    weights = np.outer([1.0, 4.0, 1.0], [1.0, 4.0, 1.0]) / 36
    total = 0.0
    for j, hy in enumerate(np.diff(y_knots)):
        for i, hx in enumerate(np.diff(x_knots)):
            local = field[2 * j : 2 * j + 3, 2 * i : 2 * i + 3]
            total += hx * hy * np.sum(weights * local)
    if not np.isfinite(total):
        raise ValueError("integrated population must be finite")
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nx = np.array([0., .22, .57, 1.])\ny = np.array([0., .31, .64, 1.])\nalpha = .03\nbeta = np.array([1.7, -.9])\nf = np.array([1.,1.,2.,3.])\ng = np.array([1.,.2,.3])\np, r = .27, .18\n",
            "call": "compute_carrier_population(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
            "gold_call": "_oracle_compute_carrier_population(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nx = np.array([0., .4, 1.]); y = np.array([0., .6, 1.])\nalpha = .1; beta = np.zeros(2)\nf = np.array([2., 0., 0., 0.]); g = np.zeros(3)\np, r = 1/3, .2\n",
            "call": "compute_carrier_population(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
            "gold_call": "_oracle_compute_carrier_population(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nx = np.array([0., .15, .7, 1.]); y = np.array([0., .2, .8, 1.])\nalpha = .025; beta = np.array([-.8,1.4])\nf = np.array([.5,2.,-.5,1.]); g = np.array([.8,.1,.2])\np, r = .31, .22\n",
            "call": "compute_carrier_population(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
            "gold_call": "_oracle_compute_carrier_population(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nx = np.array([0., .22, .57, 1.])\ny = np.array([0., .31, .64, 1.])\nalpha = .03\nbeta = np.array([1.7, -.9])\nf = np.array([1.,1.,2.,3.])\ng = np.array([1.,.2,.3])\np, r = .27, .18\nalpha = 0.\ndef _exception_code(function):\n    try:\n        function(x.copy(), y.copy(), alpha, beta.copy(), f.copy(), g.copy(), p, r)\n    except ValueError:\n        return 1\n    return 0\n\n",
            "call": "_exception_code(compute_carrier_population)",
            "gold_call": "_exception_code(_oracle_compute_carrier_population)",
        },
    ]
