"""
Implement build_powder_grid which returns the weighted crystallite orientation set used for powder averaging.

Orientational averages over a polycrystalline sample are integrals of an orientation-dependent
observable over the three Euler angles that carry an interaction tensor from its principal axis
frame into the rotor frame. The measure is sin(beta) d(beta) d(alpha) d(gamma), so the polar
angle is integrated in the variable cos(beta) on [-1, 1] while the two azimuthal angles are
integrated on [0, 2*pi).

This task uses a product quadrature: Gauss-Legendre nodes and weights in cos(beta), and equally
spaced nodes with equal weights in alpha and in gamma. The resulting crystallite weights are
normalized so that they sum to one, which makes a weighted sum over the returned rows a
normalized powder average.

Returns
-------
np.ndarray, real array of shape (n_beta*n_alpha*n_gamma, 4) whose columns are the Euler angles alpha, beta and gamma in radians and the normalized crystallite weight
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_powder_grid(n_beta: int, n_alpha: int, n_gamma: int) -> "np.ndarray":
    '''Build the weighted crystallite orientation set for powder averaging.

    The rows are ordered with the polar index varying slowest, then the alpha
    index, then the gamma index varying fastest.

    Parameters
    ----------
    n_beta : int
        Number of Gauss-Legendre nodes in cos(beta). Must be >= 1.
    n_alpha : int
        Number of equally spaced alpha nodes, alpha_i = 2*pi*i/n_alpha for
        i = 0, ..., n_alpha - 1. Must be >= 1.
    n_gamma : int
        Number of equally spaced gamma nodes, gamma_l = 2*pi*l/n_gamma for
        l = 0, ..., n_gamma - 1. Must be >= 1.

    Returns
    -------
    grid : np.ndarray
        Real array of shape (n_beta*n_alpha*n_gamma, 4) whose columns are
        alpha, beta and gamma in radians and the normalized crystallite
        weight. beta is taken as arccos of the Gauss-Legendre node, so it
        lies in (0, pi). The weights sum to 1.

    Raises
    ------
    ValueError
        If any of n_beta, n_alpha or n_gamma is not a positive integer.
    '''
    return grid  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_powder_grid(n_beta: int, n_alpha: int, n_gamma: int) -> "np.ndarray":
    for name, value in (("n_beta", n_beta), ("n_alpha", n_alpha), ("n_gamma", n_gamma)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if int(value) < 1:
            raise ValueError(name + " must be >= 1")

    n_beta = int(n_beta)
    n_alpha = int(n_alpha)
    n_gamma = int(n_gamma)

    nodes, weights = np.polynomial.legendre.leggauss(n_beta)
    beta = np.arccos(nodes)
    # The Gauss-Legendre weights on [-1, 1] sum to 2; halving normalizes the
    # polar integral, and the azimuthal grids contribute equal weights.
    beta_weight = weights / 2.0
    alpha = 2.0 * np.pi * np.arange(n_alpha) / n_alpha
    gamma = 2.0 * np.pi * np.arange(n_gamma) / n_gamma

    grid = np.empty((n_beta * n_alpha * n_gamma, 4), dtype=float)
    row = 0
    for k in range(n_beta):
        for i in range(n_alpha):
            for l in range(n_gamma):
                grid[row] = (alpha[i], beta[k], gamma[l],
                             beta_weight[k] / (n_alpha * n_gamma))
                row += 1
    return grid

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    common = "import numpy as np\n"
    return [
        # Normal: the graded 6 x 6 x 3 orientation set, checked entry by entry
        # on a representative slice plus the weight normalization.
        {
            "setup": common,
            "call": "[round(float(v), 10) for v in np.concatenate([build_powder_grid(6, 6, 3)[::17].ravel(), [build_powder_grid(6, 6, 3)[:, 3].sum()]])]",
            "gold_call": "[round(float(v), 10) for v in np.concatenate([_oracle_build_powder_grid(6, 6, 3)[::17].ravel(), [_oracle_build_powder_grid(6, 6, 3)[:, 3].sum()]])]",
        },
        # Boundary: the smallest possible set. A single Gauss-Legendre node sits
        # at cos(beta) = 0, i.e. beta = pi/2, and carries the full weight.
        {
            "setup": common,
            "call": "[round(float(v), 12) for v in build_powder_grid(1, 1, 1).ravel()]",
            "gold_call": "[round(float(v), 12) for v in _oracle_build_powder_grid(1, 1, 1).ravel()]",
        },
        # Edge: the quadrature must integrate the second Legendre polynomial to
        # zero, which is what makes a second-rank interaction average correctly.
        {
            "setup": common,
            "call": "(lambda g: round(float(np.sum(g[:, 3] * (3.0 * np.cos(g[:, 1]) ** 2 - 1.0) / 2.0)), 12))(build_powder_grid(4, 3, 2))",
            "gold_call": "(lambda g: round(float(np.sum(g[:, 3] * (3.0 * np.cos(g[:, 1]) ** 2 - 1.0) / 2.0)), 12))(_oracle_build_powder_grid(4, 3, 2))",
        },
        # Edge: fourth-rank exactness of a 4-node Gauss-Legendre rule; the mean
        # of the fourth Legendre polynomial also vanishes.
        {
            "setup": common,
            "call": "(lambda g: round(float(np.sum(g[:, 3] * (35.0 * np.cos(g[:, 1]) ** 4 - 30.0 * np.cos(g[:, 1]) ** 2 + 3.0) / 8.0)), 12))(build_powder_grid(4, 2, 2))",
            "gold_call": "(lambda g: round(float(np.sum(g[:, 3] * (35.0 * np.cos(g[:, 1]) ** 4 - 30.0 * np.cos(g[:, 1]) ** 2 + 3.0) / 8.0)), 12))(_oracle_build_powder_grid(4, 2, 2))",
        },
        # Normal: complete small grid, which pins the row ordering and the
        # azimuthal node placement.
        {
            "setup": common,
            "call": "[round(float(v), 12) for v in build_powder_grid(2, 3, 2).ravel()]",
            "gold_call": "[round(float(v), 12) for v in _oracle_build_powder_grid(2, 3, 2).ravel()]",
        },
        # Boundary: a single azimuthal node in each of alpha and gamma leaves the
        # polar quadrature alone and must place both azimuths at zero.
        {
            "setup": common,
            "call": "[round(float(v), 12) for v in build_powder_grid(5, 1, 1).ravel()]",
            "gold_call": "[round(float(v), 12) for v in _oracle_build_powder_grid(5, 1, 1).ravel()]",
        },
        # Edge: a larger set, summarized by its shape, angular extrema and the
        # weight of the first and last rows.
        {
            "setup": common,
            "call": "(lambda g: [float(g.shape[0]), float(g.shape[1]), round(float(g[:, 1].min()), 10), round(float(g[:, 1].max()), 10), round(float(g[:, 0].max()), 10), round(float(g[:, 2].max()), 10), round(float(g[0, 3]), 12), round(float(g[-1, 3]), 12)])(build_powder_grid(10, 4, 5))",
            "gold_call": "(lambda g: [float(g.shape[0]), float(g.shape[1]), round(float(g[:, 1].min()), 10), round(float(g[:, 1].max()), 10), round(float(g[:, 0].max()), 10), round(float(g[:, 2].max()), 10), round(float(g[0, 3]), 12), round(float(g[-1, 3]), 12)])(_oracle_build_powder_grid(10, 4, 5))",
        },
        # Invalid input must raise ValueError rather than return a grid.
        {
            "setup": "\n\nimport numpy as np\ndef _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "_exception_code(build_powder_grid, 0, 4, 2)",
            "gold_call": "_exception_code(_oracle_build_powder_grid, 0, 4, 2)",
        },
    ]
