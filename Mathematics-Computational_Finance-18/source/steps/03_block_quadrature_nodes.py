"""
Build the quadrature nodes and weights of Eq (25) for the frequency axis. Use the partition of the positive axis that the source prescribes, with the quadrature rule it assigns to each block, at n_quad points per block; the integer m sets the prescribed breakpoint. Return the blocks in increasing order of frequency, with the nodes of each block in ascending order. Weights are returned RELATIVE TO THE REPRESENTING DENSITY, so that multiplying them elementwise by that density at these nodes gives the exponential-sum weights. Raise ValueError if alpha is outside (0, 1/2), if m is not a nonnegative integer, or if n_quad is not a positive integer.

Approximating the mixing density by a finite positive quadrature turns the Volterra dynamics into a finite system of Ornstein-Uhlenbeck factors. How the frequency axis is partitioned, and which rule is used on each block, is fixed by the source, and each choice is forced by the local behaviour of the integrand. On the unbounded block use the standard Gauss-Laguerre rule shifted to start at 1, with that rule's own unit-rate weight function: nodes 1 + y_j and density-relative weights W_j exp(y_j), where (y_j, W_j) are the Gauss-Laguerre abscissae and weights. This fixes a choice the source leaves open.

Returns
-------
tuple (x, q) of two 1-D float64 ndarrays, each of length 3*n_quad.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def block_quadrature_nodes(alpha: float, m: int, n_quad: int) -> tuple:
    """Build the quadrature nodes and density-relative weights of Eq (25) for the frequency
    axis.

    Parameters
    ----------
    alpha : float
        Roughness exponent of the fractional kernel, 0 < alpha < 1/2.
    m : int
        Nonnegative integer that sets the block breakpoint.
    n_quad : int
        Positive number of quadrature points per block.

    Returns
    -------
    result : tuple
        tuple (x, q) of two 1-D float64 ndarrays, each of length 3*n_quad.

    Raises
    ------
    ValueError
        If alpha is outside (0, 1/2).
        If m is not a nonnegative integer.
        If n_quad is not a positive integer.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gamma, roots_jacobi, roots_legendre, roots_laguerre


def _oracle_block_quadrature_nodes(alpha: float, m: int, n_quad: int) -> tuple:
    """Eq (25): nodes and DENSITY-RELATIVE weights for the prescribed 3-block split.

    Blocks are [0, 2^-m], [2^-m, 1] and [1, inf), carrying Gauss-Jacobi, Gauss-Legendre
    and Gauss-Laguerre respectively, n_quad points each.

    Weights q are returned relative to the representing density: the caller forms
    omega = q * w(x). On the first block the Jacobi rule integrates the x^{a-1} endpoint
    singularity exactly, so its raw weight is divided back out by x^{1-a} to leave a
    density-relative weight.
    """
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must satisfy 0 < alpha < 1/2")
    if int(m) != m or m < 0:
        raise ValueError("m must be a nonnegative integer")
    if int(n_quad) != n_quad or n_quad < 1:
        raise ValueError("n_quad must be a positive integer")
    m, n_quad = int(m), int(n_quad)
    c = 2.0 ** (-m)

    # [0, c]: x = c(1+u)/2 turns int_0^c f(x) x^{a-1} dx into
    # (c/2)^a int_{-1}^1 f(.) (1+u)^{a-1} du, which is Gauss-Jacobi with beta = a-1.
    u, wu = roots_jacobi(n_quad, 0.0, alpha - 1.0)
    x1 = c * (1.0 + u) / 2.0
    q1 = wu * (c / 2.0) ** alpha * x1 ** (1.0 - alpha)

    # [c, 1]: smooth -> Gauss-Legendre
    v, wv = roots_legendre(n_quad)
    x2 = c + (1.0 - c) * (v + 1.0) / 2.0
    q2 = wv * (1.0 - c) / 2.0

    # [1, inf): the density decays exponentially -> Gauss-Laguerre on x = 1 + y
    y, wy = roots_laguerre(n_quad)
    x3 = 1.0 + y
    q3 = wy * np.exp(y)

    return np.concatenate([x1, x2, x3]), np.concatenate([q1, q2, q3])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nalpha, m, n_quad = 0.35, 3, 24",
            "call": "np.concatenate(block_quadrature_nodes(alpha, m, n_quad))",
            "gold_call": "np.concatenate(_oracle_block_quadrature_nodes(alpha, m, n_quad))",
        },
        {
            "setup": "import numpy as np\nalpha, m, n_quad = 0.35, 0, 4",
            "call": "block_quadrature_nodes(alpha, m, n_quad)[0]",
            "gold_call": "_oracle_block_quadrature_nodes(alpha, m, n_quad)[0]",
        },
        {
            "setup": "import numpy as np\nalpha, m, n_quad = 0.10, 5, 8",
            "call": "block_quadrature_nodes(alpha, m, n_quad)[1]",
            "gold_call": "_oracle_block_quadrature_nodes(alpha, m, n_quad)[1]",
        },
        {
            "setup": "import numpy as np\n# invalid input: n_quad = 0 must raise ValueError\ndef run_model():\n    try:\n        block_quadrature_nodes(0.35, 3, 0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_block_quadrature_nodes(0.35, 3, 0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
