"""
Return the midpoint-rule growth matrix on a mesh of bin midpoints: G[i, j] is the normal probability density of the next-census length z_i of a female whose current length is z_j, with mean mu0 + mu1 * z_j and standard deviation sd (metres), multiplied by the bin width, so that G @ f approximates the integral of the growth kernel times f(z) over z on the mesh. The density is evaluated as is: the columns are not renormalised over the finite mesh, so probability mass that falls outside the mesh is lost.

Growth in an integral projection model is a transition density from the current to the next size; a normal density around a linear (von Bertalanffy-type) mean growth is standard, and discretised by the midpoint rule it becomes the matrix that propagates the survivors to the next census.

Returns
-------
numpy.ndarray of float64 with shape (n, n): the midpoint-rule growth matrix, bin width included.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def growth_kernel(midpoints: "numpy.ndarray", width: float, mu0: float = 0.194,
                  mu1: float = 0.841, sd: float = 0.1) -> "numpy.ndarray":
    """Return the midpoint-rule growth matrix on a mesh of bin midpoints: G[i, j] is the normal probability density of the next-census length z_i of a female whose current length is z_j, with mean mu0 + mu1 * z_j and standard deviation sd (metres), multiplied by the bin width, so that G @ f approximates the integral of the growth kernel times f(z) over z on the mesh. The density is evaluated as is: the columns are not renormalised over the finite mesh, so probability mass that falls outside the mesh is lost.

    Parameters
    ----------
    midpoints : numpy.ndarray
        One-dimensional array of finite bin midpoints in metres (row 0 of step 04).
    width : float
        Positive bin width in metres (the common value in row 1 of step 04).
    mu0 : float
        Intercept of the mean next-census length in metres; default 0.194.
    mu1 : float
        Slope of the mean next-census length per metre of current length; default 0.841.
    sd : float
        Positive standard deviation of the next-census length in metres; default 0.1.

    Returns
    -------
    growth : numpy.ndarray
        Array of shape (n, n) with G[i, j] the density at z_i given z_j times the bin width (float64).

    Raises
    ------
    ValueError
        If midpoints is not a non-empty one-dimensional finite array, width or sd is not positive and finite, or mu0 or mu1 is not finite.
    """
    return growth

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_growth_kernel(midpoints: "numpy.ndarray", width: float, mu0: float = 0.194,
                          mu1: float = 0.841, sd: float = 0.1) -> "numpy.ndarray":
    """Midpoint-rule growth kernel G[i, j] = N(z_i; mu0 + mu1 z_j, sd) * width, no renormalisation."""
    z = np.asarray(midpoints, dtype=np.float64)
    if z.ndim != 1 or z.size < 1:
        raise ValueError("midpoints must be a non-empty 1-D array")
    if not np.all(np.isfinite(z)):
        raise ValueError("midpoints must be finite")
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("width must be positive and finite")
    if not (np.isfinite(mu0) and np.isfinite(mu1) and np.isfinite(sd)) or sd <= 0.0:
        raise ValueError("mu0 and mu1 must be finite, sd positive and finite")
    mean = float(mu0) + float(mu1) * z[None, :]                 # column j: source length z_j
    pdf = np.exp(-0.5 * ((z[:, None] - mean) / float(sd)) ** 2) / (float(sd) * np.sqrt(2.0 * np.pi))
    # the density times the bin width IS the midpoint rule for int G(z', z) f(z) dz; mass that the
    # normal places outside the domain is lost (no per-column renormalisation)
    return pdf * float(width)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nmesh = _oracle_midpoint_mesh(0.01, 1.5, 300)\nz, w = mesh[0], mesh[1]\nwidth = float(w[0])\n",
            "call": "growth_kernel(z, width)",
            "gold_call": "_oracle_growth_kernel(z, width)",
        },
        {
            "setup": "import numpy as np\nmesh = _oracle_midpoint_mesh(0.01, 1.5, 40)\nz, w = mesh[0], mesh[1]\nwidth = float(w[0])\n",
            "call": "growth_kernel(z, width)",
            "gold_call": "_oracle_growth_kernel(z, width)",
        },
        {
            "setup": "import numpy as np\nmesh = _oracle_midpoint_mesh(0.05, 1.2, 100)\nz, w = mesh[0], mesh[1]\nwidth = float(w[0])\nmu0, mu1, sd = 0.10, 0.90, 0.05\n",
            "call": "growth_kernel(z, width, mu0, mu1, sd)",
            "gold_call": "_oracle_growth_kernel(z, width, mu0, mu1, sd)",
        },
        {
            "setup": "import numpy as np\ndef _fx_midpoint_mesh(z_min, z_max, n_bins):\n    n = int(n_bins)\n    edges = np.linspace(float(z_min), float(z_max), n + 1)\n    mids = 0.5 * (edges[:-1] + edges[1:])\n    widths = np.full(n, (float(z_max) - float(z_min)) / n)\n    return np.vstack([mids, widths])\nmesh = _fx_midpoint_mesh(0.01, 1.5, 300)\nz, w = mesh[0], mesh[1]\nwidth = -0.1\ndef run_model():\n    try:\n        growth_kernel(z, width)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_growth_kernel(z, width)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
