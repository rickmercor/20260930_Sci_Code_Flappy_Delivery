"""
Geometric weighted densities of the confined fluid on the pore grid.

Fundamental measure theory replaces the local density by a small family of smeared densities, each one the convolution of the profile with a weight function carrying a different geometric measure of the sphere. Four of them are scalars and two are vectors that vanish in a uniform fluid and become large wherever the profile turns over, which is exactly what happens against a wall. In planar geometry the three dimensional convolutions collapse onto one dimensional integrals along the pore axis. Represent each weight on the grid by integrating it exactly across the cell around every offset, so that the weights of a uniform fluid sum to the geometric measures themselves, and evaluate the convolutions as discrete sums over those cell integrals, returning one value per grid point whatever the length of the grid relative to the sphere. Return the six weighted densities stacked in the order n0, n1, n2, n3, nV1, nV2, in reciprocal angstrom powers that follow from their geometric measures.

Returns
-------
numpy.ndarray of shape (6, N), the weighted densities n0, n1, n2, n3, nV1 and nV2 in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fmt_weighted_densities(rho: "np.ndarray", radius: float, dz: float) -> "np.ndarray":
    '''Six fundamental measure weighted densities of a planar profile.

    Parameters
    ----------
    rho : numpy.ndarray
        Segment number density on the grid in angstrom^-3.
    radius : float
        Effective hard-sphere radius in angstrom.
    dz : float
        Uniform grid spacing in angstrom.

    Returns
    -------
    numpy.ndarray
        Array of shape (6, rho.size) holding n0, n1, n2, n3, nV1 and nV2 in that order.

    Raises
    ------
    ValueError
        If radius or dz is not positive.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_fmt_weighted_densities(rho: "np.ndarray", radius: float, dz: float) -> "np.ndarray":
    if radius <= 0 or dz <= 0:
        raise ValueError("radius and dz must be positive")
    rho = np.asarray(rho, dtype=float)
    k = int(np.ceil(radius/dz - 0.5)) + 1
    j = np.arange(-k, k + 1)
    a = np.clip((j - 0.5)*dz, -radius, radius)
    b = np.clip((j + 0.5)*dz, -radius, radius)
    w3 = np.pi*((radius**2*b - b**3/3.0) - (radius**2*a - a**3/3.0))
    w2 = 2.0*np.pi*radius*(b - a)
    wv2 = np.pi*(b**2 - a**2)
    def _c(a, w):
        full = np.convolve(a, w)
        off = (w.size - 1)//2
        return full[off:off + a.size]
    n2 = _c(rho, w2)
    n3 = _c(rho, w3)
    nv2 = _c(rho, wv2)
    n1 = n2/(4.0*np.pi*radius)
    n0 = n2/(4.0*np.pi*radius**2)
    nv1 = nv2/(4.0*np.pi*radius)
    return np.vstack([n0, n1, n2, n3, nv1, nv2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nR = np.full(801, 0.01)",
            "call": "fmt_weighted_densities(R, 1.5675050365, 0.01)",
            "gold_call": "_oracle_fmt_weighted_densities(R, 1.5675050365, 0.01)",
        },
        {
            "setup": "import numpy as np\nZ = np.linspace(0.0, 14.0, 1401)\nR = 0.02*np.exp(-((Z-7.0)/1.5)**2)",
            "call": "fmt_weighted_densities(R, 1.5675050365, 0.01)",
            "gold_call": "_oracle_fmt_weighted_densities(R, 1.5675050365, 0.01)",
        },
        {
            "setup": "import numpy as np\nR = np.zeros(401)\nR[200] = 0.5",
            "call": "fmt_weighted_densities(R, 1.5675050365, 0.02)",
            "gold_call": "_oracle_fmt_weighted_densities(R, 1.5675050365, 0.02)",
        },
        {
            "setup": "import numpy as np\nZ = np.linspace(0.0, 8.0, 401)\nR = 0.015*(Z > 2.0)*(Z < 6.0)",
            "call": "fmt_weighted_densities(R, 1.4, 0.02)",
            "gold_call": "_oracle_fmt_weighted_densities(R, 1.4, 0.02)",
        },
        {
            "setup": "import numpy as np\nR = np.zeros(301)",
            "call": "fmt_weighted_densities(R, 1.5675050365, 0.01)",
            "gold_call": "_oracle_fmt_weighted_densities(R, 1.5675050365, 0.01)",
        },
    ]
