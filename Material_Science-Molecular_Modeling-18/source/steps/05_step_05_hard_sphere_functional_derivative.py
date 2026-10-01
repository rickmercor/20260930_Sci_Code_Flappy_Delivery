"""
One-body direct correlation of the hard-sphere reference along the pore axis.

The functional derivative of the hard-sphere free energy with respect to the density profile is not a pointwise quantity: each partial derivative of the free-energy density has to be carried back through the same geometric weight that produced its weighted density. The scalar weights are even in the offset and so appear as ordinary convolutions, while the two vector weights are odd, which flips the sign of their contribution relative to the scalar ones. Use the same cell integrated weights and discrete summation convention as the weighted densities themselves. The result is dimensionless and enters the equilibrium condition alongside the external field.

Returns
-------
numpy.ndarray, the dimensionless functional derivative on the grid
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hard_sphere_functional_derivative(rho: "np.ndarray", radius: float, dz: float) -> "np.ndarray":
    '''Functional derivative of the reduced hard-sphere free energy.

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
        Dimensionless functional derivative on the grid, same length as rho.

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

def _oracle_hard_sphere_functional_derivative(rho: "np.ndarray", radius: float, dz: float) -> "np.ndarray":
    if radius <= 0 or dz <= 0:
        raise ValueError("radius and dz must be positive")
    rho = np.asarray(rho, dtype=float)
    parts = _oracle_white_bear_partials(_oracle_fmt_weighted_densities(rho, radius, dz))
    d0, d1, d2, d3, dv1, dv2 = parts
    k = int(np.ceil(radius/dz - 0.5)) + 1
    j = np.arange(-k, k + 1)
    a = np.clip((j - 0.5)*dz, -radius, radius)
    b = np.clip((j + 0.5)*dz, -radius, radius)
    w3 = np.pi*((radius**2*b - b**3/3.0) - (radius**2*a - a**3/3.0))
    w2 = 2.0*np.pi*radius*(b - a)
    wv2 = np.pi*(b**2 - a**2)
    w1 = w2/(4.0*np.pi*radius)
    w0 = w2/(4.0*np.pi*radius**2)
    wv1 = wv2/(4.0*np.pi*radius)
    def _c(arr, w):
        full = np.convolve(arr, w)
        off = (w.size - 1)//2
        return full[off:off + arr.size]
    return (_c(d0, w0) + _c(d1, w1) + _c(d2, w2) + _c(d3, w3)
            - _c(dv1, wv1) - _c(dv2, wv2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nR=np.full(801,0.01)",
            "call": "hard_sphere_functional_derivative(R, 1.5675050365, 0.01)",
            "gold_call": "_oracle_hard_sphere_functional_derivative(R, 1.5675050365, 0.01)",
        },
        {
            "setup": "import numpy as np\nZ=np.linspace(0.0,14.0,1401)\nR=0.02*np.exp(-((Z-7.0)/1.5)**2)",
            "call": "hard_sphere_functional_derivative(R, 1.5675050365, 0.01)",
            "gold_call": "_oracle_hard_sphere_functional_derivative(R, 1.5675050365, 0.01)",
        },
        {
            "setup": "import numpy as np\nZ=np.linspace(0.0,8.0,401)\nR=0.015*(Z>2.0)*(Z<6.0)",
            "call": "hard_sphere_functional_derivative(R, 1.4, 0.02)",
            "gold_call": "_oracle_hard_sphere_functional_derivative(R, 1.4, 0.02)",
        },
        {
            "setup": "import numpy as np\nR=np.zeros(301)",
            "call": "hard_sphere_functional_derivative(R, 1.5675050365, 0.01)",
            "gold_call": "_oracle_hard_sphere_functional_derivative(R, 1.5675050365, 0.01)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(11)\nR=rng.uniform(0.0,0.03,501)",
            "call": "hard_sphere_functional_derivative(R, 1.5675050365, 0.01)",
            "gold_call": "_oracle_hard_sphere_functional_derivative(R, 1.5675050365, 0.01)",
        },
    ]
