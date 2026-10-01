"""
Sensitivity of the hard-sphere free-energy density to each weighted density.

The excess free-energy density of the hard-sphere reference is an algebraic function of the six weighted densities, built so that it reproduces the accepted bulk equation of state when the vector measures vanish. What the Euler-Lagrange equation needs is not that density itself but its six partial derivatives. Two of the terms couple a scalar measure to a vector one and therefore contribute with the sign that the vector weight carries. The third group of terms gathers a prefactor that depends only on the packing measure and multiplies a combination of the surface measures; that prefactor approaches a finite limit as the packing vanishes even though it is written as a ratio whose numerator and denominator both go to zero, so it must stay well behaved in the empty part of the pore. Return the six partial derivatives in the same order as their weighted densities.

Returns
-------
numpy.ndarray of shape (6, N), the partial derivative of the reduced free-energy density with respect to each weighted density, in the same order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def white_bear_partials(weighted: "np.ndarray") -> "np.ndarray":
    '''Partial derivatives of the hard-sphere free-energy density.

    Parameters
    ----------
    weighted : numpy.ndarray
        Array of shape (6, N) holding n0, n1, n2, n3, nV1 and nV2 on the grid.

    Returns
    -------
    numpy.ndarray
        Array of shape (6, N) holding the partial derivative of the reduced free-energy density
        with respect to each of the six weighted densities, in the same order.

    Raises
    ------
    ValueError
        If weighted does not have six rows.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_white_bear_partials(weighted: "np.ndarray") -> "np.ndarray":
    w = np.asarray(weighted, dtype=float)
    if w.shape[0] != 6:
        raise ValueError("weighted must have six rows")
    n0, n1, n2, n3, nv1, nv2 = w
    x = np.clip(n3, 0.0, 1.0 - 1e-10)
    om = 1.0 - x
    small = x < 1e-5
    xs = np.where(small, 1.0, x)
    oms = 1.0 - xs
    num = xs + oms**2*np.log(oms)
    nump = xs - 2.0*oms*np.log(oms)
    P = np.where(small, 1.5 - x/3.0 - x**2/12.0 - x**3/30.0, num/xs**2)
    dP = np.where(small, -1.0/3.0 - x/6.0 - x**2/10.0, (nump*xs - 2.0*num)/xs**3)
    f = P/(36.0*np.pi*om**2)
    df = (dP*om + 2.0*P)/(36.0*np.pi*om**3)
    d0 = -np.log(om)
    d1 = n2/om
    d2 = n1/om + f*(3.0*n2**2 - 3.0*nv2**2)
    d3 = n0/om + (n1*n2 - nv1*nv2)/om**2 + df*(n2**3 - 3.0*n2*nv2**2)
    dv1 = -nv2/om
    dv2 = -nv1/om - 6.0*f*n2*nv2
    return np.vstack([d0, d1, d2, d3, dv1, dv2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nR=1.5675050365\nrb=0.01\nW=np.array([[rb],[R*rb],[4*np.pi*R**2*rb],[(4/3)*np.pi*R**3*rb],[0.0],[0.0]])",
            "call": "white_bear_partials(W)",
            "gold_call": "_oracle_white_bear_partials(W)",
        },
        {
            "setup": "import numpy as np\nR=1.5675050365\nrb=0.03\nW=np.array([[rb],[R*rb],[4*np.pi*R**2*rb],[(4/3)*np.pi*R**3*rb],[0.2*R*rb],[0.2*4*np.pi*R**2*rb]])",
            "call": "white_bear_partials(W)",
            "gold_call": "_oracle_white_bear_partials(W)",
        },
        {
            "setup": "import numpy as np\nW=np.zeros((6,5))",
            "call": "white_bear_partials(W)",
            "gold_call": "_oracle_white_bear_partials(W)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(3)\nR=1.5675050365\nrb=rng.uniform(0.001,0.02,7)\nW=np.vstack([rb,R*rb,4*np.pi*R**2*rb,(4/3)*np.pi*R**3*rb,-0.3*R*rb,-0.3*4*np.pi*R**2*rb])",
            "call": "white_bear_partials(W)",
            "gold_call": "_oracle_white_bear_partials(W)",
        },
        {
            "setup": "import numpy as np\nR=1.5675050365\nrb=np.array([1e-9,1e-6,1e-3])\nW=np.vstack([rb,R*rb,4*np.pi*R**2*rb,(4/3)*np.pi*R**3*rb,0*rb,0*rb])",
            "call": "white_bear_partials(W)",
            "gold_call": "_oracle_white_bear_partials(W)",
        },
    ]
