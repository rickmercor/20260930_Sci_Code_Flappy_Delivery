"""
Contact pair correlation of the reference fluid, corrected for local anisotropy.

The strength of a hydrogen bond between two neighbouring molecules is proportional to how often the reference fluid actually places them in contact, so the association term needs the contact value of the pair correlation function evaluated locally rather than at the reservoir density. Carrying the bulk expression over to an inhomogeneous fluid requires two changes. The bulk packing moments are replaced by the corresponding geometric weighted densities, and the result is corrected by a scalar built from the vector weighted density that measures how one sided the local environment is: it equals one in a uniform fluid and falls towards zero where the profile is strongly layered, which suppresses contact against a wall. Return that correction factor and the contact pair correlation stacked in that order.

Returns
-------
numpy.ndarray of shape (2, N), the anisotropy factor then the contact pair correlation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contact_pair_correlation(weighted: "np.ndarray", hs_diameter: float) -> "np.ndarray":
    '''Local anisotropy factor and contact pair correlation of the hard-sphere reference.

    Parameters
    ----------
    weighted : numpy.ndarray
        Array of shape (6, N) holding n0, n1, n2, n3, nV1 and nV2 on the grid.
    hs_diameter : float
        Effective hard-sphere diameter in angstrom.

    Returns
    -------
    numpy.ndarray
        Array of shape (2, N); first row the dimensionless anisotropy factor, second row the
        contact pair correlation function.

    Raises
    ------
    ValueError
        If weighted does not have six rows or hs_diameter is not positive.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_contact_pair_correlation(weighted: "np.ndarray", hs_diameter: float) -> "np.ndarray":
    w = np.asarray(weighted, dtype=float)
    if w.shape[0] != 6:
        raise ValueError("weighted must have six rows")
    if hs_diameter <= 0:
        raise ValueError("hs_diameter must be positive")
    n0, n1, n2, n3, nv1, nv2 = w
    d = hs_diameter
    zeta = 1.0 - (nv2**2)/np.where(n2 > 0.0, n2**2, 1.0)
    zeta = np.clip(zeta, 0.0, 1.0)
    om = np.clip(1.0 - n3, 1e-10, None)
    g = (1.0/om
         + (d/4.0)*n2*zeta/om**2
         + (d**2/72.0)*n2**2*zeta/om**3)
    return np.vstack([zeta, g])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nR=1.5675050365\nrb=0.01\nW=np.array([[rb],[R*rb],[4*np.pi*R**2*rb],[(4/3)*np.pi*R**3*rb],[0.0],[0.0]])",
            "call": "contact_pair_correlation(W, 3.135010073)",
            "gold_call": "_oracle_contact_pair_correlation(W, 3.135010073)",
        },
        {
            "setup": "import numpy as np\nR=1.5675050365\nrb=0.02\nn2=4*np.pi*R**2*rb\nW=np.array([[rb],[R*rb],[n2],[(4/3)*np.pi*R**3*rb],[0.3*R*rb],[0.5*n2]])",
            "call": "contact_pair_correlation(W, 3.135010073)",
            "gold_call": "_oracle_contact_pair_correlation(W, 3.135010073)",
        },
        {
            "setup": "import numpy as np\nW=np.zeros((6,4))",
            "call": "contact_pair_correlation(W, 3.135010073)",
            "gold_call": "_oracle_contact_pair_correlation(W, 3.135010073)",
        },
        {
            "setup": "import numpy as np\nZ=np.linspace(0.0,14.0,1401)\nRHO=0.02*np.exp(-((Z-7.0)/1.5)**2)\nW_MODEL=fmt_weighted_densities(RHO,1.5675050365,0.01)\nW_GOLD=_oracle_fmt_weighted_densities(RHO,1.5675050365,0.01)",
            "call": "contact_pair_correlation(W_MODEL, 3.135010073)",
            "gold_call": "_oracle_contact_pair_correlation(W_GOLD, 3.135010073)",
        },
        {
            "setup": "import numpy as np\nR=1.5675050365\nrb=np.array([0.001,0.01,0.03])\nn2=4*np.pi*R**2*rb\nW=np.vstack([rb,R*rb,n2,(4/3)*np.pi*R**3*rb,-0.2*R*rb,-0.9*n2])",
            "call": "contact_pair_correlation(W, 3.135010073)",
            "gold_call": "_oracle_contact_pair_correlation(W, 3.135010073)",
        },
    ]
