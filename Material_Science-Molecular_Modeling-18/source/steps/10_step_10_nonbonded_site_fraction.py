"""
Fraction of association sites left unbonded at each position in the pore.

Wertheim's first-order treatment closes the association problem with a mass-action condition: the fraction of sites of one type that remain free is set by the density of complementary partners available to bond with, weighted by the association strength. The density that enters is not the local density but the geometric weighted density that measures how many segment centres are within reach, reduced by the same anisotropy factor that corrects the contact value. Water carries two donor and two acceptor sites and a donor may bond only to an acceptor, so the two site types share one free fraction and the mass-action condition collapses to a single quadratic with one physical root. That root must tend to unity as the association strength vanishes.

Returns
-------
numpy.ndarray, the fraction of non-bonded sites on the grid
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonbonded_site_fraction(n0: "np.ndarray", zeta: "np.ndarray", delta: "np.ndarray") -> "np.ndarray":
    '''Fraction of association sites that are not hydrogen bonded.

    Parameters
    ----------
    n0 : numpy.ndarray
        Zeroth geometric weighted density on the grid in angstrom^-3.
    zeta : numpy.ndarray
        Dimensionless local anisotropy factor on the grid.
    delta : numpy.ndarray
        Association strength on the grid in angstrom^3.

    Returns
    -------
    numpy.ndarray
        Fraction of non-bonded sites on the grid, between zero and one.

    Raises
    ------
    ValueError
        If the three inputs do not have the same shape.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_nonbonded_site_fraction(n0: "np.ndarray", zeta: "np.ndarray", delta: "np.ndarray") -> "np.ndarray":
    a = np.asarray(n0, dtype=float)
    b = np.asarray(zeta, dtype=float)
    c = np.asarray(delta, dtype=float)
    if not (a.shape == b.shape == c.shape):
        raise ValueError("n0, zeta and delta must have the same shape")
    u = a*b*c
    return 2.0/(1.0 + np.sqrt(1.0 + 8.0*u))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nN0=np.array([0.01])\nZ=np.array([1.0])\nD=np.array([68.9252])",
            "call": "nonbonded_site_fraction(N0, Z, D)",
            "gold_call": "_oracle_nonbonded_site_fraction(N0, Z, D)",
        },
        {
            "setup": "import numpy as np\nN0=np.array([0.0,0.005,0.02,0.05])\nZ=np.ones(4)\nD=np.full(4,70.0)",
            "call": "nonbonded_site_fraction(N0, Z, D)",
            "gold_call": "_oracle_nonbonded_site_fraction(N0, Z, D)",
        },
        {
            "setup": "import numpy as np\nN0=np.full(4,0.02)\nZ=np.array([1.0,0.9,0.6,0.0])\nD=np.full(4,120.0)",
            "call": "nonbonded_site_fraction(N0, Z, D)",
            "gold_call": "_oracle_nonbonded_site_fraction(N0, Z, D)",
        },
        {
            "setup": "import numpy as np\nN0=np.full(3,0.01)\nZ=np.ones(3)\nD=np.array([0.0,1e-8,1e4])",
            "call": "nonbonded_site_fraction(N0, Z, D)",
            "gold_call": "_oracle_nonbonded_site_fraction(N0, Z, D)",
        },
        {
            "setup": "import numpy as np\nrng=np.random.default_rng(5)\nN0=rng.uniform(0,0.05,7)\nZ=rng.uniform(0.5,1.0,7)\nD=rng.uniform(10,300,7)",
            "call": "nonbonded_site_fraction(N0, Z, D)",
            "gold_call": "_oracle_nonbonded_site_fraction(N0, Z, D)",
        },
        {
            "setup": "import numpy as np\nN0=np.array([1e-20,1e-16,1e-12])\nZ=np.ones(3)\nD=np.ones(3)",
            "call": "nonbonded_site_fraction(N0, Z, D)",
            "gold_call": "_oracle_nonbonded_site_fraction(N0, Z, D)",
        },
    ]
