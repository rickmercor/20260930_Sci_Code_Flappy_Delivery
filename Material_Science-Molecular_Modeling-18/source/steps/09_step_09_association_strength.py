"""
Strength of an association interaction between two sites on neighbouring segments.

The association strength collects the probability that two segments sit close enough for their sites to overlap and the Boltzmann gain when they do. For a square-well site-site interaction it factorises into the Mayer factor of the well and a purely geometric bonding volume, multiplied by the contact pair correlation of the reference fluid. The bonding volume follows from integrating the overlap of two site shells over the separations at which a bond can form, which leaves a closed algebraic expression in the site displacement, the well range and the hard-sphere diameter. It carries a logarithm of a ratio of those lengths together with a polynomial in them, and it is normalised so that the strength has the dimensions of a volume. Distances are in angstrom, well depths as temperatures in kelvin, and the returned strength is in cubic angstrom.

Returns
-------
numpy.ndarray, the association strength on the grid in angstrom^3
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def association_strength(g_contact: "np.ndarray", hs_diameter: float, sigma: float, epsilon_hb: float, r_site: float, r_cut: float, temperature: float) -> "np.ndarray":
    '''Association strength between two complementary sites.

    Parameters
    ----------
    g_contact : numpy.ndarray
        Contact pair correlation of the reference fluid on the grid.
    hs_diameter : float
        Effective hard-sphere diameter in angstrom.
    sigma : float
        Mie segment diameter in angstrom.
    epsilon_hb : float
        Association well depth divided by the Boltzmann constant, in kelvin.
    r_site : float
        Distance from the segment centre to an association site in angstrom.
    r_cut : float
        Range of the square-well site-site interaction in angstrom.
    temperature : float
        Absolute temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        Association strength on the grid in angstrom^3.

    Raises
    ------
    ValueError
        If hs_diameter, sigma, r_site, r_cut or temperature is not positive.
    '''
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_association_strength(g_contact: "np.ndarray", hs_diameter: float, sigma: float, epsilon_hb: float, r_site: float, r_cut: float, temperature: float) -> "np.ndarray":
    if min(hs_diameter, sigma, r_site, r_cut, temperature) <= 0:
        raise ValueError("hs_diameter, sigma, r_site, r_cut and temperature must be positive")
    g = np.asarray(g_contact, dtype=float)
    d, rc, rd = hs_diameter, r_cut, r_site
    t1 = np.log((rc + 2.0*rd)/d)*(6.0*rc**3 + 18.0*rc**2*rd - 24.0*rd**3)
    t2 = (rc + 2.0*rd - d)*(22.0*rd**2 - 5.0*rc*rd - 7.0*rd*d - 8.0*rc**2 + rc*d + d**2)
    kappa = 4.0*np.pi*d**2*(t1 + t2)/(72.0*rd**2*sigma**3)
    return sigma**3*(np.exp(epsilon_hb/temperature) - 1.0)*kappa*g

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nG=np.array([1.55848])",
            "call": "association_strength(G, 3.135010073, 3.161, 1210.0, 1.2644, 1.8440374, 425.0)",
            "gold_call": "_oracle_association_strength(G, 3.135010073, 3.161, 1210.0, 1.2644, 1.8440374, 425.0)",
        },
        {
            "setup": "import numpy as np\nG=np.array([1.0,2.0,5.0,10.0])",
            "call": "association_strength(G, 3.135010073, 3.161, 1210.0, 1.2644, 1.8440374, 425.0)",
            "gold_call": "_oracle_association_strength(G, 3.135010073, 3.161, 1210.0, 1.2644, 1.8440374, 425.0)",
        },
        {
            "setup": "import numpy as np\nG=np.zeros(3)",
            "call": "association_strength(G, 3.135010073, 3.161, 1210.0, 1.2644, 1.8440374, 425.0)",
            "gold_call": "_oracle_association_strength(G, 3.135010073, 3.161, 1210.0, 1.2644, 1.8440374, 425.0)",
        },
        {
            "setup": "import numpy as np\nG=np.array([1.55848])",
            "call": "association_strength(G, 3.135010073, 3.161, 1210.0, 1.2644, 1.8440374, 700.0)",
            "gold_call": "_oracle_association_strength(G, 3.135010073, 3.161, 1210.0, 1.2644, 1.8440374, 700.0)",
        },
        {
            "setup": "import numpy as np\nG=np.linspace(1.0,4.0,6)",
            "call": "association_strength(G, 3.0, 3.05, 1500.0, 1.22, 1.70, 350.0)",
            "gold_call": "_oracle_association_strength(G, 3.0, 3.05, 1500.0, 1.22, 1.70, 350.0)",
        },
    ]
