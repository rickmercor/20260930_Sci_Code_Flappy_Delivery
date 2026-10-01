"""
Return the Bjerrum length, the lattice cell volume and the reduced surface parameter of a uniformly charged plane facing a solvent of given permittivity and temperature.

A planar wall carrying a uniform surface charge density sigma faces a solvent of relative permittivity eps_r at absolute temperature T. The Bjerrum length is the separation at which the Coulomb energy of two elementary charges in that medium equals the thermal energy, l_B = e**2 / (4 pi eps_0 eps_r k_B T), reported in angstrom. The solvent is coarse-grained onto a cubic lattice of cell edge a, and the cell volume a**3 is the volume of one solvent molecule.

Every later step sees the wall only through the dimensionless reduced surface parameter

zeta = 2 pi l_B a**3 (sigma / e)**2,

with l_B and a in angstrom and sigma / e in elementary charges per square angstrom. Because it is quadratic in sigma, walls of either sign with the same magnitude of charge share one value of zeta, and an uncharged wall has zeta = 0.

Use e = 1.602176634e-19 C, k_B = 1.380649e-23 J/K and eps_0 = 8.8541878128e-12 F/m.

Returns
-------
np.ndarray of length 3: the Bjerrum length (angstrom), the cell volume (cubic angstrom) and the reduced surface parameter zeta
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def electrostatic_scales(eps_r: float, T_K: float, a_ang: float, sigma_e_per_A2: float) -> "np.ndarray":
    '''Bjerrum length, lattice cell volume and reduced surface parameter of a charged plane.

    Parameters
    ----------
    eps_r : float
        Relative permittivity of the solvent; strictly positive.
    T_K : float
        Absolute temperature in kelvin; strictly positive.
    a_ang : float
        Lattice cell edge in angstrom; strictly positive.
    sigma_e_per_A2 : float
        Surface charge density in elementary charges per square angstrom; any sign, zero allowed.

    Returns
    -------
    result : np.ndarray
        Real array of length 3 holding, in order, the Bjerrum length in angstrom, the cell volume
        in cubic angstrom and the dimensionless reduced surface parameter zeta.

    Raises
    ------
    ValueError
        If eps_r, T_K or a_ang is not strictly positive.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_electrostatic_scales(eps_r: float, T_K: float, a_ang: float, sigma_e_per_A2: float) -> "np.ndarray":
    eps_r = float(eps_r)
    T_K = float(T_K)
    a_ang = float(a_ang)
    sigma = float(sigma_e_per_A2)
    if eps_r <= 0.0 or T_K <= 0.0 or a_ang <= 0.0:
        raise ValueError("eps_r, T_K and a_ang must all be strictly positive")
    e_charge = 1.602176634e-19
    k_boltzmann = 1.380649e-23
    eps_vacuum = 8.8541878128e-12
    lB = e_charge ** 2 / (4.0 * np.pi * eps_vacuum * eps_r * k_boltzmann * T_K) * 1.0e10
    a3 = a_ang ** 3
    zeta = 2.0 * np.pi * lB * a3 * sigma * sigma
    return np.array([lB, a3, zeta])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # normal, the aqueous electrolyte of the task at a strongly charged wall
        {
            "setup": "import numpy as np",
            "call": "electrostatic_scales(78.5, 298.15, 5.00, 0.0325)",
            "gold_call": "_oracle_electrostatic_scales(78.5, 298.15, 5.00, 0.0325)",
        },
        # normal, a water-sized cell at a weaker surface charge
        {
            "setup": "import numpy as np",
            "call": "electrostatic_scales(78.5, 298.15, 3.107, 0.020)",
            "gold_call": "_oracle_electrostatic_scales(78.5, 298.15, 3.107, 0.020)",
        },
        # boundary, an uncharged wall has zeta equal to zero
        {
            "setup": "import numpy as np",
            "call": "electrostatic_scales(78.5, 298.15, 5.00, 0.0)",
            "gold_call": "_oracle_electrostatic_scales(78.5, 298.15, 5.00, 0.0)",
        },
        # edge, a low-permittivity hot solvent and a negatively charged wall with a large cell
        {
            "setup": "import numpy as np",
            "call": "electrostatic_scales(2.0, 400.0, 7.5, -0.055)",
            "gold_call": "_oracle_electrostatic_scales(2.0, 400.0, 7.5, -0.055)",
        },
        # contract, a vanishing permittivity must raise ValueError
        {
            "setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0",
            "call": "_raises(lambda: electrostatic_scales(0.0, 298.15, 5.0, 0.01))",
            "gold_call": "_raises(lambda: _oracle_electrostatic_scales(0.0, 298.15, 5.0, 0.01))",
        },
        # contract, a negative cell edge must raise ValueError
        {
            "setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0",
            "call": "_raises(lambda: electrostatic_scales(78.5, 298.15, -5.0, 0.01))",
            "gold_call": "_raises(lambda: _oracle_electrostatic_scales(78.5, 298.15, -5.0, 0.01))",
        },
    ]
