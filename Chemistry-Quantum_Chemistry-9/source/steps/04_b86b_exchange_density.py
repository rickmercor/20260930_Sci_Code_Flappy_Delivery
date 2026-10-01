"""
Return the gradient-corrected semilocal exchange energy density of one spin channel, evaluated pointwise. Write rho for the density and x for the absolute gradient divided by rho**(4/3). The value is minus (3/2)*(3/(4*pi))**(1/3) times rho**(4/3), minus 0.00375 times rho**(4/3) times x**2 divided by (1 + 0.007*x**2)**(4/5). The channel enters on its own and the two channels are summed only later, so no spin-scaling factor is applied here.

The leading term is the uniform-gas exchange energy density written per spin channel, and the second is a gradient correction whose denominator exponent is what makes the functional recover the correct large-gradient behaviour of an exponentially decaying density. The two constants are not free: they were fixed once against rare-gas atoms and have been carried unchanged ever since.

Returns
-------
ndarray of shape (len(density),): the semilocal exchange energy density of that spin channel, in hartree per bohr cubed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def b86b_exchange_density(density: "np.ndarray", gradient: "np.ndarray") -> "np.ndarray":
    '''Return the gradient-corrected semilocal exchange energy density of one spin channel, evaluated pointwise. Write rho for the density and x for the absolute gradient divided by rho**(4/3). The value is minus (3/2)*(3/(4*pi))**(1/3) times rho**(4/3), minus 0.00375 times rho**(4/3) times x**2 divided by (1 + 0.007*x**2)**(4/5). The channel enters on its own and the two channels are summed only later, so no spin-scaling factor is applied here.

    Parameters
    ----------
    density : np.ndarray
        Strictly positive spin density in inverse bohr cubed.
    gradient : np.ndarray
        Radial derivative of that density, same length.

    Returns
    -------
    energy_density : np.ndarray
        ndarray of shape (len(density),): the semilocal exchange energy density of that spin channel, in hartree per bohr cubed.

    Raises
    ------
    ValueError
        if density and gradient have different lengths, or if any density value is not positive.
    '''
    return energy_density  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_b86b_exchange_density(density: "np.ndarray", gradient: "np.ndarray") -> "np.ndarray":
    """B86b exchange energy density of one spin channel."""
    rho = np.asarray(density, dtype=float).ravel()
    g = np.asarray(gradient, dtype=float).ravel()
    if rho.size != g.size:
        raise ValueError("density and gradient must have the same length")
    if np.any(rho <= 0.0):
        raise ValueError("every density value must be positive")
    cx = 1.5 * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    r43 = rho ** (4.0 / 3.0)
    x2 = g * g / rho ** (8.0 / 3.0)
    return -cx * r43 - 0.00375 * r43 * x2 / (1.0 + 0.007 * x2) ** 0.8

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)",
         "call": "b86b_exchange_density(FA[0], FA[1])",
         "gold_call": "_oracle_b86b_exchange_density(FA[0], FA[1])"},   # normal
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)",
         "call": "b86b_exchange_density(FB[0], np.zeros(RG.size))",
         "gold_call": "_oracle_b86b_exchange_density(FB[0], np.zeros(RG.size))"},   # boundary
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)",
         "call": "b86b_exchange_density(FB[0], FB[1])",
         "gold_call": "_oracle_b86b_exchange_density(FB[0], FB[1])"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        b86b_exchange_density(np.array([1.0,-1.0]), np.array([0.1,0.1]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_b86b_exchange_density(np.array([1.0,-1.0]), np.array([0.1,0.1]))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
