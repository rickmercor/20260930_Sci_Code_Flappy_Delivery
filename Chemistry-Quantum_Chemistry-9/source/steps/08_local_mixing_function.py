"""
Return the position-dependent fraction of exact exchange the local hybrid mixes in, evaluated pointwise. Write rho for the density, eps for the semilocal exchange energy density of the same channel, and n for the exchange-correlation hole normalisation of the same channel. Form the length z as the absolute value of rho divided by eps. Form the dimensionless scaling factor s as b_param times the square of the sine of pi times n, plus one, so that s equals one wherever n is a whole number and rises to b_param plus one halfway between two whole numbers. The mixing fraction at each point is the error function of c_param times s times z, so the scaling factor multiplies the argument of the error function and not the function itself. Return one value per point, each between zero and one.

The ratio of a density to its own semilocal exchange energy density carries units of length and grows in the exponentially decaying tail while staying small near the nucleus, so an increasing, bounded function of it mixes in little exact exchange where a semilocal functional is reliable and almost all of it far out. The squared sine leaves that behaviour untouched in regions holding a whole exchange electron and turns the mixing up sharply where a region holds a fraction of one, which is where a semilocal functional spreads charge it should not.

Returns
-------
ndarray of shape (len(density),): the local exact-exchange mixing fraction, dimensionless and between zero and one.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def local_mixing_function(density: "np.ndarray", b86b_energy_density: "np.ndarray", xc_hole_norm: "np.ndarray", c_param: float, b_param: float) -> "np.ndarray":
    '''Return the position-dependent fraction of exact exchange the local hybrid mixes in, evaluated pointwise. Write rho for the density, eps for the semilocal exchange energy density of the same channel, and n for the exchange-correlation hole normalisation of the same channel. Form the length z as the absolute value of rho divided by eps. Form the dimensionless scaling factor s as b_param times the square of the sine of pi times n, plus one, so that s equals one wherever n is a whole number and rises to b_param plus one halfway between two whole numbers. The mixing fraction at each point is the error function of c_param times s times z, so the scaling factor multiplies the argument of the error function and not the function itself. Return one value per point, each between zero and one.

    Parameters
    ----------
    density : np.ndarray
        Strictly positive spin density in inverse bohr cubed.
    b86b_energy_density : np.ndarray
        Non-zero semilocal exchange energy density of the same channel, same length.
    xc_hole_norm : np.ndarray
        Effective exchange-correlation hole normalisation of the same channel, same length.
    c_param : float
        Positive coefficient multiplying the length inside the mixing function.
    b_param : float
        Non-negative constant scaling the hole-normalisation factor.

    Returns
    -------
    mixing : np.ndarray
        ndarray of shape (len(density),): the local exact-exchange mixing fraction, dimensionless and between zero and one.

    Raises
    ------
    ValueError
        if density, b86b_energy_density and xc_hole_norm do not all have the same length, if any density value is not positive, if any semilocal exchange energy density is zero, if c_param is not positive, or if b_param is negative.
    '''
    return mixing  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _erf_array(values):
    """Error function evaluated elementwise, using only the standard library."""
    arr = np.asarray(values, dtype=float)
    flat = np.array([math.erf(float(v)) for v in arr.ravel()], dtype=float)
    return flat.reshape(arr.shape)


def _oracle_local_mixing_function(density: "np.ndarray", b86b_energy_density: "np.ndarray", xc_hole_norm: "np.ndarray", c_param: float, b_param: float) -> "np.ndarray":
    """Position-dependent exact-exchange mixing fraction of the LHnz local hybrid."""
    rho = np.asarray(density, dtype=float).ravel()
    eb = np.asarray(b86b_energy_density, dtype=float).ravel()
    n = np.asarray(xc_hole_norm, dtype=float).ravel()
    if not (rho.size == eb.size == n.size):
        raise ValueError("density, b86b_energy_density and xc_hole_norm must have the same length")
    if np.any(rho <= 0.0):
        raise ValueError("every density value must be positive")
    if np.any(eb == 0.0):
        raise ValueError("no B86b exchange energy density may vanish")
    if float(c_param) <= 0.0:
        raise ValueError("c_param must be positive")
    if float(b_param) < 0.0:
        raise ValueError("b_param must not be negative")
    z = np.abs(rho / eb)
    s = float(b_param) * np.sin(math.pi * n) ** 2 + 1.0
    return _erf_array(float(c_param) * s * z)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)\nNA=_oracle_effective_hole_normalisation(FA[0],FA[4],XA); NB=_oracle_effective_hole_normalisation(FB[0],FB[4],XB)\nBA=_oracle_b86b_exchange_density(FA[0],FA[1]); BB=_oracle_b86b_exchange_density(FB[0],FB[1])\nNC=_oracle_xc_hole_normalisation(NA,NB)",
         "call": "local_mixing_function(FA[0], BA, NC, 0.10, 4.6)",
         "gold_call": "_oracle_local_mixing_function(FA[0], BA, NC, 0.10, 4.6)"},   # normal
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nBA=_oracle_b86b_exchange_density(FA[0],FA[1]); BB=_oracle_b86b_exchange_density(FB[0],FB[1])",
         "call": "local_mixing_function(FB[0], BB, np.ones(RG.size), 0.10, 4.6)",
         "gold_call": "_oracle_local_mixing_function(FB[0], BB, np.ones(RG.size), 0.10, 4.6)"},   # boundary
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nBA=_oracle_b86b_exchange_density(FA[0],FA[1]); BB=_oracle_b86b_exchange_density(FB[0],FB[1])",
         "call": "local_mixing_function(FA[0], BA, 0.5*np.ones(RG.size), 0.25, 0.0)",
         "gold_call": "_oracle_local_mixing_function(FA[0], BA, 0.5*np.ones(RG.size), 0.25, 0.0)"},   # edge
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nBA=_oracle_b86b_exchange_density(FA[0],FA[1]); BB=_oracle_b86b_exchange_density(FB[0],FB[1])\ndef _exc(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_exc(lambda: local_mixing_function(FA[0], BA, np.ones(RG.size), 0.0, 4.6))",
         "gold_call": "_exc(lambda: _oracle_local_mixing_function(FA[0], BA, np.ones(RG.size), 0.0, 4.6))"},   # invalid input
    ]
