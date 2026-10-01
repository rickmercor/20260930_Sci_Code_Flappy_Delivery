"""
Return the surface charge density, inside a given bracket, at which two ionic species have equal mean adsorption distances at a positively charged wall.

The mean adsorption distances of step 05 change with the surface charge density sigma at different rates for different species, so the order in which two species sit relative to the wall can reverse as the wall is charged more strongly. Two species exchange places at the surface charge density where their mean adsorption distances are equal.

Given the indices of two species and a bracket [sigma_lo, sigma_hi] across which the difference of their mean adsorption distances changes sign exactly once, return the surface charge density inside the bracket at which the two mean adsorption distances coincide. The result must be accurate to a relative 1e-9, and a bracket across which the difference does not change sign is not a valid input.

Returns
-------
float: the surface charge density in e per square angstrom at which the two mean adsorption distances are equal
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exchange_charge(eps_r: float, T_K: float, a_ang: float, z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", i: int, j: int, sigma_lo: float, sigma_hi: float) -> float:
    '''Surface charge density at which two species have equal mean adsorption distances.

    Parameters
    ----------
    eps_r : float
        Relative permittivity of the solvent; strictly positive.
    T_K : float
        Absolute temperature in kelvin; strictly positive.
    a_ang : float
        Lattice cell edge in angstrom; strictly positive.
    z : np.ndarray
        Real array of length N holding the signed valency of each species.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the cell volume;
        every entry strictly positive.
    phi_bulk : np.ndarray
        Real array of length N holding the reservoir volume fractions of an electroneutral
        reservoir; every entry strictly positive, with a sum strictly less than one.
    i : int
        Zero-based index of the first species.
    j : int
        Zero-based index of the second species, different from i.
    sigma_lo : float
        Lower end of the bracket in elementary charges per square angstrom; strictly positive.
    sigma_hi : float
        Upper end of the bracket in elementary charges per square angstrom; larger than sigma_lo.

    Returns
    -------
    sigma_x : float
        Surface charge density in elementary charges per square angstrom, between sigma_lo and
        sigma_hi, at which the mean adsorption distances of species i and j are equal.

    Raises
    ------
    ValueError
        If i and j are equal or out of range, if the bracket is not 0 < sigma_lo < sigma_hi, or if
        the difference of the two mean adsorption distances has the same sign at both ends of the
        bracket.
    '''
    return sigma_x  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _distance_gap(s, eps_r, T_K, a_ang, zz, v, phi_bulk, i, j):
    """Mean adsorption distance of species i minus that of species j at surface charge s."""
    n = zz.size
    m = _oracle_adsorption_moments(eps_r, T_K, a_ang, s, zz, v, phi_bulk)
    return float(m[n + i] - m[n + j])


def _oracle_exchange_charge(eps_r: float, T_K: float, a_ang: float, z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", i: int, j: int, sigma_lo: float, sigma_hi: float) -> float:
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    n = zz.size
    i = int(i)
    j = int(j)
    lo = float(sigma_lo)
    hi = float(sigma_hi)
    if i == j or not (0 <= i < n and 0 <= j < n):
        raise ValueError("i and j must be distinct valid species indices")
    if not (0.0 < lo < hi):
        raise ValueError("the bracket must satisfy 0 < sigma_lo < sigma_hi")
    args = (eps_r, T_K, a_ang, zz, v, phi_bulk, i, j)
    if _distance_gap(lo, *args) * _distance_gap(hi, *args) > 0.0:
        raise ValueError("the mean adsorption distances do not exchange inside the bracket")
    return float(brentq(_distance_gap, lo, hi, args=args, xtol=1e-300, rtol=1e-12, maxiter=200))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nZ = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nPB = np.array([2.0e-3, 1.0e-3, 5.0e-4, 5.1875e-3])",
            "call": "np.log(exchange_charge(78.5, 298.15, 5.00, Z, V, PB, 0, 1, 0.025, 0.040))",
            "gold_call": "np.log(_oracle_exchange_charge(78.5, 298.15, 5.00, Z, V, PB, 0, 1, 0.025, 0.040))",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nZ = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nPB = np.array([2.0e-3, 1.0e-3, 5.0e-4, 5.1875e-3])",
            "call": "np.log(exchange_charge(78.5, 298.15, 5.00, Z, V, PB, 2, 1, 0.004, 0.012))",
            "gold_call": "np.log(_oracle_exchange_charge(78.5, 298.15, 5.00, Z, V, PB, 2, 1, 0.004, 0.012))",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nZ = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nPB = np.array([2.0e-3, 1.0e-3, 5.0e-4, 5.1875e-3])",
            "call": "np.log(exchange_charge(78.5, 298.15, 5.00, Z, V, PB, 0, 1, 0.0318, 0.060))",
            "gold_call": "np.log(_oracle_exchange_charge(78.5, 298.15, 5.00, Z, V, PB, 0, 1, 0.0318, 0.060))",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nZ = np.array([-2.0, 1.0, -1.0])\nV = np.array([6.0, 0.3, 0.8])\nPB = np.array([4.0e-3, 1.6e-3, 3.2e-3])",
            "call": "np.log(exchange_charge(40.0, 330.0, 4.50, Z, V, PB, 2, 0, 0.002, 0.030))",
            "gold_call": "np.log(_oracle_exchange_charge(40.0, 330.0, 4.50, Z, V, PB, 2, 0, 0.002, 0.030))",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0\nZ = np.array([-1.0, -2.0, -3.0, 1.0])\nV = np.array([0.5, 2.0, 8.0, 1.0])\nPB = np.array([2.0e-3, 1.0e-3, 5.0e-4, 5.1875e-3])",
            "call": "_raises(lambda: exchange_charge(78.5, 298.15, 5.00, Z, V, PB, 0, 1, 0.040, 0.060))",
            "gold_call": "_raises(lambda: _oracle_exchange_charge(78.5, 298.15, 5.00, Z, V, PB, 0, 1, 0.040, 0.060))",
        },
    ]
