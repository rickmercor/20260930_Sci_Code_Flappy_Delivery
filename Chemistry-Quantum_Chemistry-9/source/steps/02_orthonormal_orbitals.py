"""
Build two orthonormal radial orbitals out of Slater primitives and return them as one padded coefficient table. A primitive carrying label one is sqrt(z**3/pi)*exp(-z*r) and a primitive carrying label two is sqrt(z**5/(3*pi))*r*exp(-z*r), each already scaled so that the integral of its square over all space is one. Define the overlap of two radial functions as the integral from zero to infinity of four pi r**2 times their product. The inner orbital is the sum of the label-one primitives built from inner_exponents weighted by inner_coefficients, divided by the square root of its own overlap with itself. The outer orbital starts as the single label-two primitive built from outer_exponent, then has the inner orbital subtracted from it once, weighted by the overlap of that starting primitive with the already-scaled inner orbital, and is only then divided by the square root of its own overlap with itself. The inner orbital is never modified, so the order of those two operations is part of the contract. Return an array whose first index selects the orbital with the inner one first, whose second index runs over terms, and whose third index holds the primitive label, the exponent and the coefficient in that order. List the inner orbital's terms in the order of inner_exponents, list the outer orbital's label-two term first and its label-one terms after it in that same order, and pad the inner orbital with trailing rows of all zeros so both orbitals carry the same number of terms.

Two radial functions drawn from different Slater exponents are not orthogonal, and a one-electron exchange hole is only exactly the density when the occupied set is orthonormal. Removing the inner function from the outer one and rescaling afterwards is not the same map as rescaling first, because the subtraction changes the norm.

Returns
-------
ndarray of shape (2, len(inner_exponents)+1, 3): orbital, term, then primitive label, Slater exponent and coefficient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def orthonormal_orbitals(inner_exponents: "np.ndarray", inner_coefficients: "np.ndarray", outer_exponent: float) -> "np.ndarray":
    '''Build two orthonormal radial orbitals out of Slater primitives and return them as one padded coefficient table. A primitive carrying label one is sqrt(z**3/pi)*exp(-z*r) and a primitive carrying label two is sqrt(z**5/(3*pi))*r*exp(-z*r), each already scaled so that the integral of its square over all space is one. Define the overlap of two radial functions as the integral from zero to infinity of four pi r**2 times their product. The inner orbital is the sum of the label-one primitives built from inner_exponents weighted by inner_coefficients, divided by the square root of its own overlap with itself. The outer orbital starts as the single label-two primitive built from outer_exponent, then has the inner orbital subtracted from it once, weighted by the overlap of that starting primitive with the already-scaled inner orbital, and is only then divided by the square root of its own overlap with itself. The inner orbital is never modified, so the order of those two operations is part of the contract. Return an array whose first index selects the orbital with the inner one first, whose second index runs over terms, and whose third index holds the primitive label, the exponent and the coefficient in that order. List the inner orbital's terms in the order of inner_exponents, list the outer orbital's label-two term first and its label-one terms after it in that same order, and pad the inner orbital with trailing rows of all zeros so both orbitals carry the same number of terms.

    Parameters
    ----------
    inner_exponents : np.ndarray
        Positive Slater exponents of the inner orbital's label-one primitives, in inverse bohr.
    inner_coefficients : np.ndarray
        Weights of those primitives, same length as inner_exponents.
    outer_exponent : float
        Positive Slater exponent of the outer orbital's label-two primitive, in inverse bohr.

    Returns
    -------
    table : np.ndarray
        ndarray of shape (2, len(inner_exponents)+1, 3): orbital, term, then primitive label, Slater exponent and coefficient.

    Raises
    ------
    ValueError
        if inner_exponents and inner_coefficients have different lengths or are empty, or if any Slater exponent is not positive.
    '''
    return table  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _sto_norm(principal, exponent):
    """Norm of the radial Slater primitive r**(principal-1) * exp(-exponent r)."""
    if principal == 1:
        return math.sqrt(exponent ** 3 / math.pi)
    return math.sqrt(exponent ** 5 / (3.0 * math.pi))


def _pair_terms(a, b):
    """Collect the product of two orbital tables as {(power, decay): coefficient}."""
    out = {}
    for pa, za, ca in a:
        if ca == 0.0:
            continue
        for pb, zb, cb in b:
            if cb == 0.0:
                continue
            key = (int(round(pa)) + int(round(pb)) - 2, float(za) + float(zb))
            w = ca * cb * _sto_norm(int(round(pa)), float(za)) * _sto_norm(int(round(pb)), float(zb))
            out[key] = out.get(key, 0.0) + w
    return out


def _overlap(a, b):
    """Overlap of two orbital tables, integrated over all space."""
    s = 0.0
    for (power, decay), w in _pair_terms(a, b).items():
        s += w * 4.0 * math.pi * math.factorial(power + 2) / decay ** (power + 3)
    return s


def _oracle_orthonormal_orbitals(inner_exponents: "np.ndarray", inner_coefficients: "np.ndarray", outer_exponent: float) -> "np.ndarray":
    """Normalised inner orbital and the outer orbital orthogonalised against it."""
    ze = np.asarray(inner_exponents, dtype=float).ravel()
    zc = np.asarray(inner_coefficients, dtype=float).ravel()
    if ze.size != zc.size or ze.size == 0:
        raise ValueError("inner_exponents and inner_coefficients must have the same nonzero length")
    if np.any(ze <= 0.0) or float(outer_exponent) <= 0.0:
        raise ValueError("every Slater exponent must be positive")
    nin = ze.size
    inner = [[1.0, ze[i], zc[i]] for i in range(nin)]
    s = math.sqrt(_overlap(inner, inner))
    inner = [[1.0, ze[i], zc[i] / s] for i in range(nin)]
    outer_raw = [[2.0, float(outer_exponent), 1.0]]
    mix = _overlap(outer_raw, inner)
    outer = outer_raw + [[1.0, ze[i], -mix * inner[i][2]] for i in range(nin)]
    t = math.sqrt(_overlap(outer, outer))
    outer = [[row[0], row[1], row[2] / t] for row in outer]
    width = nin + 1
    table = np.zeros((2, width, 3))
    for i, row in enumerate(inner):
        table[0, i] = row
    for i, row in enumerate(outer):
        table[1, i] = row
    return table

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np",
         "call": "orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)",
         "gold_call": "_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)"},   # normal
        {"setup": "import numpy as np",
         "call": "orthonormal_orbitals([3.0],[1.0],0.8)",
         "gold_call": "_oracle_orthonormal_orbitals([3.0],[1.0],0.8)"},   # boundary
        {"setup": "import numpy as np",
         "call": "orthonormal_orbitals([9.1,4.4,2.05],[0.02,0.21,0.83],0.41)",
         "gold_call": "_oracle_orthonormal_orbitals([9.1,4.4,2.05],[0.02,0.21,0.83],0.41)"},   # edge
        {"setup": "import numpy as np\ndef _c():\n    try:\n        orthonormal_orbitals([1.0,2.0],[1.0],0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_orthonormal_orbitals([1.0,2.0],[1.0],0.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]
