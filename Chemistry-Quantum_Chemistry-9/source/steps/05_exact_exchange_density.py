"""
Return the exact exchange energy density of one spin channel, in the gauge that takes the integrand as it stands with no function added to it. A row (p, z, w) of an orbital table contributes w times the primitive it names, where label p equal to one means sqrt(z**3/pi)*exp(-z*r) and label p equal to two means sqrt(z**5/(3*pi))*r*exp(-z*r); a row whose coefficient is zero is padding and contributes nothing, whatever its other entries. The value at radius r is minus one half of the sum over every ordered pair (i, j) of orbitals of n_i n_j phi_i(r) phi_j(r) times the electrostatic potential at r of the product density phi_i phi_j. Fractional occupations enter as the plain product n_i n_j and not as its square root, which is what makes a single fractionally occupied orbital cancel its own classical repulsion exactly. Every product of two of these orbitals is spherically symmetric, and the potential at radius r of a spherical density f is four pi times the quantity (the integral from zero to r of s**2 f(s) ds, divided by r, plus the integral from r to infinity of s f(s) ds).

Written this way the exchange energy density integrates to the Hartree-Fock exchange energy of the channel, and for one orbital it is exactly minus one half of the density times the potential that orbital's own charge creates. Because every radial product here is a power of r times a decaying exponential, both radial integrals close in elementary functions and no numerical inner quadrature is needed.

Returns
-------
ndarray of shape (len(radii),): the exact exchange energy density of that spin channel, in hartree per bohr cubed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def exact_exchange_density(orbitals: "np.ndarray", occupations: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    '''Return the exact exchange energy density of one spin channel, in the gauge that takes the integrand as it stands with no function added to it. A row (p, z, w) of an orbital table contributes w times the primitive it names, where label p equal to one means sqrt(z**3/pi)*exp(-z*r) and label p equal to two means sqrt(z**5/(3*pi))*r*exp(-z*r); a row whose coefficient is zero is padding and contributes nothing, whatever its other entries. The value at radius r is minus one half of the sum over every ordered pair (i, j) of orbitals of n_i n_j phi_i(r) phi_j(r) times the electrostatic potential at r of the product density phi_i phi_j. Fractional occupations enter as the plain product n_i n_j and not as its square root, which is what makes a single fractionally occupied orbital cancel its own classical repulsion exactly. Every product of two of these orbitals is spherically symmetric, and the potential at radius r of a spherical density f is four pi times the quantity (the integral from zero to r of s**2 f(s) ds, divided by r, plus the integral from r to infinity of s f(s) ds).

    Parameters
    ----------
    orbitals : np.ndarray
        Padded coefficient table of shape (n_orbitals, n_terms, 3), as returned by the orbital step.
    occupations : np.ndarray
        Occupation of each orbital, each between zero and one.
    radii : np.ndarray
        Strictly positive radii in bohr at which to evaluate.

    Returns
    -------
    energy_density : np.ndarray
        ndarray of shape (len(radii),): the exact exchange energy density of that spin channel, in hartree per bohr cubed.

    Raises
    ------
    ValueError
        if orbitals does not have shape (n_orbitals, n_terms, 3), if orbitals and occupations describe different numbers of orbitals, if any occupation lies outside zero to one, or if any radius is not positive.
    '''
    return energy_density  # placeholder

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


def _sto_eval(principal, exponent, r, order):
    """Value (order 0), d/dr (order 1) or d2/dr2 (order 2) of a normalised primitive."""
    c = _sto_norm(principal, exponent)
    e = np.exp(-exponent * r)
    if principal == 1:
        if order == 0:
            return c * e
        if order == 1:
            return -exponent * c * e
        return exponent * exponent * c * e
    if order == 0:
        return c * r * e
    if order == 1:
        return c * (1.0 - exponent * r) * e
    return c * (exponent * exponent * r - 2.0 * exponent) * e


def _expand(orbital, r, order):
    """Evaluate one padded (nterm, 3) orbital table on r."""
    out = np.zeros_like(r, dtype=float)
    for principal, exponent, coef in orbital:
        if coef == 0.0:
            continue
        out = out + coef * _sto_eval(int(round(principal)), float(exponent), r, order)
    return out


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


def _gamma_lo(n, x):
    """Lower incomplete gamma P(n+1, x) for integer n, by its finite exponential sum."""
    s = np.zeros_like(x)
    term = np.ones_like(x)
    for m in range(n + 1):
        s = s + term
        term = term * x / (m + 1)
    return 1.0 - np.exp(-x) * s


def _gamma_hi(n, x):
    s = np.zeros_like(x)
    term = np.ones_like(x)
    for m in range(n + 1):
        s = s + term
        term = term * x / (m + 1)
    return np.exp(-x) * s


def _radial_coulomb(power, decay, r):
    """Potential at radius r of the spherical density s**power * exp(-decay s)."""
    x = decay * r
    lo = math.factorial(power + 2) / decay ** (power + 3) * _gamma_lo(power + 2, x)
    hi = math.factorial(power + 1) / decay ** (power + 2) * _gamma_hi(power + 1, x)
    return 4.0 * math.pi * (lo / r + hi)


def _oracle_exact_exchange_density(orbitals: "np.ndarray", occupations: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    """Exact (Hartree-Fock) exchange energy density of one spin channel."""
    orb = np.asarray(orbitals, dtype=float)
    occ = np.asarray(occupations, dtype=float).ravel()
    r = np.asarray(radii, dtype=float).ravel()
    if orb.ndim != 3 or orb.shape[2] != 3:
        raise ValueError("orbitals must have shape (n_orbitals, n_terms, 3)")
    if orb.shape[0] != occ.size:
        raise ValueError("orbitals and occupations must describe the same number of orbitals")
    if np.any(occ < 0.0) or np.any(occ > 1.0):
        raise ValueError("each occupation must lie between zero and one")
    if np.any(r <= 0.0):
        raise ValueError("every radius must be positive")
    total = np.zeros_like(r)
    for i in range(orb.shape[0]):
        fi = _expand(orb[i], r, 0)
        for j in range(orb.shape[0]):
            fj = _expand(orb[j], r, 0)
            pot = np.zeros_like(r)
            for (power, decay), w in _pair_terms(orb[i], orb[j]).items():
                pot = pot + w * _radial_coulomb(power, decay, r)
            total = total + occ[i] * occ[j] * fi * fj * pot
    return -0.5 * total

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])",
         "call": "exact_exchange_density(OB, OA, RG)",
         "gold_call": "_oracle_exact_exchange_density(OB, OA, RG)"},   # normal
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])",
         "call": "exact_exchange_density(OB[:1], OCB, RG)",
         "gold_call": "_oracle_exact_exchange_density(OB[:1], OCB, RG)"},   # boundary
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)",
         "call": "exact_exchange_density(OB, np.array([1.0,1.0]), RG)",
         "gold_call": "_oracle_exact_exchange_density(OB, np.array([1.0,1.0]), RG)"},   # edge
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\ndef _exc(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_exc(lambda: exact_exchange_density(OB, OA, np.array([0.0,1.0])))",
         "gold_call": "_exc(lambda: _oracle_exact_exchange_density(OB, OA, np.array([0.0,1.0])))"},   # invalid input
    ]
