"""
Evaluate the five radial fields that one spin channel contributes, at each radius. A row (p, z, w) of an orbital table contributes w times the primitive it names, where label p equal to one means sqrt(z**3/pi)*exp(-z*r) and label p equal to two means sqrt(z**5/(3*pi))*r*exp(-z*r); a row whose coefficient is zero is padding and contributes nothing, whatever its other entries. Write phi_k for the k-th orbital of the table, taken as that sum over its rows, n_k for its occupation and a prime for d/dr. The density is the sum over k of n_k phi_k**2. The gradient is the sum over k of 2 n_k phi_k phi_k'. The Laplacian is the sum over k of 2 n_k times (phi_k (phi_k'' + 2 phi_k'/r) + phi_k'**2). The orbital kinetic density is the sum over k of n_k phi_k'**2, carrying no factor of one half. The fifth field is the model-hole curvature: the Laplacian minus twice (the orbital kinetic density minus a quarter of the gradient squared divided by the density), the whole difference divided by six. Return the five fields as rows in exactly that order.

A model exchange hole that is expanded about its own reference point is fixed by the density and by the curvature of the spherically averaged hole at that point, and that curvature is the combination written above. The kinetic density that enters it is the sum of squared orbital gradients as it appears in that construction, which is twice the quantity usually written as the kinetic energy density.

Returns
-------
ndarray of shape (5, len(radii)): density, gradient, Laplacian, orbital kinetic density and model-hole curvature, in hartree atomic units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spin_channel_fields(orbitals: "np.ndarray", occupations: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    '''Evaluate the five radial fields that one spin channel contributes, at each radius. A row (p, z, w) of an orbital table contributes w times the primitive it names, where label p equal to one means sqrt(z**3/pi)*exp(-z*r) and label p equal to two means sqrt(z**5/(3*pi))*r*exp(-z*r); a row whose coefficient is zero is padding and contributes nothing, whatever its other entries. Write phi_k for the k-th orbital of the table, taken as that sum over its rows, n_k for its occupation and a prime for d/dr. The density is the sum over k of n_k phi_k**2. The gradient is the sum over k of 2 n_k phi_k phi_k'. The Laplacian is the sum over k of 2 n_k times (phi_k (phi_k'' + 2 phi_k'/r) + phi_k'**2). The orbital kinetic density is the sum over k of n_k phi_k'**2, carrying no factor of one half. The fifth field is the model-hole curvature: the Laplacian minus twice (the orbital kinetic density minus a quarter of the gradient squared divided by the density), the whole difference divided by six. Return the five fields as rows in exactly that order.

    Parameters
    ----------
    orbitals : np.ndarray
        Padded coefficient table of shape (n_orbitals, n_terms, 3); each row is primitive label, exponent, coefficient.
    occupations : np.ndarray
        Occupation of each orbital, each between zero and one.
    radii : np.ndarray
        Strictly positive radii in bohr at which to evaluate.

    Returns
    -------
    fields : np.ndarray
        ndarray of shape (5, len(radii)): density, gradient, Laplacian, orbital kinetic density and model-hole curvature, in hartree atomic units.

    Raises
    ------
    ValueError
        if orbitals does not have shape (n_orbitals, n_terms, 3), if orbitals and occupations describe different numbers of orbitals, if any occupation lies outside zero to one, or if any radius is not positive.
    '''
    return fields  # placeholder

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


def _oracle_spin_channel_fields(orbitals: "np.ndarray", occupations: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    """Density, gradient, Laplacian, kinetic-energy density and hole curvature."""
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
    rho = np.zeros_like(r)
    grad = np.zeros_like(r)
    lap = np.zeros_like(r)
    tau = np.zeros_like(r)
    for k in range(orb.shape[0]):
        f0 = _expand(orb[k], r, 0)
        f1 = _expand(orb[k], r, 1)
        f2 = _expand(orb[k], r, 2)
        rho = rho + occ[k] * f0 * f0
        grad = grad + 2.0 * occ[k] * f0 * f1
        lap = lap + 2.0 * occ[k] * (f0 * (f2 + 2.0 * f1 / r) + f1 * f1)
        tau = tau + occ[k] * f1 * f1
    safe = np.where(rho > 0.0, rho, 1.0)
    d = tau - 0.25 * grad * grad / safe
    curv = (lap - 2.0 * d) / 6.0
    return np.vstack((rho, grad, lap, tau, curv))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])",
         "call": "spin_channel_fields(OB, OA, RG)",
         "gold_call": "_oracle_spin_channel_fields(OB, OA, RG)"},   # normal
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])",
         "call": "spin_channel_fields(OB[:1], OCB, RG)",
         "gold_call": "_oracle_spin_channel_fields(OB[:1], OCB, RG)"},   # boundary
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)",
         "call": "spin_channel_fields(OB, np.array([1.0,0.0]), RG)",
         "gold_call": "_oracle_spin_channel_fields(OB, np.array([1.0,0.0]), RG)"},   # edge
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\ndef _exc(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_exc(lambda: spin_channel_fields(OB, np.array([1.0,1.5]), RG))",
         "gold_call": "_exc(lambda: _oracle_spin_channel_fields(OB, np.array([1.0,1.5]), RG))"},   # invalid input
    ]
