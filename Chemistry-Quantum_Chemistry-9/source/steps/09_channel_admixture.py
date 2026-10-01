"""
Return one spin channel's contribution to the exact-exchange admixture energy. It is four pi times the quadrature sum over the supplied nodes of weight times radius squared times the mixing fraction times (the exact exchange energy density minus the semilocal one) at that node. The difference is taken in that order, exact first, and the four pi times radius squared is the spherical measure that turns a radial quadrature into an integral over all space. Return a plain float.

Both energy densities are per unit volume, so the radial rule can only be applied after the angular measure has been folded in; for a spherically symmetric density that measure is the full solid angle. The two energy densities are close to each other and much larger than their difference, so the subtraction has to happen pointwise before the quadrature rather than between two separately integrated energies.

Returns
-------
float: that channel's contribution to the admixture energy, in hartree.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def channel_admixture(mixing: "np.ndarray", exact_density: "np.ndarray", semilocal_density: "np.ndarray", radii: "np.ndarray", weights: "np.ndarray") -> float:
    '''Return one spin channel's contribution to the exact-exchange admixture energy. It is four pi times the quadrature sum over the supplied nodes of weight times radius squared times the mixing fraction times (the exact exchange energy density minus the semilocal one) at that node. The difference is taken in that order, exact first, and the four pi times radius squared is the spherical measure that turns a radial quadrature into an integral over all space. Return a plain float.

    Parameters
    ----------
    mixing : np.ndarray
        Local exact-exchange mixing fraction at each node, each between zero and one.
    exact_density : np.ndarray
        Exact exchange energy density at each node, same length.
    semilocal_density : np.ndarray
        Semilocal exchange energy density at each node, same length.
    radii : np.ndarray
        Strictly positive radii in bohr, same length.
    weights : np.ndarray
        Radial quadrature weights in bohr, same length.

    Returns
    -------
    contribution : float
        float: that channel's contribution to the admixture energy, in hartree.

    Raises
    ------
    ValueError
        if mixing, the two energy densities, radii and weights do not all have the same length, if any radius is not positive, or if any mixing fraction lies outside zero to one.
    '''
    return contribution  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _oracle_channel_admixture(mixing: "np.ndarray", exact_density: "np.ndarray", semilocal_density: "np.ndarray", radii: "np.ndarray", weights: "np.ndarray") -> float:
    """One spin channel's contribution to the exact-exchange admixture energy."""
    g = np.asarray(mixing, dtype=float).ravel()
    ex = np.asarray(exact_density, dtype=float).ravel()
    eb = np.asarray(semilocal_density, dtype=float).ravel()
    r = np.asarray(radii, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if not (g.size == ex.size == eb.size == r.size == w.size):
        raise ValueError("mixing, both energy densities, radii and weights must have the same length")
    if np.any(r <= 0.0):
        raise ValueError("every radius must be positive")
    if np.any(g < 0.0) or np.any(g > 1.0):
        raise ValueError("every mixing fraction must lie between zero and one")
    return 4.0 * math.pi * float(np.sum(w * r ** 2 * g * (ex - eb)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)\nNA=_oracle_effective_hole_normalisation(FA[0],FA[4],XA); NB=_oracle_effective_hole_normalisation(FB[0],FB[4],XB)\nBA=_oracle_b86b_exchange_density(FA[0],FA[1]); BB=_oracle_b86b_exchange_density(FB[0],FB[1])\nGA=_oracle_local_mixing_function(FA[0],BA,_oracle_xc_hole_normalisation(NA,NB),0.10,4.6)\nWG=_oracle_radial_quadrature(RG.size,1.0)[1]",
         "call": "channel_admixture(GA, XA, BA, RG, WG)",
         "gold_call": "_oracle_channel_admixture(GA, XA, BA, RG, WG)"},   # normal
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)\nBA=_oracle_b86b_exchange_density(FA[0],FA[1]); BB=_oracle_b86b_exchange_density(FB[0],FB[1])\nWG=_oracle_radial_quadrature(RG.size,1.0)[1]",
         "call": "channel_admixture(0.25*np.ones(RG.size), XA, BA, RG, WG)",
         "gold_call": "_oracle_channel_admixture(0.25*np.ones(RG.size), XA, BA, RG, WG)"},   # boundary
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)\nBA=_oracle_b86b_exchange_density(FA[0],FA[1]); BB=_oracle_b86b_exchange_density(FB[0],FB[1])\nWG=_oracle_radial_quadrature(RG.size,1.0)[1]",
         "call": "channel_admixture(np.ones(RG.size), XB, BB, RG, WG)",
         "gold_call": "_oracle_channel_admixture(np.ones(RG.size), XB, BB, RG, WG)"},   # edge
        {"setup": "import numpy as np\nRG=np.array([0.10,0.35,0.80,1.50,3.00,6.00])\nOB=_oracle_orthonormal_orbitals([4.70,2.45],[0.15,0.90],0.66)\nOA=np.array([1.0,0.5]); OCB=np.array([1.0])\nFA=_oracle_spin_channel_fields(OB,OA,RG); FB=_oracle_spin_channel_fields(OB[:1],OCB,RG)\nXA=_oracle_exact_exchange_density(OB,OA,RG); XB=_oracle_exact_exchange_density(OB[:1],OCB,RG)\nBA=_oracle_b86b_exchange_density(FA[0],FA[1]); BB=_oracle_b86b_exchange_density(FB[0],FB[1])\nWG=_oracle_radial_quadrature(RG.size,1.0)[1]\ndef _exc(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_exc(lambda: channel_admixture(2.0*np.ones(RG.size), XA, BA, RG, WG))",
         "gold_call": "_exc(lambda: _oracle_channel_admixture(2.0*np.ones(RG.size), XA, BA, RG, WG))"},   # invalid input
    ]
