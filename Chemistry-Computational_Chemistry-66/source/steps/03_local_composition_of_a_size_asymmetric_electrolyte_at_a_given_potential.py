"""
Return the local volume fraction of every ionic species, and the logarithm of the local solvent fraction relative to its reservoir value, at a point of given reduced electrostatic potential.

Several ionic species and a solvent share a cubic lattice of cell volume a**3 that is full everywhere. Species i carries the signed valency z_i, occupies v_i cells per ion and has local volume fraction phi_i, and the solvent fraction is eta = 1 - sum_i phi_i. The free energy per unit volume is the electrostatic energy of the field and the ions together with the Flory-Huggins mixing term

(k_B T / a**3) [ sum_i (phi_i / v_i) ln(phi_i) + eta ln(eta) ],

in which each ion counts as one particle and the solvent counts per cell. The electrolyte is in equilibrium with a reservoir of volume fractions phi_i^b and solvent fraction eta^b, where the potential is zero, so at a point where the reduced potential Psi = e psi / (k_B T) takes a given value the local composition is the one that minimises the free energy with respect to each ion density at that fixed potential.

Return the physically admissible composition, with every ion fraction positive and the solvent fraction between zero and one. Far from a charged wall the potential is tiny and the composition departs from the reservoir only slightly; every returned entry, the logarithm included, must keep ten significant figures there as well, for potentials as small in magnitude as 1e-9, and at zero potential the reservoir composition is returned exactly.

Returns
-------
np.ndarray of length N + 1: the N local volume fractions, then ln(eta / eta^b)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def local_composition(z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", Psi: float) -> "np.ndarray":
    '''Local composition of a size-asymmetric lattice electrolyte at a given reduced potential.

    Parameters
    ----------
    z : np.ndarray
        Real array of length N holding the signed valency of each species.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the cell
        volume; every entry strictly positive.
    phi_bulk : np.ndarray
        Real array of length N holding the reservoir volume fraction of each species; every entry
        strictly positive, with a sum strictly less than one.
    Psi : float
        Reduced electrostatic potential e psi / (k_B T) at the point, measured from the reservoir;
        any sign, with magnitude up to 60.

    Returns
    -------
    result : np.ndarray
        Real array of length N + 1 holding the N local volume fractions in the input species order,
        followed by ln(eta / eta^b), the natural logarithm of the local solvent fraction divided by
        the reservoir solvent fraction.

    Raises
    ------
    ValueError
        If the three arrays differ in length, if any relative volume or reservoir fraction is not
        strictly positive, or if the reservoir fractions sum to one or more.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _oracle_local_composition(z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", Psi: float) -> "np.ndarray":
    zz = np.atleast_1d(np.asarray(z, dtype=float))
    vv = np.atleast_1d(np.asarray(v, dtype=float))
    pb = np.atleast_1d(np.asarray(phi_bulk, dtype=float))
    if not (zz.size == vv.size == pb.size):
        raise ValueError("z, v and phi_bulk must have the same length")
    if np.any(vv <= 0.0) or np.any(pb <= 0.0):
        raise ValueError("relative volumes and reservoir fractions must be strictly positive")
    eb = 1.0 - float(pb.sum())
    if eb <= 0.0:
        raise ValueError("reservoir fractions must sum to less than one")
    Psi = float(Psi)
    if Psi == 0.0:
        return np.concatenate([pb, [0.0]])
    # phi_i = phi_i^b exp(-z_i Psi + v_i d) with d = ln(eta / eta^b); filling the lattice leaves one
    # increasing convex equation in d, written with expm1 so the far field keeps its digits
    f = lambda d: eb * np.expm1(d) + float((pb * np.expm1(-zz * Psi + vv * d)).sum())
    hi = -np.log(eb)
    lo = -1e-12
    while f(lo) > 0.0:
        lo = 4.0 * lo - 1.0
    d = brentq(f, lo, hi, xtol=1e-300, rtol=4.0 * np.finfo(float).eps, maxiter=1000)
    return np.concatenate([pb * np.exp(-zz * Psi + vv * d), [d]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    task = """import numpy as np
Z = np.array([-1.0, -2.0, -3.0, 1.0])
V = np.array([0.5, 2.0, 8.0, 1.0])
PB = np.array([2.0e-3, 1.0e-3, 5.0e-4, 5.1875e-3])
def _positive_log_state(value):
    value = np.asarray(value, dtype=float)
    fractions = value[:-1]
    flags = (fractions > 0.0).astype(float)
    safe = np.maximum(fractions, np.finfo(float).tiny)
    return np.concatenate((flags, np.log(safe), value[-1:]))
def _signed_log_response(value, base, potential):
    response = (np.asarray(value, dtype=float) - np.asarray(base, dtype=float)) / potential
    flags = (response != 0.0).astype(float)
    signs = np.sign(response)
    safe = np.maximum(np.abs(response), np.finfo(float).tiny)
    return np.concatenate((flags, signs, np.log(safe)))
"""
    return [
        # normal, compare ion fractions on a logarithmic scale so every species carries relative weight
        {
            "setup": task,
            "call": "_positive_log_state(local_composition(Z, V, PB, 1.0))",
            "gold_call": "_positive_log_state(_oracle_local_composition(Z, V, PB, 1.0))",
            "tol": 1e-11,
        },
        # normal, saturated wall-like state; the positivity flags reject zero or negative trace fractions
        {
            "setup": task,
            "call": "_positive_log_state(local_composition(Z, V, PB, 8.5))",
            "gold_call": "_positive_log_state(_oracle_local_composition(Z, V, PB, 8.5))",
            "tol": 1e-11,
        },
        # boundary, zero potential returns the reservoir composition exactly
        {
            "setup": task,
            "call": "local_composition(Z, V, PB, 0.0)",
            "gold_call": "_oracle_local_composition(Z, V, PB, 0.0)",
            "tol": 1e-12,
        },
        # edge, scale the full departure from the reservoir at Psi=1e-9 before taking signed logs
        {
            "setup": task,
            "call": "_signed_log_response(local_composition(Z, V, PB, 1e-9), np.r_[PB, 0.0], 1e-9)",
            "gold_call": "_signed_log_response(_oracle_local_composition(Z, V, PB, 1e-9), np.r_[PB, 0.0], 1e-9)",
            "tol": 1e-11,
        },
        # normal, negative potential with strict positivity retained for depleted anions
        {
            "setup": task,
            "call": "_positive_log_state(local_composition(Z, V, PB, -6.0))",
            "gold_call": "_positive_log_state(_oracle_local_composition(Z, V, PB, -6.0))",
            "tol": 1e-11,
        },
        # edge, very bulky counterion and an exponentially depleted co-ion
        {
            "setup": """import numpy as np
Z = np.array([-2.0, 1.0])
V = np.array([30.0, 0.25])
PB = np.array([0.03, 5.0e-4])
def _positive_log_state(value):
    value = np.asarray(value, dtype=float)
    fractions = value[:-1]
    flags = (fractions > 0.0).astype(float)
    safe = np.maximum(fractions, np.finfo(float).tiny)
    return np.concatenate((flags, np.log(safe), value[-1:]))
""",
            "call": "_positive_log_state(local_composition(Z, V, PB, 25.0))",
            "gold_call": "_positive_log_state(_oracle_local_composition(Z, V, PB, 25.0))",
            "tol": 1e-11,
        },
        # contract, an overfull reservoir must raise ValueError
        {
            "setup": "import numpy as np\ndef _raises(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1.0\n    return 0.0",
            "call": "_raises(lambda: local_composition(np.array([-1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.7, 0.7]), 1.0))",
            "gold_call": "_raises(lambda: _oracle_local_composition(np.array([-1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.7, 0.7]), 1.0))",
        },
    ]
