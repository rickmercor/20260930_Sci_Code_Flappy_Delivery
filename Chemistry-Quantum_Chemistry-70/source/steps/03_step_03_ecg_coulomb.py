"""
Step 03: Coulomb potential-energy matrix. Coulomb potential-energy matrix, nuclear attraction plus electron repulsion, between antisymmetrized L = 1 correlated Gaussians.

For a nucleus of charge Z held fixed at the origin the potential energy of the two electrons is V = -Z/r1 - Z/r2 + 1/r12 in hartree atomic units. Each term is the inverse of the length of one linear combination of the electron coordinates, and the repulsion is what makes the two-electron problem non-separable and the correlation factor in the basis worth carrying.

The basis functions are the same unnormalized correlated Gaussians of definite parity used for the overlap and the kinetic energy.

Returns
-------
numpy.ndarray of shape (n, m): Coulomb matrix -Z/r1 - Z/r2 + 1/r12 between the same functions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ecg_coulomb(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str, Z: float) -> "np.ndarray":
    '''Coulomb matrix -Z/r1 - Z/r2 + 1/r12 of antisymmetrized L = 1 correlated Gaussians.

    Parameters
    ----------
    exps_bra : np.ndarray
        Array of shape (n, 3), n >= 1, of exponents (a, b, c) in bohr^-2; finite, non-negative, a b + c (a + b) > 0.
    exps_ket : np.ndarray
        Array of shape (m, 3), m >= 1, with the same rules.
    kind : str
        "even" for phi = (1 - P12)[(r1_vec x r2_vec)_z g] or "odd" for phi = (1 - P12)[z1 g], with
        g = exp(-a r1^2 - b r2^2 - c r12^2).
    Z : float
        Nuclear charge in units of the proton charge, Z > 0.

    Returns
    -------
    result : np.ndarray
        Real array of shape (n, m) with entries V_ij = integral of phi_i (-Z/r1 - Z/r2 + 1/r12) phi_j over both
        electrons, in hartree times the overlap units of the unnormalized functions.

    Raises
    ------
    ValueError
        If an exponent array is malformed or non-integrable, if kind is not "even" or "odd", or if Z is not positive.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _one_coulomb(k, kind, swap, charge_z):
    sig = k["S"]
    total = 0.0
    for q, charge in (((1.0, 0.0), -charge_z), ((0.0, 1.0), -charge_z), ((1.0, -1.0), 1.0)):
        beta = _quad(q, sig, q)
        prefactor = charge * (2.0 / np.sqrt(np.pi)) * k["N"] / np.sqrt(2.0 * beta)
        if kind == "even":
            total = total + (-1.0 if swap else 1.0) * prefactor * (4.0 / 3.0) * k["dS"]
        else:
            v, w = _odd_vectors(swap)
            sq = (sig[0] * q[0] + sig[1] * q[1], sig[1] * q[0] + sig[2] * q[1])
            m0 = _quad(v, sig, w)
            m1 = (v[0] * sq[0] + v[1] * sq[1]) * (w[0] * sq[0] + w[1] * sq[1]) / beta
            total = total + prefactor * (m0 - m1 / 3.0)
    return total


def _oracle_ecg_coulomb(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str, Z: float) -> "np.ndarray":
    charge_z = float(Z)
    if not charge_z > 0.0:
        raise ValueError("Z must be positive")
    eb, ek = _ecg_exponents(exps_bra), _ecg_exponents(exps_ket)
    direct = _one_coulomb(_ecg_blocks(eb, ek, kind, False), kind, False, charge_z)
    exchanged = _one_coulomb(_ecg_blocks(eb, ek, kind, True), kind, True, charge_z)
    return np.asarray(2.0 * (direct - exchanged), dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
bra = np.array([[0.8, 0.3, 0.0], [0.5, 0.5, 0.2], [1.6, 0.1, 0.05]])
ket = np.array([[0.3, 0.9, 0.1], [2.0, 0.4, 0.0], [0.7, 0.7, 0.6], [0.2, 1.1, 0.3]])
"""
    return [
        # --- Normal: helium nucleus, unnatural parity ---
        {"setup": base, "call": "ecg_coulomb(bra.copy(), ket.copy(), 'even', 2.0)",
         "gold_call": "_oracle_ecg_coulomb(bra.copy(), ket.copy(), 'even', 2.0)", "tol": 1e-9},
        # --- Normal: helium nucleus, natural parity ---
        {"setup": base, "call": "ecg_coulomb(bra.copy(), ket.copy(), 'odd', 2.0)",
         "gold_call": "_oracle_ecg_coulomb(bra.copy(), ket.copy(), 'odd', 2.0)", "tol": 1e-9},
        # --- Boundary: Z = 1 (H-), where the attraction and the repulsion nearly cancel for diffuse pairs ---
        {"setup": "import numpy as np\ne = np.array([[0.05, 0.05, 0.0], [0.02, 0.3, 0.1], [0.3, 0.02, 0.0]])\n",
         "call": "ecg_coulomb(e.copy(), e.copy(), 'odd', 1.0)", "gold_call": "_oracle_ecg_coulomb(e.copy(), e.copy(), 'odd', 1.0)", "tol": 1e-9},
        # --- Edge: highly charged nucleus with a pure r12 Gaussian in the ket ---
        {"setup": "import numpy as np\nb = np.array([[2.5, 1.5, 0.0], [0.4, 0.6, 0.2]])\nk = np.array([[0.0, 0.8, 1.2]])\n",
         "call": "ecg_coulomb(b.copy(), k.copy(), 'even', 10.0)", "gold_call": "_oracle_ecg_coulomb(b.copy(), k.copy(), 'even', 10.0)", "tol": 1e-9},
        # --- Error: non-positive nuclear charge ---
        {"setup": "import numpy as np\ne = np.array([[1.0, 0.5, 0.1]])\n"
                  "def _probe(fn):\n    try:\n        fn(e, e, 'odd', 0.0)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(ecg_coulomb)", "gold_call": "_probe(_oracle_ecg_coulomb)"},
    ]
