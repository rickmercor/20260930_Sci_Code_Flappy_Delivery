"""
Step 01: Overlap matrix of antisymmetrized L = 1 correlated Gaussians. Overlap matrix between spatially antisymmetric two-electron explicitly correlated Gaussian functions of total orbital angular momentum one, for either parity.

Each basis function carries an exponent triple (a, b, c) and the Gaussian g(r1_vec, r2_vec) = exp(-a r1^2 - b r2^2 - c r12^2), with r1_vec and r2_vec the electron positions measured from an infinitely heavy nucleus and r12 = |r1_vec - r2_vec|. The functions are not normalized, and the same triple is used throughout the calculation.

An angular factor linear in each vector it contains fixes the total orbital angular momentum and the parity. The z component of r1_vec x r2_vec gives a 1^+ function, even under inversion and therefore of unnatural parity, as in the 2p^2 configuration; the z coordinate of electron 1 gives a 1^- function, as in the 1snp and 1s ep configurations. A spin triplet needs a spatial wavefunction that changes sign when the electrons are exchanged, so each primitive is acted on by 1 - P12, where P12 swaps r1_vec and r2_vec.

Gaussian expansions of this kind are the working tool for few-body Coulomb systems. Exponents laid out in geometric progressions cover compact bound states, extended Rydberg states and a square-integrable discretization of the continuum on the same footing, and sets spanning several decades are common.

Returns
-------
numpy.ndarray of shape (n, m): overlap matrix of the antisymmetrized L = 1 correlated Gaussians of the chosen parity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ecg_overlap(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str) -> "np.ndarray":
    '''Overlap matrix of antisymmetrized L = 1 correlated Gaussians.

    Parameters
    ----------
    exps_bra : np.ndarray
        Array of shape (n, 3), n >= 1. Row i holds the exponents (a, b, c) of bra function i, in bohr^-2.
        Every exponent must be finite and non-negative, with a b + c (a + b) > 0.
    exps_ket : np.ndarray
        Array of shape (m, 3), m >= 1, with the same rules, for the ket functions.
    kind : str
        "even" for phi = (1 - P12)[(r1_vec x r2_vec)_z g] (total L = 1, even parity) or
        "odd" for phi = (1 - P12)[z1 g] (total L = 1, odd parity), with
        g = exp(-a r1^2 - b r2^2 - c r12^2) and P12 the exchange of the two electron positions.

    Returns
    -------
    result : np.ndarray
        Real array of shape (n, m) with entries S_ij = integral of phi_i phi_j over r1_vec and r2_vec
        (unnormalized functions, bohr units).

    Raises
    ------
    ValueError
        If an exponent array has the wrong shape, holds a negative or non-finite exponent, gives a
        non-integrable Gaussian (a b + c (a + b) <= 0), or if kind is not "even" or "odd".
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _ecg_exponents(exps):
    """Validate an (n, 3) exponent array and return it as floats."""
    e = np.asarray(exps, dtype=float)
    if e.ndim != 2 or e.shape[1] != 3 or e.shape[0] < 1:
        raise ValueError("exponents must have shape (n, 3) with n >= 1")
    if not np.all(np.isfinite(e)) or np.any(e < 0.0):
        raise ValueError("exponents must be finite and non-negative")
    if np.any(e[:, 0] * e[:, 1] + e[:, 2] * (e[:, 0] + e[:, 1]) <= 0.0):
        raise ValueError("every Gaussian must be square integrable")
    return e


def _ecg_blocks(eb, ek, kind, swap):
    """Pair quantities for bra exponents eb and ket exponents ek; swap exchanges the ket electrons."""
    if kind not in ("even", "odd"):
        raise ValueError("kind must be 'even' or 'odd'")
    if swap:
        ek = ek[:, [1, 0, 2]]
    ab, bb, cb = eb[:, 0][:, None], eb[:, 1][:, None], eb[:, 2][:, None]
    ak, bk, ck = ek[:, 0][None, :], ek[:, 1][None, :], ek[:, 2][None, :]
    a_mat = (2.0 * (ab + cb), -2.0 * cb, 2.0 * (bb + cb))
    b_mat = (2.0 * (ak + ck), -2.0 * ck, 2.0 * (bk + ck))
    c00, c01, c11 = a_mat[0] + b_mat[0], a_mat[1] + b_mat[1], a_mat[2] + b_mat[2]
    det = c00 * c11 - c01 ** 2
    sig = (c11 / det, -c01 / det, c00 / det)
    return {"A": a_mat, "B": b_mat, "S": sig, "N": (2.0 * np.pi) ** 3 * det ** -1.5,
            "dS": sig[0] * sig[2] - sig[1] ** 2}


def _odd_vectors(swap):
    """Coefficients of (r1, r2) in the z-coordinate prefactor of the bra and of the (possibly swapped) ket."""
    return (1.0, 0.0), ((0.0, 1.0) if swap else (1.0, 0.0))


def _quad(v, m, w):
    """v^T M w for a symmetric 2 x 2 array M stored as (M00, M01, M11)."""
    return v[0] * (m[0] * w[0] + m[1] * w[1]) + v[1] * (m[1] * w[0] + m[2] * w[1])


def _one_overlap(k, kind, swap):
    if kind == "even":
        return (-1.0 if swap else 1.0) * k["N"] * 2.0 * k["dS"]
    v, w = _odd_vectors(swap)
    return k["N"] * _quad(v, k["S"], w)


def _oracle_ecg_overlap(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str) -> "np.ndarray":
    eb, ek = _ecg_exponents(exps_bra), _ecg_exponents(exps_ket)
    direct = _one_overlap(_ecg_blocks(eb, ek, kind, False), kind, False)
    exchanged = _one_overlap(_ecg_blocks(eb, ek, kind, True), kind, True)
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
        # --- Normal: rectangular overlap, unnatural parity ---
        {"setup": base, "call": "ecg_overlap(bra.copy(), ket.copy(), 'even')",
         "gold_call": "_oracle_ecg_overlap(bra.copy(), ket.copy(), 'even')", "tol": 1e-9},
        # --- Normal: rectangular overlap, natural parity ---
        {"setup": base, "call": "ecg_overlap(bra.copy(), ket.copy(), 'odd')",
         "gold_call": "_oracle_ecg_overlap(bra.copy(), ket.copy(), 'odd')", "tol": 1e-9},
        # --- Boundary: a = b with c = 0; the odd function survives antisymmetrization, the square matrix is symmetric ---
        {"setup": "import numpy as np\ne = np.array([[1.0, 1.0, 0.0], [0.25, 0.25, 0.0], [0.4, 3.0, 0.0]])\n",
         "call": "ecg_overlap(e.copy(), e.copy(), 'odd')", "gold_call": "_oracle_ecg_overlap(e.copy(), e.copy(), 'odd')", "tol": 1e-9},
        # --- Edge: only the r12 Gaussian together with one vanishing orbital exponent ---
        {"setup": "import numpy as np\nb = np.array([[0.0, 0.6, 0.9]])\nk = np.array([[0.4, 0.0, 1.5], [0.05, 0.02, 0.01]])\n",
         "call": "ecg_overlap(b.copy(), k.copy(), 'even')", "gold_call": "_oracle_ecg_overlap(b.copy(), k.copy(), 'even')", "tol": 1e-9},
        # --- Error: non-integrable Gaussian (a = 0 and c = 0) ---
        {"setup": "import numpy as np\ne = np.array([[0.0, 1.0, 0.0]])\n"
                  "def _probe(fn):\n    try:\n        fn(e, e, 'odd')\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(ecg_overlap)", "gold_call": "_probe(_oracle_ecg_overlap)"},
        # --- Error: unknown kind ---
        {"setup": "import numpy as np\ne = np.array([[1.0, 1.0, 0.1]])\n"
                  "def _probe(fn):\n    try:\n        fn(e, e, 'singlet')\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(ecg_overlap)", "gold_call": "_probe(_oracle_ecg_overlap)"},
    ]
