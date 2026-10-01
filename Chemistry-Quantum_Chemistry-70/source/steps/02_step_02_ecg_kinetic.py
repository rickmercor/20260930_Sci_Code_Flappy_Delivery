"""
Step 02: Kinetic-energy matrix. Kinetic-energy matrix between the antisymmetrized L = 1 correlated Gaussians of either parity, for two electrons around an infinitely heavy nucleus.

With the nucleus held fixed there is no mass-polarization term, and the kinetic energy of the two electrons is T = -(1/2) (nabla_1^2 + nabla_2^2) in hartree atomic units. The basis functions are the same unnormalized correlated Gaussians of definite parity used for the overlap, and they vanish at infinity.

Together with the Coulomb matrix this is one of the two pieces of the Hamiltonian that every variational and complex-rotated calculation on this basis needs.

Returns
-------
numpy.ndarray of shape (n, m): kinetic-energy matrix -(nabla_1^2 + nabla_2^2)/2 between the same functions
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ecg_kinetic(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str) -> "np.ndarray":
    '''Kinetic-energy matrix of antisymmetrized L = 1 correlated Gaussians.

    Parameters
    ----------
    exps_bra : np.ndarray
        Array of shape (n, 3), n >= 1, of exponents (a, b, c) in bohr^-2 for the bra functions; finite, non-negative,
        with a b + c (a + b) > 0.
    exps_ket : np.ndarray
        Array of shape (m, 3), m >= 1, with the same rules, for the ket functions.
    kind : str
        "even" for phi = (1 - P12)[(r1_vec x r2_vec)_z g] or "odd" for phi = (1 - P12)[z1 g], with
        g = exp(-a r1^2 - b r2^2 - c r12^2) and P12 the exchange of the two electron positions.

    Returns
    -------
    result : np.ndarray
        Real array of shape (n, m) with entries T_ij = integral of phi_i [-(nabla_1^2 + nabla_2^2)/2] phi_j over
        r1_vec and r2_vec, in hartree times the overlap units of the unnormalized functions.

    Raises
    ------
    ValueError
        If an exponent array is malformed or non-integrable, or if kind is not "even" or "odd".
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _sym_times(x, y):
    """Product X Y of two symmetric 2 x 2 arrays (00, 01, 11), returned as a general (00, 01, 10, 11) array."""
    return (x[0] * y[0] + x[1] * y[1], x[0] * y[1] + x[1] * y[2],
            x[1] * y[0] + x[2] * y[1], x[1] * y[1] + x[2] * y[2])


def _gen_times_sym(f, y):
    """Product F Y of a general (00, 01, 10, 11) array and a symmetric (00, 01, 11) array."""
    return (f[0] * y[0] + f[1] * y[1], f[0] * y[1] + f[1] * y[2],
            f[2] * y[0] + f[3] * y[1], f[2] * y[1] + f[3] * y[2])


def _gen_quad(v, f, w):
    """v^T F w for a general 2 x 2 array F stored as (00, 01, 10, 11)."""
    return v[0] * (f[0] * w[0] + f[1] * w[1]) + v[1] * (f[2] * w[0] + f[3] * w[1])


def _one_kinetic(k, kind, swap):
    a_mat, b_mat, sig = k["A"], k["B"], k["S"]
    asb = _gen_times_sym(_sym_times(a_mat, sig), b_mat)
    trace_asb = asb[0] + asb[3]
    if kind == "even":
        value = (5.0 * k["dS"] * trace_asb - k["dS"] * (a_mat[0] + a_mat[2] + b_mat[0] + b_mat[2])
                 + sig[0] + sig[2])
        return (-1.0 if swap else 1.0) * k["N"] * value
    v, w = _odd_vectors(swap)
    sa, sb, bs = _sym_times(sig, a_mat), _sym_times(sig, b_mat), _sym_times(b_mat, sig)
    sabs = _gen_times_sym(_gen_times_sym(sa, b_mat), sig)
    sbas = _gen_times_sym(_gen_times_sym(sb, a_mat), sig)
    value = (1.5 * trace_asb * _quad(v, sig, w) + 0.5 * _gen_quad(v, sabs, w) + 0.5 * _gen_quad(v, sbas, w)
             + 0.5 * (v[0] * w[0] + v[1] * w[1]) - 0.5 * _gen_quad(v, sa, w) - 0.5 * _gen_quad(v, bs, w))
    return k["N"] * value


def _oracle_ecg_kinetic(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str) -> "np.ndarray":
    eb, ek = _ecg_exponents(exps_bra), _ecg_exponents(exps_ket)
    direct = _one_kinetic(_ecg_blocks(eb, ek, kind, False), kind, False)
    exchanged = _one_kinetic(_ecg_blocks(eb, ek, kind, True), kind, True)
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
        # --- Normal: rectangular kinetic matrix, unnatural parity ---
        {"setup": base, "call": "ecg_kinetic(bra.copy(), ket.copy(), 'even')",
         "gold_call": "_oracle_ecg_kinetic(bra.copy(), ket.copy(), 'even')", "tol": 1e-9},
        # --- Normal: rectangular kinetic matrix, natural parity ---
        {"setup": base, "call": "ecg_kinetic(bra.copy(), ket.copy(), 'odd')",
         "gold_call": "_oracle_ecg_kinetic(bra.copy(), ket.copy(), 'odd')", "tol": 1e-9},
        # --- Boundary: uncorrelated functions (c = 0), square and symmetric ---
        {"setup": "import numpy as np\ne = np.array([[4.0, 0.25, 0.0], [1.0, 1.0, 0.0], [0.25, 4.0, 0.0]])\n",
         "call": "ecg_kinetic(e.copy(), e.copy(), 'even')", "gold_call": "_oracle_ecg_kinetic(e.copy(), e.copy(), 'even')", "tol": 1e-9},
        # --- Edge: strongly correlated diffuse pair with a very compact partner ---
        {"setup": "import numpy as np\nb = np.array([[0.01, 0.02, 3.0]])\nk = np.array([[30.0, 0.05, 0.0], [0.0, 0.3, 0.4]])\n",
         "call": "ecg_kinetic(b.copy(), k.copy(), 'odd')", "gold_call": "_oracle_ecg_kinetic(b.copy(), k.copy(), 'odd')", "tol": 1e-9},
        # --- Error: negative exponent ---
        {"setup": "import numpy as np\ne = np.array([[1.0, -0.1, 0.5]])\n"
                  "def _probe(fn):\n    try:\n        fn(e, e, 'even')\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(ecg_kinetic)", "gold_call": "_probe(_oracle_ecg_kinetic)"},
    ]
