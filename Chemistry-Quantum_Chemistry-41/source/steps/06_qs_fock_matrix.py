"""
Implement qs_fock_matrix, which assembles, in the atomic-orbital basis, the effective
one-particle matrix of one quasiparticle-self-consistency cycle of the renormalized
second-order Green's-function method for given orbitals and orbital energies.

Atomic-orbital matrices refer to the real, non-orthogonal basis of compute_ao_integrals and
orbital quantities to the columns of C; all energies are in hartree.

Returns
-------
np.ndarray of shape (n_bf, n_bf): symmetric atomic-orbital matrix of the closed-shell Fock operator of the current orbitals plus the renormalized static self-energy of srg_self_energy, in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def qs_fock_matrix(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", coefficients: "np.ndarray", orbital_energies: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    '''Atomic-orbital matrix of the Fock operator plus the renormalized static self-energy.

    Parameters
    ----------
    overlap : np.ndarray
        Atomic-orbital overlap matrix S, shape (n_bf, n_bf).
    core_hamiltonian : np.ndarray
        Core Hamiltonian h, shape (n_bf, n_bf), in hartree.
    eri : np.ndarray
        Atomic-orbital electron-repulsion integrals (mu nu|lambda sigma) in chemists'
        notation, shape (n_bf, n_bf, n_bf, n_bf), in hartree.
    coefficients : np.ndarray
        Current orbital coefficients C as columns, shape (n_bf, n_bf), with C^T S C = 1; the
        first n_occ columns are the doubly occupied orbitals.
    orbital_energies : np.ndarray
        Current orbital energies, shape (n_bf,), in hartree, in the column order of C.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= n_bf - 1.
    s, c_ss, c_os : float
        Flow parameter (hartree^-2) and same-/opposite-spin scaling factors, as in
        srg_self_energy.

    Returns
    -------
    fock : np.ndarray
        Symmetric matrix of shape (n_bf, n_bf), in hartree:
        the atomic-orbital representation of the closed-shell Fock operator of the determinant
        in which the first n_occ columns of C are doubly occupied, plus that of the operator
        whose matrix in the orbitals C is sigma, the result of srg_self_energy for the
        integrals transformed to the orbitals C and the orbital energies given. The
        atomic-orbital representation of an operator is the matrix of its elements between
        atomic orbitals, so that C^T fock C is the matrix of the same operator in the orbitals
        C.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, or n_occ, s, c_ss or c_os is invalid as in
        srg_self_energy.
    '''
    return fock

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_qs_fock_matrix(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", coefficients: "np.ndarray", orbital_energies: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    overlap = np.asarray(overlap, dtype=float)
    core_hamiltonian = np.asarray(core_hamiltonian, dtype=float)
    eri = np.asarray(eri, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    n_bf = overlap.shape[0]
    if (overlap.shape != (n_bf, n_bf) or core_hamiltonian.shape != (n_bf, n_bf)
            or eri.shape != (n_bf,) * 4 or coefficients.shape != (n_bf, n_bf)):
        raise ValueError("inconsistent matrix shapes")
    if int(n_occ) != n_occ or not 1 <= n_occ <= n_bf - 1:
        raise ValueError("n_occ must be an integer between 1 and n_bf - 1")
    n_occ = int(n_occ)
    density = 2.0 * coefficients[:, :n_occ] @ coefficients[:, :n_occ].T
    coulomb = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    exchange = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, coefficients, coefficients, coefficients, coefficients, optimize=True)
    sigma = _oracle_srg_self_energy(eri_mo, orbital_energies, n_occ, s, c_ss, c_os)
    sc = overlap @ coefficients
    fock = core_hamiltonian + coulomb - 0.5 * exchange + sc @ sigma @ sc.T
    return 0.5 * (fock + fock.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    helper = """import numpy as np
from scipy.linalg import eigh
from scipy.special import erf
def s_integrals(charges, coords, shells):
    # closed-form integrals over contracted s Gaussians; coefficients multiply normalized primitives
    charges = np.asarray(charges, float); coords = np.asarray(coords, float)
    fns = []
    for atom, exps, coefs in shells:
        a = np.asarray(exps, float); c = np.asarray(coefs, float) * (2 * a / np.pi) ** 0.75
        s = np.sum(c[:, None] * c[None, :] * (np.pi / (a[:, None] + a[None, :])) ** 1.5)
        fns.append((coords[atom], a, c / np.sqrt(s)))
    def f0(t):
        t = np.asarray(t, float)
        return np.where(t < 1e-12, 1.0 - t / 3.0, 0.5 * np.sqrt(np.pi / np.maximum(t, 1e-300)) * erf(np.sqrt(t)))
    def pair(i, j):
        A, a, ca = fns[i]; B, b, cb = fns[j]
        p = a[:, None] + b[None, :]; mu = a[:, None] * b[None, :] / p
        K = np.exp(-mu * np.sum((A - B) ** 2)); P = (a[:, None, None] * A + b[None, :, None] * B) / p[..., None]
        return p, mu, K * ca[:, None] * cb[None, :], P, np.sum((A - B) ** 2)
    n = len(fns); S = np.zeros((n, n)); H = np.zeros((n, n)); prs = {}
    for i in range(n):
        for j in range(n):
            p, mu, w, P, r2 = pair(i, j); prs[i, j] = (p, w, P)
            S[i, j] = np.sum(w * (np.pi / p) ** 1.5)
            H[i, j] = np.sum(w * mu * (3 - 2 * mu * r2) * (np.pi / p) ** 1.5)
            for Z, C in zip(charges, coords):
                H[i, j] -= Z * np.sum(w * 2 * np.pi / p * f0(p * np.sum((P - C) ** 2, axis=-1)))
    eri = np.zeros((n, n, n, n))
    for (i, j), (p, w1, P) in prs.items():
        for (k, l), (q, w2, Q) in prs.items():
            pp = p.reshape(-1, 1); qq = q.reshape(1, -1)
            d2 = np.sum((P.reshape(-1, 1, 3) - Q.reshape(1, -1, 3)) ** 2, axis=-1)
            eri[i, j, k, l] = np.sum(np.outer(w1, w2) * 2 * np.pi ** 2.5 / (pp * qq * np.sqrt(pp + qq)) * f0(pp * qq / (pp + qq) * d2))
    return S, H, eri
H_S = [([0.100112428, 0.2430767471, 0.6259552659, 1.822142904, 6.513143725, 35.52322122],
        [0.1303340841, 0.4164915298, 0.3705627997, 0.1685383049, 0.0493614929, 0.0091635963]),
       ([0.0654284286, 0.100112428, 0.2430767471, 0.6259552659], [4.8518572723, 0.4054347007, 0.0547707665, 0.1436562367]),
       ([0.0654284286, 0.100112428, 0.2430767471, 0.6259552659], [1.746076269, -5.7363906541, -0.3296656157, 0.1098876093])]
def hydrogens(n_atoms):
    return [(a, e, c) for a in range(n_atoms) for (e, c) in H_S]
def core_orbitals(S, H):
    eps, C = eigh(H, S)
    return eps, C
"""
    invalid = """
def run_model():
    try:
        qs_fock_matrix(S.copy(), H.copy(), eri.copy(), C.copy(), eps.copy(), n_occ, 1.4, 0.0, 1.0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_qs_fock_matrix(S.copy(), H.copy(), eri.copy(), C.copy(), eps.copy(), n_occ, 1.4, 0.0, 1.0)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Typical: H2 at 1.4 bohr with core-Hamiltonian orbitals, opposite-spin scaling ---
        {
            "setup": helper + """
S, H, eri = s_integrals([1.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], hydrogens(2))
eps, C = core_orbitals(S, H)
""",
            "call": "qs_fock_matrix(S.copy(), H.copy(), eri.copy(), C.copy(), eps.copy(), 1, 1.4, 0.0, 1.0)",
            "gold_call": "_oracle_qs_fock_matrix(S.copy(), H.copy(), eri.copy(), C.copy(), eps.copy(), 1, 1.4, 0.0, 1.0)",
            "tol": 1e-10,
        },
        # --- Typical: H4 chain, rotated orbitals (a unitary mix of the core orbitals) and shifted
        #     orbital energies, unscaled self-energy ---
        {
            "setup": helper + """
S, H, eri = s_integrals([1.0] * 4, [[0, 0, 0.0], [0, 0, 1.5], [0, 0, 3.4], [0, 0, 4.9]], hydrogens(4))
eps, C = core_orbitals(S, H)
theta = 0.3
rot = np.eye(len(eps)); rot[1, 1] = rot[2, 2] = np.cos(theta); rot[1, 2] = -np.sin(theta); rot[2, 1] = np.sin(theta)
C = C @ rot
eps = eps + np.linspace(-0.2, 0.3, len(eps))
""",
            "call": "qs_fock_matrix(S.copy(), H.copy(), eri.copy(), C.copy(), eps.copy(), 2, 0.525, 1.0, 1.0)",
            "gold_call": "_oracle_qs_fock_matrix(S.copy(), H.copy(), eri.copy(), C.copy(), eps.copy(), 2, 0.525, 1.0, 1.0)",
            "tol": 1e-10,
        },
        # --- Boundary: s = 0 leaves the closed-shell Fock matrix of the given orbitals ---
        {
            "setup": helper + """
S, H, eri = s_integrals([1.0] * 4, [[0, 0, 0.0], [0, 0, 1.4], [0, 1.6, 0.2], [0.3, 1.5, 1.6]], hydrogens(4))
eps, C = core_orbitals(S, H)
""",
            "call": "qs_fock_matrix(S.copy(), H.copy(), eri.copy(), C.copy(), eps.copy(), 2, 0.0, 0.6, 1.0)",
            "gold_call": "_oracle_qs_fock_matrix(S.copy(), H.copy(), eri.copy(), C.copy(), eps.copy(), 2, 0.0, 0.6, 1.0)",
            "tol": 1e-10,
        },
        # --- Edge: stretched H2 (4.5 bohr) with a small gap, both spin components scaled ---
        {
            "setup": helper + """
S, H, eri = s_integrals([1.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 4.5]], hydrogens(2))
eps, C = core_orbitals(S, H)
""",
            "call": "qs_fock_matrix(S.copy(), H.copy(), eri.copy(), C.copy(), eps.copy(), 1, 0.7, 0.6, 1.0)",
            "gold_call": "_oracle_qs_fock_matrix(S.copy(), H.copy(), eri.copy(), C.copy(), eps.copy(), 1, 0.7, 0.6, 1.0)",
            "tol": 1e-10,
        },
        # --- Invalid: no virtual orbital (n_occ = n_bf) ---
        {
            "setup": helper + """
S, H, eri = s_integrals([1.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], hydrogens(2))
eps, C = core_orbitals(S, H)
n_occ = 6
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
