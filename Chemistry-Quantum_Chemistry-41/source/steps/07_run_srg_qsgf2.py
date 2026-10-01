"""
Implement run_srg_qsgf2, which iterates the quasiparticle-self-consistent cycle of the
renormalized second-order Green's-function method from a restricted Hartree-Fock start to
self-consistency and returns the converged quasiparticle energies.

Energies are in hartree, and orbital coefficients and matrices refer to the atomic-orbital
basis of compute_ao_integrals.

Returns
-------
np.ndarray of shape (n_bf,): converged quasiparticle energies in ascending order, in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_srg_qsgf2(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    '''Converged quasiparticle energies of the renormalized qs second-order scheme.

    Parameters
    ----------
    overlap : np.ndarray
        Atomic-orbital overlap matrix S, shape (n_bf, n_bf).
    core_hamiltonian : np.ndarray
        Core Hamiltonian h, shape (n_bf, n_bf), in hartree.
    eri : np.ndarray
        Atomic-orbital electron-repulsion integrals in chemists' notation, shape
        (n_bf, n_bf, n_bf, n_bf), in hartree.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= n_bf - 1.
    s, c_ss, c_os : float
        Flow parameter (hartree^-2) and same-/opposite-spin scaling factors, as in
        srg_self_energy.

    Returns
    -------
    qp_energies : np.ndarray
        Shape (n_bf,), in hartree, ascending: the quasiparticle energies at self-consistency,
        i.e. the generalized eigenvalues, with respect to the overlap S, of the matrix F of
        qs_fock_matrix evaluated with its own S-orthonormal eigenvectors as orbitals (the n_occ
        lowest occupied) and its own eigenvalues as orbital energies. The cycle starts from the
        converged restricted Hartree-Fock orbitals and orbital energies of run_rhf and is
        converged until no quasiparticle energy changes by more than 1e-10 hartree between
        cycles and the orbital-gradient matrix F P S - S P F (P the density matrix) has no
        element larger than 1e-9.

    Raises
    ------
    ValueError
        If the inputs are invalid as in qs_fock_matrix or run_rhf, or the cycle does not
        converge within 300 iterations.
    '''
    return qp_energies

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh


def _oracle_run_srg_qsgf2(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    overlap = np.asarray(overlap, dtype=float)
    n_bf = overlap.shape[0]
    if int(n_occ) != n_occ or not 1 <= n_occ <= n_bf - 1:
        raise ValueError("n_occ must be an integer between 1 and n_bf - 1")
    n_occ = int(n_occ)
    _, energies, coefficients = _oracle_run_rhf(overlap, core_hamiltonian, eri, 0.0, n_occ)
    history = []
    for _ in range(300):
        fock = _oracle_qs_fock_matrix(overlap, core_hamiltonian, eri, coefficients, energies, n_occ, s, c_ss, c_os)
        density = 2.0 * coefficients[:, :n_occ] @ coefficients[:, :n_occ].T
        error = fock @ density @ overlap - overlap @ density @ fock
        history = (history + [(fock, error)])[-8:]
        new_energies, new_coefficients = eigh(_diis_extrapolate(history), overlap)
        change = np.abs(new_energies - energies).max()
        energies, coefficients = new_energies, new_coefficients
        if change < 1e-10 and np.abs(error).max() < 1e-9:
            return energies
    raise ValueError("quasiparticle self-consistency did not converge")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    helper = """import numpy as np
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
"""
    invalid = """
def run_model():
    try:
        run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), n_occ, 1.4, 0.0, 1.0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), n_occ, 1.4, 0.0, 1.0)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Typical: H2 at 1.4 bohr, opposite-spin-scaled parametrization ---
        {
            "setup": helper + """
S, H, eri = s_integrals([1.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], hydrogens(2))
""",
            "call": "run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), 1, 1.4, 0.0, 1.0)",
            "gold_call": "_oracle_run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), 1, 1.4, 0.0, 1.0)",
            "tol": 1e-8,
        },
        # --- Typical: H4 chain, unscaled self-energy with s = 0.525 ---
        {
            "setup": helper + """
S, H, eri = s_integrals([1.0] * 4, [[0, 0, 0.0], [0, 0, 1.5], [0, 0, 3.4], [0, 0, 4.9]], hydrogens(4))
""",
            "call": "run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), 2, 0.525, 1.0, 1.0)",
            "gold_call": "_oracle_run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), 2, 0.525, 1.0, 1.0)",
            "tol": 1e-8,
        },
        # --- Typical: bent H4 with both spin components weighted (exchange terms active) ---
        {
            "setup": helper + """
S, H, eri = s_integrals([1.0] * 4, [[0, 0, 0.0], [0, 0, 1.4], [0, 1.6, 0.2], [0.3, 1.5, 1.6]], hydrogens(4))
""",
            "call": "run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), 2, 0.7, 0.6, 1.0)",
            "gold_call": "_oracle_run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), 2, 0.7, 0.6, 1.0)",
            "tol": 1e-8,
        },
        # --- Boundary: s = 0 returns the Hartree-Fock orbital energies ---
        {
            "setup": helper + """
S, H, eri = s_integrals([1.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], hydrogens(2))
""",
            "call": "run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), 1, 0.0, 0.0, 1.0)",
            "gold_call": "_oracle_run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), 1, 0.0, 0.0, 1.0)",
            "tol": 1e-8,
        },
        # --- Edge: H2 stretched to 3.0 bohr (smaller gap, stronger regularization effect) ---
        {
            "setup": helper + """
S, H, eri = s_integrals([1.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], hydrogens(2))
""",
            "call": "run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), 1, 1.4, 0.0, 1.0)",
            "gold_call": "_oracle_run_srg_qsgf2(S.copy(), H.copy(), eri.copy(), 1, 1.4, 0.0, 1.0)",
            "tol": 1e-8,
        },
        # --- Invalid: no occupied orbital ---
        {
            "setup": helper + """
S, H, eri = s_integrals([1.0, 1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], hydrogens(2))
n_occ = 0
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
