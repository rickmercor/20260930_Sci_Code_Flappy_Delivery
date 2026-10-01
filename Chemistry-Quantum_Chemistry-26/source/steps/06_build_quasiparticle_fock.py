"""
Assemble the total quasiparticle Fock matrix of the source method in the atomic-orbital basis from the current orbitals and quasiparticle energies.

In quasiparticle self-consistent Green's-function theory the orbitals and quasiparticle energies
are the eigenvectors and eigenvalues of an effective one-particle operator: the Hartree-Fock
Fock operator plus a static correlation potential. The correlation potential is naturally
defined in the basis of the current orbitals, whereas the eigenvalue problem is solved in the
non-orthogonal atomic-orbital basis.

Returns
-------
np.ndarray of shape (nbf, nbf): the total quasiparticle Fock matrix (Hartree-Fock plus static self-energy) in the atomic-orbital basis, in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_quasiparticle_fock(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", C: "np.ndarray", eps: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    '''Total quasiparticle Fock matrix (Hartree-Fock plus static self-energy) in the AO basis.

    Parameters
    ----------
    S : np.ndarray
        AO overlap matrix, shape (nbf, nbf).
    h : np.ndarray
        AO core Hamiltonian, shape (nbf, nbf), hartree.
    eri : np.ndarray
        AO electron-repulsion integrals in chemists' notation, eri[m, n, l, s] holding (mn|ls),
        shape (nbf, nbf, nbf, nbf).
    C : np.ndarray
        Current orbital coefficients, shape (nbf, nbf); the columns are orbitals,
        orthonormal in the overlap metric. Columns 0 .. n_occ-1 are the doubly occupied orbitals.
    eps : np.ndarray
        Current quasiparticle energies of the columns of ``C`` (hartree), shape (nbf,).
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= nbf.
    s : float
        Flow parameter of the source method (hartree^-2), finite and non-negative.
    c_ss : float
        Same-spin scaling factor of the self-energy.
    c_os : float
        Opposite-spin scaling factor of the self-energy.

    Returns
    -------
    fock : np.ndarray
        AO-basis matrix of shape (nbf, nbf), in hartree, of the source method's total
        quasiparticle operator: the closed-shell Hartree-Fock Fock operator of the aufbau
        density of the first n_occ columns of ``C``, plus the source method's
        flow-regularised static self-energy evaluated in the orbitals ``C`` with the
        energies ``eps`` (all orbitals correlated).

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, n_occ lies outside [1, nbf], or ``s`` is
        negative or not finite.
    '''
    return fock

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_quasiparticle_fock(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", C: "np.ndarray", eps: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    S = np.asarray(S, dtype=float)
    h = np.asarray(h, dtype=float)
    eri = np.asarray(eri, dtype=float)
    C = np.asarray(C, dtype=float)
    nbf = S.shape[0]
    if (S.shape != (nbf, nbf) or h.shape != (nbf, nbf) or C.shape != (nbf, nbf)
            or eri.shape != (nbf,) * 4 or np.asarray(eps).size != nbf):
        raise ValueError("inconsistent array shapes")
    if not 1 <= int(n_occ) <= nbf:
        raise ValueError("n_occ must lie in [1, nbf]")
    c_occ = C[:, : int(n_occ)]
    J, K = _coulomb_exchange(eri, 2.0 * c_occ @ c_occ.T)
    eri_mo = np.einsum("mnls,mp,nq,lr,st->pqrt", eri, C, C, C, C, optimize=True)
    sigma = _oracle_compute_static_self_energy(eps, eri_mo, n_occ, s, c_ss, c_os)
    sc = S @ C
    return h + J - 0.5 * K + sc @ sigma @ sc.T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    s_basis = '''import numpy as np
from scipy.special import erf

H_STO3G = ([3.425250914, 0.6239137298, 0.1688554040], [0.1543289673, 0.5353281423, 0.4446345422])
H_631G = [([18.73113696, 2.825394365, 0.6401216923], [0.03349460434, 0.2347269535, 0.8137573261]),
          ([0.1612777588], [1.0])]
HE_631G = [([38.421634, 5.77803, 1.241774], [0.04013973935, 0.261246097, 0.7931846246]),
           ([0.297964], [1.0])]

def s_shell_integrals(coords, charges, shells):
    """S, core Hamiltonian, chemists' ERIs and nuclear repulsion for contracted s shells."""
    coords = np.asarray(coords, dtype=float)
    funcs = []
    for center, (exps, coefs) in shells:
        a = np.asarray(exps, dtype=float)
        d = np.asarray(coefs, dtype=float) * (2.0 * a / np.pi) ** 0.75
        d = d / np.sqrt(np.sum(np.outer(d, d) * (np.pi / np.add.outer(a, a)) ** 1.5))
        funcs.append((a, d, coords[center]))

    def boys0(t):
        t_safe = np.where(t < 1e-12, 1.0, t)
        return np.where(t < 1e-12, 1.0 - t / 3.0, 0.5 * np.sqrt(np.pi / t_safe) * erf(np.sqrt(t_safe)))

    def pair(f, g):
        (a, da, A), (b, db, B) = f, g
        p = np.add.outer(a, b).ravel()
        mu = np.multiply.outer(a, b).ravel() / p
        r2 = float(np.sum((A - B) ** 2))
        P = ((a[:, None, None] * A + b[None, :, None] * B).reshape(-1, 3)) / p[:, None]
        return p, mu, P, np.outer(da, db).ravel() * np.exp(-mu * r2), r2

    n = len(funcs)
    pairs = [[pair(funcs[i], funcs[j]) for j in range(n)] for i in range(n)]
    S = np.zeros((n, n))
    h = np.zeros((n, n))
    eri = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            p, mu, P, c, r2 = pairs[i][j]
            ov = c * (np.pi / p) ** 1.5
            S[i, j] = ov.sum()
            h[i, j] = np.sum(ov * mu * (3.0 - 2.0 * mu * r2))
            for C, Z in zip(coords, charges):
                h[i, j] -= Z * np.sum(c * 2.0 * np.pi / p * boys0(p * np.sum((P - C) ** 2, axis=1)))
            for k in range(n):
                for l in range(n):
                    q, _, Q, d, _ = pairs[k][l]
                    pp, qq = np.meshgrid(p, q, indexing="ij")
                    t = pp * qq / (pp + qq) * np.sum((P[:, None, :] - Q[None, :, :]) ** 2, axis=2)
                    eri[i, j, k, l] = np.sum(np.outer(c, d) * 2.0 * np.pi ** 2.5
                                             / (pp * qq * np.sqrt(pp + qq)) * boys0(t))
    e_nuc = sum(charges[a] * charges[b] / np.linalg.norm(coords[a] - coords[b])
                for a in range(len(charges)) for b in range(a))
    return S, h, eri, float(e_nuc)

def roothaan_orbitals(S, h, eri, n_occ, n_iter):
    """Orbitals and energies after n_iter plain Roothaan steps from the core guess (C^T S C = I)."""
    w, u = np.linalg.eigh(S)
    X = (u / np.sqrt(w)) @ u.T
    eps, C = np.linalg.eigh(X @ h @ X)
    C = X @ C
    for _ in range(n_iter):
        P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        F = h + np.einsum("mnls,ls->mn", eri, P) - 0.5 * np.einsum("mlns,ls->mn", eri, P)
        eps, C = np.linalg.eigh(X @ F @ X)
        C = X @ C
    return eps, C
'''
    return [
        # --- Normal: H2 / STO-3G at R = 1.4 bohr, converged RHF orbitals, unscaled ---
        {
            "setup": s_basis + """
S, h, eri, e_nuc = s_shell_integrals([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1.0, 1.0],
                                     [(0, H_STO3G), (1, H_STO3G)])
eps, C = roothaan_orbitals(S, h, eri, 1, 60)
""",
            "call": "build_quasiparticle_fock(S.copy(), h.copy(), eri.copy(), C.copy(), eps.copy(), 1, 0.6, 1.0, 1.0)",
            "gold_call": "_oracle_build_quasiparticle_fock(S.copy(), h.copy(), eri.copy(), C.copy(), eps.copy(), 1, 0.6, 1.0, 1.0)",
            "tol": 1e-9,
        },
        # --- Normal: rectangular H4 / 6-31G, two occupied orbitals, spin-component scaled ---
        {
            "setup": s_basis + """
coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 2.8, 0.0], [0.0, 2.8, 1.4]]
S, h, eri, e_nuc = s_shell_integrals(coords, [1.0] * 4, [(k, H_631G[m]) for k in range(4) for m in range(2)])
eps, C = roothaan_orbitals(S, h, eri, 2, 60)
""",
            "call": "build_quasiparticle_fock(S.copy(), h.copy(), eri.copy(), C.copy(), eps.copy(), 2, 0.9, 0.5, 1.2)",
            "gold_call": "_oracle_build_quasiparticle_fock(S.copy(), h.copy(), eri.copy(), C.copy(), eps.copy(), 2, 0.9, 0.5, 1.2)",
            "tol": 1e-9,
        },
        # --- Edge: HeH+ / 6-31G with non-self-consistent orbitals (one Roothaan step),
        #     shifted energies and arbitrary orbital signs ---
        {
            "setup": s_basis + """
S, h, eri, e_nuc = s_shell_integrals([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]], [2.0, 1.0],
                                     [(0, HE_631G[0]), (0, HE_631G[1]), (1, H_631G[0]), (1, H_631G[1])])
eps, C = roothaan_orbitals(S, h, eri, 1, 1)
eps = eps + np.array([0.05, -0.03, 0.02, 0.0])
C = C * np.array([1.0, -1.0, -1.0, 1.0])
""",
            "call": "build_quasiparticle_fock(S.copy(), h.copy(), eri.copy(), C.copy(), eps.copy(), 1, 1.2, 0.0, 1.1)",
            "gold_call": "_oracle_build_quasiparticle_fock(S.copy(), h.copy(), eri.copy(), C.copy(), eps.copy(), 1, 1.2, 0.0, 1.1)",
            "tol": 1e-9,
        },
        # --- Boundary: s = 0 reduces to the Hartree-Fock Fock matrix of the supplied orbitals ---
        {
            "setup": s_basis + """
S, h, eri, e_nuc = s_shell_integrals([[0.0, 0.0, 0.0], [0.0, 0.0, 2.2]], [1.0, 1.0],
                                     [(0, H_631G[0]), (0, H_631G[1]), (1, H_631G[0]), (1, H_631G[1])])
eps, C = roothaan_orbitals(S, h, eri, 1, 3)
""",
            "call": "build_quasiparticle_fock(S.copy(), h.copy(), eri.copy(), C.copy(), eps.copy(), 1, 0.0, 1.0, 1.0)",
            "gold_call": "_oracle_build_quasiparticle_fock(S.copy(), h.copy(), eri.copy(), C.copy(), eps.copy(), 1, 0.0, 1.0, 1.0)",
            "tol": 1e-9,
        },
    ]
