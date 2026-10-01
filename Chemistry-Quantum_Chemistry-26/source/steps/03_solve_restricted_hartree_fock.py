"""
Solve the closed-shell restricted Hartree-Fock (Roothaan-Hall) equations self-consistently and return the total energy, the canonical orbital energies and the orbital coefficients.

The canonical Hartree-Fock orbitals and energies are the reference and the starting point of the
quasiparticle self-consistent iterations in the later steps. Individual orbitals are fixed only
up to sign (and up to rotations within degenerate sets), so the well-defined results are the
energy, the orbital energies and the density matrix.

Returns
-------
tuple (energy, eps, C): the total RHF energy as a float in hartree (including nuclear repulsion), the ascending orbital energies of shape (nbf,), and the orbital coefficients of shape (nbf, nbf), orthonormal in the overlap metric
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_restricted_hartree_fock(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", n_occ: int, e_nuc: float) -> tuple:
    '''Closed-shell restricted Hartree-Fock ground state with aufbau occupation.

    Parameters
    ----------
    S : np.ndarray
        AO overlap matrix, shape (nbf, nbf), symmetric positive definite.
    h : np.ndarray
        AO core Hamiltonian (kinetic plus nuclear attraction), shape (nbf, nbf), hartree.
    eri : np.ndarray
        AO electron-repulsion integrals in chemists' notation, eri[m, n, l, s] holding (mn|ls),
        shape (nbf, nbf, nbf, nbf), with the eightfold symmetry of real orbitals.
    n_occ : int
        Number of doubly occupied spatial orbitals, 1 <= n_occ <= nbf.
    e_nuc : float
        Nuclear repulsion energy (hartree), added to the electronic energy.

    Returns
    -------
    result : tuple
        ``(energy, eps, C)``: ``energy`` is the converged total RHF energy (float,
        hartree, including ``e_nuc``); ``eps`` is the array of the nbf canonical orbital
        energies in ascending order, shape (nbf,); ``C`` is the (nbf, nbf) array whose
        column k holds the AO coefficients of the orbital with energy eps[k], normalised
        to be orthonormal in the overlap metric. The n_occ lowest orbitals are occupied. The self-consistent
        field is started from the core-Hamiltonian guess and converged at least to energy
        changes below 1e-10 hartree and a largest element of the orbital gradient
        (the commutator of the Fock and density matrices, in an orthonormalised basis)
        below 1e-8.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, S is not positive definite, or n_occ is
        outside [1, nbf].
    RuntimeError
        If the self-consistent field does not converge.
    '''
    return energy, eps, C

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _symmetric_orthogonalizer(S):
    """Loewdin S^(-1/2); raises ValueError for a non-positive-definite overlap."""
    w, u = np.linalg.eigh(S)
    if w.min() <= 1e-12 * max(1.0, w.max()):
        raise ValueError("overlap matrix is not positive definite")
    return (u / np.sqrt(w)) @ u.T


def _coulomb_exchange(eri, P):
    """Coulomb and exchange matrices J[P], K[P] in the AO basis."""
    J = np.einsum("mnls,ls->mn", eri, P, optimize=True)
    K = np.einsum("mlns,ls->mn", eri, P, optimize=True)
    return J, K


def _diis_extrapolate(focks, errors):
    """Pulay DIIS combination of stored Fock matrices."""
    m = len(focks)
    if m < 2:
        return focks[-1]
    b = -np.ones((m + 1, m + 1))
    b[m, m] = 0.0
    for i in range(m):
        for j in range(m):
            b[i, j] = np.sum(errors[i] * errors[j])
    rhs = np.zeros(m + 1)
    rhs[m] = -1.0
    coef = np.linalg.lstsq(b, rhs, rcond=None)[0][:m]
    return sum(c * f for c, f in zip(coef, focks))


def _oracle_solve_restricted_hartree_fock(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", n_occ: int, e_nuc: float) -> tuple:
    S = np.asarray(S, dtype=float)
    h = np.asarray(h, dtype=float)
    eri = np.asarray(eri, dtype=float)
    nbf = S.shape[0]
    if S.shape != (nbf, nbf) or h.shape != (nbf, nbf) or eri.shape != (nbf,) * 4:
        raise ValueError("inconsistent array shapes")
    if not 1 <= int(n_occ) <= nbf:
        raise ValueError("n_occ must lie in [1, nbf]")
    n_occ = int(n_occ)
    X = _symmetric_orthogonalizer(S)
    eps, ct = np.linalg.eigh(X @ h @ X)
    C = X @ ct
    focks, errors = [], []
    energy_old = None
    for _ in range(500):
        P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
        J, K = _coulomb_exchange(eri, P)
        F = h + J - 0.5 * K
        energy = 0.5 * np.sum(P * (h + F)) + float(e_nuc)
        err = X @ (F @ P @ S - S @ P @ F) @ X
        if energy_old is not None and abs(energy - energy_old) < 1e-12 and np.abs(err).max() < 1e-10:
            eps, ct = np.linalg.eigh(X @ F @ X)
            return float(energy), eps, X @ ct
        energy_old = energy
        focks.append(F)
        errors.append(err)
        if len(focks) > 8:
            focks.pop(0)
            errors.pop(0)
        eps, ct = np.linalg.eigh(X @ _diis_extrapolate(focks, errors) @ X)
        C = X @ ct
    raise RuntimeError("RHF did not converge")

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

def rhf_invariants(result, n_occ):
    """Energy, orbital energies and density matrix (sign- and rotation-invariant)."""
    energy, eps, C = result
    C = np.asarray(C, dtype=float)
    P = 2.0 * C[:, :n_occ] @ C[:, :n_occ].T
    return np.concatenate([[float(energy)], np.asarray(eps, dtype=float), P.ravel()])
'''
    return [
        # --- Normal: H2 / STO-3G at R = 1.4 bohr (minimal basis, one occupied orbital) ---
        {
            "setup": s_basis + """
S, h, eri, e_nuc = s_shell_integrals([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1.0, 1.0],
                                     [(0, H_STO3G), (1, H_STO3G)])
n_occ = 1
""",
            "call": "rhf_invariants(solve_restricted_hartree_fock(S.copy(), h.copy(), eri.copy(), n_occ, e_nuc), n_occ)",
            "gold_call": "rhf_invariants(_oracle_solve_restricted_hartree_fock(S.copy(), h.copy(), eri.copy(), n_occ, e_nuc), n_occ)",
            "tol": 1e-7,
        },
        # --- Normal: HeH+ / 6-31G at R = 1.4632 bohr (heteronuclear cation, 4 functions) ---
        {
            "setup": s_basis + """
S, h, eri, e_nuc = s_shell_integrals([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]], [2.0, 1.0],
                                     [(0, HE_631G[0]), (0, HE_631G[1]), (1, H_631G[0]), (1, H_631G[1])])
n_occ = 1
""",
            "call": "rhf_invariants(solve_restricted_hartree_fock(S.copy(), h.copy(), eri.copy(), n_occ, e_nuc), n_occ)",
            "gold_call": "rhf_invariants(_oracle_solve_restricted_hartree_fock(S.copy(), h.copy(), eri.copy(), n_occ, e_nuc), n_occ)",
            "tol": 1e-7,
        },
        # --- Normal: rectangular H4 / 6-31G (two occupied orbitals, 8 functions) ---
        {
            "setup": s_basis + """
coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 2.8, 0.0], [0.0, 2.8, 1.4]]
shells = [(k, H_631G[m]) for k in range(4) for m in range(2)]
S, h, eri, e_nuc = s_shell_integrals(coords, [1.0] * 4, shells)
n_occ = 2
""",
            "call": "rhf_invariants(solve_restricted_hartree_fock(S.copy(), h.copy(), eri.copy(), n_occ, e_nuc), n_occ)",
            "gold_call": "rhf_invariants(_oracle_solve_restricted_hartree_fock(S.copy(), h.copy(), eri.copy(), n_occ, e_nuc), n_occ)",
            "tol": 1e-7,
        },
        # --- Boundary: H- in a single STO-3G function (nbf = n_occ = 1, no virtual orbitals) ---
        {
            "setup": s_basis + """
S, h, eri, e_nuc = s_shell_integrals([[0.0, 0.0, 0.0]], [1.0], [(0, H_STO3G)])
n_occ = 1
""",
            "call": "rhf_invariants(solve_restricted_hartree_fock(S.copy(), h.copy(), eri.copy(), n_occ, e_nuc), n_occ)",
            "gold_call": "rhf_invariants(_oracle_solve_restricted_hartree_fock(S.copy(), h.copy(), eri.copy(), n_occ, e_nuc), n_occ)",
            "tol": 1e-7,
        },
        # --- Edge: stretched H2 / 6-31G at R = 4.0 bohr (small HOMO-LUMO gap) ---
        {
            "setup": s_basis + """
S, h, eri, e_nuc = s_shell_integrals([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]], [1.0, 1.0],
                                     [(0, H_631G[0]), (0, H_631G[1]), (1, H_631G[0]), (1, H_631G[1])])
n_occ = 1
""",
            "call": "rhf_invariants(solve_restricted_hartree_fock(S.copy(), h.copy(), eri.copy(), n_occ, e_nuc), n_occ)",
            "gold_call": "rhf_invariants(_oracle_solve_restricted_hartree_fock(S.copy(), h.copy(), eri.copy(), n_occ, e_nuc), n_occ)",
            "tol": 1e-7,
        },
    ]
