"""
Iterate the source method's quasiparticle equations to self-consistency and return the converged quasiparticle energies and orbitals.

Quasiparticle self-consistency replaces the energy-dependent Dyson equation by an effective
one-particle eigenvalue problem whose operator depends on its own solutions. The converged
occupied quasiparticle energies approximate minus the vertical ionisation energies, and the
virtual ones approximate minus the electron affinities.

Returns
-------
tuple (eps, C): the converged quasiparticle energies in ascending order, shape (nbf,), and the matching orbital coefficients, shape (nbf, nbf), orthonormal in the overlap metric
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def iterate_quasiparticle_self_consistency(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", n_occ: int, C0: "np.ndarray", eps0: "np.ndarray", s: float, c_ss: float, c_os: float, conv_tol: float = 1e-9, max_iter: int = 200) -> tuple:
    '''Converge the source method's quasiparticle self-consistency cycle.

    Parameters
    ----------
    S : np.ndarray
        AO overlap matrix, shape (nbf, nbf).
    h : np.ndarray
        AO core Hamiltonian, shape (nbf, nbf), hartree.
    eri : np.ndarray
        AO electron-repulsion integrals, chemists' notation (mn|ls), shape (nbf,) * 4.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= nbf.
    C0 : np.ndarray
        Starting orbital coefficients, shape (nbf, nbf), orthonormal in the overlap
        metric, with columns ordered by ascending ``eps0`` (typically the restricted Hartree-Fock solution).
    eps0 : np.ndarray
        Starting orbital energies (hartree), shape (nbf,), ascending.
    s : float
        Flow parameter of the source method (hartree^-2), finite and non-negative.
    c_ss : float
        Same-spin scaling factor of the self-energy.
    c_os : float
        Opposite-spin scaling factor of the self-energy.
    conv_tol : float
        Convergence threshold (hartree): the cycle stops at the first iterate (C, eps)
        for which the total quasiparticle Fock matrix, expressed in the orbitals C,
        deviates from the diagonal matrix of eps by less than ``conv_tol`` in every
        element, where the total quasiparticle Fock matrix is
        (Hartree-Fock matrix of the aufbau density of C plus the back-transformed static
        self-energy built from C and eps).
    max_iter : int
        Maximum number of Fock-matrix builds.

    Returns
    -------
    result : tuple
        ``(eps, C)``: the converged quasiparticle energies in ascending order, shape
        (nbf,), and the corresponding orbital coefficients, shape (nbf, nbf),
        orthonormal in the overlap metric. The occupied quasiparticle orbitals are the first n_occ columns.

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, n_occ lies outside [1, nbf], or ``s`` is
        negative or not finite.
    RuntimeError
        If the cycle does not converge within ``max_iter`` iterations.
    '''
    return eps, C

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_iterate_quasiparticle_self_consistency(S: "np.ndarray", h: "np.ndarray", eri: "np.ndarray", n_occ: int, C0: "np.ndarray", eps0: "np.ndarray", s: float, c_ss: float, c_os: float, conv_tol: float = 1e-9, max_iter: int = 200) -> tuple:
    S = np.asarray(S, dtype=float)
    C = np.array(C0, dtype=float)
    eps = np.array(eps0, dtype=float).ravel()
    X = _symmetric_orthogonalizer(S)
    focks, errors = [], []
    for _ in range(int(max_iter)):
        fock = _oracle_build_quasiparticle_fock(S, h, eri, C, eps, n_occ, s, c_ss, c_os)
        residual = C.T @ fock @ C - np.diag(eps)
        if np.abs(residual).max() < conv_tol:
            return eps, C
        u = X @ S @ C  # S^(1/2) C: maps the orbital basis to the Loewdin-orthonormal basis
        focks.append(fock)
        errors.append(u @ residual @ u.T)
        if len(focks) > 10:
            focks.pop(0)
            errors.pop(0)
        eps, ct = np.linalg.eigh(X @ _diis_extrapolate(focks, errors) @ X)
        C = X @ ct
    raise RuntimeError("quasiparticle self-consistency did not converge")

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

def qp_invariants(result, n_occ):
    """Quasiparticle energies and aufbau density matrix (invariant to orbital signs)."""
    eps, C = result
    C = np.asarray(C, dtype=float)
    return np.concatenate([np.asarray(eps, dtype=float), (2.0 * C[:, :n_occ] @ C[:, :n_occ].T).ravel()])
'''
    return [
        # --- Normal: H2 / STO-3G at R = 1.4 bohr from the RHF solution, unscaled ---
        {
            "setup": s_basis + """
S, h, eri, e_nuc = s_shell_integrals([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], [1.0, 1.0],
                                     [(0, H_STO3G), (1, H_STO3G)])
eps0, C0 = roothaan_orbitals(S, h, eri, 1, 60)
""",
            "call": "qp_invariants(iterate_quasiparticle_self_consistency(S.copy(), h.copy(), eri.copy(), 1, C0.copy(), eps0.copy(), 0.6, 1.0, 1.0, 1e-10, 200), 1)",
            "gold_call": "qp_invariants(_oracle_iterate_quasiparticle_self_consistency(S.copy(), h.copy(), eri.copy(), 1, C0.copy(), eps0.copy(), 0.6, 1.0, 1.0, 1e-10, 200), 1)",
            "tol": 1e-7,
        },
        # --- Normal: HeH+ / 6-31G at R = 1.4632 bohr, spin-component scaled ---
        {
            "setup": s_basis + """
S, h, eri, e_nuc = s_shell_integrals([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]], [2.0, 1.0],
                                     [(0, HE_631G[0]), (0, HE_631G[1]), (1, H_631G[0]), (1, H_631G[1])])
eps0, C0 = roothaan_orbitals(S, h, eri, 1, 60)
""",
            "call": "qp_invariants(iterate_quasiparticle_self_consistency(S.copy(), h.copy(), eri.copy(), 1, C0.copy(), eps0.copy(), 0.9, 0.5, 1.2, 1e-10, 200), 1)",
            "gold_call": "qp_invariants(_oracle_iterate_quasiparticle_self_consistency(S.copy(), h.copy(), eri.copy(), 1, C0.copy(), eps0.copy(), 0.9, 0.5, 1.2, 1e-10, 200), 1)",
            "tol": 1e-7,
        },
        # --- Normal: rectangular H4 / 6-31G, two occupied orbitals, opposite-spin channel only ---
        {
            "setup": s_basis + """
coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4], [0.0, 2.8, 0.0], [0.0, 2.8, 1.4]]
S, h, eri, e_nuc = s_shell_integrals(coords, [1.0] * 4, [(k, H_631G[m]) for k in range(4) for m in range(2)])
eps0, C0 = roothaan_orbitals(S, h, eri, 2, 60)
""",
            "call": "qp_invariants(iterate_quasiparticle_self_consistency(S.copy(), h.copy(), eri.copy(), 2, C0.copy(), eps0.copy(), 1.2, 0.0, 1.1, 1e-10, 200), 2)",
            "gold_call": "qp_invariants(_oracle_iterate_quasiparticle_self_consistency(S.copy(), h.copy(), eri.copy(), 2, C0.copy(), eps0.copy(), 1.2, 0.0, 1.1, 1e-10, 200), 2)",
            "tol": 1e-7,
        },
        # --- Edge: stretched H2 / 6-31G (R = 3.0 bohr, strong correlation) from a crude start ---
        {
            "setup": s_basis + """
S, h, eri, e_nuc = s_shell_integrals([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], [1.0, 1.0],
                                     [(0, H_631G[0]), (0, H_631G[1]), (1, H_631G[0]), (1, H_631G[1])])
eps0, C0 = roothaan_orbitals(S, h, eri, 1, 1)
""",
            "call": "qp_invariants(iterate_quasiparticle_self_consistency(S.copy(), h.copy(), eri.copy(), 1, C0.copy(), eps0.copy(), 0.7, 1.0, 1.0, 1e-10, 200), 1)",
            "gold_call": "qp_invariants(_oracle_iterate_quasiparticle_self_consistency(S.copy(), h.copy(), eri.copy(), 1, C0.copy(), eps0.copy(), 0.7, 1.0, 1.0, 1e-10, 200), 1)",
            "tol": 1e-7,
        },
        # --- Boundary: s = 0, the fixed point is the Hartree-Fock solution itself ---
        {
            "setup": s_basis + """
S, h, eri, e_nuc = s_shell_integrals([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]], [2.0, 1.0],
                                     [(0, HE_631G[0]), (0, HE_631G[1]), (1, H_631G[0]), (1, H_631G[1])])
eps0, C0 = roothaan_orbitals(S, h, eri, 1, 2)
""",
            "call": "qp_invariants(iterate_quasiparticle_self_consistency(S.copy(), h.copy(), eri.copy(), 1, C0.copy(), eps0.copy(), 0.0, 1.0, 1.0, 1e-10, 200), 1)",
            "gold_call": "qp_invariants(_oracle_iterate_quasiparticle_self_consistency(S.copy(), h.copy(), eri.copy(), 1, C0.copy(), eps0.copy(), 0.0, 1.0, 1.0, 1e-10, 200), 1)",
            "tol": 1e-7,
        },
    ]
