"""
Step 07 - Complete third-order secular matrix for restricted orbitals, split by spin symmetry and diagonalised.

For restricted orbitals the spin orbitals come in pairs, p = 2k with alpha spin and p = 2k + 1 with beta spin for the same spatial orbital k, and the Hamiltonian conserves the spin projection. States with Ms = 0 are therefore described in the space of the single excitations i -> a with equal spins and the double excitations i < j -> a < b whose spins add up to zero. In that space assemble the complete strict third-order matrix: the singles-singles block of the previous step, the singles-doubles coupling through second order and the doubles-doubles block through first order. As in the singles block, the first-order doubles-doubles part involves the Fock matrix of the input and not the zeroth-order energies, while the coupling keeps the form it has in Møller-Plesset-based schemes, with first-order doubles built on the zeroth-order energies.

Exchanging the alpha and beta labels of every spin orbital maps this space onto itself and commutes with the matrix. On the basis states it sends (i, a) to (i', a') and (i, j, a, b) to (i', j', a', b'), where p' = p xor 1 is the partner of p with the other spin; when a pair comes out in descending order it is swapped back, and each swap contributes a factor -1. Every eigenvector is either symmetric (+1: singlets, together with quintets, which have no single-excitation part) or antisymmetric (-1: triplets). Diagonalise the matrix within the requested sector. The single-excitation weight of an eigenvector is the squared norm of its singles part when the whole vector is normalised in the orthonormal intermediate-state basis.

Returns
-------
numpy.ndarray of shape (2, m_sector): excitation energies in hartree in ascending order (row 0) and their single-excitation weights (row 1)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def adc3_spin_sector_spectrum(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike, eri: npt.ArrayLike, n_occ: int, flip_parity: int) -> np.ndarray:
    '''Eigenvalues and single-excitation weights of one spin sector of the strict third-order matrix.

    Parameters
    ----------
    orbital_energies : array_like
        Zeroth-order one-particle energies e_p of the m spin orbitals in
        hartree, occupied spin orbitals first. H0 = sum_p e_p a+_p a_p.
        Spin orbitals are ordered p = 2k + s (s = 0 alpha, s = 1 beta).
    fock : array_like
        Fock matrix f_pq of the reference determinant in the same spin-orbital
        basis, shape (m, m), symmetric, with a vanishing occupied-virtual block.
        It need not be diagonal.
    eri : array_like
        Antisymmetrised two-electron integrals <pq||rs> = <pq|rs> - <pq|sr> in
        physicists' notation over real orbitals, shape (m, m, m, m).
    n_occ : int
        Number of occupied spin orbitals; they are the first n_occ.
    flip_parity : int
        +1 for the sector symmetric under exchange of alpha and beta labels
        (singlets), -1 for the antisymmetric sector (triplets).

    Returns
    -------
    spectrum : numpy.ndarray
        Array of shape (2, m_sector), m_sector being the number of Ms = 0
        single and double excitations in the requested sector. Row 0 holds
        the excitation energies in hartree in ascending order, row 1 the
        single-excitation weight of each eigenvector.

    Raises
    ------
    ValueError
        If orbital_energies is not one-dimensional with at least two entries,
        fock or eri does not have the matching shape, any input is not finite,
        n_occ is not an integer strictly between 0 and m, fock is not
        symmetric or has a non-zero occupied-virtual block (tolerance 1e-10),
        eri is not antisymmetric within each index pair or not symmetric
        under exchange of the two pairs (tolerance 1e-10), or some virtual
        orbital energy does not lie above every occupied one. Also if
        flip_parity is not +1 or -1, the number of spin orbitals or n_occ
        is odd, alpha and beta orbital energies differ, fock is not
        spin-diagonal with identical alpha and beta blocks, or eri does
        not conserve spin or changes under exchange of alpha and beta
        labels (tolerance 1e-10).
    '''
    return spectrum

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _check_spin_adapted(e, f, g, no):
    """Raise unless the spin-orbital arrays come from restricted spatial orbitals, p = 2k + s."""
    import numpy as np
    m = e.size
    if m % 2 or no % 2:
        raise ValueError("spin-adapted input needs an even number of spin orbitals and of occupied ones")
    s = np.arange(m) % 2
    flip = np.arange(m) ^ 1
    if np.max(np.abs(e - e[flip])) > 1.0e-10:
        raise ValueError("alpha and beta orbital energies differ")
    if (np.max(np.abs(np.where(s[:, None] != s[None, :], f, 0.0))) > 1.0e-10
            or np.max(np.abs(f - f[np.ix_(flip, flip)])) > 1.0e-10):
        raise ValueError("fock must be spin-diagonal with identical alpha and beta blocks")
    sp, sq, sr, ss = s[:, None, None, None], s[None, :, None, None], s[None, None, :, None], s[None, None, None, :]
    conserving = ((sp == sr) & (sq == ss)) | ((sp == ss) & (sq == sr))
    if (np.max(np.abs(np.where(conserving, 0.0, g))) > 1.0e-10
            or np.max(np.abs(g - g[np.ix_(flip, flip, flip, flip)])) > 1.0e-10):
        raise ValueError("eri must conserve spin and be invariant under exchanging alpha and beta")


def _oracle_adc3_spin_sector_spectrum(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike, eri: npt.ArrayLike,
                                      n_occ: int, flip_parity: int) -> np.ndarray:
    import numpy as np
    from itertools import combinations
    e, f, g, no = _checked_spin_orbital_hamiltonian(orbital_energies, fock, eri, n_occ)
    if flip_parity not in (1, -1):
        raise ValueError("flip_parity must be +1 or -1")
    _check_spin_adapted(e, f, g, no)
    nv = e.size - no
    o = slice(0, no)
    v = slice(no, no + nv)
    ph = [(i, a) for i in range(no) for a in range(nv) if i % 2 == a % 2]
    dd = [(i, j, a, b) for i, j in combinations(range(no), 2) for a, b in combinations(range(nv), 2)
          if (i % 2) + (j % 2) == (a % 2) + (b % 2)]
    n1, n2 = len(ph), len(dd)
    iph = np.array([x[0] for x in ph])
    aph = np.array([x[1] for x in ph])
    M11 = _oracle_adc3_singles_block(e, f, g, no)
    rows = iph * nv + aph
    M = np.zeros((n1 + n2, n1 + n2))
    M[:n1, :n1] = M11[np.ix_(rows, rows)]
    if n2 > 0:
        di, dj, da, db = (np.array([x[k] for x in dd]) for k in range(4))
        t = g[o, o, v, v] / _pair_denominators(e[:no], e[no:])
        gooov, govvv, govov = g[o, o, o, v], g[o, v, v, v], g[o, v, o, v]
        Yu = np.zeros((n2, no, no, nv, nv))
        z = np.arange(n2)
        Yu[z, di, dj, da, db] = 1.0
        Yu[z, dj, di, da, db] = -1.0
        Yu[z, di, dj, db, da] = -1.0
        Yu[z, dj, di, db, da] = 1.0
        ZI = np.einsum('ijbc,kabc->ijka', t, govvv)
        ZII = np.einsum('ilab,lkjb->ijka', t, gooov)
        ZA = 0.5 * ZI + ZII - ZII.transpose(1, 0, 2, 3)
        ZVI = np.einsum('jkbc,jkia->iabc', t, gooov)
        ZVII = np.einsum('ijbd,jcad->iabc', t, govvv)
        ZB = -0.5 * ZVI + ZVII - ZVII.transpose(0, 1, 3, 2)
        w = (np.einsum('zjkab,jkib->zia', Yu, gooov) + np.einsum('zijbc,jabc->zia', Yu, govvv)
             + np.einsum('zijbc,jabc->zia', Yu, ZB) - np.einsum('zjkab,jkib->zia', Yu, ZA)
             + np.einsum('zjkbc,jlbc,ilka->zia', Yu, t, gooov, optimize=True)
             + np.einsum('zjkbc,jkbd,icad->zia', Yu, t, govvv, optimize=True))
        M12 = -0.5 * w[:, iph, aph].T
        X1 = np.einsum('zjkab,ik->zijab', Yu, f[o, o]) + np.einsum('zijac,bc->zijab', Yu, f[v, v])
        X2 = np.einsum('zjkac,ickb->zijab', Yu, govov) - np.einsum('zikac,jckb->zijab', Yu, govov)
        W1 = (X1 + X1.transpose(0, 2, 1, 4, 3) + X2 + X2.transpose(0, 2, 1, 4, 3)
              + 0.5 * np.einsum('zklab,ijkl->zijab', Yu, g[o, o, o, o])
              + 0.5 * np.einsum('zijcd,abcd->zijab', Yu, g[v, v, v, v]))
        M22 = W1[:, di, dj, da, db].T
        M[:n1, n1:] = M12
        M[n1:, :n1] = M12.T
        M[n1:, n1:] = 0.5 * (M22 + M22.T)
    P = np.zeros((n1 + n2, n1 + n2))
    index1 = {x: k for k, x in enumerate(ph)}
    index2 = {x: k for k, x in enumerate(dd)}
    for k, (i, a) in enumerate(ph):
        P[index1[(i ^ 1, a ^ 1)], k] = 1.0
    for k, (i, j, a, b) in enumerate(dd):
        sign = 1.0
        i2, j2, a2, b2 = i ^ 1, j ^ 1, a ^ 1, b ^ 1
        if i2 > j2:
            i2, j2, sign = j2, i2, -sign
        if a2 > b2:
            a2, b2, sign = b2, a2, -sign
        P[n1 + index2[(i2, j2, a2, b2)], n1 + k] = sign
    lam, Q = np.linalg.eigh(P)
    Qs = Q[:, np.abs(lam - flip_parity) < 0.5]
    energies, V = np.linalg.eigh(Qs.T @ M @ Qs)
    X = Qs @ V
    weights = np.sum(X[:n1] ** 2, axis=0)
    return np.vstack([energies, weights])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: symmetric (singlet) sector of a restricted model Hamiltonian ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(11)
n, n_occ_spatial = 4, 2
A = rng.normal(scale=0.005, size=(n, n, n, n))
A = A + A.transpose(1, 0, 2, 3)
A = A + A.transpose(0, 1, 3, 2)
A = A + A.transpose(2, 3, 0, 1)
A = A + 0.25 * np.einsum('pq,rs->pqrs', np.eye(n), np.eye(n))
k = np.arange(2 * n) // 2
s = np.arange(2 * n) % 2
V = A[np.ix_(k, k, k, k)].transpose(0, 2, 1, 3)
V = V * ((s[:, None, None, None] == s[None, None, :, None]) & (s[None, :, None, None] == s[None, None, None, :]))
eri = V - V.transpose(0, 1, 3, 2)
F = np.diag([-1.05, -0.62, 0.28, 0.83])
F[0, 1] = F[1, 0] = 0.035
F[2, 3] = F[3, 2] = -0.045
orbital_energies = np.repeat(np.diag(F) + np.array([-0.02, 0.015, -0.01, 0.02]), 2)
fock = np.kron(F, np.eye(2))
n_occ = 2 * n_occ_spatial
flip_parity = 1
""",
            "call": "adc3_spin_sector_spectrum(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ, flip_parity)",
            "gold_call": "_oracle_adc3_spin_sector_spectrum(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ, flip_parity)",
            "tol": 1e-08,
        },
        # --- Normal: antisymmetric (triplet) sector of the same Hamiltonian ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(11)
n, n_occ_spatial = 4, 2
A = rng.normal(scale=0.005, size=(n, n, n, n))
A = A + A.transpose(1, 0, 2, 3)
A = A + A.transpose(0, 1, 3, 2)
A = A + A.transpose(2, 3, 0, 1)
A = A + 0.25 * np.einsum('pq,rs->pqrs', np.eye(n), np.eye(n))
k = np.arange(2 * n) // 2
s = np.arange(2 * n) % 2
V = A[np.ix_(k, k, k, k)].transpose(0, 2, 1, 3)
V = V * ((s[:, None, None, None] == s[None, None, :, None]) & (s[None, :, None, None] == s[None, None, None, :]))
eri = V - V.transpose(0, 1, 3, 2)
F = np.diag([-1.05, -0.62, 0.28, 0.83])
F[0, 1] = F[1, 0] = 0.035
F[2, 3] = F[3, 2] = -0.045
orbital_energies = np.repeat(np.diag(F) + np.array([-0.02, 0.015, -0.01, 0.02]), 2)
fock = np.kron(F, np.eye(2))
n_occ = 2 * n_occ_spatial
flip_parity = -1
""",
            "call": "adc3_spin_sector_spectrum(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ, flip_parity)",
            "gold_call": "_oracle_adc3_spin_sector_spectrum(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ, flip_parity)",
            "tol": 1e-08,
        },
        # --- Boundary: one occupied spatial orbital, the smallest doubles space ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(7)
n, n_occ_spatial = 4, 1
A = rng.normal(scale=0.005, size=(n, n, n, n))
A = A + A.transpose(1, 0, 2, 3)
A = A + A.transpose(0, 1, 3, 2)
A = A + A.transpose(2, 3, 0, 1)
A = A + 0.25 * np.einsum('pq,rs->pqrs', np.eye(n), np.eye(n))
k = np.arange(2 * n) // 2
s = np.arange(2 * n) % 2
V = A[np.ix_(k, k, k, k)].transpose(0, 2, 1, 3)
V = V * ((s[:, None, None, None] == s[None, None, :, None]) & (s[None, :, None, None] == s[None, None, None, :]))
eri = V - V.transpose(0, 1, 3, 2)
F = np.diag([-0.88, 0.21, 0.66, 1.12])
F[1, 2] = F[2, 1] = 0.03
F[2, 3] = F[3, 2] = -0.035
orbital_energies = np.repeat(np.diag(F) + np.array([0.01, -0.015, 0.02, 0.0]), 2)
fock = np.kron(F, np.eye(2))
n_occ = 2 * n_occ_spatial
flip_parity = 1
""",
            "call": "adc3_spin_sector_spectrum(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ, flip_parity)",
            "gold_call": "_oracle_adc3_spin_sector_spectrum(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ, flip_parity)",
            "tol": 1e-08,
        },
        # --- Edge: canonical limit with a larger virtual space, triplet sector ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(23)
n, n_occ_spatial = 5, 2
A = rng.normal(scale=0.005, size=(n, n, n, n))
A = A + A.transpose(1, 0, 2, 3)
A = A + A.transpose(0, 1, 3, 2)
A = A + A.transpose(2, 3, 0, 1)
A = A + 0.25 * np.einsum('pq,rs->pqrs', np.eye(n), np.eye(n))
k = np.arange(2 * n) // 2
s = np.arange(2 * n) % 2
V = A[np.ix_(k, k, k, k)].transpose(0, 2, 1, 3)
V = V * ((s[:, None, None, None] == s[None, None, :, None]) & (s[None, :, None, None] == s[None, None, None, :]))
eri = V - V.transpose(0, 1, 3, 2)
F = np.diag([-0.98, -0.57, 0.31, 0.74, 1.26])
orbital_energies = np.repeat(np.diag(F), 2)
fock = np.kron(F, np.eye(2))
n_occ = 2 * n_occ_spatial
flip_parity = -1
""",
            "call": "adc3_spin_sector_spectrum(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ, flip_parity)",
            "gold_call": "_oracle_adc3_spin_sector_spectrum(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ, flip_parity)",
            "tol": 1e-08,
        },
    ]
