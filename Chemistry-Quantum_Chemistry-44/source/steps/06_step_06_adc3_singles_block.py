"""
Step 06 - Singles-singles block of the third-order intermediate-state secular matrix.

Excited states come from the algebraic-diagrammatic construction for the polarisation propagator in its intermediate-state form. Take H(lambda) = H0 + lambda (H - H0) with the same H0 as in the ground-state step, and let Psi_0(lambda) be its exact ground state with energy E_0(lambda). The precursor states a+_a a_i Psi_0 and a+_a a+_b a_j a_i Psi_0 (i < j, a < b) are made orthogonal to Psi_0, the doubles are also made orthogonal to every single, and each class is then orthonormalised symmetrically (Löwdin) within itself. The secular matrix M_IJ = <Psi~_I | H - E_0 | Psi~_J> is expanded in powers of lambda and evaluated at lambda = 1. The strict third-order scheme keeps the singles-singles block through third order, the singles-doubles coupling through second order and the doubles-doubles block through first order, and its eigenvalues are excitation energies.

This step builds the first of those blocks for every occupied-virtual spin-orbital pair. Through first order it is the configuration-interaction-singles matrix formed with the Fock matrix of the input, not with the zeroth-order energies, and the reference energy cancels. The second- and third-order parts are the standard strict third-order contributions, written with the first-order doubles (whose denominators use the zeroth-order energies), the second-order doubles and singles of the ground-state step and the second-order one-particle density. Nothing is truncated beyond the order counting.

Returns
-------
numpy.ndarray of shape (n_occ * n_vir, n_occ * n_vir): the singles-singles block through third order in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def adc3_singles_block(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike, eri: npt.ArrayLike, n_occ: int) -> np.ndarray:
    '''Singles-singles block of the strict third-order secular matrix.

    Parameters
    ----------
    orbital_energies : array_like
        Zeroth-order one-particle energies e_p of the m spin orbitals in
        hartree, occupied spin orbitals first. H0 = sum_p e_p a+_p a_p.
    fock : array_like
        Fock matrix f_pq of the reference determinant in the same spin-orbital
        basis, shape (m, m), symmetric, with a vanishing occupied-virtual block.
        It need not be diagonal.
    eri : array_like
        Antisymmetrised two-electron integrals <pq||rs> = <pq|rs> - <pq|sr> in
        physicists' notation over real orbitals, shape (m, m, m, m).
    n_occ : int
        Number of occupied spin orbitals; they are the first n_occ.

    Returns
    -------
    singles_block : numpy.ndarray
        Symmetric array of shape (n_occ * n_vir, n_occ * n_vir) in hartree,
        with n_vir = m - n_occ. Row and column i * n_vir + a belong to the
        single excitation from occupied spin orbital i to virtual spin orbital
        a (virtual indices counted from 0 within the virtual block); every
        pair is included whatever the spins.

    Raises
    ------
    ValueError
        If orbital_energies is not one-dimensional with at least two entries,
        fock or eri does not have the matching shape, any input is not finite,
        n_occ is not an integer strictly between 0 and m, fock is not
        symmetric or has a non-zero occupied-virtual block (tolerance 1e-10),
        eri is not antisymmetric within each index pair or not symmetric
        under exchange of the two pairs (tolerance 1e-10), or some virtual
        orbital energy does not lie above every occupied one.
    '''
    return singles_block

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _occupied_delta(A, n_vir):
    """A_ij delta_ab laid out as an (i, a, j, b) array."""
    import numpy as np
    return A[:, None, :, None] * np.eye(n_vir)[None, :, None, :]


def _virtual_delta(A, n_occ):
    """A_ab delta_ij laid out as an (i, a, j, b) array."""
    import numpy as np
    return A[None, :, None, :] * np.eye(n_occ)[:, None, :, None]


def _oracle_adc3_singles_block(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike, eri: npt.ArrayLike,
                               n_occ: int) -> np.ndarray:
    import numpy as np
    e, f, g, no = _checked_spin_orbital_hamiltonian(orbital_energies, fock, eri, n_occ)
    nv = e.size - no
    o = slice(0, no)
    v = slice(no, no + nv)
    gs = _oracle_second_order_ground_state(e, f, g, no)
    t2 = gs[2:2 + no * no * nv * nv].reshape(no, no, nv, nv)
    ts = gs[2 + no * no * nv * nv:].reshape(no, nv)
    t = g[o, o, v, v] / _pair_denominators(e[:no], e[no:])
    roo = -0.5 * np.einsum('ikab,jkab->ij', t, t)
    rvv = 0.5 * np.einsum('ijac,ijbc->ab', t, t)
    goooo, goovv, govov, gvvvv = g[o, o, o, o], g[o, o, v, v], g[o, v, o, v], g[v, v, v, v]
    gooov, govvv = g[o, o, o, v], g[o, v, v, v]
    M = (_virtual_delta(f[v, v], no) - _occupied_delta(f[o, o], nv) - np.einsum('ibja->iajb', govov))
    X2 = (-0.5 * np.einsum('ikac,jkbc->iajb', t, goovv)
          + _virtual_delta(0.25 * np.einsum('klac,klbc->ab', t, goovv), no)
          + _occupied_delta(0.25 * np.einsum('ikcd,jkcd->ij', t, goovv), nv))
    M = M + X2 + X2.transpose(2, 3, 0, 1)
    Z4 = np.einsum('jkac,kbic->ijab', t, govov)
    Z3 = np.einsum('klab,ijkl->ijab', t, goooo)
    Z5 = np.einsum('ijcd,abcd->ijab', t, gvvvv)
    Zsq = np.einsum('ikac,jkbc->iajb', t, t)
    Y = (np.einsum('ka,jkib->iajb', ts, gooov)
         + np.einsum('ic,jabc->iajb', ts, govvv)
         + 0.5 * np.einsum('ikac,jkbc->iajb', t, Z4)
         + 0.5 * np.einsum('ikac,kjcb->iajb', t, Z4)
         + 0.5 * np.einsum('ibjc,ac->iajb', govov, rvv)
         - np.einsum('ibkc,jcka->iajb', govov, Zsq)
         - 0.5 * np.einsum('ikac,jkbc->iajb', t2, goovv)
         - 0.5 * np.einsum('ibka,jk->iajb', govov, roo)
         + 0.25 * np.einsum('ikac,jkbc->iajb', t, Z3)
         + 0.25 * np.einsum('ikac,jkbc->iajb', t, Z5)
         + _virtual_delta(0.5 * np.einsum('klac,klcb->ab', t, Z4), no)
         + _occupied_delta(0.5 * np.einsum('ikcd,jkdc->ij', t, Z4), nv)
         - _occupied_delta(np.einsum('kc,ikjc->ij', ts, gooov), nv)
         - _virtual_delta(np.einsum('kc,kabc->ab', ts, govvv), no)
         - _virtual_delta(0.125 * np.einsum('klac,klbc->ab', t, Z5), no)
         - _occupied_delta(0.125 * np.einsum('ikcd,jkcd->ij', t, Z3), nv)
         + _virtual_delta(0.25 * np.einsum('klac,klbc->ab', t2, goovv), no)
         + _occupied_delta(0.25 * np.einsum('ikcd,jkcd->ij', t2, goovv), nv))
    M = M + Y + Y.transpose(2, 3, 0, 1)
    M = M + (-np.einsum('adbc,icjd->iajb', gvvvv, Zsq)
             - np.einsum('iljk,kalb->iajb', goooo, Zsq)
             + _virtual_delta(np.einsum('adbc,cd->ab', gvvvv, rvv), no)
             + _virtual_delta(np.einsum('kalb,kl->ab', govov, roo), no)
             + 0.5 * np.einsum('klac,klbd,icjd->iajb', t, t, govov, optimize=True)
             + 0.5 * np.einsum('ikcd,jlcd,kalb->iajb', t, t, govov, optimize=True)
             - _occupied_delta(np.einsum('icjd,cd->ij', govov, rvv), nv)
             - _occupied_delta(np.einsum('iljk,kl->ij', goooo, roo), nv))
    return M.reshape(no * nv, no * nv)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: restricted model Hamiltonian with a non-diagonal Fock matrix ---
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
""",
            "call": "adc3_singles_block(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "gold_call": "_oracle_adc3_singles_block(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "tol": 1e-09,
        },
        # --- Boundary: diagonal Fock matrix equal to H0, the canonical limit ---
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
orbital_energies = np.repeat(np.diag(F), 2)
fock = np.kron(F, np.eye(2))
n_occ = 2 * n_occ_spatial
""",
            "call": "adc3_singles_block(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "gold_call": "_oracle_adc3_singles_block(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "tol": 1e-09,
        },
        # --- Edge: general spin-orbital Hamiltonian without spin structure ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(17)
m, n_occ = 5, 2
A = rng.normal(scale=0.005, size=(m, m, m, m))
A = A + A.transpose(1, 0, 2, 3)
A = A + A.transpose(0, 1, 3, 2)
A = A + A.transpose(2, 3, 0, 1)
A = A + 0.25 * np.einsum('pq,rs->pqrs', np.eye(m), np.eye(m))
V = A.transpose(0, 2, 1, 3)
eri = V - V.transpose(0, 1, 3, 2)
fock = np.diag([-0.95, -0.6, 0.3, 0.75, 1.2])
fock[0, 1] = fock[1, 0] = 0.025
fock[2, 4] = fock[4, 2] = -0.03
orbital_energies = np.diag(fock) + np.array([-0.015, 0.02, 0.01, 0.0, -0.02])
""",
            "call": "adc3_singles_block(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "gold_call": "_oracle_adc3_singles_block(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "tol": 1e-09,
        },
    ]
