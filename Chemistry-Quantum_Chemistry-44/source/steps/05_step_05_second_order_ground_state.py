"""
Step 05 - Second-order ground-state amplitudes and energies for a non-canonical zeroth-order Hamiltonian.

Once the orbitals are dressed, the zeroth-order Hamiltonian H0 = sum over p of e_p a+_p a_p is diagonal, but the Fock matrix of the reference determinant is not: it keeps off-diagonal elements inside the occupied block and inside the virtual block, while its occupied-virtual block still vanishes because the reference is a Hartree-Fock determinant. The perturbation is everything in H - H0, so its one-particle part is f - diag(e) and acts within the occupied and within the virtual space.

This step takes spin-orbital arrays directly, which makes it independent of how they were produced: the zeroth-order energies e_p, the Fock matrix f_pq and the antisymmetrised integrals <pq||rs>, with the n_occ occupied spin orbitals listed first. Use Rayleigh-Schrödinger perturbation theory in intermediate normalisation and write the ground state as

Psi = Phi_0 + sum over i, a of t_i^a a+_a a_i Phi_0 + (1/4) sum over i, j, a, b of t_ij^ab a+_a a+_b a_i a_j Phi_0

With this convention the first-order doubles are t_ij^ab(1) = <ij||ab> / (e_a + e_b - e_i - e_j) and there are no first-order singles. Return E(2), the third-order energy E(3) of the same expansion, the second-order doubles t_ij^ab(2) and the second-order singles t_i^a(2). The second-order doubles carry a contribution from the one-particle part of the perturbation that canonical Møller-Plesset theory does not have; derive it from the definitions above.

Returns
-------
numpy.ndarray, one-dimensional: E(2) and E(3) in hartree followed by the second-order doubles and singles amplitudes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def second_order_ground_state(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike, eri: npt.ArrayLike, n_occ: int) -> np.ndarray:
    '''Second-order Rayleigh-Schrödinger ground state for a diagonal H0 and a non-diagonal Fock matrix.

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
    ground_state : numpy.ndarray
        One-dimensional array of length 2 + n_occ**2 * n_vir**2 + n_occ * n_vir,
        with n_vir = m - n_occ. Element 0 is E(2) and element 1 is E(3), in
        hartree. Then follow the second-order doubles t_ij^ab(2) for all
        occupied i, j and virtual a, b (virtual indices counted from 0 within
        the virtual block) in C order of (i, j, a, b), and finally the
        second-order singles t_i^a(2) in C order of (i, a). Amplitudes are
        coefficients of a+_a a+_b a_i a_j Phi_0 (with the factor 1/4 of the
        expansion) and of a+_a a_i Phi_0 in the second-order correction
        Psi(2) to the intermediate-normalised wavefunction.

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
    return ground_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _checked_spin_orbital_hamiltonian(orbital_energies, fock, eri, n_occ):
    """Validate zeroth-order energies, Fock matrix and antisymmetrised integrals; return arrays."""
    import numpy as np
    e = np.asarray(orbital_energies, dtype=float)
    f = np.asarray(fock, dtype=float)
    g = np.asarray(eri, dtype=float)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("orbital_energies must be one-dimensional with at least two entries")
    m = e.size
    if f.shape != (m, m) or g.shape != (m, m, m, m):
        raise ValueError("fock must be (m, m) and eri (m, m, m, m) for m orbital energies")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(f)) and np.all(np.isfinite(g))):
        raise ValueError("inputs must be finite")
    no = float(n_occ)
    if no != int(no) or not (0 < int(no) < m):
        raise ValueError("n_occ must be an integer strictly between 0 and the number of orbitals")
    no = int(no)
    if np.max(np.abs(f - f.T)) > 1.0e-10 or np.max(np.abs(f[:no, no:])) > 1.0e-10:
        raise ValueError("fock must be symmetric with a vanishing occupied-virtual block")
    if (np.max(np.abs(g + g.transpose(1, 0, 2, 3))) > 1.0e-10
            or np.max(np.abs(g + g.transpose(0, 1, 3, 2))) > 1.0e-10
            or np.max(np.abs(g - g.transpose(2, 3, 0, 1))) > 1.0e-10):
        raise ValueError("eri must be antisymmetric in each index pair and symmetric under pair exchange")
    if np.min(e[no:]) <= np.max(e[:no]):
        raise ValueError("every virtual orbital energy must lie above every occupied one")
    return e, f, g, no


def _oracle_second_order_ground_state(orbital_energies: npt.ArrayLike, fock: npt.ArrayLike,
                                      eri: npt.ArrayLike, n_occ: int) -> np.ndarray:
    import numpy as np
    e, f, g, no = _checked_spin_orbital_hamiltonian(orbital_energies, fock, eri, n_occ)
    nv = e.size - no
    o = slice(0, no)
    v = slice(no, no + nv)
    eo, ev = e[:no], e[no:]
    D = _pair_denominators(eo, ev)
    t = g[o, o, v, v] / D
    X = np.einsum('ikac,kbjc->ijab', t, g[o, v, o, v])
    X = X - X.transpose(1, 0, 2, 3) - X.transpose(0, 1, 3, 2) + X.transpose(1, 0, 3, 2)
    X = X - 0.5 * np.einsum('klab,ijkl->ijab', t, g[o, o, o, o]) - 0.5 * np.einsum('ijcd,abcd->ijab', t, g[v, v, v, v])
    F = (np.einsum('ac,ijcb->ijab', f[v, v], t) + np.einsum('bc,ijac->ijab', f[v, v], t)
         - np.einsum('ik,kjab->ijab', f[o, o], t) - np.einsum('jk,ikab->ijab', f[o, o], t))
    t2 = (X - F) / D + t
    t1 = -(0.5 * np.einsum('ijbc,jabc->ia', t, g[o, v, v, v])
           + 0.5 * np.einsum('jkab,jkib->ia', t, g[o, o, o, v])) / (ev[None, :] - eo[:, None])
    E2 = -0.25 * np.sum(g[o, o, v, v] * t)
    E3 = -0.25 * np.sum(g[o, o, v, v] * t2)
    return np.concatenate([np.array([E2, E3]), t2.ravel(), t1.ravel()])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: restricted model Hamiltonian with a non-diagonal Fock matrix ---
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
F[0, 1] = F[1, 0] = -0.03
F[2, 3] = F[3, 2] = 0.04
F[3, 4] = F[4, 3] = -0.025
F[2, 4] = F[4, 2] = 0.02
orbital_energies = np.repeat(np.diag(F) + np.array([0.015, -0.02, 0.01, -0.015, 0.02]), 2)
fock = np.kron(F, np.eye(2))
n_occ = 2 * n_occ_spatial
""",
            "call": "second_order_ground_state(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "gold_call": "_oracle_second_order_ground_state(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "tol": 1e-09,
        },
        # --- Boundary: diagonal Fock matrix equal to H0, the canonical limit ---
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
""",
            "call": "second_order_ground_state(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "gold_call": "_oracle_second_order_ground_state(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "tol": 1e-09,
        },
        # --- Edge: no spin structure at all and an odd number of occupied orbitals ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
m, n_occ = 6, 3
A = rng.normal(scale=0.005, size=(m, m, m, m))
A = A + A.transpose(1, 0, 2, 3)
A = A + A.transpose(0, 1, 3, 2)
A = A + A.transpose(2, 3, 0, 1)
A = A + 0.25 * np.einsum('pq,rs->pqrs', np.eye(m), np.eye(m))
V = A.transpose(0, 2, 1, 3)
eri = V - V.transpose(0, 1, 3, 2)
fock = np.diag([-1.1, -0.8, -0.55, 0.25, 0.6, 1.05])
fock[0, 2] = fock[2, 0] = 0.03
fock[1, 2] = fock[2, 1] = -0.02
fock[3, 5] = fock[5, 3] = 0.04
orbital_energies = np.diag(fock) + np.array([0.02, -0.01, 0.015, -0.02, 0.01, 0.0])
""",
            "call": "second_order_ground_state(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "gold_call": "_oracle_second_order_ground_state(orbital_energies.copy(), fock.copy(), eri.copy(), n_occ)",
            "tol": 1e-09,
        },
    ]
