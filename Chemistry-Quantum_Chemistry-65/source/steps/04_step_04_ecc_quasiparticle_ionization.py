"""
Obtain the highest-occupied quasiparticle ionization energy and its spectral weight from the equation-of-motion treatment of the doubly similarity-transformed electron-boson Hamiltonian.

In the electron-boson picture of the GW approximation an added or removed electron couples linearly to the direct-ring
particle-hole bosons. Transforming the electron-boson Hamiltonian first with the ring amplitudes t and then with the
extended coupled-cluster de-excitation amplitudes z block-diagonalizes the boson part, and an equation-of-motion
treatment in the space of one-hole and one-particle configurations plus hole-boson (2h1p) and particle-boson (2p1h)
configurations gives a non-Hermitian effective Hamiltonian. Its one-hole/one-particle block is the Fock matrix, its
2h1p and 2p1h blocks contain the orbital energy of the hole or particle minus or plus the dressed boson block, and the
couplings between the blocks are the direct electron-boson integrals dressed by t and z. Its eigenvalues coincide
exactly with the charged excitation energies of the full-frequency G0W0 supermatrix built on the same orbitals, and
supplying a modified Fock matrix in the one-hole/one-particle block is how static self-energy corrections enter.

Because the effective Hamiltonian is not symmetric, the spectral weight of an eigenstate on an orbital is the product
of the matching components of its left and right eigenvectors, normalized so that the left and right eigenvectors are
biorthonormal. The quasiparticle of the highest occupied orbital is the eigenstate with the largest such weight on that
orbital, and its ionization energy is minus its eigenvalue.

Returns
-------
numpy.ndarray of shape (2,): [highest-occupied quasiparticle ionization energy in eV, biorthogonal spectral weight]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ecc_quasiparticle_ionization(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int,
                                 fock: "np.ndarray") -> "np.ndarray":
    '''Ionization energy and spectral weight of the highest-occupied-orbital quasiparticle from the ECC effective Hamiltonian.

    Parameters
    ----------
    orbital_energies : np.ndarray
        Shape (n,), canonical Hartree-Fock orbital energies in ascending order (eV); orbitals 0 to n_occ-1 are doubly
        occupied.
    eri : np.ndarray
        Shape (n, n, n, n), real two-electron integrals (pq|rs) in chemists' notation in the same orbital basis (eV).
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ < n.
    fock : np.ndarray
        Shape (n, n), symmetric matrix placed in the one-hole/one-particle block (eV): diag(orbital_energies) for plain
        G0W0, or that matrix plus a static self-energy correction. The 2h1p and 2p1h blocks always use
        orbital_energies.

    Returns
    -------
    quasiparticle : np.ndarray
        Shape (2,), [ionization energy in eV, spectral weight]. The effective Hamiltonian is built from the closed-shell
        (spin-adapted) direct-ring amplitudes t and z of the same orbitals and screening, so its eigenvalues equal those
        of the corresponding G0W0 supermatrix. The weight of an eigenstate on orbital n_occ-1 is the product of the
        left and right eigenvector components on that orbital with the eigenvectors biorthonormal; the returned state
        is the one with the largest weight, and the ionization energy is minus its eigenvalue.

    Raises
    ------
    ValueError
        If n_occ is outside 1 <= n_occ < n, or eri or fock has the wrong shape.
    '''
    return quasiparticle

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ecc_quasiparticle_ionization(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int,
                                         fock: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    e = np.asarray(orbital_energies, dtype=float)
    v4 = np.asarray(eri, dtype=float)
    f = np.asarray(fock, dtype=float)
    n = e.shape[0]
    o = int(n_occ)
    if not 1 <= o < n:
        raise ValueError("n_occ must satisfy 1 <= n_occ < n")
    if v4.shape != (n, n, n, n) or f.shape != (n, n):
        raise ValueError("eri must have shape (n, n, n, n) and fock shape (n, n)")
    v = n - o
    m = o * v
    amps = _oracle_ring_ecc_amplitudes(e, v4, o)
    t, z = amps[0], amps[1]
    gap = (e[o:][None, :] - e[:o][:, None]).reshape(-1)
    k = v4[:o, o:, :o, o:].reshape(m, m)
    a_blk = np.diag(gap) + 2.0 * k
    b_blk = 2.0 * k
    eye = np.eye(m)
    # spin-adapted direct electron-boson coupling V_{pq,nu} = sqrt(2) (pq|ia)
    coup = np.sqrt(2.0) * v4[:, :, :o, o:].reshape(n, n, m)
    right = coup @ (eye + t)                  # N blocks
    left = coup @ (eye + z + t @ z)           # N-tilde blocks, (eye + z + t z) = (eye - t)^-1
    d_hole = np.kron(np.diag(e[:o]), eye) - np.kron(np.eye(o), a_blk + t @ b_blk)
    d_part = np.kron(np.diag(e[o:]), eye) + np.kron(np.eye(v), a_blk + b_blk @ t)
    zero = np.zeros((o * m, v * m))
    heff = np.block([
        [f, left[:, :o, :].reshape(n, o * m), right[:, o:, :].reshape(n, v * m)],
        [right[:, :o, :].reshape(n, o * m).T, d_hole, zero],
        [left[:, o:, :].reshape(n, v * m).T, zero.T, d_part],
    ])
    w, vr = np.linalg.eig(heff)
    vl = np.linalg.inv(vr)                    # rows are left eigenvectors, biorthonormal to the columns of vr
    weight = (vl[:, o - 1] * vr[o - 1, :]).real
    best = int(np.argmax(weight))
    return np.array([-w[best].real, weight[best]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    make = ("import numpy as np\n"
            "def mo_system(n, o, seed, scale):\n"
            "    rng = np.random.default_rng(seed)\n"
            "    q, _ = np.linalg.qr(rng.normal(size=(n, n)))\n"
            "    pos = np.cumsum(np.full(n, 1.4))\n"
            "    g = scale * 11.13 / np.sqrt(1.0 + (11.13 * np.abs(pos[:, None] - pos[None]) / 14.397) ** 2)\n"
            "    eri = np.einsum('mp,mq,mn,nr,ns->pqrs', q, q, g, q, q)\n"
            "    e = np.sort(rng.uniform(-12.0, -8.0, o)).tolist() + np.sort(rng.uniform(-2.0, 1.0, n - o)).tolist()\n"
            "    return np.array(e), eri\n")
    return [
        # --- Normal: plain G0W0 on four orbitals with two occupied ---
        {
            "setup": make + "e, eri = mo_system(4, 2, 3, 1.0)\nfock = np.diag(e)\n",
            "call": "ecc_quasiparticle_ionization(e.copy(), eri.copy(), 2, fock.copy())",
            "gold_call": "_oracle_ecc_quasiparticle_ionization(e, eri, 2, fock)",
            "tol": 1e-7,
        },
        # --- Boundary: a single occupied orbital and a Fock block with a static off-diagonal correction ---
        {
            "setup": make + "e, eri = mo_system(3, 1, 8, 1.2)\n"
                            "s = np.array([[0.06, -0.02, 0.01], [-0.02, -0.03, 0.015], [0.01, 0.015, 0.02]])\n"
                            "fock = np.diag(e) + s\n",
            "call": "ecc_quasiparticle_ionization(e.copy(), eri.copy(), 1, fock.copy())",
            "gold_call": "_oracle_ecc_quasiparticle_ionization(e, eri, 1, fock)",
            "tol": 1e-7,
        },
        # --- Edge: three occupied and three virtual orbitals with strong coupling that lowers the weight ---
        {
            "setup": make + "e, eri = mo_system(6, 3, 21, 2.2)\nfock = np.diag(e)\n",
            "call": "ecc_quasiparticle_ionization(e.copy(), eri.copy(), 3, fock.copy())",
            "gold_call": "_oracle_ecc_quasiparticle_ionization(e, eri, 3, fock)",
            "tol": 1e-7,
        },
    ]
