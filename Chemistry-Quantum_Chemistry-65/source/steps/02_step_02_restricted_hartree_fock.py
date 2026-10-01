"""
Solve the closed-shell restricted Hartree-Fock equations of a zero-differential-overlap pi Hamiltonian.

With zero differential overlap the two-electron integrals in the site basis reduce to (pq|rs) = delta_pq delta_rs
gamma_pr, so the Fock matrix of a closed-shell determinant needs only the spin-summed density matrix
D = 2 C_occ C_occ^T. The Coulomb term puts the potential of all site charges on the diagonal and the exchange term
couples sites in proportion to their density-matrix element and their repulsion. The canonical orbitals diagonalize
the converged Fock matrix, the lowest n_occ orbitals are doubly occupied, and the orbital energies later serve as the
quasiparticle reference for the Green's function and coupled-cluster treatments.

Canonical orbitals are defined only up to sign. Correlated quantities built from them are sign-independent, but
orbital-basis integrals are not, so a fixed phase convention is used.

Returns
-------
numpy.ndarray of shape (n + 1, n): row 0 ascending orbital energies in eV, rows 1..n the sign-fixed orbital coefficient matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def restricted_hartree_fock(h: "np.ndarray", gamma: "np.ndarray", n_occ: int) -> "np.ndarray":
    '''Canonical closed-shell Hartree-Fock orbital energies and orbitals of a zero-differential-overlap Hamiltonian.

    Parameters
    ----------
    h : np.ndarray
        Shape (n, n), symmetric one-electron core Hamiltonian in the orthonormal site basis (eV).
    gamma : np.ndarray
        Shape (n, n), symmetric site repulsion matrix (eV); the only nonzero two-electron integrals are
        (pp|qq) = gamma_pq.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ < n.

    Returns
    -------
    result : np.ndarray
        Shape (n + 1, n). Row 0 holds the orbital energies in ascending order. Rows 1 to n hold the orbital coefficient
        matrix C, whose column k is orbital k in the site basis, normalized, and signed so that its first component
        (lowest site index) with magnitude above 1e-6 is positive. The Fock matrix is
        F = h + diag(gamma @ diag(D)) - D * gamma / 2 (elementwise product) with D = 2 C_occ C_occ^T, converged until
        no element of D changes by more than 1e-12 between iterations.

    Raises
    ------
    ValueError
        If h and gamma do not have the same square shape or n_occ is outside 1 <= n_occ < n.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_restricted_hartree_fock(h: "np.ndarray", gamma: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    h = np.asarray(h, dtype=float)
    g = np.asarray(gamma, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or g.shape != h.shape:
        raise ValueError("h and gamma must be square arrays of the same shape")
    n = h.shape[0]
    o = int(n_occ)
    if not 1 <= o < n:
        raise ValueError("n_occ must satisfy 1 <= n_occ < n")

    def _fock(dens):
        return h + np.diag(g @ np.diag(dens)) - 0.5 * dens * g

    e, c = np.linalg.eigh(h)
    dens = 2.0 * c[:, :o] @ c[:, :o].T
    for _ in range(2000):
        e, c = np.linalg.eigh(_fock(dens))
        new = 2.0 * c[:, :o] @ c[:, :o].T
        converged = np.abs(new - dens).max() < 1e-12
        dens = 0.5 * (dens + new)   # damping keeps strongly scaled repulsions from oscillating
        if converged:
            dens = new
            break
    e, c = np.linalg.eigh(_fock(dens))
    lead = np.argmax(np.abs(c) > 1e-6, axis=0)     # first site with a non-negligible coefficient
    c = c * np.sign(c[lead, np.arange(n)])
    return np.vstack([e[None, :], c])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    chain = ("import numpy as np\n"
             "def chain(n, lam, w=11.16, u=11.13, ts=-2.6, tl=-2.2, rs=1.35, rl=1.46):\n"
             "    pos = np.zeros((n, 2))\n"
             "    for k in range(n - 1):\n"
             "        r = rs if k % 2 == 0 else rl\n"
             "        a = np.deg2rad(30.0 if k % 2 == 0 else -30.0)\n"
             "        pos[k + 1] = pos[k] + r * np.array([np.cos(a), np.sin(a)])\n"
             "    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)\n"
             "    g = lam * u / np.sqrt(1.0 + (u * d / 14.397) ** 2)\n"
             "    h = np.diag([ts if k % 2 == 0 else tl for k in range(n - 1)], 1)\n"
             "    h = h + h.T - np.diag(w + g.sum(1) - np.diag(g))\n"
             "    return h, g\n")
    return [
        # --- Normal: hexatriene at the standard interaction strength ---
        {
            "setup": chain + "h, g = chain(6, 1.0)\n",
            "call": "restricted_hartree_fock(h.copy(), g.copy(), 3)",
            "gold_call": "_oracle_restricted_hartree_fock(h, g, 3)",
            "tol": 1e-7,
        },
        # --- Boundary: ethylene, one occupied orbital and a 2 x 2 problem ---
        {
            "setup": chain + "h, g = chain(2, 1.4)\n",
            "call": "restricted_hartree_fock(h.copy(), g.copy(), 1)",
            "gold_call": "_oracle_restricted_hartree_fock(h, g, 1)",
            "tol": 1e-7,
        },
        # --- Edge: octatetraene with a doubled repulsion, where the self-consistent field is far from the core guess ---
        {
            "setup": chain + "h, g = chain(8, 2.0)\n",
            "call": "restricted_hartree_fock(h.copy(), g.copy(), 4)",
            "gold_call": "_oracle_restricted_hartree_fock(h, g, 4)",
            "tol": 1e-7,
        },
        # --- Invalid: as many occupied orbitals as sites must raise ValueError ---
        {
            "setup": chain + "h, g = chain(4, 1.0)\n"
                             "def run(fn):\n"
                             "    try:\n"
                             "        fn(h, g, 4)\n"
                             "        return 0\n"
                             "    except ValueError:\n"
                             "        return 1\n",
            "call": "run(restricted_hartree_fock)",
            "gold_call": "run(_oracle_restricted_hartree_fock)",
        },
    ]
