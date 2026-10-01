"""
Compute the direct-ring extended coupled-cluster doubles amplitudes that block-diagonalize the quasi-boson Hamiltonian.

When particle-hole excitations of a closed-shell reference are treated as ideal bosons, the direct (ring-only, no
exchange) part of the electron repulsion turns into a quadratic boson Hamiltonian with a number-conserving block A and
a pair-creation block B. For a spin-restricted reference the singlet particle-hole pairs (ia) of spatial orbitals are
the boson modes, and summing over the two spin channels doubles the direct integrals in both blocks. A similarity
transformation with the doubles excitation operator t removes the pair-creation terms when t solves the quadratic
Riccati equation B + A t + t A + t B t = 0; the physical root is the one built from the positive-frequency eigenvectors
of the direct random-phase approximation, t = Y X^-1, which is symmetric. A second transformation with the
de-excitation operator z of the extended coupled-cluster ansatz removes the remaining pair-annihilation terms. Both
transformations terminate exactly because the boson Hamiltonian is quadratic, and the left amplitudes follow in closed
form from the right ones.

Returns
-------
numpy.ndarray of shape (2, m, m) with m = n_occ (n - n_occ): [ring amplitudes t, ECC de-excitation amplitudes z]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ring_ecc_amplitudes(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    '''Right (t) and left (z) direct-ring extended coupled-cluster doubles amplitudes of a closed-shell reference.

    Parameters
    ----------
    orbital_energies : np.ndarray
        Shape (n,), canonical Hartree-Fock orbital energies in ascending order (eV); orbitals 0 to n_occ-1 are doubly
        occupied and every virtual energy exceeds every occupied energy.
    eri : np.ndarray
        Shape (n, n, n, n), real two-electron integrals (pq|rs) in chemists' notation in the same orbital basis (eV).
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ < n.

    Returns
    -------
    amplitudes : np.ndarray
        Shape (2, m, m) with m = n_occ * (n - n_occ). Element [0] is t and element [1] is z. The composite index of the
        spatial occupied-virtual pair (i, a) is i * (n - n_occ) + (a - n_occ). A and B are the spin-summed singlet
        direct-ring blocks, t is the symmetric root of B + A t + t A + t B t = 0 given by t = Y X^-1 from the
        positive-frequency direct random-phase eigenvectors, and z = t (1 - t t)^-1.

    Raises
    ------
    ValueError
        If n_occ is outside 1 <= n_occ < n or eri does not have shape (n, n, n, n).
    '''
    return amplitudes

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ring_ecc_amplitudes(orbital_energies: "np.ndarray", eri: "np.ndarray", n_occ: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    e = np.asarray(orbital_energies, dtype=float)
    v4 = np.asarray(eri, dtype=float)
    n = e.shape[0]
    o = int(n_occ)
    if not 1 <= o < n:
        raise ValueError("n_occ must satisfy 1 <= n_occ < n")
    if v4.shape != (n, n, n, n):
        raise ValueError("eri must have shape (n, n, n, n)")
    v = n - o
    gap = (e[o:][None, :] - e[:o][:, None]).reshape(-1)
    k = v4[:o, o:, :o, o:].reshape(o * v, o * v)
    a_blk = np.diag(gap) + 2.0 * k
    b_blk = 2.0 * k
    # positive-frequency eigenvectors of the direct RPA through the symmetric form (A-B)^1/2 (A+B) (A-B)^1/2
    root = np.sqrt(gap)
    omega2, vec = np.linalg.eigh(root[:, None] * (a_blk + b_blk) * root[None, :])
    omega = np.sqrt(omega2)
    x_plus_y = (root[:, None] * vec) / np.sqrt(omega)[None, :]
    x_minus_y = (vec / root[:, None]) * np.sqrt(omega)[None, :]
    x = 0.5 * (x_plus_y + x_minus_y)
    y = 0.5 * (x_plus_y - x_minus_y)
    t = y @ np.linalg.inv(x)
    t = 0.5 * (t + t.T)
    z = t @ np.linalg.inv(np.eye(o * v) - t @ t)
    return np.stack([t, z])

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
        # --- Normal: four orbitals, two occupied, ordinary screening ---
        {
            "setup": make + "e, eri = mo_system(4, 2, 3, 1.0)\n",
            "call": "ring_ecc_amplitudes(e.copy(), eri.copy(), 2)",
            "gold_call": "_oracle_ring_ecc_amplitudes(e, eri, 2)",
            "tol": 1e-8,
        },
        # --- Boundary: a single particle-hole pair (two orbitals, one occupied) ---
        {
            "setup": make + "e, eri = mo_system(2, 1, 11, 1.0)\n",
            "call": "ring_ecc_amplitudes(e.copy(), eri.copy(), 1)",
            "gold_call": "_oracle_ring_ecc_amplitudes(e, eri, 1)",
            "tol": 1e-8,
        },
        # --- Edge: asymmetric occupation (two occupied, five virtual) with strong coupling and large amplitudes ---
        {
            "setup": make + "e, eri = mo_system(7, 2, 5, 2.5)\n",
            "call": "ring_ecc_amplitudes(e.copy(), eri.copy(), 2)",
            "gold_call": "_oracle_ring_ecc_amplitudes(e, eri, 2)",
            "tol": 1e-8,
        },
    ]
