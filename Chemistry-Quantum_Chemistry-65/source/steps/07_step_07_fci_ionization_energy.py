"""
Compute the exact (full configuration interaction) first ionization energy of a zero-differential-overlap pi Hamiltonian.

Full configuration interaction diagonalizes the Hamiltonian in the space of all Slater determinants with fixed numbers
of spin-up and spin-down electrons, which gives the exact energies within the model. For a Hamiltonian written in an
orthonormal site basis with zero differential overlap, a determinant is a pair of occupation strings, one per spin. The
diagonal element collects the site energies, the on-site repulsion of doubly occupied sites and the repulsion between
the charges of different sites, and the only off-diagonal elements are single hops of one electron between sites with
nonzero one-electron coupling, carrying the fermionic sign from the electrons of the same spin that it passes. The
first vertical ionization energy is the lowest energy of the system with one spin-down electron removed minus the lowest
energy of the neutral closed-shell system, both with the same Hamiltonian. The spin sector with one more spin-up than
spin-down electron contains every cation spin state, so its lowest root is the cation ground state.

Returns
-------
float, full configuration interaction first ionization energy in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fci_ionization_energy(h: "np.ndarray", gamma: "np.ndarray", n_occ: int) -> float:
    '''Full configuration interaction first ionization energy of a closed-shell zero-differential-overlap Hamiltonian.

    Parameters
    ----------
    h : np.ndarray
        Shape (n, n), symmetric one-electron Hamiltonian in the orthonormal site basis (eV).
    gamma : np.ndarray
        Shape (n, n), symmetric site repulsion matrix (eV). The two-electron energy of an occupation pattern is
        sum_p gamma_pp n_p,up n_p,down + sum_{p<q} gamma_pq n_p n_q, where n_p is the total occupation of site p.
    n_occ : int
        Number of electrons of each spin in the neutral system, 1 <= n_occ <= n.

    Returns
    -------
    ionization_energy : float
        E0(n_occ spin-up, n_occ - 1 spin-down) - E0(n_occ spin-up, n_occ spin-down) in eV, each E0 the lowest
        eigenvalue of the full configuration interaction matrix in that sector, as a Python float.

    Raises
    ------
    ValueError
        If h and gamma do not have the same square shape or n_occ is outside 1 <= n_occ <= n.
    '''
    return ionization_energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def _fci_lowest_energy(h, g, n_up, n_down):
    """Lowest eigenvalue of the zero-differential-overlap Hamiltonian with n_up and n_down electrons."""
    import itertools
    import numpy as np
    import scipy.sparse
    import scipy.sparse.linalg
    n = h.shape[0]

    def _strings(k):
        return [sum(1 << p for p in comb) for comb in itertools.combinations(range(n), k)]

    def _hop_matrix(strs):
        index = {s: j for j, s in enumerate(strs)}
        rows, cols, vals = [], [], []
        for j, s in enumerate(strs):
            for p in range(n):
                for q in range(n):
                    if p == q or h[p, q] == 0.0 or not (s >> q) & 1 or (s >> p) & 1:
                        continue
                    lo, hi = min(p, q), max(p, q)
                    passed = s & (((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1))
                    rows.append(index[s ^ (1 << q) ^ (1 << p)])
                    cols.append(j)
                    vals.append((-1.0) ** bin(passed).count('1') * h[p, q])
        return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(len(strs), len(strs)))

    up, down = _strings(n_up), _strings(n_down)
    occ_up = np.array([[(s >> p) & 1 for p in range(n)] for s in up], dtype=float).reshape(len(up), n)
    occ_dn = np.array([[(s >> p) & 1 for p in range(n)] for s in down], dtype=float).reshape(len(down), n)
    off = g - np.diag(np.diag(g))
    e_up = occ_up @ np.diag(h) + 0.5 * np.einsum('ap,pq,aq->a', occ_up, off, occ_up)
    e_dn = occ_dn @ np.diag(h) + 0.5 * np.einsum('bp,pq,bq->b', occ_dn, off, occ_dn)
    diag = e_up[:, None] + e_dn[None, :] + occ_up @ g @ occ_dn.T
    hmat = (scipy.sparse.kron(_hop_matrix(up), scipy.sparse.identity(len(down)))
            + scipy.sparse.kron(scipy.sparse.identity(len(up)), _hop_matrix(down))
            + scipy.sparse.diags(diag.reshape(-1)))
    if hmat.shape[0] <= 3000:
        return float(np.linalg.eigvalsh(hmat.toarray())[0])
    vals = scipy.sparse.linalg.eigsh(hmat.tocsr(), k=1, which='SA', tol=1e-12)[0]
    return float(vals[0])


def _oracle_fci_ionization_energy(h: "np.ndarray", gamma: "np.ndarray", n_occ: int) -> float:
    """Reference implementation."""
    import numpy as np
    h = np.asarray(h, dtype=float)
    g = np.asarray(gamma, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or g.shape != h.shape:
        raise ValueError("h and gamma must be square arrays of the same shape")
    n = h.shape[0]
    o = int(n_occ)
    if not 1 <= o <= n:
        raise ValueError("n_occ must satisfy 1 <= n_occ <= n")
    return _fci_lowest_energy(h, g, o, o - 1) - _fci_lowest_energy(h, g, o, o)

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
            "call": "fci_ionization_energy(h.copy(), g.copy(), 3)",
            "gold_call": "_oracle_fci_ionization_energy(h, g, 3)",
            "tol": 1e-8,
        },
        # --- Boundary: ethylene, where the cation has a single electron ---
        {
            "setup": chain + "h, g = chain(2, 1.5)\n",
            "call": "fci_ionization_energy(h.copy(), g.copy(), 1)",
            "gold_call": "_oracle_fci_ionization_energy(h, g, 1)",
            "tol": 1e-8,
        },
        # --- Edge: a four-site ring with a hop that closes the ring and a strong repulsion ---
        {
            "setup": chain + "h, g = chain(4, 1.8)\nh[0, 3] = h[3, 0] = -1.1\n",
            "call": "fci_ionization_energy(h.copy(), g.copy(), 2)",
            "gold_call": "_oracle_fci_ionization_energy(h, g, 2)",
            "tol": 1e-8,
        },
        # --- Normal: octatetraene with a weakened repulsion, a larger determinant space ---
        {
            "setup": chain + "h, g = chain(8, 0.6)\n",
            "call": "fci_ionization_energy(h.copy(), g.copy(), 4)",
            "gold_call": "_oracle_fci_ionization_energy(h, g, 4)",
            "tol": 1e-8,
        },
    ]
