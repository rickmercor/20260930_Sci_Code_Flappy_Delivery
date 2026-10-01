"""
Build the flow-regularised second-order static self-energy matrix of the source method for a closed-shell reference, in its spin-integrated, spin-component-scaled form over spatial molecular orbitals.

In quasiparticle self-consistent second-order Green's-function theory the frequency-dependent
second-order self-energy is replaced by a static matrix that is added to the Hartree-Fock Fock
matrix. It contains a two-hole-one-particle and a two-particle-one-hole contribution. For a
closed-shell reference it can be written over spatial orbitals, and spin-component scaling
multiplies the same-spin and opposite-spin electron-pair contributions by separate factors
c_SS and c_OS.

Returns
-------
np.ndarray of shape (n, n): the static self-energy matrix over all molecular orbitals, in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_static_self_energy(eps: "np.ndarray", eri_mo: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    '''Flow-regularised, spin-component-scaled static second-order self-energy (MO basis).

    Implements the source method's static self-energy for a closed-shell reference in
    spatial orbitals: the spin integration of its spin-orbital expression, with the
    same-spin and opposite-spin pair contributions scaled by ``c_ss`` and ``c_os`` and
    the energy denominators formed from ``eps`` as in the source method. All orbitals
    are correlated (no frozen core).

    Parameters
    ----------
    eps : np.ndarray
        Orbital (quasiparticle) energies in hartree, shape (n,). Orbitals 0 .. n_occ-1
        are doubly occupied and n_occ .. n-1 are virtual.
    eri_mo : np.ndarray
        Two-electron integrals over the n real spatial orbitals in chemists' notation,
        eri_mo[p, q, r, s] holding (pq|rs), shape (n, n, n, n), with eightfold permutational
        symmetry.
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= n.
    s : float
        Flow parameter in hartree^-2, finite and non-negative.
    c_ss : float
        Scaling factor of the same-spin contributions.
    c_os : float
        Scaling factor of the opposite-spin contributions.

    Returns
    -------
    sigma : np.ndarray
        Static self-energy matrix F_pq(s) in hartree over all n orbitals (occupied and
        virtual), shape (n, n). It vanishes for s = 0 and when there are no virtual
        orbitals.

    Raises
    ------
    ValueError
        If the shapes of ``eps`` and ``eri_mo`` are inconsistent, n_occ lies outside
        [1, n], or ``s`` is negative or not finite.
    '''
    return sigma

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_static_self_energy(eps: "np.ndarray", eri_mo: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    eps = np.asarray(eps, dtype=float).ravel()
    g = np.asarray(eri_mo, dtype=float)
    n = eps.size
    if g.shape != (n, n, n, n):
        raise ValueError("eri_mo must have shape (n, n, n, n) with n = len(eps)")
    if not 1 <= int(n_occ) <= n:
        raise ValueError("n_occ must lie in [1, n]")
    n_occ = int(n_occ)
    occ, vir = slice(0, n_occ), slice(n_occ, n)
    e_o, e_v = eps[occ], eps[vir]
    c_same, c_opp = float(c_ss), float(c_os)

    # 2h1p term: A[p, i, a, j] = (pi|aj); exchange partner (qj|ai) = A[q, j, a, i]
    a_h = g[:, occ, vir, occ]
    b_h = (c_same + c_opp) * a_h - c_same * a_h.transpose(0, 3, 2, 1)
    d_h = (eps[:, None, None, None] - e_o[None, :, None, None]
           + e_v[None, None, :, None] - e_o[None, None, None, :])
    w_h = _oracle_compute_flow_regularized_weight(d_h[:, None], d_h[None, :], s)
    sigma = np.einsum("pqiaj,piaj,qiaj->pq", w_h, a_h, b_h, optimize=True)

    # 2p1h term: A[p, a, i, b] = (pa|ib); exchange partner (qb|ia) = A[q, b, i, a]
    a_p = g[:, vir, occ, vir]
    b_p = (c_same + c_opp) * a_p - c_same * a_p.transpose(0, 3, 2, 1)
    d_p = (eps[:, None, None, None] - e_v[None, :, None, None]
           + e_o[None, None, :, None] - e_v[None, None, None, :])
    w_p = _oracle_compute_flow_regularized_weight(d_p[:, None], d_p[None, :], s)
    sigma = sigma + np.einsum("pqaib,paib,qaib->pq", w_p, a_p, b_p, optimize=True)
    return 0.5 * (sigma + sigma.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    synthetic = """import numpy as np
def symmetric_eri(n, seed, scale):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, n, n, n))
    g = g + g.transpose(1, 0, 2, 3)
    g = g + g.transpose(0, 1, 3, 2)
    g = g + g.transpose(2, 3, 0, 1)
    return scale * g / 8.0
"""
    return [
        # --- Normal: H2 / STO-3G at R = 1.4 bohr, textbook MO integrals, unscaled ---
        {
            "setup": """import numpy as np
eps = np.array([-0.578, 0.6703])
g = np.zeros((2, 2, 2, 2))
g[0, 0, 0, 0] = 0.6746
g[1, 1, 1, 1] = 0.6975
g[0, 0, 1, 1] = g[1, 1, 0, 0] = 0.6636
for idx in [(0, 1, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0), (1, 0, 1, 0)]:
    g[idx] = 0.1813
""",
            "call": "compute_static_self_energy(eps.copy(), g.copy(), 1, 0.5, 1.0, 1.0)",
            "gold_call": "_oracle_compute_static_self_energy(eps.copy(), g.copy(), 1, 0.5, 1.0, 1.0)",
            "tol": 1e-10,
        },
        # --- Normal: 7 orbitals, 3 occupied, generic spin-component scaling ---
        {
            "setup": synthetic + """
eps = np.array([-1.9, -0.85, -0.47, 0.21, 0.55, 1.1, 2.4])
g = symmetric_eri(7, 11, 0.3)
""",
            "call": "compute_static_self_energy(eps.copy(), g.copy(), 3, 0.9, 0.45, 1.25)",
            "gold_call": "_oracle_compute_static_self_energy(eps.copy(), g.copy(), 3, 0.9, 0.45, 1.25)",
            "tol": 1e-10,
        },
        # --- Edge: exact energy coincidences give vanishing denominators (removable 0/0),
        #     opposite-spin channel only ---
        {
            "setup": synthetic + """
eps = np.array([-2.0, -1.0, -0.5, 0.5, 1.5])
g = symmetric_eri(5, 5, 0.4)
""",
            "call": "compute_static_self_energy(eps.copy(), g.copy(), 3, 1.1, 0.0, 1.3)",
            "gold_call": "_oracle_compute_static_self_energy(eps.copy(), g.copy(), 3, 1.1, 0.0, 1.3)",
            "tol": 1e-10,
        },
        # --- Edge: same energy coincidences with only the same-spin channel, one virtual orbital ---
        {
            "setup": synthetic + """
eps = np.array([-2.0, -1.0, -0.5, 0.5])
g = symmetric_eri(4, 9, 0.5)
""",
            "call": "compute_static_self_energy(eps.copy(), g.copy(), 3, 0.6, 0.8, 0.0)",
            "gold_call": "_oracle_compute_static_self_energy(eps.copy(), g.copy(), 3, 0.6, 0.8, 0.0)",
            "tol": 1e-10,
        },
        # --- Boundary: s = 0 (no flow, the correlation contribution vanishes) ---
        {
            "setup": synthetic + """
eps = np.array([-1.3, -0.6, 0.4, 0.9])
g = symmetric_eri(4, 3, 0.3)
""",
            "call": "compute_static_self_energy(eps.copy(), g.copy(), 2, 0.0, 1.0, 1.0)",
            "gold_call": "_oracle_compute_static_self_energy(eps.copy(), g.copy(), 2, 0.0, 1.0, 1.0)",
            "tol": 1e-10,
        },
    ]
