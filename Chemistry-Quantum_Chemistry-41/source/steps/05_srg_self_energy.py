"""
Implement srg_self_energy, which builds the renormalized static second-order self-energy
matrix, with separate scaling of its same-spin and opposite-spin parts, in the basis of the
current molecular orbitals.

The self-energy is spin integrated and expressed over the real spatial orbitals of a closed-shell
reference. As in spin-component-scaled second-order perturbation theory, the contributions
of same-spin and opposite-spin electron pairs can be weighted separately.

Returns
-------
np.ndarray of shape (n, n): symmetric spin-component-scaled renormalized static second-order self-energy in the MO basis, in hartree
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def srg_self_energy(eri_mo: "np.ndarray", orbital_energies: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    '''Spin-component-scaled renormalized static second-order self-energy (MO basis).

    Parameters
    ----------
    eri_mo : np.ndarray
        Electron-repulsion integrals over the current real spatial orbitals in chemists'
        notation, eri_mo[p, q, r, s] = (pq|rs), shape (n, n, n, n), in hartree.
    orbital_energies : np.ndarray
        Current orbital energies e_p, shape (n,), in hartree; orbitals 0..n_occ-1 are the
        doubly occupied ones (indices i, j), the others are virtual (indices a, b).
    n_occ : int
        Number of doubly occupied orbitals, 1 <= n_occ <= n - 1.
    s : float
        Flow parameter in hartree^-2 (finite, non-negative).
    c_ss : float
        Same-spin scaling factor.
    c_os : float
        Opposite-spin scaling factor.

    Returns
    -------
    sigma : np.ndarray
        Symmetric matrix of shape (n, n), in hartree:
        sigma[p, q] = sum_{i,j,a} f(D^{pa}_{ij}, D^{qa}_{ij}; s) (pi|aj) [(c_ss + c_os)(qi|aj) - c_ss (qj|ai)]
                    + sum_{a,b,i} f(D^{pi}_{ab}, D^{qi}_{ab}; s) (pa|ib) [(c_ss + c_os)(qa|ib) - c_ss (qb|ia)],
        with D^{pa}_{ij} = e_p + e_a - e_i - e_j, D^{pi}_{ab} = e_p + e_i - e_a - e_b and f the
        factor of srg_regulator.

    Raises
    ------
    ValueError
        If the shapes are inconsistent, n_occ is outside 1..n-1, c_ss or c_os is not finite,
        or s is invalid as in srg_regulator.
    '''
    return sigma

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_srg_self_energy(eri_mo: "np.ndarray", orbital_energies: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    eri_mo = np.asarray(eri_mo, dtype=float)
    eps = np.asarray(orbital_energies, dtype=float).ravel()
    n = eps.size
    if eri_mo.shape != (n, n, n, n):
        raise ValueError("eri_mo must have shape (n, n, n, n) matching orbital_energies")
    if int(n_occ) != n_occ or not 1 <= n_occ <= n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    if not (np.isfinite(c_ss) and np.isfinite(c_os)):
        raise ValueError("scaling factors must be finite")
    n_occ = int(n_occ)
    occ, vir = slice(0, n_occ), slice(n_occ, n)
    e_occ, e_vir = eps[occ], eps[vir]
    # two-hole-one-particle part: (pi|aj) with D^{pa}_{ij}
    hole = eri_mo[:, occ, vir, occ]
    hole_scaled = (c_ss + c_os) * hole - c_ss * hole.transpose(0, 3, 2, 1)
    d_hole = eps[:, None, None, None] + e_vir[None, None, :, None] - e_occ[None, :, None, None] - e_occ[None, None, None, :]
    f_hole = _oracle_srg_regulator(d_hole[:, None], d_hole[None, :], s)
    sigma = np.einsum("piaj,pqiaj,qiaj->pq", hole, f_hole, hole_scaled, optimize=True)
    # two-particle-one-hole part: (pa|ib) with D^{pi}_{ab}
    part = eri_mo[:, vir, occ, vir]
    part_scaled = (c_ss + c_os) * part - c_ss * part.transpose(0, 3, 2, 1)
    d_part = eps[:, None, None, None] + e_occ[None, None, :, None] - e_vir[None, :, None, None] - e_vir[None, None, None, :]
    f_part = _oracle_srg_regulator(d_part[:, None], d_part[None, :], s)
    sigma = sigma + np.einsum("paib,pqaib,qaib->pq", part, f_part, part_scaled, optimize=True)
    return 0.5 * (sigma + sigma.T)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    helper = """import numpy as np
def model(n, n_occ, seed, gap=0.6, spread=0.9):
    # 8-fold symmetric positive-semidefinite 'integrals' and an ordered orbital spectrum
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(2 * n, n, n)) * 0.12
    L = L + L.transpose(0, 2, 1)
    eri = np.einsum('xpq,xrs->pqrs', L, L)
    occ = -np.sort(rng.uniform(0.3, 0.3 + spread, n_occ))[::-1] - gap / 2
    vir = np.sort(rng.uniform(0.0, spread, n - n_occ)) + gap / 2
    return eri, np.concatenate([np.sort(occ), vir])
"""
    invalid = """
def run_model():
    try:
        srg_self_energy(eri.copy(), eps.copy(), n_occ, 1.4, 0.0, 1.0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_srg_self_energy(eri.copy(), eps.copy(), n_occ, 1.4, 0.0, 1.0)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Typical: opposite-spin-only scaling with s = 1.4 (7 orbitals, 3 occupied) ---
        {
            "setup": helper + """
eri, eps = model(7, 3, seed=11)
""",
            "call": "srg_self_energy(eri.copy(), eps.copy(), 3, 1.4, 0.0, 1.0)",
            "gold_call": "_oracle_srg_self_energy(eri.copy(), eps.copy(), 3, 1.4, 0.0, 1.0)",
            "tol": 1e-10,
        },
        # --- Typical: unscaled self-energy (c_ss = c_os = 1) with s = 0.525 ---
        {
            "setup": helper + """
eri, eps = model(8, 4, seed=5)
""",
            "call": "srg_self_energy(eri.copy(), eps.copy(), 4, 0.525, 1.0, 1.0)",
            "gold_call": "_oracle_srg_self_energy(eri.copy(), eps.copy(), 4, 0.525, 1.0, 1.0)",
            "tol": 1e-10,
        },
        # --- Typical: both spin components weighted differently (exchange terms active) ---
        {
            "setup": helper + """
eri, eps = model(6, 2, seed=23)
""",
            "call": "srg_self_energy(eri.copy(), eps.copy(), 2, 0.7, 0.6, 1.0)",
            "gold_call": "_oracle_srg_self_energy(eri.copy(), eps.copy(), 2, 0.7, 0.6, 1.0)",
            "tol": 1e-10,
        },
        # --- Boundary: s = 0 gives the zero matrix ---
        {
            "setup": helper + """
eri, eps = model(6, 3, seed=2)
""",
            "call": "srg_self_energy(eri.copy(), eps.copy(), 3, 0.0, 0.0, 1.0)",
            "gold_call": "_oracle_srg_self_energy(eri.copy(), eps.copy(), 3, 0.0, 0.0, 1.0)",
            "tol": 1e-12,
        },
        # --- Edge: a single occupied orbital and a nearly closed gap (small denominators),
        #     with a large flow parameter ---
        {
            "setup": helper + """
eri, eps = model(6, 1, seed=31, gap=0.02, spread=0.4)
""",
            "call": "srg_self_energy(eri.copy(), eps.copy(), 1, 50.0, 0.33, 1.2)",
            "gold_call": "_oracle_srg_self_energy(eri.copy(), eps.copy(), 1, 50.0, 0.33, 1.2)",
            "tol": 1e-10,
        },
        # --- Invalid: every orbital occupied (no virtual orbital) ---
        {
            "setup": helper + """
eri, eps = model(5, 4, seed=3)
n_occ = 5
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
