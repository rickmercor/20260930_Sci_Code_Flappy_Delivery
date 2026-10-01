"""
Transform a Hubbard Hamiltonian to unrestricted orbitals and extract the coefficients of its local-seniority-conserving part.

Pairing each spin-up orbital $(p,\mathrm{up})$ with the spin-down orbital $(p,\mathrm{dn})$ of the same level defines, for every level, the operators $N_p=n_{p,\mathrm{up}}+n_{p,\mathrm{dn}}$, $S^z_p=\tfrac12(n_{p,\mathrm{up}}-n_{p,\mathrm{dn}})$, $P^\dagger_p=c^\dagger_{p,\mathrm{up}}c^\dagger_{p,\mathrm{dn}}$ and $S^+_p=c^\dagger_{p,\mathrm{up}}c_{p,\mathrm{dn}}$, which generate an so(4) algebra on each level. Keeping only the terms of the electronic Hamiltonian that conserve the seniority of every level gives



$$H_{\delta\Omega=0}=E_0+\sum_p \epsilon_p N_p+\sum_{pq}L_{pq}P^\dagger_p P_q+\frac14\sum_{p\neq q}W_{pq}N_pN_q+\sum_p B_pS^z_p-\sum_{p\neq q}K^{\mathrm{ud}}_{pq}S^+_pS^-_q+\sum_{p\neq q}B_{pq}S^z_pS^z_q+\sum_{p\neq q}X_{pq}N_pS^z_q ,$$



whose coefficients are combinations of the one-electron integrals $h^\sigma_{pp}$ and of Coulomb and exchange integrals $J^{\sigma\sigma'}_{pq}=v^{\sigma\sigma'}_{pqpq}$ and $K^{\sigma\sigma'}_{pq}=v^{\sigma\sigma'}_{pqqp}$ in the unrestricted orbital basis. The term with $X_{pq}$ couples the charge of one level to the spin of another and vanishes when the spin-up and spin-down orbitals coincide.

Returns
-------
np.ndarray of shape (7, M, M): [diag(eps), diag(B), L, W, Bz, Kud, X]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def seniority_coefficients(hopping: "np.ndarray", onsite_U: float, orbitals: "np.ndarray") -> "np.ndarray":
    '''Return the coefficients of the local-seniority-conserving Hubbard Hamiltonian.

    Pair spin-up and spin-down orbitals with the same level index and use the unrestricted
    orbital basis stored in ``orbitals``. Column ``p`` of ``orbitals[0]`` is the spin-up
    orbital for level ``p`` and column ``p`` of ``orbitals[1]`` is the corresponding
    spin-down orbital. For the site-basis Hubbard Hamiltonian the scalar constant is E0 = 0
    and is not returned.

    Parameters
    ----------
    hopping : np.ndarray
        Real symmetric M x M one-electron matrix in the site basis.
    onsite_U : float
        On-site Hubbard interaction strength.
    orbitals : np.ndarray
        Real array of shape (2, M, M) containing the spin-up and spin-down orbital matrices.

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (7, M, M), ordered as
        [diag(eps), diag(B), L, W, Bz, Kud, X] for the Hamiltonian written in the step
        background. The first two entries are diagonal matrices; W, Bz, Kud and X have
        zero diagonals, while L includes its diagonal terms.

    Raises
    ------
    ValueError
        If hopping is not square and symmetric or orbitals does not have shape (2, M, M).
    '''
    return coefficients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_seniority_coefficients(hopping: "np.ndarray", onsite_U: float, orbitals: "np.ndarray") -> "np.ndarray":
    h = np.asarray(hopping)
    c = np.asarray(orbitals)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.allclose(h, h.T, atol=1e-12):
        raise ValueError("hopping must be a square symmetric matrix")
    m = h.shape[0]
    if c.shape != (2, m, m):
        raise ValueError("orbitals must have shape (2, M, M)")
    c_up, c_dn = c[0], c[1]
    h_up = c_up.T @ h @ c_up
    h_dn = c_dn.T @ h @ c_dn
    sq_up, sq_dn, mixed = c_up * c_up, c_dn * c_dn, c_up * c_dn
    j_uu = onsite_U * sq_up.T @ sq_up
    j_dd = onsite_U * sq_dn.T @ sq_dn
    j_ud = onsite_U * sq_up.T @ sq_dn
    k_uu, k_dd = j_uu, j_dd
    k_ud = onsite_U * mixed.T @ mixed
    pair = onsite_U * mixed.T @ mixed
    off = 1.0 - np.eye(m)
    eps = 0.5 * (np.diag(h_up) + np.diag(h_dn))
    b_single = np.diag(h_up) - np.diag(h_dn)
    w = 0.5 * j_uu + 0.5 * j_dd + j_ud - 0.5 * k_uu - 0.5 * k_dd
    b_pair = 0.5 * j_uu + 0.5 * j_dd - j_ud - 0.5 * k_uu - 0.5 * k_dd
    x = 0.5 * (j_uu - j_dd) - 0.5 * (k_uu - k_dd) - 0.5 * (j_ud - j_ud.T)
    return np.stack([np.diag(eps), np.diag(b_single), pair, w * off, b_pair * off, k_ud * off, x * off])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    ring = """import numpy as np
from scipy.linalg import expm
def ring(m, t):
    h = np.zeros((m, m))
    for i in range(m):
        h[i, (i + 1) % m] = h[(i + 1) % m, i] = -t
    return h
def rot(seed, m, scale):
    a = np.random.default_rng(seed).normal(scale=scale, size=(m, m))
    return expm(a - a.T)
"""
    return [
        # Normal: 8-site ring, unrestricted orbitals (X coupling nonzero)
        {
            "setup": ring + "h = ring(8, 1.0)\ncu = rot(0, 8, 0.6)\ncd = cu @ rot(1, 8, 0.4)\norb = np.stack([cu, cd])\n",
            "call": "seniority_coefficients(h.copy(), 4.0, orb.copy())",
            "gold_call": "_oracle_seniority_coefficients(h.copy(), 4.0, orb.copy())",
        },
        # Normal: open chain with site-dependent hopping, 6 levels
        {
            "setup": ring + "h = np.diag([0.3, -0.2, 0.1, 0.0, 0.2, -0.1]) - np.diag([1.0, 0.8, 1.2, 0.9, 1.1], 1) - np.diag([1.0, 0.8, 1.2, 0.9, 1.1], -1)\norb = np.stack([rot(2, 6, 0.8), rot(3, 6, 0.8)])\n",
            "call": "seniority_coefficients(h.copy(), 2.5, orb.copy())",
            "gold_call": "_oracle_seniority_coefficients(h.copy(), 2.5, orb.copy())",
        },
        # Boundary: restricted orbitals (C_up = C_dn): B and X vanish, Bz = -Kud
        {
            "setup": ring + "h = ring(6, 1.0)\nc = rot(4, 6, 0.9)\norb = np.stack([c, c])\n",
            "call": "seniority_coefficients(h.copy(), 4.0, orb.copy())",
            "gold_call": "_oracle_seniority_coefficients(h.copy(), 4.0, orb.copy())",
        },
        # Edge: site basis (identity orbitals): only on-site L survives among the two-electron terms
        {
            "setup": ring + "h = ring(4, 1.0)\norb = np.stack([np.eye(4), np.eye(4)])\n",
            "call": "seniority_coefficients(h.copy(), 3.0, orb.copy())",
            "gold_call": "_oracle_seniority_coefficients(h.copy(), 3.0, orb.copy())",
        },
        # Held-out generalization: odd number of levels and dense non-circulant hopping
        {
            "setup": ring + "rng = np.random.default_rng(51)\na = rng.normal(scale=0.45, size=(5, 5))\nh = 0.5 * (a + a.T)\norb = np.stack([rot(52, 5, 0.55), rot(53, 5, 0.75)])\n",
            "call": "seniority_coefficients(h.copy(), 1.7, orb.copy())",
            "gold_call": "_oracle_seniority_coefficients(h.copy(), 1.7, orb.copy())",
        },
        # Held-out generalization: seven levels with unrelated unrestricted rotations
        {
            "setup": ring + "rng = np.random.default_rng(61)\na = rng.normal(scale=0.35, size=(7, 7))\nh = 0.5 * (a + a.T)\norb = np.stack([rot(62, 7, 0.35), rot(63, 7, 0.95)])\n",
            "call": "seniority_coefficients(h.copy(), 3.3, orb.copy())",
            "gold_call": "_oracle_seniority_coefficients(h.copy(), 3.3, orb.copy())",
        },
        # Boundary: zero interaction on a dense five-level one-electron Hamiltonian
        {
            "setup": ring + "rng = np.random.default_rng(71)\na = rng.normal(scale=0.4, size=(5, 5))\nh = 0.5 * (a + a.T)\norb = np.stack([rot(72, 5, 0.6), rot(73, 5, 0.4)])\n",
            "call": "seniority_coefficients(h.copy(), 0.0, orb.copy())",
            "gold_call": "_oracle_seniority_coefficients(h.copy(), 0.0, orb.copy())",
        },
        # Invalid: non-symmetric hopping
        {
            "setup": """import numpy as np
h = np.array([[0.0, -1.0], [-0.5, 0.0]])
orb = np.stack([np.eye(2), np.eye(2)])
def run_model():
    try:
        seniority_coefficients(h.copy(), 1.0, orb.copy())
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_seniority_coefficients(h.copy(), 1.0, orb.copy())
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
