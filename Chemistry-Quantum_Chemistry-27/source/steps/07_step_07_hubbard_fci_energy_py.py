"""
Compute the exact (full configuration interaction) ground-state energy of a Hubbard model in a fixed particle-number and spin sector.

The full configuration interaction energy is the lowest eigenvalue of the electronic Hamiltonian in the complete space of Slater determinants with $N_{\mathrm{up}}$ up and $N_{\mathrm{dn}}$ down electrons. For a Hubbard model on $M$ sites the space has dimension $\binom{M}{N_{\mathrm{up}}}\binom{M}{N_{\mathrm{dn}}}$, and the result does not depend on the orbital basis, which makes it the reference against which orbital-optimized, seniority-restricted approximations are measured.

Returns
-------
float: exact ground-state energy at fixed (N_up, N_down)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hubbard_fci_energy(hopping: "np.ndarray", onsite_U: float, n_up: int, n_down: int) -> float:
    '''Return the lowest eigenvalue of a Hubbard Hamiltonian at fixed (N_up, N_down).
 
    H = sum_{ij,s} hopping[i,j] c+_{is} c_{js} + U sum_i n_{i up} n_{i down}, diagonalized
    in the space of all determinants with n_up up and n_down down electrons on
    M = hopping.shape[0] sites. The result must be accurate to 1e-10 (any exact or
    iterative eigensolver may be used).
 
    Parameters
    ----------
    hopping : np.ndarray
        Real symmetric M x M one-electron matrix.
    onsite_U : float
        On-site interaction U.
    n_up : int
        Number of up electrons, 0 <= n_up <= M.
    n_down : int
        Number of down electrons, 0 <= n_down <= M.
 
    Returns
    -------
    energy : float
        Ground-state energy in the (n_up, n_down) sector.
 
    Raises
    ------
    ValueError
        If hopping is not square and symmetric or an electron count is outside [0, M].
    '''
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np
from scipy.sparse import coo_matrix, diags, identity, kron
from scipy.sparse.linalg import eigsh
 
def _spin_strings(m: int, n: int) -> tuple:
    strings = [sum(1 << i for i in occ) for occ in itertools.combinations(range(m), n)]
    return strings, {s: i for i, s in enumerate(strings)}
 
def _one_body_spin(h: "np.ndarray", strings: list, index: dict) -> "np.ndarray":
    m = h.shape[0]
    rows, cols, vals = [], [], []
    for j, s in enumerate(strings):
        for q in range(m):
            if not s >> q & 1:
                continue
            removed = s ^ (1 << q)
            sign_q = (-1) ** bin(s & ((1 << q) - 1)).count("1")
            for p in range(m):
                if h[p, q] == 0.0 or (removed >> p & 1):
                    continue
                sign_p = (-1) ** bin(removed & ((1 << p) - 1)).count("1")
                rows.append(index[removed | (1 << p)])
                cols.append(j)
                vals.append(sign_q * sign_p * h[p, q])
    return coo_matrix((vals, (rows, cols)), shape=(len(strings), len(strings))).toarray()
 
def _oracle_hubbard_fci_energy(hopping: "np.ndarray", onsite_U: float, n_up: int, n_down: int) -> float:
    h = np.asarray(hopping, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.allclose(h, h.T, atol=1e-12):
        raise ValueError("hopping must be a square symmetric matrix")
    m = h.shape[0]
    if not (0 <= n_up <= m and 0 <= n_down <= m):
        raise ValueError("electron counts must lie in [0, M]")
    up, up_index = _spin_strings(m, int(n_up))
    dn, dn_index = _spin_strings(m, int(n_down))
    h_up = _one_body_spin(h, up, up_index)
    h_dn = _one_body_spin(h, dn, dn_index)
    occ_up = np.array([[s >> i & 1 for i in range(m)] for s in up], dtype=float)
    occ_dn = np.array([[s >> i & 1 for i in range(m)] for s in dn], dtype=float)
    double = onsite_U * (occ_up @ occ_dn.T).ravel()
    dim = len(up) * len(dn)
    ham = kron(coo_matrix(h_up), identity(len(dn))) + kron(identity(len(up)), coo_matrix(h_dn)) + diags(double)
    if dim <= 400:
        return float(np.linalg.eigvalsh(ham.toarray())[0])
    return float(eigsh(ham.tocsr(), k=1, which="SA", tol=1e-13, maxiter=100000)[0][0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    ring = """import numpy as np
def ring(m, t):
    h = np.zeros((m, m))
    for i in range(m):
        h[i, (i + 1) % m] = h[(i + 1) % m, i] = -t
    return h
"""
    return [
        # Normal: benchmark, 3 up and 3 down electrons on the 8-site ring (3136 determinants)
        {"setup": ring + "h = ring(8, 1.0)\n", "call": "hubbard_fci_energy(h.copy(), 4.0, 3, 3)",
         "gold_call": "_oracle_hubbard_fci_energy(h.copy(), 4.0, 3, 3)"},
        # Normal: 6-site ring with two electrons per spin (antiperiodic sign on the wrap-around hop)
        {"setup": ring + "h = ring(6, 1.0)\n", "call": "hubbard_fci_energy(h.copy(), 4.0, 2, 2)",
         "gold_call": "_oracle_hubbard_fci_energy(h.copy(), 4.0, 2, 2)"},
        # Boundary: dense random symmetric one-body matrix (long-range hops) with unequal spin counts
        {"setup": ring + "rng = np.random.default_rng(3)\na = rng.normal(size=(5, 5))\nh = 0.5 * (a + a.T)\n",
         "call": "hubbard_fci_energy(h.copy(), 6.0, 3, 2)",
         "gold_call": "_oracle_hubbard_fci_energy(h.copy(), 6.0, 3, 2)"},
        # Edge: two-site dimer at half filling (closed form -(sqrt(U^2 + 16 t^2) - U)/2)
        {"setup": ring + "h = -np.array([[0.0, 1.0], [1.0, 0.0]])\n", "call": "hubbard_fci_energy(h.copy(), 3.0, 1, 1)",
         "gold_call": "_oracle_hubbard_fci_energy(h.copy(), 3.0, 1, 1)"},
        # Invalid: more up electrons than sites
        {"setup": """import numpy as np
h = -np.eye(3, k=1) - np.eye(3, k=-1)
def run_model():
    try:
        hubbard_fci_energy(h.copy(), 1.0, 4, 1)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_hubbard_fci_energy(h.copy(), 1.0, 4, 1)
        return 0
    except ValueError:
        return 1
""", "call": "run_model()", "gold_call": "run_oracle()"},
    ]
