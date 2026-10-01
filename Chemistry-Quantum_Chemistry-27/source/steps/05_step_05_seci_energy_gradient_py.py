"""
Evaluate the seniority eigenstate configuration interaction energy of a Hubbard model for given orbital-rotation parameters, together with its exact gradient with respect to those parameters.

For fixed orbitals the SECI energy is the lowest eigenvalue of the local-seniority-conserving Hamiltonian in the chosen sector, $E(\theta)=\lambda_{\min}\left[H_{\delta\Omega=0}(\theta)\right]$, where $\theta=(x,y)$ parametrize $C^{\mathrm{up}}=e^{X}$ and $C^{\mathrm{dn}}=e^{X}e^{Y}$. The orbitals and the configuration-interaction coefficients are optimized together, so the energy must be differentiated with respect to the orbital parameters. When the ground state is nondegenerate, the Hellmann–Feynman theorem gives $\partial E/\partial\theta_k=\langle\Psi|\partial H_{\delta\Omega=0}/\partial\theta_k|\Psi\rangle$; the dependence on $\theta$ enters through the transformed integrals and through the derivative of the matrix exponential.

Returns
-------
np.ndarray of shape (1 + M(M−1)/2 + k(k−1)/2,): [E, dE/dx, dE/dy]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def seci_energy_gradient(x_params: "np.ndarray", y_params: "np.ndarray", hopping: "np.ndarray", onsite_U: float, n_pairs: int, n_spin_levels: int) -> "np.ndarray":
    '''Return the SECI energy and its gradient with respect to the orbital parameters.
 
    Let M = hopping.shape[0] and k = n_spin_levels (k even). The orbitals are
    C_up = expm(X), C_dn = expm(X) expm(Y), with X built from x_params (order of
    numpy.triu_indices(M, 1), X = A - A^T) and Y zero except its last k x k block built
    from y_params (order of numpy.triu_indices(k, 1)). Levels 0..M-k-1 are pairing
    levels holding n_pairs pairs; levels M-k..M-1 are spin levels with k/2 up and k/2
    down electrons (S^z = 0). The energy E(x, y) is the lowest eigenvalue of the
    local-seniority-conserving part of the Hubbard Hamiltonian
        H = sum_{ij,s} hopping[i,j] c+_{is} c_{js} + U sum_i n_{i up} n_{i down}
    restricted to this sector (equivalently, of the full H projected onto the sector).
    The gradient is the exact derivative of E with respect to every entry of x_params
    and y_params.
 
    Parameters
    ----------
    x_params : np.ndarray
        M(M-1)/2 rotation parameters for X.
    y_params : np.ndarray
        k(k-1)/2 rotation parameters for Y.
    hopping : np.ndarray
        Real symmetric M x M site-basis one-electron matrix.
    onsite_U : float
        On-site repulsion U.
    n_pairs : int
        Number of electron pairs on the pairing levels (0 <= n_pairs <= M - k).
    n_spin_levels : int
        Even number k of spin levels, 0 <= k <= M.
 
    Returns
    -------
    result : np.ndarray
        Array of length 1 + M(M-1)/2 + k(k-1)/2: [E, dE/dx_1, ..., dE/dy_last]. The
        energy must be accurate to 1e-10 and each gradient component to 1e-7.
 
    Raises
    ------
    ValueError
        If k is odd or outside [0, M], n_pairs is outside [0, M - k], a parameter array
        has the wrong length, or the lowest eigenvalue is degenerate (gap below 1e-9),
        so that the gradient is undefined.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm, expm_frechet
 
def _sector_moves(states: "np.ndarray") -> tuple:
    index = {tuple(row): i for i, row in enumerate(states)}
    pair_moves, spin_moves = [], []
    m = states.shape[1]
    for j, row in enumerate(states):
        for p in range(m):
            for q in range(m):
                if p == q:
                    continue
                if row[p] == 0 and row[q] == 2:
                    target = row.copy()
                    target[p], target[q] = 2, 0
                    pair_moves.append((index[tuple(target)], j, p, q))
                if row[p] == -1 and row[q] == 1:
                    target = row.copy()
                    target[p], target[q] = 1, -1
                    spin_moves.append((index[tuple(target)], j, p, q))
    return np.array(pair_moves, dtype=int).reshape(-1, 4), np.array(spin_moves, dtype=int).reshape(-1, 4)
 
def _energy_gradient_core(theta: "np.ndarray", hopping: "np.ndarray", onsite_U: float, states: "np.ndarray", moves: tuple, n_spin_levels: int) -> "np.ndarray":
    m, k = hopping.shape[0], int(n_spin_levels)
    nx = m * (m - 1) // 2
    orb = _oracle_unrestricted_orbitals(theta[:nx], theta[nx:], m, k)
    c_up, c_dn = orb[0], orb[1]
    gen_x = _antisymmetric(theta[:nx], m)
    gen_y = np.zeros((m, m))
    if k > 1:
        gen_y[m - k:, m - k:] = _antisymmetric(theta[nx:], k)
    coef = _oracle_seniority_coefficients(hopping, onsite_U, orb)
    ham = _oracle_seci_hamiltonian(coef, states)
    values, vectors = np.linalg.eigh(ham)
    if len(values) > 1 and values[1] - values[0] < 1e-9:
        raise ValueError("degenerate SECI ground state; gradient undefined")
    psi = vectors[:, 0]
    prob = psi ** 2
    charge = np.where(states == 2, 2.0, np.where(states == 0, 0.0, 1.0))
    spin = np.where(states == 1, 0.5, np.where(states == -1, -0.5, 0.0))
    off = 1.0 - np.eye(m)
    # expectation values multiplying each coefficient (Hellmann-Feynman weights)
    w_eps = prob @ charge
    w_b = prob @ spin
    w_pair = np.diag(prob @ (states == 2))
    pair_moves, spin_moves = moves
    if len(pair_moves):
        np.add.at(w_pair, (pair_moves[:, 2], pair_moves[:, 3]), psi[pair_moves[:, 0]] * psi[pair_moves[:, 1]])
    w_w = 0.25 * np.einsum("i,ip,iq->pq", prob, charge, charge) * off
    w_bz = np.einsum("i,ip,iq->pq", prob, spin, spin) * off
    w_x = np.einsum("i,ip,iq->pq", prob, charge, spin) * off
    w_k = np.zeros((m, m))
    if len(spin_moves):
        np.add.at(w_k, (spin_moves[:, 2], spin_moves[:, 3]), -psi[spin_moves[:, 0]] * psi[spin_moves[:, 1]])
    w_k *= off
    # same-spin Coulomb and exchange cancel for the Hubbard interaction, leaving
    # E = sum (w_eps/2 + w_b) h_up_pp + (w_eps/2 - w_b) h_dn_pp + U [sum Om_pq (uu^T dd)_pq + sum Lam_pq (P^T P)_pq]
    omega = (w_w - w_bz - 0.5 * (w_x - w_x.T)) * off
    lam = w_pair + w_k
    sq_up, sq_dn, mixed = c_up * c_up, c_dn * c_dn, c_up * c_dn
    grad_up = 2.0 * hopping @ c_up * (0.5 * w_eps + w_b)[None, :] + onsite_U * (2.0 * c_up * (sq_dn @ omega.T) + c_dn * (mixed @ (lam + lam.T)))
    grad_dn = 2.0 * hopping @ c_dn * (0.5 * w_eps - w_b)[None, :] + onsite_U * (2.0 * c_dn * (sq_up @ omega) + c_up * (mixed @ (lam + lam.T)))
    exp_y = expm(gen_y)
    gx = expm_frechet(gen_x.T, grad_up + grad_dn @ exp_y.T, compute_expm=False)
    grad_x = (gx - gx.T)[np.triu_indices(m, 1)]
    if k > 1:
        gy = expm_frechet(gen_y.T, expm(gen_x).T @ grad_dn, compute_expm=False)[m - k:, m - k:]
        grad_y = (gy - gy.T)[np.triu_indices(k, 1)]
    else:
        grad_y = np.zeros(0)
    return np.concatenate([[values[0]], grad_x, grad_y])
 
def _sector_setup(hopping: "np.ndarray", n_pairs: int, n_spin_levels: int) -> tuple:
    h = np.asarray(hopping, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("hopping must be square")
    m, k, npair = h.shape[0], int(n_spin_levels), int(n_pairs)
    if k % 2 or not 0 <= k <= m or not 0 <= npair <= m - k:
        raise ValueError("invalid seniority sector")
    states = _oracle_seniority_sector_basis(m - k, npair, k, k // 2)
    return h, states, _sector_moves(states)
 
def _oracle_seci_energy_gradient(x_params: "np.ndarray", y_params: "np.ndarray", hopping: "np.ndarray", onsite_U: float, n_pairs: int, n_spin_levels: int) -> "np.ndarray":
    h, states, moves = _sector_setup(hopping, n_pairs, n_spin_levels)
    m, k = h.shape[0], int(n_spin_levels)
    x = np.asarray(x_params, dtype=float).ravel()
    y = np.asarray(y_params, dtype=float).ravel()
    if x.size != m * (m - 1) // 2 or y.size != k * (k - 1) // 2:
        raise ValueError("rotation parameter arrays have the wrong length")
    return _energy_gradient_core(np.concatenate([x, y]), h, float(onsite_U), states, moves, k)

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
        # Normal: benchmark sector (8-site ring, 1 pair, 4 spin levels), random orbitals
        {"setup": ring + "rng = np.random.default_rng(0)\nx = rng.normal(size=28)\ny = rng.normal(size=6)\nh = ring(8, 1.0)\n",
         "call": "seci_energy_gradient(x.copy(), y.copy(), h.copy(), 4.0, 1, 4)",
         "gold_call": "_oracle_seci_energy_gradient(x.copy(), y.copy(), h.copy(), 4.0, 1, 4)",
         "tol": 1e-6},
        # Normal: maximal seniority on the half-filled 6-site ring
        {"setup": ring + "rng = np.random.default_rng(1)\nx = rng.normal(size=15)\ny = rng.normal(size=15)\nh = ring(6, 1.0)\n",
         "call": "seci_energy_gradient(x.copy(), y.copy(), h.copy(), 4.0, 0, 6)",
         "gold_call": "_oracle_seci_energy_gradient(x.copy(), y.copy(), h.copy(), 4.0, 0, 6)",
         "tol": 1e-6},
        # Boundary: seniority zero (restricted DOCI; no spin levels, empty y)
        {"setup": ring + "rng = np.random.default_rng(2)\nx = 0.5 * rng.normal(size=15)\ny = np.zeros(0)\nh = ring(6, 1.0)\n",
         "call": "seci_energy_gradient(x.copy(), y.copy(), h.copy(), 4.0, 3, 0)",
         "gold_call": "_oracle_seci_energy_gradient(x.copy(), y.copy(), h.copy(), 4.0, 3, 0)",
         "tol": 1e-6},
        # Edge: smallest mixed sector (2 pairing + 2 spin levels) on an open 4-site chain
        {"setup": """import numpy as np
h = -np.diag(np.ones(3), 1) - np.diag(np.ones(3), -1)
x = np.array([0.3, -0.2, 0.1, 0.25, -0.15, 0.05])
y = np.array([0.3])
""",
         "call": "seci_energy_gradient(x.copy(), y.copy(), h.copy(), 2.0, 1, 2)",
         "gold_call": "_oracle_seci_energy_gradient(x.copy(), y.copy(), h.copy(), 2.0, 1, 2)",
         "tol": 1e-6},
        # Held-out generalization: dense five-level Hamiltonian, mixed seniority sector
        {"setup": """import numpy as np
rng = np.random.default_rng(55)
a = rng.normal(scale=0.6, size=(5, 5))
h = 0.5 * (a + a.T)
x = rng.normal(scale=0.35, size=10)
y = rng.normal(scale=0.25, size=1)
""",
         "call": "seci_energy_gradient(x.copy(), y.copy(), h.copy(), 3.2, 1, 2)",
         "gold_call": "_oracle_seci_energy_gradient(x.copy(), y.copy(), h.copy(), 3.2, 1, 2)",
         "tol": 1e-6},
        # Held-out generalization: seven levels, four spin levels and dense hopping
        {"setup": """import numpy as np
rng = np.random.default_rng(77)
a = rng.normal(scale=0.4, size=(7, 7))
h = 0.5 * (a + a.T)
x = rng.normal(scale=0.2, size=21)
y = rng.normal(scale=0.2, size=6)
""",
         "call": "seci_energy_gradient(x.copy(), y.copy(), h.copy(), 2.3, 1, 4)",
         "gold_call": "_oracle_seci_energy_gradient(x.copy(), y.copy(), h.copy(), 2.3, 1, 4)",
         "tol": 1e-6},
        # Invalid: odd number of spin levels (no S^z = 0 sector)
        {"setup": """import numpy as np
h = -np.eye(4, k=1) - np.eye(4, k=-1)
def run_model():
    try:
        seci_energy_gradient(np.zeros(6), np.zeros(3), h.copy(), 1.0, 0, 3)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_seci_energy_gradient(np.zeros(6), np.zeros(3), h.copy(), 1.0, 0, 3)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
    ]
