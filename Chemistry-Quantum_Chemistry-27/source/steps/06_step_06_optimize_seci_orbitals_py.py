"""
Find the variationally optimal seniority eigenstate configuration interaction energy of a Hubbard model by minimizing over all orbital rotations.

As in doubly occupied configuration interaction, the SECI orbitals are chosen to minimize the energy, and this minimization also decides which orbitals act as pairing levels and which as spin levels. The energy surface over the rotation parameters $(x,y)$ is not convex. It has several local minima of different energy, and it has stationary points with spin-restricted spin levels at which the maximal-seniority energy of a half-filled Hubbard model is exactly zero for any $U$. The SECI energy is the global minimum of this surface.

Returns
-------
float: global-minimum SECI energy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def optimize_seci_orbitals(hopping: "np.ndarray", onsite_U: float, n_pairs: int, n_spin_levels: int, seed: int = 20260927) -> float:
    '''Return the orbital-optimized SECI energy (global minimum over orbital rotations).
 
    With M = hopping.shape[0], k = n_spin_levels (even), pairing levels 0..M-k-1 holding
    n_pairs pairs and spin levels M-k..M-1 holding k/2 up and k/2 down electrons, let
    E(x, y) be the lowest eigenvalue of the local-seniority-conserving part of the
    Hubbard Hamiltonian
        H = sum_{ij,s} hopping[i,j] c+_{is} c_{js} + U sum_i n_{i up} n_{i down}
    in this sector, for orbitals C_up = expm(X) and C_dn = expm(X) expm(Y), where X is a
    real antisymmetric M x M matrix and Y a real antisymmetric matrix that is zero
    outside its last k x k block. Return
        E_SECI = min over all real antisymmetric X and block-restricted Y of E(x, y),
    the global minimum, accurate to 1e-8. The minimizing orbitals are not unique and are
    not returned. At the reported minimum the gradient norm is below 1e-6. The energy
    and gradient at any (x, y) are those of seci_energy_gradient. Random starting points,
    if used, are drawn from numpy.random.default_rng(seed); the returned global minimum
    must not depend on the seed.
 
    Parameters
    ----------
    hopping : np.ndarray
        Real symmetric M x M site-basis one-electron matrix.
    onsite_U : float
        On-site repulsion U.
    n_pairs : int
        Number of electron pairs on the pairing levels (0 <= n_pairs <= M - k).
    n_spin_levels : int
        Even number k of spin levels, 0 <= k <= M.
    seed : int, optional
        Seed for the random starting points of the global search (default 20260927).
 
    Returns
    -------
    energy : float
        Global minimum of the SECI energy.
 
    Raises
    ------
    ValueError
        If k is odd or outside [0, M], or n_pairs is outside [0, M - k].
    '''
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize
 
def _oracle_optimize_seci_orbitals(hopping: "np.ndarray", onsite_U: float, n_pairs: int, n_spin_levels: int, seed: int = 20260927) -> float:
    h = np.asarray(hopping, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("hopping must be square")
    m, k = h.shape[0], int(n_spin_levels)
    if k % 2 or not 0 <= k <= m or not 0 <= int(n_pairs) <= m - k:
        raise ValueError("invalid seniority sector")
    n_x, n_y = m * (m - 1) // 2, k * (k - 1) // 2
 
    def _objective(theta):
        # energy and gradient from the Step-5 function; at a degenerate ground state the
        # gradient is undefined, so evaluate at a nearby point
        shift = np.cos(np.arange(theta.size) + 1.0)
        for scale in (0.0, 1e-7, 1e-6, 1e-5, 1e-4):
            point = theta + scale * shift
            try:
                out = _oracle_seci_energy_gradient(point[:n_x], point[n_x:], h, onsite_U, n_pairs, k)
                return out[0], out[1:]
            except ValueError:
                continue
        raise ValueError("degenerate SECI ground state throughout the neighbourhood")
 
    if n_x + n_y == 0:
        return float(_objective(np.zeros(0))[0])
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(12):
        run = minimize(_objective, rng.normal(size=n_x + n_y), jac=True, method="L-BFGS-B",
                       options={"maxiter": 3000, "ftol": 1e-13, "gtol": 1e-8})
        if best is None or run.fun < best.fun:
            best = run
    polish = minimize(_objective, best.x, jac=True, method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
    return float(min(polish.fun, best.fun))

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
        # Normal: benchmark, 6 electrons on the 8-site ring at seniority 4 (two competing minima)
        {"setup": ring + "h = ring(8, 1.0)\n",
         "call": "optimize_seci_orbitals(h.copy(), 4.0, 1, 4)",
         "gold_call": "_oracle_optimize_seci_orbitals(h.copy(), 4.0, 1, 4)",
         "tol": 1e-7},
        # Normal: intermediate seniority on the half-filled 6-site ring at strong coupling (competing minima)
        {"setup": ring + "h = ring(6, 1.0)\n",
         "call": "optimize_seci_orbitals(h.copy(), 8.0, 1, 4)",
         "gold_call": "_oracle_optimize_seci_orbitals(h.copy(), 8.0, 1, 4)",
         "tol": 1e-7},
        # Property: a different seed must reach the same global minimum (6-site ring, two pairs, competing minima)
        {"setup": ring + "h = ring(6, 1.0)\n",
         "call": "optimize_seci_orbitals(h.copy(), 4.0, 2, 2, seed=7)",
         "gold_call": "_oracle_optimize_seci_orbitals(h.copy(), 4.0, 2, 2)",
         "tol": 1e-7},
        # Normal: low seniority (two pairs, two spin levels) on the half-filled 6-site ring (competing minima)
        {"setup": ring + "h = ring(6, 1.0)\n",
         "call": "optimize_seci_orbitals(h.copy(), 4.0, 2, 2)",
         "gold_call": "_oracle_optimize_seci_orbitals(h.copy(), 4.0, 2, 2)",
         "tol": 1e-7},
        # Boundary: two-site Hubbard model at maximal seniority
        {"setup": ring + "h = -np.array([[0.0, 1.0], [1.0, 0.0]])\n",
         "call": "optimize_seci_orbitals(h.copy(), 3.0, 0, 2)",
         "gold_call": "_oracle_optimize_seci_orbitals(h.copy(), 3.0, 0, 2)",
         "tol": 1e-7},
        # Edge: seniority zero (orbital-optimized restricted DOCI) on the 6-site ring
        {"setup": ring + "h = ring(6, 1.0)\n",
         "call": "optimize_seci_orbitals(h.copy(), 4.0, 3, 0)",
         "gold_call": "_oracle_optimize_seci_orbitals(h.copy(), 4.0, 3, 0)",
         "tol": 1e-7},
        # Edge: one site holding one pair, no orbital parameters (energy 2*h + U)
        {"setup": "import numpy as np\nh = np.array([[-0.5]])\n",
         "call": "optimize_seci_orbitals(h.copy(), 3.0, 1, 0)",
         "gold_call": "_oracle_optimize_seci_orbitals(h.copy(), 3.0, 1, 0)",
         "tol": 1e-7},
        # Held-out generalization: dense three-level Hamiltonian (not a ring or chain)
        {"setup": """import numpy as np
rng = np.random.default_rng(3133)
a = rng.normal(scale=0.5, size=(3, 3))
h = 0.5 * (a + a.T)
""",
         "call": "optimize_seci_orbitals(h.copy(), 2.7, 1, 2)",
         "gold_call": "_oracle_optimize_seci_orbitals(h.copy(), 2.7, 1, 2)",
         "tol": 1e-7},
        # Held-out generalization: dense five-level Hamiltonian with an odd number of sites
        {"setup": """import numpy as np
rng = np.random.default_rng(3152)
a = rng.normal(scale=0.5, size=(5, 5))
h = 0.5 * (a + a.T)
""",
         "call": "optimize_seci_orbitals(h.copy(), 2.5, 1, 2)",
         "gold_call": "_oracle_optimize_seci_orbitals(h.copy(), 2.5, 1, 2)",
         "tol": 1e-7},
        # Property: a different seed reaches the same dense four-level global minimum
        {"setup": """import numpy as np
rng = np.random.default_rng(3142)
a = rng.normal(scale=0.5, size=(4, 4))
h = 0.5 * (a + a.T)
""",
         "call": "optimize_seci_orbitals(h.copy(), 3.2, 1, 2, seed=17)",
         "gold_call": "_oracle_optimize_seci_orbitals(h.copy(), 3.2, 1, 2, seed=20260927)",
         "tol": 1e-7},
        # Invalid: more pairs than pairing levels
        {"setup": ring + """h = ring(4, 1.0)
def run_model():
    try:
        optimize_seci_orbitals(h.copy(), 2.0, 3, 2)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_optimize_seci_orbitals(h.copy(), 2.0, 3, 2)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
    ]
