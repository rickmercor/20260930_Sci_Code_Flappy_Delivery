"""
It assembles the concrete system, converts all lengths to Bohr, performs the near/far lattice split, compresses the far field with K-means and runs the MM induced-moment SCF to the converged coupled electrostatic-plus-induction energy, it returns that single scalar, in Hartree.

The pipeline brings together the three ingredients of the periodic polarizable-embedding scheme -- isotropic damping of the QM--MM boundary, orientation-independent Gaussian tensor damping within the classical sublattice, and a K-means-compressed expansion of the distant periodic images -- into one self-consistent induced-moment solve, with the QM region reduced to a single fixed point multipole.  The output is the converged total electrostatic-plus-induction energy of the coupled, 2-D-periodic system.

Returns
-------
return energy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
ANGSTROM_TO_BOHR = 1.8897261246
_TOTAL = 121
 
def coupled_qmmm_energy(beta_inv_angstrom: float = 0.291, g_angstrom: float = 0.32,
                        n_clusters: int = 3, mixing: float = 0.35, scf_tol: float = 1e-8) -> float:
    '''Full pipeline: converged coupled QM/MM electrostatic-plus-induction energy.
 
    Parameters
    ----------
    beta_inv_angstrom : float
        Isotropic QM--MM damping parameter beta, in A^-1 (default 0.291).
    g_angstrom : float
        Gaussian MM--MM tensor damping width, in A (default 0.32).
    n_clusters : int
        Number of K-means far-field expansion centres (default 3).
    mixing : float
        SCF linear mixing parameter t (default 0.35).
    scf_tol : float
        SCF convergence threshold on the largest induced-component change (default 1e-8).
 
    Returns
    -------
    energy : float
        The single requested scalar: the fully converged total
        electrostatic-plus-induction energy of the coupled QM/MM system, in Hartree.
        
    Raises
    ------
    ValueError
        If ``beta_inv_angstrom`` or ``g_angstrom`` is not finite or is not strictly
        positive, or if ``n_clusters`` is not a positive integer.
    '''
    return energy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
ANGSTROM_TO_BOHR = 1.8897261246
_TOTAL = 121
 
def _reference_system():
    """The prompt's geometry / moments / parameters in Angstrom and atomic units."""
    return dict(
        qm_pos=np.array([2.10, 2.10, 1.50]),
        qm_mu=np.array([0.10, 0.05, -0.60]),
        qm_theta=np.array([0.20, 0.15, -0.05, 0.10, -0.08]),   # xx, yy, xy, xz, yz
        mm_pos=np.array([[0.55, 0.40, 0.00],
                         [2.95, 1.35, 0.30],
                         [1.65, 3.10, -0.25]]),
        mm_mu=np.array([[0.42, -0.28, 0.09],
                        [-0.18, 0.52, -0.06],
                        [0.06, -0.38, 0.55]]),
        mm_theta=np.array([[0.75, -0.45, 0.22, -0.12, 0.05],
                           [-0.38, 0.85, -0.09, 0.18, -0.22],
                           [0.28, 0.12, 0.06, -0.28, 0.14]]),
        a1=np.array([4.20, 0.0, 0.0]),
        a2=np.array([0.0, 4.20, 0.0]),
        alpha=9.85, C=24.5,
    )
 
 
def _q2m(theta) -> np.ndarray:
    """Coerce a quadrupole given as (3,3) or [xx, yy, xy, xz, yz] to a traceless (3,3) matrix."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape == (3, 3):
        return 0.5 * (theta + theta.T)
    xx, yy, xy, xz, yz = theta
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, -(xx + yy)]], dtype=float)
 
 
def _far_field_sites(base_pos, base_mu, base_th, a1, a2, near_cells, far_window):
    """Pool the far-field periodic images of the four base sites (QM + 3 MM)."""
    near = {(int(a), int(b)) for a, b in near_cells}
    xy, z, mu, th = [], [], [], []
    for nx in far_window:
        for ny in far_window:
            if (nx, ny) in near:
                continue
            shift = nx * a1 + ny * a2
            for b in range(base_pos.shape[0]):
                p = base_pos[b] + shift
                xy.append(p[:2]); z.append(p[2]); mu.append(base_mu[b]); th.append(base_th[b])
    return np.array(xy), np.array(z), np.array(mu), np.array(th)
 
def _oracle_coupled_qmmm_energy(beta_inv_angstrom: float = 0.291, g_angstrom: float = 0.32,
                                n_clusters: int = 3, mixing: float = 0.35, scf_tol: float = 1e-8) -> float:
    """Reference orchestrator: near/far split, then steps 5 -> 4 -> 1/2/3 -> 6."""
    if not np.isfinite(beta_inv_angstrom) or float(beta_inv_angstrom) <= 0.0:
        raise ValueError("beta_inv_angstrom must be finite and positive")
    if not np.isfinite(g_angstrom) or float(g_angstrom) <= 0.0:
        raise ValueError("g_angstrom must be finite and positive")
    if not isinstance(n_clusters, (int, np.integer)) or n_clusters < 1:
        raise ValueError("n_clusters must be a positive integer")
 
    B = ANGSTROM_TO_BOHR
    s = _reference_system()
    qm_pos = s["qm_pos"] * B
    mm_pos = s["mm_pos"] * B
    a1, a2 = s["a1"] * B, s["a2"] * B
    qm_mu = s["qm_mu"]
    qm_th = _q2m(s["qm_theta"])
    mm_mu = s["mm_mu"]
    mm_th = np.stack([_q2m(r) for r in s["mm_theta"]])
    alpha, C = s["alpha"], s["C"]
    beta = float(beta_inv_angstrom) / B
    g = float(g_angstrom) * B
    K = int(n_clusters)
 
    near_cells = [(nx, ny) for nx in (-1, 0, 1) for ny in (-1, 0, 1)]
    far_window = (-2, -1, 0, 1, 2)
 
    base_pos = np.vstack([qm_pos, mm_pos])
    base_mu = np.vstack([qm_mu, mm_mu])
    base_th = np.concatenate([qm_th[None, :, :], mm_th], axis=0)
 
    # ---- far field: step 5 (K-means clustering) then step 4 (origin shift) ----
    xy, z, fmu, fth = _far_field_sites(base_pos, base_mu, base_th, a1, a2, near_cells, far_window)
    packed = _oracle_kmeans_expansion_centers(xy, z, n_clusters=K, random_state=0, n_init=10, tol=1e-6)
    cluster_pos = packed[:3 * K].reshape(K, 3)
    labels = packed[3 * K:].astype(int)
 
    cluster_mu = np.zeros((K, 3))
    cluster_th = np.zeros((K, 3, 3))
    for i in range(xy.shape[0]):
        k = labels[i]
        site = np.array([xy[i, 0], xy[i, 1], z[i]])
        shifted = np.asarray(_oracle_translate_multipole(fmu[i], fth[i], cluster_pos[k] - site, q=0.0),
                             dtype=float).reshape(12)
        cluster_mu[k] += shifted[:3]
        cluster_th[k] += shifted[3:].reshape(3, 3)
 
    # ---- near field: periodic instances of the QM site and the 3 MM sites ----
    def _inst(bp):
        return [(bp + nx * a1 + ny * a2, nx, ny) for (nx, ny) in near_cells]
    qm_inst = _inst(qm_pos)
    mm_inst = [(p, nx, ny, j) for j in range(3) for (p, nx, ny) in _inst(mm_pos[j])]
 
    # ---- assemble the inputs for step 6 (steps 1, 2, 3) ----
    home = [qm_pos, mm_pos[0], mm_pos[1], mm_pos[2]]
    perm_mu = np.vstack([qm_mu, mm_mu])
    perm_th = np.concatenate([qm_th[None, :, :], mm_th], axis=0)
    v_const = np.zeros((4, 3))
    g_const = np.zeros((4, 3, 3))
    mm_terms = [[], [], [], []]
 
    for tgt in range(4):
        is_qm = (tgt == 0)
        tp = home[tgt]
 
        if not is_qm:
            for (p, nx, ny) in qm_inst:
                rvec = tp - p
                d = float(np.sqrt(rvec @ rvec))
                S = float(_oracle_isotropic_qmmm_damping(np.array([d]), beta)[0])
                res = _oracle_multipole_potential_field(rvec, 0.0, qm_mu, qm_th, tensors=None, iso=S)
                v_const[tgt] += res[1:4]
                g_const[tgt] += res[4:13].reshape(3, 3)
 
        for k in range(K):
            rvec = tp - cluster_pos[k]
            res = _oracle_multipole_potential_field(rvec, 0.0, cluster_mu[k], cluster_th[k], tensors=None, iso=1.0)
            v_const[tgt] += res[1:4]
            g_const[tgt] += res[4:13].reshape(3, 3)
 
        for (p, nx, ny, j) in mm_inst:
            if (not is_qm) and j == tgt - 1 and nx == 0 and ny == 0:
                continue
            rvec = tp - p
            if is_qm:
                d = float(np.sqrt(rvec @ rvec))
                iso = float(_oracle_isotropic_qmmm_damping(np.array([d]), beta)[0])
                flat = _oracle_gaussian_damped_tensors(rvec, None)          # bare QM--MM tensor
            else:
                iso = 1.0
                flat = _oracle_gaussian_damped_tensors(rvec, g)             # Gaussian-damped MM--MM tensor
            flat = np.asarray(flat, dtype=float)
            T2 = flat[4:13].reshape(3, 3)
            T3 = flat[13:40].reshape(3, 3, 3)
            T4 = flat[40:121].reshape(3, 3, 3, 3)
            mm_terms[tgt].append((j, T2, T3, T4, iso))
 
    # ---- step 6: MM SCF + converged coupled energy ----
    energy = _oracle_mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms,
                           alpha, C, mix=float(mixing), tol=float(scf_tol))
    return float(energy)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test-case specifications."""
    return [
        # --- Normal scenario: the prompt's benchmark configuration (final answer) ---
        {
            "setup": "pass",
            "call": "coupled_qmmm_energy()",
            "gold_call": "_oracle_coupled_qmmm_energy()",
            "tol": 1e-7,
        },
        # --- Stronger isotropic damping (beta swept up) ---
        {
            "setup": "b = 0.50",
            "call": "coupled_qmmm_energy(beta_inv_angstrom=b)",
            "gold_call": "_oracle_coupled_qmmm_energy(beta_inv_angstrom=b)",
            "tol": 1e-7,
        },
        # --- Boundary: a single far-field expansion centre (K = 1) ---
        {
            "setup": "pass",
            "call": "coupled_qmmm_energy(n_clusters=1)",
            "gold_call": "_oracle_coupled_qmmm_energy(n_clusters=1)",
            "tol": 1e-7,
        },
        # --- Fewer clusters + a looser SCF tolerance ---
        {
            "setup": "pass",
            "call": "coupled_qmmm_energy(n_clusters=2, scf_tol=1e-7)",
            "gold_call": "_oracle_coupled_qmmm_energy(n_clusters=2, scf_tol=1e-7)",
            "tol": 1e-7,
        },
        # --- Edge: wider tensor damping and heavier mixing, all knobs stated. K = 5 is
        # chosen because the reference's fixed 64-restart k-means reaches the same global
        # optimum that scikit-learn's KMeans(n_init=10) finds at this K (it does not at
        # K = 4 or K = 6, where the two land on different local optima). ---
        {
            "setup": "pass",
            "call": "coupled_qmmm_energy(0.35, 0.60, 5, 0.5, 1e-9)",
            "gold_call": "_oracle_coupled_qmmm_energy(0.35, 0.60, 5, 0.5, 1e-9)",
            "tol": 1e-7,
        },
        # --- Invalid: non-positive isotropic damping parameter ---
        {
            "setup": """def run_model():
    try:
        coupled_qmmm_energy(beta_inv_angstrom=0.0); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_coupled_qmmm_energy(beta_inv_angstrom=0.0); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive number of expansion centres ---
        {
            "setup": """def run_model():
    try:
        coupled_qmmm_energy(n_clusters=0); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_coupled_qmmm_energy(n_clusters=0); return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
