"""
In the interacting-bath formulation of density matrix embedding, the cluster Hamiltonian of the embedded site i is the exact projection of the full Hamiltonian H onto the space spanned by the determinants    |Phi_core Phi_alpha>,  where Phi_core is the product of the occupied reference (gKS) orbitals that have no overlap with the impurity (they are also orthogonal to the bath, because the impurity plus bath space is invariant under gamma), and Phi_alpha runs over all determinants that distribute two electrons among the impurity orbital I = chi_i and the bath orbital B = b^(i).

In the interacting-bath formulation of density matrix embedding, the cluster Hamiltonian of the embedded site i is the exact projection of the full Hamiltonian H onto the space spanned by the determinants

  |Phi_core Phi_alpha>,

where Phi_core is the product of the occupied reference (gKS) orbitals that have no overlap with the impurity (they are also orthogonal to the bath, because the impurity plus bath space is invariant under gamma), and Phi_alpha runs over all determinants that distribute two electrons among the impurity orbital I = chi_i and the bath orbital B = b^(i). The core is frozen: it is doubly occupied in every determinant of the space. The projected Hamiltonian therefore acts on a half-filled two-orbital cluster and, for a site-local repulsion U n_{j up} n_{j down}, it contains

  * the one-body matrix of h in the cluster basis (I, B), i.e. h_II = h_ii,

    h_IB = sum_j h_ij b_j and h_BB = sum_{jk} b_j h_jk b_k;

  * the mean field exerted by the frozen core on the cluster electrons, which

    for an on-site interaction is the local potential U gamma^core_jj (the

    core occupation per spin on site j) contracted with the cluster orbitals:

    it vanishes on I (the core has no amplitude on site i) and adds

    U sum_j b_j^2 gamma^core_jj to h_BB, with gamma^core = (1 - Q) gamma (1 - Q)

    and Q the projector on span{I, B};

  * the two-body terms that survive the projection: U n_{I up} n_{I down} on the

    impurity and U_B n_{B up} n_{B down} on the bath with U_B = U sum_j b_j^4

    (there are no mixed impurity-bath interaction terms because the bath has

    no weight on site i);

  * a constant core energy, which does not affect densities and is dropped.

In the S_z = 0 two-electron basis, ordered as

  |I up, I down>, |I up, B down>, |B up, I down>, |B up, B down>,

with each state written as c+_{a up} c+_{b down}|0>, the projected Hamiltonian is the real symmetric 4 x 4 matrix

  [[2 e_I + U,   tau,         tau,         0        ],

   [tau,         e_I + e_B,   0,           tau      ],

   [tau,         0,           e_I + e_B,   tau      ],

   [0,           tau,         tau,         2 e_B + U_B]],

with e_I = h_II, e_B = h_BB + U sum_j b_j^2 gamma^core_jj and tau = h_IB. The impurity chemical potential term of the embedding is not included here; it is added to this matrix in a later step.

Returns
-------
np.ndarray of float with shape (4, 4): the projected cluster Hamiltonian in the basis |I up, I down>, |I up, B down>, |B up, I down>, |B up, B down>.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def cluster_hamiltonian(h, U, gamma, site):
    '''Projected two-electron cluster Hamiltonian of the embedded site.

    Parameters
    ----------
    h : array_like of float, shape (L, L)
        Real symmetric one-electron matrix of the ring.
    U : float
        On-site repulsion.
    gamma : array_like of float, shape (L, L)
        Idempotent per-spin reference density matrix (the bath is built from
        its row `site`, and its projection on the orthogonal complement of the
        impurity plus bath space defines the frozen core).
    site : int
        Index of the embedded orbital, 0 <= site < L.

    Returns
    -------
    h_cl : np.ndarray of float, shape (4, 4)
        Matrix of the projected Hamiltonian (without chemical potential term
        and without the constant core energy) in the S_z = 0 basis
        |I up, I down>, |I up, B down>, |B up, I down>, |B up, B down>.
        Raises ValueError if gamma is not symmetric and idempotent (to 1e-6),
        if the cluster does not hold exactly one electron per spin, if site
        is out of range, or if no bath orbital exists for the site.
    '''
    return np.zeros((4, 4), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cluster_hamiltonian(h, U, gamma, site):
    h = np.asarray(h, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or gamma.shape != h.shape:
        raise ValueError("h and gamma must be square matrices of the same shape")
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be finite and symmetric")
    if not np.all(np.isfinite(gamma)) or not np.allclose(gamma, gamma.T, rtol=0.0, atol=1e-8):
        raise ValueError("gamma must be finite and symmetric")
    if np.max(np.abs(gamma @ gamma - gamma)) > 1e-6:
        raise ValueError("gamma must be idempotent")
    U = float(U)
    if not np.isfinite(U):
        raise ValueError("U must be finite")
    if isinstance(site, bool) or not isinstance(site, (int, np.integer)):
        raise ValueError("site must be an integer")
    L = h.shape[0]
    if site < 0 or site >= L:
        raise ValueError("site index out of range")
    i = int(site)
    b = gamma[i, :].copy()
    b[i] = 0.0
    norm = np.linalg.norm(b)
    if norm < 1e-12:
        raise ValueError("the embedded site is decoupled: no bath orbital exists")
    b = b / norm
    cmat = np.zeros((L, 2))
    cmat[i, 0] = 1.0
    cmat[:, 1] = b
    q = cmat @ cmat.T
    if abs(np.trace(q @ gamma) - 1.0) > 1e-6:
        raise ValueError("the cluster must hold exactly one electron per spin")
    p = np.eye(L) - q
    gamma_core = p @ gamma @ p
    h_eff = h + np.diag(U * np.diag(gamma_core))
    hc = cmat.T @ h_eff @ cmat
    e_i, e_b, tau = hc[0, 0], hc[1, 1], hc[0, 1]
    u_b = U * float(np.sum(b ** 4))
    return np.array([
        [2.0 * e_i + U, tau, tau, 0.0],
        [tau, e_i + e_b, 0.0, tau],
        [tau, 0.0, e_i + e_b, tau],
        [0.0, tau, tau, 2.0 * e_b + u_b],
    ])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def _ring(L, t, v):
    h = np.diag(np.asarray(v, dtype=float))
    for i in range(L):
        h[i, (i + 1) % L] -= t
        h[(i + 1) % L, i] -= t
    return h
def _gamma(h, n_occ):
    e, c = np.linalg.eigh(h)
    return c[:, :n_occ] @ c[:, :n_occ].T
h6 = _ring(6, 1.0, [-1.0, 2.0, -2.0, 3.0, -3.0, 1.0])
g6 = _gamma(h6 + np.diag([0.4, -0.5, 0.5, -0.4, 0.4, -0.5]), 3)
"""
    return [
        # --- Normal: site 2 of the task ring at U = 7 ---
        {
            "setup": setup,
            "call": "cluster_hamiltonian(h6, 7.0, g6, 2)",
            "gold_call": "_oracle_cluster_hamiltonian(h6, 7.0, g6, 2)",
        },
        # --- Normal: site 5 at U = 3 (periodic wrap-around bond) ---
        {
            "setup": setup,
            "call": "cluster_hamiltonian(h6, 3.0, g6, 5)",
            "gold_call": "_oracle_cluster_hamiltonian(h6, 3.0, g6, 5)",
        },
        # --- Boundary: U = 0 leaves only the projected one-body matrix ---
        {
            "setup": setup,
            "call": "cluster_hamiltonian(h6, 0.0, g6, 1)",
            "gold_call": "_oracle_cluster_hamiltonian(h6, 0.0, g6, 1)",
        },
        # --- Edge: five-site ring with two electrons (core is empty) ---
        {
            "setup": setup + "h5 = _ring(5, 0.8, [0.3, -0.4, 0.8, -0.2, 0.1])\ng5 = _gamma(h5, 1)\n",
            "call": "cluster_hamiltonian(h5, 2.5, g5, 3)",
            "gold_call": "_oracle_cluster_hamiltonian(h5, 2.5, g5, 3)",
        },
        # --- Invalid: a density matrix that is not idempotent ---
        {
            "setup": setup + """
def run_model():
    try:
        cluster_hamiltonian(h6, 7.0, 0.5 * g6, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_cluster_hamiltonian(h6, 7.0, 0.5 * g6, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: site index out of range ---
        {
            "setup": setup + """
def run_model():
    try:
        cluster_hamiltonian(h6, 7.0, g6, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_cluster_hamiltonian(h6, 7.0, g6, -1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
