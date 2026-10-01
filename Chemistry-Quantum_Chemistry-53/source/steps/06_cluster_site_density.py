"""
The embedding cluster of site i is described by the Hamiltonian    H^(i) = P^(i) H P^(i) - mu_imp^(i) n_I,  where the first term is the projected cluster Hamiltonian of the previous steps (a 4 x 4 matrix in the S_z = 0 basis |I up, I down>, |I up, B down>, |B up, I down>, |B up, B down>) and n_I = n_{I up} + n_{I down} counts the electrons on the impurity orbital.

The embedding cluster of site i is described by the Hamiltonian

  H^(i) = P^(i) H P^(i) - mu_imp^(i) n_I,

where the first term is the projected cluster Hamiltonian of the previous steps (a 4 x 4 matrix in the S_z = 0 basis |I up, I down>, |I up, B down>, |B up, I down>, |B up, B down>) and n_I = n_{I up} + n_{I down} counts the electrons on the impurity orbital. In that basis n_I is the diagonal matrix diag(2, 1, 1, 0). The cluster is solved exactly: the 4 x 4 matrix is diagonalized and its lowest eigenvector psi gives the correlated impurity occupation

  n_i^cl = <psi| n_I |psi> = 2 psi_1^2 + psi_2^2 + psi_3^2,

which is the spin-summed density of site i predicted by the cluster. A more negative mu_imp raises the cost of putting electrons on the impurity and lowers n_i^cl, a more positive one raises it; this is the handle through which the embedding transfers the correlation potential to the correlated densities. The ground level of the 4 x 4 problem must be non-degenerate for the occupation to be defined; a degenerate ground level (gap below 1e-10) is treated as invalid input.

Returns
-------
float: the ground-state impurity occupation <n_I> (spin summed, in [0, 2]) of the cluster Hamiltonian h_cl - mu * diag(2, 1, 1, 0).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def cluster_site_density(h_cl, mu):
    '''Correlated impurity occupation of one embedding cluster.

    Parameters
    ----------
    h_cl : array_like of float, shape (4, 4)
        Real symmetric projected cluster Hamiltonian in the basis
        |I up, I down>, |I up, B down>, |B up, I down>, |B up, B down>.
    mu : float
        Impurity chemical potential; the cluster Hamiltonian solved is
        h_cl - mu * diag(2, 1, 1, 0).

    Returns
    -------
    n_cl : float
        Spin-summed occupation of the impurity orbital in the ground state
        of the cluster, between 0 and 2. Raises ValueError if h_cl is not a
        finite symmetric 4 x 4 matrix, if mu is not finite, or if the ground
        level of the cluster is degenerate (gap below 1e-10).
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cluster_site_density(h_cl, mu):
    h_cl = np.asarray(h_cl, dtype=float)
    if h_cl.shape != (4, 4):
        raise ValueError("h_cl must be a 4 x 4 matrix")
    if not np.all(np.isfinite(h_cl)) or not np.allclose(h_cl, h_cl.T, rtol=0.0, atol=1e-10):
        raise ValueError("h_cl must be finite and symmetric")
    mu = float(mu)
    if not np.isfinite(mu):
        raise ValueError("mu must be finite")
    n_op = np.diag([2.0, 1.0, 1.0, 0.0])
    w, v = np.linalg.eigh(h_cl - mu * n_op)
    if w[1] - w[0] < 1e-10:
        raise ValueError("degenerate cluster ground level: occupation undefined")
    psi = v[:, 0]
    return float(2.0 * psi[0] ** 2 + psi[1] ** 2 + psi[2] ** 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def _hcl(e_i, e_b, tau, u_i, u_b):
    return np.array([[2 * e_i + u_i, tau, tau, 0.0],
                     [tau, e_i + e_b, 0.0, tau],
                     [tau, 0.0, e_i + e_b, tau],
                     [0.0, tau, tau, 2 * e_b + u_b]])
h_a = _hcl(-2.0, 0.9, -0.45, 7.0, 1.6)
"""
    return [
        # --- Normal: a cluster resembling site 2 of the task ring with a negative mu ---
        {
            "setup": setup,
            "call": "cluster_site_density(h_a, -1.34)",
            "gold_call": "_oracle_cluster_site_density(h_a, -1.34)",
        },
        # --- Normal: same cluster without chemical potential ---
        {
            "setup": setup,
            "call": "cluster_site_density(h_a, 0.0)",
            "gold_call": "_oracle_cluster_site_density(h_a, 0.0)",
        },
        # --- Boundary: a very negative mu empties the impurity (occupation close to 0) ---
        {
            "setup": setup,
            "call": "cluster_site_density(h_a, -60.0)",
            "gold_call": "_oracle_cluster_site_density(h_a, -60.0)",
        },
        # --- Boundary: a very positive mu fills the impurity (occupation close to 2) ---
        {
            "setup": setup,
            "call": "cluster_site_density(h_a, 60.0)",
            "gold_call": "_oracle_cluster_site_density(h_a, 60.0)",
        },
        # --- Edge: decoupled orbitals (tau = 0) with a unique ground state ---
        {
            "setup": setup + "h_d = _hcl(-1.5, 0.5, 0.0, 1.0, 0.7)\n",
            "call": "cluster_site_density(h_d, 0.2)",
            "gold_call": "_oracle_cluster_site_density(h_d, 0.2)",
        },
        # --- Invalid: degenerate ground level (singlet and triplet coincide at tau = 0) ---
        {
            "setup": setup + """h_deg = _hcl(0.0, 0.0, 0.0, 1.0, 1.0)
def run_model():
    try:
        cluster_site_density(h_deg, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_cluster_site_density(h_deg, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: wrong matrix shape ---
        {
            "setup": setup + """
def run_model():
    try:
        cluster_site_density(np.eye(3), 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_cluster_site_density(np.eye(3), 0.0)
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
