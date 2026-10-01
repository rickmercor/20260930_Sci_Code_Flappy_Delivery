"""
The self-consistency condition of gLPFET is a set of local density constraints: for every site i, the occupation of site i in the gKS reference determinant must equal the correlated occupation of that site in its own embedding cluster,    <n_i>_gKS = <n_i>_H^(i)   for all i,  both being spin-summed densities (n_i^gKS = 2 gamma_ii).

Step scientific background

The self-consistency condition of gLPFET is a set of local density constraints: for every site i, the occupation of site i in the gKS reference determinant must equal the correlated occupation of that site in its own embedding cluster,

  <n_i>_gKS = <n_i>_H^(i)   for all i,

both being spin-summed densities (n_i^gKS = 2 gamma_ii). The objective of the outer optimization is the Euclidean norm of the mismatch vector,

  Delta(v_c) = sqrt( sum_i ( n_i^cl(v_c) - n_i^gKS(v_c) )^2 ).

Both sides depend on the correlation potential v_c, the gKS densities directly through the Fock matrix and the cluster densities through the bath orbitals, the core, the projected Hamiltonians and the impurity chemical potentials (which are functionals of v_c and of the bath orbitals). This step assembles the whole chain for a given v_c: solve the gKS reference, build the bath orbital and the projected cluster Hamiltonian of every site, evaluate the impurity chemical potential of every cluster, solve every cluster exactly and return the mismatch vector site by site. Note that the projected cluster Hamiltonian contains the bare Hamiltonian only: neither v_c nor the Hartree-Fock potential of the reference enters the cluster explicitly; v_c acts on the clusters only through the impurity chemical potential and through the reference density matrix.

At U = 0 the clusterisation of the reference determinant is exact, so the mismatch vanishes at v_c = 0; for a general v_c it does not, because the clusters are built from the bare Hamiltonian while the reference contains v_c.

Returns
-------
np.ndarray of float with shape (L,): cluster density minus gKS density on every site, for the given correlation potential v_c.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def density_mismatch(h, U, v_c, n_elec):
    '''Site-resolved gLPFET density mismatch for a given correlation potential.

    Parameters
    ----------
    h : array_like of float, shape (L, L)
        Real symmetric one-electron matrix of the ring.
    U : float
        On-site repulsion.
    v_c : array_like of float, shape (L,)
        Local correlation potential of the gKS reference.
    n_elec : int
        Even number of electrons, 2 <= n_elec <= 2L - 2.

    Returns
    -------
    r : np.ndarray of float, shape (L,)
        r[i] = n_i^cl - n_i^gKS, the spin-summed correlated occupation of
        site i from its embedding cluster minus the gKS occupation of the
        same site. Raises ValueError for invalid h, U, v_c or n_elec (same
        rules as the reference and cluster steps).
    '''
    return np.zeros(np.asarray(h).shape[0], dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _gks_reference(h, U, v_c, n_occ):
    L = h.shape[0]
    gamma = np.zeros((L, L))
    focks, errors = [], []
    for _ in range(5000):
        fock = h + np.diag(v_c) + np.diag(U * np.diag(gamma))
        err = fock @ gamma - gamma @ fock
        focks.append(fock)
        errors.append(err.ravel())
        if len(focks) > 8:
            focks.pop(0)
            errors.pop(0)
        fock_use = fock
        m = len(focks)
        if m > 1:
            bmat = -np.ones((m + 1, m + 1))
            bmat[m, m] = 0.0
            for a in range(m):
                for b in range(m):
                    bmat[a, b] = errors[a] @ errors[b]
            rhs = np.zeros(m + 1)
            rhs[m] = -1.0
            try:
                coef = np.linalg.solve(bmat, rhs)[:m]
                if np.all(np.isfinite(coef)):
                    fock_use = sum(c * f for c, f in zip(coef, focks))
            except np.linalg.LinAlgError:
                pass
        _, cmat = np.linalg.eigh(fock_use)
        g_new = cmat[:, :n_occ] @ cmat[:, :n_occ].T
        if np.max(np.abs(g_new - gamma)) < 1e-12 and np.max(np.abs(err)) < 1e-10:
            gamma = g_new
            break
        gamma = g_new
    else:
        raise RuntimeError("gKS self-consistency did not converge")
    fock = h + np.diag(v_c) + np.diag(U * np.diag(gamma))
    levels = np.linalg.eigvalsh(fock)
    if levels[n_occ] - levels[n_occ - 1] < 1e-8:
        raise ValueError("degenerate Fermi level: closed-shell occupation undefined")
    return gamma


def _cluster_density(h, U, gamma, v_c, i):
    L = h.shape[0]
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
    gamma_core = (np.eye(L) - q) @ gamma @ (np.eye(L) - q)
    hc = cmat.T @ (h + np.diag(U * np.diag(gamma_core))) @ cmat
    e_i, e_b, tau = hc[0, 0], hc[1, 1], hc[0, 1]
    u_b = U * float(np.sum(b ** 4))
    mu = float(np.sum(b * b * v_c))
    hm = np.array([
        [2.0 * e_i + U - 2.0 * mu, tau, tau, 0.0],
        [tau, e_i + e_b - mu, 0.0, tau],
        [tau, 0.0, e_i + e_b - mu, tau],
        [0.0, tau, tau, 2.0 * e_b + u_b],
    ])
    w, v = np.linalg.eigh(hm)
    if w[1] - w[0] < 1e-10:
        raise ValueError("degenerate cluster ground level: occupation undefined")
    psi = v[:, 0]
    return float(2.0 * psi[0] ** 2 + psi[1] ** 2 + psi[2] ** 2)


def _oracle_density_mismatch(h, U, v_c, n_elec):
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2:
        raise ValueError("h must be a square matrix")
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be finite and symmetric")
    L = h.shape[0]
    U = float(U)
    if not np.isfinite(U):
        raise ValueError("U must be finite")
    v_c = np.asarray(v_c, dtype=float)
    if v_c.ndim != 1 or v_c.shape[0] != L or not np.all(np.isfinite(v_c)):
        raise ValueError("v_c must be a finite vector with L entries")
    if isinstance(n_elec, bool) or not isinstance(n_elec, (int, np.integer)):
        raise ValueError("n_elec must be an integer")
    if n_elec % 2 != 0 or n_elec < 2 or n_elec > 2 * L - 2:
        raise ValueError("n_elec must be even and satisfy 2 <= n_elec <= 2L - 2")
    gamma = _gks_reference(h, U, v_c, int(n_elec) // 2)
    n_gks = 2.0 * np.diag(gamma)
    n_cl = np.array([_cluster_density(h, U, gamma, v_c, i) for i in range(L)])
    return n_cl - n_gks

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    ring = """import numpy as np
def _ring(L, t, v):
    h = np.diag(np.asarray(v, dtype=float))
    for i in range(L):
        h[i, (i + 1) % L] -= t
        h[(i + 1) % L, i] -= t
    return h
h6 = _ring(6, 1.0, [-1.0, 2.0, -2.0, 3.0, -3.0, 1.0])
"""
    return [
        # --- Normal: the task ring at U = 7 with a vanishing correlation potential ---
        {
            "setup": ring,
            "call": "np.round(density_mismatch(h6, 7.0, np.zeros(6), 6), 6)",
            "gold_call": "np.round(_oracle_density_mismatch(h6, 7.0, np.zeros(6), 6), 6)",
        },
        # --- Normal: a trial correlation potential at U = 4 ---
        {
            "setup": ring + "vc = np.array([0.5, -0.6, 0.6, -0.4, 0.4, -0.5])\n",
            "call": "np.round(density_mismatch(h6, 4.0, vc, 6), 6)",
            "gold_call": "np.round(_oracle_density_mismatch(h6, 4.0, vc, 6), 6)",
        },
        # --- Normal: four-site non-uniform ring at U = 3 with two electrons ---
        {
            "setup": ring + "h4 = _ring(4, 1.0, [0.5, -0.5, 1.5, -1.5])\n",
            "call": "np.round(density_mismatch(h4, 3.0, np.array([0.1, -0.1, 0.2, -0.2]), 2), 6)",
            "gold_call": "np.round(_oracle_density_mismatch(h4, 3.0, np.array([0.1, -0.1, 0.2, -0.2]), 2), 6)",
        },
        # --- Boundary: U = 0 with a non-trivial potential (the mismatch does not vanish) ---
        {
            "setup": ring + "vc = np.array([0.3, -0.2, 0.1, 0.4, -0.5, 0.2])\n",
            "call": "np.round(density_mismatch(h6, 0.0, vc, 6), 8)",
            "gold_call": "np.round(_oracle_density_mismatch(h6, 0.0, vc, 6), 8)",
        },
        # --- Edge: below 1e-6 at the converged U = 7 potential, above 1e-3 away from it ---
        {
            "setup": ring + """vc7 = np.array([0.7496959882, -1.2732482393, 1.3056007117,
                -1.5616000843, 1.5511455089, -0.7807338046])
""",
            "call": "int(np.linalg.norm(density_mismatch(h6, 7.0, vc7, 6)) < 1e-6 and np.linalg.norm(density_mismatch(h6, 7.0, 0.9 * vc7, 6)) > 1e-3)",
            "gold_call": "int(np.linalg.norm(_oracle_density_mismatch(h6, 7.0, vc7, 6)) < 1e-6 and np.linalg.norm(_oracle_density_mismatch(h6, 7.0, 0.9 * vc7, 6)) > 1e-3)",
        },
        # --- Invalid: odd electron number ---
        {
            "setup": ring + """
def run_model():
    try:
        density_mismatch(h6, 7.0, np.zeros(6), 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_density_mismatch(h6, 7.0, np.zeros(6), 3)
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
