"""
The outer loop of gLPFET determines the local correlation potential v_c from the local density constraints: the L unknowns v_c[0..L-1] are fixed by the L equations    n_i^cl(v_c) - n_i^gKS(v_c) = 0,   i = 0..L-1.

Step scientific background

The outer loop of gLPFET determines the local correlation potential v_c from the local density constraints: the L unknowns v_c[0..L-1] are fixed by the L equations

  n_i^cl(v_c) - n_i^gKS(v_c) = 0,   i = 0..L-1.

Unlike in DET, where a constant shift of the Hxc potential is compensated by the global chemical potential and the potential is only defined up to that shift, in gLPFET a constant shift of v_c changes every impurity chemical potential (because the bath orbitals are normalized) and therefore changes the cluster densities, so the potential is uniquely determined and no additional electron-number constraint is needed. Starting from v_c = 0 (the Hartree-Fock reference), the mismatch vector is driven to zero with a standard multidimensional root finder (or, equivalently, the mismatch norm Delta of the previous step is minimized to zero). Every evaluation of the mismatch re-solves the gKS reference self-consistently, because the Hartree-Fock potential, the bath orbitals and the cores all respond to v_c. The converged potential reproduces in the gKS determinant the correlated densities of the clusters. At U = 0 the mismatch vanishes at the starting point v_c = 0 (for a non-zero trial potential it does not, since the clusters see the bare Hamiltonian while the reference contains v_c), so the solver returns the zero potential.

Returns
-------
np.ndarray of float with shape (L,): the converged gLPFET correlation potential (mismatch norm below tol), starting from v_c = 0.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_correlation_potential(h, U, n_elec, tol=1e-10):
    '''Self-consistent gLPFET correlation potential of the ring.

    Parameters
    ----------
    h : array_like of float, shape (L, L)
        Real symmetric one-electron matrix of the ring.
    U : float
        On-site repulsion.
    n_elec : int
        Even number of electrons, 2 <= n_elec <= 2L - 2.
    tol : float
        Required upper bound on the converged mismatch norm Delta.

    Returns
    -------
    v_c : np.ndarray of float, shape (L,)
        Local correlation potential for which the cluster densities and the
        gKS densities agree on every site (mismatch norm below tol), obtained
        from the starting point v_c = 0. Raises ValueError for invalid inputs
        (same rules as the reference step) and RuntimeError if the density
        mapping cannot be converged.
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


def _mismatch(v_c, h, U, n_occ):
    gamma = _gks_reference(h, U, v_c, n_occ)
    n_gks = 2.0 * np.diag(gamma)
    n_cl = np.array([_cluster_density(h, U, gamma, v_c, i) for i in range(h.shape[0])])
    return n_cl - n_gks


def _oracle_solve_correlation_potential(h, U, n_elec, tol=1e-10):
    from scipy.optimize import root
    h = np.asarray(h, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] < 2:
        raise ValueError("h must be a square matrix")
    if not np.all(np.isfinite(h)) or not np.allclose(h, h.T, rtol=0.0, atol=1e-10):
        raise ValueError("h must be finite and symmetric")
    L = h.shape[0]
    U = float(U)
    if not np.isfinite(U):
        raise ValueError("U must be finite")
    if isinstance(n_elec, bool) or not isinstance(n_elec, (int, np.integer)):
        raise ValueError("n_elec must be an integer")
    if n_elec % 2 != 0 or n_elec < 2 or n_elec > 2 * L - 2:
        raise ValueError("n_elec must be even and satisfy 2 <= n_elec <= 2L - 2")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be a positive number")
    n_occ = int(n_elec) // 2
    v0 = np.zeros(L)
    if np.linalg.norm(_mismatch(v0, h, U, n_occ)) < tol:
        return v0
    sol = root(_mismatch, v0, args=(h, U, n_occ), method="hybr", tol=1e-13)
    v_c = np.asarray(sol.x, dtype=float)
    if np.linalg.norm(_mismatch(v_c, h, U, n_occ)) > tol:
        raise RuntimeError("the gLPFET density mapping did not converge")
    return v_c

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
        # --- Normal: the task ring at U = 7 ---
        {
            "setup": ring,
            "call": "np.round(solve_correlation_potential(h6, 7.0, 6), 5)",
            "gold_call": "np.round(_oracle_solve_correlation_potential(h6, 7.0, 6), 5)",
        },
        # --- Normal: weak coupling U = 2 ---
        {
            "setup": ring,
            "call": "np.round(solve_correlation_potential(h6, 2.0, 6), 5)",
            "gold_call": "np.round(_oracle_solve_correlation_potential(h6, 2.0, 6), 5)",
        },
        # --- Normal: four-site non-uniform ring, two electrons, U = 3 ---
        {
            "setup": ring + "h4 = _ring(4, 1.0, [0.5, -0.5, 1.5, -1.5])\n",
            "call": "np.round(solve_correlation_potential(h4, 3.0, 2), 5)",
            "gold_call": "np.round(_oracle_solve_correlation_potential(h4, 3.0, 2), 5)",
        },
        # --- Boundary: U = 0 returns the zero potential; an interacting solve is
        # appended so the case cannot be passed by always returning zeros ---
        {
            "setup": ring + "h4 = _ring(4, 1.0, [0.5, -0.5, 1.5, -1.5])\n",
            "call": "np.round(np.concatenate([solve_correlation_potential(h6, 0.0, 6), solve_correlation_potential(h4, 3.0, 2)]), 5)",
            "gold_call": "np.round(np.concatenate([_oracle_solve_correlation_potential(h6, 0.0, 6), _oracle_solve_correlation_potential(h4, 3.0, 2)]), 5)",
        },
        # --- Edge: the converged potential makes the densities match (encoded check) ---
        {
            "setup": ring + """
def _check(fn):
    vc = fn(h6, 10.0, 6)
    return int(np.all(np.isfinite(vc)) and vc.shape == (6,) and abs(vc[0] - 0.8507124) < 1e-5)
""",
            "call": "_check(solve_correlation_potential)",
            "gold_call": "_check(_oracle_solve_correlation_potential)",
        },
        # --- Invalid: electron number larger than 2L - 2 ---
        {
            "setup": ring + """
def run_model():
    try:
        solve_correlation_potential(h6, 7.0, 12)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_correlation_potential(h6, 7.0, 12)
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
