"""
Solve the restricted closed-shell Hartree-Fock problem of the task's Pariser-Parr-Pople chain at half filling (one electron per site): n_sites sites on a line with unit spacing, nearest-neighbour hopping t_k = -t (1 + delta (-1)^k) on bond k (k = 0 joins sites 0 and 1), on-site repulsion U between opposite spins, and the Ohno inter-site repulsion V_ij = U / sqrt(1 + (U |i - j| / kappa)^2) in its charge-neutral form sum_{i<j} V_ij (n_i - 1)(n_j - 1), so that the one-body matrix carries the on-site shift -sum_{j != i} V_ij and the model has the constant sum_{i<j} V_ij. Start from the eigenvectors of the one-body matrix and iterate the Fock matrix to self-consistency until the one-particle density matrix changes by less than 1e-10 in every element. Return an array whose row 0 holds the n_sites canonical orbital energies in ascending order and whose rows 1 to n_sites hold the orbital coefficient matrix C (C[i, p] = coefficient of site i in orbital p, orbitals in the same ascending order), each column with its phase fixed so that its coefficient on site 0 is positive.

The Pariser-Parr-Pople model is the standard semi-empirical description of conjugated pi systems: a tight-binding chain with an on-site Hubbard repulsion and a long-range, distance-dependent inter-site repulsion; in the charge-neutral form the long-range term contributes a one-body on-site shift and a constant. Closed-shell restricted Hartree-Fock on such a model is a small self-consistent eigenvalue problem whose canonical orbitals and orbital energies are the mean-field reference on which the correlation treatment of the source is built.

Returns
-------
numpy.ndarray of float64 with shape (n_sites + 1, n_sites): row 0 the orbital energies, rows 1 to n_sites the matrix C.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ppp_rhf(n_sites: int, t: float, delta: float, U: float, kappa: float) -> "np.ndarray":
    """Solve the restricted closed-shell Hartree-Fock problem of the task's Pariser-Parr-Pople chain at half filling (one electron per site): n_sites sites on a line with unit spacing, nearest-neighbour hopping t_k = -t (1 + delta (-1)^k) on bond k (k = 0 joins sites 0 and 1), on-site repulsion U between opposite spins, and the Ohno inter-site repulsion V_ij = U / sqrt(1 + (U |i - j| / kappa)^2) in its charge-neutral form sum_{i<j} V_ij (n_i - 1)(n_j - 1), so that the one-body matrix carries the on-site shift -sum_{j != i} V_ij and the model has the constant sum_{i<j} V_ij. Start from the eigenvectors of the one-body matrix and iterate the Fock matrix to self-consistency until the one-particle density matrix changes by less than 1e-10 in every element. Return an array whose row 0 holds the n_sites canonical orbital energies in ascending order and whose rows 1 to n_sites hold the orbital coefficient matrix C (C[i, p] = coefficient of site i in orbital p, orbitals in the same ascending order), each column with its phase fixed so that its coefficient on site 0 is positive.

    Parameters
    ----------
    n_sites : int
        Even number of sites (at least 2); the chain holds n_sites electrons.
    t : float
        Positive hopping scale; energies are returned in the units of t, U and kappa.
    delta : float
        Bond alternation, |delta| < 1.
    U : float
        Positive on-site repulsion.
    kappa : float
        Positive Ohno range parameter.

    Returns
    -------
    orbitals : numpy.ndarray
        Array of shape (n_sites + 1, n_sites), float64: row 0 the canonical orbital energies in ascending order, rows 1 to n_sites the coefficient matrix C with C[i, p] the coefficient of site i in orbital p, each column with a positive coefficient on site 0.

    Raises
    ------
    ValueError
        If n_sites is odd or below 2, |delta| >= 1, or t, U or kappa is not positive.
    """
    return orbitals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_scalar(x, name, positive=False, nonneg=False):
    x = float(x)
    if not np.isfinite(x):
        raise ValueError("%s must be finite" % name)
    if positive and x <= 0:
        raise ValueError("%s must be positive" % name)
    if nonneg and x < 0:
        raise ValueError("%s must be non-negative" % name)
    return x


def _check_int(n, name, minimum=0):
    if int(n) != n or n < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return int(n)


def _ppp_site_hamiltonian(n_sites, t, delta, U, kappa):
    """PPP chain in the neutral form: H = sum_k t_k (a+_k a_k+1 + h.c.) + U sum_i n_i,up n_i,dn
    + sum_{i<j} V_ij (n_i - 1)(n_j - 1); t_k = -t (1 + delta (-1)^k), Ohno V_ij = U / sqrt(1 + (U |i-j| / kappa)^2).
    Returns the one-body site matrix (with the -sum_j V_ij on-site shift), the site-pair interaction V,
    and the constant sum_{i<j} V_ij."""
    h = np.zeros((n_sites, n_sites))
    for k in range(n_sites - 1):
        h[k, k + 1] = h[k + 1, k] = -t * (1.0 + delta * (-1) ** k)
    idx = np.arange(n_sites)
    V = U / np.sqrt(1.0 + (U * np.abs(idx[:, None] - idx[None, :]) / kappa) ** 2)
    for i in range(n_sites):
        h[i, i] = -(np.sum(V[i]) - V[i, i])
    E_const = 0.5 * (np.sum(V) - np.trace(V))
    return h, V, E_const


def _oracle_ppp_rhf(n_sites: int, t: float, delta: float, U: float, kappa: float) -> "np.ndarray":
    n_sites = _check_int(n_sites, "n_sites", 2)
    t = _check_scalar(t, "t", positive=True)
    delta = _check_scalar(delta, "delta")
    U = _check_scalar(U, "U", positive=True)
    kappa = _check_scalar(kappa, "kappa", positive=True)
    if n_sites % 2 or abs(delta) >= 1:
        raise ValueError("n_sites must be even and |delta| < 1")
    h, V, _ = _ppp_site_hamiltonian(n_sites, t, delta, U, kappa)
    n_occ = n_sites // 2
    eps, C = np.linalg.eigh(h)                       # Hueckel guess
    D_old = None
    for _ in range(2000):
        Cocc = C[:, :n_occ]
        D = Cocc @ Cocc.T                            # per-spin density in the site basis
        J = np.diag(V @ np.diag(D))                  # (pq|rs) = d_pq d_rs V_pr in the site basis
        K = V * D
        F = h + 2.0 * J - K
        if D_old is not None:
            F = 0.7 * F + 0.3 * F_old                # mild damping for robustness
        eps, C = np.linalg.eigh(F)
        if D_old is not None and np.max(np.abs(D - D_old)) < 1e-13:
            break
        D_old, F_old = D, F
    else:
        raise ValueError("SCF did not converge")
    Cocc = C[:, :n_occ]
    D = Cocc @ Cocc.T
    F = h + 2.0 * np.diag(V @ np.diag(D)) - V * D
    eps, C = np.linalg.eigh(F)
    for p in range(n_sites):                         # phase: coefficient on site 0 positive
        if C[0, p] < 0:
            C[:, p] *= -1.0
    return np.vstack([eps[None, :], C])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "n_sites, t, delta, U, kappa = 6, 1.0, 0.1, 3.0, 4.0\n",
            "call": "ppp_rhf(n_sites, t, delta, U, kappa)",
            "gold_call": "_oracle_ppp_rhf(n_sites, t, delta, U, kappa)",
            "tol": 1e-07,
        },
        {
            "setup": "n_sites, t, delta, U, kappa = 8, 1.0, 0.07, 4.0, 4.0\n",
            "call": "ppp_rhf(n_sites, t, delta, U, kappa)",
            "gold_call": "_oracle_ppp_rhf(n_sites, t, delta, U, kappa)",
            "tol": 1e-07,
        },
        {
            "setup": "n_sites, t, delta, U, kappa = 10, 1.0, 0.07, 4.0, 6.0\n",
            "call": "ppp_rhf(n_sites, t, delta, U, kappa)",
            "gold_call": "_oracle_ppp_rhf(n_sites, t, delta, U, kappa)",
            "tol": 1e-07,
        },
        {
            "setup": "n_sites, t, delta, U, kappa = 2, 1.0, 0.0, 2.0, 4.0\n",
            "call": "ppp_rhf(n_sites, t, delta, U, kappa)",
            "gold_call": "_oracle_ppp_rhf(n_sites, t, delta, U, kappa)",
            "tol": 1e-07,
        },
        {
            "setup": "n_sites, t, delta, U, kappa = 7, 1.0, 0.07, 4.0, 6.0\ndef run_model():\n    try:\n        ppp_rhf(n_sites, t, delta, U, kappa)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_ppp_rhf(n_sites, t, delta, U, kappa)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
