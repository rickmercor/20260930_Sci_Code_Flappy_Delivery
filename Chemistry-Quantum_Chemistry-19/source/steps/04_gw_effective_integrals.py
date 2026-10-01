"""
Each dRPA eigenvector is defined up to an overall sign, which propagates to M(p, q, nu) for that mode. Every physical quantity built later is bilinear in the integrals of a given mode, so this sign drops out. The tests of this step therefore compare sign-insensitive combinations; the oracle fixes the sign by making the largest-magnitude component of X + Y of each mode positive, and the modes are ordered by increasing excitation energy.

In the GW approximation the dynamically screened interaction is expanded over the dRPA excitations, and every self-energy expression can be written with a single set of three-index quantities, the GW effective integrals M(p, q, nu). They are the transition densities of the dRPA modes contracted with the bare Coulomb integrals: in the spatial-orbital, spin-restricted formalism of this task M(p, q, nu) is the square root of two times the sum over occupied-virtual pairs (i, a) of bracket p a bar q i X(ia, nu) plus bracket p i bar q a Y(ia, nu), where (X, Y) are the dRPA eigenvectors with the standard normalization X^T X - Y^T Y = 1. The square root of two is the spin adaptation factor of the singlet screening; its omission halves every self-energy contribution. With these integrals the GW self-energy is a sum-over-states with poles at eps_i - Omega_nu (hole branch) and eps_a + Omega_nu (particle branch) and residues M(p, i, nu) M(q, i, nu) and M(p, a, nu) M(q, a, nu).

Returns
-------
np.ndarray of shape (n, n, n_occ * (n - n_occ)): the GW effective integrals M[p, q, nu], modes in increasing excitation energy, in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gw_effective_integrals(orbital_energies, eri, n_occ):
    '''GW effective integrals M(p, q, nu) built from the singlet dRPA modes.

    Parameters
    ----------
    orbital_energies : array_like of float, shape (n,)
        Canonical orbital energies in increasing order, in eV.
    eri : array_like of float, shape (n, n, n, n)
        Two-electron integrals bracket p q bar r s in Dirac notation, in eV.
    n_occ : int
        Number of doubly occupied orbitals (the first n_occ entries).

    Returns
    -------
    M : np.ndarray, shape (n, n, n_occ * (n - n_occ))
        M[p, q, nu] with the modes nu ordered by increasing dRPA excitation
        energy and each mode phased so that the largest-magnitude component
        of X + Y is positive, in eV.
        Raises ValueError on inconsistent shapes, non-finite input, or
        n_occ outside 1 .. n - 1.
    '''
    n = len(orbital_energies)
    M = np.zeros((n, n, int(n_occ) * (n - int(n_occ))))
    return M

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_gw_effective_integrals(orbital_energies, eri, n_occ):
    eps = np.asarray(orbital_energies, dtype=float)
    g = np.asarray(eri, dtype=float)
    n = eps.shape[0]
    if eps.ndim != 1 or n < 2 or g.shape != (n, n, n, n):
        raise ValueError("orbital_energies must have n entries and eri shape (n, n, n, n)")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(g))):
        raise ValueError("inputs must be finite")
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    n_occ = int(n_occ)
    pairs = [(i, a) for i in range(n_occ) for a in range(n_occ, n)]
    nov = len(pairs)
    A = np.zeros((nov, nov))
    B = np.zeros((nov, nov))
    for k, (i, a) in enumerate(pairs):
        for l, (j, b) in enumerate(pairs):
            A[k, l] = (eps[a] - eps[i]) * (1.0 if k == l else 0.0) + 2.0 * g[a, j, i, b]
            B[k, l] = 2.0 * g[a, b, i, j]
    AmB = A - B
    w_amb, U = np.linalg.eigh(0.5 * (AmB + AmB.T))
    if np.any(w_amb <= 0.0):
        raise ValueError("A - B is not positive definite")
    S = np.dot(U * np.sqrt(w_amb), U.T)
    Sinv = np.dot(U / np.sqrt(w_amb), U.T)
    w2, Z = np.linalg.eigh(np.linalg.multi_dot([S, A + B, S]))
    if np.any(w2 <= 0.0):
        raise ValueError("dRPA problem has no real positive spectrum")
    order = np.argsort(w2)
    w2 = w2[order]
    Z = Z[:, order]
    Om = np.sqrt(w2)
    XpY = np.dot(S, Z) / np.sqrt(Om)
    XmY = np.dot(Sinv, Z) * np.sqrt(Om)
    for v in range(nov):
        k = np.argmax(np.abs(XpY[:, v]))
        if XpY[k, v] < 0.0:
            XpY[:, v] *= -1.0
            XmY[:, v] *= -1.0
    X = 0.5 * (XpY + XmY)
    Y = 0.5 * (XpY - XmY)
    M = np.zeros((n, n, nov))
    for k, (i, a) in enumerate(pairs):
        M += np.sqrt(2.0) * (g[:, a, :, i][:, :, None] * X[k][None, None, :] + g[:, i, :, a][:, :, None] * Y[k][None, None, :])
    return M

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
def _model(lengths, hops, site, U, n_occ):
    pos = np.concatenate([[0.0], np.cumsum(lengths)])
    r = np.abs(pos[:, None] - pos[None, :])
    V = 14.397 / np.sqrt((14.397 / U) ** 2 + r ** 2)
    n = len(site)
    h = np.zeros((n, n))
    for k in range(n - 1):
        h[k, k + 1] = h[k + 1, k] = hops[k]
    for m in range(n):
        h[m, m] = site[m] - (np.sum(V[m]) - V[m, m])
    _, C = np.linalg.eigh(h)
    P = 2.0 * np.dot(C[:, :n_occ], C[:, :n_occ].T)
    for _ in range(2000):
        F = h + np.diag(np.dot(V, np.diag(P))) - 0.5 * P * V
        _, C = np.linalg.eigh(F)
        Pn = 2.0 * np.dot(C[:, :n_occ], C[:, :n_occ].T)
        if np.max(np.abs(Pn - P)) < 1e-12:
            P = Pn
            break
        P = 0.5 * (P + Pn)
    F = h + np.diag(np.dot(V, np.diag(P))) - 0.5 * P * V
    eps, C = np.linalg.eigh(F)
    for p in range(n):
        k = np.argmax(np.abs(C[:, p]))
        if C[k, p] < 0.0:
            C[:, p] *= -1.0
    eri = np.einsum("mp,nq,mr,ns,mn->pqrs", C, C, C, C, V, optimize=True)
    return eps, eri
def _inv(M):
    # sign-insensitive view of M: magnitudes per mode and the mode-summed quadratic form
    M = np.asarray(M, dtype=float)
    return np.round(np.concatenate([np.abs(M).ravel(), np.einsum('pqn,rsn->pqrs', M, M).ravel()]), 8)
eps6, eri6 = _model([1.34, 1.47, 1.36, 1.46, 1.38], [-2.23, -1.62, -2.02, -2.05, -2.10], [-14.2, -11.2, -10.5, -11.9, -10.4, -10.4], 14.0, 3)
eps4, eri4 = _model([1.35, 1.45, 1.37], [-2.4, -1.9, -2.3], [-12.6, -11.0, -11.5, -10.2], 11.0, 2)
"""
    return [
        # --- Normal: the reference chain ---
        {
            "setup": setup,
            "call": "_inv(gw_effective_integrals(eps6, eri6, 3))",
            "gold_call": "_inv(_oracle_gw_effective_integrals(eps6, eri6, 3))",
        },
        # --- Normal: four-centre chain ---
        {
            "setup": setup,
            "call": "_inv(gw_effective_integrals(eps4, eri4, 2))",
            "gold_call": "_inv(_oracle_gw_effective_integrals(eps4, eri4, 2))",
        },
        # --- Boundary: weak interaction (Y small but not zero) ---
        {
            "setup": setup,
            "call": "_inv(gw_effective_integrals(eps6, 0.05 * eri6, 3))",
            "gold_call": "_inv(_oracle_gw_effective_integrals(eps6, 0.05 * eri6, 3))",
        },
        # --- Edge: two centres, one mode (X and Y are scalars) ---
        {
            "setup": setup + "eps2, eri2 = _model([1.40], [-2.5], [-11.0, -12.4], 10.0, 1)\n",
            "call": "_inv(gw_effective_integrals(eps2, eri2, 1))",
            "gold_call": "_inv(_oracle_gw_effective_integrals(eps2, eri2, 1))",
        },
        # --- Edge: stronger interaction, where the Y amplitudes are sizeable ---
        {
            "setup": setup,
            "call": "_inv(gw_effective_integrals(eps6, 1.7 * eri6, 3))",
            "gold_call": "_inv(_oracle_gw_effective_integrals(eps6, 1.7 * eri6, 3))",
        },
        # --- Invalid: n_occ out of range ---
        {
            "setup": setup + """
def run_model():
    try:
        gw_effective_integrals(eps6, eri6, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_gw_effective_integrals(eps6, eri6, 0)
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
