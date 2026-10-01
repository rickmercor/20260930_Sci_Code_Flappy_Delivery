"""
This step returns the dRPA excitation energies in increasing order. They are the "dressed" energies that enter every pole of the self-energy of this task: the two-hole-one-particle configurations sit at eps_i - Omega_nu, the two-particle-one-hole ones at eps_a + Omega_nu, and the three-body ones at eps_i - Omega_nu - Omega_mu and eps_a + Omega_nu + Omega_mu. Replacing the dRPA by its Tamm-Dancoff approximation (dropping B) changes all of them.

The screened interaction of the GW family is built from the neutral excitations of the direct random-phase approximation (dRPA). In the spin-restricted, spatial-orbital formulation used here the singlet dRPA problem couples the occupied-virtual pairs (i, a) through the matrices A(ia, jb) = (eps_a - eps_i) delta_ij delta_ab + 2 bracket a j bar i b and B(ia, jb) = 2 bracket a b bar i j, with no exchange integrals at all (this is what makes the approximation "direct"). The excitation energies Omega_nu are the positive eigenvalues of the non-Hermitian pair problem (A, B; -B, -A), which for real orbitals is conveniently solved through the symmetric matrix (A - B)^(1/2) (A + B) (A - B)^(1/2), whose eigenvalues are the squares of the Omega_nu. For a direct-only kernel A - B is diagonal and positive, so the square root is trivial.

Returns
-------
np.ndarray of shape (n_occ * (n - n_occ),): the singlet direct-RPA excitation energies in increasing order, in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def drpa_excitation_energies(orbital_energies, eri, n_occ):
    '''Singlet direct-RPA excitation energies of a closed-shell reference.

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
    omega : np.ndarray, shape (n_occ * (n - n_occ),)
        dRPA excitation energies in increasing order, in eV.
        Raises ValueError if the shapes are inconsistent, if n_occ is not an
        integer between 1 and n - 1, or if the pair problem has no real
        positive spectrum.
    '''
    n = len(orbital_energies)
    omega = np.zeros(int(n_occ) * (n - int(n_occ)))
    return omega

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _drpa_pairs(eps, eri, n_occ):
    n = eps.shape[0]
    pairs = [(i, a) for i in range(n_occ) for a in range(n_occ, n)]
    nov = len(pairs)
    A = np.zeros((nov, nov))
    B = np.zeros((nov, nov))
    for k, (i, a) in enumerate(pairs):
        for l, (j, b) in enumerate(pairs):
            A[k, l] = (eps[a] - eps[i]) * (1.0 if k == l else 0.0) + 2.0 * eri[a, j, i, b]
            B[k, l] = 2.0 * eri[a, b, i, j]
    return pairs, A, B


def _oracle_drpa_excitation_energies(orbital_energies, eri, n_occ):
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
    _, A, B = _drpa_pairs(eps, g, n_occ)
    AmB = A - B
    w_amb, U = np.linalg.eigh(0.5 * (AmB + AmB.T))
    if np.any(w_amb <= 0.0):
        raise ValueError("A - B is not positive definite")
    S = np.dot(U * np.sqrt(w_amb), U.T)
    w2 = np.linalg.eigvalsh(np.linalg.multi_dot([S, A + B, S]))
    if np.any(w2 <= 0.0):
        raise ValueError("dRPA problem has no real positive spectrum")
    return np.sort(np.sqrt(w2))

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
eps6, eri6 = _model([1.34, 1.47, 1.36, 1.46, 1.38], [-2.23, -1.62, -2.02, -2.05, -2.10], [-14.2, -11.2, -10.5, -11.9, -10.4, -10.4], 14.0, 3)
eps4, eri4 = _model([1.35, 1.45, 1.37], [-2.4, -1.9, -2.3], [-12.6, -11.0, -11.5, -10.2], 11.0, 2)
"""
    return [
        # --- Normal: the reference chain (nine excitations) ---
        {
            "setup": setup,
            "call": "np.round(drpa_excitation_energies(eps6, eri6, 3), 8)",
            "gold_call": "np.round(_oracle_drpa_excitation_energies(eps6, eri6, 3), 8)",
        },
        # --- Normal: four-centre chain (four excitations) ---
        {
            "setup": setup,
            "call": "np.round(drpa_excitation_energies(eps4, eri4, 2), 8)",
            "gold_call": "np.round(_oracle_drpa_excitation_energies(eps4, eri4, 2), 8)",
        },
        # --- Boundary: non-interacting limit returns the bare orbital gaps ---
        {
            "setup": setup,
            "call": "np.round(drpa_excitation_energies(eps6, np.zeros_like(eri6), 3), 8)",
            "gold_call": "np.round(_oracle_drpa_excitation_energies(eps6, np.zeros_like(eri6), 3), 8)",
        },
        # --- Edge: a single occupied orbital (one of the two centres) ---
        {
            "setup": setup + "eps2, eri2 = _model([1.40], [-2.5], [-11.0, -12.4], 10.0, 1)\n",
            "call": "np.round(drpa_excitation_energies(eps2, eri2, 1), 8)",
            "gold_call": "np.round(_oracle_drpa_excitation_energies(eps2, eri2, 1), 8)",
        },
        # --- Edge: a scaled interaction (the B matrix matters; the Tamm-Dancoff
        #     result would differ) ---
        {
            "setup": setup,
            "call": "np.round(drpa_excitation_energies(eps6, 1.7 * eri6, 3), 8)",
            "gold_call": "np.round(_oracle_drpa_excitation_energies(eps6, 1.7 * eri6, 3), 8)",
        },
        # --- Invalid: n_occ out of range ---
        {
            "setup": setup + """
def run_model():
    try:
        drpa_excitation_energies(eps6, eri6, 6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_drpa_excitation_energies(eps6, eri6, 6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: inconsistent shapes ---
        {
            "setup": setup + """
def run_model():
    try:
        drpa_excitation_energies(eps4, eri6, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_drpa_excitation_energies(eps4, eri6, 2)
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
