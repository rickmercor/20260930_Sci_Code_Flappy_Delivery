"""
This step builds that Hamiltonian and solves the closed-shell restricted Hartree-Fock problem for it. The Fock matrix in the site basis is the one-body matrix plus the Coulomb term (the Ohno matrix applied to the vector of site populations, on the diagonal) minus one half of the elementwise product of the spin-summed density matrix with the Ohno matrix. Iterate to self-consistency with a tight threshold (change of the density matrix below 1e-10) so that the Fock matrix is reproducible to better than 1e-8 eV; plain iteration with averaging of successive density matrices converges for the chains of this task. The converged Fock matrix is the reference for everything that follows: its eigenvalues are the orbital energies and its eigenvectors the canonical orbitals.

The pi electrons of a conjugated chain are described by a Pariser-Parr-Pople (PPP) Hamiltonian: one orbital per centre, a site energy on every centre, nearest-neighbour hopping along the bonds, an on-site repulsion U between opposite-spin electrons on the same centre, and an Ohno-screened Coulomb repulsion between the net charges of different centres. The net-charge form of the intersite term, one half of the sum over ordered pairs of distinct centres of V times (n_mu - 1)(n_nu - 1), contains a neutral background of one positive charge per centre; expanding it moves a constant and a one-body shift of minus the sum of V over the other centres onto every site energy. The centres lie on a straight line at the given bond lengths, the Ohno interaction between two centres a distance r apart is e^2 divided by the square root of (e^2 / U)^2 + r^2 with e^2 = 14.397 eV Angstrom, and the on-site value of the Ohno matrix is U itself.

Returns
-------
np.ndarray of shape (n, n): the converged restricted Hartree-Fock Fock matrix of the chain in the site basis, in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ppp_fock_matrix(bond_lengths, hoppings, site_energies, hubbard_u, n_occ):
    '''Converged restricted Hartree-Fock Fock matrix of a linear PPP chain.

    Parameters
    ----------
    bond_lengths : array_like of float, shape (n - 1,)
        Distances in Angstrom between consecutive centres along the line.
    hoppings : array_like of float, shape (n - 1,)
        Hopping integrals in eV between consecutive centres (bond k joins
        centres k and k + 1).
    site_energies : array_like of float, shape (n,)
        Site energies in eV of the n centres (before the background shift).
    hubbard_u : float
        On-site repulsion U in eV (also the on-site value of the Ohno matrix).
    n_occ : int
        Number of doubly occupied orbitals (the chain holds 2 n_occ electrons).

    Returns
    -------
    fock : np.ndarray, shape (n, n)
        Converged Fock matrix in the site basis, in eV.
        Raises ValueError if the array lengths are inconsistent, if any
        bond length is not positive, if hubbard_u is not positive, or if
        n_occ is not an integer between 1 and n.
    '''
    fock = np.zeros((len(site_energies), len(site_energies)))
    return fock

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_ppp_fock_matrix(bond_lengths, hoppings, site_energies, hubbard_u, n_occ):
    lengths = np.asarray(bond_lengths, dtype=float)
    t = np.asarray(hoppings, dtype=float)
    eps_site = np.asarray(site_energies, dtype=float)
    n = eps_site.shape[0]
    if eps_site.ndim != 1 or n < 2:
        raise ValueError("site_energies must be a vector with at least two entries")
    if lengths.shape != (n - 1,) or t.shape != (n - 1,):
        raise ValueError("bond_lengths and hoppings must have n - 1 entries")
    if not (np.all(np.isfinite(lengths)) and np.all(np.isfinite(t)) and np.all(np.isfinite(eps_site))):
        raise ValueError("inputs must be finite")
    if np.any(lengths <= 0.0):
        raise ValueError("bond lengths must be positive")
    if not np.isfinite(hubbard_u) or hubbard_u <= 0.0:
        raise ValueError("hubbard_u must be positive")
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n:
        raise ValueError("n_occ must be an integer between 1 and n")
    n_occ = int(n_occ)
    e2 = 14.397
    pos = np.concatenate([[0.0], np.cumsum(lengths)])
    r = np.abs(pos[:, None] - pos[None, :])
    V = e2 / np.sqrt((e2 / hubbard_u) ** 2 + r ** 2)
    h = np.zeros((n, n))
    for k in range(n - 1):
        h[k, k + 1] = t[k]
        h[k + 1, k] = t[k]
    for m in range(n):
        h[m, m] = eps_site[m] - (np.sum(V[m]) - V[m, m])
    _, C = np.linalg.eigh(h)
    P = 2.0 * np.dot(C[:, :n_occ], C[:, :n_occ].T)
    for _ in range(2000):
        F = h + np.diag(np.dot(V, np.diag(P))) - 0.5 * P * V
        _, C = np.linalg.eigh(F)
        P_new = 2.0 * np.dot(C[:, :n_occ], C[:, :n_occ].T)
        if np.max(np.abs(P_new - P)) < 1e-12:
            P = P_new
            break
        P = 0.5 * (P + P_new)
    F = h + np.diag(np.dot(V, np.diag(P))) - 0.5 * P * V
    return F

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
lengths = [1.34, 1.47, 1.36, 1.46, 1.38]
hops = [-2.23, -1.62, -2.02, -2.05, -2.10]
site = [-14.2, -11.2, -10.5, -11.9, -10.4, -10.4]
"""
    return [
        # --- Normal: the reference chain of the task ---
        {
            "setup": setup,
            "call": "np.round(ppp_fock_matrix(lengths, hops, site, 14.0, 3), 6)",
            "gold_call": "np.round(_oracle_ppp_fock_matrix(lengths, hops, site, 14.0, 3), 6)",
        },
        # --- Normal: a four-centre chain with a different interaction strength ---
        {
            "setup": setup + "l4 = [1.35, 1.45, 1.37]\nh4 = [-2.4, -1.9, -2.3]\ns4 = [-12.6, -11.0, -11.5, -10.2]\n",
            "call": "np.round(ppp_fock_matrix(l4, h4, s4, 11.0, 2), 6)",
            "gold_call": "np.round(_oracle_ppp_fock_matrix(l4, h4, s4, 11.0, 2), 6)",
        },
        # --- Boundary: a two-centre chain (one doubly occupied orbital) ---
        {
            "setup": setup,
            "call": "np.round(ppp_fock_matrix([1.40], [-2.5], [-11.0, -12.0], 10.0, 1), 6)",
            "gold_call": "np.round(_oracle_ppp_fock_matrix([1.40], [-2.5], [-11.0, -12.0], 10.0, 1), 6)",
        },
        # --- Edge: the Coulomb-free limit of the intersite term (very long bonds) only
        #     keeps U and the hoppings; the background shift must still be applied ---
        {
            "setup": setup + "far = [40.0, 40.0, 40.0]\n",
            "call": "np.round(ppp_fock_matrix(far, [-1.0, -0.5, -1.2], [-11.0, -11.5, -10.5, -12.0], 9.0, 2), 6)",
            "gold_call": "np.round(_oracle_ppp_fock_matrix(far, [-1.0, -0.5, -1.2], [-11.0, -11.5, -10.5, -12.0], 9.0, 2), 6)",
        },
        # --- Edge: fully occupied chain (n_occ = n) ---
        {
            "setup": setup,
            "call": "np.round(ppp_fock_matrix([1.4, 1.4], [-2.0, -2.0], [-11.0, -11.0, -11.0], 10.0, 3), 6)",
            "gold_call": "np.round(_oracle_ppp_fock_matrix([1.4, 1.4], [-2.0, -2.0], [-11.0, -11.0, -11.0], 10.0, 3), 6)",
        },
        # --- Invalid: inconsistent array lengths ---
        {
            "setup": setup + """
def run_model():
    try:
        ppp_fock_matrix([1.4, 1.4], [-2.0], [-11.0, -11.0, -11.0], 10.0, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_ppp_fock_matrix([1.4, 1.4], [-2.0], [-11.0, -11.0, -11.0], 10.0, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive U ---
        {
            "setup": setup + """
def run_model():
    try:
        ppp_fock_matrix(lengths, hops, site, 0.0, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_ppp_fock_matrix(lengths, hops, site, 0.0, 3)
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
