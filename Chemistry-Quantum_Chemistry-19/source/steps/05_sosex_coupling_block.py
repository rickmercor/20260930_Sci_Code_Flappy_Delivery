"""
The first step beyond GW adds the second-order screened exchange diagrams. In the ADC form they appear as a correction to the coupling block only, and the scheme that keeps this correction together with the GW blocks is the ADC-2SOSEX scheme; it is equivalent to the positive-semidefinite GW+2SOSEX self-energy. The correction to the 2h1p coupling of configuration (i, nu) with orbital q is a sum over occupied-virtual pairs (k, c) of two terms, each a GW effective integral of the mode nu times a bare two-electron integral, divided by a dressed denominator that combines the orbital gap eps_c - eps_k with plus or minus Omega_nu; the two terms differ in the index order of the effective integral and of the bare integral and in the sign of Omega_nu in the denominator. The 2p1h block is its particle-hole mirror image. The explicit expressions are Eqs. (16a) and (16b) of the source paper; this step implements them for one branch.

The algebraic-diagrammatic construction (ADC) writes each branch of the dynamical self-energy as U^dagger (omega - K - C)^(-1) U, with a coupling block U between the one-particle space and the excited configurations, a diagonal block K of zeroth-order configuration energies and a block C of configuration couplings. The upfolded GW self-energy is the simplest member of this family: the configurations are the two-hole-one-particle (2h1p) and two-particle-one-hole (2p1h) states labelled by an orbital and a dRPA mode, K holds the dressed energies eps_i - Omega_nu and eps_a + Omega_nu, the coupling is the GW effective integral itself, and C vanishes.

Returns
-------
np.ndarray of shape (n_rows * n_modes, n): the second-order screened-exchange correction to the coupling block of the requested branch, in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sosex_coupling_block(orbital_energies, eri, M, omega, n_occ, branch):
    '''Second-order screened-exchange correction to the ADC coupling block.

    Parameters
    ----------
    orbital_energies : array_like of float, shape (n,)
        Canonical orbital energies in increasing order, in eV.
    eri : array_like of float, shape (n, n, n, n)
        Two-electron integrals bracket p q bar r s in Dirac notation, in eV.
    M : array_like of float, shape (n, n, n_modes)
        GW effective integrals M[p, q, nu], in eV.
    omega : array_like of float, shape (n_modes,)
        dRPA excitation energies, in eV, in the same order as the modes of M.
    n_occ : int
        Number of doubly occupied orbitals (the first n_occ entries).
    branch : str
        "hole" for the 2h1p block (rows (i, nu), i occupied) or "particle"
        for the 2p1h block (rows (a, nu), a virtual).

    Returns
    -------
    U2 : np.ndarray, shape (n_rows * n_modes, n)
        The correction block, rows ordered with the orbital index outermost
        (increasing) and the mode index innermost, columns over all n
        orbitals, in eV.
        Raises ValueError for inconsistent shapes, non-finite input, n_occ
        outside 1 .. n - 1, or an unknown branch.
    '''
    n = len(orbital_energies)
    n_rows = int(n_occ) if branch == "hole" else n - int(n_occ)
    U2 = np.zeros((n_rows * len(omega), n))
    return U2

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_block_inputs_sosex(orbital_energies, M, omega, n_occ, branch, eri=None):
    eps = np.asarray(orbital_energies, dtype=float)
    Mm = np.asarray(M, dtype=float)
    Om = np.asarray(omega, dtype=float)
    n = eps.shape[0]
    if eps.ndim != 1 or n < 2:
        raise ValueError("orbital_energies must be a vector with at least two entries")
    if Om.ndim != 1 or Mm.shape != (n, n, Om.shape[0]) or Om.shape[0] < 1:
        raise ValueError("M must have shape (n, n, n_modes) and omega n_modes entries")
    if eri is not None and np.asarray(eri).shape != (n, n, n, n):
        raise ValueError("eri must have shape (n, n, n, n)")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(Mm)) and np.all(np.isfinite(Om))):
        raise ValueError("inputs must be finite")
    if eri is not None and not np.all(np.isfinite(np.asarray(eri, dtype=float))):
        raise ValueError("inputs must be finite")
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    if branch not in ("hole", "particle"):
        raise ValueError("branch must be 'hole' or 'particle'")
    return eps, Mm, Om, int(n_occ)


def _oracle_sosex_coupling_block(orbital_energies, eri, M, omega, n_occ, branch):
    eps, M, Om, n_occ = _check_block_inputs_sosex(orbital_energies, M, omega, n_occ, branch, eri=eri)
    g = np.asarray(eri, dtype=float)
    n = eps.shape[0]
    occ = range(n_occ)
    vir = range(n_occ, n)
    rows = list(occ) if branch == "hole" else list(vir)
    nov = Om.shape[0]
    U = np.zeros((len(rows) * nov, n))
    r = 0
    for p in rows:
        for v in range(nov):
            for q in range(n):
                s = 0.0
                for k in occ:
                    for c in vir:
                        if branch == "hole":
                            s += M[c, k, v] * g[p, c, k, q] / (eps[c] - eps[k] + Om[v])
                            s += M[k, c, v] * g[p, k, c, q] / (eps[c] - eps[k] - Om[v])
                        else:
                            s += M[c, k, v] * g[p, k, c, q] / (eps[c] - eps[k] + Om[v])
                            s += M[k, c, v] * g[p, c, k, q] / (eps[c] - eps[k] - Om[v])
                U[r, q] = s
            r += 1
    return U

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
rng = np.random.default_rng(11)
n, n_occ = 5, 2
eps = np.array([-13.0, -11.2, -8.9, -0.4, 2.1])
Om = np.array([9.35, 10.45, 12.25, 13.75, 15.55, 16.8])
Mraw = rng.normal(size=(n, n, 6))
M = 0.5 * (Mraw + Mraw.transpose(1, 0, 2))
Vs = rng.uniform(3.0, 6.0, size=(n, n)); Vs = 0.5 * (Vs + Vs.T); np.fill_diagonal(Vs, 11.0)
Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
eri = np.einsum("mp,nq,mr,ns,mn->pqrs", Q, Q, Q, Q, Vs, optimize=True)
"""
    return [
        # --- Normal: hole branch (2h1p rows) ---
        {
            "setup": setup,
            "call": "sosex_coupling_block(eps, eri, M, Om, n_occ, 'hole')",
            "gold_call": "_oracle_sosex_coupling_block(eps, eri, M, Om, n_occ, 'hole')",
        },
        # --- Normal: particle branch (2p1h rows) ---
        {
            "setup": setup,
            "call": "sosex_coupling_block(eps, eri, M, Om, n_occ, 'particle')",
            "gold_call": "_oracle_sosex_coupling_block(eps, eri, M, Om, n_occ, 'particle')",
        },
        # --- Boundary: a minimal two-orbital, one-mode problem ---
        {
            "setup": setup + "e2 = np.array([-10.0, 1.5])\nO2 = np.array([12.3])\nM2 = np.array([[[0.8], [-0.35]], [[-0.35], [0.6]]])\ng2 = np.einsum('mp,nq,mr,ns,mn->pqrs', np.eye(2), np.eye(2), np.eye(2), np.eye(2), np.array([[10.0, 5.5], [5.5, 10.0]]))\n",
            "call": "sosex_coupling_block(e2, g2, M2, O2, 1, 'hole')",
            "gold_call": "_oracle_sosex_coupling_block(e2, g2, M2, O2, 1, 'hole')",
        },
        # --- Edge: a mode whose energy lies close to an orbital gap (large but
        #     finite dressed denominator) ---
        {
            "setup": setup + "Om_close = Om.copy(); Om_close[0] = eps[3] - eps[1] + 0.04\n",
            "call": "sosex_coupling_block(eps, eri, M, Om_close, n_occ, 'particle')",
            "gold_call": "_oracle_sosex_coupling_block(eps, eri, M, Om_close, n_occ, 'particle')",
        },
        # --- Edge: vanishing effective integrals give a zero block of the right shape ---
        {
            "setup": setup,
            "call": "sosex_coupling_block(eps, eri, np.zeros_like(M), Om, n_occ, 'hole')",
            "gold_call": "_oracle_sosex_coupling_block(eps, eri, np.zeros_like(M), Om, n_occ, 'hole')",
        },
        # --- Invalid: unknown branch ---
        {
            "setup": setup + """
def run_model():
    try:
        sosex_coupling_block(eps, eri, M, Om, n_occ, 'both')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sosex_coupling_block(eps, eri, M, Om, n_occ, 'both')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: mode count of omega and M disagree ---
        {
            "setup": setup + """
def run_model():
    try:
        sosex_coupling_block(eps, eri, M, Om[:5], n_occ, 'hole')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_sosex_coupling_block(eps, eri, M, Om[:5], n_occ, 'hole')
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
