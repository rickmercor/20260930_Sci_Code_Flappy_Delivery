"""
Completing the full G3W2 self-energy within the ADC form requires the remaining pieces of its hole and particle branches, those that are cubic in the GW effective integrals. Part of them enters as a further correction to the coupling between the one-particle space and the 2h1p (or 2p1h) configurations. For the hole branch this correction to the coupling of configuration (i, nu) with orbital q is a sum over an extra dRPA mode mu and over pairs of orbitals of four distinct terms, each a product of three effective integrals (two of them carrying the extra mode mu, one carrying the configuration mode nu) divided by a product of two dressed denominators; three of the terms run over an occupied orbital k and a virtual orbital c, the fourth over two virtual orbitals c and d, and the terms carry different signs, a factor one half on the first, and denominators in which Omega_mu enters with either sign and in one case together with Omega_nu. The particle branch is the particle-hole mirror image with two occupied orbitals k and l in the last term. The explicit expressions are Eqs. (18a) and (18b) of the source paper; this step implements them for one branch.

Because of its scaling (a double mode sum with three orbital indices) this is the most expensive block of the scheme, but for the small chains of this task a direct loop over all indices is perfectly adequate. Together with the three-body configurations of the next step, this block is what turns the ADC(3)-G3W2 scheme into the full ADC-G3W2 scheme.

Returns
-------
np.ndarray of shape (n_rows * n_modes, n): the cubic correction to the coupling block of the requested branch, in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def third_order_coupling_block(orbital_energies, M, omega, n_occ, branch):
    '''Cubic (third) contribution to the ADC coupling block of one branch.

    Parameters
    ----------
    orbital_energies : array_like of float, shape (n,)
        Canonical orbital energies in increasing order, in eV.
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
    U3 : np.ndarray, shape (n_rows * n_modes, n)
        The cubic correction to the coupling block, rows ordered with the
        orbital index outermost (increasing) and the mode index innermost,
        columns over all n orbitals, in eV.
        Raises ValueError for inconsistent shapes, non-finite input, n_occ
        outside 1 .. n - 1, or an unknown branch.
    '''
    n = len(orbital_energies)
    n_rows = int(n_occ) if branch == "hole" else n - int(n_occ)
    U3 = np.zeros((n_rows * len(omega), n))
    return U3

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _check_block_inputs(orbital_energies, M, omega, n_occ, branch):
    eps = np.asarray(orbital_energies, dtype=float)
    Mm = np.asarray(M, dtype=float)
    Om = np.asarray(omega, dtype=float)
    n = eps.shape[0]
    if eps.ndim != 1 or n < 2:
        raise ValueError("orbital_energies must be a vector with at least two entries")
    if Om.ndim != 1 or Mm.shape != (n, n, Om.shape[0]) or Om.shape[0] < 1:
        raise ValueError("M must have shape (n, n, n_modes) and omega n_modes entries")
    if not (np.all(np.isfinite(eps)) and np.all(np.isfinite(Mm)) and np.all(np.isfinite(Om))):
        raise ValueError("inputs must be finite")
    if int(n_occ) != n_occ or n_occ < 1 or n_occ > n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    if branch not in ("hole", "particle"):
        raise ValueError("branch must be 'hole' or 'particle'")
    return eps, Mm, Om, int(n_occ)


def _oracle_third_order_coupling_block(orbital_energies, M, omega, n_occ, branch):
    eps, M, Om, n_occ = _check_block_inputs(orbital_energies, M, omega, n_occ, branch)
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
                if branch == "hole":
                    i = p
                    for k in occ:
                        for c in vir:
                            for mu in range(nov):
                                s += 0.5 * M[i, c, mu] * M[k, c, v] * M[q, k, mu] / ((eps[c] - eps[k] - Om[v]) * (eps[c] - eps[i] - Om[mu]))
                                s -= M[c, i, mu] * M[k, c, v] * M[k, q, mu] / ((eps[c] - eps[k] - Om[v]) * (eps[c] - eps[i] + Om[mu]))
                                s -= M[k, i, mu] * M[c, k, v] * M[c, q, mu] / ((eps[c] - eps[i] + Om[v] + Om[mu]) * (eps[c] - eps[k] + Om[v]))
                    for c in vir:
                        for d in vir:
                            for mu in range(nov):
                                s += M[d, i, mu] * M[c, d, v] * M[c, q, mu] / ((eps[c] - eps[i] + Om[v] + Om[mu]) * (eps[d] - eps[i] + Om[mu]))
                else:
                    a = p
                    for k in occ:
                        for c in vir:
                            for mu in range(nov):
                                s += 0.5 * M[k, a, mu] * M[k, c, v] * M[c, q, mu] / ((eps[c] - eps[k] - Om[v]) * (eps[a] - eps[k] - Om[mu]))
                                s -= M[a, k, mu] * M[k, c, v] * M[q, c, mu] / ((eps[c] - eps[k] - Om[v]) * (eps[a] - eps[k] + Om[mu]))
                                s -= M[a, c, mu] * M[c, k, v] * M[q, k, mu] / ((eps[a] - eps[k] + Om[v] + Om[mu]) * (eps[c] - eps[k] + Om[v]))
                    for k in occ:
                        for l in occ:
                            for mu in range(nov):
                                s += M[a, l, mu] * M[l, k, v] * M[q, k, mu] / ((eps[a] - eps[k] + Om[v] + Om[mu]) * (eps[a] - eps[l] + Om[mu]))
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
rng = np.random.default_rng(37)
n, n_occ = 5, 2
eps = np.array([-13.0, -11.2, -8.9, -0.4, 2.1])
Om = np.array([9.35, 10.45, 12.25, 13.75, 15.55, 16.8])
Mraw = rng.normal(size=(n, n, 6))
M = 0.5 * (Mraw + Mraw.transpose(1, 0, 2))
"""
    return [
        # --- Normal: hole branch ---
        {
            "setup": setup,
            "call": "third_order_coupling_block(eps, M, Om, n_occ, 'hole')",
            "gold_call": "_oracle_third_order_coupling_block(eps, M, Om, n_occ, 'hole')",
        },
        # --- Normal: particle branch ---
        {
            "setup": setup,
            "call": "third_order_coupling_block(eps, M, Om, n_occ, 'particle')",
            "gold_call": "_oracle_third_order_coupling_block(eps, M, Om, n_occ, 'particle')",
        },
        # --- Boundary: three occupied orbitals, two virtuals (the two-virtual term
        #     of the hole branch has only four (c, d) pairs) ---
        {
            "setup": setup,
            "call": "third_order_coupling_block(eps, M, Om, 3, 'hole')",
            "gold_call": "_oracle_third_order_coupling_block(eps, M, Om, 3, 'hole')",
        },
        # --- Edge: a single mode, so the extra mode sum has one term ---
        {
            "setup": setup,
            "call": "third_order_coupling_block(eps, M[:, :, 2:3], Om[2:3], n_occ, 'particle')",
            "gold_call": "_oracle_third_order_coupling_block(eps, M[:, :, 2:3], Om[2:3], n_occ, 'particle')",
        },
        # --- Edge: an effective-integral tensor that vanishes on the occupied-occupied
        #     block removes the two-occupied term of the particle branch entirely ---
        {
            "setup": setup + "Moo = M.copy(); Moo[:2, :2, :] = 0.0\n",
            "call": "third_order_coupling_block(eps, Moo, Om, n_occ, 'particle')",
            "gold_call": "_oracle_third_order_coupling_block(eps, Moo, Om, n_occ, 'particle')",
        },
        # --- Invalid: unknown branch ---
        {
            "setup": setup + """
def run_model():
    try:
        third_order_coupling_block(eps, M, Om, n_occ, 'electron')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_third_order_coupling_block(eps, M, Om, n_occ, 'electron')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-finite input ---
        {
            "setup": setup + """
Mbad = M.copy(); Mbad[0, 1, 0] = np.nan
def run_model():
    try:
        third_order_coupling_block(eps, Mbad, Om, n_occ, 'hole')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_third_order_coupling_block(eps, Mbad, Om, n_occ, 'hole')
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
