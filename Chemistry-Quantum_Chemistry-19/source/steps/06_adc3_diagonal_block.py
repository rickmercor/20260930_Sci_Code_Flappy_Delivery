"""
The block is Hermitian and, for a real orbital basis, symmetric. Each of its elements is a sum over the orbitals of the opposite type (virtual c for the hole block, occupied k for the particle block) of a product of two GW effective integrals, one connecting the row orbital to c in the column mode and one connecting the column orbital to c in the row mode, each product divided by a dressed denominator that combines an orbital gap with one excitation energy, symmetrized over the two configurations with a factor one half. The explicit expressions are Eqs. (17a) and (17b) of the source paper; this step implements them for one branch.

As long as only the coupling blocks are corrected, the ADC construction adds a finite number of diagrams to the self-energy. The character of the method changes when the configuration space itself acquires a coupling block C: the resolvent (omega - K - C)^(-1) then resums an infinite class of diagrams, and the quasiparticle and satellite energies follow from a genuinely non-perturbative eigenvalue problem. For the screened expansion of this task, the first such block couples two 2h1p configurations (i, nu) and (j, mu) (and, mirrored, two 2p1h configurations) and originates from the term of the G3W2 self-energy that carries two dressed resolvents in a row. The scheme that contains it together with the GW and screened-exchange couplings is called ADC(3)-G3W2 in the source paper.

Returns
-------
np.ndarray of shape (n_rows * n_modes, n_rows * n_modes): the symmetric configuration-coupling block of the requested branch, in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adc3_diagonal_block(orbital_energies, M, omega, n_occ, branch):
    '''Configuration-coupling block C among the 2h1p (or 2p1h) configurations.

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
        "hole" for the block among 2h1p configurations (i, nu) or
        "particle" for the block among 2p1h configurations (a, nu).

    Returns
    -------
    C : np.ndarray, shape (n_rows * n_modes, n_rows * n_modes)
        The coupling block, configurations ordered with the orbital index
        outermost (increasing) and the mode index innermost, in eV.
        Raises ValueError for inconsistent shapes, non-finite input, n_occ
        outside 1 .. n - 1, or an unknown branch.
    '''
    n = len(orbital_energies)
    n_rows = int(n_occ) if branch == "hole" else n - int(n_occ)
    C = np.zeros((n_rows * len(omega), n_rows * len(omega)))
    return C

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


def _oracle_adc3_diagonal_block(orbital_energies, M, omega, n_occ, branch):
    eps, M, Om, n_occ = _check_block_inputs(orbital_energies, M, omega, n_occ, branch)
    n = eps.shape[0]
    occ = list(range(n_occ))
    vir = list(range(n_occ, n))
    rows = occ if branch == "hole" else vir
    other = vir if branch == "hole" else occ
    nov = Om.shape[0]
    conf = [(p, v) for p in rows for v in range(nov)]
    C = np.zeros((len(conf), len(conf)))
    for r, (i, v) in enumerate(conf):
        for s_, (j, mu) in enumerate(conf):
            val = 0.0
            for c in other:
                if branch == "hole":
                    prod = M[i, c, mu] * M[j, c, v]
                    val += 0.5 * prod / (eps[i] - eps[c] + Om[mu])
                    val += 0.5 * prod / (eps[j] - eps[c] + Om[v])
                else:
                    prod = M[c, i, mu] * M[c, j, v]
                    val += 0.5 * prod / (eps[i] - eps[c] - Om[mu])
                    val += 0.5 * prod / (eps[j] - eps[c] - Om[v])
            C[r, s_] = val
    return C

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
rng = np.random.default_rng(23)
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
            "call": "adc3_diagonal_block(eps, M, Om, n_occ, 'hole')",
            "gold_call": "_oracle_adc3_diagonal_block(eps, M, Om, n_occ, 'hole')",
        },
        # --- Normal: particle branch ---
        {
            "setup": setup,
            "call": "adc3_diagonal_block(eps, M, Om, n_occ, 'particle')",
            "gold_call": "_oracle_adc3_diagonal_block(eps, M, Om, n_occ, 'particle')",
        },
        # --- Boundary: three occupied, two virtual orbitals (unequal branch sizes) ---
        {
            "setup": setup + "Mb = M.copy()\n",
            "call": "adc3_diagonal_block(eps, Mb, Om, 3, 'particle')",
            "gold_call": "_oracle_adc3_diagonal_block(eps, Mb, Om, 3, 'particle')",
        },
        # --- Edge: a single mode (the block is n_rows by n_rows) ---
        {
            "setup": setup,
            "call": "adc3_diagonal_block(eps, M[:, :, :1], Om[:1], n_occ, 'hole')",
            "gold_call": "_oracle_adc3_diagonal_block(eps, M[:, :, :1], Om[:1], n_occ, 'hole')",
        },
        # --- Edge: scaled-down integrals (the block is quadratic in M) ---
        {
            "setup": setup,
            "call": "adc3_diagonal_block(eps, 1e-3 * M, Om, n_occ, 'particle')",
            "gold_call": "_oracle_adc3_diagonal_block(eps, 1e-3 * M, Om, n_occ, 'particle')",
        },
        # --- Invalid: unknown branch ---
        {
            "setup": setup + """
def run_model():
    try:
        adc3_diagonal_block(eps, M, Om, n_occ, 'h')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_adc3_diagonal_block(eps, M, Om, n_occ, 'h')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: M of the wrong shape ---
        {
            "setup": setup + """
def run_model():
    try:
        adc3_diagonal_block(eps, M[:, :4, :], Om, n_occ, 'hole')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_adc3_diagonal_block(eps, M[:, :4, :], Om, n_occ, 'hole')
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
