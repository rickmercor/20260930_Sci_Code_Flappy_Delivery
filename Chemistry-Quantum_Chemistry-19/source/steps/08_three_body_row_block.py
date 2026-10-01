"""
Three new blocks come with these configurations: the direct coupling between the one-particle space and a three-body configuration, which is a single sum over the orbitals of the opposite type of a product of two effective integrals over a dressed denominator carrying the second mode (with a definite overall sign that differs between the hole and the particle branches); the coupling between a three-body configuration and a two-body configuration, which is a single effective integral of the second mode and is diagonal in the first mode; and the diagonal three-body energies. The explicit expressions are Eqs. (19), (20) and (21) of the source paper. This step returns, for one branch, the whole row of the effective Hamiltonian that belongs to the three-body configurations, as the horizontal concatenation [coupling to orbitals | coupling to two-body configurations | diagonal energies], with both configuration indices ordered orbital outermost, first mode, then second mode innermost.

The last term of the G3W2 self-energy cannot be accommodated within the 2h1p and 2p1h configuration spaces at all: its resolvent carries two dRPA excitation energies at once. The ADC construction therefore enlarges the configuration space by three-hole-two-particle (3h2p) configurations, labelled by an occupied orbital and an ordered pair of dRPA modes (i, nu, mu), and by the mirrored three-particle-two-hole (3p2h) configurations (a, nu, mu). Their zeroth-order energies are the doubly dressed values eps_i - Omega_nu - Omega_mu and eps_a + Omega_nu + Omega_mu; both orderings of the two modes are kept as distinct configurations, exactly as they arise from the double mode sum of the self-energy.

Returns
-------
np.ndarray of shape (n_rows * n_modes**2, n + n_rows * n_modes + n_rows * n_modes**2): the three-body row [coupling to orbitals | coupling to two-body configurations | diagonal energies] of the requested branch, in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def three_body_row_block(orbital_energies, M, omega, n_occ, branch):
    '''Row block of the 3h2p (or 3p2h) configurations of the ADC Hamiltonian.

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
        "hole" for the 3h2p configurations (i, nu, mu), "particle" for the
        3p2h configurations (a, nu, mu).

    Returns
    -------
    row : np.ndarray, shape (n_rows * n_modes**2, n + n_rows * n_modes + n_rows * n_modes**2)
        Horizontal concatenation of the coupling block to the n orbitals,
        the coupling block to the two-body configurations (p, lambda) of the
        same branch, and the diagonal block of three-body energies, in eV.
        Three-body configurations are ordered orbital outermost, then the
        first mode, then the second mode; two-body configurations orbital
        outermost, mode innermost.
        Raises ValueError for inconsistent shapes, non-finite input, n_occ
        outside 1 .. n - 1, or an unknown branch.
    '''
    n = len(orbital_energies)
    n_rows = int(n_occ) if branch == "hole" else n - int(n_occ)
    nm = len(omega)
    row = np.zeros((n_rows * nm * nm, n + n_rows * nm + n_rows * nm * nm))
    return row

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


def _oracle_three_body_row_block(orbital_energies, M, omega, n_occ, branch):
    eps, M, Om, n_occ = _check_block_inputs(orbital_energies, M, omega, n_occ, branch)
    n = eps.shape[0]
    occ = list(range(n_occ))
    vir = list(range(n_occ, n))
    rows = occ if branch == "hole" else vir
    other = vir if branch == "hole" else occ
    nov = Om.shape[0]
    conf3 = [(p, v, mu) for p in rows for v in range(nov) for mu in range(nov)]
    conf2 = [(p, v) for p in rows for v in range(nov)]
    U2 = np.zeros((len(conf3), n))
    C21 = np.zeros((len(conf3), len(conf2)))
    K2 = np.zeros((len(conf3), len(conf3)))
    sgn = -1.0 if branch == "hole" else 1.0
    for r, (p, v, mu) in enumerate(conf3):
        K2[r, r] = eps[p] + sgn * (Om[v] + Om[mu])
        for q in range(n):
            val = 0.0
            for c in other:
                if branch == "hole":
                    val -= M[c, p, mu] * M[q, c, v] / (eps[c] - eps[p] + Om[mu])
                else:
                    val += M[p, c, mu] * M[c, q, v] / (eps[p] - eps[c] + Om[mu])
            U2[r, q] = val
        for s_, (j, lam) in enumerate(conf2):
            if lam == v:
                C21[r, s_] = M[j, p, mu] if branch == "hole" else M[p, j, mu]
    return np.hstack([U2, C21, K2])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    setup = """import numpy as np
rng = np.random.default_rng(41)
n, n_occ = 4, 2
eps = np.array([-12.5, -9.8, -0.7, 2.4])
Om = np.array([9.6, 11.3, 13.9, 16.2])
Mraw = rng.normal(size=(n, n, 4))
M = 0.5 * (Mraw + Mraw.transpose(1, 0, 2))
"""
    return [
        # --- Normal: hole branch (3h2p rows) ---
        {
            "setup": setup,
            "call": "three_body_row_block(eps, M, Om, n_occ, 'hole')",
            "gold_call": "_oracle_three_body_row_block(eps, M, Om, n_occ, 'hole')",
        },
        # --- Normal: particle branch (3p2h rows) ---
        {
            "setup": setup,
            "call": "three_body_row_block(eps, M, Om, n_occ, 'particle')",
            "gold_call": "_oracle_three_body_row_block(eps, M, Om, n_occ, 'particle')",
        },
        # --- Boundary: one occupied orbital, three virtuals ---
        {
            "setup": setup,
            "call": "three_body_row_block(eps, M, Om, 1, 'hole')",
            "gold_call": "_oracle_three_body_row_block(eps, M, Om, 1, 'hole')",
        },
        # --- Edge: two modes only (four ordered mode pairs per orbital) ---
        {
            "setup": setup,
            "call": "three_body_row_block(eps, M[:, :, :2], Om[:2], n_occ, 'particle')",
            "gold_call": "_oracle_three_body_row_block(eps, M[:, :, :2], Om[:2], n_occ, 'particle')",
        },
        # --- Edge: vanishing effective integrals leave only the diagonal energies ---
        {
            "setup": setup,
            "call": "three_body_row_block(eps, np.zeros_like(M), Om, n_occ, 'hole')",
            "gold_call": "_oracle_three_body_row_block(eps, np.zeros_like(M), Om, n_occ, 'hole')",
        },
        # --- Invalid: unknown branch ---
        {
            "setup": setup + """
def run_model():
    try:
        three_body_row_block(eps, M, Om, n_occ, '3h2p')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_three_body_row_block(eps, M, Om, n_occ, '3h2p')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: n_occ equal to n ---
        {
            "setup": setup + """
def run_model():
    try:
        three_body_row_block(eps, M, Om, 4, 'hole')
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_three_body_row_block(eps, M, Om, 4, 'hole')
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
