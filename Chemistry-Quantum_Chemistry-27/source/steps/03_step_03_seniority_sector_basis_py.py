"""
Enumerate the product states of a fixed-local-seniority sector: pair configurations on the pairing levels times spin configurations on the singly occupied levels.

A seniority eigenstate configuration interaction (SECI) wave function fixes the seniority of every level. The pairing levels have seniority zero (each is empty or doubly occupied) and the spin levels have seniority one (each holds exactly one electron, with spin up or down). The wave function is expanded as $|\Psi\rangle=\sum_{\Gamma\Lambda}C_{\Gamma\Lambda}|\Xi_\Gamma\rangle\otimes|\Phi_\Lambda\rangle$, where $|\Phi_\Lambda\rangle$ runs over ways of placing the electron pairs on the pairing levels and $|\Xi_\Gamma\rangle$ over ways of assigning up and down spins to the spin levels at fixed total $S^z$. The sector dimension is therefore a product of two binomial coefficients

Returns
-------
np.ndarray of int, shape (D, P + k): 2/0 on pairing levels, +1/−1 on spin levels
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def seniority_sector_basis(n_pairing_levels: int, n_pairs: int, n_spin_levels: int, n_up: int) -> "np.ndarray":
    '''Return the ordered product-state basis of a fixed-local-seniority sector.
 
    Levels 0..P-1 are pairing levels (P = n_pairing_levels) and levels P..P+k-1 are spin
    levels (k = n_spin_levels). Each basis state is a row of length P + k with entries
        2  : doubly occupied pairing level,   0 : empty pairing level,
        1  : spin level holding an up electron,   -1 : spin level holding a down electron.
    Exactly n_pairs pairing levels are doubly occupied and exactly n_up spin levels hold
    up electrons. Rows are ordered with the pair configuration as the outer index and
    the spin configuration as the inner index; each is enumerated in the order of
    itertools.combinations over the occupied pairing levels (0..P-1) and over the up-spin
    levels (numbered 0..k-1 within the spin block), respectively.
 
    Parameters
    ----------
    n_pairing_levels : int
        P >= 0.
    n_pairs : int
        Number of electron pairs, 0 <= n_pairs <= P.
    n_spin_levels : int
        k >= 0.
    n_up : int
        Number of up electrons on the spin levels, 0 <= n_up <= k.
 
    Returns
    -------
    basis : np.ndarray
        Integer array of shape (C(P, n_pairs) * C(k, n_up), P + k).
 
    Raises
    ------
    ValueError
        If any count is negative, n_pairs > P, n_up > k, or P + k = 0.
    '''
    return basis

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import numpy as np
 
def _oracle_seniority_sector_basis(n_pairing_levels: int, n_pairs: int, n_spin_levels: int, n_up: int) -> "np.ndarray":
    p, npair, k, nup = int(n_pairing_levels), int(n_pairs), int(n_spin_levels), int(n_up)
    if min(p, npair, k, nup) < 0 or npair > p or nup > k or p + k == 0:
        raise ValueError("invalid sector specification")
    rows = []
    for occupied in itertools.combinations(range(p), npair):
        for ups in itertools.combinations(range(k), nup):
            row = np.zeros(p + k, dtype=int)
            row[list(occupied)] = 2
            spins = -np.ones(k, dtype=int)
            spins[list(ups)] = 1
            row[p:] = spins
            rows.append(row)
    return np.array(rows, dtype=int).reshape(-1, p + k)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: benchmark sector, 4 pairing levels with one pair and 4 spin levels at Sz = 0
        {"setup": "", "call": "seniority_sector_basis(4, 1, 4, 2)", "gold_call": "_oracle_seniority_sector_basis(4, 1, 4, 2)"},
        # Normal: maximal seniority, six spin levels and no pairing levels
        {"setup": "", "call": "seniority_sector_basis(0, 0, 6, 3)", "gold_call": "_oracle_seniority_sector_basis(0, 0, 6, 3)"},
        # Boundary: seniority zero (pairing levels only)
        {"setup": "", "call": "seniority_sector_basis(6, 3, 0, 0)", "gold_call": "_oracle_seniority_sector_basis(6, 3, 0, 0)"},
        # Edge: nonzero Sz (three up, one down) and all pairing levels filled
        {"setup": "", "call": "seniority_sector_basis(2, 2, 4, 3)", "gold_call": "_oracle_seniority_sector_basis(2, 2, 4, 3)"},
        # Invalid: more pairs than pairing levels
        {
            "setup": """def run_model():
    try:
        seniority_sector_basis(2, 3, 2, 1)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_seniority_sector_basis(2, 3, 2, 1)
        return 0
    except ValueError:
        return 1
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
