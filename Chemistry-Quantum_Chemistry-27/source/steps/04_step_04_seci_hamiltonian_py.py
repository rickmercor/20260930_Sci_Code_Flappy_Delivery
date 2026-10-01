"""
Build the matrix of the local-seniority-conserving (so(4)) Hamiltonian in a fixed-local-seniority product-state basis.

Because every term of $H_{\delta\Omega=0}$ conserves the seniority of each level, the Hamiltonian maps a fixed-local-seniority sector into itself. On the pairing levels only the number operators $N_\mu$ and the pair-transfer terms $L_{\mu\nu}P^\dagger_\mu P_\nu$ act; on the spin levels only $N_\alpha=1$, $S^z_\alpha$ and the spin-exchange terms $-K^{\mathrm{ud}}_{\alpha\beta}S^+_\alpha S^-_\beta$ act; and the term $\sum X_{\mu\alpha}N_\mu S^z_\alpha$ couples the charge of pairing levels to the spin of spin levels. The operators $P^\dagger_p$, $P_p$, $S^\pm_p$, $N_p$ and $S^z_p$ are bilinear in the fermion operators of a single level, so operators on different levels commute and the sector can be represented by ordinary product states of pair and spin configurations without fermionic sign factors. The lowest eigenvalue of this matrix is the SECI energy for the given orbitals, and equals the lowest eigenvalue of the full electronic Hamiltonian projected onto the same sector.

Returns
-------
np.ndarray of shape (D, D): sector Hamiltonian matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def seci_hamiltonian(coefficients: "np.ndarray", sector_basis: "np.ndarray") -> "np.ndarray":
    '''Return the SECI Hamiltonian matrix in the given product-state basis.
 
    coefficients = [diag(eps), diag(B), L, W, Bz, Kud, X] (shape (7, M, M)); L and Kud
    are symmetric. Each row of sector_basis encodes one product state with entries 2
    (doubly occupied pairing level), 0 (empty pairing level), 1 (up spin level) or -1
    (down spin level). For a state, N_p = 2, 0, 1, 1 and S^z_p = 0, 0, 1/2, -1/2 for the
    four entry values. With E0 = 0 the Hamiltonian is
 
        H = sum_p eps_p N_p + sum_p B_p S^z_p + sum_{p,q} L_pq P+_p P_q
            + (1/4) sum_{p!=q} W_pq N_p N_q + sum_{p!=q} Bz_pq S^z_p S^z_q
            + sum_{p!=q} X_pq N_p S^z_q - sum_{p!=q} Kud_pq S+_p S-_q .
 
    Diagonal elements collect all N and S^z terms plus L_pp for every doubly occupied
    level. Off-diagonal elements use matrix elements +1 for the product-state
    transitions: P+_p P_q (p != q) maps a state with level q doubly occupied and level p
    empty to the state with the pair moved to p, contributing +L_pq; S+_p S-_q (p != q)
    maps a state with p down and q up to the state with p up and q down, contributing
    -Kud_pq. Element H[i, j] is the coefficient of basis state i in H applied to basis
    state j.
 
    Parameters
    ----------
    coefficients : np.ndarray
        Array of shape (7, M, M).
    sector_basis : np.ndarray
        Integer array of shape (D, M) with entries in {2, 0, 1, -1}; every state reached
        by the transitions above must be in the basis.
 
    Returns
    -------
    hamiltonian : np.ndarray
        Real symmetric D x D matrix.
 
    Raises
    ------
    ValueError
        If the shapes are inconsistent, a basis entry is not in {2, 0, 1, -1}, or a
        transition leads to a state that is not in the basis.
    '''
    return hamiltonian

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_seci_hamiltonian(coefficients: "np.ndarray", sector_basis: "np.ndarray") -> "np.ndarray":
    coef = np.asarray(coefficients)
    states = np.asarray(sector_basis)
    if states.ndim != 2 or coef.shape != (7, states.shape[1], states.shape[1]):
        raise ValueError("coefficients must have shape (7, M, M) matching the basis width")
    if not np.all(np.isin(states, [2, 0, 1, -1])):
        raise ValueError("basis entries must be 2, 0, 1 or -1")
    eps, b_single = np.diag(coef[0]), np.diag(coef[1])
    pair, w, b_pair, k_ud, x = coef[2], coef[3], coef[4], coef[5], coef[6]
    charge = np.where(states == 2, 2.0, np.where(states == 0, 0.0, 1.0))
    spin = np.where(states == 1, 0.5, np.where(states == -1, -0.5, 0.0))
    diagonal = (charge @ eps + spin @ b_single
                + 0.25 * np.einsum("ip,pq,iq->i", charge, w, charge)
                + np.einsum("ip,pq,iq->i", spin, b_pair, spin)
                + np.einsum("ip,pq,iq->i", charge, x, spin)
                + (states == 2) @ np.diag(pair))
    ham = np.diag(diagonal).astype(np.result_type(coef, float))
    index = {tuple(row): i for i, row in enumerate(states)}
    m = states.shape[1]
    for j, row in enumerate(states):
        for p in range(m):
            for q in range(m):
                if p == q:
                    continue
                if row[p] == 0 and row[q] == 2:
                    target = row.copy()
                    target[p], target[q] = 2, 0
                    if tuple(target) not in index:
                        raise ValueError("pair transfer leaves the basis")
                    ham[index[tuple(target)], j] += pair[p, q]
                if row[p] == -1 and row[q] == 1:
                    target = row.copy()
                    target[p], target[q] = 1, -1
                    if tuple(target) not in index:
                        raise ValueError("spin exchange leaves the basis")
                    ham[index[tuple(target)], j] -= k_ud[p, q]
    return ham

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    common = """import numpy as np
import itertools
def sector(p, npair, k, nup):
    rows = []
    for occ in itertools.combinations(range(p), npair):
        for ups in itertools.combinations(range(k), nup):
            r = np.zeros(p + k, dtype=int); r[list(occ)] = 2
            s = -np.ones(k, dtype=int); s[list(ups)] = 1; r[p:] = s
            rows.append(r)
    return np.array(rows, dtype=int).reshape(-1, p + k)
def coefs(seed, m):
    rng = np.random.default_rng(seed)
    sym = lambda a: 0.5 * (a + a.T)
    off = 1.0 - np.eye(m)
    return np.stack([np.diag(rng.normal(size=m)), np.diag(rng.normal(size=m)), sym(rng.normal(size=(m, m))),
                     sym(rng.normal(size=(m, m))) * off, sym(rng.normal(size=(m, m))) * off,
                     sym(rng.normal(size=(m, m))) * off, rng.normal(size=(m, m)) * off])
"""
    return [
        # Normal: benchmark-sized sector (4 pairing levels, 1 pair, 4 spin levels, Sz = 0)
        {"setup": common + "c = coefs(0, 8)\nb = sector(4, 1, 4, 2)\n",
         "call": "seci_hamiltonian(c.copy(), b.copy())",
         "gold_call": "_oracle_seci_hamiltonian(c.copy(), b.copy())"},
        # Normal: maximal seniority (spin block only, spin-exchange and Bz terms)
        {"setup": common + "c = coefs(1, 6)\nb = sector(0, 0, 6, 3)\n",
         "call": "seci_hamiltonian(c.copy(), b.copy())",
         "gold_call": "_oracle_seci_hamiltonian(c.copy(), b.copy())"},
        # Boundary: seniority zero (pair transfer and on-level L_pp terms only)
        {"setup": common + "c = coefs(2, 5)\nb = sector(5, 2, 0, 0)\n",
         "call": "seci_hamiltonian(c.copy(), b.copy())",
         "gold_call": "_oracle_seci_hamiltonian(c.copy(), b.copy())"},
        # Edge: nonzero Sz with filled pairing block (only diagonal charge-spin coupling)
        {"setup": common + "c = coefs(3, 6)\nb = sector(2, 2, 4, 3)\n",
         "call": "seci_hamiltonian(c.copy(), b.copy())",
         "gold_call": "_oracle_seci_hamiltonian(c.copy(), b.copy())"},
        # Invalid: truncated basis, a spin exchange leaves it
        {"setup": common + """c = coefs(4, 4)
b = sector(0, 0, 4, 2)[:3]
def run_model():
    try:
        seci_hamiltonian(c.copy(), b.copy())
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_seci_hamiltonian(c.copy(), b.copy())
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
    ]
