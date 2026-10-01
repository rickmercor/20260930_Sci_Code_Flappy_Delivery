"""
Build periodic spin-1/2 bond operators and the total-spin Casimir.

The isotropic exchange conserves total spin. A zero-magnetization basis contains one magnetic component of every integer-spin multiplet, so it is sufficient for constructing the reduced thermal traces.

Returns
-------
Return (bonds, total_s2), real arrays of shapes (n,d,d) and (d,d).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spin_operators(n: int) -> "tuple[np.ndarray, np.ndarray]":
    """Return (bonds, total_s2) in the sorted zero-magnetization bit basis.
    
    Parameters
    ----------
    n : int
        Even number of spin-1/2 sites, 4 <= n <= 8. Site i is bit i;
        bit 1 has Sz=+1/2. Basis integers ascend and have n/2 set bits.
    
    Returns
    -------
    bonds : float array (n, d, d), d = binomial(n, n/2)
        bonds[i] = S_i dot S_(i+1 mod n), with hbar=1. A pair contributes
        +1/4 on parallel bits, -1/4 on antiparallel bits, and an off-diagonal
        spin-exchange matrix element +1/2 on antiparallel bits.
    total_s2 : float array (d, d)
        Total-spin Casimir 3*n/4*I + 2*sum_(i<j) S_i dot S_j.
    
    Notes
    -----
    Use each periodic nearest-neighbor bond once. All inputs are valid;
    no input is mutated. Return real numerical arrays, not basis labels.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_spin_operators(n: int) -> "tuple[np.ndarray, np.ndarray]":
    """Return (bonds, total_s2) in the sorted zero-magnetization bit basis.

    Parameters
    ----------
    n : int
        Even number of spin-1/2 sites, 4 <= n <= 8. Site i is bit i;
        bit 1 has Sz=+1/2. Basis integers ascend and have n/2 set bits.

    Returns
    -------
    bonds : float array (n, d, d), d = binomial(n, n/2)
        bonds[i] = S_i dot S_(i+1 mod n), with hbar=1. A pair contributes
        +1/4 on parallel bits, -1/4 on antiparallel bits, and an off-diagonal
        spin-exchange matrix element +1/2 on antiparallel bits.
    total_s2 : float array (d, d)
        Total-spin Casimir 3*n/4*I + 2*sum_(i<j) S_i dot S_j.

    Notes
    -----
    Use each periodic nearest-neighbor bond once. All inputs are valid;
    no input is mutated. Return real numerical arrays, not basis labels.
    """
    basis = [x for x in range(1 << n) if x.bit_count() == n // 2]
    index = {x: i for i, x in enumerate(basis)}
    d = len(basis)

    def _pair(i, j):
        a = np.zeros((d, d))
        for c, x in enumerate(basis):
            opposite = ((x >> i) & 1) != ((x >> j) & 1)
            a[c, c] = -0.25 if opposite else 0.25
            if opposite:
                a[index[x ^ (1 << i) ^ (1 << j)], c] = 0.5
        return a

    bonds = np.stack([_pair(i, (i + 1) % n) for i in range(n)])
    s2 = 0.75 * n * np.eye(d)
    for i in range(n):
        for j in range(i + 1, n):
            s2 += 2.0 * _pair(i, j)
    return bonds, s2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n', 'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in spin_operators(4)])', 'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_spin_operators(4)])', 'tol': 1e-12},
     {'setup': 'import numpy as np\n', 'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in spin_operators(6)])', 'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_spin_operators(6)])', 'tol': 1e-12},
     {'setup': 'import numpy as np\n', 'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in spin_operators(8)])', 'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_spin_operators(8)])', 'tol': 1e-12}]
