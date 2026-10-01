"""
Resolve the zero-magnetization space into total-spin sectors.

Thermal degeneracy and the number of independent multiplets are different quantities. Sector projectors separate the multiplets without introducing eigenvector sign or basis-rotation dependence into the interface.

Returns
-------
Return the real projector array of shape (n/2+1,d,d), ordered by S.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spin_projectors(total_s2: "np.ndarray", n: int) -> "np.ndarray":
    """Return orthogonal projectors onto each total-spin sector in Sz=0.
    
    Parameters
    ----------
    total_s2 : finite real symmetric array (d, d)
        Casimir from spin_operators(n); d=binomial(n,n/2).
    n : int
        Even site count, 4 <= n <= 8.
    
    Returns
    -------
    projectors : float array (n/2+1, d, d)
        Index S=0,...,n/2 projects onto eigenvalue S*(S+1) of total_s2.
        Assign an eigenvalue to that sector when its absolute distance
        from S*(S+1) is below 1e-7. Projectors sum to I. They retain the
        original bit-basis coordinates and do not depend on eigenvector
        signs or rotations within a degenerate eigenspace. Inputs are
        not mutated. The trace of a projector is its multiplicity-space
        dimension; the thermal spin degeneracy is separately 2*S+1.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_spin_projectors(total_s2: "np.ndarray", n: int) -> "np.ndarray":
    """Return orthogonal projectors onto each total-spin sector in Sz=0.

    Parameters
    ----------
    total_s2 : finite real symmetric array (d, d)
        Casimir from spin_operators(n); d=binomial(n,n/2).
    n : int
        Even site count, 4 <= n <= 8.

    Returns
    -------
    projectors : float array (n/2+1, d, d)
        Index S=0,...,n/2 projects onto eigenvalue S*(S+1) of total_s2.
        Assign an eigenvalue to that sector when its absolute distance
        from S*(S+1) is below 1e-7. Projectors sum to I. They retain the
        original bit-basis coordinates and do not depend on eigenvector
        signs or rotations within a degenerate eigenspace. Inputs are
        not mutated. The trace of a projector is its multiplicity-space
        dimension; the thermal spin degeneracy is separately 2*S+1.
    """
    values, vectors = np.linalg.eigh(total_s2)
    result = []
    for spin in range(n // 2 + 1):
        u = vectors[:, np.abs(values - spin * (spin + 1)) < 1e-7]
        result.append(u @ u.T)
    return np.stack(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '_, s2 = _oracle_spin_operators(4)\n',
      'call': 'spin_projectors(s2, 4)',
      'gold_call': '_oracle_spin_projectors(s2, 4)',
      'tol': 1e-10},
     {'setup': '_, s2 = _oracle_spin_operators(6)\n',
      'call': 'spin_projectors(s2, 6)',
      'gold_call': '_oracle_spin_projectors(s2, 6)',
      'tol': 1e-10},
     {'setup': '_, s2 = _oracle_spin_operators(8)\n',
      'call': 'spin_projectors(s2, 8)',
      'gold_call': '_oracle_spin_projectors(s2, 8)',
      'tol': 1e-10}]
