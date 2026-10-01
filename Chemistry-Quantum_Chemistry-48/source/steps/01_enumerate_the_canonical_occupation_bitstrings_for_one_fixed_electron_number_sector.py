"""
Enumerate the canonical occupation bitstrings for one fixed electron-number sector.

Operator signs and determinant amplitudes must use the same occupation ordering. Here the fixed-popcount bitstrings are sorted by increasing integer value.

Returns
-------
One-dimensional int64 array of increasing fixed-popcount bitstrings.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def enumerate_fock_states(n_orbitals: int, n_electrons: int) -> np.ndarray:
    """Enumerate a fixed-particle Fock sector.

    Parameters
    ----------
    n_orbitals : int
        Number of spin orbitals, between 1 and 62.
    n_electrons : int
        Particle count between zero and n_orbitals.

    Returns
    -------
    np.ndarray
        Increasing int64 occupation bitstrings.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_enumerate_fock_states(n_orbitals: int, n_electrons: int) -> np.ndarray:
    """Reference fixed-particle bitstring enumeration."""
    from itertools import combinations
    import numpy as np

    if not isinstance(n_orbitals, (int, np.integer)):
        raise ValueError("n_orbitals must be an integer")
    if not isinstance(n_electrons, (int, np.integer)):
        raise ValueError("n_electrons must be an integer")

    m = int(n_orbitals)
    n = int(n_electrons)

    if m < 1 or m > 62 or n < 0 or n > m:
        raise ValueError("invalid orbital or electron count")

    states = [
        sum(1 << p for p in occupied)
        for occupied in combinations(range(m), n)
    ]
    return np.asarray(sorted(states), dtype=np.int64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "enumerate_fock_states(4, 2)",
            "gold_call": "_oracle_enumerate_fock_states(4, 2)",
        },
        {
            "setup": "",
            "call": "enumerate_fock_states(5, 0)",
            "gold_call": "_oracle_enumerate_fock_states(5, 0)",
        },
        {
            "setup": "",
            "call": "enumerate_fock_states(5, 5)",
            "gold_call": "_oracle_enumerate_fock_states(5, 5)",
        },
    ]
