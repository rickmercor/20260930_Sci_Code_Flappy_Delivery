"""
Expand an ordered set of orbital rows into amplitudes over a supplied occupation basis.

The coefficient of a sorted occupation in a Slater determinant is the determinant of the orbital-coefficient submatrix on those occupied spin orbitals.

Returns
-------
Floating array with one many-electron amplitude per supplied basis state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def expand_slater_determinant(
    orbital_rows: np.ndarray,
    basis_states: np.ndarray,
) -> np.ndarray:
    """Expand an ordered orbital product in an occupation basis.

    Parameters
    ----------
    orbital_rows : np.ndarray
        Real orbital coefficients with shape (n_electrons, n_orbitals).
    basis_states : np.ndarray
        Distinct bitstrings in the matching particle sector.

    Returns
    -------
    np.ndarray
        Determinant amplitudes in basis_states order.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_expand_slater_determinant(orbital_rows, basis_states):
    """Reference minor-determinant expansion."""
    import numpy as np

    orbitals = np.asarray(orbital_rows, dtype=float)
    basis = np.asarray(basis_states)

    if orbitals.ndim != 2 or orbitals.shape[1] < 1:
        raise ValueError("orbital_rows must be two-dimensional")
    if basis.ndim != 1 or basis.size < 1:
        raise ValueError(
            "basis_states must be nonempty and one-dimensional"
        )
    if not np.all(np.isfinite(orbitals)) or not np.all(np.isfinite(basis)):
        raise ValueError("inputs must be finite")
    if not np.all(basis == np.floor(basis)):
        raise ValueError("basis states must be integers")

    n, m = orbitals.shape
    basis = basis.astype(np.int64)

    if np.any(basis < 0) or len(set(map(int, basis))) != basis.size:
        raise ValueError("basis states must be distinct and nonnegative")

    amplitudes = []
    for raw in basis:
        state = int(raw)
        if state >> m or state.bit_count() != n:
            raise ValueError("basis state belongs to the wrong sector")
        occupied = [p for p in range(m) if state & (1 << p)]
        amplitudes.append(np.linalg.det(orbitals[:, occupied]))

    return np.asarray(amplitudes, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = """import numpy as np
basis_states = np.array([3,5,6,9,10,12])
orbital_rows = np.array([[1.,0.,0.,0.],[0.,0.,1.,0.]])"""

    return [
        {
            "setup": common,
            "call": "expand_slater_determinant(orbital_rows, basis_states)",
            "gold_call": "_oracle_expand_slater_determinant(orbital_rows, basis_states)",
        },
        {
            "setup": common + "\norbital_rows = orbital_rows[::-1]",
            "call": "expand_slater_determinant(orbital_rows, basis_states)",
            "gold_call": "_oracle_expand_slater_determinant(orbital_rows, basis_states)",
        },
        {
            "setup": (
                "import numpy as np\n"
                "orbital_rows=np.empty((0,3)); basis_states=np.array([0])"
            ),
            "call": "expand_slater_determinant(orbital_rows, basis_states)",
            "gold_call": "_oracle_expand_slater_determinant(orbital_rows, basis_states)",
        },
    ]
