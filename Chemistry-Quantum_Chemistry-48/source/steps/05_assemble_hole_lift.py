"""
Lift every hole determinant through all creation operators into the electron sector.

After row zero is removed, column (I,mu) is c_mu^dagger acting on hole determinant I. Reoccupation gives zero; every allowed creation carries its fermionic phase.

Returns
-------
Floating matrix with shape (n_electron_states, n_det*n_orbitals).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_hole_lift(
    hole_amplitudes: np.ndarray,
    hole_basis_states: np.ndarray,
    electron_basis_states: np.ndarray,
    n_orbitals: int,
) -> np.ndarray:
    """Assemble creation-lift columns in flattened determinant-orbital order.

    Parameters
    ----------
    hole_amplitudes : np.ndarray
        Hole-state amplitudes with shape
        (n_det, n_hole_states).
    hole_basis_states : np.ndarray
        Bitstrings for the hole sector.
    electron_basis_states : np.ndarray
        Bitstrings for the sector with one additional electron.
    n_orbitals : int
        Number of spin orbitals.

    Returns
    -------
    np.ndarray
        Lift matrix with columns ordered by
        I*n_orbitals+mu.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _create(state: int, orbital: int):
    mask = 1 << orbital
    if state & mask:
        return None
    sign = -1 if (state & (mask - 1)).bit_count() % 2 else 1
    return state | mask, sign


def _oracle_assemble_hole_lift(
    hole_amplitudes,
    hole_basis_states,
    electron_basis_states,
    n_orbitals,
):
    """Reference creation lift with occupation parity."""
    import numpy as np

    amplitudes = np.asarray(hole_amplitudes, dtype=float)
    holes = np.asarray(hole_basis_states)
    electrons = np.asarray(electron_basis_states)

    if (
        not isinstance(n_orbitals, (int, np.integer))
        or int(n_orbitals) < 1
    ):
        raise ValueError("n_orbitals must be positive")

    m = int(n_orbitals)

    if (
        amplitudes.ndim != 2
        or amplitudes.shape[0] < 1
        or amplitudes.shape[1] != holes.size
    ):
        raise ValueError("hole amplitudes and basis do not align")
    if (
        holes.ndim != 1
        or electrons.ndim != 1
        or holes.size < 1
        or electrons.size < 1
    ):
        raise ValueError(
            "basis arrays must be nonempty and one-dimensional"
        )
    if not np.all(np.isfinite(amplitudes)):
        raise ValueError("amplitudes must be finite")
    if (
        not np.all(np.isfinite(holes))
        or not np.all(holes == np.floor(holes))
    ):
        raise ValueError("hole states must be finite integers")
    if (
        not np.all(np.isfinite(electrons))
        or not np.all(electrons == np.floor(electrons))
    ):
        raise ValueError("electron states must be finite integers")

    holes = holes.astype(np.int64)
    electrons = electrons.astype(np.int64)

    hole_counts = {int(x).bit_count() for x in holes}
    electron_counts = {int(x).bit_count() for x in electrons}

    if len(hole_counts) != 1 or len(electron_counts) != 1:
        raise ValueError("each basis must have fixed particle count")
    if next(iter(electron_counts)) != next(iter(hole_counts)) + 1:
        raise ValueError("sectors must differ by one electron")

    lookup = {int(state): i for i, state in enumerate(electrons)}

    if len(lookup) != electrons.size:
        raise ValueError("electron states must be unique")

    lift = np.zeros(
        (electrons.size, amplitudes.shape[0] * m)
    )

    for determinant in range(amplitudes.shape[0]):
        for mu in range(m):
            for j, raw in enumerate(holes):
                created = _create(int(raw), mu)
                if created is not None:
                    state, sign = created
                    if state not in lookup:
                        raise ValueError(
                            "basis is not closed under creation"
                        )
                    lift[
                        lookup[state],
                        determinant * m + mu,
                    ] += sign * amplitudes[determinant, j]

    return lift

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    call = (
        "assemble_hole_lift("
        "hole_amplitudes, hole_basis_states, "
        "electron_basis_states, n_orbitals)"
    )
    gold = (
        "_oracle_assemble_hole_lift("
        "hole_amplitudes, hole_basis_states, "
        "electron_basis_states, n_orbitals)"
    )

    return [
        {
            "setup": (
                "import numpy as np\n"
                "hole_amplitudes=np.array([[2.]])\n"
                "hole_basis_states=np.array([0]); "
                "electron_basis_states=np.array([1,2,4]); "
                "n_orbitals=3"
            ),
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": (
                "import numpy as np\n"
                "hole_amplitudes=np.array([[1.,0.,0.]])\n"
                "hole_basis_states=np.array([1,2,4]); "
                "electron_basis_states=np.array([3,5,6]); "
                "n_orbitals=3"
            ),
            "call": call,
            "gold_call": gold,
        },
        {
            "setup": (
                "import numpy as np\n"
                "hole_amplitudes=np.zeros((2,3))\n"
                "hole_basis_states=np.array([1,2,4]); "
                "electron_basis_states=np.array([3,5,6]); "
                "n_orbitals=3"
            ),
            "call": call,
            "gold_call": gold,
        },
    ]
