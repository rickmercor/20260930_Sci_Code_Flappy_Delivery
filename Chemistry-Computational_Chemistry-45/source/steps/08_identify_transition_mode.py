"""
Run the complete graph-coordinate pipeline and select the transition mode.

The final benchmark ranks each mode by reactive-bond F1, then by independent nonbond motion, then by original candidate order.

Returns
-------
One-based integer index of the selected candidate mode.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def identify_transition_mode(
    reference_positions: np.ndarray,
    mode_vectors: np.ndarray,
    amplitudes: np.ndarray,
    bonds: np.ndarray,
    atomic_numbers: np.ndarray,
    expected_bonds: np.ndarray,
) -> int:
    """Identify the candidate transition mode selected by graph-coordinate analysis.

    Parameters
    ----------
    reference_positions : np.ndarray
        Reference Cartesian coordinates with shape (n_atoms, 3).
    mode_vectors : np.ndarray
        Candidate displacement fields with shape (n_modes, n_atoms, 3).
    amplitudes : np.ndarray
        Nonconstant trajectory amplitudes.
    bonds : np.ndarray
        Zero-based molecular-graph edges with shape (n_bonds, 2).
    atomic_numbers : np.ndarray
        Positive atomic numbers, one per atom.
    expected_bonds : np.ndarray
        Expected reactive bonds with shape (n_expected, 2).

    Returns
    -------
    int
        One-based index of the selected candidate mode.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_identify_transition_mode(
    reference_positions: np.ndarray,
    mode_vectors: np.ndarray,
    amplitudes: np.ndarray,
    bonds: np.ndarray,
    atomic_numbers: np.ndarray,
    expected_bonds: np.ndarray,
) -> int:
    """Compose the seven graph-coordinate steps and select a mode."""
    import numpy as np

    reference = np.asarray(reference_positions, dtype=float)
    modes = np.asarray(mode_vectors, dtype=float)
    if reference.ndim != 2 or reference.shape[0] < 2 or reference.shape[1] != 3:
        raise ValueError("reference_positions must have shape (n_atoms, 3)")
    if modes.ndim != 3 or modes.shape[0] < 1 or modes.shape[1:] != reference.shape:
        raise ValueError("mode_vectors must have shape (n_modes, n_atoms, 3)")

    coordinates = _oracle_enumerate_internal_coordinates(bonds, reference.shape[0])
    thresholds = np.array([0.4, 10.0, 20.0, 0.15], dtype=float)
    ranking = []
    for index, mode in enumerate(modes):
        trajectory = _oracle_generate_mode_trajectory(reference, mode, amplitudes)
        pair = _oracle_select_diverse_pair(trajectory)
        values = _oracle_evaluate_internal_coordinates(trajectory, coordinates)
        changes = _oracle_coordinate_change_magnitudes(values, pair, coordinates)
        statuses = _oracle_screen_graph_changes(
            changes,
            coordinates,
            atomic_numbers,
            thresholds,
        )
        f1_score = _oracle_bond_change_f1(statuses, coordinates, expected_bonds)
        independent_nonbond = int(np.count_nonzero((statuses == 3) | (statuses == 4)))
        ranking.append((-f1_score, independent_nonbond, index))

    return int(min(ranking)[2] + 1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    call = 'identify_transition_mode(\n    reference_positions,\n    mode_vectors,\n    amplitudes,\n    bonds,\n    atomic_numbers,\n    expected_bonds,\n)'
    gold_call = '_oracle_identify_transition_mode(\n    reference_positions,\n    mode_vectors,\n    amplitudes,\n    bonds,\n    atomic_numbers,\n    expected_bonds,\n)'
    return [
        {
            "setup": 'import numpy as np\nreference_positions = np.array([\n    [0.0,0.0,0.0],\n    [1.0,0.0,0.0],\n    [2.0,0.2,0.0],\n], dtype=float)\nmode_vectors = np.zeros((2,3,3), dtype=float)\nmode_vectors[0,0,0] = 0.30\nmode_vectors[1,1,0] = 0.30\namplitudes = np.array([-1.0,0.0,1.0], dtype=float)\nbonds = np.array([[0,1],[1,2]], dtype=int)\natomic_numbers = np.array([8,1,8], dtype=int)\nexpected_bonds = np.array([[0,1],[1,2]], dtype=int)\n',
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": 'import numpy as np\nreference_positions = np.array([\n    [0.0,0.0,0.0],\n    [1.0,0.0,0.0],\n    [0.0,3.0,0.0],\n    [1.0,3.0,0.0],\n    [1.0,4.0,0.0],\n], dtype=float)\nmode_vectors = np.zeros((2,5,3), dtype=float)\nmode_vectors[:,1,0] = 0.30\nmode_vectors[1,4,0] = 0.50\namplitudes = np.array([-1.0,0.0,1.0], dtype=float)\nbonds = np.array([[0,1],[2,3],[3,4]], dtype=int)\natomic_numbers = np.array([6,6,6,6,6], dtype=int)\nexpected_bonds = np.array([[0,1]], dtype=int)\n',
            "call": call,
            "gold_call": gold_call,
        },
        {
            "setup": 'import numpy as np\nreference_positions = np.array([\n    [0.0,0.0,0.0],\n    [1.0,0.0,0.0],\n], dtype=float)\nmode_vectors = np.zeros((3,2,3), dtype=float)\nmode_vectors[0,1,0] = 0.30\nmode_vectors[1,1,0] = 0.30\nmode_vectors[2,1,0] = 0.10\namplitudes = np.array([-1.0,0.0,1.0], dtype=float)\nbonds = np.array([[0,1]], dtype=int)\natomic_numbers = np.array([6,6], dtype=int)\nexpected_bonds = np.array([[0,1]], dtype=int)\n',
            "call": call,
            "gold_call": gold_call,
        },
    ]
