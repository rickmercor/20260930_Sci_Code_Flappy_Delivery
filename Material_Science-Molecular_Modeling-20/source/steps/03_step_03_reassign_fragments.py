"""
Assign compatible fragments to connection slots within the paper’s default search radius. A slot or fragment may be used only once.

MOFBuilder assigns coordinating fragments using a distance criterion and exclusive ownership. Here, eligible pairs are processed in ascending order of distance, row index, and column index.

Returns
-------
A tuple containing the assigned fragment index for each slot and the total distance in Å. Unassigned slots contain -1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reassign_fragments(distance_matrix: list) -> tuple:
    """Assign fragments without reusing columns.

    Parameters
    ----------
    distance_matrix : list
        Rectangular matrix of finite nonnegative distances in Å.
        Rows are connection slots; columns are compatible fragments.
        The search radius is 4.0 Å; accept distances <= 4.0 + 1e-8 Å.
        Sort eligible pairs by (distance, row, column).
        Accept a pair only if its row and column are unused.

    Returns
    -------
    tuple
        (assigned_columns, total_distance).
        Unassigned rows contain -1.
        Empty input returns ([], 0.0).
        Zero columns return ([-1] * number_of_rows, 0.0).

    Raises
    ------
    ValueError
        Ragged matrix, negative distance, or nonfinite distance.
    """
    return ([], 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reassign_fragments(distance_matrix: list) -> tuple:
    import math
    if not distance_matrix:
        return ([], 0.0)
    ncols = len(distance_matrix[0])
    if any((len(row) != ncols for row in distance_matrix)):
        raise ValueError('ragged distance matrix')
    if any((not math.isfinite(distance) or distance < 0 for row in distance_matrix for distance in row)):
        raise ValueError('invalid distance')
    cutoff = 4.0
    pairs = sorted(((distance, i, j) for i, row in enumerate(distance_matrix) for j, distance in enumerate(row) if distance <= cutoff + 1e-08))
    assignment = [-1] * len(distance_matrix)
    used_columns = set()
    total_distance = 0.0
    for distance, i, j in pairs:
        if assignment[i] < 0 and j not in used_columns:
            assignment[i] = j
            used_columns.add(j)
            total_distance += distance
    return (assignment, float(total_distance))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'reassign_fragments([[1.3, 2, 5], [1, 5, 3]])', 'gold_call': '_oracle_reassign_fragments([[1.3, 2, 5], [1, 5, 3]])'}, {'setup': '', 'call': 'reassign_fragments([[4]])', 'gold_call': '_oracle_reassign_fragments([[4]])'}, {'setup': '', 'call': 'reassign_fragments([[4.000000005]])', 'gold_call': '_oracle_reassign_fragments([[4.000000005]])'}, {'setup': '', 'call': 'reassign_fragments([[4.00000002]])', 'gold_call': '_oracle_reassign_fragments([[4.00000002]])'}, {'setup': '', 'call': 'reassign_fragments([[1, 1], [1, 1]])', 'gold_call': '_oracle_reassign_fragments([[1, 1], [1, 1]])'}, {'setup': '', 'call': 'reassign_fragments([[1, 2], [1.1, 5]])', 'gold_call': '_oracle_reassign_fragments([[1, 2], [1.1, 5]])'}, {'setup': '', 'call': 'reassign_fragments([])', 'gold_call': '_oracle_reassign_fragments([])'}, {'setup': '', 'call': 'reassign_fragments([[], []])', 'gold_call': '_oracle_reassign_fragments([[], []])'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: reassign_fragments([[],[1]]))', 'gold_call': '_check_error(lambda: _oracle_reassign_fragments([[],[1]]))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: reassign_fragments([[-1]]))', 'gold_call': '_check_error(lambda: _oracle_reassign_fragments([[-1]]))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: reassign_fragments([[float("inf")]]))', 'gold_call': '_check_error(lambda: _oracle_reassign_fragments([[float("inf")]]))'}]
