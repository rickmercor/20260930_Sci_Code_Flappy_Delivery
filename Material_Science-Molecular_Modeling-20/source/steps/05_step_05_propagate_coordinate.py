"""
Calculate the target edge length from Eq. (4) and propagate an endpoint from the supplied starting position.

Once orientations are fixed, MOFBuilder propagates coordinates along the graph. Each connection is scaled using the linker length, bond constant, and geometric offset. The direction must be normalized before propagation.

Returns
-------
A tuple containing the target length and the endpoint coordinates, all in Å.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def propagate_coordinate(
    start: list,
    direction: list,
    linker_length: float,
    const_length: float,
    offset_length: float,
) -> tuple:
    """Return the target length and propagated endpoint.

    Parameters
    ----------
    start, direction : list
        Finite Cartesian 3-vectors.
        Normalize direction before propagation.
    linker_length, const_length, offset_length : float
        Finite nonnegative lengths in Å.
        target = linker_length + 2*const_length + 2*offset_length.
        endpoint = start + target * direction / norm(direction).

    Returns
    -------
    tuple
        (target_length, (end_x, end_y, end_z)), in Å.

    Raises
    ------
    ValueError
        Invalid vectors, direction norm <= 1e-8,
        negative/nonfinite lengths, or nonpositive total target length.
    """
    return (0.0, (0.0, 0.0, 0.0))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_propagate_coordinate(start: list, direction: list, linker_length: float, const_length: float, offset_length: float) -> tuple:
    import math
    if any((len(vector) != 3 or not all((math.isfinite(x) for x in vector)) for vector in (start, direction))):
        raise ValueError('invalid vector')
    if any((not math.isfinite(length) or length < 0 for length in (linker_length, const_length, offset_length))):
        raise ValueError('invalid length')
    norm = math.sqrt(sum((x * x for x in direction)))
    if norm <= 1e-08:
        raise ValueError('zero direction')
    target = linker_length + 2 * const_length + 2 * offset_length
    if target <= 0:
        raise ValueError('nonpositive target')
    endpoint = tuple((start[k] + target * direction[k] / norm for k in range(3)))
    return (float(target), endpoint)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'propagate_coordinate([1,2,3], [0,0,5], 1.6, 0.2, 0.1)', 'gold_call': '_oracle_propagate_coordinate([1,2,3], [0,0,5], 1.6, 0.2, 0.1)'}, {'setup': '', 'call': 'propagate_coordinate([1,2,3], [3,4,0], 1, 0, 0)', 'gold_call': '_oracle_propagate_coordinate([1,2,3], [3,4,0], 1, 0, 0)'}, {'setup': '', 'call': 'propagate_coordinate([0,0,0], [-2,0,0], 0, 0.5, 0)', 'gold_call': '_oracle_propagate_coordinate([0,0,0], [-2,0,0], 0, 0.5, 0)'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: propagate_coordinate([0,0,0], [0,0,0], 1, 0, 0))', 'gold_call': '_check_error(lambda: _oracle_propagate_coordinate([0,0,0], [0,0,0], 1, 0, 0))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: propagate_coordinate([0,0,0], [1,0,0], -1, 0, 0))', 'gold_call': '_check_error(lambda: _oracle_propagate_coordinate([0,0,0], [1,0,0], -1, 0, 0))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: propagate_coordinate([0,0,0], [1,0,0], 1, -1, 0))', 'gold_call': '_check_error(lambda: _oracle_propagate_coordinate([0,0,0], [1,0,0], 1, -1, 0))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: propagate_coordinate([0,0,0], [1,0,0], 0, 0, 0))', 'gold_call': '_check_error(lambda: _oracle_propagate_coordinate([0,0,0], [1,0,0], 0, 0, 0))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: propagate_coordinate([0,0], [1,0,0], 1, 0, 0))', 'gold_call': '_check_error(lambda: _oracle_propagate_coordinate([0,0], [1,0,0], 1, 0, 0))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: propagate_coordinate([0,0,0], [1,0,0], float("inf"), 0, 0))', 'gold_call': '_check_error(lambda: _oracle_propagate_coordinate([0,0,0], [1,0,0], float("inf"), 0, 0))'}]
