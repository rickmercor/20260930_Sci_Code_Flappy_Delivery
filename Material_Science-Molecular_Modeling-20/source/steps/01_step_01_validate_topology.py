"""
Check the candidate connections against the topology rule for the given linker type. Return the indices of connections with the required endpoint labels and an E site inside the segment.

MOFBuilder uses V–E–V for ditopic connections and V–E–EC for multitopic connections. The input coordinates are already unwrapped. Each connection is checked in Cartesian space.

Returns
-------
A list of valid connection indices in input order. Return [] if none pass.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def validate_topology(points: list, kinds: list, edge_points: list, candidate_edges: list, linker_type: str) -> list[int]:
    """Return valid connection indices.

    Parameters
    ----------
    points : list
        N finite Cartesian 3-vectors in Å.
    kinds : list
        N labels, each V or EC.
    edge_points : list
        Finite Cartesian E-site 3-vectors.
    candidate_edges : list
        Distinct oriented index pairs (i, j).
    linker_type : str
        'ditopic' requires V at both ends.
        'multitopic' requires V followed by EC.
        An E site must lie within 1e-8 Å of the segment, with its
        projection strictly between the endpoints.
        Count each edge once, regardless of the number of matching E sites.

    Returns
    -------
    list[int]
        Valid edge indices in input order; [] if none pass.

    Raises
    ------
    ValueError
        Unsupported linker type, inconsistent labels, malformed or
        nonfinite points, duplicate edges, invalid indices, or
        endpoints separated by at most 1e-8 Å.
    """
    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_validate_topology(points: list, kinds: list, edge_points: list, candidate_edges: list, linker_type: str) -> list[int]:
    import math
    if linker_type not in ('ditopic', 'multitopic'):
        raise ValueError('unsupported linker type')
    if len(kinds) != len(points) or any((kind not in ('V', 'EC') for kind in kinds)):
        raise ValueError('inconsistent labels')
    if any((len(p) != 3 or not all((math.isfinite(x) for x in p)) for p in points + edge_points)):
        raise ValueError('invalid point')
    seen = set()
    valid = []
    for edge_index, pair in enumerate(candidate_edges):
        if len(pair) != 2 or any((type(i) is not int or i < 0 or i >= len(points) for i in pair)):
            raise ValueError('invalid edge')
        i, j = pair
        if (i, j) in seen:
            raise ValueError('duplicate edge')
        seen.add((i, j))
        a, b = (points[i], points[j])
        vector = [b[k] - a[k] for k in range(3)]
        length_sq = sum((x * x for x in vector))
        if length_sq <= 1e-16:
            raise ValueError('coincident endpoints')
        end_kind = 'V' if linker_type == 'ditopic' else 'EC'
        if kinds[i] != 'V' or kinds[j] != end_kind:
            continue
        for point in edge_points:
            projection = sum(((point[k] - a[k]) * vector[k] for k in range(3))) / length_sq
            distance_sq = sum(((point[k] - a[k] - projection * vector[k]) ** 2 for k in range(3)))
            if 0.0 < projection < 1.0 and distance_sq <= 1e-16:
                valid.append(edge_index)
                break
    return valid

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': "validate_topology([[0,0,0],[2,0,0]], ['V','V'], [[1,0,0]], [[0,1]], 'ditopic')", 'gold_call': "_oracle_validate_topology([[0,0,0],[2,0,0]], ['V','V'], [[1,0,0]], [[0,1]], 'ditopic')"}, {'setup': '', 'call': "validate_topology([[0,0,0],[2,0,0]], ['V','EC'], [[1,0,0]], [[0,1]], 'multitopic')", 'gold_call': "_oracle_validate_topology([[0,0,0],[2,0,0]], ['V','EC'], [[1,0,0]], [[0,1]], 'multitopic')"}, {'setup': '', 'call': "validate_topology([[0,0,0],[2,0,0]], ['V','V'], [[1,0,0]], [[0,1]], 'multitopic')", 'gold_call': "_oracle_validate_topology([[0,0,0],[2,0,0]], ['V','V'], [[1,0,0]], [[0,1]], 'multitopic')"}, {'setup': '', 'call': "validate_topology([[0,0,0],[2,0,0]], ['EC','V'], [[1,0,0]], [[0,1]], 'multitopic')", 'gold_call': "_oracle_validate_topology([[0,0,0],[2,0,0]], ['EC','V'], [[1,0,0]], [[0,1]], 'multitopic')"}, {'setup': '', 'call': "validate_topology([[0,0,0],[2,0,0]], ['V','V'], [[0,0,0],[2,0,0],[3,0,0]], [[0,1]], 'ditopic')", 'gold_call': "_oracle_validate_topology([[0,0,0],[2,0,0]], ['V','V'], [[0,0,0],[2,0,0],[3,0,0]], [[0,1]], 'ditopic')"}, {'setup': '', 'call': "validate_topology([[0,0,0],[2,0,0]], ['V','V'], [[1,1e-9,0],[1,0,0]], [[0,1]], 'ditopic')", 'gold_call': "_oracle_validate_topology([[0,0,0],[2,0,0]], ['V','V'], [[1,1e-9,0],[1,0,0]], [[0,1]], 'ditopic')"}, {'setup': '', 'call': "validate_topology([[0,0,0],[2,0,0]], ['V','V'], [[1,2e-8,0]], [[0,1]], 'ditopic')", 'gold_call': "_oracle_validate_topology([[0,0,0],[2,0,0]], ['V','V'], [[1,2e-8,0]], [[0,1]], 'ditopic')"}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': "_check_error(lambda: validate_topology([[0,0,0]], ['V'], [], [], 'bad'))", 'gold_call': "_check_error(lambda: _oracle_validate_topology([[0,0,0]], ['V'], [], [], 'bad'))"}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': "_check_error(lambda: validate_topology([[0,0,0]], [], [], [], 'ditopic'))", 'gold_call': "_check_error(lambda: _oracle_validate_topology([[0,0,0]], [], [], [], 'ditopic'))"}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': "_check_error(lambda: validate_topology([[0,0]], ['V'], [], [], 'ditopic'))", 'gold_call': "_check_error(lambda: _oracle_validate_topology([[0,0]], ['V'], [], [], 'ditopic'))"}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': "_check_error(lambda: validate_topology([[0,0,0]], ['V'], [], [[0,1]], 'ditopic'))", 'gold_call': "_check_error(lambda: _oracle_validate_topology([[0,0,0]], ['V'], [], [[0,1]], 'ditopic'))"}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': "_check_error(lambda: validate_topology([[0,0,0]], ['V'], [], [[0,0]], 'ditopic'))", 'gold_call': "_check_error(lambda: _oracle_validate_topology([[0,0,0]], ['V'], [], [[0,0]], 'ditopic'))"}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': "_check_error(lambda: validate_topology([[0,0,0],[1,0,0]], ['V','V'], [], [[0,1],[0,1]], 'ditopic'))", 'gold_call': "_check_error(lambda: _oracle_validate_topology([[0,0,0],[1,0,0]], ['V','V'], [], [[0,1],[0,1]], 'ditopic'))"}, {'setup': '', 'call': "validate_topology([], [], [], [], 'ditopic')", 'gold_call': "_oracle_validate_topology([], [], [], [], 'ditopic')"}]
