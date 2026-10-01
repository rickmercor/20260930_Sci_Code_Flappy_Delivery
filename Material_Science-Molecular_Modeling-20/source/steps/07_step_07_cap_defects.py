"""
Place and orient the complete cap fragment at each vacant site.

MOFBuilder centers a capping fragment on its marked atom and rotates it toward the vacant coordination direction. The same rigid transformation is applied to every atom in the fragment.

Returns
-------
A list containing one transformed atom-coordinate list per site, with the original atom order preserved.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cap_defects(fragment: list, marked_index: int, axis_index: int, sites: list, vacant_vectors: list) -> list:
    """Place a complete cap fragment at each site.

    Parameters
    ----------
    fragment : list
        Nonempty list of finite Cartesian 3-vectors.
    marked_index, axis_index : int
        Distinct valid atom indices.
        The source axis points from marked_index to axis_index.
    sites, vacant_vectors : list
        Equally long lists of finite Cartesian 3-vectors.
        Use the shortest proper rotation from source to vacancy axis.
        For antiparallel unit axes with cross norm <= 1e-12,
        choose the Cartesian basis least parallel to the source,
        breaking ties in x, y, z order. Normalize source cross basis
        and rotate by pi about that axis.

    Returns
    -------
    list
        One transformed atom-coordinate list per site.
        Preserve atom order. Empty sites return [] after validation.

    Raises
    ------
    ValueError
        Invalid coordinates or atom indices, unequal site/vector counts,
        source-axis norm <= 1e-8, or vacancy norm <= 1e-8.
    """
    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_cap_defects(fragment: list, marked_index: int, axis_index: int, sites: list, vacant_vectors: list) -> list:
    import math
    if not fragment or any((type(i) is not int or i < 0 or i >= len(fragment) for i in (marked_index, axis_index))) or marked_index == axis_index:
        raise ValueError('invalid atom indices')
    if len(sites) != len(vacant_vectors):
        raise ValueError('site count mismatch')
    if any((len(point) != 3 or not all((math.isfinite(x) for x in point)) for point in fragment + sites + vacant_vectors)):
        raise ValueError('invalid coordinate')

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])

    def unit(vector):
        norm = math.sqrt(sum((x * x for x in vector)))
        if norm <= 1e-08:
            raise ValueError('zero axis')
        return [x / norm for x in vector]
    origin = fragment[marked_index]
    local = [[point[k] - origin[k] for k in range(3)] for point in fragment]
    source = unit(local[axis_index])
    result = []
    for site, vacant in zip(sites, vacant_vectors):
        target = unit(vacant)
        normal = cross(source, target)
        sine = math.sqrt(sum((x * x for x in normal)))
        cosine = max(-1.0, min(1.0, sum((source[k] * target[k] for k in range(3)))))
        if sine <= 1e-12:
            if cosine >= 0:
                rotated = local
            else:
                index = min(range(3), key=lambda k: (abs(source[k]), k))
                basis = [float(k == index) for k in range(3)]
                axis = unit(cross(source, basis))
                rotated = [[2 * sum((axis[k] * point[k] for k in range(3))) * axis[j] - point[j] for j in range(3)] for point in local]
        else:
            axis = [x / sine for x in normal]
            rotated = []
            for point in local:
                cross_term = cross(axis, point)
                dot = sum((axis[k] * point[k] for k in range(3)))
                rotated.append([point[k] * cosine + cross_term[k] * sine + axis[k] * dot * (1 - cosine) for k in range(3)])
        result.append([tuple((site[k] + point[k] for k in range(3))) for point in rotated])
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[0,0,0]], [[0,0,2]])', 'gold_call': '_oracle_cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[0,0,0]], [[0,0,2]])'}, {'setup': '', 'call': 'cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[4,5,6]], [[1,0,0]])', 'gold_call': '_oracle_cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[4,5,6]], [[1,0,0]])'}, {'setup': '', 'call': 'cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[0,0,0]], [[0,0,-1]])', 'gold_call': '_oracle_cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[0,0,0]], [[0,0,-1]])'}, {'setup': '', 'call': 'cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [], [])', 'gold_call': '_oracle_cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [], [])'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[0,0,0]], []))', 'gold_call': '_check_error(lambda: _oracle_cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[0,0,0]], []))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[0,0,0]], [[0,0,0]]))', 'gold_call': '_check_error(lambda: _oracle_cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[0,0,0]], [[0,0,0]]))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 0, [], []))', 'gold_call': '_check_error(lambda: _oracle_cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 0, [], []))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: cap_defects([[1,2,3],[1,2,4],[2,2,3]], -1, 1, [], []))', 'gold_call': '_check_error(lambda: _oracle_cap_defects([[1,2,3],[1,2,4],[2,2,3]], -1, 1, [], []))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: cap_defects([[0,0,0],[0,0,0]], 0, 1, [], []))', 'gold_call': '_check_error(lambda: _oracle_cap_defects([[0,0,0],[0,0,0]], 0, 1, [], []))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[0,0]], [[1,0,0]]))', 'gold_call': '_check_error(lambda: _oracle_cap_defects([[1,2,3],[1,2,4],[2,2,3]], 0, 1, [[0,0]], [[1,0,0]]))'}]
