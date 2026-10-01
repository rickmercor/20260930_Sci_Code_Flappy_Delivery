"""
Find the closest positive-distance periodic image of the second point relative to the first. Return its shift and Cartesian distance.

The connection search includes the central cell and its 26 neighbors. Coincident positions are excluded. Cell vectors are stored as rows, so Cartesian displacements are obtained from fractional displacements using x = f @ T.

Returns
-------
A tuple containing the integer image shift and its Cartesian distance in Å.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def periodic_neighbor(a_frac: list, b_frac: list, cell: list) -> tuple:
    """Return the closest positive-distance image.

    Parameters
    ----------
    a_frac, b_frac : list
        Finite fractional 3-vectors in [0, 1).
    cell : list
        Finite nonsingular 3x3 row-lattice matrix in Å.
        Search shifts in {-1, 0, 1}^3.
        Ignore distances <= 1e-8 Å.
        Among distances within 1e-12 Å of the global minimum,
        choose the lexicographically smallest shift.

    Returns
    -------
    tuple
        ((sx, sy, sz), distance).

    Raises
    ------
    ValueError
        Invalid coordinates or cell, abs(det(cell)) <= 1e-12 Å^3,
        or no positive-distance image.
    """
    return ((0, 0, 0), 0.0)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_periodic_neighbor(a_frac: list, b_frac: list, cell: list) -> tuple:
    import itertools
    import math
    if any((len(point) != 3 or any((not math.isfinite(x) or not 0 <= x < 1 for x in point)) for point in (a_frac, b_frac))):
        raise ValueError('invalid fractional point')
    if len(cell) != 3 or any((len(row) != 3 or not all((math.isfinite(x) for x in row)) for row in cell)):
        raise ValueError('invalid cell')
    a, b, c = cell
    determinant = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])
    if abs(determinant) <= 1e-12:
        raise ValueError('singular cell')
    images = []
    for shift in itertools.product((-1, 0, 1), repeat=3):
        fractional_delta = [b_frac[k] + shift[k] - a_frac[k] for k in range(3)]
        cartesian_delta = [sum((fractional_delta[k] * cell[k][j] for k in range(3))) for j in range(3)]
        distance = math.sqrt(sum((x * x for x in cartesian_delta)))
        if distance > 1e-08:
            images.append((shift, distance))
    if not images:
        raise ValueError('no positive image')
    minimum = min((distance for shift, distance in images))
    return min(((shift, distance) for shift, distance in images if distance <= minimum + 1e-12))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': '',
            'call': 'periodic_neighbor([0.1,0.1,0.1], [0.2,0.1,0.1], [[10,0,0],[0,10,0],[0,0,10]])',
            'gold_call': '_oracle_periodic_neighbor([0.1,0.1,0.1], [0.2,0.1,0.1], [[10,0,0],[0,10,0],[0,0,10]])',
        },
        {
            'setup': '',
            'call': 'periodic_neighbor([0.1,0.1,0.1], [0.9,0.1,0.1], [[10,0,0],[0,10,0],[0,0,10]])',
            'gold_call': '_oracle_periodic_neighbor([0.1,0.1,0.1], [0.9,0.1,0.1], [[10,0,0],[0,10,0],[0,0,10]])',
        },
        {
            'setup': '',
            'call': 'periodic_neighbor([0,0,0], [0,0,0], [[10,0,0],[0,10,0],[0,0,10]])',
            'gold_call': '_oracle_periodic_neighbor([0,0,0], [0,0,0], [[10,0,0],[0,10,0],[0,0,10]])',
        },
        {
            'setup': '',
            'call': 'periodic_neighbor([0,0,0], [0.5,0,0], [[10,0,0],[0,10,0],[0,0,10]])',
            'gold_call': '_oracle_periodic_neighbor([0,0,0], [0.5,0,0], [[10,0,0],[0,10,0],[0,0,10]])',
        },
        {
            'setup': '',
            'call': 'periodic_neighbor([0,0,0], [0.1,0.2,0], [[2,0,0],[1,3,0],[0,0,4]])',
            'gold_call': '_oracle_periodic_neighbor([0,0,0], [0.1,0.2,0], [[2,0,0],[1,3,0],[0,0,4]])',
        },
        {
            'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n',
            'call': '_check_error(lambda: periodic_neighbor([1,0,0], [0,0,0], [[10,0,0],[0,10,0],[0,0,10]]))',
            'gold_call': '_check_error(lambda: _oracle_periodic_neighbor([1,0,0], [0,0,0], [[10,0,0],[0,10,0],[0,0,10]]))',
        },
        {
            'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n',
            'call': '_check_error(lambda: periodic_neighbor([0,0,0], [0,0,0], [[0,0,0],[0,1,0],[0,0,1]]))',
            'gold_call': '_check_error(lambda: _oracle_periodic_neighbor([0,0,0], [0,0,0], [[0,0,0],[0,1,0],[0,0,1]]))',
        },
        {
            'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n',
            'call': '_check_error(lambda: periodic_neighbor([0,0,0], [0,0,0], [[1,0],[0,1]]))',
            'gold_call': '_check_error(lambda: _oracle_periodic_neighbor([0,0,0], [0,0,0], [[1,0],[0,1]]))',
        },
        {
            'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n',
            'call': '_check_error(lambda: periodic_neighbor([float("nan"),0,0], [0,0,0], [[10,0,0],[0,10,0],[0,0,10]]))',
            'gold_call': '_check_error(lambda: _oracle_periodic_neighbor([float("nan"),0,0], [0,0,0], [[10,0,0],[0,10,0],[0,0,10]]))',
        },
    ]
