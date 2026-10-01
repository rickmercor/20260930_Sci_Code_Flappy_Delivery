"""
Calculate the cell loss for corresponding original and propagated graph nodes using Eq. (5).

The cell objective compares the original and updated node positions in fractional space. Each coordinate set is converted using its own cell matrix. Node identities and unwrapped image branches stay fixed.

Returns
-------
A float containing the dimensionless sum of squared fractional-coordinate differences.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fractional_cell_loss(old_cart: list, new_cart: list, old_cell: list, new_cell: list) -> float:
    """Evaluate the fractional-coordinate cell loss.

    Parameters
    ----------
    old_cart, new_cart : list
        Equally long lists of finite Cartesian 3-vectors in Å.
    old_cell, new_cell : list
        Finite nonsingular 3x3 row-lattice matrices in Å.
        Use x = f @ T.
        Compare the supplied unwrapped branches without wrapping
        fractional differences.

    Returns
    -------
    float
        Dimensionless sum of squared fractional-coordinate differences.
        Zero nodes return 0.0 after validating both cells.

    Raises
    ------
    ValueError
        Unequal node counts, malformed/nonfinite data,
        or abs(det(T)) <= 1e-12 Å^3 for either cell.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fractional_cell_loss(old_cart: list, new_cart: list, old_cell: list, new_cell: list) -> float:
    import math
    for cell in (old_cell, new_cell):
        if len(cell) != 3 or any((len(row) != 3 or not all((math.isfinite(x) for x in row)) for row in cell)):
            raise ValueError('invalid cell')
    if any((len(point) != 3 or not all((math.isfinite(x) for x in point)) for point in old_cart + new_cart)):
        raise ValueError('invalid coordinate')
    if len(old_cart) != len(new_cart):
        raise ValueError('shape mismatch')

    def inverse(m):
        determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        if abs(determinant) <= 1e-12:
            raise ValueError('singular cell')
        inv = [[0.0] * 3 for _ in range(3)]
        inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / determinant
        inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / determinant
        inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / determinant
        inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / determinant
        inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / determinant
        inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / determinant
        inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / determinant
        inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / determinant
        inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / determinant
        return inv
    inv_old = inverse(old_cell)
    inv_new = inverse(new_cell)
    total = 0.0
    for old_point, new_point in zip(old_cart, new_cart):
        old_fractional = [sum((old_point[k] * inv_old[k][j] for k in range(3))) for j in range(3)]
        new_fractional = [sum((new_point[k] * inv_new[k][j] for k in range(3))) for j in range(3)]
        total += sum(((new_fractional[j] - old_fractional[j]) ** 2 for j in range(3)))
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'fractional_cell_loss([[1,1,1]], [[2,1,1]], [[10,0,0],[0,10,0],[0,0,10]], [[10,0,0],[0,10,0],[0,0,10]])', 'gold_call': '_oracle_fractional_cell_loss([[1,1,1]], [[2,1,1]], [[10,0,0],[0,10,0],[0,0,10]], [[10,0,0],[0,10,0],[0,0,10]])'}, {'setup': '', 'call': 'fractional_cell_loss([[2,3,4]], [[4,6,8]], [[2,0,0],[0,3,0],[0,0,4]], [[4,0,0],[0,6,0],[0,0,8]])', 'gold_call': '_oracle_fractional_cell_loss([[2,3,4]], [[4,6,8]], [[2,0,0],[0,3,0],[0,0,4]], [[4,0,0],[0,6,0],[0,0,8]])'}, {'setup': '', 'call': 'fractional_cell_loss([[0,0,0]], [[3,3,0]], [[2,0,0],[1,3,0],[0,0,4]], [[2,0,0],[1,3,0],[0,0,4]])', 'gold_call': '_oracle_fractional_cell_loss([[0,0,0]], [[3,3,0]], [[2,0,0],[1,3,0],[0,0,4]], [[2,0,0],[1,3,0],[0,0,4]])'}, {'setup': '', 'call': 'fractional_cell_loss([[3,3,0]], [[0,0,0]], [[2,0,0],[1,3,0],[0,0,4]], [[2,0,0],[1,3,0],[0,0,4]])', 'gold_call': '_oracle_fractional_cell_loss([[3,3,0]], [[0,0,0]], [[2,0,0],[1,3,0],[0,0,4]], [[2,0,0],[1,3,0],[0,0,4]])'}, {'setup': '', 'call': 'fractional_cell_loss([[0,0,0]], [[10,0,0]], [[10,0,0],[0,10,0],[0,0,10]], [[10,0,0],[0,10,0],[0,0,10]])', 'gold_call': '_oracle_fractional_cell_loss([[0,0,0]], [[10,0,0]], [[10,0,0],[0,10,0],[0,0,10]], [[10,0,0],[0,10,0],[0,0,10]])'}, {'setup': '', 'call': 'fractional_cell_loss([], [], [[10,0,0],[0,10,0],[0,0,10]], [[10,0,0],[0,10,0],[0,0,10]])', 'gold_call': '_oracle_fractional_cell_loss([], [], [[10,0,0],[0,10,0],[0,0,10]], [[10,0,0],[0,10,0],[0,0,10]])'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: fractional_cell_loss([], [[0,0,0]], [[10,0,0],[0,10,0],[0,0,10]], [[10,0,0],[0,10,0],[0,0,10]]))', 'gold_call': '_check_error(lambda: _oracle_fractional_cell_loss([], [[0,0,0]], [[10,0,0],[0,10,0],[0,0,10]], [[10,0,0],[0,10,0],[0,0,10]]))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: fractional_cell_loss([[0,0]], [[0,0,0]], [[10,0,0],[0,10,0],[0,0,10]], [[10,0,0],[0,10,0],[0,0,10]]))', 'gold_call': '_check_error(lambda: _oracle_fractional_cell_loss([[0,0]], [[0,0,0]], [[10,0,0],[0,10,0],[0,0,10]], [[10,0,0],[0,10,0],[0,0,10]]))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: fractional_cell_loss([], [], [[0,0,0],[0,1,0],[0,0,1]], [[10,0,0],[0,10,0],[0,0,10]]))', 'gold_call': '_check_error(lambda: _oracle_fractional_cell_loss([], [], [[0,0,0],[0,1,0],[0,0,1]], [[10,0,0],[0,10,0],[0,0,10]]))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: fractional_cell_loss([], [], [[10,0,0],[0,10,0],[0,0,10]], [[1,0],[0,1]]))', 'gold_call': '_check_error(lambda: _oracle_fractional_cell_loss([], [], [[10,0,0],[0,10,0],[0,0,10]], [[1,0],[0,1]]))'}, {'setup': '\ndef _check_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_check_error(lambda: fractional_cell_loss([], [], [[10,0,0],[0,10,0],[0,0,10]], [[1,0,0],[0,float("nan"),0],[0,0,1]]))', 'gold_call': '_check_error(lambda: _oracle_fractional_cell_loss([], [], [[10,0,0],[0,10,0],[0,0,10]], [[1,0,0],[0,float("nan"),0],[0,0,1]]))'}]
