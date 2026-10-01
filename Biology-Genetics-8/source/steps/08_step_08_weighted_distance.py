"""
Take the Euclidean norm of the difference of the two weighted encodings.

Each network is now described by two matrices of the same shape, the integer encoding and

the weights, and the timed description of the network is their entrywise product. The

distance is a matrix norm applied to the difference of the two products. The order matters

and is the whole content of the step: each encoding is multiplied by its own weight matrix

first, and only then are the two products subtracted. Subtracting the encodings first and

then applying a weight would require the two networks to share a clock, which is exactly

the assumption the timed distance is designed to avoid. Because both weight matrices vanish

on and above the diagonal, only the strictly lower triangle contributes, so any diagonal

agreement between the encodings has no effect on the result. Under the Euclidean norm the

outcome is the square root of the summed squared entrywise differences, and the sign

convention used for the weights cancels, since the same convention is applied to both

networks and the difference is squared.

Returns
-------
float, the Euclidean norm of the difference of the two weighted encodings, rounded to 6 decimal places
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def weighted_distance(matrix_a: list[list[int]], weights_a: list[list[float]],
                      matrix_b: list[list[int]], weights_b: list[list[float]]) -> float:
    '''Distance between two timed ranked networks under the Euclidean matrix norm.
    Parameters
    ----------
    matrix_a, matrix_b : list[list[int]]
        The triangular encodings of the two networks, of equal size.
    weights_a, weights_b : list[list[float]]
        The corresponding weight matrices, of the same size.
    Returns
    -------
    distance : float
        The norm of the difference of the two weighted encodings, rounded to
        6 decimal places.
    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_weighted_distance(matrix_a, weights_a, matrix_b, weights_b):
    import math
    N = len(matrix_a)
    if not (len(weights_a) == len(matrix_b) == len(weights_b) == N):
        raise ValueError("all four matrices must have the same size")
    total = 0.0
    for i in range(N):
        for j in range(N):
            total += (matrix_a[i][j] * weights_a[i][j]
                      - matrix_b[i][j] * weights_b[i][j]) ** 2
    return round(math.sqrt(total), 6)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = '''
FA = [[1,0,0,0,0,0,0,0],[0,2,0,0,0,0,0,0],[0,1,3,0,0,0,0,0],[0,0,2,4,0,0,0,0],
      [0,0,1,3,5,0,0,0],[0,0,1,2,4,6,0,0],[0,0,0,0,2,4,5,0],[0,0,0,0,1,2,3,4]]
FB = [[1,0,0,0,0,0,0,0],[0,2,0,0,0,0,0,0],[0,1,3,0,0,0,0,0],[0,0,2,4,0,0,0,0],
      [0,0,2,3,5,0,0,0],[0,0,2,3,4,6,0,0],[0,0,2,3,3,4,5,0],[0,0,2,3,3,3,3,4]]
WA = [[0,0,0,0,0,0,0,0],[2,0,0,0,0,0,0,0],[3,2,0,0,0,0,0,0],[4,3,2,0,0,0,0,0],
      [5,4,3,2,0,0,0,0],[10,9,8,7,6,0,0,0],[11,10,9,8,7,6,0,0],[12,11,10,9,8,7,2,0]]
WB = [[0,0,0,0,0,0,0,0],[4,0,0,0,0,0,0,0],[6,4,0,0,0,0,0,0],[7,5,3,0,0,0,0,0],
      [8,6,4,2,0,0,0,0],[9,7,5,3,2,0,0,0],[11,9,7,5,4,3,0,0],[12,10,8,6,5,4,3,0]]
'''
    return [
        {   # normal: the two timed networks of the task
            "setup": setup,
            "call": 'weighted_distance(FA, WA, FB, WB)',
            "gold_call": '_oracle_weighted_distance(FA, WA, FB, WB)',
        },
        {   # boundary: a network against itself
            "setup": setup,
            "call": 'weighted_distance(FA, WA, FA, WA)',
            "gold_call": '_oracle_weighted_distance(FA, WA, FA, WA)',
        },
        {   # normal: the arguments swapped
            "setup": setup,
            "call": 'weighted_distance(FB, WB, FA, WA)',
            "gold_call": '_oracle_weighted_distance(FB, WB, FA, WA)',
        },
        {   # edge: a hand case where only one entry contributes
            "setup": setup,
            "call": 'weighted_distance([[1,0],[0,2]], [[0,0],[3,0]], [[1,0],[1,2]], [[0,0],[4,0]])',
            "gold_call": '_oracle_weighted_distance([[1,0],[0,2]], [[0,0],[3,0]], [[1,0],[1,2]], [[0,0],[4,0]])',
        },
    ]
