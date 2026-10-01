"""
Return the sum over all entries of the absolute difference between the two networks' weighted counts.

Once both networks are described by matrices of the same size, and each entry has been multiplied by its elapsed time to give a total branch length, the two are compared entry by entry and the absolute differences are summed. This is an ordinary matrix norm of the difference, which is what makes the result a metric and not merely a score. Because the encoding is a complete invariant, a distance of zero means the two histories are identical in shape and in timing.

Returns
-------
float - the summed absolute difference of the weighted entries
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def weighted_l1(matrix_a, weights_a, matrix_b, weights_b):

    """Compute the weighted L1 distance between two aligned F-matrices.



    Parameters

    ----------

    matrix_a, matrix_b : list of list of int

        Aligned lower-triangular lineage-count matrices.

    weights_a, weights_b : list of list of float

        Corresponding lower-triangular elapsed-time weight matrices.



    Returns

    -------

    float

        The sum of all lower-triangular weighted absolute differences.



    Raises

    ------

    ValueError

        If the four matrices are empty, non-square, or have different shapes;

        if count entries are invalid; if weight entries are non-numeric or

        non-finite; or if any matrix has nonzero entries above its diagonal.

    """

    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_weighted_l1(matrix_a, weights_a, matrix_b, weights_b):

    """Return the L1 distance between two weighted count matrices."""

    import math

    import numpy as np



    matrices = [matrix_a, weights_a, matrix_b, weights_b]



    if any(not isinstance(matrix, (list, tuple)) for matrix in matrices):

        raise ValueError("all inputs must be list or tuple matrices")



    sizes = [len(matrix) for matrix in matrices]



    if any(size == 0 for size in sizes) or len(set(sizes)) != 1:

        raise ValueError("all matrices must be non-empty and have the same size")



    size = sizes[0]



    if any(

        not isinstance(row, (list, tuple)) or len(row) != size

        for matrix in matrices

        for row in matrix

    ):

        raise ValueError("all matrices must be square")



    for matrix in (matrix_a, matrix_b):

        for row_index, row in enumerate(matrix):

            for column_index, value in enumerate(row):

                if (

                    not isinstance(value, (int, np.integer))

                    or isinstance(value, bool)

                    or int(value) < 0

                ):

                    raise ValueError(

                        "count-matrix entries must be non-negative integers"

                    )

                if column_index > row_index and value != 0:

                    raise ValueError("count matrices must be lower triangular")



    for matrix in (weights_a, weights_b):

        for row_index, row in enumerate(matrix):

            for column_index, value in enumerate(row):

                if (

                    not isinstance(value, (int, float, np.integer, np.floating))

                    or isinstance(value, bool)

                    or not math.isfinite(float(value))

                    or float(value) < 0.0

                ):

                    raise ValueError(

                        "weight-matrix entries must be finite and non-negative"

                    )

                if column_index >= row_index and value != 0.0:

                    raise ValueError(

                        "weight matrices must be zero on and above the diagonal"

                    )



    return sum(

        abs(

            matrix_a[row][column] * weights_a[row][column]

            - matrix_b[row][column] * weights_b[row][column]

        )

        for row in range(size)

        for column in range(size)

    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    """Return weighted-L1 test specifications."""

    incompatible_shapes = """ma = [[1, 0], [1, 1]]

wa = [[0.0, 0.0], [1.5, 0.0]]

mb = [[1, 0, 0], [0, 2, 0], [0, 1, 3]]

wb = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]



def run_model():

    try:

        weighted_l1(ma, wa, mb, wb)

        return 0

    except ValueError:

        return 1

    except Exception:

        return 2



def run_gold():

    try:

        _oracle_weighted_l1(ma, wa, mb, wb)

        return 0

    except ValueError:

        return 1

    except Exception:

        return 2

"""



    return [

        {

            # Normal: one nonzero lower-triangular difference.

            "setup": """ma = [[1, 0], [1, 1]]

wa = [[0.0, 0.0], [1.5, 0.0]]

mb = [[1, 0], [0, 2]]

wb = [[0.0, 0.0], [1.5, 0.0]]

""",

            "call": "weighted_l1(ma, wa, mb, wb)",

            "gold_call": "_oracle_weighted_l1(ma, wa, mb, wb)",

        },

        {

            # Normal: identical counts but different elapsed-time weights.

            "setup": """ma = [[1, 0, 0], [0, 2, 0], [0, 1, 3]]

wa = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 2.0, 0.0]]

mb = [[1, 0, 0], [0, 2, 0], [0, 1, 3]]

wb = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 2.0, 0.0]]

""",

            "call": "weighted_l1(ma, wa, mb, wb)",

            "gold_call": "_oracle_weighted_l1(ma, wa, mb, wb)",

        },

        {

            # Boundary: two-by-two matrices with distinct count and weight terms.

            "setup": """ma = [[2, 0], [3, 4]]

wa = [[0.0, 0.0], [2.5, 0.0]]

mb = [[1, 0], [1, 1]]

wb = [[0.0, 0.0], [0.5, 0.0]]

""",

            "call": "weighted_l1(ma, wa, mb, wb)",

            "gold_call": "_oracle_weighted_l1(ma, wa, mb, wb)",

        },

        {

            # Invalid: both networks must use a common reconciled dimension.

            "setup": incompatible_shapes,

            "call": "run_model()",

            "gold_call": "run_gold()",

        },

    ]
