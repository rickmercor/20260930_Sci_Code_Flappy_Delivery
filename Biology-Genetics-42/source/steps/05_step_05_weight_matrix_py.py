"""
Return the matrix of elapsed times attached to each entry of a count matrix, given the interval times.

The counts say how many lineages persist across a span but nothing about how long the span lasted, so each entry is paired with an elapsed time and the product is a total branch length. Which two boundaries that elapsed time runs between is a convention of the encoding and not a matter of taste: it is fixed by the source, and reading it differently changes every weighted entry at once. Entries that describe coexistence within a single interval rather than persistence across a span carry no weight.

Returns
-------
list of list of float - a square matrix of elapsed times, one row per interval
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def weight_matrix(times):

    """Compute the lower-triangular time-weight matrix for aligned intervals.



    If ``times`` contains ``N + 1`` strictly decreasing interval-boundary

    times, the returned matrix has shape ``N x N``. With zero-based Python

    indices, for row ``i`` and column ``j < i``,

    ``W[i][j] = times[j] - times[i + 1]``. Diagonal and upper-triangular

    entries are zero. This is the zero-based form of the paper's

    ``W_{i,j} = u_{j-1} - u_i`` convention.



    Parameters

    ----------

    times : list of float

        ``N + 1`` aligned interval-boundary times ordered from root to tip.



    Returns

    -------

    list of list of float

        An ``N x N`` lower-triangular matrix of nonnegative elapsed-time

        weights.



    Raises

    ------

    ValueError

        If ``times`` contains fewer than two entries, contains non-numeric or

        non-finite values, or is not strictly decreasing from root to tip.

    """

    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_weight_matrix(times):

    """Return the elapsed-time matrix for a reconciled time sequence."""

    import math

    import numpy as np



    if not isinstance(times, (list, tuple, np.ndarray)) or len(times) < 2:

        raise ValueError("times must contain at least a root and a tip time")



    if any(

        not isinstance(time, (int, float, np.integer, np.floating))

        or isinstance(time, bool)

        or not math.isfinite(float(time))

        for time in times

    ):

        raise ValueError("times must be finite real numbers")



    times = [float(time) for time in times]



    if any(

        times[index] <= times[index + 1]

        for index in range(len(times) - 1)

    ):

        raise ValueError("times must be strictly decreasing from root to tips")



    n_intervals = len(times) - 1



    return [

        [

            times[column - 1] - times[row]

            if column < row

            else 0.0

            for column in range(1, n_intervals + 1)

        ]

        for row in range(1, n_intervals + 1)

    ]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    """Return weight-matrix test specifications."""

    invalid_times = """times = [6.0, 4.0, 4.5, 1.0]



def run_model():

    try:

        weight_matrix(times)

        return 0

    except ValueError:

        return 1

    except Exception:

        return 2



def run_gold():

    try:

        _oracle_weight_matrix(times)

        return 0

    except ValueError:

        return 1

    except Exception:

        return 2

"""



    return [

        {

            # Normal: A after artificial-event time augmentation.

            "setup": (

                "times = [10.0, 9.25, 8.5, 7.5, 6.5, 6.0, "

                "5.5, 3.0, 1.5, 1.0]"

            ),

            "call": "weight_matrix(times)",

            "gold_call": "_oracle_weight_matrix(times)",

        },

        {

            # Normal: B, with no artificial-event time augmentation.

            "setup": (

                "times = [10.0, 9.0, 8.5, 8.0, 7.5, 6.0, "

                "5.0, 4.5, 3.0, 1.0]"

            ),

            "call": "weight_matrix(times)",

            "gold_call": "_oracle_weight_matrix(times)",

        },

        {

            # Boundary: three intervals.

            "setup": "times = [6.0, 4.0, 2.0, 1.0]",

            "call": "weight_matrix(times)",

            "gold_call": "_oracle_weight_matrix(times)",

        },

        {

            # Invalid: times must decrease from root to tips.

            "setup": invalid_times,

            "call": "run_model()",

            "gold_call": "run_gold()",

        },

    ]
