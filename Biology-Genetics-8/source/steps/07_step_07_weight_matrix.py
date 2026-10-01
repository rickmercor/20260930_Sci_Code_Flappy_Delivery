"""
Build the weight matrix that converts the integer encoding into units of time.

The encoding counts lineages, so a difference between two encodings is measured in

lineages and is blind to how much time separates the events involved. When the networks

carry branch lengths, each entry of the encoding should instead be charged by the span of

time over which the persistence it records is being asserted. That span begins at the

moment the earlier of the two intervals begins, not at the moment it ends, since the entry

concerns lineages present throughout that interval; and it ends at the later event itself.

The diagonal is a different kind of statement: it records how many lineages coexist within

a single interval rather than anything persisting across a span, so it carries no weight at

all. The matrix is therefore zero on and above the diagonal and strictly positive below it,

and because two networks with the same number of events may have quite different tempos,

each network's weight matrix must be built from its own event times.

Returns
-------
list of N lists of numbers; entry (i, j) for j < i is the positive elapsed time from the start of interval j to the event of rank i, and every entry on or above the diagonal is 0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def weight_matrix(times: list[float]) -> list[list[float]]:
    '''Build the time-weight matrix of one ranked network.

    Parameters
    ----------
    times : list[float]
        Strictly increasing event times u_0 to u_N, where u_0 is the time at the
        top of the stem above the root, u_k is the time of the event of rank k,
        and u_N is the sampling time of the leaves.

    Returns
    -------
    weights : list[list[float]]
        An N by N matrix. Entry (i, j) with j strictly less than i is the elapsed
        time from the start of interval j to the event of rank i, expressed as a
        positive branch length. All other entries are zero.
    '''
    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_weight_matrix(times):
    if len(times) < 2:
        raise ValueError("times must list u_0 through u_N")
    N = len(times) - 1
    for k in range(1, len(times)):
        if times[k] <= times[k - 1]:
            raise ValueError("event times must be strictly increasing")
    return [[(times[i] - times[j - 1]) if j < i else 0
             for j in range(1, N + 1)] for i in range(1, N + 1)]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = '''UA = [0,1,2,3,4,5,10,11,12]
UB = [0,2,4,6,7,8,9,11,12]
'''
    return [
        {   # normal: the nine event times of network A
            "setup": setup,
            "call": 'weight_matrix(UA)',
            "gold_call": '_oracle_weight_matrix(UA)',
        },
        {   # normal: network B, whose clock runs differently
            "setup": setup,
            "call": 'weight_matrix(UB)',
            "gold_call": '_oracle_weight_matrix(UB)',
        },
        {   # boundary: two intervals, a single nonzero weight
            "setup": setup,
            "call": 'weight_matrix([0,1,2])',
            "gold_call": '_oracle_weight_matrix([0,1,2])',
        },
        {   # edge: one interval, so the matrix is the single zero
            "setup": setup,
            "call": 'weight_matrix([0,5])',
            "gold_call": '_oracle_weight_matrix([0,5])',
        },
    ]
