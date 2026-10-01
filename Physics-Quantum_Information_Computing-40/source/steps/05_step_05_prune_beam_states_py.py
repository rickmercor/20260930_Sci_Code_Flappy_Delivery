"""
Keep at most the requested beam width of highest-reliability candidate paths.

The bounded-search method controls exponential branching by ranking candidates by path score and retaining the most reliable paths. Stable equal-score ordering is a benchmark tie convention disclosed in the prompt.

Returns
-------
real `np.ndarray` containing at most `beam_width` states in stable decreasing-score order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def prune_beam_states(states: "np.ndarray", beam_width: int) -> "np.ndarray":
    '''Keep the highest-scoring states using stable descending order.

    Parameters
    ----------
    states : np.ndarray
        Float array of shape (P,D) in the common path-state layout; column 2 is score.
    beam_width : int
        Positive maximum number of rows to retain.

    Returns
    -------
    pruned : np.ndarray
        At most beam_width rows sorted by decreasing score. Equal scores preserve
        their original input order.
    '''
    return pruned

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_prune_beam_states(states: "np.ndarray", beam_width: int) -> "np.ndarray":
    x = np.asarray(states, dtype=float)
    order = np.argsort(-x[:, 2], kind="stable")
    return x[order[:min(int(beam_width), x.shape[0])]].copy()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    '''Return list of test case specifications.'''
    return [
        {
            "setup": '''import numpy as np\nx=np.array([[0,3,2.0,1],[0,2,5.0,2],[1,1,3.0,3],[0,4,4.0,4]],dtype=float)\n''',
            "call": "prune_beam_states(x.copy(),2)",
            "gold_call": "_oracle_prune_beam_states(x.copy(),2)",
            "tol": 0.0,
        },
        {
            "setup": '''import numpy as np\nx=np.array([[0,1,4.0,0],[0,1,4.0,1],[0,1,4.0,2]],dtype=float)\n''',
            "call": "prune_beam_states(x.copy(),2)",
            "gold_call": "_oracle_prune_beam_states(x.copy(),2)",
            "tol": 0.0,
        },
        {
            "setup": '''import numpy as np\nx=np.array([[0,1,-1.0,0],[0,1,2.0,1]],dtype=float)\n''',
            "call": "prune_beam_states(x.copy(),5)",
            "gold_call": "_oracle_prune_beam_states(x.copy(),5)",
            "tol": 0.0,
        },
    ]
