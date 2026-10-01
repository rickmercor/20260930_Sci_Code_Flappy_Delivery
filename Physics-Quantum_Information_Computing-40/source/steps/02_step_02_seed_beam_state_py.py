"""
Convert the initial BP trace into the single numerical seed path used by the bounded search.

The source seed keeps the final BP edge messages, has no fixed nodes, initializes score to zero, and selects its first branching variable from the cumulative posterior-LLR reliability over the initial run.

Returns
-------
real `np.ndarray` of length `4+2*N+E` encoding success, iterations, score, next position, masks, edge messages, and correction bits
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def seed_beam_state(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", initial_iters: int) -> "np.ndarray":
    '''Return the seed-path state built from the initial BP run.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (M,N).
    p : np.ndarray
        Error probabilities of shape (N,).
    syndrome : np.ndarray
        Binary syndrome of shape (M,).
    initial_iters : int
        Positive BP iteration cap.

    Returns
    -------
    state : np.ndarray
        Float vector of length 4+2*N+E with layout
        [success, iters, score, next_pos, mask_values(N), edge_msgs(E), correction(N)].
        mask_values are -1 for every unmasked node, score is 0, and next_pos is
        the smallest index minimizing the absolute cumulative posterior LLR over
        the iterations actually executed.
    '''
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_seed_beam_state(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", initial_iters: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int)
    N = H.shape[1]
    E = int(np.count_nonzero(H))
    trace = _oracle_min_sum_bp_trace(H, p, syndrome, int(initial_iters))
    posterior = trace[:, :N]
    final = trace[-1]
    hard = final[N:2 * N]
    edge_msgs = final[2 * N:2 * N + E]
    cumulative = np.sum(posterior, axis=0)
    next_pos = int(np.argmin(np.abs(cumulative)))
    masks = -np.ones(N, dtype=float)
    return np.concatenate([
        [final[-1], float(trace.shape[0]), 0.0, float(next_pos)],
        masks,
        edge_msgs,
        hard,
    ])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    '''Return list of test case specifications.'''
    return [
        {
            "setup": '''import numpy as np\nH=np.array([[1,1,0,1],[0,1,1,1],[1,0,1,1]],dtype=int)\np=np.array([.08,.12,.05,.16])\ns=np.array([1,0,1],dtype=int)\n''',
            "call": "seed_beam_state(H.copy(),p.copy(),s.copy(),4)",
            "gold_call": "_oracle_seed_beam_state(H.copy(),p.copy(),s.copy(),4)",
            "tol": 1e-12,
        },
        {
            "setup": '''import numpy as np\nH=np.array([[1,1,1],[1,0,1],[0,1,1]],dtype=int)\np=np.array([.02,.20,.11])\ns=np.array([0,0,0],dtype=int)\n''',
            "call": "seed_beam_state(H.copy(),p.copy(),s.copy(),3)",
            "gold_call": "_oracle_seed_beam_state(H.copy(),p.copy(),s.copy(),3)",
            "tol": 1e-12,
        },
        {
            "setup": '''import numpy as np\nH=np.array([[1,1,1,0,0],[0,1,1,1,0],[0,0,1,1,1],[1,0,0,1,1]],dtype=int)\np=np.array([.13,.07,.18,.05,.09])\ns=np.array([1,0,1,1],dtype=int)\n''',
            "call": "seed_beam_state(H.copy(),p.copy(),s.copy(),6)",
            "gold_call": "_oracle_seed_beam_state(H.copy(),p.copy(),s.copy(),6)",
            "tol": 1e-12,
        },
    ]
