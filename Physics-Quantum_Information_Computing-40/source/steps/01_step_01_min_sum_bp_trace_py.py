"""
Run standard min-sum BP and return posterior LLRs, hard decisions, final warm-start edge messages, and convergence status for every executed iteration.

The source decoder is initialized by the detector-to-error and error-to-detector updates of Appendix A. Exposing the iteration trace preserves the posterior history needed for the later cumulative reliability metric.

Returns
-------
real `np.ndarray` of shape `(T,2*N+E+1)` with posterior LLRs, hard bits, row-major error-to-detector messages, and a success flag
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def min_sum_bp_trace(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", max_iters: int) -> "np.ndarray":
    '''Return the standard min-sum BP trace in deterministic row-major edge order.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (M,N); each detector row has degree at least two.
    p : np.ndarray
        Error-source probabilities of shape (N,), all strictly between 0 and 0.5.
    syndrome : np.ndarray
        Binary syndrome vector of shape (M,).
    max_iters : int
        Positive maximum number of BP iterations.

    Returns
    -------
    trace : np.ndarray
        Float array of shape (T, 2*N + E + 1), T <= max_iters, where E is the
        number of nonzero entries in H. Columns are posterior LLRs, hard bits,
        error-to-detector messages in row-major (detector,error) edge order,
        and a success flag. Stop at the first syndrome match. Use sign(0)=+1
        and hard bit 0 for posterior>0, otherwise 1.
    '''
    return trace

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_min_sum_bp_trace(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", max_iters: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int) % 2
    p = np.asarray(p, dtype=float)
    s = np.asarray(syndrome, dtype=int) % 2
    M, N = H.shape
    edges = [(i, j) for i in range(M) for j in range(N) if H[i, j]]
    edge_index = {edge: k for k, edge in enumerate(edges)}
    row_neighbors = [np.flatnonzero(H[i]).tolist() for i in range(M)]
    col_neighbors = [np.flatnonzero(H[:, j]).tolist() for j in range(N)]
    prior = np.log((1.0 - p) / p)
    edge_msgs = np.asarray([prior[j] for i, j in edges], dtype=float)
    rows = []
    for _ in range(int(max_iters)):
        detector_msgs = np.zeros(len(edges), dtype=float)
        for k, (i, j) in enumerate(edges):
            incoming = np.asarray(
                [edge_msgs[edge_index[(i, jp)]] for jp in row_neighbors[i] if jp != j],
                dtype=float,
            )
            signs = np.where(incoming >= 0.0, 1.0, -1.0)
            detector_msgs[k] = ((-1.0) ** int(s[i])) * float(np.prod(signs)) * float(np.min(np.abs(incoming)))
        new_edge_msgs = edge_msgs.copy()
        for k, (i, j) in enumerate(edges):
            new_edge_msgs[k] = prior[j] + sum(
                detector_msgs[edge_index[(ip, j)]] for ip in col_neighbors[j] if ip != i
            )
        posterior = np.asarray([
            prior[j] + sum(detector_msgs[edge_index[(i, j)]] for i in col_neighbors[j])
            for j in range(N)
        ], dtype=float)
        hard = (posterior <= 0.0).astype(int)
        success = bool(np.array_equal((H @ hard) % 2, s))
        rows.append(np.concatenate([posterior, hard.astype(float), new_edge_msgs, [float(success)]]))
        edge_msgs = new_edge_msgs
        if success:
            break
    return np.vstack(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    '''Return list of test case specifications.'''
    return [
        {
            "setup": '''import numpy as np\nH=np.array([[1,1,0,1],[0,1,1,1],[1,0,1,1]],dtype=int)\np=np.array([.08,.12,.05,.16])\ns=np.array([1,0,1],dtype=int)\n''',
            "call": "min_sum_bp_trace(H.copy(),p.copy(),s.copy(),5)",
            "gold_call": "_oracle_min_sum_bp_trace(H.copy(),p.copy(),s.copy(),5)",
            "tol": 1e-12,
        },
        {
            "setup": '''import numpy as np\nH=np.array([[1,1,1],[1,0,1],[0,1,1]],dtype=int)\np=np.array([.02,.20,.11])\ns=np.array([0,0,0],dtype=int)\n''',
            "call": "min_sum_bp_trace(H.copy(),p.copy(),s.copy(),4)",
            "gold_call": "_oracle_min_sum_bp_trace(H.copy(),p.copy(),s.copy(),4)",
            "tol": 1e-12,
        },
        {
            "setup": '''import numpy as np\nH=np.array([[1,1,0,1,0],[0,1,1,0,1],[1,0,1,0,1],[0,1,0,1,1]],dtype=int)\np=np.array([.17,.06,.13,.09,.04])\ns=np.array([1,1,0,1],dtype=int)\n''',
            "call": "min_sum_bp_trace(H.copy(),p.copy(),s.copy(),7)",
            "gold_call": "_oracle_min_sum_bp_trace(H.copy(),p.copy(),s.copy(),7)",
            "tol": 1e-12,
        },
    ]
