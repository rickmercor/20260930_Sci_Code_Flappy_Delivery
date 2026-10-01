"""
Run min-sum BP after applying fixed-node masks, source-defined syndrome modification, and parent-message warm start.

Appendix B recasts every fixed-one variable by flipping its adjacent syndrome bits, removes messages to or from all fixed nodes, and initializes the remaining messages from the parent path rather than restarting from priors. This is a central source-specific computation.

Returns
-------
real `np.ndarray` of shape `(T,2*N+E+1)` containing the masked posterior/history state and success flag
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def masked_bp_trace(
    H: "np.ndarray",
    p: "np.ndarray",
    syndrome: "np.ndarray",
    edge_msgs: "np.ndarray",
    mask_values: "np.ndarray",
    max_iters: int,
) -> "np.ndarray":
    """Return the warm-started masked-BP trace.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (M, N).
    p : np.ndarray
        Error probabilities of shape (N,).
    syndrome : np.ndarray
        Original binary syndrome of shape (M,).
    edge_msgs : np.ndarray
        Warm-start error-to-detector messages for all E row-major Tanner
        edges. These values initialize the message state before iteration.
    mask_values : np.ndarray
        Integer-like vector of length N with -1 for unmasked nodes and
        0 or 1 for fixed nodes. Every computed detector-to-error message
        has at least one other unmasked neighbor for the supplied inputs.
    max_iters : int
        Positive masked-BP iteration cap.

    Returns
    -------
    trace : np.ndarray
        Float array of shape (T, 2*N + E + 1), where T <= max_iters.
        Columns contain posterior LLRs, hard bits, error-to-detector
        messages in row-major edge order, and a success flag, respectively.
        Masked posterior LLRs and masked hard bits are zero in every row.
        Message slots incident to masked error nodes retain their input
        edge_msgs values unchanged in every row; those slots do not enter
        any BP update. Unmasked message slots contain the updated messages.
        For each fixed-one node, flip its adjacent syndrome bits before
        iteration. Ignore messages to or from masked nodes. Stop at the
        first match to that modified syndrome. Use sign(0)=+1 and set an
        unmasked hard bit to 0 for positive posterior LLR, otherwise to 1.
    """
    return trace

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_masked_bp_trace(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", edge_msgs: "np.ndarray", mask_values: "np.ndarray", max_iters: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int) % 2
    p = np.asarray(p, dtype=float)
    s = np.asarray(syndrome, dtype=int).copy() % 2
    masks = np.asarray(mask_values, dtype=int)
    M, N = H.shape
    edges = [(i, j) for i in range(M) for j in range(N) if H[i, j]]
    edge_index = {edge: k for k, edge in enumerate(edges)}
    row_neighbors = [np.flatnonzero(H[i]).tolist() for i in range(M)]
    col_neighbors = [np.flatnonzero(H[:, j]).tolist() for j in range(N)]
    prior = np.log((1.0 - p) / p)
    active = masks < 0
    for j in np.flatnonzero(masks == 1):
        s ^= H[:, j]
    work = np.asarray(edge_msgs, dtype=float).copy()
    rows = []
    for _ in range(int(max_iters)):
        detector_msgs = np.zeros(len(edges), dtype=float)
        for k, (i, j) in enumerate(edges):
            if not active[j]:
                continue
            incoming = np.asarray([
                work[edge_index[(i, jp)]]
                for jp in row_neighbors[i]
                if jp != j and active[jp]
            ], dtype=float)
            signs = np.where(incoming >= 0.0, 1.0, -1.0)
            detector_msgs[k] = ((-1.0) ** int(s[i])) * float(np.prod(signs)) * float(np.min(np.abs(incoming)))
        new_work = work.copy()
        for k, (i, j) in enumerate(edges):
            if not active[j]:
                continue
            new_work[k] = prior[j] + sum(
                detector_msgs[edge_index[(ip, j)]] for ip in col_neighbors[j] if ip != i
            )
        posterior = np.zeros(N, dtype=float)
        for j in np.flatnonzero(active):
            posterior[j] = prior[j] + sum(
                detector_msgs[edge_index[(i, j)]] for i in col_neighbors[j]
            )
        hard = np.zeros(N, dtype=int)
        hard[active] = (posterior[active] <= 0.0).astype(int)
        success = bool(np.array_equal((H @ hard) % 2, s))
        rows.append(np.concatenate([posterior, hard.astype(float), new_work, [float(success)]]))
        work = new_work
        if success:
            break
    return np.vstack(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    '''Return list of test case specifications.'''
    common = '''import numpy as np\nH=np.array([[0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,0,0],[1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,1,0,0],[0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0],[0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,0],[0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,1,1,1,0],[0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,1],[1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1,0,1,0],[0,1,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,1,0,1],[0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,1,0,0,0,0],[0,1,0,0,1,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0],[0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1]],dtype=int)\np=np.array([0.099321675569,0.055717560406,0.074255307877,0.070190994539,0.151229358583,0.145537983086,0.035252957104,0.085929973968,0.086129312934,0.159482638903,0.153475804526,0.075766077846,0.144920993860,0.135244812128,0.140972421803,0.131736174287,0.114588720979,0.098089955076,0.082017411514,0.155506630793])\n'''
    return [
        {
            "setup": common + '''s=np.array([1,0,1,0,1,1,0,1,0,1,1],dtype=int)\nseed=_oracle_seed_beam_state(H,p,s,3)\nN=H.shape[1];E=int(H.sum());warm=seed[4+N:4+N+E].copy();m=np.full(N,-1);m[int(seed[3])]=0\n''',
            "call": "masked_bp_trace(H.copy(),p.copy(),s.copy(),warm.copy(),m.copy(),4)",
            "gold_call": "_oracle_masked_bp_trace(H.copy(),p.copy(),s.copy(),warm.copy(),m.copy(),4)",
            "tol": 1e-12,
        },
        {
            "setup": common + '''s=np.array([1,0,0,0,0,0,1,0,1,1,1],dtype=int)\nseed=_oracle_seed_beam_state(H,p,s,3)\nN=H.shape[1];E=int(H.sum());warm=seed[4+N:4+N+E].copy();m=np.full(N,-1);m[int(seed[3])]=1\n''',
            "call": "masked_bp_trace(H.copy(),p.copy(),s.copy(),warm.copy(),m.copy(),4)",
            "gold_call": "_oracle_masked_bp_trace(H.copy(),p.copy(),s.copy(),warm.copy(),m.copy(),4)",
            "tol": 1e-12,
        },
        {
            "setup": common + '''s=np.array([1,1,1,0,1,0,1,1,1,0,0],dtype=int)\nseed=_oracle_seed_beam_state(H,p,s,3)\nN=H.shape[1];E=int(H.sum());warm=seed[4+N:4+N+E].copy();m=np.full(N,-1);m[int(seed[3])]=0\n''',
            "call": "masked_bp_trace(H.copy(),p.copy(),s.copy(),warm.copy(),m.copy(),3)",
            "gold_call": "_oracle_masked_bp_trace(H.copy(),p.copy(),s.copy(),warm.copy(),m.copy(),3)",
            "tol": 1e-12,
        },
    ]
