"""
Branch one active path at its selected node, evaluate fixed values zero and one, and compute each child's next branch and reliability score.

The source uses cumulative posterior LLR magnitude for the next unmasked node and normalizes the summed unmasked reliability by the number of masked-BP iterations actually executed. The two children inherit the parent's message state and mask history.

Returns
-------
real `np.ndarray` of shape `(2,4+2*N+E)` containing the two child states in value-0 then value-1 order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def expand_beam_state(
    H: "np.ndarray",
    p: "np.ndarray",
    syndrome: "np.ndarray",
    state: "np.ndarray",
    iters_per_round: int,
) -> "np.ndarray":
    """Return the two child path states in fixed-value order 0 then 1.

    Parameters
    ----------
    H : np.ndarray
        Binary parity-check matrix of shape (M, N).
    p : np.ndarray
        Error probabilities of shape (N,).
    syndrome : np.ndarray
        Original binary syndrome.
    state : np.ndarray
        Parent state in the layout returned by seed_beam_state:
        [success, iters, score, next_pos, mask_values(N), edge_msgs(E),
        correction(N)]. The selected branch node is unmasked.
    iters_per_round : int
        Positive masked-BP iteration cap for each child.

    Returns
    -------
    children : np.ndarray
        Float array of shape (2, 4 + 2*N + E), using the common state
        layout. Each child fixes the parent's next_pos to its branch value
        and starts masked BP from the parent's edge messages. Message slots
        incident to any masked node, including the new branch node, retain
        their parent edge-message values unchanged.
        next_pos is the smallest-index unmasked node minimizing the
        absolute cumulative posterior LLR over the executed iterations.
        score is the sum of those magnitudes over unmasked nodes divided
        by the number of masked-BP iterations actually executed.
        If no unmasked nodes remain, next_pos is -1 and score is zero.
        On success, correction is the full valid correction with all fixed
        mask values restored. On failure, correction contains the final
        masked-BP hard bits, including zero at each masked position.
    """
    return children

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_expand_beam_state(H: "np.ndarray", p: "np.ndarray", syndrome: "np.ndarray", state: "np.ndarray", iters_per_round: int) -> "np.ndarray":
    H = np.asarray(H, dtype=int)
    N = H.shape[1]
    E = int(np.count_nonzero(H))
    parent = np.asarray(state, dtype=float)
    branch_pos = int(round(parent[3]))
    parent_masks = parent[4:4 + N].astype(int)
    parent_edge = parent[4 + N:4 + N + E]
    out = []
    for val in (0, 1):
        masks = parent_masks.copy()
        masks[branch_pos] = val
        trace = _oracle_masked_bp_trace(H, p, syndrome, parent_edge.copy(), masks, int(iters_per_round))
        posterior = trace[:, :N]
        final = trace[-1]
        correction = final[N:2 * N].astype(int)
        success = int(round(final[-1])) == 1
        if success:
            correction[masks >= 0] = masks[masks >= 0]
        edge = final[2 * N:2 * N + E]
        unmasked = np.flatnonzero(masks < 0)
        if unmasked.size:
            cumulative = np.sum(posterior[:, unmasked], axis=0)
            reliability = np.abs(cumulative)
            next_pos = int(unmasked[int(np.argmin(reliability))])
            score = float(np.sum(reliability) / trace.shape[0])
        else:
            next_pos = -1
            score = 0.0
        out.append(np.concatenate([
            [float(success), float(trace.shape[0]), score, float(next_pos)],
            masks.astype(float), edge, correction.astype(float),
        ]))
    return np.vstack(out)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three independent child-expansion cases."""
    common = """import numpy as np

H = np.array(
    [
        [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    ],
    dtype=int,
)
p = np.array(
    [
        0.099321675569, 0.055717560406, 0.074255307877, 0.070190994539,
        0.151229358583, 0.145537983086, 0.035252957104, 0.085929973968,
        0.086129312934, 0.159482638903, 0.153475804526, 0.075766077846,
        0.14492099386, 0.135244812128, 0.140972421803, 0.131736174287,
        0.114588720979, 0.098089955076, 0.082017411514, 0.155506630793,
    ]
)
"""
    return [
        {
            "setup": common + """
s = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1], dtype=int)
st = _oracle_seed_beam_state(
    H.copy(), p.copy(), s.copy(), 3
)
""",
            "call": """
expand_beam_state(
    H.copy(), p.copy(), s.copy(), st.copy(), 4
)
""",
            "gold_call": """
_oracle_expand_beam_state(
    H.copy(), p.copy(), s.copy(), st.copy(), 4
)
""",
            "tol": 1e-12,
        },
        {
            "setup": common + """
s = np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1], dtype=int)
st = _oracle_seed_beam_state(
    H.copy(), p.copy(), s.copy(), 3
)
""",
            "call": """
expand_beam_state(
    H.copy(), p.copy(), s.copy(), st.copy(), 4
)
""",
            "gold_call": """
_oracle_expand_beam_state(
    H.copy(), p.copy(), s.copy(), st.copy(), 4
)
""",
            "tol": 1e-12,
        },
        {
            "setup": common + """
s = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0], dtype=int)

N = H.shape[1]
E = int(H.sum())
seed = _oracle_seed_beam_state(H.copy(), p.copy(), s.copy(), 3)
masks = seed[4:4 + N].astype(int)
masks[int(seed[3])] = 1
warm = seed[4 + N:4 + N + E].copy()
trace = _oracle_masked_bp_trace(
    H.copy(), p.copy(), s.copy(), warm, masks.copy(), 4
)
last = trace[-1]
unmasked = np.flatnonzero(masks < 0)
reliability = np.abs(trace[:, unmasked].sum(axis=0))
next_pos = int(unmasked[np.argmin(reliability)])
score = float(reliability.sum() / trace.shape[0])
correction = last[N:2 * N].copy()
if bool(last[-1]):
    correction[masks >= 0] = masks[masks >= 0]
st2 = np.concatenate(
    [
        [last[-1], float(trace.shape[0]), score, float(next_pos)],
        masks.astype(float),
        last[2 * N:2 * N + E],
        correction,
    ]
)
""",
            "call": """
expand_beam_state(
    H.copy(), p.copy(), s.copy(), st2.copy(), 4
)
""",
            "gold_call": """
_oracle_expand_beam_state(
    H.copy(), p.copy(), s.copy(), st2.copy(), 4
)
""",
            "tol": 1e-12,
        },
    ]
