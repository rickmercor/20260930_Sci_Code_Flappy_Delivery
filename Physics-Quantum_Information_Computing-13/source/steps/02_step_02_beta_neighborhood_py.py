"""
Compute one strict fractional neighborhood on either side of the bipartite graph.

A vertex enters a $\beta$-neighborhood only when strictly more than a $\beta$ fraction of all of its incident edges terminate in the supplied source set. This strict inequality is essential at threshold cases.

Returns
-------
list[int], ascending indices in the opposite bipartite side satisfying the strict beta threshold
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def beta_neighborhood(
    H: "np.ndarray",
    vertices: list[int],
    source_side: str,
    beta: float,
) -> list[int]:
    """Return the strict beta-neighborhood on the opposite graph side.

    Parameters
    ----------
    H : np.ndarray
        Binary B-by-A adjacency matrix.
    vertices : list[int]
        Active vertices on source_side.
    source_side : str
        Either "A" or "B".
    beta : float
        Threshold in [0, 1).

    Returns
    -------
    result : list[int]
        Ascending qualifying opposite-side vertex indices.
    """
    return []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_beta_neighborhood(
    H: np.ndarray,
    vertices: list[int],
    source_side: str,
    beta: float,
) -> list[int]:
    import numpy as np

    H = np.asarray(H)
    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")
    if source_side not in ("A", "B"):
        raise ValueError("source_side must be 'A' or 'B'")
    if not (0.0 <= float(beta) < 1.0):
        raise ValueError("beta must lie in [0,1)")
    if np.any(H.sum(axis=0) == 0) or np.any(H.sum(axis=1) == 0):
        raise ValueError("isolated graph vertices are not supported")

    vertices = [int(v) for v in vertices]
    if len(vertices) != len(set(vertices)):
        raise ValueError("vertices must be unique")

    if source_side == "A":
        if any(v < 0 or v >= H.shape[1] for v in vertices):
            raise ValueError("A vertex out of range")
        counts = H[:, vertices].sum(axis=1) if vertices else np.zeros(H.shape[0], dtype=int)
        degrees = H.sum(axis=1)
    else:
        if any(v < 0 or v >= H.shape[0] for v in vertices):
            raise ValueError("B vertex out of range")
        counts = H[vertices, :].sum(axis=0) if vertices else np.zeros(H.shape[1], dtype=int)
        degrees = H.sum(axis=0)

    return np.flatnonzero(counts > float(beta) * degrees).astype(int).tolist()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
H_candidate = np.array([[1,1,0],[0,1,1]], dtype=int)
H_oracle = H_candidate.copy()
v_candidate = [0,1]
v_oracle = v_candidate.copy()
""",
            "call": "np.asarray(beta_neighborhood(H_candidate, v_candidate, 'A', 0.5), dtype=int)",
            "gold_call": "np.asarray(_oracle_beta_neighborhood(H_oracle, v_oracle, 'A', 0.5), dtype=int)",
        },
        {
            "setup": """import numpy as np
H_candidate = np.array([[1,1]], dtype=int)
H_oracle = H_candidate.copy()
v_candidate = [0]
v_oracle = v_candidate.copy()
""",
            "call": "np.asarray(beta_neighborhood(H_candidate, v_candidate, 'A', 0.5), dtype=int)",
            "gold_call": "np.asarray(_oracle_beta_neighborhood(H_oracle, v_oracle, 'A', 0.5), dtype=int)",
        },
        {
            "setup": """import numpy as np
H_candidate = np.eye(2, dtype=int)
H_oracle = H_candidate.copy()
v_candidate = []
v_oracle = []
""",
            "call": "np.asarray(beta_neighborhood(H_candidate, v_candidate, 'B', 0.5), dtype=int)",
            "gold_call": "np.asarray(_oracle_beta_neighborhood(H_oracle, v_oracle, 'B', 0.5), dtype=int)",
        },
    ]
