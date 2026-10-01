"""
Construct the benchmark's uniquely specified peeling and partner set.

A peeling removes an active vertex through a neighbor that is unique with respect to the still-active set. The direction of the uniqueness condition reverses between an $A$-side set and a $B$-side set.

Returns
-------
tuple[list[tuple[int,int]], list[int]], ordered peeling edges plus sorted opposite-side partner indices
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def deterministic_peeling(
    H: "np.ndarray",
    active: list[int],
    active_side: str,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Compute the deterministic lexicographic peeling and partner set."""
    return [], []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_deterministic_peeling(
    H: np.ndarray,
    active: list[int],
    active_side: str,
) -> tuple[list[tuple[int, int]], list[int]]:
    import numpy as np

    H = np.asarray(H)
    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")
    if active_side not in ("A", "B"):
        raise ValueError("active_side must be 'A' or 'B'")

    active = [int(v) for v in active]
    if len(active) != len(set(active)):
        raise ValueError("active indices must be unique")

    limit = H.shape[1] if active_side == "A" else H.shape[0]
    if any(v < 0 or v >= limit for v in active):
        raise ValueError("active vertex out of range")

    remaining = set(active)
    pairs = []

    while remaining:
        if active_side == "A":
            candidates = []
            for b in range(H.shape[0]):
                nbrs = [a for a in remaining if H[b, a] == 1]
                if len(nbrs) == 1:
                    candidates.append((nbrs[0], b))

            if not candidates:
                raise ValueError("active A-set is not peelable")

            a, b = min(candidates)
            remaining.remove(a)

        else:
            candidates = []
            for a in range(H.shape[1]):
                nbrs = [b for b in remaining if H[b, a] == 1]
                if len(nbrs) == 1:
                    candidates.append((nbrs[0], a))

            if not candidates:
                raise ValueError("active B-set is not peelable")

            b, a = min(candidates)
            remaining.remove(b)

        pairs.append((int(a), int(b)))

    if active_side == "A":
        partners = sorted(b for a, b in pairs)
    else:
        partners = sorted(a for a, b in pairs)

    return pairs, partners

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
def _pack(result):
    pairs, partners = result
    P = np.asarray(pairs, dtype=int).reshape((-1,2))
    return np.concatenate([
        np.asarray([P.shape[0], P.shape[1]], dtype=int),
        P.ravel(),
        np.asarray([len(partners)], dtype=int),
        np.asarray(partners, dtype=int),
    ])
H_candidate = np.array([[1,1,0],[0,1,1]], dtype=int)
H_oracle = H_candidate.copy()
active_candidate = [0,2]
active_oracle = active_candidate.copy()
""",
            "call": "_pack(deterministic_peeling(H_candidate, active_candidate, 'A'))",
            "gold_call": "_pack(_oracle_deterministic_peeling(H_oracle, active_oracle, 'A'))",
        },
        {
            "setup": """import numpy as np
def _pack(result):
    pairs, partners = result
    P = np.asarray(pairs, dtype=int).reshape((-1,2))
    return np.concatenate([
        np.asarray([P.shape[0], P.shape[1]], dtype=int),
        P.ravel(),
        np.asarray([len(partners)], dtype=int),
        np.asarray(partners, dtype=int),
    ])
H_candidate = np.eye(2, dtype=int)
H_oracle = H_candidate.copy()
active_candidate = [0,1]
active_oracle = active_candidate.copy()
""",
            "call": "_pack(deterministic_peeling(H_candidate, active_candidate, 'B'))",
            "gold_call": "_pack(_oracle_deterministic_peeling(H_oracle, active_oracle, 'B'))",
        },
        {
            "setup": """import numpy as np
def _pack(result):
    pairs, partners = result
    P = np.asarray(pairs, dtype=int).reshape((-1,2))
    return np.concatenate([
        np.asarray([P.shape[0], P.shape[1]], dtype=int),
        P.ravel(),
        np.asarray([len(partners)], dtype=int),
        np.asarray(partners, dtype=int),
    ])
H_candidate = np.eye(2, dtype=int)
H_oracle = H_candidate.copy()
active_candidate = []
active_oracle = []
""",
            "call": "_pack(deterministic_peeling(H_candidate, active_candidate, 'A'))",
            "gold_call": "_pack(_oracle_deterministic_peeling(H_oracle, active_oracle, 'A'))",
        },
    ]
