"""
Run the alternating neighborhood expansion for a fixed number of rounds.

Growth from an $A$ seed and growth from a $B$ seed are asymmetric recurrences. One side is threshold-selected and the opposite side is enlarged by ordinary graph adjacency; the process is monotone on the side being accumulated.

Returns
-------
tuple[list[int], list[int]], final ascending A-region and B-region after exactly rounds updates
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def grow_beta_region(
    H: "np.ndarray",
    seed: list[int],
    seed_side: str,
    beta: float,
    rounds: int,
) -> tuple[list[int], list[int]]:
    """Run the source-defined alternating beta-region growth.

    Returns
    -------
    result : tuple[list[int], list[int]]
        Final ascending A-side region and B-side region.
    """
    return [], []

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_grow_beta_region(
    H: np.ndarray,
    seed: list[int],
    seed_side: str,
    beta: float,
    rounds: int,
) -> tuple[list[int], list[int]]:
    import numpy as np

    H = np.asarray(H)
    if H.ndim != 2 or H.size == 0:
        raise ValueError("H must be a nonempty 2D matrix")
    if not np.all((H == 0) | (H == 1)):
        raise ValueError("H must be binary")
    if seed_side not in ("A", "B"):
        raise ValueError("seed_side must be 'A' or 'B'")
    if not (0.0 <= float(beta) < 1.0):
        raise ValueError("beta must lie in [0,1)")
    if not isinstance(rounds, (int, np.integer)) or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer")

    seed = [int(v) for v in seed]
    if len(seed) != len(set(seed)):
        raise ValueError("seed indices must be unique")

    if seed_side == "A":
        if any(v < 0 or v >= H.shape[1] for v in seed):
            raise ValueError("A seed out of range")
        A_region = set(seed)
        B_region = set()

        for _ in range(rounds):
            B_region = set(
                _oracle_beta_neighborhood(H, sorted(A_region), "A", beta)
            )
            if B_region:
                A_region |= set(
                    np.flatnonzero(
                        np.any(H[sorted(B_region), :], axis=0)
                    ).astype(int).tolist()
                )
    else:
        if any(v < 0 or v >= H.shape[0] for v in seed):
            raise ValueError("B seed out of range")
        B_region = set(seed)
        A_region = set()

        for _ in range(rounds):
            A_region = set(
                _oracle_beta_neighborhood(H, sorted(B_region), "B", beta)
            )
            if A_region:
                B_region |= set(
                    np.flatnonzero(
                        np.any(H[:, sorted(A_region)], axis=1)
                    ).astype(int).tolist()
                )

    return sorted(A_region), sorted(B_region)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
def _pack(result):
    A, B = result
    return np.asarray([len(A), *A, len(B), *B], dtype=int)
H_candidate = np.array([[1,1,0],[0,1,1]], dtype=int)
H_oracle = H_candidate.copy()
seed_candidate = [0]
seed_oracle = seed_candidate.copy()
""",
            "call": "_pack(grow_beta_region(H_candidate, seed_candidate, 'A', 0.5, 2))",
            "gold_call": "_pack(_oracle_grow_beta_region(H_oracle, seed_oracle, 'A', 0.5, 2))",
        },
        {
            "setup": """import numpy as np
def _pack(result):
    A, B = result
    return np.asarray([len(A), *A, len(B), *B], dtype=int)
H_candidate = np.array([[1,1,0],[0,1,1]], dtype=int)
H_oracle = H_candidate.copy()
seed_candidate = [0,2]
seed_oracle = seed_candidate.copy()
""",
            "call": "_pack(grow_beta_region(H_candidate, seed_candidate, 'A', 0.5, 0))",
            "gold_call": "_pack(_oracle_grow_beta_region(H_oracle, seed_oracle, 'A', 0.5, 0))",
        },
        {
            "setup": """import numpy as np
def _pack(result):
    A, B = result
    return np.asarray([len(A), *A, len(B), *B], dtype=int)
H_candidate = np.array([[1,1,0],[0,1,1]], dtype=int)
H_oracle = H_candidate.copy()
seed_candidate = [1]
seed_oracle = seed_candidate.copy()
""",
            "call": "_pack(grow_beta_region(H_candidate, seed_candidate, 'B', 0.5, 1))",
            "gold_call": "_pack(_oracle_grow_beta_region(H_oracle, seed_oracle, 'B', 0.5, 1))",
        },
    ]
