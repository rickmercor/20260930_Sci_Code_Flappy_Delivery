"""
Generate the sparse randomized embedding used by the adaptive Krylov computation.

Randomized sketching projects high-dimensional vectors into a smaller space while approximately preserving the information needed by the numerical computation. This step constructs the sparse random embedding used by the benchmark.

Returns
-------
float, Frobenius norm of S as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sparse_sign_sketch(
    state: dict,
    seed: int,
    rows: int,
) -> float:
    """Generate a deterministic sparse-sign sketch.

    Parameters
    ----------
    state : dict
        Mutable pipeline state containing the ambient dimension.
    seed : int
        Random seed for the sketch stream.
    rows : int
        Number of sketch rows.

    Returns
    -------
    float
        Frobenius norm of the sketch.
    """
    result = 0.0
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _draw_sparse_sign(
    rng: np.random.Generator,
    rows: int,
    cols: int,
) -> np.ndarray:
    zeta = min(rows, 8)
    S = np.zeros((rows, cols), dtype=float)

    for j in range(cols):
        idx = rng.choice(rows, size=zeta, replace=False)
        signs = rng.choice(np.array([-1.0, 1.0]), size=zeta)
        S[idx, j] = signs / np.sqrt(zeta)

    return S


def _oracle_sparse_sign_sketch(
    state: dict,
    seed: int,
    rows: int,
) -> float:
    import numpy as np

    if not isinstance(state, dict):
        raise ValueError("state must be a dict")
    if "N" not in state:
        raise ValueError("operator must be constructed first")
    if not (isinstance(rows, (int, np.integer)) and rows >= 1):
        raise ValueError("rows must be positive")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(int(seed))
    S = _draw_sparse_sign(rng, rows, int(state["N"]))

    state["rng"] = rng
    state["S"] = S
    state["sketch_rows"] = rows
    state["sketch_seed"] = int(seed)

    return float(np.linalg.norm(S, ord="fro"))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
state = {"N": 9}
seed = 2026
rows = 4
""",
            "call": "sparse_sign_sketch(state, seed, rows)",
            "gold_call": "_oracle_sparse_sign_sketch(state, seed, rows)",
        },
        {
            "setup": """import numpy as np
state = {"N": 1}
seed = 0
rows = 1
""",
            "call": "sparse_sign_sketch(state, seed, rows)",
            "gold_call": "_oracle_sparse_sign_sketch(state, seed, rows)",
        },
        {
            "setup": """import numpy as np
state = {"N": 12}
seed = 7
rows = 30
""",
            "call": "sparse_sign_sketch(state, seed, rows)",
            "gold_call": "_oracle_sparse_sign_sketch(state, seed, rows)",
        },
    ]
