"""
Generate the deterministic sequence of sampled column blocks.

Each iteration uses a randomly selected subset of columns. The sampling sequence is fixed by one deterministic NumPy generator, with distinct columns sampled without replacement at each iteration.

Returns
-------
np.ndarray, shape (n_iters, block_size), containing the sequential sampled column indices
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def generate_column_blocks(
    n_cols: int,
    block_size: int,
    n_iters: int,
    seed: int = 271828,
) -> np.ndarray:
    """Generate sequential random column-index blocks.

    Parameters
    ----------
    n_cols : int
        Number of available columns.
    block_size : int
        Number of distinct columns sampled per iteration.
    n_iters : int
        Number of iterations.
    seed : int
        Seed for numpy default_rng.

    Returns
    -------
    np.ndarray
        Integer array of shape (n_iters, block_size).

    Raises
    ------
    ValueError
        If n_cols is not positive, block_size is outside the range
        1 through n_cols, n_iters is not positive, or seed is not an
        integer.

    Notes
    -----
    The result is deterministic for identical inputs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_generate_column_blocks(
    n_cols: int,
    block_size: int,
    n_iters: int,
    seed: int = 271828,
) -> np.ndarray:
    """Reference implementation."""
    import numpy as np

    if not isinstance(n_cols, (int, np.integer)) or n_cols < 1:
        raise ValueError("n_cols must be positive")

    if not isinstance(block_size, (int, np.integer)):
        raise ValueError("block_size must be an integer")

    if block_size < 1 or block_size > n_cols:
        raise ValueError("invalid block_size")

    if not isinstance(n_iters, (int, np.integer)) or n_iters < 1:
        raise ValueError("n_iters must be positive")

    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(int(seed))

    blocks = np.empty(
        (int(n_iters), int(block_size)),
        dtype=np.int64,
    )

    for k in range(int(n_iters)):
        blocks[k] = rng.choice(
            int(n_cols),
            size=int(block_size),
            replace=False,
        )

    return blocks

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
n_cols = 5
block_size = 2
n_iters = 6
seed = 271828
""",
            "call": "generate_column_blocks(n_cols, block_size, n_iters, seed)",
            "gold_call": "_oracle_generate_column_blocks(n_cols, block_size, n_iters, seed)",
        },
        {
            "setup": """import numpy as np
n_cols = 1
block_size = 1
n_iters = 1
seed = 0
""",
            "call": "generate_column_blocks(n_cols, block_size, n_iters, seed)",
            "gold_call": "_oracle_generate_column_blocks(n_cols, block_size, n_iters, seed)",
        },
        {
            "setup": """import numpy as np
n_cols = 4
block_size = 0
n_iters = 3
seed = 1

def catches_value_error(fn):
    try:
        fn(n_cols, block_size, n_iters, seed)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "catches_value_error(generate_column_blocks)",
            "gold_call": "catches_value_error(_oracle_generate_column_blocks)",
        },
    ]
