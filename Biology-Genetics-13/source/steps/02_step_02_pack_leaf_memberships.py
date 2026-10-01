"""
Shared candidate discovery carries one compatibility bit per update. If carrier matrix entry X_{i,s} is one, bit i mod w is set in word floor(i/w) for leaf s, where w is the selected word width. The resulting q = ceil(k/w) words let one bitwise operation update several mutation states at once.

Inputs

------

carrier_matrix: Binary array of shape (k, n_samples).

word_size: Number of active bits in each packed word.

Returns

-------

packed_memberships: Integer array of shape (n_samples, ceil(k / word_size)).

Returns
-------
np.ndarray of shape (n_samples, ceil(k / word_size)), packed membership words as int64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pack_leaf_memberships(carrier_matrix: np.ndarray, word_size: int = 32) -> np.ndarray:
    """Pack per-update leaf memberships into fixed-width words.

    Parameters
    ----------
    carrier_matrix : np.ndarray
        Binary carrier matrix with shape (k, n_samples).
    word_size : int, optional
        Number of update bits assigned to each word.

    Raises
    ------
    ValueError
        If `carrier_matrix` is empty, is not two dimensional, or is not binary,
        or if `word_size` is outside [1, 62].

    Returns
    -------
    packed_memberships : np.ndarray
        int64 array with one row per sample and one column per word.
    """
    return packed_memberships  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_pack_leaf_memberships(carrier_matrix: np.ndarray, word_size: int = 32) -> np.ndarray:
    """Reference implementation."""
    carriers = np.asarray(carrier_matrix)
    if carriers.ndim != 2 or carriers.shape[0] == 0 or carriers.shape[1] == 0:
        raise ValueError("carrier_matrix must be a nonempty two-dimensional array")
    if not np.all((carriers == 0) | (carriers == 1)):
        raise ValueError("carrier_matrix must be binary")
    if not isinstance(word_size, (int, np.integer)) or not 1 <= int(word_size) <= 62:
        raise ValueError("word_size must be an integer in [1, 62]")

    word_size = int(word_size)
    n_updates, n_samples = carriers.shape
    n_words = (n_updates + word_size - 1) // word_size
    packed = np.zeros((n_samples, n_words), dtype=np.int64)
    for update in range(n_updates):
        word = update // word_size
        offset = update % word_size
        packed[:, word] |= carriers[update].astype(np.int64) << offset
    return packed

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
carriers = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)
word_size = 4
""",
            "call": "pack_leaf_memberships(carriers, word_size=word_size).tolist()",
            "gold_call": "_oracle_pack_leaf_memberships(carriers, word_size=word_size).tolist()",
        },
        {
            "setup": """import numpy as np
carriers = np.array([[1]], dtype=np.uint8)
word_size = 1
""",
            "call": "pack_leaf_memberships(carriers, word_size=word_size).tolist()",
            "gold_call": "_oracle_pack_leaf_memberships(carriers, word_size=word_size).tolist()",
        },
        {
            "setup": """import numpy as np
carriers = np.array([[1, 2]], dtype=int)
def run_model():
    try:
        pack_leaf_memberships(carriers, word_size=4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_pack_leaf_memberships(carriers, word_size=4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
