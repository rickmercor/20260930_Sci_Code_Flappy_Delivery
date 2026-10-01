"""
Pack a batch of mutation memberships into fixed-width words.

Shared mutation discovery stores one compatibility bit per update at each

active node. Consecutive updates occupy consecutive low-to-high bits in

fixed-width words, so update i uses word floor(i / w) and bit i modulo w. Leaf

words are seeded from replacement carrier membership.

Inputs

Returns
-------
np.ndarray of shape (n_samples, ceil(n_updates / word_size)), packed membership words as int64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pack_batch_memberships(update_carriers: "np.ndarray", word_size: int) -> "np.ndarray":
    '''Pack leaf membership across a mutation-update batch.

    Parameters
    ----------
    update_carriers : np.ndarray
        Binary carrier rows for the batch.
    word_size : int
        Number of usable bits per packed word, from 1 through 62.

    Raises
    ------
    ValueError
        If update_carriers is not a nonempty two-dimensional binary array or
        word_size is not an integer from 1 through 62.

    Returns
    -------
    packed_memberships : np.ndarray
        Packed integer words for each sample leaf.
    '''
    return packed_memberships  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_pack_batch_memberships(update_carriers: "np.ndarray", word_size: int) -> "np.ndarray":
    """Reference implementation."""
    update_carriers = np.asarray(update_carriers)
    if update_carriers.ndim != 2 or min(update_carriers.shape) < 1:
        raise ValueError("update_carriers must be a nonempty two-dimensional array")
    if not np.all((update_carriers == 0) | (update_carriers == 1)):
        raise ValueError("update_carriers must be binary")
    if not isinstance(word_size, int) or isinstance(word_size, bool) or not (1 <= word_size <= 62):
        raise ValueError("word_size must be an integer from 1 through 62")

    update_carriers = update_carriers.astype(np.int64)
    n_updates, n_samples = update_carriers.shape
    n_words = (n_updates + word_size - 1) // word_size
    packed = np.zeros((n_samples, n_words), dtype=np.int64)
    for update in range(n_updates):
        word = update // word_size
        bit = update % word_size
        packed[update_carriers[update] == 1, word] |= np.int64(1 << bit)
    return packed

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
update_carriers = np.array([
    [0,0,0,0,0,0,0,0,1,1,0,0,0,0],
    [1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,0,0],
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,1,1,1,1,1,1,1,0,0],
    [1,1,0,0,0,0,0,0,0,0,0,0,1,0],
], dtype=int)
word_size = 4
""",
            "call": "pack_batch_memberships(update_carriers, word_size).tolist()",
            "gold_call": "_oracle_pack_batch_memberships(update_carriers, word_size).tolist()",
        },
        {
            "setup": """import numpy as np
update_carriers = np.array([[1]], dtype=int)
word_size = 1
""",
            "call": "pack_batch_memberships(update_carriers, word_size).tolist()",
            "gold_call": "_oracle_pack_batch_memberships(update_carriers, word_size).tolist()",
        },
        {
            "setup": """import numpy as np
update_carriers = np.array([[0,2]], dtype=int)
word_size = 2
def run_model():
    try:
        pack_batch_memberships(update_carriers, word_size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_pack_batch_memberships(update_carriers, word_size)
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
