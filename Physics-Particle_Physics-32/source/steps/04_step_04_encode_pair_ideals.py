"""
Convert accepted pairs into denominator-factor exponent tensors.

This converts polynomial pair constraints into the monomial-in-D_i representation required by the simplified ideal-intersection procedure. Each accepted pair becomes two exponent vectors, one for each denominator factor generating that pair ideal.

Returns
-------
Return an integer NumPy array of shape (a, 2, factor_count), where a is the number of accepted pairs. Raise ValueError for malformed arrays, nonbinary flags, incompatible lengths, invalid factor indices, an invalid factor count, or an empty accepted set.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def encode_accepted_pair_ideals(
    pair_flags: np.ndarray,
    pairs: np.ndarray,
    factor_count: int,
) -> np.ndarray:
    """Encode accepted factor pairs as exponent tensors.

    Parameters
    ----------
    pair_flags : np.ndarray
        Binary integer vector of length m.
    pairs : np.ndarray
        Integer array of shape (m,2) containing factor-index pairs.
    factor_count : int
        Total number of denominator factors.

    Returns
    -------
    np.ndarray
        Integer exponent array of shape (a,2,factor_count), where a is
        the number of accepted pairs.

    Raises
    ------
    ValueError
        If inputs are malformed, flags are nonbinary, pair indices are
        invalid, or no pair is accepted.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_encode_accepted_pair_ideals(
    pair_flags,
    pairs,
    factor_count,
):
    flags = np.asarray(pair_flags)
    pairs = np.asarray(pairs)

    if (
        flags.ndim != 1
        or flags.size == 0
        or not np.issubdtype(flags.dtype, np.integer)
    ):
        raise ValueError(
            "pair_flags must be a nonempty integer vector"
        )

    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or pairs.size == 0
        or not np.issubdtype(pairs.dtype, np.integer)
    ):
        raise ValueError("pairs must be an integer (m,2) array")

    if len(flags) != len(pairs):
        raise ValueError("pair_flags and pairs must have equal length")

    if np.any((flags != 0) & (flags != 1)):
        raise ValueError("pair_flags must contain only 0 or 1")

    if (
        isinstance(factor_count, (bool, np.bool_))
        or not isinstance(factor_count, (int, np.integer))
    ):
        raise ValueError("factor_count must be an integer")

    factor_count = int(factor_count)

    if factor_count < 1:
        raise ValueError("factor_count must be positive")

    accepted = []

    for flag, pair in zip(flags, pairs):
        left = int(pair[0])
        right = int(pair[1])

        if (
            left == right
            or left < 0
            or right < 0
            or left >= factor_count
            or right >= factor_count
        ):
            raise ValueError(
                "pair indices must be distinct and in range"
            )

        if int(flag) == 0:
            continue

        exponent_tensor = np.zeros(
            (2, factor_count),
            dtype=np.int64,
        )
        exponent_tensor[0, left] = 1
        exponent_tensor[1, right] = 1
        accepted.append(exponent_tensor)

    if not accepted:
        raise ValueError("at least one pair must be accepted")

    return np.stack(accepted, axis=0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Compare results from independent equivalent input objects."""
    return [{'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'flags=np.array([1,1,0,1],dtype=np.int64)\n'
               'pairs=np.array([[0,1],[1,2],[0,3],[2,3]],dtype=np.int64)',
      'call': 'encode_accepted_pair_ideals(*deepcopy((flags, pairs, 4)))',
      'gold_call': '_oracle_encode_accepted_pair_ideals(*deepcopy((flags, pairs, 4)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'flags=np.array([1],dtype=np.int64)\n'
               'pairs=np.array([[0,1]],dtype=np.int64)',
      'call': 'encode_accepted_pair_ideals(*deepcopy((flags, pairs, 2)))',
      'gold_call': '_oracle_encode_accepted_pair_ideals(*deepcopy((flags, pairs, 2)))'},
     {'setup': 'import numpy as np\n'
               'from copy import deepcopy\n'
               'def _expect_value_error(function, arguments):\n'
               '    try:\n'
               '        function(*arguments)\n'
               '    except ValueError:\n'
               '        return 1\n'
               "    raise AssertionError('Expected ValueError')\n"
               'flags=np.array([0],dtype=np.int64)\n'
               'pairs=np.array([[0,1]],dtype=np.int64)\n'
               'arguments=(flags,pairs,2)',
      'call': '_expect_value_error(encode_accepted_pair_ideals, deepcopy(arguments))',
      'gold_call': '_expect_value_error(_oracle_encode_accepted_pair_ideals, deepcopy(arguments))'}]
