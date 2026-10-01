"""
Encode Hermitian Pauli words with their complex phases.

An unsigned tensor-product Pauli observable has a binary representation

$$

P=i^p X^{\boldsymbol a}Z^{\boldsymbol b},\qquad p=\sum_{q=0}^{n-1}a_qb_q\pmod 4.

$$

The single-qubit letters $I,X,Y,Z$ correspond to binary pairs $(0,0),(1,0),(1,1),(0,1)$, respectively. In particular, $Y=iXZ$; discarding $p$ preserves commutation but changes some products of commuting observables. Store the leftmost qubit first in each binary block.

Returns
-------
Integer array of shape $(m,2n+1)$ with rows $(p,a_0,\ldots,a_{n-1},b_0,\ldots,b_{n-1})$, in input order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def encode_pauli_words(labels: tuple[str, ...]) -> "np.ndarray":
    r"""Encode Hermitian Pauli words with their complex phases.

    Parameters
    ----------
    labels : tuple[str, ...]
        Nonempty tuple of distinct unsigned words over $I,X,Y,Z$, each of the same length $1\le n\le6$; exclude the all-identity word.

    Returns
    -------
    encoded : np.ndarray
        Integer array of shape $(m,2n+1)$ with rows $(p,a_0,\ldots,a_{n-1},b_0,\ldots,b_{n-1})$, in input order.

    Raises
    ------
    ValueError
        If labels are empty, repeated, contain unsupported symbols or the identity, have unequal lengths, or use more than six qubits.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_encode_pauli_words(labels: tuple[str, ...]) -> "np.ndarray":
    if not isinstance(labels, tuple) or not labels:
        raise ValueError("labels must be a nonempty tuple")
    if any(not isinstance(word, str) for word in labels):
        raise ValueError("each label must be a string")
    n = len(labels[0])
    if not 1 <= n <= 6 or len(set(labels)) != len(labels):
        raise ValueError("use distinct words on one to six qubits")
    if any(
        len(word) != n or set(word) - set("IXYZ") or word == "I" * n for word in labels
    ):
        raise ValueError("invalid Pauli word")
    encoded = np.zeros((len(labels), 2 * n + 1), dtype=np.int64)
    for row, word in enumerate(labels):
        encoded[row, 1 : 1 + n] = [letter in "XY" for letter in word]
        encoded[row, 1 + n :] = [letter in "YZ" for letter in word]
        encoded[row, 0] = word.count("Y") % 4
    return encoded

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical cases for this step."""
    return [
        {
            "setup": """import numpy as np
labels = ('XI', 'YY', 'ZY')
""",
            "call": "encode_pauli_words(labels)",
            "gold_call": "_oracle_encode_pauli_words(labels)",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
labels = ('Z',)
""",
            "call": "encode_pauli_words(labels)",
            "gold_call": "_oracle_encode_pauli_words(labels)",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
labels = ('YYYYYY', 'XYZXYZ')
""",
            "call": "encode_pauli_words(labels)",
            "gold_call": "_oracle_encode_pauli_words(labels)",
            "tol": 0,
        },
        {
            "setup": """import numpy as np
labels = ('XI', 'XI')

def _raises_value_error(fn):
    try:
        fn(labels)
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(encode_pauli_words)",
            "gold_call": "_raises_value_error(_oracle_encode_pauli_words)",
            "tol": 0.0,
        },
    ]
