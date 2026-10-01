"""
Encode a reduced-alphabet protein sequence as a continuous one-hot state. Each residue-class index becomes one row of a numerical matrix that can subsequently be relaxed and evolved through Hamiltonian dynamics.

The HADES acquisition procedure begins with a discrete protein sequence, whereas Hamiltonian dynamics requires a numerical position variable. For a sequence of length $L$ over $A$ permitted amino-acid classes, construct $q^{init}$ with shape $(L,A)$. Set $q^{init}_{r,c}=1$ when $c=s_r$, and set it to $0$ otherwise. Thus, each sequence position is represented by one one-hot row. All initial coordinates lie in $[0,1]$, and each row sums to $1$. Later Hamiltonian steps may relax these entries into continuous values, while $q^{init}$ remains the reference state for the surrogate’s quadratic penalty.

Returns
-------
np.ndarray, a float64 one-hot matrix of shape (L, alphabet_size)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def encode_sequence(
    sequence: np.ndarray,
    alphabet_size: int,
) -> np.ndarray:
    """Encode integer residue indices as a one-hot matrix.

    Parameters
    ----------
    sequence : np.ndarray
        One-dimensional array of zero-based residue-class indices.
    alphabet_size : int
        Number of allowed residue classes; must be positive.
    Raises
    ------
    ValueError
        If ``sequence`` is empty, not one-dimensional, contains
        non-integer or out-of-range indices, or ``alphabet_size``
        is not a positive integer.

    Returns
    -------
    one_hot : np.ndarray
        Float64 array of shape ``(len(sequence), alphabet_size)``.
    """
    return one_hot

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_encode_sequence(
    sequence: np.ndarray,
    alphabet_size: int,
) -> np.ndarray:
    """Return the deterministic one-hot encoding."""

    sequence_array = np.asarray(sequence)

    if (
        sequence_array.ndim != 1
        or sequence_array.size == 0
    ):
        raise ValueError(
            "sequence must be a non-empty one-dimensional array"
        )

    if (
        isinstance(alphabet_size, (bool, np.bool_))
        or not isinstance(
            alphabet_size,
            (int, np.integer),
        )
    ):
        raise ValueError(
            "alphabet_size must be a positive integer"
        )

    if int(alphabet_size) < 1:
        raise ValueError(
            "alphabet_size must be a positive integer"
        )

    if not np.issubdtype(
        sequence_array.dtype,
        np.integer,
    ):
        raise ValueError(
            "sequence entries must be integers"
        )

    sequence_int = sequence_array.astype(
        np.int64,
        copy=False,
    )

    if (
        np.any(sequence_int < 0)
        or np.any(
            sequence_int >= int(alphabet_size)
        )
    ):
        raise ValueError(
            "sequence entries must lie in [0, alphabet_size)"
        )

    return np.eye(
        int(alphabet_size),
        dtype=np.float64,
    )[sequence_int]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return normal, boundary, and invalid-input test cases."""

    return [
        {
            "setup": """import numpy as np

sequence = np.array([0, 1, 2], dtype=int)
alphabet_size = 3
""",
            "call": "encode_sequence(sequence, alphabet_size).tolist()",
            "gold_call": (
                "_oracle_encode_sequence(sequence, alphabet_size).tolist()"
            ),
        },
        {
            "setup": """import numpy as np

sequence = np.array([0], dtype=int)
alphabet_size = 1
""",
            "call": "encode_sequence(sequence, alphabet_size).tolist()",
            "gold_call": (
                "_oracle_encode_sequence(sequence, alphabet_size).tolist()"
            ),
        },
        {
            "setup": """import numpy as np

sequence = np.array([0, 3], dtype=int)
alphabet_size = 3

def _model_error_code():
    try:
        encode_sequence(sequence, alphabet_size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def _oracle_error_code():
    try:
        _oracle_encode_sequence(sequence, alphabet_size)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "_model_error_code()",
            "gold_call": "_oracle_error_code()",
        },
    ]
