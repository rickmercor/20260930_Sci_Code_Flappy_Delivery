"""
Position-free content attention cannot distinguish token order. This step

reproduces the framework's late-combination rotary projection using the frozen

token and QKV maps in the main prompt. The source-specific channel pairing and

phase orientation matter at nonzero positions: the correct transform leaves the

position-zero query and key unchanged, preserves their per-token Euclidean norms,

and does not rotate values.

Inputs

------

segments: packed segment array containing observed token states

embedding_dim: positive even hidden dimension

Returns

-------

projected: array ending in axes (embedding, rotary query, rotary key, value) and hidden coordinate

Returns
-------
np.ndarray ending in shape (width, 4, embedding_dim), normalized X and projected Q, K, V
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def project_rotary_qkv(
    segments: "np.ndarray", embedding_dim: int = 4, rope_base: float = 10000.0
) -> "np.ndarray":
    """Form deterministic Q, K, and V arrays with the source's rotary rule.

    Parameters
    ----------
    segments : np.ndarray
        Finite packed segment array of shape
        ``(n_samples, n_segments, width, 4)`` with integral token states from
        zero through seven in its first channel.
    embedding_dim : int
        Positive even hidden dimension.
    rope_base : float
        Finite base greater than one for the source-defined rotary frequencies.

    Raises
    ------
    ValueError
        If segments has the wrong shape or invalid token states, embedding_dim
        is not a positive even integer, or rope_base is not finite and greater
        than one.

    Returns
    -------
    projected : np.ndarray
        float64 array of shape
        ``(n_samples, n_segments, width, 4, embedding_dim)`` ordered as the
        normalized embedding, rotary query, rotary key, and value.
    """
    return projected  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_project_rotary_qkv(
    segments: "np.ndarray", embedding_dim: int = 4, rope_base: float = 10000.0
) -> "np.ndarray":
    """Reference implementation."""
    segments = np.asarray(segments, dtype=float)
    if segments.ndim != 4 or segments.shape[-1] != 4 or 0 in segments.shape[:3]:
        raise ValueError("segments must have shape (n_samples, n_segments, width, 4)")
    if not np.all(np.isfinite(segments)):
        raise ValueError("segments must be finite")
    tokens = segments[..., 0]
    if (
        not np.all(tokens == np.floor(tokens))
        or np.any(tokens < 0)
        or np.any(tokens > 7)
    ):
        raise ValueError("token states must be integers from 0 through 7")
    if (
        not isinstance(embedding_dim, (int, np.integer))
        or isinstance(embedding_dim, (bool, np.bool_))
        or embedding_dim <= 0
        or embedding_dim % 2 != 0
    ):
        raise ValueError("embedding_dim must be a positive even integer")
    if not np.isfinite(rope_base) or rope_base <= 1.0:
        raise ValueError("rope_base must be finite and greater than one")

    token_axis = np.arange(1.0, 9.0)[:, None]
    hidden_axis = np.arange(1.0, embedding_dim + 1.0)[None, :]
    table = 0.45 * np.sin(token_axis * hidden_axis) + 0.15 * np.cos(
        (token_axis + 1.0) * hidden_axis
    )
    table[7] = 0.0
    embedded = table[tokens.astype(np.int64)]
    mean = embedded.mean(axis=-1, keepdims=True)
    variance = ((embedded - mean) ** 2).mean(axis=-1, keepdims=True)
    embedded = (embedded - mean) / np.sqrt(variance + 1e-5)

    row = np.arange(1.0, embedding_dim + 1.0)[:, None]
    col = np.arange(1.0, embedding_dim + 1.0)[None, :]
    wq = 0.35 * np.sin(row * col + 0.2)
    wk = 0.30 * np.cos(row * (col + 1.0) - 0.1)
    wv = 0.40 * np.sin((row + 1.0) * col + 0.3)
    query = embedded @ wq
    key = embedded @ wk
    value = embedded @ wv

    pair_index = np.arange(embedding_dim // 2, dtype=float)
    frequencies = rope_base ** (-2.0 * pair_index / embedding_dim)
    ordinal_position = np.arange(segments.shape[2], dtype=float)
    angles = ordinal_position[:, None] * frequencies[None, :]
    cosine = np.cos(angles)[None, None, :, :]
    sine = np.sin(angles)[None, None, :, :]

    def _rotate(values):
        even = values[..., 0::2]
        odd = values[..., 1::2]
        output = np.empty_like(values)
        output[..., 0::2] = even * cosine - odd * sine
        output[..., 1::2] = even * sine + odd * cosine
        return output

    return np.stack((embedded, _rotate(query), _rotate(key), value), axis=-2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
segments = np.array(
    [
        [
            [
                [5, 5, 9, -1],
                [1, 1, 10, 0],
                [0, 2, 15, 1],
                [3, 3, 20, 2],
                [6, 6, 21, -1],
                [7, 7, 0, -1],
            ]
        ]
    ],
    dtype=float,
)
embedding_dim = 4
rope_base = 10000.0
""",
            "call": "project_rotary_qkv(segments, embedding_dim, rope_base).tolist()",
            "gold_call": "_oracle_project_rotary_qkv(segments, embedding_dim, rope_base).tolist()",
        },
        {
            "setup": """import numpy as np
segments = np.array([[[[0, 4, 1, 0]]]], dtype=float)
embedding_dim = 2
rope_base = 2.0
""",
            "call": "project_rotary_qkv(segments, embedding_dim, rope_base).tolist()",
            "gold_call": "_oracle_project_rotary_qkv(segments, embedding_dim, rope_base).tolist()",
        },
        {
            "setup": """import numpy as np
segments = np.array(
    [
        [
            [
                [5, 5, 9, -1],
                [1, 1, 10, 0],
                [0, 2, 15, 1],
                [3, 3, 20, 2],
                [6, 6, 21, -1],
                [7, 7, 0, -1],
            ]
        ]
    ],
    dtype=float,
)
def run_model():
    try:
        project_rotary_qkv(segments, embedding_dim=3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_project_rotary_qkv(segments, embedding_dim=3)
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
