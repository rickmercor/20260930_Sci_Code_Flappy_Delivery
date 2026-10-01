"""
Actual SNP coordinates enter this head as an ordered query-key displacement,

not as unsigned distance. Under the declared row-query/column-key convention,

the source-defined bias has zero diagonal and changes sign when query and key

are exchanged. Its direction and its point of composition with content attention

are method-defining choices rather than interchangeable implementation details.

Padding keys are excluded and the result passes through the frozen output

projection from the main prompt.

Inputs

------

projected: normalized embeddings and rotary Q, K, V arrays

bias_pack: normalized genomic positions and validity mask

Returns

-------

attention_output: float array with one hidden vector per token

Returns
-------
np.ndarray of shape (n_samples, n_segments, width, d), projected attention output
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_genomic_attention(
    projected: "np.ndarray", bias_pack: "np.ndarray", beta: float = -1.4
) -> "np.ndarray":
    """Apply the source-defined coordinate-aware attention and frozen output map.

    Parameters
    ----------
    projected : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, 4, d)`` ordered
        as embedding, rotary query, rotary key, and value.
    bias_pack : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, 2)`` containing
        normalized positions and binary non-padding indicators.
    beta : float
        Finite head coefficient associated with the directional coordinate prior.

    Raises
    ------
    ValueError
        If projected or bias_pack has the wrong shape, contains non-finite
        values, has a non-binary validity channel, leaves a segment with no
        valid key, or if beta is not finite.

    Returns
    -------
    attention_output : np.ndarray
        float64 array of shape ``(n_samples, n_segments, width, d)``.
    """
    return attention_output  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_apply_genomic_attention(
    projected: "np.ndarray", bias_pack: "np.ndarray", beta: float = -1.4
) -> "np.ndarray":
    """Reference implementation."""
    projected = np.asarray(projected, dtype=float)
    bias_pack = np.asarray(bias_pack, dtype=float)
    if projected.ndim != 5 or projected.shape[-2] != 4 or 0 in projected.shape:
        raise ValueError(
            "projected must have shape (n_samples, n_segments, width, 4, d)"
        )
    if bias_pack.shape != projected.shape[:3] + (2,):
        raise ValueError("bias_pack shape must match projected token axes")
    if not np.all(np.isfinite(projected)) or not np.all(np.isfinite(bias_pack)):
        raise ValueError("projected and bias_pack must be finite")
    valid = bias_pack[..., 1]
    if not np.all((valid == 0.0) | (valid == 1.0)):
        raise ValueError("validity indicators must be binary")
    if np.any(valid.sum(axis=-1) == 0):
        raise ValueError("each segment must have at least one valid key")
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")

    query = projected[..., 1, :]
    key = projected[..., 2, :]
    value = projected[..., 3, :]
    hidden_dim = projected.shape[-1]
    bias = bias_pack[..., 0]
    relative = bias[..., None, :] - bias[..., :, None]
    scores = np.einsum("...id,...jd->...ij", query, key) / np.sqrt(float(hidden_dim))
    scores = scores + float(beta) * relative
    scores = np.where(valid[..., None, :] == 1.0, scores, -np.inf)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    attended = np.einsum("...ij,...jd->...id", weights, value)

    row = np.arange(1.0, hidden_dim + 1.0)[:, None]
    col = np.arange(1.0, hidden_dim + 1.0)[None, :]
    output_projection = 0.25 * np.cos(row * col + 0.4)
    return attended @ output_projection

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
projected = (
    np.arange(1, 1 + 1 * 1 * 4 * 4 * 2, dtype=float).reshape(1, 1, 4, 4, 2) / 13.0
)
bias_pack = np.array([[[[0.0, 1.0], [0.2, 1.0], [1.0, 1.0], [0.0, 0.0]]]])
beta = -1.4
""",
            "call": "apply_genomic_attention(projected, bias_pack, beta).tolist()",
            "gold_call": "_oracle_apply_genomic_attention(projected, bias_pack, beta).tolist()",
        },
        {
            "setup": """import numpy as np
projected = (
    np.arange(1, 1 + 1 * 1 * 4 * 4 * 2, dtype=float).reshape(1, 1, 4, 4, 2) / 13.0
)[:, :, :1].copy()
bias_pack = np.array([[[[0.0, 1.0], [0.2, 1.0], [1.0, 1.0], [0.0, 0.0]]]])[:, :, :1].copy()
beta = 0.0
""",
            "call": "apply_genomic_attention(projected, bias_pack, beta).tolist()",
            "gold_call": "_oracle_apply_genomic_attention(projected, bias_pack, beta).tolist()",
        },
        {
            "setup": """import numpy as np
projected = (
    np.arange(1, 1 + 1 * 1 * 4 * 4 * 2, dtype=float).reshape(1, 1, 4, 4, 2) / 13.0
)
bias_pack = np.array([[[[0.0, 1.0], [0.2, 1.0], [1.0, 1.0], [0.0, 0.0]]]])
bias_pack[..., 1] = 0.0
def run_model():
    try:
        apply_genomic_attention(projected, bias_pack)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_apply_genomic_attention(projected, bias_pack)
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
