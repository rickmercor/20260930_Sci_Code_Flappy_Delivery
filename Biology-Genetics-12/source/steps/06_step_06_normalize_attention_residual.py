"""
DeepNet residual scaling stabilizes the GenoBERT encoder by multiplying the

incoming representation by alpha = (2N)^(1/4), where N is encoder depth, before

adding a sublayer output. Post-residual layer normalization is performed across

the hidden coordinates of each token with population variance and a fixed

epsilon. This step applies that operation to the attention branch.

Inputs

------

projected: array containing the normalized token embedding in slot zero

attention_output: output of the relative-bias attention sublayer

Returns

-------

hidden: normalized attention-residual representation

Returns
-------
np.ndarray of shape (n_samples, n_segments, width, d), normalized attention residual
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normalize_attention_residual(
    projected: "np.ndarray", attention_output: "np.ndarray", encoder_depth: int = 1
) -> "np.ndarray":
    """Apply DeepNet scaling, an attention residual, and token-wise layer normalization.

    Parameters
    ----------
    projected : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, 4, d)`` whose
        slot zero is the input embedding.
    attention_output : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, d)``.
    encoder_depth : int
        Positive encoder depth used in the residual scaling factor.

    Raises
    ------
    ValueError
        If projected or attention_output has the wrong shape or non-finite
        entries, or if encoder_depth is not a positive integer.

    Returns
    -------
    hidden : np.ndarray
        float64 array of shape ``(n_samples, n_segments, width, d)``.
    """
    return hidden  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_normalize_attention_residual(
    projected: "np.ndarray", attention_output: "np.ndarray", encoder_depth: int = 1
) -> "np.ndarray":
    """Reference implementation."""
    projected = np.asarray(projected, dtype=float)
    attention_output = np.asarray(attention_output, dtype=float)
    if projected.ndim != 5 or projected.shape[-2] != 4 or 0 in projected.shape:
        raise ValueError(
            "projected must have shape (n_samples, n_segments, width, 4, d)"
        )
    if attention_output.shape != projected.shape[:3] + (projected.shape[-1],):
        raise ValueError(
            "attention_output shape must match projected token and hidden axes"
        )
    if not np.all(np.isfinite(projected)) or not np.all(np.isfinite(attention_output)):
        raise ValueError("projected and attention_output must be finite")
    if (
        not isinstance(encoder_depth, (int, np.integer))
        or isinstance(encoder_depth, (bool, np.bool_))
        or encoder_depth <= 0
    ):
        raise ValueError("encoder_depth must be a positive integer")
    alpha = (2.0 * encoder_depth) ** 0.25
    residual = alpha * projected[..., 0, :] + attention_output
    mean = residual.mean(axis=-1, keepdims=True)
    variance = ((residual - mean) ** 2).mean(axis=-1, keepdims=True)
    return (residual - mean) / np.sqrt(variance + 1e-5)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
projected = (
    np.arange(1, 1 + 1 * 1 * 3 * 4 * 4, dtype=float).reshape(1, 1, 3, 4, 4) / 19.0
)
attention_output = np.array(
    [[[[0.2, -0.1, 0.4, 0.0], [0.3, 0.2, -0.2, 0.1], [-0.1, 0.5, 0.2, 0.3]]]]
)
encoder_depth = 6
""",
            "call": "normalize_attention_residual(projected, attention_output, encoder_depth).tolist()",
            "gold_call": "_oracle_normalize_attention_residual(projected, attention_output, encoder_depth).tolist()",
        },
        {
            "setup": """import numpy as np
projected = (
    np.arange(1, 1 + 1 * 1 * 3 * 4 * 4, dtype=float).reshape(1, 1, 3, 4, 4) / 19.0
)[:, :, :1].copy()
attention_output = np.array(
    [[[[0.2, -0.1, 0.4, 0.0], [0.3, 0.2, -0.2, 0.1], [-0.1, 0.5, 0.2, 0.3]]]]
)[:, :, :1].copy()
encoder_depth = 1
""",
            "call": "normalize_attention_residual(projected, attention_output, encoder_depth).tolist()",
            "gold_call": "_oracle_normalize_attention_residual(projected, attention_output, encoder_depth).tolist()",
        },
        {
            "setup": """import numpy as np
projected = (
    np.arange(1, 1 + 1 * 1 * 3 * 4 * 4, dtype=float).reshape(1, 1, 3, 4, 4) / 19.0
)
attention_output = np.array(
    [[[[0.2, -0.1, 0.4, 0.0], [0.3, 0.2, -0.2, 0.1], [-0.1, 0.5, 0.2, 0.3]]]]
)
def run_model():
    try:
        normalize_attention_residual(projected, attention_output, encoder_depth=0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_normalize_attention_residual(projected, attention_output, encoder_depth=0)
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
