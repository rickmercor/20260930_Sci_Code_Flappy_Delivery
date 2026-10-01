"""
This local branch changes both channel width and sequence resolution before it

returns to the encoder shape. Reconstruct the source's two-convolution topology

from the main prompt and paper: activation placement, the order of pooling versus

the second convolution, and restoration before the residual are not commutative.

The declared same-padding rule also governs short boundary cases. The completed

branch uses the framework's DeepNet residual normalization.

Inputs

------

hidden: attention-residual representation of shape (n_samples, n_segments, width, d)

bottleneck_factor: positive channel expansion factor

Returns

-------

encoded_hidden: locally aggregated and normalized representation

Returns
-------
np.ndarray with the same shape as hidden, the CNN-residual encoder representation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def apply_cnn_bottleneck(
    hidden: "np.ndarray",
    encoder_depth: int = 1,
    bottleneck_factor: float = 2.0,
    kernel_size: int = 3,
) -> "np.ndarray":
    """Apply the source-defined convolutional bottleneck and residual branch.

    Parameters
    ----------
    hidden : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, d)`` with an
        even token width.
    encoder_depth : int
        Positive encoder depth used in residual scaling.
    bottleneck_factor : float
        Finite positive channel multiplier for which
        ``int(d * bottleneck_factor)`` is at least one.
    kernel_size : int
        Positive odd convolution kernel size.

    Raises
    ------
    ValueError
        If hidden has the wrong shape, non-finite entries, or odd width; if
        encoder_depth is not a positive integer; if bottleneck_factor is not
        finite and positive or gives zero channels; or if kernel_size is not a
        positive odd integer.

    Returns
    -------
    encoded_hidden : np.ndarray
        float64 array with the same shape as hidden.
    """
    return encoded_hidden  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_apply_cnn_bottleneck(
    hidden: "np.ndarray",
    encoder_depth: int = 1,
    bottleneck_factor: float = 2.0,
    kernel_size: int = 3,
) -> "np.ndarray":
    """Reference implementation."""
    hidden = np.asarray(hidden, dtype=float)
    if hidden.ndim != 4 or 0 in hidden.shape:
        raise ValueError("hidden must have shape (n_samples, n_segments, width, d)")
    if not np.all(np.isfinite(hidden)):
        raise ValueError("hidden must be finite")
    if hidden.shape[2] % 2 != 0:
        raise ValueError("hidden token width must be even")
    if (
        not isinstance(encoder_depth, (int, np.integer))
        or isinstance(encoder_depth, (bool, np.bool_))
        or encoder_depth <= 0
    ):
        raise ValueError("encoder_depth must be a positive integer")
    if not np.isfinite(bottleneck_factor) or bottleneck_factor <= 0:
        raise ValueError("bottleneck_factor must be finite and positive")
    bottleneck_dim = int(hidden.shape[-1] * bottleneck_factor)
    if bottleneck_dim < 1:
        raise ValueError("bottleneck_factor gives zero channels")
    if (
        not isinstance(kernel_size, (int, np.integer))
        or isinstance(kernel_size, (bool, np.bool_))
        or kernel_size <= 0
        or kernel_size % 2 == 0
    ):
        raise ValueError("kernel_size must be a positive odd integer")

    def _same_cross_correlation(values, kernels):
        padding = kernels.shape[-1] // 2
        padded = np.pad(values, ((0, 0), (0, 0), (padding, padding), (0, 0)))
        output = np.zeros(values.shape[:-1] + (kernels.shape[0],), dtype=float)
        for kernel_index in range(kernels.shape[-1]):
            slab = padded[:, :, kernel_index : kernel_index + values.shape[2], :]
            output += np.einsum("...i,oi->...o", slab, kernels[:, :, kernel_index])
        return output

    input_dim = hidden.shape[-1]
    out1 = np.arange(1.0, bottleneck_dim + 1.0)[:, None, None]
    in1 = np.arange(1.0, input_dim + 1.0)[None, :, None]
    tap = np.arange(1.0, kernel_size + 1.0)[None, None, :]
    kernel1 = 0.12 * np.sin(out1 + in1 * tap)
    expanded = np.maximum(_same_cross_correlation(hidden, kernel1), 0.0)
    pooled = expanded.reshape(
        expanded.shape[0],
        expanded.shape[1],
        expanded.shape[2] // 2,
        2,
        expanded.shape[3],
    ).max(axis=3)

    out2 = np.arange(1.0, input_dim + 1.0)[:, None, None]
    in2 = np.arange(1.0, bottleneck_dim + 1.0)[None, :, None]
    kernel2 = 0.10 * np.cos(out2 * in2 + tap)
    decoded_half = np.maximum(_same_cross_correlation(pooled, kernel2), 0.0)
    decoded = np.repeat(decoded_half, 2, axis=2)

    alpha = (2.0 * encoder_depth) ** 0.25
    residual = alpha * hidden + decoded
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
hidden = np.linspace(-1.2, 1.3, 1 * 1 * 6 * 4).reshape(1, 1, 6, 4)
encoder_depth = 1
bottleneck_factor = 2.0
kernel_size = 3
""",
            "call": "apply_cnn_bottleneck(hidden, encoder_depth, bottleneck_factor, kernel_size).tolist()",
            "gold_call": "_oracle_apply_cnn_bottleneck(hidden, encoder_depth, bottleneck_factor, kernel_size).tolist()",
        },
        {
            "setup": """import numpy as np
hidden = np.array([[[[-1.0, 1.0], [0.0, 0.5]]]])
encoder_depth = 2
bottleneck_factor = 0.5
kernel_size = 1
""",
            "call": "apply_cnn_bottleneck(hidden, encoder_depth, bottleneck_factor, kernel_size).tolist()",
            "gold_call": "_oracle_apply_cnn_bottleneck(hidden, encoder_depth, bottleneck_factor, kernel_size).tolist()",
        },
        {
            "setup": """import numpy as np
hidden = np.zeros((1, 1, 3, 2))
def run_model():
    try:
        apply_cnn_bottleneck(hidden)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_apply_cnn_bottleneck(hidden)
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
