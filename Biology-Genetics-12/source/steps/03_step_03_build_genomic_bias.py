"""
Relative genomic positional bias uses actual SNP coordinates rather than only

ordinal token offsets. Within each segment, the virtual CLS and SEP coordinates

bound an affine normalization to zero and one; PAD positions stay zero and are

marked invalid. The resulting bias vector preserves irregular physical spacing

while remaining comparable across genomic windows.

Inputs

------

segments: packed segment array of shape (n_samples, n_segments, width, 4)

Returns

-------

bias_pack: array of shape (n_samples, n_segments, width, 2) containing bias and validity

Returns
-------
np.ndarray of shape (n_samples, n_segments, width, 2), normalized bias and validity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_genomic_bias(segments: "np.ndarray") -> "np.ndarray":
    """Normalize genomic coordinates and mark non-padding tokens.

    Parameters
    ----------
    segments : np.ndarray
        Packed segment array whose last axis contains observed token, target
        token, physical coordinate, and source variant index.

    Raises
    ------
    ValueError
        If segments is not a finite four-dimensional array with last dimension
        four, if token states are non-integral or outside 0 through 7, if a
        segment lacks one CLS and one SEP token in that order, if padding is not
        trailing, or if non-padding coordinates are not strictly increasing.

    Returns
    -------
    bias_pack : np.ndarray
        float64 array with normalized coordinate in channel 0 and a binary
        non-padding indicator in channel 1.
    """
    return bias_pack  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_genomic_bias(segments: "np.ndarray") -> "np.ndarray":
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
        raise ValueError("segment tokens must be integers from 0 through 7")

    result = np.zeros(segments.shape[:-1] + (2,), dtype=np.float64)
    for sample_index in range(segments.shape[0]):
        for segment_index in range(segments.shape[1]):
            row_tokens = tokens[sample_index, segment_index].astype(int)
            cls_indices = np.flatnonzero(row_tokens == 5)
            sep_indices = np.flatnonzero(row_tokens == 6)
            if (
                cls_indices.tolist() != [0]
                or len(sep_indices) != 1
                or sep_indices[0] <= 0
            ):
                raise ValueError(
                    "each segment must begin with one CLS and contain one later SEP"
                )
            sep_index = int(sep_indices[0])
            valid = row_tokens != 7
            if not np.all(valid[: sep_index + 1]) or np.any(valid[sep_index + 1 :]):
                raise ValueError("padding must be trailing after SEP")
            coordinates = segments[sample_index, segment_index, : sep_index + 1, 2]
            if np.any(np.diff(coordinates) <= 0):
                raise ValueError("non-padding coordinates must be strictly increasing")
            span = coordinates[-1] - coordinates[0]
            bias = (coordinates - coordinates[0]) / span
            result[sample_index, segment_index, : sep_index + 1, 0] = bias
            result[sample_index, segment_index, :, 1] = valid.astype(float)
    return result

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
                [4, 4, 20, 2],
                [6, 6, 21, -1],
                [7, 7, 0, -1],
            ]
        ]
    ],
    dtype=float,
)
""",
            "call": "build_genomic_bias(segments).tolist()",
            "gold_call": "_oracle_build_genomic_bias(segments).tolist()",
        },
        {
            "setup": """import numpy as np
segments = np.array([[[[5, 5, 0, -1], [1, 1, 1, 0],
                              [6, 6, 2, -1]]]], dtype=float)
""",
            "call": "build_genomic_bias(segments).tolist()",
            "gold_call": "_oracle_build_genomic_bias(segments).tolist()",
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
                [4, 4, 20, 2],
                [6, 6, 21, -1],
                [7, 7, 0, -1],
            ]
        ]
    ],
    dtype=float,
)
segments[0, 0, 3], segments[0, 0, 4] = segments[0, 0, 4].copy(), segments[0, 0, 3].copy()
def run_model():
    try:
        build_genomic_bias(segments)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_build_genomic_bias(segments)
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
