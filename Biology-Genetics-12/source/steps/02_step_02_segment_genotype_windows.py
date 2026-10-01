"""
Genome-wide genotype matrices are wide relative to cohort size, so GenoBERT

recasts each sample as a document and overlapping fixed-SNP windows as segments.

Each segment receives CLS and SEP boundary tokens, and a short terminal segment

is padded to the same width. The packed representation here retains token state,

target state, physical coordinate, and original variant index for later merging.

Inputs

------

encoded: integer array of shape (n_samples, n_variants, 2)

genomic_positions: increasing physical positions of shape (n_variants,)

Returns

-------

segments: float array of shape (n_samples, n_segments, window_size + 2, 4)

Returns
-------
np.ndarray of shape (n_samples, n_segments, window_size + 2, 4), packed segment data
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def segment_genotype_windows(
    encoded: "np.ndarray",
    genomic_positions: "np.ndarray",
    window_size: int = 6,
    overlap: int = 3,
) -> "np.ndarray":
    """Form overlapping genotype segments with CLS, SEP, and PAD tokens.

    Parameters
    ----------
    encoded : np.ndarray
        Integer array of shape ``(n_samples, n_variants, 2)`` containing
        observed and target genotype states.
    genomic_positions : np.ndarray
        Strictly increasing, finite, positive physical coordinates with one
        entry per variant.
    window_size : int
        Positive number of SNP tokens per segment.
    overlap : int
        Number of SNP tokens shared by consecutive segments, in the interval
        ``[0, window_size)``.

    Raises
    ------
    ValueError
        If encoded has the wrong shape or invalid states, genomic_positions is
        not a matching strictly increasing positive vector, window_size is not
        a positive integer, or overlap is not an integer in
        ``[0, window_size)``.

    Returns
    -------
    segments : np.ndarray
        Packed float64 array whose last axis is observed token, target token,
        physical coordinate, and zero-based source variant index.
    """
    return segments  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_segment_genotype_windows(
    encoded: "np.ndarray",
    genomic_positions: "np.ndarray",
    window_size: int = 6,
    overlap: int = 3,
) -> "np.ndarray":
    """Reference implementation."""
    encoded = np.asarray(encoded)
    genomic_positions = np.asarray(genomic_positions, dtype=float)
    if (
        encoded.ndim != 3
        or encoded.shape[2] != 2
        or encoded.shape[0] == 0
        or encoded.shape[1] == 0
    ):
        raise ValueError("encoded must have shape (n_samples, n_variants, 2)")
    if not np.all(encoded == np.floor(encoded)):
        raise ValueError("encoded states must be integers")
    observed = encoded[:, :, 0]
    target = encoded[:, :, 1]
    if not np.all((observed >= 0) & (observed <= 4)) or not np.all(
        (target >= 1) & (target <= 4)
    ):
        raise ValueError("encoded states are outside the genotype vocabulary")
    if not np.all((observed == 0) | (observed == target)):
        raise ValueError("an observed state must equal its target or be MASK")
    if genomic_positions.ndim != 1 or len(genomic_positions) != encoded.shape[1]:
        raise ValueError("genomic_positions must match the variant axis")
    if not np.all(np.isfinite(genomic_positions)) or np.any(genomic_positions <= 0):
        raise ValueError("genomic_positions must be finite and positive")
    if np.any(np.diff(genomic_positions) <= 0):
        raise ValueError("genomic_positions must be strictly increasing")
    if (
        not isinstance(window_size, (int, np.integer))
        or isinstance(window_size, (bool, np.bool_))
        or window_size <= 0
    ):
        raise ValueError("window_size must be a positive integer")
    if not isinstance(overlap, (int, np.integer)) or isinstance(
        overlap, (bool, np.bool_)
    ):
        raise ValueError("overlap must be an integer")  # noqa: TRY004
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be in [0, window_size)")

    stride = window_size - overlap
    starts = list(range(0, encoded.shape[1], stride))
    width = window_size + 2
    segments = np.empty((encoded.shape[0], len(starts), width, 4), dtype=np.float64)
    for segment_index, start in enumerate(starts):
        stop = min(start + window_size, encoded.shape[1])
        count = stop - start
        for sample_index in range(encoded.shape[0]):
            block = np.empty((width, 4), dtype=np.float64)
            block[:, 0:2] = 7.0
            block[:, 2] = 0.0
            block[:, 3] = -1.0
            block[0] = (5.0, 5.0, genomic_positions[start] - 1.0, -1.0)
            block[1 : count + 1, 0:2] = encoded[sample_index, start:stop]
            block[1 : count + 1, 2] = genomic_positions[start:stop]
            block[1 : count + 1, 3] = np.arange(start, stop, dtype=float)
            block[count + 1] = (6.0, 6.0, genomic_positions[stop - 1] + 1.0, -1.0)
            segments[sample_index, segment_index] = block
    return segments

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
encoded = np.array(
    [
        [[1, 1], [0, 2], [3, 3], [4, 4], [1, 1]],
        [[4, 4], [3, 3], [2, 2], [0, 1], [2, 2]],
    ],
    dtype=int,
)
genomic_positions = np.array([10.0, 12.0, 20.0, 21.0, 35.0])
window_size = 4
overlap = 2
""",
            "call": "segment_genotype_windows(encoded, genomic_positions, window_size, overlap).tolist()",
            "gold_call": "_oracle_segment_genotype_windows(encoded, genomic_positions, window_size, overlap).tolist()",
        },
        {
            "setup": """import numpy as np
encoded = np.array([[[0, 4]]], dtype=int)
genomic_positions = np.array([1.0])
window_size = 1
overlap = 0
""",
            "call": "segment_genotype_windows(encoded, genomic_positions, window_size, overlap).tolist()",
            "gold_call": "_oracle_segment_genotype_windows(encoded, genomic_positions, window_size, overlap).tolist()",
        },
        {
            "setup": """import numpy as np
encoded = np.array(
    [
        [[1, 1], [0, 2], [3, 3], [4, 4], [1, 1]],
        [[4, 4], [3, 3], [2, 2], [0, 1], [2, 2]],
    ],
    dtype=int,
)
genomic_positions = np.array([10.0, 12.0, 20.0, 21.0, 35.0])
def run_model():
    try:
        segment_genotype_windows(encoded, genomic_positions, window_size=3, overlap=3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_segment_genotype_windows(encoded, genomic_positions, window_size=3, overlap=3)
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
