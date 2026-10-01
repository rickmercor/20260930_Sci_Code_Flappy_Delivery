"""
The masked-language head maps each encoded token back to probabilities for the

four phased genotype states. Overlapping genomic windows can provide multiple

predictions for one masked locus, so this toy inference rule averages class

probabilities across every segment occurrence before scoring. The target phase

state is carried with each occurrence and must agree across windows.

Inputs

------

hidden: final hidden vectors for all segment tokens

segments: packed segment data containing source indices and target states

masked_sites: requested sample and source-variant pairs

Returns

-------

pooled: rows containing site identity, target state, and four averaged probabilities

Returns
-------
np.ndarray of shape (n_masked, 7), site indices, target state, and pooled class probabilities
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def merge_masked_probabilities(
    hidden: "np.ndarray", segments: "np.ndarray", masked_sites: "np.ndarray"
) -> "np.ndarray":
    """Decode genotype probabilities and average overlapping masked predictions.

    Parameters
    ----------
    hidden : np.ndarray
        Finite array of shape ``(n_samples, n_segments, width, d)``.
    segments : np.ndarray
        Finite packed array of shape ``(n_samples, n_segments, width, 4)`` with
        observed token, target token, coordinate, and source index.
    masked_sites : np.ndarray
        Distinct integral ``(sample_index, variant_index)`` rows.

    Raises
    ------
    ValueError
        If hidden and segments have incompatible shapes or non-finite entries;
        if masked_sites has the wrong shape, non-integral or duplicate rows, or
        an out-of-range sample; or if a requested site has no masked segment
        occurrence or has inconsistent target states outside 1 through 4.

    Returns
    -------
    pooled : np.ndarray
        float64 array of shape ``(n_masked, 7)`` with columns sample index,
        variant index, target state, then probabilities for states 1 through 4.
    """
    return pooled  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_merge_masked_probabilities(
    hidden: "np.ndarray", segments: "np.ndarray", masked_sites: "np.ndarray"
) -> "np.ndarray":
    """Reference implementation."""
    hidden = np.asarray(hidden, dtype=float)
    segments = np.asarray(segments, dtype=float)
    masked_sites = np.asarray(masked_sites)
    if hidden.ndim != 4 or 0 in hidden.shape:
        raise ValueError("hidden must have shape (n_samples, n_segments, width, d)")
    if segments.shape != hidden.shape[:3] + (4,):
        raise ValueError("segments shape must match hidden token axes")
    if not np.all(np.isfinite(hidden)) or not np.all(np.isfinite(segments)):
        raise ValueError("hidden and segments must be finite")
    if masked_sites.ndim != 2 or masked_sites.shape[1] != 2:
        raise ValueError("masked_sites must have shape (n_masked, 2)")
    if not np.issubdtype(masked_sites.dtype, np.integer) and (
        not np.all(np.isfinite(masked_sites))
        or not np.all(masked_sites == np.floor(masked_sites))
    ):
        raise ValueError("masked_sites must contain integers")
    masked_sites = masked_sites.astype(np.int64, copy=False)
    if len({tuple(row) for row in masked_sites.tolist()}) != len(masked_sites):
        raise ValueError("masked_sites rows must be distinct")
    if masked_sites.size and (
        np.any(masked_sites[:, 0] < 0) or np.any(masked_sites[:, 0] >= hidden.shape[0])
    ):
        raise ValueError("masked sample index out of range")

    hidden_dim = hidden.shape[-1]
    row = np.arange(1.0, hidden_dim + 1.0)[:, None]
    genotype_class = np.arange(1.0, 5.0)[None, :]
    decoder = 0.55 * np.sin(
        row * (genotype_class + 1.0) + 0.25 * (genotype_class - 1.0)
    )
    decoder_bias = np.array([0.15, -0.10, 0.05, 0.0])
    logits = hidden @ decoder + decoder_bias
    logits = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)

    pooled = np.empty((len(masked_sites), 7), dtype=np.float64)
    for row_index, (sample_index, variant_index) in enumerate(masked_sites):
        source_indices = segments[sample_index, :, :, 3]
        observed_tokens = segments[sample_index, :, :, 0]
        occurrence = (source_indices == variant_index) & (observed_tokens == 0)
        if not np.any(occurrence):
            raise ValueError(
                "each requested site must have a masked segment occurrence"
            )
        targets = segments[sample_index, :, :, 1][occurrence]
        if not np.all(targets == targets[0]) or targets[0] not in (1.0, 2.0, 3.0, 4.0):
            raise ValueError("masked occurrences must have one valid target state")
        mean_probability = probabilities[sample_index][occurrence].mean(axis=0)
        pooled[row_index] = np.concatenate(
            (
                [float(sample_index), float(variant_index), float(targets[0])],
                mean_probability,
            )
        )
    return pooled

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pin = (
        "import math\n"
        "def _pin(result):\n"
        "    flat = [v for row in result for v in row]\n"
        "    return (sum(math.sin(0.31 * (i + 1)) * v for i, v in enumerate(flat))\n"
        "            + 10000.0 * len(result))\n"
    )
    return [
        {
            "setup": pin + """import numpy as np
hidden = np.linspace(-0.8, 0.9, 1 * 2 * 4 * 4).reshape(1, 2, 4, 4)
segments = np.array(
    [
        [
            [[5, 5, 9, -1], [0, 2, 10, 0], [3, 3, 12, 1], [6, 6, 13, -1]],
            [[5, 5, 11, -1], [3, 3, 12, 1], [0, 4, 18, 2], [6, 6, 19, -1]],
        ]
    ],
    dtype=float,
)
masked_sites = np.array([[0, 0], [0, 2]])
""",
            "call": "_pin(merge_masked_probabilities(hidden, segments, masked_sites).tolist())",
            "gold_call": "_pin(_oracle_merge_masked_probabilities(hidden, segments, masked_sites).tolist())",
        },
        {
            "setup": pin + """import numpy as np
hidden = np.linspace(-0.8, 0.9, 1 * 2 * 4 * 4).reshape(1, 2, 4, 4)
segments = np.array(
    [
        [
            [[5, 5, 9, -1], [0, 2, 10, 0], [3, 3, 12, 1], [6, 6, 13, -1]],
            [[5, 5, 11, -1], [3, 3, 12, 1], [0, 4, 18, 2], [6, 6, 19, -1]],
        ]
    ],
    dtype=float,
)
masked_sites = np.empty((0, 2), dtype=int)
""",
            "call": "_pin(merge_masked_probabilities(hidden, segments, masked_sites).tolist())",
            "gold_call": "_pin(_oracle_merge_masked_probabilities(hidden, segments, masked_sites).tolist())",
        },
        {
            "setup": """import numpy as np
hidden = np.linspace(-0.8, 0.9, 1 * 2 * 4 * 4).reshape(1, 2, 4, 4)
segments = np.array(
    [
        [
            [[5, 5, 9, -1], [0, 2, 10, 0], [3, 3, 12, 1], [6, 6, 13, -1]],
            [[5, 5, 11, -1], [3, 3, 12, 1], [0, 4, 18, 2], [6, 6, 19, -1]],
        ]
    ],
    dtype=float,
)
masked_sites = np.array([[0, 1]])
def run_model():
    try:
        merge_masked_probabilities(hidden, segments, masked_sites)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_merge_masked_probabilities(hidden, segments, masked_sites)
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
