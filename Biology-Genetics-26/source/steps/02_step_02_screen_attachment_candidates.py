"""
Correct unequal branch availability and screen focal-test attachment excesses.

The more observable member of a test-uncle pair has more opportunities to receive attachments. Scaling its count to the smaller availability makes the two counts comparable. Support requires the attachment excess to meet the lower-tail cutoff in both the raw and corrected comparisons.

Returns
-------
numpy.ndarray of floats with shape (number of candidates, 5)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def screen_attachment_candidates(
    candidate_summary: np.ndarray,
    z_cutoff: float = -1.96,
) -> np.ndarray:
    '''Apply availability correction and the signed avuncular screen.

    Parameters
    ----------
    candidate_summary : np.ndarray
        Nonnegative rows [raw_test, raw_uncle, avail_test, avail_uncle, avail_focal, joint_test_focal].
    z_cutoff : float
        Finite inclusive lower-tail cutoff.

    Returns
    -------
    screen : np.ndarray
        Columns [corrected_test, corrected_uncle, raw_z_score, corrected_z_score, supported].

    Raises
    ------
    ValueError
        If candidate_summary is not a nonempty numeric array of shape (c, 6) with finite nonnegative entries, either tested availability is nonpositive, an attachment count exceeds its availability, or z_cutoff is not finite and numeric.
    '''
    return screen

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_screen_attachment_candidates(
    candidate_summary: np.ndarray,
    z_cutoff: float = -1.96,
) -> np.ndarray:
    """Reference implementation."""
    summary = np.asarray(candidate_summary)
    if summary.ndim != 2 or summary.shape[1] != 6:
        raise ValueError("candidate_summary must have shape (c, 6)")
    if summary.shape[0] == 0 or not np.issubdtype(summary.dtype, np.number):
        raise ValueError("candidate_summary must contain numeric candidate rows")
    if not np.all(np.isfinite(summary)) or np.any(summary < 0):
        raise ValueError("candidate_summary entries must be finite and nonnegative")
    if not isinstance(z_cutoff, (int, float)) or not math.isfinite(float(z_cutoff)):
        raise ValueError("z_cutoff must be finite and numeric")

    rows = []
    for row in summary:
        raw_test, raw_uncle, avail_test, avail_uncle = map(float, row[:4])
        if avail_test <= 0.0 or avail_uncle <= 0.0:
            raise ValueError("test and uncle availability must be positive")
        if raw_test > avail_test or raw_uncle > avail_uncle:
            raise ValueError("an attachment count cannot exceed its branch availability")
        raw_denominator = math.sqrt(raw_test + raw_uncle)
        raw_z_score = (
            (raw_uncle - raw_test) / raw_denominator
            if raw_denominator > 0.0
            else 0.0
        )
        corrected_test = raw_test
        corrected_uncle = raw_uncle
        if avail_test > avail_uncle:
            corrected_test *= avail_uncle / avail_test
        elif avail_uncle > avail_test:
            corrected_uncle *= avail_test / avail_uncle
        corrected_denominator = math.sqrt(corrected_test + corrected_uncle)
        corrected_z_score = (
            (corrected_uncle - corrected_test) / corrected_denominator
            if corrected_denominator > 0.0
            else 0.0
        )
        supported = float(
            raw_z_score <= float(z_cutoff)
            and corrected_z_score <= float(z_cutoff)
        )
        rows.append(
            [
                corrected_test,
                corrected_uncle,
                raw_z_score,
                corrected_z_score,
                supported,
            ]
        )
    return np.asarray(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''summary = np.array([
    [50, 32, 80, 100, 90, 72],
    [50, 30, 100, 60, 100, 100],
])
expected = np.array([
    [50.0, 25.6, -1.9877674693472376, -2.8062666079922405, 1.0],
    [30.0, 30.0, -2.23606797749979, 0.0, 0.0],
])
def check(value):
    if not np.allclose(value, expected, rtol=0.0, atol=1e-8):
        raise AssertionError("unexpected prompt-instance candidate screen")
    return 1
''',
            "call": "check(screen_attachment_candidates(summary))",
            "gold_call": "check(_oracle_screen_attachment_candidates(summary))",
        },
        {
            "setup": '''summary = np.array([[50, 40, 80, 100, 90, 72]])
expected = np.array([[50.0, 32.0, -1.0540925533894598, -1.9877674693472376, 0.0]])
def check(value):
    if not np.allclose(value, expected, rtol=0.0, atol=1e-8):
        raise AssertionError("unexpected correction-only candidate screen")
    return 2
''',
            "call": "check(screen_attachment_candidates(summary))",
            "gold_call": "check(_oracle_screen_attachment_candidates(summary))",
        },
        {
            "setup": '''summary = np.array([[2, 1, 0, 3, 3, 0]])
def error_code(function):
    try:
        function(summary)
    except ValueError:
        return 3
    except Exception:
        return 4
    return 0
''',
            "call": "error_code(screen_attachment_candidates)",
            "gold_call": "error_code(_oracle_screen_attachment_candidates)",
        },
    ]
