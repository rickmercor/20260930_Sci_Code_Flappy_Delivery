"""
Summarize discordant attachments and branch observability for each candidate pair.

An attachment to a candidate branch can be counted only in a gene tree where the relevant branch is observable. Raw focal-test and focal-uncle attachment counts therefore need their test, uncle, focal, and joint focal-test availability totals before the candidates can be compared.

Returns
-------
numpy.ndarray of integers with shape (number of candidates, 6)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def summarize_candidate_evidence(
    attachment_codes_by_candidate: list,
    branch_presence_by_candidate: list,
) -> np.ndarray:
    '''Count attachments and branch availability for every candidate.

    Parameters
    ----------
    attachment_codes_by_candidate : list
        One-dimensional arrays containing only 0, 1, and 2.
    branch_presence_by_candidate : list
        Matching binary arrays with test, uncle, and focal columns.

    Returns
    -------
    summary : np.ndarray
        Integer array with six columns in the declared order.

    Raises
    ------
    ValueError
        If the candidate sequences are empty or unequal in length, an attachment-code array is not a one-dimensional integer array containing only {0, 1, 2}, or a presence array is not a matching binary integer array of shape (n, 3).
    '''
    return summary

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_summarize_candidate_evidence(
    attachment_codes_by_candidate: list,
    branch_presence_by_candidate: list,
) -> np.ndarray:
    """Reference implementation."""
    if not isinstance(attachment_codes_by_candidate, (list, tuple)):
        raise ValueError("attachment codes must be a sequence")
    if not isinstance(branch_presence_by_candidate, (list, tuple)):
        raise ValueError("branch presence must be a sequence")
    if len(attachment_codes_by_candidate) == 0:
        raise ValueError("at least one candidate is required")
    if len(attachment_codes_by_candidate) != len(branch_presence_by_candidate):
        raise ValueError("candidate sequences must have equal length")

    rows = []
    for codes_value, presence_value in zip(
        attachment_codes_by_candidate,
        branch_presence_by_candidate,
    ):
        codes = np.asarray(codes_value)
        presence = np.asarray(presence_value)
        if codes.ndim != 1 or not np.issubdtype(codes.dtype, np.integer):
            raise ValueError("each attachment-code array must be one-dimensional integers")
        if presence.shape != (codes.size, 3) or not np.issubdtype(
            presence.dtype,
            np.integer,
        ):
            raise ValueError("each presence array must be an integer array of shape (n, 3)")
        if not np.all(np.isin(codes, (0, 1, 2))):
            raise ValueError("attachment codes must lie in {0, 1, 2}")
        if not np.all(np.isin(presence, (0, 1))):
            raise ValueError("branch-presence entries must be binary")

        raw_test = int(np.count_nonzero(codes == 1))
        raw_uncle = int(np.count_nonzero(codes == 2))
        availability = np.sum(presence, axis=0, dtype=int)
        joint_test_focal = int(np.count_nonzero((presence[:, 0] == 1) & (presence[:, 2] == 1)))
        rows.append(
            [
                raw_test,
                raw_uncle,
                int(availability[0]),
                int(availability[1]),
                int(availability[2]),
                joint_test_focal,
            ]
        )
    return np.asarray(rows, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": '''codes = [
    np.array([1] * 50 + [2] * 22 + [0] * 8 + [2] * 10 + [0] * 10),
    np.array([2] * 30 + [1] * 50 + [0] * 20),
]
presence_a = np.column_stack((
    [1] * 80 + [0] * 20,
    [1] * 100,
    [1] * 72 + [0] * 8 + [1] * 18 + [0] * 2,
))
presence_b = np.ones((100, 3), dtype=int)
presence_b[60:, 1] = 0
presence = [presence_a, presence_b]
expected = [[50, 32, 80, 100, 90, 72], [50, 30, 100, 60, 100, 100]]
def check(value):
    result = value.tolist()
    if result != expected:
        raise AssertionError("unexpected prompt-instance evidence summary")
    return result
''',
            "call": "check(summarize_candidate_evidence(codes, presence))",
            "gold_call": "check(_oracle_summarize_candidate_evidence(codes, presence))",
        },
        {
            "setup": '''codes = [np.array([1, 2, 0, 1, 0, 2, 1], dtype=int)]
presence = [np.array([
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 1],
], dtype=int)]
expected = [[3, 2, 5, 5, 5, 3]]
def check(value):
    result = value.tolist()
    if result != expected:
        raise AssertionError("unexpected partial-overlap evidence summary")
    return result
''',
            "call": "check(summarize_candidate_evidence(codes, presence))",
            "gold_call": "check(_oracle_summarize_candidate_evidence(codes, presence))",
        },
        {
            "setup": '''codes = [np.array([1, 3], dtype=int)]
presence = [np.ones((2, 3), dtype=int)]
def error_code(function):
    try:
        function(codes, presence)
    except ValueError:
        return 2
    except Exception:
        return 3
    return 0
''',
            "call": "error_code(summarize_candidate_evidence)",
            "gold_call": "error_code(_oracle_summarize_candidate_evidence)",
        },
    ]
