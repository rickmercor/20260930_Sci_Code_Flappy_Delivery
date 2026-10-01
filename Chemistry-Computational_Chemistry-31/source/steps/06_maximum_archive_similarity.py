"""
Score candidate arrow profiles against a curated mechanism archive.

The source's unordered similarity compares a candidate mechanism with each curated mechanism using the Jaccard index: the size of the shared arrow-environment set divided by the size of their union. Each candidate retains its largest archive score. Empty references are ignored, and a tie between references is resolved here by the smallest one-based reference label.

Returns
-------
Float array containing maximum similarity, best reference label, intersection size, and union size for each candidate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def maximum_archive_similarity(
    candidate_profiles: 'np.ndarray',
    reference_profiles: 'np.ndarray',
) -> 'np.ndarray':
    """Find each candidate's largest unordered archive similarity.

    Treat nonzero entries as set membership. Ignore empty reference profiles.
    For every candidate-reference pair, use intersection size divided by union
    size. If several references maximize the score, choose the smallest
    one-based reference label.

    Returns
    -------
    np.ndarray
        Float array with four columns: maximum similarity, best one-based
        reference label, intersection size, and union size.

    Raises
    ------
    ValueError
        If profile shapes are incompatible, entries are non-finite, or the
        archive contains no nonempty reference.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_maximum_archive_similarity(
    candidate_profiles: 'np.ndarray',
    reference_profiles: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    candidates = np.asarray(candidate_profiles)
    references = np.asarray(reference_profiles)

    if candidates.ndim != 2 or references.ndim != 2:
        raise ValueError(
            "profiles must be two-dimensional"
        )
    if (
        candidates.shape[1] == 0
        or references.shape[1] != candidates.shape[1]
    ):
        raise ValueError(
            "candidate and reference feature dimensions must agree"
        )

    for value, name in (
        (candidates, "candidate_profiles"),
        (references, "reference_profiles"),
    ):
        if (
            not np.issubdtype(value.dtype, np.number)
            or not np.isrealobj(value)
            or np.any(~np.isfinite(value))
        ):
            raise ValueError(f"{name} must be finite and numeric")

    candidates = candidates != 0
    references = references != 0

    valid_references = np.flatnonzero(
        np.any(references, axis=1)
    )
    if valid_references.size == 0:
        raise ValueError(
            "reference archive must contain a nonempty profile"
        )

    result_rows = []

    for candidate in candidates:
        best = None

        for reference_index in valid_references:
            reference = references[reference_index]
            intersection = int(
                np.count_nonzero(candidate & reference)
            )
            union = int(
                np.count_nonzero(candidate | reference)
            )
            score = float(intersection / union)

            row = (
                score,
                int(reference_index) + 1,
                intersection,
                union,
            )

            if best is None or score > best[0]:
                best = row

        result_rows.append(best)

    if not result_rows:
        return np.empty((0, 4), dtype=float)

    return np.asarray(result_rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nc=np.array([[1,0,1,1,0],[0,1,0,0,1]])\nr=np.array([[1,1,0,1,0],[1,0,1,0,1],[0,1,0,1,1]])",
            "call": "maximum_archive_similarity(c.copy(),r.copy())",
            "gold_call": "_oracle_maximum_archive_similarity(c.copy(),r.copy())",
        },
        {
            "setup": "import numpy as np\nc=np.array([[1,0,0,0]])\nr=np.array([[1,1,0,0],[1,0,1,0],[0,0,0,0]])",
            "call": "maximum_archive_similarity(c.copy(),r.copy())",
            "gold_call": "_oracle_maximum_archive_similarity(c.copy(),r.copy())",
        },
        {
            "setup": "import numpy as np\nc=np.ones((1,3))\nr=np.zeros((2,3))\ndef check(fn):\n try: fn(c.copy(),r.copy())\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(maximum_archive_similarity)",
            "gold_call": "check(_oracle_maximum_archive_similarity)",
        },
        {
            "setup": "import numpy as np\nc=np.empty((0,4),dtype=int)\nr=np.array([[1,0,1,0]])",
            "call": "maximum_archive_similarity(c.copy(),r.copy())",
            "gold_call": "_oracle_maximum_archive_similarity(c.copy(),r.copy())",
        },
    ]
