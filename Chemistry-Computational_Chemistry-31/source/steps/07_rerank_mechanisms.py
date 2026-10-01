"""
Re-rank parsimonious mechanisms by maximum archive similarity.

After the source workflow generates its parsimonious candidate mechanisms, they are ordered by descending maximum similarity to the curated archive. This benchmark preserves the original parsimony order whenever two candidates receive exactly the same similarity score.

Returns
-------
Float table containing the mechanisms in descending similarity order with their provenance, overlap statistics, step counts, and padded rule sequences.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rerank_mechanisms(
    mechanisms: 'np.ndarray',
    similarity_summary: 'np.ndarray',
) -> 'np.ndarray':
    """Sort mechanisms by descending similarity with a stable parsimony tie.

    Preserve input order when similarity values tie. The input mechanism table
    follows ``generate_parsimonious_mechanisms`` and the summary follows
    ``maximum_archive_similarity``.

    Returns
    -------
    np.ndarray
        Float table whose columns are similarity, original balanced-candidate
        rank, best reference label, intersection size, union size, step count,
        and the padded one-based rule sequence.

    Raises
    ------
    ValueError
        If the two tables are malformed, incompatible, or contain invalid
        similarity values.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rerank_mechanisms(
    mechanisms: 'np.ndarray',
    similarity_summary: 'np.ndarray',
) -> 'np.ndarray':
    import numpy as np

    mechs = np.asarray(mechanisms)
    summary = np.asarray(similarity_summary)

    if mechs.ndim != 2 or mechs.shape[1] < 3:
        raise ValueError(
            "mechanisms must be a padded two-dimensional table"
        )
    if (
        summary.ndim != 2
        or summary.shape != (mechs.shape[0], 4)
    ):
        raise ValueError(
            "similarity_summary must have four columns and matching rows"
        )
    if (
        not np.issubdtype(mechs.dtype, np.number)
        or not np.isrealobj(mechs)
        or np.any(~np.isfinite(mechs))
    ):
        raise ValueError(
            "mechanisms must be finite and numeric"
        )
    if (
        not np.issubdtype(summary.dtype, np.number)
        or not np.isrealobj(summary)
        or np.any(~np.isfinite(summary))
    ):
        raise ValueError(
            "similarity_summary must be finite and numeric"
        )
    if (
        np.any(summary[:, 0] < 0.0)
        or np.any(summary[:, 0] > 1.0)
    ):
        raise ValueError(
            "similarities must lie between zero and one"
        )

    order = np.lexsort(
        (
            np.arange(mechs.shape[0]),
            -summary[:, 0],
        )
    )

    rows = []

    for index in order:
        row = np.concatenate(
            (
                np.array(
                    [summary[index, 0], mechs[index, 0]],
                    dtype=float,
                ),
                summary[index, 1:4].astype(
                    float,
                    copy=False,
                ),
                mechs[index, 1:].astype(
                    float,
                    copy=False,
                ),
            )
        )
        rows.append(row)

    if not rows:
        return np.empty(
            (0, mechs.shape[1] + 4),
            dtype=float,
        )

    return np.vstack(rows)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nm=np.array([[2,2,1,4,0],[4,3,2,3,5],[7,2,6,8,0]])\ns=np.array([[.25,3,2,8],[.5,1,4,8],[.25,2,1,4]])",
            "call": "rerank_mechanisms(m.copy(),s.copy())",
            "gold_call": "_oracle_rerank_mechanisms(m.copy(),s.copy())",
        },
        {
            "setup": "import numpy as np\nm=np.array([[9,1,3],[10,1,4]])\ns=np.array([[.4,2,2,5],[.4,1,2,5]])",
            "call": "rerank_mechanisms(m.copy(),s.copy())",
            "gold_call": "_oracle_rerank_mechanisms(m.copy(),s.copy())",
        },
        {
            "setup": "import numpy as np\nm=np.empty((0,4),dtype=int)\ns=np.empty((0,4),dtype=float)",
            "call": "rerank_mechanisms(m.copy(),s.copy())",
            "gold_call": "_oracle_rerank_mechanisms(m.copy(),s.copy())",
        },
        {
            "setup": "import numpy as np\nm=np.array([[1,1,2]])\ns=np.array([[1.2,1,1,1]])\ndef check(fn):\n try: fn(m.copy(),s.copy())\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(rerank_mechanisms)",
            "gold_call": "check(_oracle_rerank_mechanisms)",
        },
    ]
