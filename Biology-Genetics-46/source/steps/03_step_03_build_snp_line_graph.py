"""
Turn the aligned fragment matrix into the ordered vertex set of the SNP line graph, the structure on which the phasing distribution is defined.

Two heterozygous SNPs are joined when at least one fragment calls an allele at both, and the vertices of the model are those joined pairs rather than the SNPs themselves. Listing the pairs in ascending genomic order gives the topological order that makes inference over the resulting graph tractable.

Returns
-------
np.ndarray of shape (U, 2), int: the SNP pairs joined by at least one fragment, in ascending genomic order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_snp_line_graph(reads: np.ndarray) -> np.ndarray:
    """Build the ordered vertex list of the SNP line graph of a fragment set.

    Parameters
    ----------
    reads : np.ndarray
        Integer array of shape (n_reads, n_snps) holding the aligned fragment
        matrix, with 0 for the reference allele, 1 for the alternate allele and
        -1 where the fragment has no called allele.

    Returns
    -------
    nodes : np.ndarray
        Integer array of shape (U, 2). Row t holds the two SNP indices of the
        t-th vertex, the smaller index first. The rows are sorted in ascending
        order of the first index, ties broken by the second index. A fragment
        set that joins no pair of SNPs yields an array of shape (0, 2).

    Raises
    ------
    ValueError
        If ``reads`` is not a non-empty two-dimensional integer-valued array
        with entries in {-1, 0, 1}.
    """
    return nodes  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_snp_line_graph(reads: np.ndarray) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import itertools

    import numpy as np

    fragments = np.asarray(reads, dtype=float)
    if fragments.ndim != 2 or fragments.size < 1:
        raise ValueError("reads must be a non-empty two-dimensional array")
    if not np.all(np.isin(fragments, (-1.0, 0.0, 1.0))):
        raise ValueError("reads entries must be -1, 0 or 1")
    fragments = fragments.astype(int)

    # A fragment covering fewer than two SNPs carries no phase information and
    # therefore contributes no edge to the SNP graph.
    pairs = set()
    for fragment in fragments:
        covered = np.flatnonzero(fragment >= 0).tolist()
        for first, second in itertools.combinations(covered, 2):
            pairs.add((int(first), int(second)))

    return np.array(sorted(pairs), dtype=int).reshape(-1, 2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: overlapping fragments over five SNPs (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
reads = np.array([[0, 1, 0, -1, -1],
                  [-1, 1, 0, 1, -1],
                  [-1, -1, 0, 1, 1],
                  [1, 0, -1, -1, -1]], dtype=int)
""",
            "call": "sig(build_snp_line_graph(reads), 10.0)",
            "gold_call": "sig(_oracle_build_snp_line_graph(reads), 10.0)",
        },
        # --- Valid: a gapped fragment, which joins the two SNPs it calls even
        #     though it skips the SNP between them ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
reads = np.array([[0, -1, 1, -1, -1],
                  [-1, 1, -1, 0, -1],
                  [1, 1, -1, -1, 0]], dtype=int)
""",
            "call": "sig(build_snp_line_graph(reads), 10.0)",
            "gold_call": "sig(_oracle_build_snp_line_graph(reads), 10.0)",
        },
        # --- Valid: two disjoint runs of SNPs, which must not be joined ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
reads = np.array([[0, 1, -1, -1, -1, -1],
                  [1, 0, -1, -1, -1, -1],
                  [-1, -1, -1, 0, 1, 1],
                  [-1, -1, -1, 1, 0, 0]], dtype=int)
""",
            "call": "sig(build_snp_line_graph(reads), 10.0)",
            "gold_call": "sig(_oracle_build_snp_line_graph(reads), 10.0)",
        },
        # --- Boundary: a fragment set in which every fragment calls a single
        #     SNP, so that no pair is joined at all ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
reads = np.array([[0, -1, -1],
                  [-1, 1, -1],
                  [-1, -1, 1]], dtype=int)
""",
            "call": "sig(build_snp_line_graph(reads), 1.0)",
            "gold_call": "sig(_oracle_build_snp_line_graph(reads), 1.0)",
        },
        # --- Edge: a single fragment covering every SNP, which makes the SNP
        #     graph complete ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
reads = np.array([[1, 0, 1, 0, 1]], dtype=int)
""",
            "call": "sig(build_snp_line_graph(reads), 10.0)",
            "gold_call": "sig(_oracle_build_snp_line_graph(reads), 10.0)",
        },
        # --- Edge: duplicated fragments, which must not duplicate vertices ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
reads = np.array([[0, 1, -1],
                  [0, 1, -1],
                  [0, 1, -1],
                  [-1, 0, 1]], dtype=int)
""",
            "call": "sig(build_snp_line_graph(reads), 1.0)",
            "gold_call": "sig(_oracle_build_snp_line_graph(reads), 1.0)",
        },
        # --- Invalid: a fragment matrix carrying an allele code of 2 ---
        {
            "setup": """import numpy as np
reads = np.array([[0, 2, -1], [1, 0, -1]], dtype=int)
def run_model():
    try:
        build_snp_line_graph(reads)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_snp_line_graph(reads)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a one-dimensional fragment vector ---
        {
            "setup": """import numpy as np
reads = np.array([0, 1, -1], dtype=int)
def run_model():
    try:
        build_snp_line_graph(reads)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_snp_line_graph(reads)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
