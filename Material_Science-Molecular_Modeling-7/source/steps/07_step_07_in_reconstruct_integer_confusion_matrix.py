"""
Reconstruct a nonnegative integer confusion matrix from row totals and arow-normalized matrix printed to a specified number of significant figures.

igure 14 reports every cell to two significant figures. A count is admissible only when its row fraction, formatted with the corresponding general-format precision, reproduces the displayed value under the same rule. Enforce the complete row total and require exactly one integer solution for every row.

Return an integer array with the same square shape as row_normalized.

Raise ValueError if an input is malformed, a displayed cell has no admissible count, or a row does not have exactly one integer reconstruction.

Returns
-------
An integer array with the same square shape as row_normalized.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def reconstruct_integer_confusion_matrix(
    row_counts: np.ndarray,
    row_normalized: np.ndarray,
    significant_figures: np.ndarray,
) -> np.ndarray:
    return np.empty_like(row_normalized, dtype=int)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reconstruct_integer_confusion_matrix(
    row_counts,
    row_normalized,
    significant_figures,
):
    counts = np.asarray(row_counts)
    shown = np.asarray(row_normalized, dtype=float)
    sigfigs = np.asarray(significant_figures)

    if (
        counts.ndim != 1
        or counts.size < 2
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts <= 0)
    ):
        raise ValueError("row_counts must be a positive integer vector")

    k = counts.size
    if (
        shown.shape != (k, k)
        or not np.all(np.isfinite(shown))
        or np.any(shown < 0)
        or np.any(shown > 1)
    ):
        raise ValueError("row_normalized must be a finite k by k array in [0,1]")

    if (
        sigfigs.shape != (k, k)
        or not np.issubdtype(sigfigs.dtype, np.integer)
        or np.any(sigfigs < 1)
        or np.any(sigfigs > 15)
    ):
        raise ValueError(
            "significant_figures must be an integer k by k array in [1,15]"
        )

    out = np.zeros((k, k), dtype=int)
    for i, n0 in enumerate(counts.tolist()):
        n = int(n0)
        candidates = []
        for displayed, s0 in zip(shown[i], sigfigs[i]):
            s = int(s0)
            target = format(float(displayed), f".{s}g")
            values = [
                x
                for x in range(n + 1)
                if format(x / n, f".{s}g") == target
            ]
            if not values:
                raise ValueError(
                    "a displayed cell has no admissible integer count"
                )
            candidates.append(values)

        solutions = []

        def search(j, remaining, prefix):
            if len(solutions) > 1:
                return
            if j == k - 1:
                if remaining in candidates[j]:
                    solutions.append(prefix + [remaining])
                return
            for value in candidates[j]:
                if value <= remaining:
                    search(j + 1, remaining - value, prefix + [value])

        search(0, n, [])
        if len(solutions) != 1:
            raise ValueError(
                "each displayed row must have a unique integer reconstruction"
            )
        out[i] = solutions[0]

    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
row_counts=np.array([19,51,25,37,8,9],dtype=int)
row_normalized=np.array([[.79,0,.11,.11,0,0],[0,.84,0,.14,.02,0],[.12,.04,.56,.24,.04,0],[.081,.11,.081,.7,0,.027],[.12,0,0,0,.88,0],[0,.11,0,0,.11,.78]],float)
significant_figures=np.full((6,6),2,dtype=int)""",
            "call": "reconstruct_integer_confusion_matrix(row_counts,row_normalized,significant_figures)",
            "gold_call": "_oracle_reconstruct_integer_confusion_matrix(row_counts,row_normalized,significant_figures)",
        },
        {
            "setup": """import numpy as np
row_counts=np.array([10,10],dtype=int)
row_normalized=np.array([[.7,.3],[.2,.8]],float)
significant_figures=np.full((2,2),1,dtype=int)""",
            "call": "reconstruct_integer_confusion_matrix(row_counts,row_normalized,significant_figures)",
            "gold_call": "_oracle_reconstruct_integer_confusion_matrix(row_counts,row_normalized,significant_figures)",
        },
        {
            "setup": """import numpy as np
row_counts=np.array([8,8],dtype=int)
row_normalized=np.array([[.12,.88],[.5,.5]],float)
significant_figures=np.full((2,2),2,dtype=int)""",
            "call": "reconstruct_integer_confusion_matrix(row_counts,row_normalized,significant_figures)",
            "gold_call": "_oracle_reconstruct_integer_confusion_matrix(row_counts,row_normalized,significant_figures)",
        },
        {
            "setup": """import numpy as np
row_counts=np.array([100,100],dtype=int)
row_normalized=np.array([[.5,.5],[.5,.5]],float)
significant_figures=np.full((2,2),1,dtype=int)
def run_model():
 try: reconstruct_integer_confusion_matrix(row_counts,row_normalized,significant_figures); return 0
 except ValueError:return 1
 except Exception:return 2
def run_gold():
 try: _oracle_reconstruct_integer_confusion_matrix(row_counts,row_normalized,significant_figures); return 0
 except ValueError:return 1
 except Exception:return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
row_counts=np.array([2.0,3.0])
row_normalized=np.eye(2)
significant_figures=np.full((2,2),2,dtype=int)
def run_model():
 try: reconstruct_integer_confusion_matrix(row_counts,row_normalized,significant_figures); return 0
 except ValueError:return 1
 except Exception:return 2
def run_gold():
 try: _oracle_reconstruct_integer_confusion_matrix(row_counts,row_normalized,significant_figures); return 0
 except ValueError:return 1
 except Exception:return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
