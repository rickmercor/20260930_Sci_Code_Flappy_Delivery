"""
Calculate the ratio of unweighted multiclass Cohen kappa for a synthetic confusion matrix and a paper confusion matrix.

The synthetic input is the augmented matrix returned by score_classification_agreement: its upper-left block is the count matrix, its last column and row contain marginals, and its bottom-right element is the raw agreement. Verify this augmentation before calculating kappa. The paper input is an integer square count matrix of the same class dimension.

Returns
-------
One finite Python float equal to kappa_synthetic divided by kappa_paper.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_kappa_retention(
    synthetic_score_matrix: np.ndarray,
    paper_confusion: np.ndarray,
) -> float:
    """
    Calculate chance-corrected agreement retention from two confusion matrices.

    Parameters
    ----------
    synthetic_score_matrix : np.ndarray
        Augmented square matrix produced by score_classification_agreement.
        The upper-left block contains synthetic counts, the last column and
        last row contain row and column totals, and the bottom-right entry is
        the raw agreement.
    paper_confusion : np.ndarray
        Nonnegative integer square confusion matrix with the same number of
        classes as the synthetic count block.

    Returns
    -------
    float
        Finite value kappa_synthetic / kappa_paper.

    Raises
    ------
    ValueError
        If the synthetic matrix is not a valid augmented square matrix, if its
        stored row totals, column totals or raw agreement disagree with its
        count block, if paper_confusion has the wrong class dimension or is not
        a valid count matrix, if either kappa denominator is invalid, if the
        paper kappa is zero, or if the final ratio is not finite.
    """
    return float("nan")

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from fractions import Fraction


def _oracle_compute_kappa_retention(
    synthetic_score_matrix,
    paper_confusion,
):
    def confusion_kappa(matrix):
        a = np.asarray(matrix, dtype=float)

        if (
            a.ndim != 2
            or a.shape[0] != a.shape[1]
            or a.shape[0] < 2
            or not np.all(np.isfinite(a))
        ):
            raise ValueError(
                "confusion matrix must be a finite square array"
            )

        if (
            np.any(a < 0)
            or not np.allclose(
                a,
                np.rint(a),
                atol=1e-12,
                rtol=0,
            )
        ):
            raise ValueError(
                "confusion counts must be nonnegative integers"
            )

        z = np.rint(a).astype(object)
        n = int(np.sum(z))

        if n <= 0:
            raise ValueError(
                "confusion matrix must contain observations"
            )

        diagonal = sum(
            int(z[i, i])
            for i in range(z.shape[0])
        )

        rows = [
            sum(
                int(v)
                for v in z[i, :]
            )
            for i in range(z.shape[0])
        ]

        cols = [
            sum(
                int(v)
                for v in z[:, j]
            )
            for j in range(z.shape[1])
        ]

        po = Fraction(
            diagonal,
            n,
        )

        pe = Fraction(
            sum(
                r * c
                for r, c in zip(
                    rows,
                    cols,
                )
            ),
            n * n,
        )

        if pe >= 1:
            raise ValueError(
                "kappa denominator must be positive"
            )

        return (
            po - pe
        ) / (
            1 - pe
        )

    score = np.asarray(
        synthetic_score_matrix,
        dtype=float,
    )

    paper = np.asarray(
        paper_confusion
    )

    if (
        score.ndim != 2
        or score.shape[0] != score.shape[1]
        or score.shape[0] < 3
    ):
        raise ValueError(
            "synthetic_score_matrix must be an augmented square matrix"
        )

    k = score.shape[0] - 1

    if paper.shape != (k, k):
        raise ValueError(
            "paper matrix class dimension must match"
        )

    syn = score[
        :k,
        :k,
    ]

    total = float(
        np.sum(syn)
    )

    if not np.allclose(
        score[:k, k],
        np.sum(
            syn,
            axis=1,
        ),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError(
            "synthetic row totals are inconsistent"
        )

    if not np.allclose(
        score[k, :k],
        np.sum(
            syn,
            axis=0,
        ),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError(
            "synthetic column totals are inconsistent"
        )

    if (
        total <= 0
        or not np.isclose(
            score[k, k],
            np.trace(syn) / total,
            atol=1e-12,
            rtol=0,
        )
    ):
        raise ValueError(
            "synthetic raw agreement is inconsistent"
        )

    kappa_synthetic = confusion_kappa(
        syn
    )

    kappa_paper = confusion_kappa(
        paper
    )

    if abs(
        kappa_paper
    ) <= 1e-15:
        raise ValueError(
            "paper kappa must be nonzero"
        )

    result = float(
        kappa_synthetic
        / kappa_paper
    )

    if not np.isfinite(
        result
    ):
        raise ValueError(
            "kappa retention must be finite"
        )

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
C=np.array([[1,0,0,0,1,0],[0,2,0,0,0,0],[0,0,1,1,0,0],[0,1,1,0,0,0],[0,0,0,0,1,1],[0,0,0,0,0,2]],float)
S=np.zeros((7,7),float); S[:6,:6]=C; S[:6,6]=C.sum(1); S[6,:6]=C.sum(0); S[6,6]=np.trace(C)/C.sum()
N=np.array([[15,0,2,2,0,0],[0,43,0,7,1,0],[3,1,14,6,1,0],[3,4,3,26,0,1],[1,0,0,0,7,0],[0,1,0,0,1,7]],int)""",'call':'compute_kappa_retention(S,N)','gold_call':'_oracle_compute_kappa_retention(S,N)'},
        {'setup':"""import numpy as np
C=np.diag([2,3,4]).astype(float); S=np.zeros((4,4)); S[:3,:3]=C; S[:3,3]=C.sum(1); S[3,:3]=C.sum(0); S[3,3]=1.; N=np.diag([5,6,7])""",'call':'compute_kappa_retention(S,N)','gold_call':'_oracle_compute_kappa_retention(S,N)'},
        {'setup':"""import numpy as np
C=np.ones((2,2),float); S=np.zeros((3,3)); S[:2,:2]=C; S[:2,2]=C.sum(1); S[2,:2]=C.sum(0); S[2,2]=.5; N=np.diag([4,4])""",'call':'compute_kappa_retention(S,N)','gold_call':'_oracle_compute_kappa_retention(S,N)'},
        {'setup':"""import numpy as np
C=np.diag([2,2]).astype(float); S=np.zeros((3,3)); S[:2,:2]=C; S[:2,2]=C.sum(1); S[2,:2]=C.sum(0); S[2,2]=.5; N=np.diag([3,3])
def run_model():
 try: compute_kappa_retention(S,N); return 0
 except ValueError:return 1
 except Exception:return 2
def run_gold():
 try: _oracle_compute_kappa_retention(S,N); return 0
 except ValueError:return 1
 except Exception:return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
C=np.diag([2,2]).astype(float); S=np.zeros((3,3)); S[:2,:2]=C; S[:2,2]=C.sum(1); S[2,:2]=C.sum(0); S[2,2]=1.; N=np.ones((2,2),int)
def run_model():
 try: compute_kappa_retention(S,N); return 0
 except ValueError:return 1
 except Exception:return 2
def run_gold():
 try: _oracle_compute_kappa_retention(S,N); return 0
 except ValueError:return 1
 except Exception:return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
