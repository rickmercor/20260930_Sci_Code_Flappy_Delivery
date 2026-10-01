"""
Infer the earlier location given an adult location at sampling.

An adult sampled at a present-day location may have laid eggs at any earlier site. With equal prior weights for earlier sites, conditioning on the known destination requires a destination-column normalization.

Returns
-------
(N,) float array: conditional probabilities of each earlier adult location.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reverse_origin(M: "np.ndarray", current: int, t0: int, t1: int) -> "np.ndarray":
    """Infer the earlier location given an adult location at sampling.

    Parameters
    ----------
    M : row-stochastic daily movement matrix
    current : integer sampled node index
    t0 : integer earlier day
    t1 : integer sampling day

    Returns
    -------
    (N,) float array, conditional probabilities for each earlier node

    Notes
    -----
    Condition on the sampled destination by normalizing the destination column of the forward transition matrix over possible origin nodes. This uses equal origin weights as in the source expression.
    For an earlier node i and sampled current node j, return (M ** (1+t1-t0))[i,j] / sum_h (M ** (1+t1-t0))[h,j]. Do not normalize an origin row.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math

def _oracle_reverse_origin(M: "np.ndarray", current: int, t0: int, t1: int) -> "np.ndarray":
    P = _oracle_adult_transition(M, t0, t1)
    col = P[:, current]
    return col / col.sum()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Three independent source-method scenarios."""
    return [
        {"setup": 'import numpy as np\nimport copy\ncoords=np.array([[0.,0.],[0.,20.],[20.,0.],[20.,20.]])\nM=np.array([[0.6794616371962452, 0.1838483498370596, 0.08640872442341802, 0.050281288543277264], [0.1838483498370596, 0.6794616371962452, 0.050281288543277264, 0.08640872442341802], [0.08640872442341802, 0.050281288543277264, 0.6794616371962452, 0.1838483498370596], [0.05028128854327726, 0.086408724423418, 0.18384834983705958, 0.6794616371962451]],dtype=float)\nM_asym=np.array([[0.789603048166292, 0.17268941606185376, 0.037707535771854234], [0.15266858315453943, 0.6980600280382888, 0.14927138880717172], [0.03785299345582488, 0.16949804651195421, 0.7926489600322209]],dtype=float)\nmo_rows=[(0, 3, 1, 4, 5, 25, 0), (2, 3, 3, 4, 5, 25, 0), (0, 4, 2, 5, 5, 25, 1), (2, 4, 0, 5, 5, 25, 0), (0, 5, 3, 6, 5, 25, 0), (2, 5, 1, 6, 5, 25, 0), (0, 6, 0, 7, 5, 25, 2), (2, 6, 2, 7, 5, 25, 1), (0, 7, 1, 8, 5, 25, 2), (2, 7, 3, 8, 5, 25, 0), (0, 8, 2, 9, 5, 25, 0), (2, 8, 0, 9, 5, 25, 0)]\nfs_rows=[(0, 3, 2, 3, 60, 0), (2, 3, 0, 3, 60, 0), (0, 4, 3, 5, 60, 0), (2, 4, 1, 5, 60, 1), (0, 5, 0, 5, 60, 0), (2, 5, 2, 5, 60, 1), (0, 6, 1, 7, 60, 0), (2, 6, 3, 7, 60, 0), (0, 7, 2, 7, 60, 0), (2, 7, 0, 7, 60, 1), (0, 8, 3, 9, 60, 0), (2, 8, 1, 9, 60, 0)]\ncoords_oracle=coords.copy()\nM_oracle=M.copy()\nM_asym_oracle=M_asym.copy()\nmo_rows_oracle=copy.deepcopy(mo_rows)\nfs_rows_oracle=copy.deepcopy(fs_rows)\n', "call": 'reverse_origin(M,2,1,5)', "gold_call": '_oracle_reverse_origin(M_oracle,2,1,5)', "tol": 1e-09},
        {"setup": 'import numpy as np\nimport copy\ncoords=np.array([[0.,0.],[0.,20.],[20.,0.],[20.,20.]])\nM=np.array([[0.6794616371962452, 0.1838483498370596, 0.08640872442341802, 0.050281288543277264], [0.1838483498370596, 0.6794616371962452, 0.050281288543277264, 0.08640872442341802], [0.08640872442341802, 0.050281288543277264, 0.6794616371962452, 0.1838483498370596], [0.05028128854327726, 0.086408724423418, 0.18384834983705958, 0.6794616371962451]],dtype=float)\nM_asym=np.array([[0.789603048166292, 0.17268941606185376, 0.037707535771854234], [0.15266858315453943, 0.6980600280382888, 0.14927138880717172], [0.03785299345582488, 0.16949804651195421, 0.7926489600322209]],dtype=float)\nmo_rows=[(0, 3, 1, 4, 5, 25, 0), (2, 3, 3, 4, 5, 25, 0), (0, 4, 2, 5, 5, 25, 1), (2, 4, 0, 5, 5, 25, 0), (0, 5, 3, 6, 5, 25, 0), (2, 5, 1, 6, 5, 25, 0), (0, 6, 0, 7, 5, 25, 2), (2, 6, 2, 7, 5, 25, 1), (0, 7, 1, 8, 5, 25, 2), (2, 7, 3, 8, 5, 25, 0), (0, 8, 2, 9, 5, 25, 0), (2, 8, 0, 9, 5, 25, 0)]\nfs_rows=[(0, 3, 2, 3, 60, 0), (2, 3, 0, 3, 60, 0), (0, 4, 3, 5, 60, 0), (2, 4, 1, 5, 60, 1), (0, 5, 0, 5, 60, 0), (2, 5, 2, 5, 60, 1), (0, 6, 1, 7, 60, 0), (2, 6, 3, 7, 60, 0), (0, 7, 2, 7, 60, 0), (2, 7, 0, 7, 60, 1), (0, 8, 3, 9, 60, 0), (2, 8, 1, 9, 60, 0)]\ncoords_oracle=coords.copy()\nM_oracle=M.copy()\nM_asym_oracle=M_asym.copy()\nmo_rows_oracle=copy.deepcopy(mo_rows)\nfs_rows_oracle=copy.deepcopy(fs_rows)\n', "call": 'reverse_origin(M,0,3,3)', "gold_call": '_oracle_reverse_origin(M_oracle,0,3,3)', "tol": 1e-09},
        {"setup": 'import numpy as np\nimport copy\ncoords=np.array([[0.,0.],[0.,20.],[20.,0.],[20.,20.]])\nM=np.array([[0.6794616371962452, 0.1838483498370596, 0.08640872442341802, 0.050281288543277264], [0.1838483498370596, 0.6794616371962452, 0.050281288543277264, 0.08640872442341802], [0.08640872442341802, 0.050281288543277264, 0.6794616371962452, 0.1838483498370596], [0.05028128854327726, 0.086408724423418, 0.18384834983705958, 0.6794616371962451]],dtype=float)\nM_asym=np.array([[0.789603048166292, 0.17268941606185376, 0.037707535771854234], [0.15266858315453943, 0.6980600280382888, 0.14927138880717172], [0.03785299345582488, 0.16949804651195421, 0.7926489600322209]],dtype=float)\nmo_rows=[(0, 3, 1, 4, 5, 25, 0), (2, 3, 3, 4, 5, 25, 0), (0, 4, 2, 5, 5, 25, 1), (2, 4, 0, 5, 5, 25, 0), (0, 5, 3, 6, 5, 25, 0), (2, 5, 1, 6, 5, 25, 0), (0, 6, 0, 7, 5, 25, 2), (2, 6, 2, 7, 5, 25, 1), (0, 7, 1, 8, 5, 25, 2), (2, 7, 3, 8, 5, 25, 0), (0, 8, 2, 9, 5, 25, 0), (2, 8, 0, 9, 5, 25, 0)]\nfs_rows=[(0, 3, 2, 3, 60, 0), (2, 3, 0, 3, 60, 0), (0, 4, 3, 5, 60, 0), (2, 4, 1, 5, 60, 1), (0, 5, 0, 5, 60, 0), (2, 5, 2, 5, 60, 1), (0, 6, 1, 7, 60, 0), (2, 6, 3, 7, 60, 0), (0, 7, 2, 7, 60, 0), (2, 7, 0, 7, 60, 1), (0, 8, 3, 9, 60, 0), (2, 8, 1, 9, 60, 0)]\ncoords_oracle=coords.copy()\nM_oracle=M.copy()\nM_asym_oracle=M_asym.copy()\nmo_rows_oracle=copy.deepcopy(mo_rows)\nfs_rows_oracle=copy.deepcopy(fs_rows)\n', "call": 'reverse_origin(M_asym,1,2,5)', "gold_call": '_oracle_reverse_origin(M_asym_oracle,1,2,5)', "tol": 1e-09},
    ]
