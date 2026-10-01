"""
Calculate spatial mother to larva kinship probability.

A sampled female can be the mother of a sampled larva only if an admissible egg-laying date falls within her possible adult life. Each feasible date contributes adult survival, conditional earlier location and surviving larval output.

Returns
-------
float: probability that the sampled female is mother of the sampled larva.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mother_larva(M: "np.ndarray", x1: int, t1: int, x2: int, t2: int, nf: float, beta: float, te: int, tl: int, ta: int, mu_a: float, mu_e: float, mu_l: float) -> float:
    """Calculate spatial mother to larva kinship probability.

    Parameters
    ----------
    M : adult daily movement matrix
    x1,t1 : sampled mother node and day
    x2,t2 : sampled larva node and day
    nf,beta,te,tl,ta,mu_a,mu_e,mu_l : demographic and stage parameters

    Returns
    -------
    float, probability a sampled female is mother of the sampled larva

    Notes
    -----
    Marginalize integer egg-laying days consistent with larval age and possible maternal age. Apply adult survival, backward maternal origin probability, egg and larval survival, then divide by stationary larval output.
    For each integer y from t2-te-(tl-1) through t2-te, keep t1-ta < y <= t1; add (1-mu_a)**(t1-y) * reverse_origin(M,x1,y,t1)[x2] * beta * (1-mu_e)**te * (1-mu_l)**(t2-y-te), then divide the sum by E_L.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math

def _oracle_mother_larva(M: "np.ndarray", x1: int, t1: int, x2: int, t2: int, nf: float, beta: float, te: int, tl: int, ta: int, mu_a: float, mu_e: float, mu_l: float) -> float:
    total = 0.0
    for y in range(t2-te-(tl-1), t2-te+1):
        if t1-ta < y <= t1:
            q = _oracle_reverse_origin(M, x1, y, t1)[x2]
            total += (1-mu_a)**(t1-y) * q * beta * (1-mu_e)**te * (1-mu_l)**(t2-y-te)
    return total / _oracle_larval_denominator(nf,beta,te,tl,mu_e,mu_l)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Three independent source-method scenarios."""
    return [
        {"setup": 'import numpy as np\nimport copy\ncoords=np.array([[0.,0.],[0.,20.],[20.,0.],[20.,20.]])\nM=np.array([[0.6794616371962452, 0.1838483498370596, 0.08640872442341802, 0.050281288543277264], [0.1838483498370596, 0.6794616371962452, 0.050281288543277264, 0.08640872442341802], [0.08640872442341802, 0.050281288543277264, 0.6794616371962452, 0.1838483498370596], [0.05028128854327726, 0.086408724423418, 0.18384834983705958, 0.6794616371962451]],dtype=float)\nM_asym=np.array([[0.789603048166292, 0.17268941606185376, 0.037707535771854234], [0.15266858315453943, 0.6980600280382888, 0.14927138880717172], [0.03785299345582488, 0.16949804651195421, 0.7926489600322209]],dtype=float)\nmo_rows=[(0, 3, 1, 4, 5, 25, 0), (2, 3, 3, 4, 5, 25, 0), (0, 4, 2, 5, 5, 25, 1), (2, 4, 0, 5, 5, 25, 0), (0, 5, 3, 6, 5, 25, 0), (2, 5, 1, 6, 5, 25, 0), (0, 6, 0, 7, 5, 25, 2), (2, 6, 2, 7, 5, 25, 1), (0, 7, 1, 8, 5, 25, 2), (2, 7, 3, 8, 5, 25, 0), (0, 8, 2, 9, 5, 25, 0), (2, 8, 0, 9, 5, 25, 0)]\nfs_rows=[(0, 3, 2, 3, 60, 0), (2, 3, 0, 3, 60, 0), (0, 4, 3, 5, 60, 0), (2, 4, 1, 5, 60, 1), (0, 5, 0, 5, 60, 0), (2, 5, 2, 5, 60, 1), (0, 6, 1, 7, 60, 0), (2, 6, 3, 7, 60, 0), (0, 7, 2, 7, 60, 0), (2, 7, 0, 7, 60, 1), (0, 8, 3, 9, 60, 0), (2, 8, 1, 9, 60, 0)]\ncoords_oracle=coords.copy()\nM_oracle=M.copy()\nM_asym_oracle=M_asym.copy()\nmo_rows_oracle=copy.deepcopy(mo_rows)\nfs_rows_oracle=copy.deepcopy(fs_rows)\n', "call": 'mother_larva(M,0,4,2,5,nf=25,beta=20,te=2,tl=5,ta=8,mu_a=.09,mu_e=.175,mu_l=.554)', "gold_call": '_oracle_mother_larva(M_oracle,0,4,2,5,nf=25,beta=20,te=2,tl=5,ta=8,mu_a=.09,mu_e=.175,mu_l=.554)', "tol": 1e-09},
        {"setup": 'import numpy as np\nimport copy\ncoords=np.array([[0.,0.],[0.,20.],[20.,0.],[20.,20.]])\nM=np.array([[0.6794616371962452, 0.1838483498370596, 0.08640872442341802, 0.050281288543277264], [0.1838483498370596, 0.6794616371962452, 0.050281288543277264, 0.08640872442341802], [0.08640872442341802, 0.050281288543277264, 0.6794616371962452, 0.1838483498370596], [0.05028128854327726, 0.086408724423418, 0.18384834983705958, 0.6794616371962451]],dtype=float)\nM_asym=np.array([[0.789603048166292, 0.17268941606185376, 0.037707535771854234], [0.15266858315453943, 0.6980600280382888, 0.14927138880717172], [0.03785299345582488, 0.16949804651195421, 0.7926489600322209]],dtype=float)\nmo_rows=[(0, 3, 1, 4, 5, 25, 0), (2, 3, 3, 4, 5, 25, 0), (0, 4, 2, 5, 5, 25, 1), (2, 4, 0, 5, 5, 25, 0), (0, 5, 3, 6, 5, 25, 0), (2, 5, 1, 6, 5, 25, 0), (0, 6, 0, 7, 5, 25, 2), (2, 6, 2, 7, 5, 25, 1), (0, 7, 1, 8, 5, 25, 2), (2, 7, 3, 8, 5, 25, 0), (0, 8, 2, 9, 5, 25, 0), (2, 8, 0, 9, 5, 25, 0)]\nfs_rows=[(0, 3, 2, 3, 60, 0), (2, 3, 0, 3, 60, 0), (0, 4, 3, 5, 60, 0), (2, 4, 1, 5, 60, 1), (0, 5, 0, 5, 60, 0), (2, 5, 2, 5, 60, 1), (0, 6, 1, 7, 60, 0), (2, 6, 3, 7, 60, 0), (0, 7, 2, 7, 60, 0), (2, 7, 0, 7, 60, 1), (0, 8, 3, 9, 60, 0), (2, 8, 1, 9, 60, 0)]\ncoords_oracle=coords.copy()\nM_oracle=M.copy()\nM_asym_oracle=M_asym.copy()\nmo_rows_oracle=copy.deepcopy(mo_rows)\nfs_rows_oracle=copy.deepcopy(fs_rows)\n', "call": 'mother_larva(M_asym,1,4,2,5,nf=25,beta=20,te=2,tl=5,ta=8,mu_a=.09,mu_e=.175,mu_l=.554)', "gold_call": '_oracle_mother_larva(M_asym_oracle,1,4,2,5,nf=25,beta=20,te=2,tl=5,ta=8,mu_a=.09,mu_e=.175,mu_l=.554)', "tol": 1e-09},
        {"setup": 'import numpy as np\nimport copy\ncoords=np.array([[0.,0.],[0.,20.],[20.,0.],[20.,20.]])\nM=np.array([[0.6794616371962452, 0.1838483498370596, 0.08640872442341802, 0.050281288543277264], [0.1838483498370596, 0.6794616371962452, 0.050281288543277264, 0.08640872442341802], [0.08640872442341802, 0.050281288543277264, 0.6794616371962452, 0.1838483498370596], [0.05028128854327726, 0.086408724423418, 0.18384834983705958, 0.6794616371962451]],dtype=float)\nM_asym=np.array([[0.789603048166292, 0.17268941606185376, 0.037707535771854234], [0.15266858315453943, 0.6980600280382888, 0.14927138880717172], [0.03785299345582488, 0.16949804651195421, 0.7926489600322209]],dtype=float)\nmo_rows=[(0, 3, 1, 4, 5, 25, 0), (2, 3, 3, 4, 5, 25, 0), (0, 4, 2, 5, 5, 25, 1), (2, 4, 0, 5, 5, 25, 0), (0, 5, 3, 6, 5, 25, 0), (2, 5, 1, 6, 5, 25, 0), (0, 6, 0, 7, 5, 25, 2), (2, 6, 2, 7, 5, 25, 1), (0, 7, 1, 8, 5, 25, 2), (2, 7, 3, 8, 5, 25, 0), (0, 8, 2, 9, 5, 25, 0), (2, 8, 0, 9, 5, 25, 0)]\nfs_rows=[(0, 3, 2, 3, 60, 0), (2, 3, 0, 3, 60, 0), (0, 4, 3, 5, 60, 0), (2, 4, 1, 5, 60, 1), (0, 5, 0, 5, 60, 0), (2, 5, 2, 5, 60, 1), (0, 6, 1, 7, 60, 0), (2, 6, 3, 7, 60, 0), (0, 7, 2, 7, 60, 0), (2, 7, 0, 7, 60, 1), (0, 8, 3, 9, 60, 0), (2, 8, 1, 9, 60, 0)]\ncoords_oracle=coords.copy()\nM_oracle=M.copy()\nM_asym_oracle=M_asym.copy()\nmo_rows_oracle=copy.deepcopy(mo_rows)\nfs_rows_oracle=copy.deepcopy(fs_rows)\n', "call": 'mother_larva(M,0,0,2,20,nf=25,beta=20,te=2,tl=5,ta=8,mu_a=.09,mu_e=.175,mu_l=.554)', "gold_call": '_oracle_mother_larva(M_oracle,0,0,2,20,nf=25,beta=20,te=2,tl=5,ta=8,mu_a=.09,mu_e=.175,mu_l=.554)', "tol": 1e-09},
    ]
