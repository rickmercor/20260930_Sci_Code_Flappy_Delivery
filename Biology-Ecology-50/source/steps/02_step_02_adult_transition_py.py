"""
Propagate adult locations using the inclusive daily movement convention.

Adult movement is indexed by discrete days. A comparison of an earlier and a later adult location allows movement on both endpoint days, which changes the required matrix power.

Returns
-------
(N,N) float array: adult location transition probabilities between two days.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def adult_transition(M: "np.ndarray", t0: int, t1: int) -> "np.ndarray":
    """Propagate adult locations using the inclusive daily movement convention.

    Parameters
    ----------
    M : row-stochastic daily movement matrix
    t0 : integer origin day
    t1 : integer destination day with t1 >= t0

    Returns
    -------
    (N,N) float array, adult transition probabilities

    Notes
    -----
    Return the paper's forward movement probability for the interval: the daily movement matrix raised to 1+t1-t0. The same-day transition therefore includes one movement opportunity. Raise ValueError if t1<t0.
    For all ordered nodes i,j, the transition from day t0 to t1 is (M ** (1+t1-t0))[i,j], where ** denotes matrix power.

    Raises
    ------
    ValueError : if the destination day precedes the origin day.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import math

def _oracle_adult_transition(M: "np.ndarray", t0: int, t1: int) -> "np.ndarray":
    if t1 < t0:
        raise ValueError("Adult movement must be forward in time")
    return np.linalg.matrix_power(M, 1 + t1 - t0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Three independent source-method scenarios."""
    return [
        {"setup": 'import numpy as np\nM=np.array([[0.6794616371962452, 0.1838483498370596, 0.08640872442341802, 0.050281288543277264], [0.1838483498370596, 0.6794616371962452, 0.050281288543277264, 0.08640872442341802], [0.08640872442341802, 0.050281288543277264, 0.6794616371962452, 0.1838483498370596], [0.05028128854327726, 0.086408724423418, 0.18384834983705958, 0.6794616371962451]],dtype=float)\nM_oracle=M.copy()\n', "call": 'adult_transition(M,2,5)', "gold_call": '_oracle_adult_transition(M_oracle,2,5)', "tol": 1e-09},
        {"setup": 'import numpy as np\nM=np.array([[0.6794616371962452, 0.1838483498370596, 0.08640872442341802, 0.050281288543277264], [0.1838483498370596, 0.6794616371962452, 0.050281288543277264, 0.08640872442341802], [0.08640872442341802, 0.050281288543277264, 0.6794616371962452, 0.1838483498370596], [0.05028128854327726, 0.086408724423418, 0.18384834983705958, 0.6794616371962451]],dtype=float)\nM_oracle=M.copy()\n', "call": 'adult_transition(M,3,3)', "gold_call": '_oracle_adult_transition(M_oracle,3,3)', "tol": 1e-09},
        {"setup": 'import numpy as np\nM=np.array([[0.6794616371962452, 0.1838483498370596, 0.08640872442341802, 0.050281288543277264], [0.1838483498370596, 0.6794616371962452, 0.050281288543277264, 0.08640872442341802], [0.08640872442341802, 0.050281288543277264, 0.6794616371962452, 0.1838483498370596], [0.05028128854327726, 0.086408724423418, 0.18384834983705958, 0.6794616371962451]],dtype=float)\nM_oracle=M.copy()\n', "call": 'adult_transition(np.eye(2),0,3)', "gold_call": '_oracle_adult_transition(np.eye(2),0,3)', "tol": 1e-09},
    ]
