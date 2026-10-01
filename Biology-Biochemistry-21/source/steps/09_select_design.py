"""
Select the best eligible design and its eligible runner-up.

Returns
-------
winner index and score, then runner-up index and score.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_design(panel: "np.ndarray", tie_tolerance: float) -> "np.ndarray":
    """Select eligible designs by cost-normalized robust score.

    Parameters
    ----------
    panel
        Finite design-by-thirteen matrix. Column zero is eligibility and
        column one is score.
    tie_tolerance
        Finite nonnegative tolerance. Among scores within this distance of a
        maximum, the lowest design index is selected.

    Returns
    -------
    np.ndarray
        Winner index, winner score, runner-up index, and runner-up score. If
        only one design is eligible, both runner-up entries are -1.

    Raises
    ------
    ValueError
        If panel shape, eligibility indicators, scores or tolerance are
        invalid, or if no design is eligible.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _sd9_pick(indices,scores,tolerance):
    maximum=float(np.max(scores)); tied=indices[scores>=maximum-tolerance]
    chosen=int(np.min(tied))
    return chosen,float(scores[np.flatnonzero(indices==chosen)[0]])


def _oracle_select_design(panel: "np.ndarray", tie_tolerance: float) -> "np.ndarray":
    panel=np.asarray(panel,dtype=np.float64)
    if (panel.ndim!=2 or panel.shape[0]==0 or panel.shape[1]!=13
            or not np.isfinite(panel).all() or not np.isfinite(tie_tolerance)
            or tie_tolerance<0.0
            or np.any((panel[:,0]!=0.0)&(panel[:,0]!=1.0))):
        raise ValueError("selection inputs violate the stated contract")
    eligible=np.flatnonzero(panel[:,0]==1.0)
    if eligible.size==0:
        raise ValueError("at least one design must be eligible")
    winner,winner_score=_sd9_pick(eligible,panel[eligible,1],tie_tolerance)
    remaining=eligible[eligible!=winner]
    if remaining.size==0:
        runner,runner_score=-1,-1.0
    else:
        runner,runner_score=_sd9_pick(remaining,panel[remaining,1],tie_tolerance)
    return np.array([float(winner),winner_score,float(runner),runner_score])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
p=np.zeros((4,13))
p[:,0]=[1,0,1,1]
p[:,1]=[.3,.9,.5,.4]
cp=p.copy()
gp=p.copy()""",
            "call": "select_design(cp,1e-10)",
            "gold_call": "_oracle_select_design(gp,1e-10)",
            "tol": 0.0,
        },
        {
            "setup": """import numpy as np
p=np.zeros((3,13))
p[:,0]=1
p[:,1]=[.5,.50000000004,.4]
cp=p.copy()
gp=p.copy()""",
            "call": "select_design(cp,1e-10)",
            "gold_call": "_oracle_select_design(gp,1e-10)",
            "tol": 0.0,
        },
        {
            "setup": """import numpy as np
p=np.zeros((2,13))
p[1,0]=1
p[:,1]=[.7,.2]
cp=p.copy()
gp=p.copy()""",
            "call": "select_design(cp,0.)",
            "gold_call": "_oracle_select_design(gp,0.)",
            "tol": 0.0,
        },
        {
            "setup": """import numpy as np
p=np.zeros((2,13))
cp=p.copy()
gp=p.copy()
def case_raises(fn,*args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0""",
            "call": "case_raises(select_design,cp,0.)",
            "gold_call": "case_raises(_oracle_select_design,gp,0.)",
            "tol": 0.0,
        },
    ]
