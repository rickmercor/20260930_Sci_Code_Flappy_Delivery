"""
Apply the cross-panel soundness and interval gate, then select the maximin eligible candidate

Select a robust candidate from certificates of shape (S,K+1,4), with S,K>=1 and the pure-geometric baseline at setup zero. Certificate columns are [nonnegative_integer_count, lower_bound, upper_bound, soundness_flag]; bounds obey 0<=lower<=upper<=1 and the flag is 0 or 1. A candidate is eligible only if every system has flag 1 and its lower bound is strictly larger than that system's baseline upper bound. Its score is its smallest lower bound across systems. Maximize that score among eligible candidates; exact ties choose the smallest one-based candidate index. Return a float vector [winner_index, winning_score, eligibility_flags_in_candidate_order]. Nonfinite or malformed certificates and a panel with no eligible candidate raise ValueError. Do not mutate inputs.

Returns
-------
selection : np.ndarray — float vector [winner_index, winning_score, eligibility_flags_in_candidate_order] of length K+2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def select_robust_candidate(certificates: np.ndarray) -> np.ndarray:
    """Return the one-based maximin winner, score, and eligibility flags.
 
    Parameters
    ----------
    certificates
        Valid array of shape (S, K+1, 4), baseline at setup zero.
 
    Returns
    -------
    np.ndarray
        Float vector of length K+2.
    """
    return selection

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_robust_candidate(certificates: np.ndarray) -> np.ndarray:
    import numpy as np
    c=np.asarray(certificates,dtype=float)
    if c.ndim!=3 or c.shape[0]<1 or c.shape[1]<2 or c.shape[2]!=4 or not np.all(np.isfinite(c)):
        raise ValueError('certificates[system,setup,4], baseline at setup zero required')
    if np.any(c[:,:,0]<0) or np.any(c[:,:,0]!=np.floor(c[:,:,0])) or np.any(c[:,:,1:3]<0) or np.any(c[:,:,1:3]>1) or np.any(c[:,:,1]>c[:,:,2]) or np.any(~np.isin(c[:,:,3],[0,1])):
        raise ValueError('invalid certificate values')
    eligible=np.all((c[:,1:,3]==1)&(c[:,1:,1]>c[:,0:1,2]),axis=0)
    score=np.min(c[:,1:,1],axis=0)
    if not np.any(eligible): raise ValueError('no eligible candidate')
    winner=int(np.flatnonzero(eligible)[np.argmax(score[eligible])])+1
    return np.concatenate([[winner,score[winner-1]],eligible.astype(float)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'c=np.array([[[1,.01,.2,1],[8,.6,.9,1],[9,.7,.95,1]],[[1,.01,.2,1],[8,.6,.9,1],[9,.7,.95,0]]])\n',
      'call': 'select_robust_candidate(c)',
      'gold_call': '_oracle_select_robust_candidate(c)'},
     {'setup': 'import numpy as np\nc=np.array([[[0,0,.1,1],[8,.6,.9,1],[8,.6,.9,1]]])\n',
      'call': 'select_robust_candidate(c)',
      'gold_call': '_oracle_select_robust_candidate(c)'},
     {'setup': 'import numpy as np\n'
               'c=np.array([[[0,0,.25,1],[5,.25,.8,1],[5,np.nextafter(.25,1.),.8,1]]])\n',
      'call': 'select_robust_candidate(c)',
      'gold_call': '_oracle_select_robust_candidate(c)'},
     {'setup': 'import numpy as np\n'
               'c=np.array([[[0,0,.3,1],[1,.2,.4,1]]])\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        select_robust_candidate(c)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_select_robust_candidate(c)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
