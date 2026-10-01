"""
Return the largest retained predictive path score for a circulant detector fixture. Compose detector_matrix(modulus,offsets) with branch_trajectory() using the supplied log odds and syndrome, then return its first retained score as a native float. The modulus and offset domain is that of detector_matrix(), and all remaining inputs obey branch_trajectory(), including row degree at least rounds+2, n>rounds, finite prior magnitude<=16, nonboolean rounds in [1,6], width in [1,8] and nonboolean integer-valued iteration budgets in [1,32]. The synthetic domain keeps the distinct-result collection limit 64 unreachable. Every upstream domain failure raises ValueError. Return the unrounded score and do not mutate inputs.

The scalar measures the leading retained path's current conditional evidence, not the most likely correction or a global decoding optimum. Its dependence on finite pruning and on the signed within-run reliability construction is the purpose of this deterministic benchmark.

Returns
-------
float, largest retained predictive reliability score
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def terminal_score(modulus, offsets, prior, syndrome, initial_iters=12, inner_iters=8, rounds=6, width=4):
    """Compose the detector construction and finite branching trajectory.

    Parameters
    ----------
    modulus : int
        Circulant modulus in [3,16].
    offsets : array_like
        Distinct within-row circulant support offsets.
    prior : array_like
        Exact input log odds in fault-column order.
    syndrome : array_like
        Binary detector outcomes.
    initial_iters : int
        Root iteration budget in [1,32].
    inner_iters : int
        Child iteration budget in [1,32].
    rounds : int
        Branching depth in [1,6].
    width : int
        Retained population limit in [1,8].

    Returns
    -------
    float
        Largest terminal predictive reliability score, unrounded.

    Raises
    ------
    ValueError
        If any upstream construction or numerical bound fails.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_terminal_score(modulus, offsets, prior, syndrome, initial_iters=12, inner_iters=8, rounds=6, width=4):
    H=_oracle_detector_matrix(modulus,offsets)
    result=_oracle_branch_trajectory(H,prior,syndrome,initial_iters,inner_iters,rounds,width)
    return float(result[0][0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\na=np.array([[0,1,2],[0,1,3],[0,2,4]]);L=np.array([29,47,23,61,37,53,31,43,59,19,41,67,17,71,73,79,83,89,97,101,103])/16;s=np.array([1,0,0,1,1,0,0])',
            'call': 'terminal_score(7,a.copy(),L.copy(),s.copy(),12,8,6,4)',
            'gold_call': '_oracle_terminal_score(7,a.copy(),L.copy(),s.copy(),12,8,6,4)',
        },
        {
            'setup': 'import numpy as np\na=np.array([[0,1,2]]);L=np.array([.25,1.,2.]);s=np.ones(3,int)',
            'call': 'terminal_score(3,a.copy(),L.copy(),s.copy(),1,1,1,1)',
            'gold_call': '_oracle_terminal_score(3,a.copy(),L.copy(),s.copy(),1,1,1,1)',
        },
        {
            'setup': 'import numpy as np\na=np.array([[0,1,2],[0,2,3]]);L=np.array([.25,1.,2.,-.5,1.5,-.25,.75,1.25]);s=np.array([0,1,0,1])',
            'call': 'terminal_score(4,a.copy(),L.copy(),s.copy(),3,4,3,2)',
            'gold_call': '_oracle_terminal_score(4,a.copy(),L.copy(),s.copy(),3,4,3,2)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try: terminal_score(3,[[0,1]],[1,2,3],[0,0,0],1,1,1,1)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_gold():\n    try: _oracle_terminal_score(3,[[0,1]],[1,2,3],[0,0,0],1,1,1,1)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
