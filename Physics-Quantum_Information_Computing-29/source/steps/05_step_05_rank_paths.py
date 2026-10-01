"""
Return indices of the highest-scoring paths in deterministic order. Sort scores descending, break equal scores by the lexicographic sequence of (fault index,fixed value) pairs in each path and break identical keys by original candidate order. Keep at most width indices. scores is a finite nonnegative length-c array bounded by 1e12, with 1<=c<=64. paths has shape (c,d,2), where 0<=d<=6; entries are integer-valued, fault indices are in [0,63], values are binary and no index repeats inside a path. width is a nonboolean integer in [1,32]. Return an integer vector of length min(c,width). Invalid inputs raise ValueError; inputs are not mutated.

Pruning imposes a finite computational budget on the branching search. Ties are benchmark conventions rather than additional evidence for a correction, so they must not depend on container iteration order. The path history provides an explicit reproducible key.

Returns
-------
numpy.ndarray, ordered retained candidate indices
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def rank_paths(scores, paths, width):
    """Rank candidate histories by score then lexicographic path.

    Parameters
    ----------
    scores : array_like
        Finite nonnegative path scores.
    paths : array_like
        Candidate history array of shape (c,d,2).
    width : int
        Maximum retained population in [1,32].

    Returns
    -------
    numpy.ndarray
        Retained indices in exploration order.

    Raises
    ------
    ValueError
        If any score, path or width constraint fails.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_rank_paths(scores, paths, width):
    s=np.asarray(scores);p=np.asarray(paths)
    if s.dtype.kind not in 'iuf' or s.ndim!=1 or not 1<=len(s)<=64 or not np.all(np.isfinite(s)) or np.any(s<0) or np.any(s>1e12):raise ValueError('scores')
    if p.dtype.kind not in 'iuf' or p.ndim!=3 or p.shape[0]!=len(s) or not 0<=p.shape[1]<=6 or p.shape[2]!=2 or not np.all(np.isfinite(p)) or np.any(p!=np.floor(p)):raise ValueError('paths')
    if np.any(p[:,:,0]<0) or np.any(p[:,:,0]>63) or np.any((p[:,:,1]!=0)&(p[:,:,1]!=1)):raise ValueError('path values')
    if any(len(set(a[:,0].tolist()))!=len(a) for a in p):raise ValueError('repeated index')
    if isinstance(width,(bool,np.bool_)) or not isinstance(width,(int,np.integer)) or not 1<=width<=32:raise ValueError('width')
    return np.array(sorted(range(len(s)),key=lambda i:(-float(s[i]),tuple(map(tuple,p[i].tolist())),i))[:width],dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\ns=np.array([4.,9.,9.,2.]);p=np.array([[[0,1]],[[3,0]],[[1,1]],[[2,0]]])',
            'call': 'rank_paths(s.copy(),p.copy(),2)',
            'gold_call': '_oracle_rank_paths(s.copy(),p.copy(),2)',
        },
        {
            'setup': 'import numpy as np\ns=np.array([0.]);p=np.empty((1,0,2),int)',
            'call': 'rank_paths(s.copy(),p.copy(),1)',
            'gold_call': '_oracle_rank_paths(s.copy(),p.copy(),1)',
        },
        {
            'setup': 'import numpy as np\ns=np.array([3.,3.,3.]);p=np.array([[[2,1],[0,0]],[[2,0],[1,1]],[[2,1],[0,0]]])',
            'call': 'rank_paths(s.copy(),p.copy(),3)',
            'gold_call': '_oracle_rank_paths(s.copy(),p.copy(),3)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try: rank_paths([1],[[[2,0],[2,1]]],1)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_gold():\n    try: _oracle_rank_paths([1],[[[2,0],[2,1]]],1)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
