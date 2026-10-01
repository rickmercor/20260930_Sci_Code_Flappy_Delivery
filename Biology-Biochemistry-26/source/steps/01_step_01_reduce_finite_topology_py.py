"""
Reduce a finite simplicial filtration to dimension-wise total finite persistence.

For a nonempty finite simplicial filtration, return the three total-persistence observables used by the matching protein-assembly treatment, in dimension order 0, 1, 2. Use coefficients in the field with two elements; exclude essential classes and zero-length intervals. Each simplex is a sorted tuple of distinct nonnegative integer vertex labels, of size 1 through 4. Every nonempty face must occur once, with filtration no larger than its coface. filtration is a finite float vector aligned to the supplied, possibly shuffled simplex order. Resolve equal filtration values by dimension and then lexicographic simplex order. Return a float array of shape (3,). Invalid shapes, noninteger vertices, duplicates, missing faces, nonfinite values or nonmonotonicity raise ValueError. Negative filtration values are valid. Do not mutate inputs.

Returns
-------
persistence_totals : np.ndarray — float array of shape (3,) in dimension order 0, 1, 2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def reduce_finite_topology(
    simplices: list[tuple[int, ...]],
    filtration: np.ndarray,
) -> np.ndarray:
    """Return finite total persistence in dimensions zero through two.
 
    Parameters
    ----------
    simplices
        Nonempty face-complete collection of sorted simplex tuples.
    filtration
        Finite filtration values aligned with ``simplices``.
 
    Returns
    -------
    np.ndarray
        Float array of shape (3,) ordered by homology dimension.
    """
    return persistence_totals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reduce_finite_topology(
    simplices: list[tuple[int, ...]],
    filtration: np.ndarray,
) -> np.ndarray:
    import numpy as np
    import itertools
    if any(any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) for v in x) for x in simplices):
        raise ValueError('integer vertex labels required')
    s=[tuple(int(v) for v in x) for x in simplices]
    f=np.asarray(filtration,dtype=float)
    if f.shape!=(len(s),) or not len(s) or not np.all(np.isfinite(f)):
        raise ValueError('finite filtration vector must match nonempty simplex list')
    if len(set(s))!=len(s) or any(not 1<=len(x)<=4 or tuple(sorted(set(x)))!=x or min(x)<0 for x in s):
        raise ValueError('unique nonnegative sorted simplices of dimensions zero to three required')
    lookup={x:i for i,x in enumerate(s)}
    for j,x in enumerate(s):
        for face in itertools.combinations(x,len(x)-1) if len(x)>1 else []:
            if face not in lookup or f[lookup[face]]>f[j]:
                raise ValueError('face missing or filtration not monotone')
    order=sorted(range(len(s)),key=lambda j:(f[j],len(s[j]),s[j]))
    ss=[s[j] for j in order]; ff=f[order]
    loc={x:i for i,x in enumerate(ss)}
    pivots={}; total=np.zeros(3)
    for j,x in enumerate(ss):
        col=0
        if len(x)>1:
            for face in itertools.combinations(x,len(x)-1):
                col ^= 1<<loc[face]
        while col:
            low=col.bit_length()-1
            if low not in pivots: break
            col ^= pivots[low]
        if col:
            low=col.bit_length()-1
            pivots[low]=col
            dim=len(ss[low])-1
            if dim<3 and ff[j]>ff[low]: total[dim]+=ff[j]-ff[low]
    return total

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               's=[(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)]\n'
               'f=np.array([-.5,0,.1,1.,1.2,1.4,2.5])\n',
      'call': 'reduce_finite_topology(s,f)',
      'gold_call': '_oracle_reduce_finite_topology(s,f)'},
     {'setup': 'import numpy as np\n'
               'import itertools\n'
               's=[x for k in range(1,5) for x in itertools.combinations(range(4),k)]\n'
               'f=np.array([len(x)-1.0 for x in s])\n',
      'call': 'reduce_finite_topology(s,f)',
      'gold_call': '_oracle_reduce_finite_topology(s,f)'},
     {'setup': 'import numpy as np\n'
               's=[(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)]\n'
               'f=np.array([-.5,0,.1,1.,1.2,1.4,2.5])\n'
               'f[:]=0\n',
      'call': 'reduce_finite_topology(s,f)',
      'gold_call': '_oracle_reduce_finite_topology(s,f)'},
     {'setup': 'import numpy as np\n'
               's=[(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)]\n'
               'f=np.array([-.5,0,.1,1.,1.2,1.4,2.5])\n'
               's=s[:-1];f=f[:-1]\n',
      'call': 'reduce_finite_topology(s,f)',
      'gold_call': '_oracle_reduce_finite_topology(s,f)'},
     {'setup': 'import numpy as np\n'
               's=[(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)]\n'
               'f=np.array([-.5,0,.1,1.,1.2,1.4,2.5])\n'
               's[0]=(0.5,)\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        reduce_finite_topology(s,f)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_reduce_finite_topology(s,f)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
