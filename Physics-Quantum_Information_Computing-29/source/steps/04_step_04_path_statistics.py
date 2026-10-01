"""
Return the next branching index, predictive path score and unnormalized reliability sum from one conditional run. For the unmasked index set U, select the smallest index minimizing abs(posterior_sum[j]); compute A=sum over U of abs(posterior_sum[j]) and score=A/iterations. The accumulator contains signed posteriors from the current run only, not a sum of magnitudes or a cumulative path history. posterior_sum is a finite real length-n array with magnitude<=1e12 and 2<=n<=64. fixed is a length-n array in {-1,0,1}, at least one entry is -1 and masked accumulator entries are zero. iterations is a nonboolean integer-valued real scalar in [1,32]. Return three native numeric scalars (index,score,A). Invalid inputs raise ValueError.

Oscillating positive and negative evidence cancels before its absolute value is taken. Dividing the path sum by actual iterations compares runs with different early stopping times, while this division is unnecessary for choosing a node within one run.

Returns
-------
tuple, selected index and two reliability scalars
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def path_statistics(posterior_sum, fixed, iterations):
    """Summarize one run's predictive reliability.

    Parameters
    ----------
    posterior_sum : array_like
        Signed within-run posterior sums, with the bounds above.
    fixed : array_like
        Mask values in {-1,0,1}.
    iterations : int
        Executed iteration count in [1,32].

    Returns
    -------
    tuple
        Selected index, normalized score and absolute reliability sum.

    Raises
    ------
    ValueError
        If the numerical or masking contract is violated.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_path_statistics(posterior_sum, fixed, iterations):
    vals=[]
    for x in (posterior_sum,fixed):
        a=np.asarray(x)
        if a.dtype.kind not in 'biufc' or np.any(a.imag!=0):raise ValueError('nonreal input')
        a=np.asarray(a.real,dtype=float)
        if a.ndim!=1 or not np.all(np.isfinite(a)):raise ValueError('invalid vector')
        vals.append(a)
    a,f=vals
    if not 2<=len(a)<=64 or f.shape!=a.shape or np.any(np.abs(a)>1e12):raise ValueError('shape or magnitude')
    if np.any((f!=-1)&(f!=0)&(f!=1)) or not np.any(f==-1) or np.any(a[f!=-1]!=0):raise ValueError('mask')
    t=np.asarray(iterations)
    if t.ndim!=0 or t.dtype.kind not in 'iuf' or not np.isfinite(t) or t!=np.floor(t) or not 1<=t<=32:raise ValueError('iterations')
    U=np.flatnonzero(f==-1);j=int(U[np.argmin(np.abs(a[U]))]);A=float(np.sum(np.abs(a[U])))
    return j,A/float(t),A

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\na=np.array([-2.,28.125,-9.,39.375]);f=np.full(4,-1)',
            'call': 'path_statistics(a.copy(),f.copy(),12)',
            'gold_call': '_oracle_path_statistics(a.copy(),f.copy(),12)',
        },
        {
            'setup': 'import numpy as np\na=np.zeros(2);f=np.array([-1,0])',
            'call': 'path_statistics(a.copy(),f.copy(),1)',
            'gold_call': '_oracle_path_statistics(a.copy(),f.copy(),1)',
        },
        {
            'setup': 'import numpy as np\na=np.array([0.,-3.,3.,15.]);f=np.array([1,-1,-1,-1])',
            'call': 'path_statistics(a.copy(),f.copy(),3)',
            'gold_call': '_oracle_path_statistics(a.copy(),f.copy(),3)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try: path_statistics([1,2],[0,-1],2)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_gold():\n    try: _oracle_path_statistics([1,2],[0,-1],2)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
