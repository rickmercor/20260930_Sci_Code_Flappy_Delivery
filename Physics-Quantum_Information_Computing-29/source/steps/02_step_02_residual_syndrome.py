"""
Return the residual syndrome after fixing selected binary faults. XOR the original syndrome with each column of H whose fixed value is one. H is a binary array with shape (m,n), 1<=m<=32 and 2<=n<=64. The binary syndrome has length m and fixed has length n with entries -1, 0 or 1; -1 denotes an unmasked fault. Every detector must retain at least two unmasked neighbors. These synthetic domain restrictions keep every later min-sum reduction nonempty. Real numeric array inputs, including zero-imaginary complex arrays, are accepted. Invalid shape, finiteness, value or live-degree conditions raise ValueError. Return a length-m integer array without mutating inputs.

Conditioning on a known fault toggles exactly the detectors it flips. Removing fixed-one columns without toggling the syndrome would solve a different conditional inference problem. Fixed-zero columns do not alter the residual parity.

Returns
-------
numpy.ndarray, residual detector parity
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def residual_syndrome(H, syndrome, fixed):
    """Condition binary detector parity on fixed faults.

    Parameters
    ----------
    H : array_like
        Binary matrix with the stated synthetic size limits.
    syndrome : array_like
        Original binary detector outcomes.
    fixed : array_like
        Mask values -1, 0 or 1.

    Returns
    -------
    numpy.ndarray
        Residual binary syndrome.

    Raises
    ------
    ValueError
        If any domain or live-degree condition fails.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_residual_syndrome(H, syndrome, fixed):
    arrays=[]
    for x in (H,syndrome,fixed):
        a=np.asarray(x)
        if a.dtype.kind not in 'biufc' or np.any(np.imag(a)!=0):raise ValueError('nonreal input')
        a=np.asarray(a.real,dtype=float)
        if not np.all(np.isfinite(a)):raise ValueError('nonfinite input')
        arrays.append(a)
    h,s,f=arrays
    if h.ndim!=2 or not 1<=h.shape[0]<=32 or not 2<=h.shape[1]<=64:raise ValueError('matrix shape')
    m,n=h.shape
    if s.shape!=(m,) or f.shape!=(n,):raise ValueError('vector shape')
    if np.any((h!=0)&(h!=1)) or np.any((s!=0)&(s!=1)) or np.any((f!=-1)&(f!=0)&(f!=1)):raise ValueError('invalid discrete value')
    if np.any(h[:,f<0].sum(axis=1)<2):raise ValueError('too few live neighbors')
    return ((s+h[:,f==1].sum(axis=1))%2).astype(int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nH=np.array([[1,1,1,0],[0,1,1,1]]);s=np.array([1,0]);f=np.array([-1,-1,-1,1])',
            'call': 'residual_syndrome(H.copy(),s.copy(),f.copy())',
            'gold_call': '_oracle_residual_syndrome(H.copy(),s.copy(),f.copy())',
        },
        {
            'setup': 'import numpy as np\nH=np.ones((1,2),int);s=np.array([1]);f=np.array([-1,-1])',
            'call': 'residual_syndrome(H.copy(),s.copy(),f.copy())',
            'gold_call': '_oracle_residual_syndrome(H.copy(),s.copy(),f.copy())',
        },
        {
            'setup': 'import numpy as np\nH=np.array([[1,1,1,1,0],[0,1,1,1,1]]);s=np.array([0,1]);f=np.array([1,-1,-1,1,0])',
            'call': 'residual_syndrome(H.copy(),s.copy(),f.copy())',
            'gold_call': '_oracle_residual_syndrome(H.copy(),s.copy(),f.copy())',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try: residual_syndrome([[1,1]],[0],[1,-1])\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_gold():\n    try: _oracle_residual_syndrome([[1,1]],[0],[1,-1])\n    except ValueError: return 1\n    except Exception: return 2\n    return 0',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
