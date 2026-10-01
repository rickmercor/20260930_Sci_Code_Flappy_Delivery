"""
Construct the exact Fourier multiplication matrix of a periodic interval with width fraction f and center c. Retain orders -M through M in increasing order and define T[p,q]=f sinc((p-q)f) exp(-2 pi i (p-q)c), where sinc(x)=sin(pi x)/(pi x) with sinc(0)=1. M is an integer in [0,4], f is a finite real scalar in [0,1] and c is a finite real scalar in [-1,1]. Boolean M is invalid. Return the complex Hermitian matrix; invalid inputs raise ValueError.

The analytic Fourier integral includes every harmonic difference through twice the retained order. Truncating the coefficient list itself at M would discard valid corner entries of this multiplication matrix. Interval centers are fractional periodic coordinates, so translations affect phases rather than the indicator eigenvalues.

Returns
-------
complex ndarray, interval multiplication matrix
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def indicator_matrix(order: int, fraction: float, center: float) -> "np.ndarray":
    """Return the interval's exact finite Fourier multiplication matrix.

    Parameters
    ----------
    order : int
        Maximum retained harmonic from 0 through 4, excluding booleans.
    fraction : float
        Real interval width in [0,1].
    center : float
        Real interval center in [-1,1].

    Returns
    -------
    ndarray
        Complex matrix of shape (2*order+1,2*order+1).

    Raises
    ------
    ValueError
        For invalid type, range or nonfinite scalar.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_indicator_matrix(order: int, fraction: float, center: float) -> "np.ndarray":
    if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or not 0<=order<=4:raise ValueError('order')
    try:
        values=[]
        for x in (fraction,center):
            a=np.asarray(x)
            if a.ndim or np.iscomplexobj(a):raise ValueError('real scalar')
            values.append(float(a))
        f,c=values
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError('scalar') from exc
    if not np.isfinite(f) or not np.isfinite(c) or not 0<=f<=1 or not -1<=c<=1:raise ValueError('range')
    h=np.arange(-order,order+1);d=h[:,None]-h[None,:]
    if f==0:return np.zeros(d.shape,dtype=complex)
    if f==1:return np.eye(len(h),dtype=complex)
    return f*np.sinc(d*f)*np.exp(-2j*np.pi*d*c)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases=[
        {'setup':'import numpy as np','call':'indicator_matrix(1,.43,.11)','gold_call':'_oracle_indicator_matrix(1,.43,.11)'},
        {'setup':'import numpy as np','call':'indicator_matrix(0,.37,-.08)','gold_call':'_oracle_indicator_matrix(0,.37,-.08)'},
        {'setup':'import numpy as np','call':'indicator_matrix(4,1,1)','gold_call':'_oracle_indicator_matrix(4,1,1)'},
        {'setup':'import numpy as np','call':'indicator_matrix(3,0,-1)','gold_call':'_oracle_indicator_matrix(3,0,-1)'},
        {'setup':'import numpy as np','call':'indicator_matrix(2,.00001,.99)','gold_call':'_oracle_indicator_matrix(2,.00001,.99)'}]
    for a in ('True,.4,0','5,.4,0','1,float("nan"),0','1,.5,1j','1,.5,2'):
        setup='import numpy as np\n'
        for name,fn in [('run_model','indicator_matrix'),('run_gold','_oracle_indicator_matrix')]:
            setup+=f'def {name}():\n    try:\n        {fn}({a})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    return cases
