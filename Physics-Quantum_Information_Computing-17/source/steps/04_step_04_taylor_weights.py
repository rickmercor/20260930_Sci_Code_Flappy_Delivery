"""
Return the finite-order Taylor mitigation weights for the selected control-reversal construction.

The weights are indexed by increasing amplification setting. They convert the finite set of noisy expectation measurements into the prescribed zero-noise estimate; the physical layer construction is handled separately.

Returns
-------
numpy.ndarray    Real vector of length order+1 in ascending amplification index.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def taylor_weights(order: int) -> "np.ndarray":
    """Return the ordered odd-node Taylor extrapolation weights.

    Parameters
    ----------
    order : int
        Nonboolean integer type in [0,8]; floats are invalid.

    Returns
    -------
    numpy.ndarray
        Real vector of length order+1 in ascending amplification index.

    Raises
    ------
    ValueError
        If order is outside its type or range contract.
    """
    return None


# EXPECTED RETURN LINE
# numpy.ndarray, Taylor coefficients for settings zero through order

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_taylor_weights(order: int) -> "np.ndarray":
    if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or not 0<=order<=8:
        raise ValueError('invalid order')
    m=int(order)
    numerator=math.prod(range(1,2*m+2,2))
    return np.array([(-1)**j*numerator/(2**m*(2*j+1)*math.factorial(j)*math.factorial(m-j)) for j in range(m+1)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases=[
        {'setup':'','call':'taylor_weights(4)','gold_call':'_oracle_taylor_weights(4)'},
        {'setup':'','call':'taylor_weights(0)','gold_call':'_oracle_taylor_weights(0)'},
        {'setup':'','call':'taylor_weights(8)','gold_call':'_oracle_taylor_weights(8)'},
    ]
    for value in ('True','2.0','9','-1'):
        setup=''
        for name,fn in (('run_model','taylor_weights'),('run_gold','_oracle_taylor_weights')):
            setup+=f'def {name}():\n    try:\n        {fn}({value})\n    except ValueError:\n        return 1\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    for case in cases:case['tol']=1e-11
    return cases
