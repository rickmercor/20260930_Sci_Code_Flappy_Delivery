"""
Integrate a squared affine deviation over a rational interval. cell is a four-by-two integer table encoding lower endpoint a, upper endpoint b, intercept u and slope v as reduced or unreduced numerator/positive-denominator pairs. Require -1<=a<b<=1, |u|<=1e12 and |v|<=1e12. reference is a finite nonboolean real scalar with magnitude<=1e12 and is interpreted as its exact binary64 value. Return the native float integral of (u+v*x-reference)^2 on [a,b]. Compute this as (b-a)*(z_a^2+z_a*z_b+z_b^2)/3 using exact rational arithmetic before the final float conversion, where z_a=u+v*a-reference and z_b=u+v*b-reference. Invalid inputs raise ValueError.

On an interval with unchanged comparison decisions, the min-sum decoder score is affine in the uncertain prior. Squaring its displacement from the unperturbed score gives a quadratic integrand, so interval integration avoids smearing a jump across its boundary. Values at isolated endpoints have zero probability under the continuous uncertainty distribution.

Returns
-------
float, integrated quadratic deviation over one parameter cell
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def cell_squared_integral(cell, reference):
    """Integrate the squared deviation on one affine cell.

    Parameters
    ----------
    cell : array_like
        Four rational pairs for lower, upper, intercept and slope.
    reference : float
        Reference score within the stated finite bound.

    Returns
    -------
    float
        Integral over the cell, without probability normalization.

    Raises
    ------
    ValueError
        If rational shapes, denominator signs or numerical bounds fail.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cell_squared_integral(cell, reference):
    from fractions import Fraction as F
    c=np.asarray(cell,dtype=object)
    if c.shape!=(4,2):raise ValueError('cell shape')
    for x in c.flat:
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)):raise ValueError('integer rational pair')
    if any(row[1]<=0 for row in c):raise ValueError('denominator')
    a,b,u,v=[F(int(n),int(d)) for n,d in c]
    q=np.asarray(reference)
    if q.ndim!=0 or q.dtype.kind not in 'iuf' or not np.isfinite(q) or abs(float(q))>1e12:raise ValueError('reference')
    if not -1<=a<b<=1 or abs(u)>1e12 or abs(v)>1e12:raise ValueError('coefficient domain')
    r=F(float(q));za=u+v*a-r;zb=u+v*b-r
    return float((b-a)*(za*za+za*zb+zb*zb)/3)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\nc=[[-1,16],[1,16],[5,2],[3,1]]',
            'call': 'cell_squared_integral([row[:] for row in c],2.)',
            'gold_call': '_oracle_cell_squared_integral([row[:] for row in c],2.)',
        },
        {
            'setup': 'import numpy as np\nc=[[0,1],[1,2],[0,1],[0,1]]',
            'call': 'cell_squared_integral([row[:] for row in c],0.)',
            'gold_call': '_oracle_cell_squared_integral([row[:] for row in c],0.)',
        },
        {
            'setup': 'import numpy as np\nc=[[1,3],[2,3],[1000000000000,1],[-3,1]]',
            'call': 'cell_squared_integral([row[:] for row in c],1e12)',
            'gold_call': '_oracle_cell_squared_integral([row[:] for row in c],1e12)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try: cell_squared_integral([[0,0],[1,1],[1,1],[1,1]],0.)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_gold():\n    try: _oracle_cell_squared_integral([[0,0],[1,1],[1,1],[1,1]],0.)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
