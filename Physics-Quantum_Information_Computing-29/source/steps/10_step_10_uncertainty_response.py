"""
Compute the uniform mean squared terminal-score change for one uncertain prior. Build H with detector_matrix(), evaluate reference=terminal_score() on the base priors using the supplied schedule and partition the uncertainty interval into exact affine comparison cells using trace_score_cell(). Sum cell_squared_integral(cell,reference) and divide by interval width. Return(mean_squared_change,reference,negative_conditional_mean,positive_conditional_mean) as native floats. lower and upper are integer rational pairs with -1<=lower<0<upper<=1; uncertain_index is a nonboolean integer in [0,n-1]. The direction is one at uncertain_index and zero elsewhere. Use trace_score_cell's domain: m<=8,n<=24,|base|<=16, initial_iters and inner_iters in[1,6], rounds in[1,3], width in[1,2], row degree>=rounds+2. Construction and binary-syndrome limits also apply. Require at most 4096 trace cells; if a new cell cannot be found by testing l+(h-l)/d for d=2,...,257 on an uncovered interval(l,h), raise ValueError. A sample at a nonconstant equality is skipped. Subtract the exact returned rational cell from the uncovered interval, split integration at zero and continue until covered. Return only after full interval coverage. Upstream failures or cap exhaustion raise ValueError.

The expectation averages the square of a statistic difference across uncertain calibrations, not the square of an averaged statistic difference. Reinitializing the decoder at each parameter value is essential: carrying the state across parameter values defines a different experiment. Exact trace cells preserve discontinuous changes in retained histories, while the rational quadratic integral avoids fixed-grid aliasing.

Returns
-------
tuple, squared-response expectation and three determining scalars
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def uncertainty_response(modulus, offsets, base, syndrome, uncertain_index=12, lower=(-1,16), upper=(1,16), initial_iters=4, inner_iters=3, rounds=3, width=2):
    """Integrate squared score changes under a uniform prior perturbation.

    Parameters
    ----------
    modulus : int
        Circulant modulus in [3,8].
    offsets : array_like
        Support offsets satisfying construction and trace-cell size bounds.
    base : array_like
        Base log odds in fault-column order.
    syndrome : array_like
        Binary detector outcomes.
    uncertain_index : int
        Index whose log odds are shifted by the uncertainty variable.
    lower : array_like
        Negative rational lower endpoint as an integer pair.
    upper : array_like
        Positive rational upper endpoint as an integer pair.
    initial_iters : int
        Root budget in [1,6].
    inner_iters : int
        Child budget in [1,6].
    rounds : int
        Branching depth in [1,3].
    width : int
        Retained width in [1,2].

    Returns
    -------
    tuple
        Mean squared change, base score and two half-interval conditional means.

    Raises
    ------
    ValueError
        For unsupported inputs, numerical bounds or partition-cap exhaustion.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_uncertainty_response(modulus, offsets, base, syndrome, uncertain_index=12, lower=(-1,16), upper=(1,16), initial_iters=4, inner_iters=3, rounds=3, width=2):
    from fractions import Fraction as F
    import math
    H=_oracle_detector_matrix(modulus,offsets);n=H.shape[1]
    if H.shape[0]>8 or n>24:raise ValueError('trace size')
    for x,hi in ((initial_iters,6),(inner_iters,6),(rounds,3),(width,2)):
        if isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or not 1<=x<=hi:raise ValueError('schedule')
    if isinstance(uncertain_index,(bool,np.bool_)) or not isinstance(uncertain_index,(int,np.integer)) or not 0<=uncertain_index<n:raise ValueError('uncertain index')
    ends=[]
    for pair in (lower,upper):
        a=np.asarray(pair,dtype=object)
        if a.shape!=(2,) or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in a) or a[1]<=0:raise ValueError('endpoint')
        ends.append(F(int(a[0]),int(a[1])))
    lo,hi=ends
    if not -1<=lo<0<hi<=1:raise ValueError('interval')
    reference=_oracle_terminal_score(modulus,offsets,base,syndrome,initial_iters,inner_iters,rounds,width)
    raw_base=np.asarray(base)
    if raw_base.dtype.kind not in 'biuf':raise ValueError('real base required')
    direction=np.zeros(n);direction[uncertain_index]=1
    pending=[(lo,hi)];left_parts=[];right_parts=[];count=0
    def rp(x):return [int(x.numerator),int(x.denominator)]
    while pending:
        l,h=pending.pop()
        found=False
        for den in range(2,258):
            x=l+(h-l)/den
            try:out=_oracle_trace_score_cell(H,base,direction,syndrome,rp(x),[rp(l),rp(h)],initial_iters,inner_iters,rounds,width)
            except ValueError:
                continue
            found=True;break
        if not found:raise ValueError('cannot select cell interior')
        a,b,u,v=[F(int(p),int(q)) for p,q in out]
        if not l<=a<x<b<=h:raise ValueError('invalid cell')
        count+=1
        if count>4096:raise ValueError('partition cap')
        for aa,bb in ((a,min(b,F(0))),(max(a,F(0)),b)):
            if aa>=bb:continue
            val=_oracle_cell_squared_integral([rp(aa),rp(bb),rp(u),rp(v)],reference)
            (left_parts if bb<=0 else right_parts).append(val)
        if a>l:pending.append((l,a))
        if b<h:pending.append((b,h))
    left=math.fsum(left_parts);right=math.fsum(right_parts)
    return (left+right)/float(hi-lo),reference,left/float(-lo),right/float(hi)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'import numpy as np\na=np.array([[0,1,2]]);b=np.array([.25,1.,2.]);s=np.ones(3,int)',
            'call': 'uncertainty_response(3,a.copy(),b.copy(),s.copy(),1,(-1,16),(1,16),1,1,1,1)',
            'gold_call': '_oracle_uncertainty_response(3,a.copy(),b.copy(),s.copy(),1,(-1,16),(1,16),1,1,1,1)',
        },
        {
            'setup': 'import numpy as np\na=np.array([[0,1,2]]);b=np.zeros(3);s=np.zeros(3,int)',
            'call': 'uncertainty_response(3,a.copy(),b.copy(),s.copy(),0,(-1,32),(1,32),1,1,1,1)',
            'gold_call': '_oracle_uncertainty_response(3,a.copy(),b.copy(),s.copy(),0,(-1,32),(1,32),1,1,1,1)',
        },
        {
            'setup': 'import numpy as np\na=np.array([[0,1,2],[0,2,3]]);b=np.array([.25,1.,2.,-.5,1.5,-.25,.75,1.25]);s=np.array([0,1,0,1])',
            'call': 'uncertainty_response(4,a.copy(),b.copy(),s.copy(),2,(-1,64),(1,32),2,2,1,2)',
            'gold_call': '_oracle_uncertainty_response(4,a.copy(),b.copy(),s.copy(),2,(-1,64),(1,32),2,2,1,2)',
        },
        {
            'setup': 'import numpy as np\ndef run_model():\n    try: uncertainty_response(3,[[0,1,2]],[0,0,0],[0,0,0],3)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0\ndef run_gold():\n    try: _oracle_uncertainty_response(3,[[0,1,2]],[0,0,0],[0,0,0],3)\n    except ValueError: return 1\n    except Exception: return 2\n    return 0',
            'call': 'run_model()',
            'gold_call': 'run_gold()',
        },
    ]
