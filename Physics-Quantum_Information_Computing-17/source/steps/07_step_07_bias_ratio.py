"""
Return the ordinary bivariate power-series coefficients of the signed residual bias fraction defined in the global instance.

The expectation stack contains one coefficient rectangle per increasing amplification setting, with the unamplified expectation first. The supplied weights define the finite-order estimate and the ideal reference is constant in both noise perturbations. All inputs after binary64 conversion are treated as exact real values, including near cancellation; weights are not renormalized.

Returns
-------
numpy.ndarray    Real ratio coefficient rectangle (P+1,Q+1), where [a,b] multiplies    x^a*y^b. Require absolute and relative accuracy 1e-9 even under    cancellation. Composition must meet the final residual tolerance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bias_ratio(expectations: "np.ndarray", weights: "np.ndarray", ideal: float) -> "np.ndarray":
    """Return power-series coefficients of the signed residual ratio.

    Parameters
    ----------
    expectations : array_like
        Real exact expectation coefficients (J,P+1,Q+1).
    weights : array_like
        Matching real extrapolation weights (J,).
    ideal : float
        Constant noise-free reference expectation.

    Returns
    -------
    numpy.ndarray
        Real ratio coefficient rectangle (P+1,Q+1), where [a,b] multiplies
        x^a*y^b. Require absolute and relative accuracy 1e-9 even under
        cancellation. Composition must meet the final residual tolerance.

    Raises
    ------
    ValueError
        If expectations is not (J,P+1,Q+1) with J in [1,9], P,Q in [0,2],
        weights is not (J,), or ideal is not a nonboolean real scalar.
        All inputs must be finite real with magnitudes<=100; weights sum
        to one within absolute tolerance 1e-10 and the unamplified constant
        expectation minus ideal must have magnitude>1e-10.
        Nonfinite output also raises ValueError.
    """
    return None


# EXPECTED RETURN LINE
# numpy.ndarray, bivariate power-series coefficient rectangle of the signed residual ratio

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bias_ratio(expectations: "np.ndarray", weights: "np.ndarray", ideal: float) -> "np.ndarray":
    from fractions import Fraction as F
    e=np.asarray(expectations);w=np.asarray(weights);i=np.asarray(ideal)
    for a in (e.copy(),w.copy(),i):
        if a.dtype.kind not in 'iuf' or not np.all(np.isfinite(a)) or np.any(np.abs(a)>100):
            raise ValueError('invalid numeric input')
    if e.ndim!=3 or not 1<=e.shape[0]<=9 or not 1<=e.shape[1]<=3 or not 1<=e.shape[2]<=3 or w.shape!=(e.shape[0],) or i.ndim!=0:
        raise ValueError('invalid shape')
    weights_exact=[F(float(x)) for x in w]
    if abs(sum(weights_exact,F(0))-1)>F(1e-10):
        raise ValueError('invalid weight sum')
    p,q=e.shape[1:];ideal_exact=F(float(i))
    a=[[sum((weights_exact[j]*F(float(e[j,r,s])) for j in range(len(w))),F(0)) for s in range(q)] for r in range(p)]
    b=[[F(float(e[0,r,s])) for s in range(q)] for r in range(p)]
    a[0][0]-=ideal_exact;b[0][0]-=ideal_exact
    if abs(b[0][0])<=F(1e-10):raise ValueError('raw bias too small')
    result=[[F(0) for _ in range(q)] for _ in range(p)]
    for r in range(p):
        for s in range(q):
            value=a[r][s]
            for u in range(r+1):
                for v in range(s+1):
                    if u or v:value-=b[u][v]*result[r-u][s-v]
            result[r][s]=value/b[0][0]
    out=np.array([[float(x) for x in row] for row in result])
    if not np.all(np.isfinite(out)):raise ValueError('non-finite ratio')
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base='e=np.array([[[.7,.02,-.001],[.03,.004,.0002],[-.002,.0003,.0001]],[[.8,.05,-.002],[.08,.009,.0005],[-.004,.0007,.0002]]]);w=np.array([1.5,-.5])'
    cases=[
        {'setup':base,'call':'bias_ratio(e.copy(),w.copy(),.65)','gold_call':'_oracle_bias_ratio(e.copy(),w.copy(),.65)'},
        {'setup':'e=np.array([[[.4]]]);w=np.array([1.])','call':'bias_ratio(e.copy(),w.copy(),.2)','gold_call':'_oracle_bias_ratio(e.copy(),w.copy(),.2)'},
        {'setup':base+';e=e[:,:,:1]','call':'bias_ratio(e.copy(),w.copy(),.65)','gold_call':'_oracle_bias_ratio(e.copy(),w.copy(),.65)'},
        {'setup':'e=np.array([.70000000017,.70000000051,.7000000008499999,.7000000011899999,.70000000153])[:,None,None];w=np.array([2.4609375,-3.28125,2.953125,-1.40625,.2734375])','call':'bias_ratio(e.copy(),w.copy(),.7)[0,0]','gold_call':'_oracle_bias_ratio(e.copy(),w.copy(),.7)[0,0]'},
    ]
    for change in ('e[0,0,0]=.65','w*=.9','e[0,0,1]=np.nan'):
        setup=base+'\n'+change+'\n'
        for name,fn in (('run_model','bias_ratio'),('run_gold','_oracle_bias_ratio')):
            setup+=f'def {name}():\n    try:\n        {fn}(e.copy(),w.copy(),.65)\n    except ValueError:\n        return 1\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    for case in cases:
        case['setup']='import numpy as np\n'+case['setup']
        case['tol']=1e-9
    return cases
