"""
Return the scaled mixed noise susceptibility of the specified two-qutrit dynamic circuit by composing the seven upstream operations.

The same model, fixed three-pulse layer family and outcome-conditioned feedback as in the global instance are used, with configurable noise scale, total forward duration, even layer count and mitigation order. The layer count is split equally around the instantaneous measurement. The two perturbations multiply the first two rates by 1+x and the remaining four rates by 1+y. The ideal reference uses the same controls and instrument with zero rates. The supported numerical envelope bounds noise scale and duration away from vanishing raw bias and limits the cancellation order; it is a synthetic precision domain, not a claimed experimental validity range.

Returns
-------
float    Unrounded 1000*d^(px+py)R/(dx^px dy^py) at zero, including the    ordinary-series factorial conversion. Required absolute and    relative accuracy is 1e-8. Passing upstream isolated tolerances    does not establish this end-to-end accuracy.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def residual_bias(noise_scale: float = 0.06, total_time: float = 1.0, layers: int = 6, order: int = 4, feedforward: float = 0.8, px: int = 2, py: int = 2) -> float:
    """Return 1000 times the mixed derivative of the signed residual ratio.

    Parameters
    ----------
    noise_scale : float
        Finite nonboolean rate multiplier in [0.04,0.3], default 0.06.
    total_time : float
        Finite nonboolean forward duration in [0.6,2], default 1.0.
    layers : int
        Nonboolean even integer in [2,12], default 6.
    order : int
        Nonboolean integer Taylor mitigation order in [0,4], default 4.
    feedforward : float
        Finite nonboolean feedback amplitude in [-2,2], default 0.8.
    px, py : int
        Nonboolean derivative orders in [0,2], both default 2.

    Returns
    -------
    float
        Unrounded 1000*d^(px+py)R/(dx^px dy^py) at zero, including the
        ordinary-series factorial conversion. Required absolute and
        relative accuracy is 1e-8. Passing upstream isolated tolerances
        does not establish this end-to-end accuracy.

    Raises
    ------
    ValueError
        If a domain fails, a raw bias has magnitude<=1e-10 or an upstream
        validity/finite-arithmetic condition fails. Floats are invalid
        integer arguments.
    """
    return None

# EXPECTED RETURN LINE
# float, unrounded scaled mixed susceptibility of the signed residual ratio

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_residual_bias(noise_scale: float = 0.06, total_time: float = 1.0, layers: int = 6, order: int = 4, feedforward: float = 0.8, px: int = 2, py: int = 2) -> float:
    vals=[]
    for value,lo,hi in ((noise_scale,.04,.3),(total_time,.6,2)):
        v=np.asarray(value)
        if v.ndim!=0 or v.dtype.kind not in 'iuf' or not np.isfinite(v) or not lo<=float(v)<=hi:
            raise ValueError('invalid scalar')
        vals.append(float(v))
    xi,time=vals
    if isinstance(layers,(bool,np.bool_)) or not isinstance(layers,(int,np.integer)) or not 2<=layers<=12 or layers%2:
        raise ValueError('invalid layer count')
    for degree in (px,py):
        if isinstance(degree,(bool,np.bool_)) or not isinstance(degree,(int,np.integer)) or not 0<=degree<=2:
            raise ValueError('invalid derivative order')
    if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or not 0<=order<=4:
        raise ValueError('invalid mitigation order')
    weights=_oracle_taylor_weights(order)
    h,hf,jumps,rates,instrument,rho,obs=_oracle_qutrit_model(xi,feedforward)
    _,dx=_oracle_lindblad_generators(h,jumps,rates*np.array([1,1,0,0,0,0]))
    _,dy=_oracle_lindblad_generators(h,jumps,rates*np.array([0,0,1,1,1,1]))
    pairs=[];ideal_pairs=[]
    dt=np.full(3,time/(3*int(layers)))
    for hs in (h,h+hf):
        if pairs and np.array_equal(hs,h):
            pairs.append(pairs[0]);ideal_pairs.append(ideal_pairs[0])
            continue
        coherent,dissipative=_oracle_lindblad_generators(hs,jumps,rates)
        pairs.append(_oracle_pulse_channels(coherent,dissipative,dx,dy,dt,px,py))
        zero=np.zeros_like(dissipative)
        ideal_pairs.append(_oracle_pulse_channels(coherent,zero,zero,zero,dt,0,0))
    half=int(layers)//2
    expectations=[]
    for table,m in ((pairs,order),(ideal_pairs,0)):
        pre=np.repeat(table[0][0][None,...],half,axis=0)
        qi=np.repeat(table[0][1][None,...],half,axis=0)
        post=np.array([np.repeat(pair[0][None,...],half,axis=0) for pair in table])
        qp=np.array([np.repeat(pair[1][None,...],half,axis=0) for pair in table])
        expectations.append(_oracle_dynamic_amplification(pre,qi,post,qp,instrument,rho,obs,m))
    result=_oracle_bias_ratio(expectations[0],weights,float(expectations[1][0,0,0]))
    return float(1000*math.factorial(int(px))*math.factorial(int(py))*result[int(px),int(py)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    cases=[
        {'setup':'','call':'residual_bias(.06,1.,6,4,.8)','gold_call':'_oracle_residual_bias(.06,1.,6,4,.8)','tol':1e-8},
        {'setup':'','call':'residual_bias(.08,.7,2,0,0.)','gold_call':'_oracle_residual_bias(.08,.7,2,0,0.)','tol':1e-8},
        {'setup':'','call':'residual_bias(.12,1.3,4,2,-.6)','gold_call':'_oracle_residual_bias(.12,1.3,4,2,-.6)','tol':1e-8},
        {'setup':'','call':'residual_bias(.06,1.,6,4,.8,0,0)','gold_call':'_oracle_residual_bias(.06,1.,6,4,.8,0,0)','tol':1e-8},
        {'setup':'','call':'residual_bias(.06,1.,6,4,.8,2,1)','gold_call':'_oracle_residual_bias(.06,1.,6,4,.8,2,1)','tol':1e-8},
        {'setup':'','call':'residual_bias(.04,.6,12,4,2.)','gold_call':'_oracle_residual_bias(.04,.6,12,4,2.)','tol':1e-8},
    ]
    for args in ('0.,1.,6,4,.8','.06,1.,3,4,.8','.06,1.,6,9,.8','.0001,1.,6,4,.8'): 
        setup=''
        for name,fn in (('run_model','residual_bias'),('run_gold','_oracle_residual_bias')):
            setup+=f'def {name}():\n    try:\n        {fn}({args})\n    except ValueError:\n        return 1\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()','tol':1e-8})
    return cases
