"""
Return rectangular bivariate channel coefficients for a forward control layer and its physical inverse under the selected control-reversal construction.

The supplied forward generator family is coherent[l]+dissipative+x*direction_x+y*direction_y, with chronological pulse index zero earliest and duration durations[l]. The same noise family acts during both experiments. Density vectors use row-major order. The coefficient axes track independent noise perturbations, not amplification settings, and retain the full rectangle including [2,2]. Physical generator families are promised; no Choi test is required.

Returns
-------
tuple    Forward and physical inverse coefficient arrays (px+1,py+1,n,n).    Entry [a,b] multiplies x^a*y^b, not its derivative. Both px and py    default to 2. Required absolute and relative accuracy is 1e-9 in    isolation; composition must meet the final residual tolerance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pulse_channels(coherent: "np.ndarray", dissipative: "np.ndarray", direction_x: "np.ndarray", direction_y: "np.ndarray", durations: "np.ndarray", px: int = 2, py: int = 2) -> tuple:
    """Compose the bivariate response of forward and inverse control layers.

    Parameters
    ----------
    coherent : array_like
        Chronological coherent generators (L,n,n).
    dissipative : array_like
        Base noise generator (n,n).
    direction_x, direction_y : array_like
        Affine noise perturbation generators (n,n).
    durations : array_like
        Real chronological durations (L,) in [0,10].
    px, py : int
        Nonboolean integer coefficient orders in [0,2].

    Returns
    -------
    tuple
        Forward and physical inverse coefficient arrays (px+1,py+1,n,n).
        Entry [a,b] multiplies x^a*y^b, not its derivative. Both px and py
        default to 2. Required absolute and relative accuracy is 1e-9 in
        isolation; composition must meet the final residual tolerance.

    Raises
    ------
    ValueError
        If a declared shape fails, L<1, n is not d*d for an integer d>=2,
        orders are invalid, any array is nonnumeric/nonfinite or has
        magnitude>100, or a duration is not real in [0,10]. Also if a pulse
        duration times the sum of infinity norms of its coherent generator,
        base dissipator and two directions exceeds 100, or output is nonfinite.
    """
    return None


# EXPECTED RETURN LINE
# tuple of complex coefficient arrays for the forward and physical inverse layer

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pulse_channels(coherent: "np.ndarray", dissipative: "np.ndarray", direction_x: "np.ndarray", direction_y: "np.ndarray", durations: "np.ndarray", px: int = 2, py: int = 2) -> tuple:
    for order in (px,py):
        if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or not 0<=order<=2:
            raise ValueError('invalid coefficient order')
    arrays=[]
    for value in (coherent,dissipative,direction_x,direction_y,durations):
        a=np.asarray(value)
        if a.dtype.kind not in 'iufc' or not np.all(np.isfinite(a)) or np.any(np.abs(a)>100):
            raise ValueError('invalid numeric array')
        arrays.append(a)
    c,d,dx,dy,t=arrays
    if c.ndim!=3 or c.shape[0]<1 or c.shape[1]!=c.shape[2]:
        raise ValueError('invalid coherent shape')
    n=c.shape[1];p=int(px);q=int(py)
    if n<4 or int(np.sqrt(n))**2!=n or any(a.shape!=(n,n) for a in (d,dx,dy)) or t.shape!=(len(c),) or t.dtype.kind not in 'iuf' or np.any(t<0) or np.any(t>10):
        raise ValueError('invalid shape or duration')
    norms=[float(dt)*(np.linalg.norm(cp,np.inf)+sum(np.linalg.norm(a,np.inf) for a in (d,dx,dy))) for cp,dt in zip(c,t)]
    if any(value>100 for value in norms):
        raise ValueError('exponent outside numeric domain')
    def multiply(a,b):
        out=np.zeros_like(a)
        for i in range(p+1):
            for j in range(q+1):
                for k in range(i+1):
                    for ell in range(j+1):
                        out[i,j]+=np.einsum('ik,kj->ij',a[k,ell],b[i-k,j-ell],optimize=False)
        return out
    def identity():
        out=np.zeros((p+1,q+1,n,n),complex);out[0,0]=np.eye(n);return out
    def exponential(cp,dt):
        z=identity()*0;z[0,0]=(cp+d)*dt
        if p:z[1,0]=dx*dt
        if q:z[0,1]=dy*dt
        norm=float(dt)*(np.linalg.norm(cp+d,np.inf)+np.linalg.norm(dx,np.inf)+np.linalg.norm(dy,np.inf))
        scaling=max(0,int(np.ceil(np.log2(max(norm,1e-100)/.25))))
        z/=2**scaling
        result=identity();term=result.copy()
        for k in range(1,65):
            next_term=np.zeros_like(term)
            for i in range(p+1):
                for j in range(q+1):
                    next_term[i,j]=np.einsum('ik,kj->ij',term[i,j],z[0,0],optimize=False)
                    if i:next_term[i,j]+=np.einsum('ik,kj->ij',term[i-1,j],z[1,0],optimize=False)
                    if j:next_term[i,j]+=np.einsum('ik,kj->ij',term[i,j-1],z[0,1],optimize=False)
            term=next_term/k;result+=term
            if np.max(np.abs(term))<1e-18:break
        else:raise ValueError('coefficient exponential did not converge')
        for _ in range(scaling):result=multiply(result,result)
        return result
    forward=identity();inverse=identity()
    for cp,dt in zip(c,t):forward=multiply(exponential(cp,dt),forward)
    for cp,dt in zip(c[::-1],t[::-1]):inverse=multiply(exponential(-cp,dt),inverse)
    if not np.all(np.isfinite(forward)) or not np.all(np.isfinite(inverse)):
        raise ValueError('non-finite coefficient')
    return forward,inverse

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base='x=np.array([[0.,1.],[1.,0.]]);z=np.diag([1.,-1.]);i=np.eye(2);c=np.array([-1j*(np.kron(x,i)-np.kron(i,x.T)),-.7j*(np.kron(z,i)-np.kron(i,z.T))]);dx=.13*(np.kron(z,z)-np.eye(4));dy=.09*(np.kron(x,x)-np.eye(4));d=dx+dy;t=np.array([.12,.27])'
    cases=[
        {'setup':base,'call':'pulse_channels(c.copy(),d.copy(),dx.copy(),dy.copy(),t.copy(),2,2)','gold_call':'_oracle_pulse_channels(c.copy(),d.copy(),dx.copy(),dy.copy(),t.copy(),2,2)'},
        {'setup':base+';t[:]=0','call':'pulse_channels(c.copy(),d.copy(),dx.copy(),dy.copy(),t.copy(),2,1)','gold_call':'_oracle_pulse_channels(c.copy(),d.copy(),dx.copy(),dy.copy(),t.copy(),2,1)'},
        {'setup':base+';t=np.array([.001,.9]);d*=3','call':'pulse_channels(c.copy(),d.copy(),dx.copy(),dy.copy(),t.copy(),0,2)','gold_call':'_oracle_pulse_channels(c.copy(),d.copy(),dx.copy(),dy.copy(),t.copy(),0,2)'},
    ]
    for change in ('t[0]=-1','d[0,0]=np.inf','t=np.array([.1])'):
        setup=base+'\n'+change+'\n'
        for name,fn in (('run_model','pulse_channels'),('run_gold','_oracle_pulse_channels')):
            setup+=f'def {name}():\n    try:\n        {fn}(c.copy(),d.copy(),dx.copy(),dy.copy(),t.copy(),2,2)\n    except ValueError:\n        return 1\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    for case in cases:
        case['setup']='import numpy as np\n'+case['setup']
        case['tol']=1e-9
    return cases
