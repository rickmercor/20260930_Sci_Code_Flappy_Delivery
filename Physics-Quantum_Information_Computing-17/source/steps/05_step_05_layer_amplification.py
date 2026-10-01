"""
Construct amplified-layer response coefficients from paired forward and physical-inverse layers for the selected finite-order control-reversal construction.

Each supplied layer is one complete control unit. The output retains chronological layer order and an increasing amplification-setting axis. Ordinary bivariate coefficient axes use row-major density action. Analytic physical channel families near zero are promised; no Choi-positivity test is required. An empty layer sequence retains the declared shape.

Returns
-------
numpy.ndarray    Complex amplified layers (order+1,L,P+1,Q+1,n,n), with ordinary    power-series coefficients. P,Q lie in [0,2], L>=0 and n=d*d for    an integer d>=2. Absolute and relative accuracy is 1e-9 in    isolation; composition must meet the final residual tolerance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def layer_amplification(forward: "np.ndarray", inverse: "np.ndarray", order: int) -> "np.ndarray":
    """Return amplified layer coefficients with one common setting axis.

    Parameters
    ----------
    forward, inverse : numpy.ndarray
        Paired layer series (L,P+1,Q+1,n,n), with row-major density action.
    order : int
        Nonboolean maximum amplification index in [0,8].

    Returns
    -------
    numpy.ndarray
        Complex amplified layers (order+1,L,P+1,Q+1,n,n), with ordinary
        power-series coefficients. P,Q lie in [0,2], L>=0 and n=d*d for
        an integer d>=2. Absolute and relative accuracy is 1e-9 in
        isolation; composition must meet the final residual tolerance.

    Raises
    ------
    ValueError
        If paired shapes or coefficient/matrix axes fail, order is not a
        nonboolean integer in [0,8], any input is nonnumeric/nonfinite or
        has magnitude>100, or an amplified coefficient is nonfinite.
    """
    return None


# EXPECTED RETURN LINE
# numpy.ndarray, amplified layer series with a common leading setting axis

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_layer_amplification(forward: "np.ndarray", inverse: "np.ndarray", order: int) -> "np.ndarray":
    if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or not 0<=order<=8:
        raise ValueError('invalid amplification order')
    arrays=[]
    for value in (forward,inverse):
        a=np.asarray(value)
        if a.dtype.kind not in 'iufc' or not np.all(np.isfinite(a)) or np.any(np.abs(a)>100):
            raise ValueError('invalid channel array')
        arrays.append(a.astype(complex))
    f,v=arrays
    if f.ndim!=5 or v.shape!=f.shape:
        raise ValueError('invalid paired shapes')
    count,p,q,n,m=f.shape
    if not 1<=p<=3 or not 1<=q<=3 or n!=m or n<4 or int(np.sqrt(n))**2!=n:
        raise ValueError('invalid coefficient or matrix axes')
    def multiply(a,b):
        out=np.zeros_like(a)
        for i in range(p):
            for j in range(q):
                for r in range(i+1):
                    for s in range(j+1):out[i,j]+=np.einsum('ik,kj->ij',a[r,s],b[i-r,j-s],optimize=False)
        return out
    result=np.empty((int(order)+1,count,p,q,n,n),complex)
    cache=[]
    for ell in range(count):
        for oldf,oldv,oldresult in cache:
            if np.array_equal(f[ell],oldf) and np.array_equal(v[ell],oldv):
                result[:,ell]=oldresult
                break
        else:
            result[0,ell]=f[ell]
            if order:
                pair=multiply(v[ell],f[ell])
                for j in range(1,int(order)+1):result[j,ell]=multiply(result[j-1,ell],pair)
            cache.append((f[ell],v[ell],result[:,ell]))
    if not np.all(np.isfinite(result)):raise ValueError('non-finite amplification')
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base='import numpy as np\ni=np.eye(2);x=np.array([[0.,1.],[1.,0.]]);z=np.diag([1.,-1.]);u=np.cos(.31)*i-1j*np.sin(.31)*x;v=np.diag(np.exp(-1j*np.array([.24,-.24])));delta=np.outer(np.diag([1.,0.]).reshape(-1),i.reshape(-1))-np.eye(4);f=np.zeros((2,3,3,4,4),complex);q=f.copy();f[0,0,0]=(np.eye(4)+.11*delta)@np.kron(u,u.conj());f[1,0,0]=(np.eye(4)+.17*delta)@np.kron(v,v.conj());q[0,0,0]=(np.eye(4)+.11*delta)@np.kron(u.conj().T,u.T);q[1,0,0]=(np.eye(4)+.17*delta)@np.kron(v.conj().T,v.T);f[0,1,0]=.11*delta@np.kron(u,u.conj());f[1,0,1]=.17*delta@np.kron(v,v.conj());q[0,0,1]=.11*delta@np.kron(u.conj().T,u.T);q[1,1,0]=.17*delta@np.kron(v.conj().T,v.T)'
    cases=[
        {'setup':base,'call':'layer_amplification(f.copy(),q.copy(),4)','gold_call':'_oracle_layer_amplification(f.copy(),q.copy(),4)'},
        {'setup':base+';f=f[:0];q=q[:0]','call':'layer_amplification(f.copy(),q.copy(),0)','gold_call':'_oracle_layer_amplification(f.copy(),q.copy(),0)'},
        {'setup':base+';f=f[:,:1];q=q[:,:1]','call':'layer_amplification(f.copy(),q.copy(),8)','gold_call':'_oracle_layer_amplification(f.copy(),q.copy(),8)'},
    ]
    for value in ('True','9','2.0'):
        setup=base+'\n'
        for name,fn in (('run_model','layer_amplification'),('run_gold','_oracle_layer_amplification')):
            setup+=f'def {name}():\n    try:\n        {fn}(f.copy(),q.copy(),{value})\n    except ValueError:\n        return 1\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    for case in cases:case['tol']=1e-9
    return cases
