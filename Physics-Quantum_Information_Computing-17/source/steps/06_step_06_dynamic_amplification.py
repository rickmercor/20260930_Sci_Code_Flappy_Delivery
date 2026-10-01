"""
Compute exact observable-response coefficients for the dynamic circuit at every setting of the selected layerwise control-reversal construction.

The two coefficient axes describe independent physical noise perturbations. Coefficients use ordinary powers with row-major density vectorization. Temporal sequences are chronological with index zero earliest. The supplied constant instrument separates pre-measurement and outcome-conditioned post-measurement layers. Channels are promised analytic CPTP families near zero; individual nonconstant coefficient matrices need not be CPTP. No Choi test is required.

Returns
-------
numpy.ndarray    Real array (order+1,P+1,Q+1), ordered by increasing amplification    setting. Entry [j,a,b] multiplies x^a*y^b, not its derivative.    P,Q are inferred from pre and lie in [0,2]. Empty temporal    sequences preserve their stated shapes. Absolute and relative    accuracy is 1e-9 in isolation. Composition must meet the final tolerance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def dynamic_amplification(pre: "np.ndarray", pre_inverse: "np.ndarray", post: "np.ndarray", post_inverse: "np.ndarray", instrument: "np.ndarray", rho: "np.ndarray", observable: "np.ndarray", order: int) -> "np.ndarray":
    """Return unextrapolated dynamic-circuit expectation coefficients.

    Parameters
    ----------
    pre, pre_inverse : array_like
        Paired chronological layers (Lpre,P+1,Q+1,d*d,d*d).
    post, post_inverse : array_like
        Paired outcome-dependent layers (B,Lpost,P+1,Q+1,d*d,d*d).
    instrument : array_like
        Complete constant Kraus instrument (B,d,d), B>=1.
    rho : array_like
        Initial density matrix (d,d), d>=2, trace one within 1e-10,
        Hermitian within absolute tolerance 1e-10 and eigenvalues>=-1e-10.
    observable : array_like
        Hermitian observable (d,d), absolute Hermiticity tolerance 1e-10.
    order : int
        Nonboolean integer amplification order in [0,8].

    Returns
    -------
    numpy.ndarray
        Real array (order+1,P+1,Q+1), ordered by increasing amplification
        setting. Entry [j,a,b] multiplies x^a*y^b, not its derivative.
        P,Q are inferred from pre and lie in [0,2]. Empty temporal
        sequences preserve their stated shapes. Absolute and relative
        accuracy is 1e-9 in isolation. Composition must meet the final tolerance.

    Raises
    ------
    ValueError
        If shapes or domains fail, any input is nonnumeric/nonfinite or
        has magnitude>100, the instrument completeness error exceeds
        absolute elementwise tolerance 1e-10, or arithmetic is nonfinite.
        An expectation coefficient with imaginary magnitude>1e-9 is invalid.
    """
    return None

# EXPECTED RETURN LINE
# numpy.ndarray, unextrapolated bivariate expectations for each setting

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_dynamic_amplification(pre: "np.ndarray", pre_inverse: "np.ndarray", post: "np.ndarray", post_inverse: "np.ndarray", instrument: "np.ndarray", rho: "np.ndarray", observable: "np.ndarray", order: int) -> "np.ndarray":
    if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or not 0<=order<=8:
        raise ValueError('invalid order')
    arrays=[]
    for value in (pre,pre_inverse,post,post_inverse,instrument,rho,observable):
        a=np.asarray(value)
        if a.dtype.kind not in 'iufc' or not np.all(np.isfinite(a)) or np.any(np.abs(a)>100):
            raise ValueError('invalid numeric array')
        arrays.append(a.astype(complex))
    pre,pre_inverse,post,post_inverse,instrument,rho,observable=arrays
    if rho.ndim!=2 or rho.shape[0]<2 or rho.shape[0]!=rho.shape[1]:
        raise ValueError('invalid density shape')
    d=len(rho);n=d*d
    if observable.shape!=(d,d) or instrument.ndim!=3 or instrument.shape[0]<1 or instrument.shape[1:]!=(d,d):
        raise ValueError('invalid observable/instrument')
    b=len(instrument)
    if pre.ndim!=5 or pre.shape[-2:]!=(n,n) or pre_inverse.shape!=pre.shape:
        raise ValueError('invalid pre shape')
    p,q=pre.shape[1:3]
    if not 1<=p<=3 or not 1<=q<=3 or post.ndim!=6 or post.shape[0]!=b or post.shape[2:]!=(p,q,n,n) or post_inverse.shape!=post.shape:
        raise ValueError('invalid coefficient/post shape')
    if not np.allclose(rho,rho.conj().T,atol=1e-10,rtol=0) or not np.allclose(observable,observable.conj().T,atol=1e-10,rtol=0):
        raise ValueError('non-Hermitian state/observable')
    if abs(np.trace(rho)-1)>1e-10 or np.linalg.eigvalsh((rho+rho.conj().T)/2).min() < -1e-10:
        raise ValueError('invalid density')
    if not np.allclose(sum(c.conj().T@c for c in instrument),np.eye(d),atol=1e-10,rtol=0):
        raise ValueError('incomplete instrument')
    def multiply(a,z):
        out=np.zeros((p,q,a.shape[-2],z.shape[-1]),complex)
        for i in range(p):
            for j in range(q):
                for k in range(i+1):
                    for ell in range(j+1):
                        out[i,j]+=np.einsum('ik,kj->ij',a[k,ell],z[i-k,j-ell],optimize=False)
        return out
    all_forward=np.concatenate([pre]+[post[branch] for branch in range(b)],axis=0)
    all_inverse=np.concatenate([pre_inverse]+[post_inverse[branch] for branch in range(b)],axis=0)
    amplified=_oracle_layer_amplification(all_forward,all_inverse,order)
    npre=len(pre);npost=post.shape[1]
    out=[]
    for j in range(int(order)+1):
        state=np.zeros((p,q,n,1),complex);state[0,0,:,0]=rho.reshape(-1)
        for k in amplified[j,:npre]:state=multiply(k,state)
        total=np.zeros((p,q),complex)
        for branch,c in enumerate(instrument):
            v=np.zeros_like(state)
            for a in range(p):
                for ell in range(q):v[a,ell,:,0]=(c@state[a,ell,:,0].reshape(d,d)@c.conj().T).reshape(-1)
            for k in amplified[j,npre+branch*npost:npre+(branch+1)*npost]:v=multiply(k,v)
            for a in range(p):
                for ell in range(q):total[a,ell]+=np.trace(observable@v[a,ell,:,0].reshape(d,d))
        if not np.all(np.isfinite(total)) or np.max(np.abs(total.imag))>1e-9:
            raise ValueError('invalid output coefficient')
        out.append(total.real)
    return np.array(out)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base='import numpy as np\ni=np.eye(2);x=np.array([[0.,1.],[1.,0.]]);z=np.diag([1.,-1.]);u=np.cos(.31)*i-1j*np.sin(.31)*x;v=np.diag(np.exp(-1j*np.array([.24,-.24])));reset=np.outer(np.diag([1.,0.]).reshape(-1),i.reshape(-1));delta=reset-np.eye(4);k=np.zeros((2,3,3,4,4),complex);q=k.copy();k[0,0,0]=(np.eye(4)+.11*delta)@np.kron(u,u.conj());k[1,0,0]=(np.eye(4)+.17*delta)@np.kron(v,v.conj());q[0,0,0]=(np.eye(4)+.11*delta)@np.kron(u.conj().T,u.T);q[1,0,0]=(np.eye(4)+.17*delta)@np.kron(v.conj().T,v.T);k[0,1,0]=.11*delta@np.kron(u,u.conj());k[1,0,1]=.17*delta@np.kron(v,v.conj());q[0,0,1]=.11*delta@np.kron(u.conj().T,u.T);q[1,1,0]=.17*delta@np.kron(v.conj().T,v.T);post=np.array([k[::-1],k]);qp=np.array([q[::-1],q]);c=np.array([np.diag([1.,0.]),np.diag([0.,1.])]);rho=np.array([[.7,.12+.08j],[.12-.08j,.3]]);o=.4*x+.7*z'
    call='dynamic_amplification(k.copy(),q.copy(),post.copy(),qp.copy(),c.copy(),rho.copy(),o.copy(),4)'
    cases=[
        {'setup':base,'call':call,'gold_call':'_oracle_'+call},
        {'setup':base+';k=np.empty((0,3,3,4,4));q=k.copy();post=np.empty((2,0,3,3,4,4));qp=post.copy()','call':call,'gold_call':'_oracle_'+call},
        {'setup':base+';c=np.array([np.diag([1.,np.sqrt(.6)]),[[0.,np.sqrt(.4)],[0.,0.]]])','call':call,'gold_call':'_oracle_'+call},
    ]
    for change in ('rho[0,0]=2','c*=.8','o[0,0]=np.nan'):
        setup=base+'\n'+change+'\n'
        for name,fn in (('run_model','dynamic_amplification'),('run_gold','_oracle_dynamic_amplification')):
            setup+=f'def {name}():\n    try:\n        {fn}(k.copy(),q.copy(),post.copy(),qp.copy(),c.copy(),rho.copy(),o.copy(),4)\n    except ValueError:\n        return 1\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    for case in cases:case['tol']=1e-9
    return cases
