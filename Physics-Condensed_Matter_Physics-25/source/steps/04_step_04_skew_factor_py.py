"""
Factor a reordered skew matrix by stable two-by-two congruence pivots.

Main paper Section III.B: P S P^T = scale * L T L^T, with unit lower-triangular L and skew 2x2 blocks T. Numerical pivot exchanges may change the supplied initial ordering. Equivalent stable factors are accepted through their defining identity and Pfaffian invariant. Use a positive scale to retain the sign and logarithmic magnitude for uniformly very large or small matrices; scale=0 is reserved for S=0.

Returns
-------
return result  # real ndarray (n+2,n). Rows 0:n hold L; row n is final original-vertex permutation p; row n+1 entries 0:n/2 hold upper-right T-block pivots, entry n/2 holds nonnegative scale, and all remaining entries are zero. The identity is S[p,p]=scale*L*T*L.T. Singular inputs have one or more zero pivots.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def skew_factor(skew: ArrayLike, ordering: ArrayLike) -> np.ndarray:
    'Factor a reordered skew matrix by stable two-by-two congruence pivots.\n\nParameters\n----------\nskew : real finite (n,n), skew-symmetric, positive even n.\nordering : integer permutation (n,), initial matched row/column order.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. Skew symmetry uses relative max-entry tolerance 1e-12. The ordering contains each index 0,...,n-1 exactly once.\n\nReturns\n-------\nreal ndarray (n+2,n). Rows 0:n hold L; row n is final original-vertex permutation p; row n+1 entries 0:n/2 hold upper-right T-block pivots, entry n/2 holds nonnegative scale, and all remaining entries are zero. The identity is S[p,p]=scale*L*T*L.T. Singular inputs have one or more zero pivots.'
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def _oracle_skew_factor(skew: ArrayLike, ordering: ArrayLike) -> np.ndarray:
    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_permutation(ordering, n, packed=False):
        a = _checked_numeric(ordering, 'permutation')
        if a.shape != (n,) or (not packed and np.asarray(ordering).dtype.kind not in 'iu'):
            raise ValueError('permutation must have the declared shape and integer type')
        if np.any(a != np.floor(a)) or np.any(a < 0) or np.any(a >= n):
            raise ValueError('permutation entries must be integer indices from 0 through n-1')
        p = a.astype(int)
        if not np.array_equal(np.sort(p), np.arange(n)):
            raise ValueError('permutation must contain each index exactly once')
        return p

    def _checked_skew(skew):
        a = _checked_numeric(skew, 'skew')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0 or len(a) % 2:
            raise ValueError('skew must be square with positive even size')
        scale = float(np.max(abs(a)))
        if scale and np.max(abs(a / scale + a.T / scale)) > 1e-12:
            raise ValueError('matrix must be skew-symmetric to relative max-entry tolerance 1e-12')
        return a

    s=_checked_skew(skew);n=len(s);p=_checked_permutation(ordering,n).copy();scale=float(np.max(abs(s)))
    l=np.eye(n);d=np.zeros(n//2)
    if scale==0:
        out=np.zeros((n+2,n));out[:n]=l;out[n]=p;return out
    a=s[np.ix_(p,p)]/scale
    def _swap(i,j,k):
        if i==j:return
        a[[i,j],:]=a[[j,i],:];a[:,[i,j]]=a[:,[j,i]]
        l[[i,j],:k]=l[[j,i],:k];p[i],p[j]=p[j],p[i]
    for k in range(0,n,2):
        tail=abs(np.triu(a[k:,k:],1));ind=np.unravel_index(np.argmax(tail),tail.shape)
        peak=tail[ind]
        if peak==0:break
        if abs(a[k,k+1])<.1*peak:
            i,j=k+ind[0],k+ind[1];_swap(k,i,k)
            if j==k:j=i
            _swap(k+1,j,k)
        pivot=a[k,k+1];d[k//2]=pivot
        if k+2<n:
            cross=a[k+2:,k:k+2].copy()
            multipliers=np.column_stack((cross[:,1]/pivot,-cross[:,0]/pivot))
            l[k+2:,k:k+2]=multipliers
            t=np.array([[0,pivot],[-pivot,0.]])
            b=a[k+2:,k+2:]-multipliers@t@multipliers.T
            a[k+2:,k+2:]=(b-b.T)/2
    out=np.zeros((n+2,n));out[:n]=l;out[n]=p;out[n+1,:n//2]=d;out[n+1,n//2]=scale
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ndef _factor_metrics(s,f):\n    f=np.asarray(f);n=len(s)\n    if f.shape!=(n+2,n) or not np.isfinite(f).all():return np.ones(5)*999\n    p=f[n]\n    if not np.all(p==np.rint(p)) or not np.array_equal(np.sort(p),np.arange(n)):return np.ones(5)*999\n    p=p.astype(int);l=f[:n];d=f[n+1,:n//2];scale=f[n+1,n//2]\n    t=np.zeros((n,n))\n    for i,v in enumerate(d):t[2*i,2*i+1]=v;t[2*i+1,2*i]=-v\n    norm=max(np.max(abs(s)),1e-300)\n    err=np.max(abs(s[np.ix_(p,p)]/norm-(scale/norm)*(l@t@l.T)))\n    tri=np.max(abs(np.triu(l,1)))+np.max(abs(np.diag(l)-1))\n    parity=(-1)**sum(np.count_nonzero(p[i+1:]<p[i]) for i in range(n))\n    sign=parity*np.prod(np.sign(d)) if scale>0 else 0\n    logabs=np.sum(np.log(abs(d)))+(n//2)*np.log(scale) if sign else 0.\n    reserved=np.max(abs(f[n+1,n//2+1:])) if n>2 else 0.\n    return np.array([err,tri,sign,logabs,reserved])\n\ns=np.array([[0.0, 2.0], [-2.0, 0.0]],dtype=float)\np=np.array([0, 1],dtype=int)', 'call': '_factor_metrics(s,skew_factor(s,p))', 'gold_call': '_factor_metrics(s,_oracle_skew_factor(s,p))'}, {'setup': 'import numpy as np\ndef _factor_metrics(s,f):\n    f=np.asarray(f);n=len(s)\n    if f.shape!=(n+2,n) or not np.isfinite(f).all():return np.ones(5)*999\n    p=f[n]\n    if not np.all(p==np.rint(p)) or not np.array_equal(np.sort(p),np.arange(n)):return np.ones(5)*999\n    p=p.astype(int);l=f[:n];d=f[n+1,:n//2];scale=f[n+1,n//2]\n    t=np.zeros((n,n))\n    for i,v in enumerate(d):t[2*i,2*i+1]=v;t[2*i+1,2*i]=-v\n    norm=max(np.max(abs(s)),1e-300)\n    err=np.max(abs(s[np.ix_(p,p)]/norm-(scale/norm)*(l@t@l.T)))\n    tri=np.max(abs(np.triu(l,1)))+np.max(abs(np.diag(l)-1))\n    parity=(-1)**sum(np.count_nonzero(p[i+1:]<p[i]) for i in range(n))\n    sign=parity*np.prod(np.sign(d)) if scale>0 else 0\n    logabs=np.sum(np.log(abs(d)))+(n//2)*np.log(scale) if sign else 0.\n    reserved=np.max(abs(f[n+1,n//2+1:])) if n>2 else 0.\n    return np.array([err,tri,sign,logabs,reserved])\n\ns=np.array([[0.0, -2.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0], [0.0, 0.0, -3.0, 0.0]],dtype=float)\np=np.array([1, 0, 2, 3],dtype=int)', 'call': '_factor_metrics(s,skew_factor(s,p))', 'gold_call': '_factor_metrics(s,_oracle_skew_factor(s,p))'}, {'setup': 'import numpy as np\ndef _factor_metrics(s,f):\n    f=np.asarray(f);n=len(s)\n    if f.shape!=(n+2,n) or not np.isfinite(f).all():return np.ones(5)*999\n    p=f[n]\n    if not np.all(p==np.rint(p)) or not np.array_equal(np.sort(p),np.arange(n)):return np.ones(5)*999\n    p=p.astype(int);l=f[:n];d=f[n+1,:n//2];scale=f[n+1,n//2]\n    t=np.zeros((n,n))\n    for i,v in enumerate(d):t[2*i,2*i+1]=v;t[2*i+1,2*i]=-v\n    norm=max(np.max(abs(s)),1e-300)\n    err=np.max(abs(s[np.ix_(p,p)]/norm-(scale/norm)*(l@t@l.T)))\n    tri=np.max(abs(np.triu(l,1)))+np.max(abs(np.diag(l)-1))\n    parity=(-1)**sum(np.count_nonzero(p[i+1:]<p[i]) for i in range(n))\n    sign=parity*np.prod(np.sign(d)) if scale>0 else 0\n    logabs=np.sum(np.log(abs(d)))+(n//2)*np.log(scale) if sign else 0.\n    reserved=np.max(abs(f[n+1,n//2+1:])) if n>2 else 0.\n    return np.array([err,tri,sign,logabs,reserved])\n\ns=np.array([[0.0, -0.7841626433654723, -1.144467078730327, 0.08067075348268377, 0.29692240796515135, 0.050519663033510215, 0.09573282514121262, -1.0221758573458943], [0.7841626433654723, 0.0, -2.71281241345953, 2.44571039879547, 1.1920110838964397, 1.5839774605892651, -1.7136512214347865, -1.3548219539938726], [1.144467078730327, 2.71281241345953, 0.0, -1.2603740057724215, 0.8385393459313748, -1.2015489658236167, 0.900061720763162, -3.3497681542517164], [-0.08067075348268377, -2.44571039879547, 1.2603740057724215, 0.0, 0.4270326199188317, 0.4178819906220006, 0.6805960600624872, 1.5042834978983475], [-0.29692240796515135, -1.1920110838964397, -0.8385393459313748, -0.4270326199188317, 0.0, -0.09367853351054295, -3.192895261719509, 0.38330279460098104], [-0.050519663033510215, -1.5839774605892651, 1.2015489658236167, -0.4178819906220006, 0.09367853351054295, 0.0, -0.594591535134559, -0.9187687530553712], [-0.09573282514121262, 1.7136512214347865, -0.900061720763162, -0.6805960600624872, 3.192895261719509, 0.594591535134559, 0.0, 0.10267068282868197], [1.0221758573458943, 1.3548219539938726, 3.3497681542517164, -1.5042834978983475, -0.38330279460098104, 0.9187687530553712, -0.10267068282868197, 0.0]],dtype=float)\np=np.array([0, 1, 2, 3, 4, 5, 6, 7],dtype=int)', 'call': '_factor_metrics(s,skew_factor(s,p))', 'gold_call': '_factor_metrics(s,_oracle_skew_factor(s,p))'}, {'setup': 'import numpy as np\ndef _factor_metrics(s,f):\n    f=np.asarray(f);n=len(s)\n    if f.shape!=(n+2,n) or not np.isfinite(f).all():return np.ones(5)*999\n    p=f[n]\n    if not np.all(p==np.rint(p)) or not np.array_equal(np.sort(p),np.arange(n)):return np.ones(5)*999\n    p=p.astype(int);l=f[:n];d=f[n+1,:n//2];scale=f[n+1,n//2]\n    t=np.zeros((n,n))\n    for i,v in enumerate(d):t[2*i,2*i+1]=v;t[2*i+1,2*i]=-v\n    norm=max(np.max(abs(s)),1e-300)\n    err=np.max(abs(s[np.ix_(p,p)]/norm-(scale/norm)*(l@t@l.T)))\n    tri=np.max(abs(np.triu(l,1)))+np.max(abs(np.diag(l)-1))\n    parity=(-1)**sum(np.count_nonzero(p[i+1:]<p[i]) for i in range(n))\n    sign=parity*np.prod(np.sign(d)) if scale>0 else 0\n    logabs=np.sum(np.log(abs(d)))+(n//2)*np.log(scale) if sign else 0.\n    reserved=np.max(abs(f[n+1,n//2+1:])) if n>2 else 0.\n    return np.array([err,tri,sign,logabs,reserved])\n\ns=np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, -0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -3.0], [0.0, 0.0, 0.0, 0.0, 3.0, 0.0]],dtype=float)\np=np.array([0, 2, 1, 3, 4, 5],dtype=int)', 'call': '_factor_metrics(s,skew_factor(s,p))', 'gold_call': '_factor_metrics(s,_oracle_skew_factor(s,p))'}, {'setup': 'import numpy as np\ndef _factor_metrics(s,f):\n    f=np.asarray(f);n=len(s)\n    if f.shape!=(n+2,n) or not np.isfinite(f).all():return np.ones(5)*999\n    p=f[n]\n    if not np.all(p==np.rint(p)) or not np.array_equal(np.sort(p),np.arange(n)):return np.ones(5)*999\n    p=p.astype(int);l=f[:n];d=f[n+1,:n//2];scale=f[n+1,n//2]\n    t=np.zeros((n,n))\n    for i,v in enumerate(d):t[2*i,2*i+1]=v;t[2*i+1,2*i]=-v\n    norm=max(np.max(abs(s)),1e-300)\n    err=np.max(abs(s[np.ix_(p,p)]/norm-(scale/norm)*(l@t@l.T)))\n    tri=np.max(abs(np.triu(l,1)))+np.max(abs(np.diag(l)-1))\n    parity=(-1)**sum(np.count_nonzero(p[i+1:]<p[i]) for i in range(n))\n    sign=parity*np.prod(np.sign(d)) if scale>0 else 0\n    logabs=np.sum(np.log(abs(d)))+(n//2)*np.log(scale) if sign else 0.\n    reserved=np.max(abs(f[n+1,n//2+1:])) if n>2 else 0.\n    return np.array([err,tri,sign,logabs,reserved])\n\ns=np.array([[0.0, -7.841626433654723e+159, -1.144467078730327e+160, 8.067075348268377e+158, 2.9692240796515134e+159, 5.051966303351022e+158, 9.573282514121261e+158, -1.0221758573458944e+160], [7.841626433654723e+159, 0.0, -2.7128124134595302e+160, 2.44571039879547e+160, 1.1920110838964398e+160, 1.5839774605892652e+160, -1.7136512214347865e+160, -1.3548219539938727e+160], [1.144467078730327e+160, 2.7128124134595302e+160, 0.0, -1.2603740057724215e+160, 8.385393459313749e+159, -1.2015489658236168e+160, 9.00061720763162e+159, -3.3497681542517167e+160], [-8.067075348268377e+158, -2.44571039879547e+160, 1.2603740057724215e+160, 0.0, 4.270326199188317e+159, 4.178819906220006e+159, 6.805960600624872e+159, 1.5042834978983474e+160], [-2.9692240796515134e+159, -1.1920110838964398e+160, -8.385393459313749e+159, -4.270326199188317e+159, 0.0, -9.367853351054295e+158, -3.1928952617195094e+160, 3.8330279460098105e+159], [-5.051966303351022e+158, -1.5839774605892652e+160, 1.2015489658236168e+160, -4.178819906220006e+159, 9.367853351054295e+158, 0.0, -5.94591535134559e+159, -9.187687530553712e+159], [-9.573282514121261e+158, 1.7136512214347865e+160, -9.00061720763162e+159, -6.805960600624872e+159, 3.1928952617195094e+160, 5.94591535134559e+159, 0.0, 1.0267068282868196e+159], [1.0221758573458944e+160, 1.3548219539938727e+160, 3.3497681542517167e+160, -1.5042834978983474e+160, -3.8330279460098105e+159, 9.187687530553712e+159, -1.0267068282868196e+159, 0.0]],dtype=float)\np=np.array([7, 6, 5, 4, 3, 2, 1, 0],dtype=int)', 'call': '_factor_metrics(s,skew_factor(s,p))', 'gold_call': '_factor_metrics(s,_oracle_skew_factor(s,p))'}, {'setup': 'import numpy as np\ndef _factor_metrics(s,f):\n    f=np.asarray(f);n=len(s)\n    if f.shape!=(n+2,n) or not np.isfinite(f).all():return np.ones(5)*999\n    p=f[n]\n    if not np.all(p==np.rint(p)) or not np.array_equal(np.sort(p),np.arange(n)):return np.ones(5)*999\n    p=p.astype(int);l=f[:n];d=f[n+1,:n//2];scale=f[n+1,n//2]\n    t=np.zeros((n,n))\n    for i,v in enumerate(d):t[2*i,2*i+1]=v;t[2*i+1,2*i]=-v\n    norm=max(np.max(abs(s)),1e-300)\n    err=np.max(abs(s[np.ix_(p,p)]/norm-(scale/norm)*(l@t@l.T)))\n    tri=np.max(abs(np.triu(l,1)))+np.max(abs(np.diag(l)-1))\n    parity=(-1)**sum(np.count_nonzero(p[i+1:]<p[i]) for i in range(n))\n    sign=parity*np.prod(np.sign(d)) if scale>0 else 0\n    logabs=np.sum(np.log(abs(d)))+(n//2)*np.log(scale) if sign else 0.\n    reserved=np.max(abs(f[n+1,n//2+1:])) if n>2 else 0.\n    return np.array([err,tri,sign,logabs,reserved])\n\ns=np.array([[0.0, -7.841626433654722e-161, -1.144467078730327e-160, 8.067075348268377e-162, 2.9692240796515136e-161, 5.051966303351022e-162, 9.57328251412126e-162, -1.0221758573458944e-160], [7.841626433654722e-161, 0.0, -2.7128124134595302e-160, 2.44571039879547e-160, 1.1920110838964397e-160, 1.583977460589265e-160, -1.7136512214347866e-160, -1.3548219539938726e-160], [1.144467078730327e-160, 2.7128124134595302e-160, 0.0, -1.2603740057724215e-160, 8.385393459313748e-161, -1.2015489658236167e-160, 9.00061720763162e-161, -3.349768154251716e-160], [-8.067075348268377e-162, -2.44571039879547e-160, 1.2603740057724215e-160, 0.0, 4.2703261991883165e-161, 4.178819906220006e-161, 6.805960600624872e-161, 1.5042834978983475e-160], [-2.9692240796515136e-161, -1.1920110838964397e-160, -8.385393459313748e-161, -4.2703261991883165e-161, 0.0, -9.367853351054294e-162, -3.192895261719509e-160, 3.833027946009811e-161], [-5.051966303351022e-162, -1.583977460589265e-160, 1.2015489658236167e-160, -4.178819906220006e-161, 9.367853351054294e-162, 0.0, -5.94591535134559e-161, -9.187687530553712e-161], [-9.57328251412126e-162, 1.7136512214347866e-160, -9.00061720763162e-161, -6.805960600624872e-161, 3.192895261719509e-160, 5.94591535134559e-161, 0.0, 1.0267068282868198e-161], [1.0221758573458944e-160, 1.3548219539938726e-160, 3.349768154251716e-160, -1.5042834978983475e-160, -3.833027946009811e-161, 9.187687530553712e-161, -1.0267068282868198e-161, 0.0]],dtype=float)\np=np.array([2, 1, 0, 3, 4, 5, 6, 7],dtype=int)', 'call': '_factor_metrics(s,skew_factor(s,p))', 'gold_call': '_factor_metrics(s,_oracle_skew_factor(s,p))'}, {'setup': 'import numpy as np\ndef _factor_metrics(s,f):\n    f=np.asarray(f);n=len(s)\n    if f.shape!=(n+2,n) or not np.isfinite(f).all():return np.ones(5)*999\n    p=f[n]\n    if not np.all(p==np.rint(p)) or not np.array_equal(np.sort(p),np.arange(n)):return np.ones(5)*999\n    p=p.astype(int);l=f[:n];d=f[n+1,:n//2];scale=f[n+1,n//2]\n    t=np.zeros((n,n))\n    for i,v in enumerate(d):t[2*i,2*i+1]=v;t[2*i+1,2*i]=-v\n    norm=max(np.max(abs(s)),1e-300)\n    err=np.max(abs(s[np.ix_(p,p)]/norm-(scale/norm)*(l@t@l.T)))\n    tri=np.max(abs(np.triu(l,1)))+np.max(abs(np.diag(l)-1))\n    parity=(-1)**sum(np.count_nonzero(p[i+1:]<p[i]) for i in range(n))\n    sign=parity*np.prod(np.sign(d)) if scale>0 else 0\n    logabs=np.sum(np.log(abs(d)))+(n//2)*np.log(scale) if sign else 0.\n    reserved=np.max(abs(f[n+1,n//2+1:])) if n>2 else 0.\n    return np.array([err,tri,sign,logabs,reserved])\n\ns=np.array([[0.0, 1e-15, 2.0, 3.0], [-1e-15, 0.0, 4.0, 5.0], [-2.0, -4.0, 0.0, 6.0], [-3.0, -5.0, -6.0, 0.0]],dtype=float)\np=np.array([0, 1, 2, 3],dtype=int)', 'call': '_factor_metrics(s,skew_factor(s,p))', 'gold_call': '_factor_metrics(s,_oracle_skew_factor(s,p))'}, {'setup': 'import numpy as np\ndef _factor_metrics(s,f):\n    f=np.asarray(f);n=len(s)\n    if f.shape!=(n+2,n) or not np.isfinite(f).all():return np.ones(5)*999\n    p=f[n]\n    if not np.all(p==np.rint(p)) or not np.array_equal(np.sort(p),np.arange(n)):return np.ones(5)*999\n    p=p.astype(int);l=f[:n];d=f[n+1,:n//2];scale=f[n+1,n//2]\n    t=np.zeros((n,n))\n    for i,v in enumerate(d):t[2*i,2*i+1]=v;t[2*i+1,2*i]=-v\n    norm=max(np.max(abs(s)),1e-300)\n    err=np.max(abs(s[np.ix_(p,p)]/norm-(scale/norm)*(l@t@l.T)))\n    tri=np.max(abs(np.triu(l,1)))+np.max(abs(np.diag(l)-1))\n    parity=(-1)**sum(np.count_nonzero(p[i+1:]<p[i]) for i in range(n))\n    sign=parity*np.prod(np.sign(d)) if scale>0 else 0\n    logabs=np.sum(np.log(abs(d)))+(n//2)*np.log(scale) if sign else 0.\n    reserved=np.max(abs(f[n+1,n//2+1:])) if n>2 else 0.\n    return np.array([err,tri,sign,logabs,reserved])\n\ns=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],dtype=float)\np=np.array([2, 0, 1, 5, 4, 3],dtype=int)', 'call': '_factor_metrics(s,skew_factor(s,p))', 'gold_call': '_factor_metrics(s,_oracle_skew_factor(s,p))'}]
