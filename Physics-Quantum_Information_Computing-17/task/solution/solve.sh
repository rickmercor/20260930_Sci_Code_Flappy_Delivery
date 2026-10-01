#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def qutrit_model(noise_scale: float = 0.06, feedforward: float = 0.8) -> tuple:
    vals = []
    for value, low, high in ((noise_scale, 0, 1), (feedforward, -2, 2)):
        v = np.asarray(value)
        if v.ndim != 0 or v.dtype.kind not in 'iuf' or not np.isfinite(v) or not low <= float(v) <= high:
            raise ValueError('invalid scalar')
        vals.append(float(v))
    xi, f = vals
    eye = np.eye(3, dtype=complex)
    x = np.zeros((3, 3), complex)
    x[0, 1] = x[1, 0] = 1
    y = np.zeros((3, 3), complex)
    y[0, 1], y[1, 0] = -1j, 1j
    z = np.diag([1, -1, 0]).astype(complex)
    n = np.diag([0, 1, 2]).astype(complex)
    a = np.diag([1, np.sqrt(2)], 1).astype(complex)
    ell = np.zeros((3, 3), complex)
    ell[2, 1] = 1
    h = np.array([1.3*np.kron(x, x), 0.9*np.kron(z, eye)+0.7*np.kron(eye, y), 1.1*np.kron(y, z)+0.4*np.kron(n, eye)])
    hf = f*np.kron(x, eye)
    jumps = np.array([np.kron(a, eye), np.kron(eye, a), np.kron(z, eye), np.kron(eye, z), np.kron(ell, eye), np.kron(eye, ell)])
    rates = xi*np.array([1, 1, 0.7, 0.7, 0.5, 0.5])
    projectors = np.array([np.kron(np.diag([1, 0, 0]), eye), np.kron(np.diag([0, 1, 1]), eye)])
    rho = np.zeros((9, 9), complex)
    rho[0, 0] = 1
    return h, hf, jumps, rates, projectors, rho, rho.copy()

import numpy as np


def lindblad_generators(hamiltonians: "np.ndarray", jumps: "np.ndarray", rates: "np.ndarray") -> tuple:
    arrays = []
    for value in (hamiltonians, jumps, rates):
        a = np.asarray(value)
        if a.dtype.kind not in 'iufc' or not np.all(np.isfinite(a)) or np.any(np.abs(a)>100):
            raise ValueError('invalid numeric array')
        arrays.append(a)
    h, c, g = arrays
    if h.ndim != 3 or h.shape[0]<1 or h.shape[1]<2 or h.shape[1]!=h.shape[2]:
        raise ValueError('invalid Hamiltonian shape')
    d = h.shape[1]
    if c.ndim!=3 or c.shape[1:]!=(d,d) or g.shape!=(len(c),) or g.dtype.kind not in 'iuf' or np.any(g<0):
        raise ValueError('invalid jump/rate shape')
    if not np.allclose(h,h.conj().transpose(0,2,1),atol=1e-10,rtol=0):
        raise ValueError('Hamiltonian is not Hermitian')
    eye = np.eye(d)
    coherent = np.array([-1j*(np.kron(x,eye)-np.kron(eye,x.T)) for x in h])
    dissipative = np.zeros((d*d,d*d),complex)
    for jump, rate in zip(c,g):
        gram = jump.conj().T@jump
        dissipative += rate*(np.kron(jump,jump.conj())-.5*np.kron(gram,eye)-.5*np.kron(eye,gram.T))
    if not np.all(np.isfinite(coherent)) or not np.all(np.isfinite(dissipative)):
        raise ValueError('non-finite generator')
    return coherent, dissipative

import numpy as np


def pulse_channels(coherent: "np.ndarray", dissipative: "np.ndarray", direction_x: "np.ndarray", direction_y: "np.ndarray", durations: "np.ndarray", px: int = 2, py: int = 2) -> tuple:
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

import math
import numpy as np


def taylor_weights(order: int) -> "np.ndarray":
    if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or not 0<=order<=8:
        raise ValueError('invalid order')
    m=int(order)
    numerator=math.prod(range(1,2*m+2,2))
    return np.array([(-1)**j*numerator/(2**m*(2*j+1)*math.factorial(j)*math.factorial(m-j)) for j in range(m+1)])

import numpy as np


def layer_amplification(forward: "np.ndarray", inverse: "np.ndarray", order: int) -> "np.ndarray":
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

import numpy as np


def dynamic_amplification(pre: "np.ndarray", pre_inverse: "np.ndarray", post: "np.ndarray", post_inverse: "np.ndarray", instrument: "np.ndarray", rho: "np.ndarray", observable: "np.ndarray", order: int) -> "np.ndarray":
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
    amplified=layer_amplification(all_forward,all_inverse,order)
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

import numpy as np


def bias_ratio(expectations: "np.ndarray", weights: "np.ndarray", ideal: float) -> "np.ndarray":
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

import math
import numpy as np


def residual_bias(noise_scale: float = 0.06, total_time: float = 1.0, layers: int = 6, order: int = 4, feedforward: float = 0.8, px: int = 2, py: int = 2) -> float:
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
    weights=taylor_weights(order)
    h,hf,jumps,rates,instrument,rho,obs=qutrit_model(xi,feedforward)
    _,dx=lindblad_generators(h,jumps,rates*np.array([1,1,0,0,0,0]))
    _,dy=lindblad_generators(h,jumps,rates*np.array([0,0,1,1,1,1]))
    pairs=[];ideal_pairs=[]
    dt=np.full(3,time/(3*int(layers)))
    for hs in (h,h+hf):
        if pairs and np.array_equal(hs,h):
            pairs.append(pairs[0]);ideal_pairs.append(ideal_pairs[0])
            continue
        coherent,dissipative=lindblad_generators(hs,jumps,rates)
        pairs.append(pulse_channels(coherent,dissipative,dx,dy,dt,px,py))
        zero=np.zeros_like(dissipative)
        ideal_pairs.append(pulse_channels(coherent,zero,zero,zero,dt,0,0))
    half=int(layers)//2
    expectations=[]
    for table,m in ((pairs,order),(ideal_pairs,0)):
        pre=np.repeat(table[0][0][None,...],half,axis=0)
        qi=np.repeat(table[0][1][None,...],half,axis=0)
        post=np.array([np.repeat(pair[0][None,...],half,axis=0) for pair in table])
        qp=np.array([np.repeat(pair[1][None,...],half,axis=0) for pair in table])
        expectations.append(dynamic_amplification(pre,qi,post,qp,instrument,rho,obs,m))
    result=bias_ratio(expectations[0],weights,float(expectations[1][0,0,0]))
    return float(1000*math.factorial(int(px))*math.factorial(int(py))*result[int(px),int(py)])
SCICODE_GOLD_EOF
