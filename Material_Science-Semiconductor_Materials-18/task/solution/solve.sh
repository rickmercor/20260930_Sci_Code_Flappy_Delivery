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

def bond_ladder_model(potentials: 'np.ndarray | list | tuple', length: int, width: int, bonds: 'np.ndarray | list | tuple', hopping: float, interaction: float) -> 'np.ndarray':
    e=np.asarray(potentials,dtype=float); links=np.asarray(bonds)
    if not isinstance(length,(int,np.integer)) or not isinstance(width,(int,np.integer)) or length<1 or width<1 or length*width>18:
        raise ValueError('positive integer dimensions with at most 18 sites required')
    P=length*width
    if P%2 or links.shape!=(P//2,2) or e.shape!=(P//2,) or not np.isfinite(e).all() or not np.isfinite(links).all() or np.any(links!=np.floor(links)):
        raise ValueError('one finite potential per matched bond required')
    links=links.astype(int)
    if sorted(links.ravel().tolist())!=list(range(P)):
        raise ValueError('bonds must partition the sites')
    for i,j in links:
        if abs(i//width-j//width)+abs(i%width-j%width)!=1:
            raise ValueError('selected bonds must be nearest neighbors')
    if not np.isfinite(hopping) or hopping<=0 or not np.isfinite(interaction):
        raise ValueError('positive hopping and finite interaction required')
    model=np.zeros((2,P,P))
    for i in range(P):
        for j in range(i+1,P):
            if abs(i//width-j//width)+abs(i%width-j%width)==1:
                model[0,i,j]=model[0,j,i]=-hopping
    for energy,(i,j) in zip(e,links):
        model[0,i,i]=model[0,j,j]=energy
        model[1,i,j]=model[1,j,i]=interaction
    derivative=np.zeros_like(model)
    for i,j in links:derivative[1,i,j]=derivative[1,j,i]=1.
    return np.stack((model,derivative))

from functools import lru_cache
from itertools import combinations
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh, LinearOperator, cg
import numpy as np

@lru_cache(maxsize=64)
def _sector_maps(length, number):
    states=sorted(sum(1<<i for i in c) for c in combinations(range(length),number))
    index={s:i for i,s in enumerate(states)}
    occ=np.array([[(s>>i)&1 for i in range(length)] for s in states],dtype=float)
    lower=sorted(sum(1<<i for i in c) for c in combinations(range(length),number-1))
    lookup={s:i for i,s in enumerate(lower)}
    maps=[]
    for i in range(length):
        src=[];dst=[];sign=[]
        for j,s in enumerate(states):
            if (s>>i)&1:
                src.append(j);dst.append(lookup[s^(1<<i)]);sign.append(1-2*(bin(s&((1<<i)-1)).count('1')%2))
        maps.append((np.array(src,dtype=int),np.array(dst,dtype=int),np.array(sign)))
    hops=[]
    for i in range(length):
        for j in range(i+1,length):
            src=[];dst=[];sign=[]
            for k,s in enumerate(states):
                if not (s>>i)&1 and (s>>j)&1:
                    src.append(k);dst.append(index[s^(1<<i)^(1<<j)])
                    sign.append(1-2*(bin((s>> (i+1))&((1<<(j-i-1))-1)).count('1')%2))
            hops.append((i,j,np.array(src,dtype=int),np.array(dst,dtype=int),np.array(sign)))
    return occ,maps,hops,len(lower)

def _response_hamiltonian(model,occ,hops):
    h,U=model
    diagonal=occ@np.diag(h).real+np.einsum('bi,ij,bj->b',occ,np.triu(U.real,1),occ)
    rows=[np.arange(len(occ))];cols=[np.arange(len(occ))];vals=[diagonal]
    for i,j,src,dst,sign in hops:
        if h[i,j]!=0:
            rows.extend((dst,src));cols.extend((src,dst));vals.extend((h[i,j]*sign,h[j,i]*sign))
    return coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(occ),len(occ))).tocsr()

def sector_density_pair(model: 'np.ndarray | list | tuple', number: int) -> 'np.ndarray':
    jet=np.asarray(model)
    if jet.ndim!=4 or jet.shape[:2]!=(2,2) or jet.shape[2]!=jet.shape[3] or not 1<=jet.shape[2]<=18 or not np.isfinite(jet).all():
        raise ValueError('finite (2,2,P,P) Hamiltonian jet required')
    jet=jet.astype(complex,copy=True)
    for order in range(2):
        h,U=jet[order]
        if not np.allclose(h,h.conj().T,rtol=0,atol=1e-12) or np.max(abs(U.imag))>1e-12 or not np.allclose(U,U.T,rtol=0,atol=1e-12) or np.max(abs(np.diag(U)))>1e-12:
            raise ValueError('Hermitian h and real symmetric zero-diagonal U required')
        jet[order,0]=(h+h.conj().T)/2;jet[order,1]=(U.real+U.real.T)/2
    if not isinstance(number,(int,np.integer)) or not 1<=number<=jet.shape[-1]:
        raise ValueError('integer particle number required')
    P=jet.shape[-1];out=np.zeros((2,2,P,P),complex)
    for layer,n in enumerate((number,number-1)):
        if n==0:continue
        occ,maps,hops,smaller=_sector_maps(P,n)
        H=_response_hamiltonian(jet[0],occ,hops);W=_response_hamiltonian(jet[1],occ,hops);D=len(occ)
        if np.max(abs(jet[0,0]-np.diag(np.diag(jet[0,0]))))==0:
            energies=H.diagonal().real
            chosen=energies<=np.min(energies)+1e-9;rank=int(np.sum(chosen))
            out[layer,0]=np.diag(occ[chosen].mean(axis=0))
            for i,j,src,dst,sign in hops:
                active=chosen[src]!=chosen[dst]
                a,b=src[active],dst[active]
                w=np.asarray(W[a,b]).ravel()
                v=np.sum(sign[active]*w*(chosen[a].astype(float)-chosen[b])/(energies[a]-energies[b]))/rank
                out[layer,1,i,j]=v;out[layer,1,j,i]=np.conj(v)
            continue
        if D<=100:
            E,Q=np.linalg.eigh(H.toarray());g=np.flatnonzero(E-E[0]<=1e-9)
            R=Q[:,g];outside=np.flatnonzero(E-E[0]>1e-9)
            dR=Q[:,outside]@((Q[:,outside].conj().T@W@R)/(E[g][None,:]-E[outside,None])) if len(outside) else np.zeros_like(R)
        else:
            k=4
            while True:
                E,Q=eigsh(H,k=min(k,D-2),which='SA',tol=2e-13,v0=np.cos(np.arange(D)*np.sqrt(2))+np.sin(np.arange(D)*np.sqrt(3)))
                order=np.argsort(E);E,Q=E[order],Q[:,order]
                if E[-1]-E[0]>1e-9:break
                k*=2
                if k>=D-2:
                    E,Q=np.linalg.eigh(H.toarray());break
            R=np.linalg.qr(Q[:,E-E[0]<=1e-9])[0]
            Eg,T=np.linalg.eigh(R.conj().T@(H@R));R=R@T;dR=np.zeros(R.shape,complex)
            for a,energy in enumerate(Eg):
                rhs=-(W@R[:,a]);rhs-=R@(R.conj().T@rhs)
                op=LinearOperator((D,D),matvec=lambda z:H@z-energy*z+R@(R.conj().T@z),dtype=complex)
                dR[:,a],info=cg(op,rhs,rtol=2e-12,atol=1e-13,maxiter=3000)
                if info:raise RuntimeError('response solve did not converge')
                dR[:,a]-=R@(R.conj().T@dR[:,a])
        for a in range(R.shape[1]):
            A=np.zeros((smaller,P),complex);dA=np.zeros_like(A)
            for i,(src,dst,sign) in enumerate(maps):
                A[dst,i]=sign*R[src,a];dA[dst,i]=sign*dR[src,a]
            out[layer,0]+=A.conj().T@A/R.shape[1]
            out[layer,1]+=(dA.conj().T@A+A.conj().T@dA)/R.shape[1]
    return out

import numpy as np

def addition_density(pair: 'np.ndarray | list | tuple') -> 'np.ndarray':
    a=np.asarray(pair)
    if a.ndim!=4 or a.shape[:2]!=(2,2) or a.shape[-1]!=a.shape[-2] or a.shape[-1]<1 or not np.isfinite(a).all():
        raise ValueError('finite (2,2,P,P) sector jets required')
    return a[0]-a[1]

import numpy as np

def modular_batch_curves(deltas: 'np.ndarray | list | tuple', distances: 'np.ndarray | list | tuple', trim: int, width: int) -> 'np.ndarray':
    jets=np.asarray(deltas);distances=np.asarray(distances)
    if jets.ndim!=5 or jets.shape[2]!=2 or jets.shape[-1]!=jets.shape[-2] or min(jets.shape)<1 or not np.isfinite(jets).all():
        raise ValueError('finite (B,S,2,P,P) density jets required')
    if not isinstance(width,(int,np.integer)) or width<1 or jets.shape[-1]%width or not isinstance(trim,(int,np.integer)) or trim<0:
        raise ValueError('integer width dividing P and nonnegative trim required')
    L=jets.shape[-1]//width
    if distances.ndim!=1 or len(distances)<1 or not np.isfinite(distances).all() or np.any(distances!=np.floor(distances)) or np.any(distances<0) or np.any(distances>=L-2*trim):
        raise ValueError('valid separations required')
    distances=distances.astype(int)
    B,S,_,P,_=jets.shape;L=P//width
    z=jets[:,:,0].reshape(B,S,L,width,L,width)
    dz=jets[:,:,1].reshape(B,S,L,width,L,width)
    result=np.zeros((B,3,len(distances)))
    for k,x in enumerate(distances):
        M=np.zeros((B,width,width));plus=np.zeros_like(M);minus=np.zeros_like(M)
        for i in range(trim,L-trim-x):
            a=z[:,:,i,:,i+x,:];da=dz[:,:,i,:,i+x,:];mag=abs(a);nonzero=mag>1e-12
            tangent=np.divide((a.conj()*da).real,mag,out=np.zeros_like(mag),where=nonzero)
            M+=np.where(nonzero,mag,0).mean(axis=1)/(L-2*trim-x)
            plus+=np.where(nonzero,tangent,abs(da)).mean(axis=1)/(L-2*trim-x)
            minus+=np.where(nonzero,-tangent,abs(da)).mean(axis=1)/(L-2*trim-x)
        for b in range(B):
            K=M[b].T@M[b];E,Q=np.linalg.eigh(K);result[b,0,k]=E[-1]
            U=Q[:,E>=E[-1]-1e-10*max(E[-1],1e-300)]
            for direction,Dm in enumerate((plus[b],minus[b]),1):
                G=Dm.T@M[b]+M[b].T@Dm
                result[b,direction,k]=np.linalg.eigvalsh(U.T@G@U)[-1]
    return result

import numpy as np

def fit_decay_dictionary(distances: 'np.ndarray | list | tuple', gamma: 'np.ndarray | list | tuple', frequency: float, rates: 'np.ndarray | list | tuple', oscillation_rates: 'np.ndarray | list | tuple', radius: float) -> 'np.ndarray':
    x = np.asarray(distances, dtype=float)
    g=np.asarray(gamma,dtype=float)
    if g.shape!=(3,len(x)) or not np.isfinite(g).all() or np.any(g[0]<0) or not np.isfinite(radius) or radius<0:
        raise ValueError('finite curve jet with nonnegative base and radius required')
    y=np.maximum(0.,g[0]+radius*np.minimum(0.,np.minimum(g[1],g[2])))
    ks = np.asarray(rates, dtype=float)
    os = np.asarray(oscillation_rates, dtype=float)
    if x.ndim != 1 or x.size < 2 or y.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x < 0) or np.any(y < 0):
        raise ValueError('distances and nonzero nonnegative gamma must be matching finite vectors')
    if not np.isfinite(frequency) or not 0 <= frequency <= 1:
        raise ValueError('frequency must lie in [0, 1]')
    if any(v.ndim != 1 or v.size == 0 or not np.isfinite(v).all() or np.any(v <= 0) for v in (ks, os)) or np.max(os) < np.min(ks):
        raise ValueError('positive finite dictionaries must contain an admissible rate pair')
    scale = np.max(y)
    if scale==0:
        k=float(np.min(ks));ko=float(np.min(os[os>=k]))
        return np.array([1/k,1/ko,0.,0.,0.])
    normalized = y / scale
    records = []
    for k in sorted(set(ks.tolist())):
        for ko in sorted(set(os.tolist())):
            if ko < k:
                continue
            u = np.exp(-k * x)
            v = np.cos(2 * np.pi * frequency * x) * np.exp(-ko * x)
            D = np.column_stack((u + v, u - v))
            # p=(A+B)/2, q=(A-B)/2 makes A>=|B| equivalent to p,q>=0.
            unconstrained = np.linalg.lstsq(D, normalized, rcond=None)[0]
            candidates = [unconstrained] if np.all(unconstrained >= 0) else []
            for j in range(2):
                z = np.zeros(2)
                norm = D[:, j] @ D[:, j]
                z[j] = max(0., float(D[:, j] @ normalized / norm)) if norm > 0 else 0.
                candidates.append(z)
            candidates.append(np.zeros(2))
            # A feasible least-squares solution is already globally optimal;
            # retain its minimum-norm choice when the design is rank deficient.
            p = unconstrained if np.all(unconstrained >= 0) else min(candidates, key=lambda z: (float(np.sum((D @ z - normalized) ** 2)), float(z @ z)))
            residual = float(np.linalg.norm(D @ p - normalized) / np.linalg.norm(normalized))
            records.append((residual, k, ko, (p[0] + p[1]) * scale, (p[0] - p[1]) * scale))
    minimum = min(r[0] for r in records)
    best = min((r for r in records if r[0] <= minimum + 1e-12), key=lambda r: (r[1], r[2]))
    residual, k, ko, amplitude, oscillation_amplitude = best
    return np.array([1 / k, 1 / ko, amplitude, oscillation_amplitude, residual], dtype=float)

import numpy as np

def paired_batch_statistics(short_fits: 'np.ndarray | list | tuple', long_fits: 'np.ndarray | list | tuple') -> 'np.ndarray':
    a = np.asarray(short_fits, dtype=float)
    b = np.asarray(long_fits, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2 or a.shape[1] != 5 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a[:, 0] <= 0) or np.any(b[:, 0] <= 0) or np.any(a[:, 4] < 0) or np.any(b[:, 4] < 0):
        raise ValueError('matching (B,5) fit arrays need B>=2, finite entries, positive lengths, nonnegative residuals')
    paired = np.column_stack((a[:, 0], b[:, 0]))
    means = np.mean(paired, axis=0)
    covariance = np.cov(paired, rowvar=False, ddof=1) / len(paired)
    return np.array([means[0], means[1], np.sqrt(covariance[0, 0]), np.sqrt(covariance[1, 1]), covariance[0, 1], np.max(np.r_[a[:, 4], b[:, 4]])])

import numpy as np

def select_localization_certificate(couplings: 'np.ndarray | list | tuple', statistics: 'np.ndarray | list | tuple', drift_limit: float, fit_limit: float, penalty: float) -> 'np.ndarray':
    v = np.asarray(couplings, dtype=float)
    a = np.asarray(statistics, dtype=float)
    if v.ndim != 1 or v.size < 1 or a.shape != (len(v), 6) or not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError('couplings and finite (C,6) statistics must match')
    if np.any(a[:, :2] <= 0) or np.any(a[:, 2:4] < 0) or np.any(a[:, 5] < 0) or np.any(np.abs(a[:, 4]) > a[:, 2] * a[:, 3] + 1e-12):
        raise ValueError('statistics must represent positive lengths and a positive semidefinite covariance')
    if any(not np.isfinite(q) or q < 0 for q in (drift_limit, fit_limit, penalty)):
        raise ValueError('limits and penalty must be finite and nonnegative')
    diff_se = np.sqrt(np.maximum(0., a[:, 2] ** 2 + a[:, 3] ** 2 - 2 * a[:, 4]))
    drift = np.abs(a[:, 1] - a[:, 0]) + penalty * diff_se
    score = a[:, 1] - penalty * a[:, 3]
    ids = np.flatnonzero((drift <= drift_limit) & (a[:, 5] <= fit_limit) & (score > 0))
    if len(ids) == 0:
        return np.array([-1., 0., 0., 0., 0., 0., 0.])
    maximum = np.max(score[ids])
    tied = ids[score[ids] >= maximum - 1e-10]
    i = min(tied, key=lambda j: (v[j], j))
    return np.array([score[i], v[i], a[i, 1], a[i, 3], diff_se[i], drift[i], a[i, 5]])

import numpy as np

def localization_design(seed: int = 49273, disorder: float = 1.9, couplings: 'np.ndarray | list | tuple' = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), lengths: 'np.ndarray | list | tuple' = (6, 8), batches: int = 4, samples: int = 2, drift_limit: float = 1.0, fit_limit: float = 0.2, penalty: float = 1.0, radius: float = 0.1) -> 'np.ndarray':
    if not isinstance(seed,(int,np.integer)) or seed<0 or not np.isfinite(disorder) or disorder<0:
        raise ValueError('nonnegative integer seed and finite nonnegative disorder required')
    if len(lengths)!=2 or any(not isinstance(x,(int,np.integer)) or x not in (6,8) for x in lengths) or lengths[0]>=lengths[1]:
        raise ValueError('lengths must be (6,8)')
    if not isinstance(batches,(int,np.integer)) or batches<2 or not isinstance(samples,(int,np.integer)) or samples<1:
        raise ValueError('integer batches>=2 and samples>=1 required')
    couplings=np.asarray(couplings,dtype=float)
    if couplings.ndim!=1 or len(couplings)<1 or not np.isfinite(couplings).all():
        raise ValueError('nonempty finite couplings required')
    potentials=disorder*np.where(np.random.default_rng(seed).uniform(-1,1,(batches,samples,lengths[1]))<0,-1.,1.)
    x=np.arange(1,5);rates=np.arange(1,41,dtype=float)/40
    oscillation_rates=np.array([.025,.05,.075,.1,.15,.2,.3,.4,.6,.8,1.,1.4,2.,3.])
    if not np.isfinite(radius) or radius<0:raise ValueError('nonnegative finite radius required')
    statistics=[];diagnostics=[]
    for V in couplings:
        fitted=[]
        for L in lengths:
            bonds=[]
            for i in range(L):
                if i%4==0:bonds.extend(((2*i,2*i+2),(2*i+1,2*i+3)))
                elif i%4 in (2,3):bonds.append((2*i,2*i+1))
            deltas=np.empty((batches,samples,2,2*L,2*L),dtype=complex)
            for b in range(batches):
                for s in range(samples):
                    model=bond_ladder_model(potentials[b,s,:L],L,2,bonds,1.,V)
                    pair=sector_density_pair(model,2*L//3)
                    deltas[b,s]=addition_density(pair)
            curves=modular_batch_curves(deltas,x,0,2)
            fitted.append(np.array([fit_decay_dictionary(x,curve,1/3,rates,oscillation_rates,radius) for curve in curves]))
        diagnostics.append(curves[0,:,1])
        statistics.append(paired_batch_statistics(*fitted))
    certificate=select_localization_certificate(couplings,np.array(statistics),drift_limit,fit_limit,penalty)
    if certificate[0]<0:return np.r_[certificate,np.zeros(3)]
    index=int(np.flatnonzero(couplings==certificate[1])[0])
    return np.r_[certificate,diagnostics[index]]
SCICODE_GOLD_EOF
