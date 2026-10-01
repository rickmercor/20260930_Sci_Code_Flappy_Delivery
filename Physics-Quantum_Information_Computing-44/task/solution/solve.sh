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
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def psk_channel(N: int, amplitude: float, eta: float, phase: float) -> np.ndarray:
    if N not in (2,4) or not np.isfinite([amplitude,eta,phase]).all() or amplitude<0 or not 0<=eta<=1:
        raise ValueError('N must be 2 or 4; amplitude >= 0; eta in [0,1]; phase finite')
    if N==2:
        p=.5*(1+erf(np.sqrt(2*eta)*amplitude*np.cos(phase)))
        return np.array([[p,1-p],[1-p,p]])
    z=np.sqrt(eta)*amplitude*np.exp(1j*(np.pi/4+phase+np.arange(4)*np.pi/2))
    a=.5*(1+erf(z.real));b=.5*(1+erf(z.imag))
    return np.array([a*b,(1-a)*b,(1-a)*(1-b),a*(1-b)])

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def cyclic_weights(N: int, mean_photons: float) -> np.ndarray:
    if N not in (2,4) or not np.isfinite(mean_photons) or not 0<=mean_photons<=16:
        raise ValueError('N in {2,4}, mean_photons in [0,16]')
    w=np.zeros(N);term=np.exp(-mean_photons);w[0]=term
    for k in range(1,192):
        term*=mean_photons/k;w[k%N]+=term
    return w/w.sum()

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def reverse_states(channel: np.ndarray, weights: np.ndarray) -> np.ndarray:
    P=np.asarray(channel,float);w=np.asarray(weights,float)
    if w.ndim!=1:
        raise ValueError('weights must be a one-dimensional normalized vector')
    N=len(w)
    if N not in (2,4) or P.shape!=(N,N) or not np.isfinite(P).all() or not np.isfinite(w).all() or np.min(P)<0 or np.min(w)<0 or not np.allclose(P.sum(axis=0),1,atol=1e-10) or not np.allclose(P.sum(axis=1),1,atol=1e-10) or abs(w.sum()-1)>1e-10:
        raise ValueError('channel must be doubly stochastic; weights must be normalized')
    v=np.sqrt(w)[:,None]*np.exp(2j*np.pi*np.arange(N)[:,None]*np.arange(N)/N)
    return np.array([(v*row)@v.conj().T for row in P])

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def _checked_state(state):
    r=np.array(state,dtype=complex,copy=True)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or len(r) not in (2,4) or not np.isfinite(r).all() or not np.allclose(r,r.conj().T,atol=1e-10) or abs(np.trace(r)-1)>1e-9 or np.linalg.eigvalsh(r).min() < -1e-10:
        raise ValueError('state must be a normalized positive Hermitian 2x2 or 4x4 matrix')
    return (r+r.conj().T)/2

def entropy_statistics(state: np.ndarray) -> np.ndarray:
    r=_checked_state(state);N=len(r);w=r.diagonal().real
    active=w>1e-15;r=r[np.ix_(active,active)];w=w[active];r/=w.sum();w=w/w.sum()
    e,u=np.linalg.eigh(r);e=np.maximum(e,0)
    loge=np.log2(np.maximum(e,1e-300));logw=np.log2(w)
    D=np.sum(e*loge)-np.sum(w*logw)
    # Spectral-overlap formula avoids forming a logarithm on a null eigenvector.
    overlaps=np.abs(u)**2
    second=np.sum((overlaps*e[None,:])*(loge[None,:]-logw[:,None])**2)
    H2=np.log2(N)-np.log2(np.sum(np.abs(r)**2/w[None,:]))
    return np.array([np.log2(N)-D,max(0,second-D*D),H2])

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def _checked_state(state):
    r=np.array(state,dtype=complex,copy=True)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or len(r) not in (2,4) or not np.isfinite(r).all() or not np.allclose(r,r.conj().T,atol=1e-10) or abs(np.trace(r)-1)>1e-9 or np.linalg.eigvalsh(r).min() < -1e-10:
        raise ValueError('state must be a normalized positive Hermitian 2x2 or 4x4 matrix')
    return (r+r.conj().T)/2

def _sand_loss(state, sigma, reciprocal_order, gradient=False):
    t=reciprocal_order;d=np.exp((t-1)*np.log(sigma)/2)
    B=d[:,None]*state*d[None,:]
    e,u=np.linalg.eigh(B);e=np.maximum(e,1e-300)
    logs=np.log(e)/t;L=logsumexp(logs)
    value=t/(1-t)*L/np.log(2)
    if not gradient:return value
    diagonal=np.abs(u)**2@np.exp(logs-L)
    return value,-diagonal/sigma/np.log(2)

def sandwiched_optimum(state: np.ndarray, reciprocal_order: float) -> np.ndarray:
    r=_checked_state(state);N=len(r);t=float(reciprocal_order)
    if not np.isfinite(t) or not 0<t<1:raise ValueError('reciprocal_order must be in (0,1)')
    w=r.diagonal().real;active=w>1e-15;rr=r[np.ix_(active,active)];ww=w[active];rr=rr/ww.sum();ww=ww/ww.sum()
    out=np.zeros(N+1)
    if len(ww)==1:out[0]=np.log2(N);out[1:][active]=1;return out
    # Optimize on the symmetry-invariant simplex; the derivative is analytic.
    fit=minimize(lambda s:_sand_loss(rr,s,t,True),ww,jac=True,method='SLSQP',
        bounds=[(1e-14,1)]*len(ww),constraints=[{'type':'eq','fun':lambda s:s.sum()-1,'jac':lambda s:np.ones_like(s)}],
        options={'ftol':3e-13,'maxiter':300})
    sigma=np.maximum(fit.x,1e-14);sigma/=sigma.sum()
    residual=_sand_loss(rr,sigma,t,True)[1]
    if np.ptp(residual)>3e-5:
        # A second, unconstrained parametrization repairs poorly scaled endpoints.
        def _fun(z):
            logits=np.r_[z,0.];s=np.exp(logits-logsumexp(logits));val,g=_sand_loss(rr,s,t,True)
            return val,(s*(g-np.dot(s,g)))[:-1]
        alt=minimize(_fun,np.log(sigma[:-1]/sigma[-1]),jac=True,method='BFGS',options={'gtol':2e-10,'maxiter':400})
        logits=np.r_[alt.x,0.];sigma=np.exp(logits-logsumexp(logits))
    out[0]=np.log2(N)-_sand_loss(rr,sigma,t);out[1:][active]=sigma
    return out

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def _checked_state(state):
    r=np.array(state,dtype=complex,copy=True)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or len(r) not in (2,4) or not np.isfinite(r).all() or not np.allclose(r,r.conj().T,atol=1e-10) or abs(np.trace(r)-1)>1e-9 or np.linalg.eigvalsh(r).min() < -1e-10:
        raise ValueError('state must be a normalized positive Hermitian 2x2 or 4x4 matrix')
    return (r+r.conj().T)/2

def min_entropy_optimum(state: np.ndarray) -> np.ndarray:
    r=_checked_state(state);N=len(r);w=r.diagonal().real;active=w>1e-15
    rr=r[np.ix_(active,active)];rr/=w[active].sum();out=np.zeros(N+1)
    if len(rr)==1:out[0]=np.log2(N);out[1:][active]=1;return out
    def _eig(z):return np.linalg.eigvalsh(np.diag(z)-rr)
    def _jac(z):return (np.abs(np.linalg.eigh(np.diag(z)-rr)[1])**2).T
    initial=np.sum(np.abs(rr),axis=1)+1e-10
    fit=minimize(lambda z:z.sum(),initial,jac=lambda z:np.ones_like(z),method='SLSQP',
      constraints=[{'type':'ineq','fun':_eig,'jac':_jac}],bounds=[(1e-14,None)]*len(rr),options={'ftol':3e-13,'maxiter':400})
    z=fit.x.copy();z+=max(0.,-_eig(z).min())+2e-14
    out[0]=np.log2(N)-np.log2(z.sum());out[1:][active]=z/z.sum()
    return out

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def _checked_state(state):
    r=np.array(state,dtype=complex,copy=True)
    if r.ndim!=2 or r.shape[0]!=r.shape[1] or len(r) not in (2,4) or not np.isfinite(r).all() or not np.allclose(r,r.conj().T,atol=1e-10) or abs(np.trace(r)-1)>1e-9 or np.linalg.eigvalsh(r).min() < -1e-10:
        raise ValueError('state must be a normalized positive Hermitian 2x2 or 4x4 matrix')
    return (r+r.conj().T)/2

def _sand_loss(state, sigma, reciprocal_order, gradient=False):
    t=reciprocal_order;d=np.exp((t-1)*np.log(sigma)/2)
    B=d[:,None]*state*d[None,:]
    e,u=np.linalg.eigh(B);e=np.maximum(e,1e-300)
    logs=np.log(e)/t;L=logsumexp(logs)
    value=t/(1-t)*L/np.log(2)
    if not gradient:return value
    diagonal=np.abs(u)**2@np.exp(logs-L)
    return value,-diagonal/sigma/np.log(2)

def robust_rate(states: np.ndarray, channels: np.ndarray, statistics: np.ndarray, n: int, epsilon: float, epsilon_prime: float) -> np.ndarray:
    rs=np.asarray(states,complex);ps=np.asarray(channels,float);st=np.asarray(statistics,float)
    if rs.ndim!=3 or rs.shape[1]!=rs.shape[2] or rs.shape[1] not in (2,4) or ps.shape!=rs.shape or st.shape!=(len(rs),3) or len(rs)==0 or not np.isfinite(ps).all() or not np.isfinite(st).all() or not np.isfinite([n,epsilon,epsilon_prime]).all() or n<1 or n!=int(n) or not 0<epsilon<1 or not 0<epsilon_prime<1:
        raise ValueError('incompatible arrays or invalid block/security parameters')
    if np.any(ps < 0) or not np.allclose(ps.sum(axis=1), 1, rtol=0, atol=1e-10) or not np.allclose(ps.sum(axis=2), 1, rtol=0, atol=1e-10):
        raise ValueError('channels must be nonnegative doubly stochastic probability matrices')
    N=rs.shape[1];S=len(rs)
    for r in rs:_checked_state(r)
    leak=np.array([-np.sum(p[p>0]*np.log2(p[p>0]))/N for p in ps])
    g=np.log2(1+np.sqrt((1-epsilon)*(1+epsilon)))-2*np.log2(epsilon)
    extract=(1+2*np.log2(epsilon_prime))/n
    mins=np.array([min_entropy_optimum(r)[0] for r in rs])
    cache={}
    def _values(t):
        key=float(t)
        if key not in cache:
            h=mins if key==0 else np.array([sandwiched_optimum(r,key)[0] for r in rs])
            cache[key]=(h,h+extract-g*key/(n*(1-key))-leak)
        return cache[key]
    grid=np.r_[0,np.linspace(.02,.98,33)]
    vals=np.array([np.min(_values(t)[1]) for t in grid])
    candidates=[(vals[i],grid[i]) for i in [0,len(grid)-1]]
    for i in range(1,len(grid)-1):
        if i==1 or (vals[i]>=vals[i-1] and vals[i]>=vals[i+1]):
            fit=minimize_scalar(lambda t:-min(_values(t)[1]),bounds=(max(1e-7,grid[i-1]),grid[i+1]),method='bounded',options={'xatol':1e-9})
            candidates.append((-fit.fun,fit.x))
    value,t=max(candidates,key=lambda z:(z[0],-z[1]))
    # Refine scenario intersections around the selected point.
    if t>0:
        for i in range(S):
            for j in range(i):
                a=max(1e-6,t-.002);b=min(.98,t+.002)
                _fun=lambda u:_values(u)[1][i]-_values(u)[1][j]
                if _fun(a)*_fun(b)<0:
                    root=brentq(_fun,a,b,xtol=1e-11);v=min(_values(root)[1])
                    if v>value-1e-11 and abs(root-t)<1e-5:value,t=v,root
    H,R=_values(t);j=int(np.flatnonzero(R<=np.min(R)+1e-7)[0]);r=rs[j];h,V,h2=st[j]
    active=r.diagonal().real>1e-15;rr=r[np.ix_(active,active)];w=rr.diagonal().real;w=w/w.sum();rr=rr/np.trace(rr).real
    if t==0:
        D=np.diag(w**-.5);down=np.log2(N)-np.log2(np.linalg.eigvalsh(D@rr@D).max())
    else:down=np.log2(N)-_sand_loss(rr,w,t)
    Rdown=down+extract-g*t/(n*(1-t))-leak[j]
    delta=4*np.log2(2+np.sqrt(N))*np.sqrt(np.log2(2)-2*np.log2(epsilon))
    RAEP=float(np.min(st[:,0]+extract-delta/np.sqrt(n)-leak))
    # The continuity expansion is a scientifically distinct comparator, optimized on (1,2).
    def _continuity(a,k):
        rk=rs[k];wk=rk.diagonal().real;ac=wk>1e-15;rk=rk[np.ix_(ac,ac)];wk=wk[ac];rk/=wk.sum();wk/=wk.sum()
        e,u=np.linalg.eigh(rk);e=np.maximum(e,0)
        trace=np.sum((np.abs(u)**2@e**a)*wk**(1-a))
        hp=np.log2(N)+np.log2(trace)/(1-a)
        hk,vk,h2k=st[k]
        logK=(a-1)*(-hp+hk)*np.log(2)+3*np.log(np.log(2**(-h2k+hk)+np.exp(2)))-np.log(6*np.log(2))-3*np.log(2-a)
        return hk-(a-1)*np.log(2)*vk/2-(a-1)**2*np.exp(logK)+extract-g/(n*(a-1))-leak[k]
    fit=minimize_scalar(lambda a:-min(_continuity(a,k) for k in range(S)),bounds=(1.00001,1.99),method='bounded',options={'xatol':2e-9})
    RB=-fit.fun
    return np.array([np.min(R),t,j,H[j],Rdown,RB,RAEP,np.min(mins+extract-leak)])

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def select_protocol(menu: np.ndarray, certificates: np.ndarray, photon_budget: float) -> np.ndarray:
    m=np.asarray(menu,float);c=np.asarray(certificates,float)
    if m.ndim!=2 or m.shape[1]!=2 or c.shape!=(len(m),8) or not np.isfinite(m).all() or not np.isfinite(c).all() or not np.isfinite(photon_budget) or photon_budget<0 or np.any(m[:,1]<0) or not np.isin(m[:,0],[2,4]).all():
        raise ValueError('invalid menu, certificate shape or photon budget')
    feasible=np.flatnonzero(m[:,1]**2<=photon_budget+1e-12)
    if not len(feasible):raise ValueError('no protocol satisfies photon budget')
    best=np.max(c[feasible,0]);i=int(feasible[np.flatnonzero(c[feasible,0]>=best-1e-9)[0]])
    z=c[i];return np.r_[z[0],i,z[2],z[1],z[3:]]

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def qkd_benchmark(menu: np.ndarray, scenarios: np.ndarray, n: int, epsilon: float, epsilon_prime: float, photon_budget: float) -> np.ndarray:
    menu=np.asarray(menu,float);scenarios=np.asarray(scenarios,float)
    if menu.ndim!=2 or menu.shape[1]!=2 or not len(menu) or scenarios.ndim!=2 or scenarios.shape[1]!=2 or not len(scenarios):raise ValueError('menu and scenarios must be nonempty two-column arrays')
    certificates=[]
    for N,amplitude in menu:
        if N not in (2,4):raise ValueError('N must be 2 or 4')
        ps=[];rs=[];stats=[]
        for eta,phase in scenarios:
            P=psk_channel(int(N),amplitude,eta,phase)
            w=cyclic_weights(int(N),(1-eta)*amplitude**2)
            rho=reverse_states(P,w)[0]
            ps.append(P);rs.append(rho);stats.append(entropy_statistics(rho))
        certificates.append(robust_rate(np.array(rs),np.array(ps),np.array(stats),n,epsilon,epsilon_prime))
    return select_protocol(menu,np.array(certificates),photon_budget)
SCICODE_GOLD_EOF
