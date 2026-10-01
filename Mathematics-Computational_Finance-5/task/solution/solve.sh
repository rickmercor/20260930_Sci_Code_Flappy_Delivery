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

def matched_arrival_rate(T,target,alpha,beta,q0):
    np=__import__('numpy')
    if not np.all(np.isfinite([T,target,alpha,beta])) or min(T,target)<=0 or not beta>alpha>0:
        raise ValueError('model inputs')
    if isinstance(q0,(bool,np.bool_)) or not np.isscalar(q0) or not np.isfinite(q0) or int(q0)!=q0 or q0<0:
        raise ValueError('q0')
    def c(x):return np.array([x,0.,0.])
    def mul(a,b):return np.array([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]])
    def div(a,b):
        inverse=np.array([1/b[0],-b[1]/b[0]**2,2*b[1]**2/b[0]**3-b[2]/b[0]**2])
        return mul(a,inverse)
    def exp(a):
        y=np.exp(a[0]);return np.array([y,y*a[1],y*(a[2]+a[1]**2)])
    al=np.array([alpha,1.,0.]);delta=c(beta)-al
    survival_integral=div(c(1)-exp(-T*delta),delta)
    immigration_count=c(T)+div(mul(al,c(T)-survival_integral),delta)
    inherited_count=q0*mul(al,survival_integral)
    result=div(c(target)-inherited_count,immigration_count)
    if result[0]<=0:raise ValueError('target implies a nonpositive baseline')
    return result

import numpy as np

def queue_kernel_curvature(v,tau,lambda_jet,alpha,beta,mu_J,sigma_J,max_state):
    np=__import__('numpy')
    v=np.asarray(v,float);lj=np.asarray(lambda_jet,float)
    if v.ndim!=1 or not v.size or not np.all(np.isfinite(v)) or lj.shape!=(3,) or not np.all(np.isfinite(lj)):
        raise ValueError('array inputs')
    if not np.all(np.isfinite([tau,alpha,beta,mu_J,sigma_J])) or tau<0 or sigma_J<0 or not beta>alpha>0 or lj[0]<=0:
        raise ValueError('model inputs')
    if isinstance(max_state,(bool,np.bool_)) or not np.isscalar(max_state) or not np.isfinite(max_state) or int(max_state)!=max_state or max_state<0:
        raise ValueError('max_state')
    n=int(max_state)+1
    out=np.zeros((3,n,n,len(v)),complex)
    if tau==0:
        out[0,np.arange(n),np.arange(n),:]=1
        return out
    def const(x):
        x=np.asarray(x,complex)
        if x.ndim==0:x=x.reshape(1)
        return np.stack([x,np.zeros_like(x),np.zeros_like(x)])
    def mul(a,b):
        return np.stack([a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]])
    def inv(a):
        return np.stack([1/a[0],-a[1]/a[0]**2,2*a[1]**2/a[0]**3-a[2]/a[0]**2])
    def div(a,b):return mul(a,inv(b))
    def exp(a):
        z=np.exp(a[0]);return np.stack([z,z*a[1],z*(a[2]+a[1]**2)])
    def log(a):
        return np.stack([np.log(a[0]),a[1]/a[0],a[2]/a[0]-(a[1]/a[0])**2])
    al=np.array([[alpha],[1.],[0.]],complex)
    lam=lj[:,None].astype(complex)
    mark=const(np.exp(1j*mu_J*v-.5*sigma_J*sigma_J*v*v))
    f=exp(.5*log(mul(al+const(beta),al+const(beta))-4*beta*mul(al,mark)))
    e=exp(-tau*f);one=const(1)
    h=mul(f,one+e)+mul(const(beta)+al,one-e)
    p=div(2*mul(mul(al,mark),one-e),h)
    u=div(2*beta*(one-e),h)
    w=div(mul(f,one+e)-mul(const(beta)+al,one-e),h)
    r=div(lam,al)
    out[:,0,0,:]=exp(.5*tau*mul(r,const(beta)-al-f)+mul(r,log(div(2*f,h))))
    for j in range(1,n):
        out[:,0,j,:]=mul(mul(out[:,0,j-1,:],r+const(j-1)),p)/j
    ratio=np.zeros((3,n,len(v)),complex);ratio[:,0,:]=u
    if n>1:ratio[:,1,:]=w+mul(u,p)
    for j in range(2,n):ratio[:,j,:]=mul(ratio[:,j-1,:],p)
    for q in range(1,n):
        for j in range(n):
            out[:,q,j,:]=sum(mul(out[:,q-1,k,:],ratio[:,j-k,:]) for k in range(j+1))
    return out

import numpy as np

def diffusion_discount_factor(v,tau,mu,sigma,rho):
    np=__import__('numpy');v=np.asarray(v,float)
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite([tau,mu,sigma,rho])) or min(tau,sigma)<0:
        raise ValueError('diffusion inputs')
    return np.exp(1j*v*(mu-.5*sigma*sigma)*tau-.5*sigma*sigma*v*v*tau-rho*tau)

import numpy as np

def call_payoff_cosine_coeffs(K, a_p, b_p, N_in):
    """Reference implementation."""
    np = __import__("numpy")
    K = float(K)
    a_p = float(a_p)
    b_p = float(b_p)
    if not np.isfinite(K) or K <= 0.0:
        raise ValueError("K must be a finite positive number")
    if not (np.isfinite(a_p) and np.isfinite(b_p)) or b_p <= a_p:
        raise ValueError("require a finite interval with b_p > a_p")
    if isinstance(N_in, bool) or int(N_in) != N_in or int(N_in) < 1:
        raise ValueError("N_in must be an integer >= 1")
    N_in = int(N_in)

    k = np.arange(N_in)
    w = k * np.pi / (b_p - a_p)
    c = max(a_p, np.log(K))
    d = b_p
    if c >= d:
        return np.zeros(N_in)
    wc = w * (c - a_p)
    wd = w * (d - a_p)
    chi = (np.cos(wd) * np.exp(d) - np.cos(wc) * np.exp(c)
           + w * np.sin(wd) * np.exp(d) - w * np.sin(wc) * np.exp(c)) \
        / (1.0 + w ** 2)
    psi = np.empty(N_in)
    psi[0] = d - c
    psi[1:] = (np.sin(wd[1:]) - np.sin(wc[1:])) / w[1:]
    return 2.0 / (b_p - a_p) * (chi - K * psi)

import numpy as np

def state_continuation_jet(kernel_jet,payoff_jet,diffusion):
    np=__import__('numpy')
    K=np.asarray(kernel_jet,complex);H=np.asarray(payoff_jet,float);D=np.asarray(diffusion,complex)
    if K.ndim!=4 or K.shape[0]!=3 or min(K.shape)<1 or H.shape!=(3,K.shape[2],K.shape[3]) or D.shape!=(K.shape[3],):
        raise ValueError('shapes')
    if not all(np.all(np.isfinite(x)) for x in [K,H,D]):raise ValueError('nonfinite coefficients')
    def product(k,h):return np.einsum('qjk,jk->qk',k,h)
    return np.stack([product(K[0],H[0]),product(K[1],H[0])+product(K[0],H[1]),
                     product(K[2],H[0])+2*product(K[1],H[1])+product(K[0],H[2])])*D

import numpy as np

def queue_exercise_boundaries(C,omega,a_next,a,b,cost):
    np=__import__('numpy');brentq=__import__('scipy.optimize',fromlist=['brentq']).brentq
    C=np.asarray(C,complex);w=np.asarray(omega,float)
    if C.ndim!=2 or min(C.shape)<1 or w.shape!=(C.shape[1],) or not np.all(np.isfinite(C)) or not np.all(np.isfinite(w)):
        raise ValueError('coefficient shapes')
    if w[0]!=0 or np.any(np.diff(w)<=0) or not np.all(np.isfinite([a_next,a,b,cost])) or b<=a or cost<=0:
        raise ValueError('ranges')
    prime=np.ones(len(w));prime[0]=.5;roots=[]
    for row in C:
        def f(x):return np.real(row*np.exp(1j*w*(x-a_next)))@prime-cost
        if not f(a)<0<f(b):raise ValueError('strict boundary bracket')
        roots.append(brentq(f,a,b,xtol=1e-12,rtol=1e-12))
    return np.array(roots)

import numpy as np

def queue_payoff_jet(C,omega,nu,a_next,a,b,roots,cost):
    np=__import__('numpy');C=np.asarray(C,complex);w=np.asarray(omega,float);nu=np.asarray(nu,float);roots=np.asarray(roots,float)
    if C.ndim!=3 or C.shape[0]!=3 or min(C.shape)<1 or w.shape!=(C.shape[2],) or roots.shape!=(C.shape[1],) or nu.ndim!=1 or not nu.size:
        raise ValueError('shapes')
    if not all(np.all(np.isfinite(x)) for x in [C,w,nu,roots]) or not np.all(np.isfinite([a_next,a,b,cost])):
        raise ValueError('nonfinite inputs')
    if b<=a or cost<=0 or w[0]!=0 or nu[0]!=0 or np.any(np.diff(w)<=0) or np.any(np.diff(nu)<=0) or np.any(roots<=a) or np.any(roots>=b):
        raise ValueError('ranges')
    prime=np.ones(len(w));prime[0]=.5
    out=np.zeros((3,len(roots),len(nu)))
    minus=w[None,:]-nu[:,None];plus=w[None,:]+nu[:,None]
    phase_minus=w[None,:]*a_next-nu[:,None]*a
    phase_plus=w[None,:]*a_next+nu[:,None]*a
    for q,r in enumerate(roots):
        length=b-r;middle=(b+r)/2
        sm=np.sinc(minus*length/(2*np.pi));sp=np.sinc(plus*length/(2*np.pi))
        Ic=length/(b-a)*(np.cos(minus*middle-phase_minus)*sm+np.cos(plus*middle-phase_plus)*sp)
        Is=length/(b-a)*(np.sin(minus*middle-phase_minus)*sm+np.sin(plus*middle-phase_plus)*sp)
        Io=2*length/(b-a)*np.cos(nu*(middle-a))*np.sinc(nu*length/(2*np.pi))
        for d in range(3):out[d,q]=(Ic*prime)@C[d,q].real-(Is*prime)@C[d,q].imag
        out[0,q]-=cost*Io
        phase=np.exp(1j*w*(r-a_next))
        slope=np.real(C[0,q]*1j*w*phase)@prime
        first=np.real(C[1,q]*phase)@prime
        if slope<=0:raise ValueError('nonpositive slope')
        out[2,q]+=2/(b-a)*np.cos(nu*(r-a))*first*first/slope
    return out

import numpy as np

def persistent_clustering_curvature(times=(.75,1.5,2.5),costs=(6.,10.),
        intervals=((1.2,8.4),(-.6,10.2),(-2.8,12.4)),N=(192,256,384),
        max_state=32,S0=100.,Kterm=105.,mu=.04,sigma=.22,rho=.08,
        target=2.4,alpha=.35,beta=.9,mu_J=.06,sigma_J=.18,q0=2):
    np=__import__('numpy')
    t=np.asarray(times,float);c=np.asarray(costs,float);I=np.asarray(intervals,float);ns=np.asarray(N)
    if t.ndim!=1 or len(t)<2 or c.shape!=(len(t)-1,) or I.shape!=(len(t),2) or ns.shape!=t.shape:
        raise ValueError('configuration shapes')
    if not all(np.all(np.isfinite(z)) for z in [t,c,I,ns]) or np.any(np.diff(np.r_[0.,t])<=0) or np.any(c<=0) or np.any(I[:,1]<=I[:,0]) or np.any(ns<16) or np.any(ns!=np.floor(ns)):
        raise ValueError('configuration ranges')
    if not np.all(np.isfinite([S0,Kterm,mu,sigma,rho])) or min(S0,Kterm,sigma)<=0:
        raise ValueError('project inputs')
    if isinstance(max_state,(bool,np.bool_)) or not np.isscalar(max_state) or not np.isfinite(max_state) or int(max_state)!=max_state or max_state<0:
        raise ValueError('max_state')
    if isinstance(q0,(bool,np.bool_)) or not np.isscalar(q0) or not np.isfinite(q0) or int(q0)!=q0 or not 0<=q0<=max_state:
        raise ValueError('q0')
    if not I[0,0]<np.log(S0)<I[0,1]:raise ValueError('initial log state')
    ns=ns.astype(int);states=int(max_state)+1;q0=int(q0);dt=np.diff(np.r_[0.,t])
    lj=matched_arrival_rate(t[-1],target,alpha,beta,q0)
    frequencies=[np.arange(n)*np.pi/(b-a) for n,(a,b) in zip(ns,I)]
    H=np.zeros((3,states,ns[-1]))
    H[0]=call_payoff_cosine_coeffs(Kterm,*I[-1],ns[-1])
    for stage in range(len(c)-1,-1,-1):
        w=frequencies[stage+1]
        K=queue_kernel_curvature(w,dt[stage+1],lj,alpha,beta,mu_J,sigma_J,max_state)
        D=diffusion_discount_factor(w,dt[stage+1],mu,sigma,rho)
        C=state_continuation_jet(K,H,D)
        roots=queue_exercise_boundaries(C[0],w,I[stage+1,0],*I[stage],c[stage])
        H=queue_payoff_jet(C,w,frequencies[stage],I[stage+1,0],*I[stage],roots,c[stage])
    w=frequencies[0]
    K=queue_kernel_curvature(w,dt[0],lj,alpha,beta,mu_J,sigma_J,max_state)
    D=diffusion_discount_factor(w,dt[0],mu,sigma,rho)
    C=state_continuation_jet(K,H,D)
    prime=np.ones(len(w));prime[0]=.5
    return float(np.real(C[2,q0]*np.exp(1j*w*(np.log(S0)-I[0,0])))@prime)
SCICODE_GOLD_EOF
