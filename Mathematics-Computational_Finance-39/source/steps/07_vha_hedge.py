"""
Assembles the variance-optimal initial stock holding for the geometric Asian call from the transform values and the Riccati-Volterra solutions on the contour, together with the held-fixing delta.

In the stock-and-cash market the option cannot be replicated; the variance-optimal strategy minimises the expected squared terminal error. The source gives it semi-explicitly in the same transform coordinates as the price, and it is not the delta of the price.

Returns
-------
A float64 array of shape (2,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vha_hedge(S0, K, N, sigma, rho, R, y, wq, h1, hz, psi2_1, psi2_z):
    r"""S0: positive float, spot. K: positive float, strike. N: positive integer, monitoring intervals.
    sigma: positive float; rho: float in $[-1, 1]$. R: float in $(0, 1)$, real part of the contour.
    y, wq: $(M,)$ contour arrays. h1: $(2, 1)$ and hz: $(2, M)$, the transform at $(1, 0)$ and at
    $(z, 0)$ for the nodes $z = R + iy$. psi2_1:
    $(2, 1, n+1)$ and psi2_z: $(2, M, n+1)$, the Riccati-Volterra solutions for the same arguments,
    at zero rate.

    Returns a numpy float64 array of shape $(2,)$: the variance-optimal initial holding in the stock
    for the fixed-strike discretely monitored geometric Asian call, by the source's representation of
    the variance-optimal strategy in transform coordinates, and the partial derivative of the price
    with respect to the spot with the spot fixing held fixed.

    Raises:
        ValueError: on non-finite or non-positive S0, K or sigma, rho outside $[-1, 1]$, R outside
            $(0, 1)$, a non-integral or non-positive N, arrays that are not finite, or inconsistent shapes.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _kernel(t, lam):
    t=np.asarray(t,dtype=np.float64); out=np.ones_like(t); m=t>0
    out[m]=-np.expm1(-lam*t[m])/(lam*t[m]); return out
def _pos(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0.0: raise ValueError("bad "+name)
    return v
def _nonneg(x,name):
    v=float(x)
    if not np.isfinite(v) or v<0.0: raise ValueError("bad "+name)
    return v
def _posint(x,name):
    v=float(x)
    if not np.isfinite(v) or v<=0 or abs(v-round(v))>0: raise ValueError("bad "+name)
    return int(round(v))
def _arr(x,name,n=None):
    a=np.asarray(x,dtype=np.float64).ravel()
    if (n is not None and a.size!=n) or a.size==0 or not np.all(np.isfinite(a)): raise ValueError("bad "+name)
    return a
def _mass(T,N,m):
    n=N*m; grid=np.linspace(0.0,T,n+1); tc=0.5*(grid[1:]+grid[:-1]); tj=np.arange(N+1)*T/N
    return np.array([np.sum(tj>=T-t) for t in tc],dtype=np.float64)/(N+1)
def _xi(nu0,kappa,theta,lam,T,n):
    grid=np.linspace(0.0,T,n+1); dt=T/n; Kd=_kernel(grid,lam); xi=np.empty(n+1); xi[0]=nu0
    for i in range(1,n+1):
        w=np.full(i+1,dt); w[0]=0.5*dt; w[-1]=0.5*dt
        known=float(np.sum(w[:i]*Kd[i-np.arange(i)]*kappa*(theta-xi[:i])))
        a=w[i]*Kd[0]*kappa; xi[i]=(nu0+known+a*theta)/(1.0+a)
    return xi
def _ric(s,w,mass,kappa,sigma,rho,lam,T):
    s=np.atleast_1d(np.asarray(s,dtype=np.complex128)); n=len(mass); dt=T/n
    Kd=_kernel(np.linspace(0.0,T,n+1),lam)
    def R(x1,x2): return 0.5*(x1*x1-x1)-kappa*x2+0.5*(sigma*sigma*x2*x2+2.0*rho*sigma*x1*x2)
    p2=np.zeros((len(s),n+1),dtype=np.complex128)
    Rl=np.zeros((len(s),n),dtype=np.complex128); Rr=np.zeros((len(s),n),dtype=np.complex128)
    for i in range(1,n+1):
        k=i-1; x1=s*mass[k]+w
        Rl[:,k]=R(x1,p2[:,k])
        if k>0:
            idx=np.arange(k)
            acc=0.5*dt*(np.einsum('mc,c->m',Rl[:,:k],Kd[i-idx])+np.einsum('mc,c->m',Rr[:,:k],Kd[i-1-idx]))
        else:
            acc=np.zeros(len(s),dtype=np.complex128)
        acc=acc+0.5*dt*Kd[1]*Rl[:,k]
        pred=acc+0.5*dt*Kd[0]*Rl[:,k]
        p2[:,i]=acc+0.5*dt*Kd[0]*R(x1,pred)
        Rr[:,k]=R(x1,p2[:,i])
    return p2
def _H0(s,w,S0,r,T,N,mass,xi,p2,sigma,rho):
    s=np.atleast_1d(np.asarray(s,dtype=np.complex128)); n=len(mass); dt=T/n
    I1=np.zeros(len(s),dtype=np.complex128); I2=np.zeros(len(s),dtype=np.complex128)
    for k in range(n):
        kt=n-1-k; x1=s*mass[kt]+w; xa,xb=xi[k],xi[k+1]; pa,pb=p2[:,n-k],p2[:,n-k-1]
        I1+=0.5*dt*x1*(xa+xb)
        qa=x1*x1+2.0*rho*sigma*x1*pa+sigma*sigma*pa*pa; qb=x1*x1+2.0*rho*sigma*x1*pb+sigma*sigma*pb*pb
        I2+=0.5*dt*(qa*xa+qb*xb)
    tj=np.arange(N+1)*T/N; rint=np.sum(tj[1:])/(N+1); G0=S0**(1.0/(N+1))
    Y0=np.log(S0)*(s*(N/(N+1))+w)+r*(s*rint+w*T)-0.5*I1+0.5*I2
    return G0**s*np.exp(Y0)
def _contour(R,Y,panels,nq):
    x,wg=np.polynomial.legendre.leggauss(nq); edges=np.linspace(-Y,Y,panels+1); ys=[]; ws=[]
    for a,b in zip(edges[:-1],edges[1:]): ys.append(0.5*(b-a)*x+0.5*(a+b)); ws.append(0.5*(b-a)*wg)
    return np.concatenate(ys), np.concatenate(ws)
def _chain(S0,K,T,r,nu0,kappa,theta,sigma,rho,lam,N,m,R,Y,panels,nq):
    mass=_mass(T,N,m); n=N*m; xi=_xi(nu0,kappa,theta,lam,T,n); y,wq=_contour(R,Y,panels,nq); z=R+1j*y
    one=np.array([1.0+0j]); p21=_ric(one,0.0,mass,kappa,sigma,rho,lam,T); h1=_H0(one,0.0,S0,r,T,N,mass,xi,p21,sigma,rho)[0]
    p2z=_ric(z,0.0,mass,kappa,sigma,rho,lam,T); hz=_H0(z,0.0,S0,r,T,N,mass,xi,p2z,sigma,rho)
    zeta=K**(1.0-z)/(2.0*np.pi*z*(z-1.0)); Emin=-(np.sum(hz*zeta*wq)).real; EG=h1.real
    price=np.exp(-r*T)*(EG-Emin); massT=N/(N+1)
    part1=((h1/S0)*massT+np.sum((hz/S0)*(z*massT)*zeta*wq)).real
    part2=((h1/S0)*rho*sigma*p21[0,-1]+np.sum((hz/S0)*(rho*sigma*p2z[:,-1])*zeta*wq)).real
    return dict(xiT=xi[-1],psi2=p21[0,-1].real,EG=EG,Emin=Emin,price=price,part1=part1,part2=part2,hedge=part1+part2)

def _oracle_vha_hedge(S0, K, N, sigma, rho, R, y, wq, h1, hz, psi2_1, psi2_z):
    S0=_pos(S0,"S0"); K=_pos(K,"K"); N=_posint(N,"N"); sigma=_pos(sigma,"sigma")
    rho=float(rho)
    if not np.isfinite(rho) or rho<-1.0 or rho>1.0: raise ValueError("bad rho")
    R=float(R)
    if not np.isfinite(R) or R<=0.0 or R>=1.0: raise ValueError("R must lie in (0,1)")
    y=_arr(y,"y"); wq=_arr(wq,"wq",y.size)
    h=np.asarray(h1,dtype=np.float64); H=np.asarray(hz,dtype=np.float64); q1=np.asarray(psi2_1,dtype=np.float64); qz=np.asarray(psi2_z,dtype=np.float64)
    if h.shape!=(2,1) or H.shape!=(2,y.size) or q1.ndim!=3 or q1.shape[:2]!=(2,1) or qz.shape!=(2,y.size,q1.shape[2]): raise ValueError("bad array shapes")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(H)) and np.all(np.isfinite(q1)) and np.all(np.isfinite(qz))): raise ValueError("arrays must be finite")
    z=R+1j*y; hzc=H[0]+1j*H[1]; h1c=h[0,0]+1j*h[1,0]
    p1T=q1[0,0,-1]+1j*q1[1,0,-1]; pzT=qz[0,:,-1]+1j*qz[1,:,-1]
    zeta=K**(1.0-z)/(2.0*np.pi*z*(z-1.0)); massT=N/(N+1)
    part1=((h1c/S0)*massT+np.sum((hzc/S0)*(z*massT)*zeta*wq)).real
    part2=((h1c/S0)*rho*sigma*p1T+np.sum((hzc/S0)*(rho*sigma*pzT)*zeta*wq)).real
    return np.array([part1+part2,part1],dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\n# near-pole contour: the smallest sampled |Im z| here is 0.0025, tighter than the design grid, so an implementation that discards or regularises the nodes beside the pole at z = R is caught\nS0,K,T,nu0,kappa,theta,sigma,rho,lam,N,m=100.0,105.0,1.0,0.03,2.5,0.06,0.5,-0.8,2.0,8,2\nmass=_oracle_vha_monitoring_mass(T,N,m); xi=_oracle_vha_forward_variance(nu0,kappa,theta,lam,T,N*m)\nRc=0.5; y=_oracle_vha_contour(Rc,1.0,16,8)[0]; wq=_oracle_vha_contour(Rc,1.0,16,8)[1]\np1=_oracle_vha_riccati_volterra(np.array([1.0]),np.array([0.0]),0.0,mass,kappa,sigma,rho,lam,T); h1=_oracle_vha_transform(np.array([1.0]),np.array([0.0]),0.0,S0,0.0,T,N,mass,xi,p1,sigma,rho)\npz=_oracle_vha_riccati_volterra(np.full(y.size,Rc),y,0.0,mass,kappa,sigma,rho,lam,T); hz=_oracle_vha_transform(np.full(y.size,Rc),y,0.0,S0,0.0,T,N,mass,xi,pz,sigma,rho)\n","call":"vha_hedge(S0,K,N,sigma,rho,Rc,y,wq,h1,hz,p1,pz)","gold_call":"_oracle_vha_hedge(S0,K,N,sigma,rho,Rc,y,wq,h1,hz,p1,pz)"},
        {"setup":"import numpy as np\nS0,K,T,nu0,kappa,theta,sigma,rho,lam,N,m=90.0,100.0,0.5,0.04,1.0,0.04,0.3,-0.7,1.0,4,2\nmass=_oracle_vha_monitoring_mass(T,N,m); xi=_oracle_vha_forward_variance(nu0,kappa,theta,lam,T,N*m)\nRc=0.25; y=_oracle_vha_contour(Rc,60.0,16,8)[0]; wq=_oracle_vha_contour(Rc,60.0,16,8)[1]\np1=_oracle_vha_riccati_volterra(np.array([1.0]),np.array([0.0]),0.0,mass,kappa,sigma,rho,lam,T); h1=_oracle_vha_transform(np.array([1.0]),np.array([0.0]),0.0,S0,0.0,T,N,mass,xi,p1,sigma,rho)\npz=_oracle_vha_riccati_volterra(np.full(y.size,Rc),y,0.0,mass,kappa,sigma,rho,lam,T); hz=_oracle_vha_transform(np.full(y.size,Rc),y,0.0,S0,0.0,T,N,mass,xi,pz,sigma,rho)\n","call":"vha_hedge(S0,K,N,sigma,rho,Rc,y,wq,h1,hz,p1,pz)","gold_call":"_oracle_vha_hedge(S0,K,N,sigma,rho,Rc,y,wq,h1,hz,p1,pz)"},
        {"setup":"import numpy as np\nS0,K,T,nu0,kappa,theta,sigma,rho,lam,N,m=120.0,110.0,2.0,0.05,3.0,0.02,0.6,-0.5,0.5,12,2\nmass=_oracle_vha_monitoring_mass(T,N,m); xi=_oracle_vha_forward_variance(nu0,kappa,theta,lam,T,N*m)\nRc=0.75; y=_oracle_vha_contour(Rc,40.0,16,8)[0]; wq=_oracle_vha_contour(Rc,40.0,16,8)[1]\np1=_oracle_vha_riccati_volterra(np.array([1.0]),np.array([0.0]),0.0,mass,kappa,sigma,rho,lam,T); h1=_oracle_vha_transform(np.array([1.0]),np.array([0.0]),0.0,S0,0.0,T,N,mass,xi,p1,sigma,rho)\npz=_oracle_vha_riccati_volterra(np.full(y.size,Rc),y,0.0,mass,kappa,sigma,rho,lam,T); hz=_oracle_vha_transform(np.full(y.size,Rc),y,0.0,S0,0.0,T,N,mass,xi,pz,sigma,rho)\n","call":"vha_hedge(S0,K,N,sigma,rho,Rc,y,wq,h1,hz,p1,pz)","gold_call":"_oracle_vha_hedge(S0,K,N,sigma,rho,Rc,y,wq,h1,hz,p1,pz)"},
        {"setup":"import numpy as np\n# boundary: zero correlation, the holding collapses to the held-fixing delta\nS0,K,T,nu0,kappa,theta,sigma,rho,lam,N,m=100.0,105.0,1.0,0.03,2.5,0.06,0.5,0.0,2.0,8,2\nmass=_oracle_vha_monitoring_mass(T,N,m); xi=_oracle_vha_forward_variance(nu0,kappa,theta,lam,T,N*m)\nRc=0.5; y=_oracle_vha_contour(Rc,60.0,16,8)[0]; wq=_oracle_vha_contour(Rc,60.0,16,8)[1]\np1=_oracle_vha_riccati_volterra(np.array([1.0]),np.array([0.0]),0.0,mass,kappa,sigma,rho,lam,T); h1=_oracle_vha_transform(np.array([1.0]),np.array([0.0]),0.0,S0,0.0,T,N,mass,xi,p1,sigma,rho)\npz=_oracle_vha_riccati_volterra(np.full(y.size,Rc),y,0.0,mass,kappa,sigma,rho,lam,T); hz=_oracle_vha_transform(np.full(y.size,Rc),y,0.0,S0,0.0,T,N,mass,xi,pz,sigma,rho)\n","call":"vha_hedge(S0,K,N,sigma,rho,Rc,y,wq,h1,hz,p1,pz)","gold_call":"_oracle_vha_hedge(S0,K,N,sigma,rho,Rc,y,wq,h1,hz,p1,pz)"},
        {"setup":"import numpy as np\n# invalid input: a correlation outside [-1, 1] must raise ValueError\ny=_oracle_vha_contour(0.5,60.0,8,6)[0]; wq=_oracle_vha_contour(0.5,60.0,8,6)[1]\nh1=np.array([[99.5],[0.0]]); hz=np.zeros((2,y.size)); p1=np.zeros((2,1,81)); pz=np.zeros((2,y.size,81))\ndef _probe():\n    try:\n        vha_hedge(100.0,105.0,8,0.5,1.5,0.5,y,wq,h1,hz,p1,pz)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe()","gold_call":"1"},
    ]
