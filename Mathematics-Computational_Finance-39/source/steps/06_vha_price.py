"""
Assembles the fixed-strike geometric Asian call price from the transform values on the contour through the source's representation of the payoff.

The call payoff on the geometric average splits into the average itself and a contour integral of its power functions; both pieces are expectations the transform provides.

Returns
-------
A float64 array of shape (3,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vha_price(K, T, r, R, y, wq, h1, hz):
    r"""K: positive float, strike. T: positive float. r: non-negative float, rate. R: float in $(0, 1)$,
    real part of the contour. y, wq: $(M,)$ arrays, the imaginary parts of the contour nodes and the
    weights from the contour step. h1: $(2, 1)$ array, the transform at the argument pair $(1, 0)$.
    hz: $(2, M)$ array, the transform at the argument pairs $(z, 0)$ for the nodes $z = R + iy$.

    Returns a numpy float64 array of shape $(3,)$: the time-zero price of the fixed-strike discretely
    monitored geometric Asian call with strike K, the expectation $E[G_T]$ of the geometric average,
    and the expectation $E[\min(G_T, K)]$, the last two undiscounted, through the source's contour
    representation of the payoff.

    Raises:
        ValueError: on non-finite or non-positive K or T, negative r, R outside $(0, 1)$, arrays that
            are not finite, or y, wq and hz of inconsistent length.
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

def _oracle_vha_price(K, T, r, R, y, wq, h1, hz):
    K=_pos(K,"K"); T=_pos(T,"T"); r=_nonneg(r,"r"); y=_arr(y,"y"); wq=_arr(wq,"wq",y.size)
    R=float(R)
    if not np.isfinite(R) or R<=0.0 or R>=1.0: raise ValueError("R must lie in (0,1)")
    h=np.asarray(h1,dtype=np.float64); H=np.asarray(hz,dtype=np.float64)
    if h.shape!=(2,1) or H.shape!=(2,y.size) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(H)): raise ValueError("bad transform arrays")
    z=R+1j*y; hzc=H[0]+1j*H[1]
    zeta=K**(1.0-z)/(2.0*np.pi*z*(z-1.0)); Emin=-(np.sum(hzc*zeta*wq)).real; EG=float(h[0,0])
    return np.array([np.exp(-r*T)*(EG-Emin),EG,Emin],dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nS0,K,T,r,nu0,kappa,theta,sigma,rho,lam,N,m=100.0,105.0,1.0,0.0,0.03,2.5,0.06,0.5,-0.8,2.0,8,50\nmass=_oracle_vha_monitoring_mass(T,N,m); xi=_oracle_vha_forward_variance(nu0,kappa,theta,lam,T,N*m)\nRc=0.5; y=_oracle_vha_contour(Rc,120.0,120,24)[0]; wq=_oracle_vha_contour(Rc,120.0,120,24)[1]\np1=_oracle_vha_riccati_volterra(np.array([1.0]),np.array([0.0]),0.0,mass,kappa,sigma,rho,lam,T); h1=_oracle_vha_transform(np.array([1.0]),np.array([0.0]),0.0,S0,r,T,N,mass,xi,p1,sigma,rho)\nhz=_oracle_vha_transform(np.full(y.size,Rc),y,0.0,S0,r,T,N,mass,xi,_oracle_vha_riccati_volterra(np.full(y.size,Rc),y,0.0,mass,kappa,sigma,rho,lam,T),sigma,rho)\n","call":"vha_price(K,T,r,Rc,y,wq,h1,hz)","gold_call":"_oracle_vha_price(K,T,r,Rc,y,wq,h1,hz)"},
        {"setup":"import numpy as np\nS0,K,T,r,nu0,kappa,theta,sigma,rho,lam,N,m=90.0,100.0,0.5,0.03,0.04,1.0,0.04,0.3,-0.7,1.0,4,10\nmass=_oracle_vha_monitoring_mass(T,N,m); xi=_oracle_vha_forward_variance(nu0,kappa,theta,lam,T,N*m)\nRc=0.25; y=_oracle_vha_contour(Rc,60.0,30,16)[0]; wq=_oracle_vha_contour(Rc,60.0,30,16)[1]\np1=_oracle_vha_riccati_volterra(np.array([1.0]),np.array([0.0]),0.0,mass,kappa,sigma,rho,lam,T); h1=_oracle_vha_transform(np.array([1.0]),np.array([0.0]),0.0,S0,r,T,N,mass,xi,p1,sigma,rho)\nhz=_oracle_vha_transform(np.full(y.size,Rc),y,0.0,S0,r,T,N,mass,xi,_oracle_vha_riccati_volterra(np.full(y.size,Rc),y,0.0,mass,kappa,sigma,rho,lam,T),sigma,rho)\n","call":"vha_price(K,T,r,Rc,y,wq,h1,hz)","gold_call":"_oracle_vha_price(K,T,r,Rc,y,wq,h1,hz)"},
        {"setup":"import numpy as np\nS0,K,T,r,nu0,kappa,theta,sigma,rho,lam,N,m=120.0,110.0,2.0,0.01,0.05,3.0,0.02,0.6,-0.5,0.5,12,5\nmass=_oracle_vha_monitoring_mass(T,N,m); xi=_oracle_vha_forward_variance(nu0,kappa,theta,lam,T,N*m)\nRc=0.75; y=_oracle_vha_contour(Rc,40.0,20,16)[0]; wq=_oracle_vha_contour(Rc,40.0,20,16)[1]\np1=_oracle_vha_riccati_volterra(np.array([1.0]),np.array([0.0]),0.0,mass,kappa,sigma,rho,lam,T); h1=_oracle_vha_transform(np.array([1.0]),np.array([0.0]),0.0,S0,r,T,N,mass,xi,p1,sigma,rho)\nhz=_oracle_vha_transform(np.full(y.size,Rc),y,0.0,S0,r,T,N,mass,xi,_oracle_vha_riccati_volterra(np.full(y.size,Rc),y,0.0,mass,kappa,sigma,rho,lam,T),sigma,rho)\n","call":"vha_price(K,T,r,Rc,y,wq,h1,hz)","gold_call":"_oracle_vha_price(K,T,r,Rc,y,wq,h1,hz)"},
        {"setup":"import numpy as np\n# boundary: a vanishing transform on the contour leaves the price equal to the expectation of the average\ny=_oracle_vha_contour(0.5,60.0,30,16)[0]; wq=_oracle_vha_contour(0.5,60.0,30,16)[1]\nh1=np.array([[99.5],[0.0]]); hz=np.zeros((2,y.size))\n","call":"vha_price(105.0,1.0,0.0,0.5,y,wq,h1,hz)","gold_call":"_oracle_vha_price(105.0,1.0,0.0,0.5,y,wq,h1,hz)"},
        {"setup":"import numpy as np\n# invalid input: weights of the wrong length must raise ValueError\ny=_oracle_vha_contour(0.5,60.0,30,16)[0]; wq=_oracle_vha_contour(0.5,60.0,30,16)[1][:-1]\nh1=np.array([[99.5],[0.0]]); hz=np.zeros((2,y.size))\ndef _probe():\n    try:\n        vha_price(105.0,1.0,0.0,0.5,y,wq,h1,hz)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe()","gold_call":"1"},
    ]
