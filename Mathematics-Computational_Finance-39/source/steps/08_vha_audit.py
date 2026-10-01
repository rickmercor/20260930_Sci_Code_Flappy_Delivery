"""
Runs the whole chain from the monitoring measure to the variance-optimal holding and reports every intermediate the construction depends on.

Reporting the forward variance, the Riccati solution, the two expectations, the price and the two holdings side by side shows how much of the final holding rides on each of the source's choices.

Returns
-------
A float64 array of shape (7,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vha_audit(S0, K, T, nu0, kappa, theta, sigma, rho, lam, N, m, R, Y, panels, nq):
    r"""Parameters as in the hedging step, at zero rate.

    The orchestrator. It must call the earlier functions rather than reimplementing them.
    Returns a numpy float64 array of shape $(7,)$: the forward variance at maturity, the real part
    of the Riccati-Volterra solution at maturity for the argument pair $(1, 0)$, the expectation of
    the geometric average, the expectation of the minimum of the average and the strike, the price of
    the call, the held-fixing delta, and the variance-optimal initial holding.

    Raises:
        ValueError: whenever any of the functions it calls would raise.
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

def _oracle_vha_audit(S0, K, T, nu0, kappa, theta, sigma, rho, lam, N, m, R, Y, panels, nq):
    S0=_pos(S0,"S0"); K=_pos(K,"K"); T=_pos(T,"T"); nu0=_nonneg(nu0,"nu0"); kappa=_pos(kappa,"kappa"); theta=_pos(theta,"theta"); sigma=_pos(sigma,"sigma"); lam=_pos(lam,"lam")
    N=_posint(N,"N"); m=_posint(m,"m")
    R=float(R)
    if not np.isfinite(R) or R<=0.0 or R>=1.0: raise ValueError("R must lie in (0,1)")
    mass=_oracle_vha_monitoring_mass(T,N,m)
    xi=_oracle_vha_forward_variance(nu0,kappa,theta,lam,T,N*m)
    c=_oracle_vha_contour(R,Y,panels,nq); y=c[0]; wq=c[1]
    one_re=np.array([1.0]); one_im=np.array([0.0])
    p1=_oracle_vha_riccati_volterra(one_re,one_im,0.0,mass,kappa,sigma,rho,lam,T)
    pz=_oracle_vha_riccati_volterra(np.full(y.size,R),y,0.0,mass,kappa,sigma,rho,lam,T)
    h1=_oracle_vha_transform(one_re,one_im,0.0,S0,0.0,T,N,mass,xi,p1,sigma,rho)
    hz=_oracle_vha_transform(np.full(y.size,R),y,0.0,S0,0.0,T,N,mass,xi,pz,sigma,rho)
    pr=_oracle_vha_price(K,T,0.0,R,y,wq,h1,hz)
    hd=_oracle_vha_hedge(S0,K,N,sigma,rho,R,y,wq,h1,hz,p1,pz)
    return np.array([xi[-1],p1[0,0,-1],pr[1],pr[2],pr[0],hd[1],hd[0]],dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nP=(100.0,105.0,1.0,0.03,2.5,0.06,0.5,-0.8,2.0,8,50,0.5,120.0,120,32)\n","call":"vha_audit(*P)","gold_call":"_oracle_vha_audit(*P)"},
        {"setup":"import numpy as np\nP=(90.0,100.0,0.5,0.04,1.0,0.04,0.3,-0.7,1.0,4,10,0.25,60.0,30,16)\n","call":"vha_audit(*P)","gold_call":"_oracle_vha_audit(*P)"},
        {"setup":"import numpy as np\nP=(120.0,110.0,2.0,0.05,3.0,0.02,0.6,-0.5,0.5,12,5,0.75,40.0,20,16)\n","call":"vha_audit(*P)","gold_call":"_oracle_vha_audit(*P)"},
        {"setup":"import numpy as np\n# boundary: zero correlation\nP=(100.0,105.0,1.0,0.03,2.5,0.06,0.5,0.0,2.0,8,10,0.5,60.0,30,16)\n","call":"vha_audit(*P)","gold_call":"_oracle_vha_audit(*P)"},
        {"setup":"import numpy as np\n# invalid input: zero monitoring intervals must raise ValueError\ndef _probe():\n    try:\n        vha_audit(100.0,105.0,1.0,0.03,2.5,0.06,0.5,-0.8,2.0,0,10,0.5,60.0,30,16)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe()","gold_call":"1"},
    ]
