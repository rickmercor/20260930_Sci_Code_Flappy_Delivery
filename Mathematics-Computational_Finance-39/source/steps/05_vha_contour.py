"""
Builds the composite Gauss-Legendre nodes and weights on the vertical contour used by the payoff representation.

The payoff is represented as a contour integral of power functions of the average against a complex measure on a vertical line inside the unit strip. The integrand has poles at distance R and 1 - R from the line, so a composite rule with many panels is needed near the real axis.

Returns
-------
A float64 array of shape (2, panels*nq).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vha_contour(R, Y, panels, nq):
    r"""R: float in $(0, 1)$, real part of the contour. Y: positive float, half-length of the imaginary
    range. panels, nq: positive integers, number of equal panels on $[-Y, Y]$ and Gauss-Legendre nodes
    per panel.

    Returns a numpy float64 array of shape $(2, \mathrm{panels}\cdot nq)$: row 0 the imaginary parts
    $y$ of the nodes $z = R + iy$ on the vertical line, row 1 the quadrature weights for integration
    against the Lebesgue measure of the imaginary part, both in increasing order of $y$.

    Raises:
        ValueError: on R outside $(0, 1)$, non-finite or non-positive Y, or non-integral or
            non-positive panels or nq.
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

def _oracle_vha_contour(R, Y, panels, nq):
    R=float(R)
    if not np.isfinite(R) or R<=0.0 or R>=1.0: raise ValueError("R must lie in (0,1)")
    Y=_pos(Y,"Y"); panels=_posint(panels,"panels"); nq=_posint(nq,"nq")
    y,w=_contour(R,Y,panels,nq)
    return np.stack([y,w]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\n","call":"vha_contour(0.5,120.0,120,32)","gold_call":"_oracle_vha_contour(0.5,120.0,120,32)"},
        {"setup":"import numpy as np\n","call":"vha_contour(0.25,60.0,30,16)","gold_call":"_oracle_vha_contour(0.25,60.0,30,16)"},
        {"setup":"import numpy as np\n","call":"vha_contour(0.75,40.0,10,8)","gold_call":"_oracle_vha_contour(0.75,40.0,10,8)"},
        {"setup":"import numpy as np\n# boundary: a single panel with two nodes\n","call":"vha_contour(0.5,1.0,1,2)","gold_call":"_oracle_vha_contour(0.5,1.0,1,2)"},
        {"setup":"import numpy as np\n# invalid input: the real part must lie strictly inside (0, 1)\ndef _probe():\n    try:\n        vha_contour(1.0,120.0,120,32)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe()","gold_call":"1"},
    ]
