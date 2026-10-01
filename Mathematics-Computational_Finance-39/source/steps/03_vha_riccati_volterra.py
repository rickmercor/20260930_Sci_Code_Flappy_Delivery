"""
Solves the Riccati-Volterra equation of the model for a piecewise-constant, measure-valued first component and a vector of complex transform arguments.

The affine structure of the model reduces the joint transform of the terminal log-price and the log-geometric average to a Riccati equation of Volterra type whose first component is driven by the monitoring measure; with a non-exponential kernel the equation has no ordinary-differential reduction and is solved on the grid.

Returns
-------
A float64 array of shape (2, M, n+1).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vha_riccati_volterra(s_re, s_im, w, mass, kappa, sigma, rho, lam, T):
    r"""s_re, s_im: float arrays of equal length $M$, real and imaginary parts of the transform
    arguments $s$. w: real float, the second transform argument. mass: $(n,)$ array, the cell values
    from the monitoring step. kappa, sigma: positive floats; rho: float in $[-1, 1]$; lam: positive
    float, kernel parameter; T: positive float.

    Returns a numpy float64 array of shape $(2, M, n + 1)$: real and imaginary parts of the solution
    $\psi_2$ of the source's Riccati-Volterra equation on the nodes $t_i = iT/n$, for the coefficient
    $\psi_1 = s\,\mathrm{mass} + w$ on each cell, with the kernel $K(t) = (1 - e^{-\lambda t})/(\lambda t)$.
    The convolution is the product-trapezoid rule on the grid with the kernel at node differences and
    the cell's own coefficient at both ends of each cell; the value at a new node is obtained from a
    predictor that repeats the left-end integrand at the right end, followed by one corrector pass.

    Raises:
        ValueError: on s_re and s_im of different or zero length, non-finite entries, a mass array that
            is empty or not finite, non-finite or non-positive kappa, sigma, lam or T, or rho outside
            $[-1, 1]$.
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

def _oracle_vha_riccati_volterra(s_re, s_im, w, mass, kappa, sigma, rho, lam, T):
    sr=_arr(s_re,"s_re"); si=_arr(s_im,"s_im")
    if sr.size!=si.size: raise ValueError("s_re and s_im must have the same length")
    w=float(w)
    if not np.isfinite(w): raise ValueError("bad w")
    mass=_arr(mass,"mass"); kappa=_pos(kappa,"kappa"); sigma=_pos(sigma,"sigma"); lam=_pos(lam,"lam"); T=_pos(T,"T")
    rho=float(rho)
    if not np.isfinite(rho) or rho<-1.0 or rho>1.0: raise ValueError("bad rho")
    p2=_ric(sr+1j*si,w,mass,kappa,sigma,rho,lam,T)
    return np.stack([p2.real,p2.imag]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nmass=_oracle_vha_monitoring_mass(1.0,8,50)\nsr=np.array([1.0,0.5,0.5]); si=np.array([0.0,3.0,-40.0])\n","call":"vha_riccati_volterra(sr,si,0.0,mass,2.5,0.5,-0.8,2.0,1.0)","gold_call":"_oracle_vha_riccati_volterra(sr,si,0.0,mass,2.5,0.5,-0.8,2.0,1.0)"},
        {"setup":"import numpy as np\nmass=_oracle_vha_monitoring_mass(0.5,4,10)\nsr=np.array([0.25,0.75]); si=np.array([10.0,-2.0])\n","call":"vha_riccati_volterra(sr,si,0.0,mass,1.0,0.3,-0.7,1.0,0.5)","gold_call":"_oracle_vha_riccati_volterra(sr,si,0.0,mass,1.0,0.3,-0.7,1.0,0.5)"},
        {"setup":"import numpy as np\n# the floating-strike argument pair (z, 1-z)\nmass=_oracle_vha_monitoring_mass(2.0,12,5)\nsr=np.array([0.5,0.5]); si=np.array([5.0,-5.0])\n","call":"vha_riccati_volterra(sr,si,0.5,mass,3.0,0.6,-0.5,0.5,2.0)","gold_call":"_oracle_vha_riccati_volterra(sr,si,0.5,mass,3.0,0.6,-0.5,0.5,2.0)"},
        {"setup":"import numpy as np\n# boundary: a zero argument gives the zero solution\nmass=_oracle_vha_monitoring_mass(1.0,8,10)\nsr=np.array([0.0]); si=np.array([0.0])\n","call":"vha_riccati_volterra(sr,si,0.0,mass,2.5,0.5,-0.8,2.0,1.0)","gold_call":"_oracle_vha_riccati_volterra(sr,si,0.0,mass,2.5,0.5,-0.8,2.0,1.0)"},
        {"setup":"import numpy as np\n# invalid input: a correlation outside [-1, 1] must raise ValueError\nmass=_oracle_vha_monitoring_mass(1.0,8,10)\ndef _probe():\n    try:\n        vha_riccati_volterra(np.array([1.0]),np.array([0.0]),0.0,mass,2.5,0.5,-1.5,2.0,1.0)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe()","gold_call":"1"},
    ]
