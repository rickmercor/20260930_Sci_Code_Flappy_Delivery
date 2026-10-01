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

def _geometry(points, weights):
    points=np.asarray(points,dtype=float);weights=np.asarray(weights,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or weights.shape!=(len(points),):
        raise ValueError('Expected points (n,2) and weights (n,), n >= 3.')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)) or np.any(weights<0):
        raise ValueError('Coordinates and nonnegative quadrature weights must be finite.')
    d=np.sum((points[:,None,:]-points[None,:,:])**2,axis=2)
    if np.any(d[np.triu_indices(len(points),1)]<=0):raise ValueError('Nodes must be distinct.')
    return points,weights,d


def emission_kernels(points: "np.ndarray", weights: "np.ndarray", mu2: float, nf: int = 3, nc: int = 3) -> "np.ndarray":
    """Weighted K_LO, L_LO and one-emission K_NLO; Eqs. 7, 13 and 21."""
    import numpy as np
    p,w,d=_geometry(points,weights);n=len(p)
    if not np.isfinite(mu2) or mu2<=0 or nc<2 or nf<0 or nc!=int(nc) or nf!=int(nf):raise ValueError('Invalid scale or color multiplicity.')
    out=np.zeros((3,n,n,n));constant=67/18-np.pi**2/6-5*nf/(9*nc)
    for i in range(n):
      for j in range(n):
       if i==j:continue
       for k in range(n):
        if k in (i,j):continue
        r,a,b=d[i,j],d[i,k],d[k,j];c=w[k]*r/(a*b)
        out[:,i,j,k]=c,c*np.log(mu2*np.sqrt(a*b)),.5*c*(constant-np.log(a/r)*np.log(b/r))
    return out

import numpy as np

def _geometry(points, weights):
    points=np.asarray(points,dtype=float);weights=np.asarray(weights,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or weights.shape!=(len(points),):
        raise ValueError('Expected points (n,2) and weights (n,), n >= 3.')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)) or np.any(weights<0):
        raise ValueError('Coordinates and nonnegative quadrature weights must be finite.')
    d=np.sum((points[:,None,:]-points[None,:,:])**2,axis=2)
    if np.any(d[np.triu_indices(len(points),1)]<=0):raise ValueError('Nodes must be distinct.')
    return points,weights,d


def pair_kernels(points: "np.ndarray", weights: "np.ndarray", nf: int = 3, nc: int = 3) -> "np.ndarray":
    """Weighted connected gluon and fermion kernels; Eqs. 20, 22 and 23."""
    import numpy as np
    p,w,d=_geometry(points,weights);n=len(p)
    if nc<2 or nf<0 or nc!=int(nc) or nf!=int(nf):raise ValueError('Invalid color multiplicity.')
    out=np.zeros((2,n,n,n,n))
    for i in range(n):
     for j in range(n):
      for k in range(n):
       for l in range(n):
        if len({i,j,k,l})<4:continue
        r,a,b,c,e,z=d[i,j],d[i,k],d[k,j],d[i,l],d[l,j],d[k,l]
        A=a*e;B=c*b;t=(A-B)/B
        if abs(t)<1e-5:
          divided=(1-t/2+t*t/3-t**3/4+t**4/5)/B
        else:divided=(np.log(A)-np.log(B))/(A-B)
        lg=np.log1p(t) if abs(t)<.5 else np.log(A)-np.log(B)
        kg=-2/z**2+((A+B-4*r*z)/z**2+r*r/A)*divided+r/(A*z)*lg
        kf=2/z**2-(A+B-r*z)/z**2*divided
        out[:,i,j,k,l]=.5*w[k]*w[l]*kg,nf/(2*nc)*w[k]*w[l]*kf
    return out

import numpy as np

def _state(s,one):
    import numpy as np
    s=np.asarray(s);one=np.asarray(one)
    if s.ndim!=2:raise ValueError('State must be a matrix.')
    n=len(s)
    if s.shape!=(n,n) or one.shape!=(3,n,n,n) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(one)):
        raise ValueError('Incompatible state or emission kernels.')
    return s,one


def dipole_fields(s: "np.ndarray", one: "np.ndarray") -> "np.ndarray":
    """Leibniz-projected LO and logarithmic rotation fields, Eqs. 9 and 14."""
    import numpy as np
    s,one=_state(s,one)
    return np.einsum('aijk,ik,kj->aij',one[:2],s,s)-s[None,:,:]*one[:2].sum(axis=3)

import numpy as np

def _state(s,one):
    import numpy as np
    s=np.asarray(s);one=np.asarray(one)
    if s.ndim!=2:raise ValueError('State must be a matrix.')
    n=len(s)
    if s.shape!=(n,n) or one.shape!=(3,n,n,n) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(one)):
        raise ValueError('Incompatible state or emission kernels.')
    return s,one


def nlo_action(s: "np.ndarray", one: "np.ndarray", two: "np.ndarray") -> "np.ndarray":
    """Project the full fixed-coupling unrotated NLO kernel, Eq. 20."""
    import numpy as np
    s,one=_state(s,one);two=np.asarray(two);n=len(s)
    if two.shape!=(2,n,n,n,n) or not np.all(np.isfinite(two)):raise ValueError('Invalid pair kernels.')
    f=np.einsum('ijk,ik,kj->ij',one[2],s,s)-s*one[2].sum(axis=2)
    f+=np.einsum('ijkl,ik,kl,lj->ij',two[0],s,s,s)-np.einsum('ijkl,ik,kj->ij',two[0],s,s)
    f+=np.einsum('ijkl,kj,il->ij',two[1],s,s)-np.einsum('ijkl,kj,ik->ij',two[1],s,s)
    return f

import numpy as np

def _state(s,one):
    import numpy as np
    s=np.asarray(s);one=np.asarray(one)
    if s.ndim!=2:raise ValueError('State must be a matrix.')
    n=len(s)
    if s.shape!=(n,n) or one.shape!=(3,n,n,n) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(one)):
        raise ValueError('Incompatible state or emission kernels.')
    return s,one


def scheme_commutator(s: "np.ndarray", one: "np.ndarray") -> "np.ndarray":
    """Project [K_LO,L_LO] after composition in dipole-chain space, Eqs. 27-30."""
    import numpy as np
    s,one=_state(s,one);f,g=dipole_fields(s,one)
    def _d(kernel,v):
        return np.einsum('ijk,ik,kj->ij',kernel,v,s)+np.einsum('ijk,ik,kj->ij',kernel,s,v)-v*kernel.sum(axis=2)
    return _d(one[0],g)-_d(one[1],f)

import numpy as np

def _state(s,one):
    import numpy as np
    s=np.asarray(s);one=np.asarray(one)
    if s.ndim!=2:raise ValueError('State must be a matrix.')
    n=len(s)
    if s.shape!=(n,n) or one.shape!=(3,n,n,n) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(one)):
        raise ValueError('Incompatible state or emission kernels.')
    return s,one


def composite_matching(s: "np.ndarray", one: "np.ndarray", alpha: float, direction: int) -> "np.ndarray":
    """First-order dipole rotation or dual hard-factor matching, Eqs. 14-15."""
    import numpy as np
    s,one=_state(s,one)
    if direction not in (-1,1) or not np.isfinite(alpha) or alpha<0:raise ValueError('Invalid matching parameters.')
    return s+direction*alpha*dipole_fields(s,one)[1]

import numpy as np

def _state(s,one):
    import numpy as np
    s=np.asarray(s);one=np.asarray(one)
    if s.ndim!=2:raise ValueError('State must be a matrix.')
    n=len(s)
    if s.shape!=(n,n) or one.shape!=(3,n,n,n) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(one)):
        raise ValueError('Incompatible state or emission kernels.')
    return s,one


def evolve_rotated_dipole(s: "np.ndarray", one: "np.ndarray", two: "np.ndarray", alpha: float, rapidity: float) -> "np.ndarray":
    """Finite-regulator NLO evolution with the unintegrated operator commutator."""
    import numpy as np
    from scipy.integrate import solve_ivp
    s,one=_state(s,one);n=len(s)
    if alpha<0 or rapidity<0 or not np.isfinite(alpha+rapidity):raise ValueError('Invalid evolution parameters.')
    if alpha==0 or rapidity==0:return s.copy()
    def _rhs(y,flat):
        state=flat.reshape(n,n)
        return (alpha*dipole_fields(state,one)[0]+alpha**2*(nlo_action(state,one,two)+scheme_commutator(state,one))).ravel()
    sol=solve_ivp(_rhs,[0,rapidity],s.ravel(),method='DOP853',rtol=2e-11,atol=2e-13)
    if not sol.success or not np.all(np.isfinite(sol.y[:,-1])):raise ValueError('Evolution failed.')
    return sol.y[:,-1].reshape(n,n)

import numpy as np

def _geometry(points, weights):
    points=np.asarray(points,dtype=float);weights=np.asarray(weights,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or weights.shape!=(len(points),):
        raise ValueError('Expected points (n,2) and weights (n,), n >= 3.')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)) or np.any(weights<0):
        raise ValueError('Coordinates and nonnegative quadrature weights must be finite.')
    d=np.sum((points[:,None,:]-points[None,:,:])**2,axis=2)
    if np.any(d[np.triu_indices(len(points),1)]<=0):raise ValueError('Nodes must be distinct.')
    return points,weights,d


def infer_rotated_cgc(points: "np.ndarray", weights: "np.ndarray", measurements: "np.ndarray", prediction: "np.ndarray", mu2: float = 1.3, nf: int = 3, nc: int = 3) -> "np.ndarray":
    """Fit the MV saturation scale/coupling and propagate the absolute observation covariance."""
    import numpy as np
    from scipy.optimize import least_squares
    p,w,d=_geometry(points,weights);data=np.asarray(measurements,dtype=float);pred=np.asarray(prediction,dtype=float)
    if data.ndim!=2 or data.shape[1]!=5 or len(data)<3 or pred.shape!=(3,):raise ValueError('Invalid measurement table or prediction.')
    if not np.all(np.isfinite(data)) or not np.all(np.isfinite(pred)) or np.any(data[:,4]<=0):raise ValueError('Invalid observations.')
    queries=np.vstack([data[:,:3],pred])
    if np.any(queries[:,0]<0) or np.any(queries[:,1]<=0) or np.any(abs(queries[:,2])>=1):raise ValueError('Invalid probe parameters.')
    one=emission_kernels(p,w,mu2,nf,nc);two=pair_kernels(p,w,nf,nc)
    dx=p[:,None,0]-p[None,:,0];dy=p[:,None,1]-p[None,:,1]
    probes=[]
    for Y,b,eps in queries:
        h=np.triu(np.exp(-d/(2*b*b))*(d+eps*(dx*dx-dy*dy)),1)
        probes.append(h/h.sum())
    radius=np.sqrt(d);safe=np.where(radius==0,1,radius)
    mv_exponent=-.25*d*np.log(1/(.2*safe)+np.e)
    def _model(theta,indices):
        Q0,alpha=theta
        physical=np.exp(Q0*Q0*mv_exponent)
        initial=composite_matching(physical,one,alpha,-1)
        values=[]
        for i in indices:
            evolved=evolve_rotated_dipole(initial,one,two,alpha,queries[i,0])
            matched=composite_matching(evolved,one,alpha,1)
            values.append(np.sum(probes[i]*(1-matched)))
        return np.array(values)
    ids=np.arange(len(data));future=[len(data)]
    def _residual(theta):return (_model(theta,ids)-data[:,3])/data[:,4]
    fits=[]
    for start in ([.72,.12],[.98,.21],[1.25,.29]):
        fit=least_squares(_residual,start,bounds=([.55,.08],[1.4,.32]),xtol=2e-12,ftol=2e-12,gtol=2e-8,diff_step=2e-5,max_nfev=120)
        fits.append(fit)
    fit=min(fits,key=lambda r:(np.dot(r.fun,r.fun),r.x[0],r.x[1]))
    theta=fit.x
    if not fit.success or np.any(theta<np.array([.55001,.08001])) or np.any(theta>np.array([1.39999,.31999])):raise ValueError('An interior identifiable fit is required.')
    jac=np.zeros((len(data),2));g=np.zeros(2)
    # Symmetric five-point derivatives checked independently against complex-step tangents.
    for j in range(2):
        step=np.zeros(2);step[j]=2e-4
        jac[:,j]=(-_residual(theta+2*step)+8*_residual(theta+step)-8*_residual(theta-step)+_residual(theta-2*step))/(12*step[j])
        g[j]=(-_model(theta+2*step,future)[0]+8*_model(theta+step,future)[0]-8*_model(theta-step,future)[0]+_model(theta-2*step,future)[0])/(12*step[j])
    information=jac.T@jac
    if np.linalg.eigvalsh(information)[0]<=1e-9:raise ValueError('The fit is not identifiable.')
    covariance=np.linalg.inv(information)
    rho=covariance[0,1]/np.sqrt(covariance[0,0]*covariance[1,1])
    error=np.sqrt(g@covariance@g)
    return np.array([theta[0],theta[1],np.dot(fit.fun,fit.fun),_model(theta,future)[0],rho,error])
SCICODE_GOLD_EOF
