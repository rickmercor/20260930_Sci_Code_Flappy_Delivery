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
from scipy.linalg import eigh

def seesaw_basis(md: 'np.ndarray', mr: 'np.ndarray') -> 'np.ndarray':
    md,mr=np.asarray(md,complex),np.asarray(mr,complex);n=md.shape[0]
    matrix=np.block([[np.zeros((n,n)),md],[md.T,mr]])
    u,_,_=np.linalg.svd(matrix)
    diag=np.diag(u.conj().T@matrix@u.conj())
    u=u*np.exp(.5j*np.angle(diag))[None,:]
    masses=np.abs(diag);order=np.argsort(masses);masses=masses[order];u=u[:,order]
    for j in range(2*n):
        pivot=np.argmax(np.abs(u[:,j]))
        pivot_value=u[pivot,j]
        sign=pivot_value.real if abs(pivot_value.real)>1e-12*abs(pivot_value) else pivot_value.imag
        if sign<0:u[:,j]*=-1
    return np.vstack([masses,u])

import numpy as np

def pv_laurent(queries: 'np.ndarray', scale: float) -> 'np.ndarray':
    out=[]
    for p2,a,b in np.asarray(queries,float):
        aa,bb=a*a,b*b
        if p2>0:
            linear=aa-bb-p2
            disc=complex(linear*linear-4*p2*bb)
            q=-.5*(linear+(1 if linear>=0 else -1)*np.sqrt(disc))
            roots=[q/p2,bb/q] if q!=0 else [0j,0j]
            def _endpoint(z):
                return 0j if abs(z)==0 else z*np.log(z+0j)
            def _root_integral(r):
                if abs(r)>2:
                    if r.imag==0:r=float(r.real)
                    return np.log(abs(r))+((1-r)*np.log1p(-1/r)).real-1
                return (_endpoint(1-r)-_endpoint(-r)-1).real
            value=2*np.log(scale)-np.log(p2)
            value-=sum(_root_integral(r) for r in roots)
            out.append([1.,value]);continue
        if aa==bb:
            val=2*np.log(scale)-np.log(aa)
        elif min(aa,bb)==0:
            val=1+2*np.log(scale)-np.log(max(aa,bb))
        else:
            large,small=max(aa,bb),min(aa,bb);t=(large-small)/small
            val=1+2*np.log(scale)-np.log(small)-(1+t)*np.log1p(t)/t
        out.append([1.,val])
    return np.array(out)

import numpy as np
from scipy.linalg import eigh

def charged_gauge(b: 'np.ndarray', m: 'np.ndarray', ml: 'np.ndarray', mw: float, alpha: float, sw2: float, xi_w: float, loops: 'np.ndarray') -> 'np.ndarray':
    b,m,ml,loops=np.asarray(b),np.asarray(m),np.asarray(ml),np.asarray(loops)
    out=np.zeros(b.shape+(2,),complex)
    for a in range(len(ml)):
        for beta in range(len(ml)):
            if a==beta:continue
            for k,z in enumerate(m):
                val=(z*z-ml[a]**2+xi_w*mw**2)*loops[a,k]-(z*z-ml[beta]**2+xi_w*mw**2)*loops[beta,k]
                out[a]+=b[a,k]*b[beta,k].conjugate()*b[beta,:,None]*val
    return alpha/(32*np.pi*sw2*mw**2)*out

import numpy as np
from scipy.linalg import eigh

def neutrino_w_gauge(b: 'np.ndarray', m: 'np.ndarray', ml: 'np.ndarray', mw: float, alpha: float, sw2: float, xi_w: float, loops: 'np.ndarray') -> 'np.ndarray':
    b,m,ml,loops=np.asarray(b),np.asarray(m),np.asarray(ml),np.asarray(loops)
    out=np.zeros(b.shape+(2,),complex)
    for i,x in enumerate(m):
        for j,y in enumerate(m):
            if i==j:continue
            for r,l in enumerate(ml):
                val=(y*y-xi_w*mw**2-l*l)*loops[j,r]-(x*x-xi_w*mw**2-l*l)*loops[i,r]
                out[:,i]+=b[:,j,None]*b[r,j].conjugate()*b[r,i]*val
    return alpha/(32*np.pi*sw2*mw**2)*out

import numpy as np
from scipy.linalg import eigh

def neutrino_z_gauge(b: 'np.ndarray', m: 'np.ndarray', mw: float, mz: float, alpha: float, sw2: float, xi_z: float, loops: 'np.ndarray') -> 'np.ndarray':
    b,m,loops=np.asarray(b),np.asarray(m),np.asarray(loops);c=b.conj().T@b
    out=np.zeros(b.shape+(2,),complex)
    for i,x in enumerate(m):
        for j,y in enumerate(m):
            if i==j:continue
            for k,z in enumerate(m):
                first=c[j,k]*c[k,i]*y*(y*y-z*z-xi_z*mz*mz)
                first-=c[j,k].conjugate()*c[k,i]*z*(y*y-z*z+xi_z*mz*mz)
                second=c[j,k]*c[k,i]*x*(x*x-z*z-xi_z*mz*mz)
                second-=c[j,k]*c[k,i].conjugate()*z*(x*x-z*z+xi_z*mz*mz)
                val=first/(2*y)*loops[j,k]-second/(2*x)*loops[i,k]
                out[:,i]+=b[:,j,None]*val
    return alpha/(32*np.pi*sw2*mw**2)*out

import numpy as np
from scipy.linalg import eigh

def majorana_diagonal_gauge(b: 'np.ndarray', m: 'np.ndarray', mw: float, mz: float, alpha: float, sw2: float, xi_z: float, loops: 'np.ndarray') -> 'np.ndarray':
    b,m,loops=np.asarray(b),np.asarray(m),np.asarray(loops);c=b.conj().T@b
    out=np.zeros(b.shape+(2,),complex)
    for i,x in enumerate(m):
        for k,z in enumerate(m):
            val=z/x*(x*x-z*z+xi_z*mz*mz)*(c[k,i]**2-c[i,k]**2)*loops[i,k]
            out[:,i]+=b[:,i,None]*val
    return -alpha/(64*np.pi*sw2*mw**2)*out

import numpy as np
from scipy.linalg import eigh

def restored_uv(b: 'np.ndarray', m: 'np.ndarray', mw: float, alpha: float, sw2: float) -> 'np.ndarray':
    b,m=np.asarray(b),np.asarray(m);c=b.conj().T@b;n=len(m)
    g=np.zeros((n,n),complex)
    for i,x in enumerate(m):
        g[i,i]=sum(z**3/x*(c[k,i]**2-c[i,k]**2) for k,z in enumerate(m))
        for j,y in enumerate(m):
            if i==j:continue
            g[j,i]=3*c[j,i]*(y*y-x*x)
            g[j,i]+=sum(z**3/(x*y)*(c[j,k].conjugate()*c[k,i]*x-c[j,k]*c[k,i].conjugate()*y) for k,z in enumerate(m))
    return alpha/(64*np.pi*sw2*mw**2)*b@g

from typing import Sequence
import numpy as np
from scipy.linalg import eigh

def scheme_response(b: 'np.ndarray', finite_a: 'np.ndarray', finite_b: 'np.ndarray', restored: 'np.ndarray', diagonal_a: 'np.ndarray', diagonal_b: 'np.ndarray', flavor: int, pair: 'Sequence[int]') -> 'np.ndarray':
    b=np.asarray(b);a=int(flavor);i,j=map(int,pair);x=b[a,i];y=b[a,j].conjugate()
    def _deriv(db):
        return float(np.imag(2*x*y*(db[a,i]*y+x*db[a,j].conjugate())))
    da,db,du=_deriv(finite_a),_deriv(finite_b),_deriv(restored)
    return np.array([(da-db)/du,da,db,du,_deriv(diagonal_a-diagonal_b)])

from typing import Sequence
import numpy as np
from scipy.linalg import eigh

def gauge_scheme_benchmark(md: 'np.ndarray', mr: 'np.ndarray', ml: 'np.ndarray', ew: 'np.ndarray', scale: float, gauges: 'np.ndarray', flavor: int=0, pair: 'Sequence[int]'=(0, 1)) -> 'np.ndarray':
    basis=seesaw_basis(md,mr);m=basis[0].real;b=basis[1:len(ml)+1]
    mw,mz,alpha=map(float,ew);sw2=1-(mw/mz)**2;n=len(m);l=len(ml)
    totals=[];diagonals=[]
    for xi_w,xi_z in np.asarray(gauges):
        queries=[[x*x,z,np.sqrt(xi_w)*mw] for x in ml for z in m]
        queries.extend([x*x,z,np.sqrt(xi_w)*mw] for x in m for z in ml)
        queries.extend([x*x,z,np.sqrt(xi_z)*mz] for x in m for z in m)
        pv=pv_laurent(queries,scale)
        bl=pv[:l*n].reshape(l,n,2);bw=pv[l*n:2*l*n].reshape(n,l,2);bz=pv[2*l*n:].reshape(n,n,2)
        charged=charged_gauge(b,m,ml,mw,alpha,sw2,xi_w,bl)
        w=neutrino_w_gauge(b,m,ml,mw,alpha,sw2,xi_w,bw)
        z=neutrino_z_gauge(b,m,mw,mz,alpha,sw2,xi_z,bz)
        diag=majorana_diagonal_gauge(b,m,mw,mz,alpha,sw2,xi_z,bz)
        totals.append((charged+w+z+diag)[:,:,1]);diagonals.append(diag[:,:,1])
    restored=restored_uv(b,m,mw,alpha,sw2)
    return scheme_response(b,totals[0],totals[1],restored,diagonals[0],diagonals[1],flavor,pair)

from typing import Sequence
import numpy as np

def coincidence_susceptibility(md: 'np.ndarray', mr: 'np.ndarray', ml: 'np.ndarray', ew: 'np.ndarray', scale: float, gauges: 'np.ndarray', box: 'np.ndarray', flavors: 'Sequence[int]'=(0, 1), pair: 'Sequence[int]'=(0, 1)) -> 'np.ndarray':
    from scipy.optimize import root
    box=np.asarray(box,float)
    def _responses(x):
        shifted=np.array(md,complex,copy=True)
        shifted[:,0]*=np.exp(1j*x[0]);shifted[:,1]*=np.exp(1j*x[1])
        return np.array([gauge_scheme_benchmark(shifted,mr,ml,ew,scale,gauges,int(f),pair) for f in flavors])
    def _residual(x):
        vals=_responses(x)
        return 1e6*(vals[:,1]-vals[:,2])
    roots=[]
    for theta in np.linspace(box[0,0],box[0,1],4):
        for eta in np.linspace(box[1,0],box[1,1],5):
            fit=root(_residual,[theta,eta],tol=1e-9,options={'maxfev':110})
            point=(fit.x+np.pi)%(2*np.pi)-np.pi
            if np.all(point>=box[:,0]-1e-9) and np.all(point<=box[:,1]+1e-9) and np.linalg.norm(_residual(point))<1e-8:
                if all(np.linalg.norm(point-prev)>1e-6 for prev in roots):roots.append(point)
    if not roots:raise ValueError('The phase rectangle contains no isolated coincidence point')
    scores=[np.linalg.norm(_responses(point)[:,3]) for point in roots]
    selected=roots[int(np.argmax(scores))]
    h=.001
    normalization=_responses(selected)[:,3]
    def _contrast(x):
        vals=_responses(x)
        return vals[:,1]-vals[:,2]
    jac=np.column_stack([(_contrast(selected-2*h*v)-8*_contrast(selected-h*v)
                          +8*_contrast(selected+h*v)-_contrast(selected+2*h*v))/(12*h)/normalization for v in np.eye(2)])
    singular=np.linalg.svd(jac,compute_uv=False)
    return np.array([singular[0],selected[0],selected[1],max(scores),singular[-1]])
SCICODE_GOLD_EOF
