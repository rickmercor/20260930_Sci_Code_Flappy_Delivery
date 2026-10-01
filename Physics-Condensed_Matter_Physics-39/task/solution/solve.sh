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
from numpy.typing import NDArray


def evaluate_bernoulli(
    arguments: NDArray[np.float64],
) -> NDArray[np.float64]:
    t = np.asarray(arguments, dtype=np.float64)
    if not np.all(np.isfinite(t)):
        raise ValueError("arguments must contain only finite values")

    values = np.empty_like(t, dtype=np.float64)
    small = np.abs(t) < 1.0e-7
    x = t[small]
    values[small] = 1.0 - x / 2.0 + x * x / 12.0 - x**4 / 720.0 + x**6 / 30240.0
    values[~small] = t[~small] / np.expm1(t[~small])
    return values

import numpy as np
from numpy.typing import NDArray


def construct_diamond_geometry(
    vertices: NDArray[np.float64],
    cell_centers: NDArray[np.float64],
    diamonds: NDArray[np.int64],
) -> NDArray[np.float64]:
    v = np.asarray(vertices, dtype=np.float64)
    x = np.asarray(cell_centers, dtype=np.float64)
    d = np.asarray(diamonds)
    if v.ndim != 2 or v.shape[1] != 2 or x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("vertices and cell_centers must have shape (n,2)")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(x)):
        raise ValueError("coordinates must be finite")
    if d.ndim != 2 or d.shape[1] != 4 or d.shape[0] == 0:
        raise ValueError("diamonds must have shape (m,4) with m >= 1")
    if not np.issubdtype(d.dtype, np.integer):
        raise ValueError("diamonds must contain integer indices")
    d = d.astype(np.int64, copy=False)
    if np.any(d[:, :2] < 0) or np.any(d[:, :2] >= x.shape[0]):
        raise ValueError("primal diamond indices are out of range")
    if np.any(d[:, 2:] < 0) or np.any(d[:, 2:] >= v.shape[0]):
        raise ValueError("dual diamond indices are out of range")

    geometry = np.empty((d.shape[0], 11), dtype=np.float64)
    for i, (K, L, Kstar, Lstar) in enumerate(d):
        sigma_vec = v[Lstar] - v[Kstar]
        sigmastar_vec = x[L] - x[K]
        sigma = float(np.linalg.norm(sigma_vec))
        sigmastar = float(np.linalg.norm(sigmastar_vec))
        if sigma <= 0.0 or sigmastar <= 0.0:
            raise ValueError("diamond edges must have positive length")
        area = 0.5 * abs(float(np.linalg.det(np.stack((sigmastar_vec, sigma_vec)))))
        scale = sigma * sigmastar
        if area <= 1.0e-14 * max(1.0, scale):
            raise ValueError("diamond geometry is degenerate")

        n_kl = np.array([-sigma_vec[1], sigma_vec[0]], dtype=np.float64) / sigma
        if float(np.dot(n_kl, sigmastar_vec)) < 0.0:
            n_kl = -n_kl
        n_star = np.array([-sigmastar_vec[1], sigmastar_vec[0]], dtype=np.float64) / sigmastar
        if float(np.dot(n_star, sigma_vec)) < 0.0:
            n_star = -n_star

        eta = float(np.dot(n_kl, n_star))
        a = sigma * sigma / (2.0 * area)
        b = sigma * sigmastar / (2.0 * area)
        c = sigmastar * sigmastar / (2.0 * area)
        geometry[i] = [sigma, sigmastar, area, n_kl[0], n_kl[1], n_star[0], n_star[1], eta, a, b, c]
    return geometry

import numpy as np
from numpy.typing import NDArray


def build_ddfv_poisson_flux_operator(
    geometry: NDArray[np.float64],
) -> NDArray[np.float64]:
    g = np.asarray(geometry, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 11 or g.shape[0] < 1 or not np.all(np.isfinite(g)):
        raise ValueError("geometry must be finite with shape (m,11), m >= 1")
    if np.any(g[:, :3] <= 0.0) or np.any(g[:, 8:11] <= 0.0):
        raise ValueError("geometry contains nonpositive lengths, areas, or coefficients")
    if np.any(np.abs(g[:, 7]) > 1.0 + 1.0e-12):
        raise ValueError("eta must be a valid normal dot product")
    # Cross-check the three metric coefficients against the stored lengths/area.
    sigma, sigstar, area = g[:,0], g[:,1], g[:,2]
    ref = np.column_stack((sigma*sigma/(2*area), sigma*sigstar/(2*area), sigstar*sigstar/(2*area)))
    if not np.allclose(g[:,8:11], ref, rtol=2e-12, atol=2e-13):
        raise ValueError("geometry coefficients are inconsistent with lengths and area")
    a,b,c,eta = g[:,8],g[:,9],g[:,10],g[:,7]
    op = np.empty((g.shape[0],2,4),dtype=np.float64)
    op[:,0,0] = -a
    op[:,0,1] =  a
    op[:,0,2] = -b*eta
    op[:,0,3] =  b*eta
    op[:,1,0] = -b*eta
    op[:,1,1] =  b*eta
    op[:,1,2] = -c
    op[:,1,3] =  c
    return op

import numpy as np
from numpy.typing import NDArray


def _step04_bernoulli_prime(z):
    z=np.asarray(z,dtype=np.float64); out=np.empty_like(z)
    small=np.abs(z)<1e-5; s=z[small]
    out[small]=-0.5+s/6.0-s**3/180.0+s**5/5040.0
    q=z[~small]
    if q.size:
        em1=np.expm1(q); eq=em1+1.0
        out[~small]=(em1-q*eq)/(em1*em1)
    return out


def compute_ddfv_ha_flux_jets(
    psi_cells: NDArray[np.float64], psi_vertices: NDArray[np.float64],
    n_cells: NDArray[np.float64], n_vertices: NDArray[np.float64],
    p_cells: NDArray[np.float64], p_vertices: NDArray[np.float64],
    diamonds: NDArray[np.int64], geometry: NDArray[np.float64], D_n: float, D_p: float,
) -> NDArray[np.float64]:
    arrays=[np.asarray(a,dtype=np.float64) for a in (psi_cells,psi_vertices,n_cells,n_vertices,p_cells,p_vertices)]
    if any(a.ndim!=1 for a in arrays): raise ValueError("all field arrays must be one-dimensional")
    pc,pv,nc,nv,hc,hv=arrays; d=np.asarray(diamonds); g=np.asarray(geometry,dtype=np.float64)
    if pc.size!=nc.size or pc.size!=hc.size: raise ValueError("psi_cells, n_cells, and p_cells must have equal lengths")
    if pv.size!=nv.size or pv.size!=hv.size: raise ValueError("psi_vertices, n_vertices, and p_vertices must have equal lengths")
    if d.ndim!=2 or d.shape[1]!=4 or not np.issubdtype(d.dtype,np.integer): raise ValueError("diamonds must have integer shape (m,4)")
    d=d.astype(np.int64,copy=False)
    if g.shape!=(d.shape[0],11) or not np.all(np.isfinite(g)): raise ValueError("geometry must be finite shape (m,11)")
    if np.any(d[:,:2]<0) or np.any(d[:,:2]>=pc.size) or np.any(d[:,2:]<0) or np.any(d[:,2:]>=pv.size): raise ValueError("diamond indices out of range")
    for arr,idx in ((pc,d[:,:2]),(nc,d[:,:2]),(hc,d[:,:2]),(pv,d[:,2:]),(nv,d[:,2:]),(hv,d[:,2:])):
        if not np.all(np.isfinite(arr[idx])): raise ValueError("referenced field values must be finite")
    Dn=float(D_n); Dp=float(D_p)
    if not np.isfinite(Dn) or not np.isfinite(Dp) or Dn<=0 or Dp<=0: raise ValueError("diffusion coefficients must be finite positive")
    K,L,Ks,Ls=d.T
    dc=pc[K]-pc[L]; dv=pv[Ks]-pv[Ls]
    Bc=evaluate_bernoulli(dc); Bcm=evaluate_bernoulli(-dc)
    Bv=evaluate_bernoulli(dv); Bvm=evaluate_bernoulli(-dv)
    dBc=_step04_bernoulli_prime(dc); dBcm=_step04_bernoulli_prime(-dc)
    dBv=_step04_bernoulli_prime(dv); dBvm=_step04_bernoulli_prime(-dv)
    gn_c=Bc*nc[K]-Bcm*nc[L]; gn_v=Bv*nv[Ks]-Bvm*nv[Ls]
    gp_c=Bcm*hc[K]-Bc*hc[L]; gp_v=Bvm*hv[Ks]-Bv*hv[Ls]
    dgnc=dBc*nc[K]+dBcm*nc[L]; dgnv=dBv*nv[Ks]+dBvm*nv[Ls]
    dgpc=-dBcm*hc[K]-dBc*hc[L]; dgpv=-dBvm*hv[Ks]-dBv*hv[Ls]
    eta=g[:,7]; a=g[:,8]; b=g[:,9]; c=g[:,10]
    jets=np.zeros((d.shape[0],4,9),dtype=np.float64)
    # helper fills one flux row from direct/cross directional coefficients.
    def fill(row,alpha,beta,gv_c,gv_v,dpsi_c,dpsi_v,ca,cb,va,vb):
        jets[:,row,0]=alpha*gv_c+beta*gv_v
        jets[:,row,1]=alpha*dpsi_c; jets[:,row,2]=-alpha*dpsi_c
        jets[:,row,3]=beta*dpsi_v; jets[:,row,4]=-beta*dpsi_v
        jets[:,row,5]=alpha*ca; jets[:,row,6]=alpha*cb
        jets[:,row,7]=beta*va; jets[:,row,8]=beta*vb
    fill(0,Dn*a,Dn*b*eta,gn_c,gn_v,dgnc,dgnv,Bc,-Bcm,Bv,-Bvm)
    fill(1,Dn*b*eta,Dn*c,gn_c,gn_v,dgnc,dgnv,Bc,-Bcm,Bv,-Bvm)
    fill(2,Dp*a,Dp*b*eta,gp_c,gp_v,dgpc,dgpv,Bcm,-Bc,Bvm,-Bv)
    fill(3,Dp*b*eta,Dp*c,gp_c,gp_v,dgpc,dgpv,Bcm,-Bc,Bvm,-Bv)
    if not np.all(np.isfinite(jets)): raise ValueError("flux jet is nonfinite")
    return jets

import numpy as np
from numpy.typing import NDArray


def _step05_validate_benchmark_topology(vertices, cell_centers, diamonds):
    if vertices.shape != (5,2) or cell_centers.shape != (8,2) or diamonds.shape != (8,4):
        raise ValueError("this residual contract requires the 5-vertex, 8-cell, 8-diamond topology")
    required={(0,2,0,4),(0,1,1,4),(1,2,2,4),(2,3,0,2),(0,4,0,1),(1,5,1,2),(3,6,2,3),(3,7,3,0)}
    rows=[tuple(map(int,r)) for r in diamonds]
    if len(set(rows))!=8 or set(rows)!=required:
        raise ValueError("diamonds do not match the prescribed benchmark connectivity")


def assemble_stationary_residual(
    state: NDArray[np.float64], vertices: NDArray[np.float64], cell_centers: NDArray[np.float64],
    diamonds: NDArray[np.int64], control_volumes: NDArray[np.float64], doping: NDArray[np.float64],
    left_state: NDArray[np.float64], right_state: NDArray[np.float64], neumann_diamonds: NDArray[np.int64],
    D_n: float, D_p: float, gamma_p: float,
) -> NDArray[np.float64]:
    x=np.asarray(state,dtype=np.float64); v=np.asarray(vertices,dtype=np.float64); centers=np.asarray(cell_centers,dtype=np.float64)
    d=np.asarray(diamonds); volumes=np.asarray(control_volumes,dtype=np.float64); dop=np.asarray(doping,dtype=np.float64)
    left=np.asarray(left_state,dtype=np.float64); right=np.asarray(right_state,dtype=np.float64); neu=np.asarray(neumann_diamonds)
    if x.shape!=(15,) or not np.all(np.isfinite(x)): raise ValueError("state must be a finite length-15 array")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(centers)): raise ValueError("mesh coordinates must be finite")
    if d.dtype.kind not in 'iu': raise ValueError("diamonds must contain integer indices")
    d=d.astype(np.int64,copy=False); _step05_validate_benchmark_topology(v,centers,d)
    if volumes.shape!=(5,) or not np.all(np.isfinite(volumes)) or np.any(volumes<=0): raise ValueError("invalid control volumes")
    if dop.shape!=(5,) or not np.all(np.isfinite(dop)): raise ValueError("invalid doping")
    if left.shape!=(3,) or right.shape!=(3,) or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)): raise ValueError("invalid contact states")
    if np.any(left[1:]<=0) or np.any(right[1:]<=0): raise ValueError("contact carrier densities must be positive")
    if neu.ndim!=1 or not np.issubdtype(neu.dtype,np.integer): raise ValueError("neumann_diamonds must be integer 1D")
    neu=neu.astype(np.int64,copy=False)
    if np.any(neu<0) or np.any(neu>=8) or np.unique(neu).size!=neu.size: raise ValueError("invalid neumann row indices")
    if {tuple(map(int,d[i])) for i in neu}!={(0,4,0,1),(3,6,2,3)}: raise ValueError("wrong insulating diamonds")
    Dn=float(D_n); Dp=float(D_p); gamma=float(gamma_p)
    if not np.isfinite(Dn) or not np.isfinite(Dp) or Dn<=0 or Dp<=0 or not np.isfinite(gamma): raise ValueError("invalid coefficients")
    psi_a=x[:5]; n_a=x[5:10]; p_a=x[10:15]
    if np.any(n_a<=0) or np.any(p_a<=0): raise ValueError("active carrier densities must be positive")

    psi_c=np.full(8,np.nan); n_c=np.full(8,np.nan); p_c=np.full(8,np.nan)
    psi_v=np.full(5,np.nan); n_v=np.full(5,np.nan); p_v=np.full(5,np.nan)
    psi_c[:4]=psi_a[:4]; n_c[:4]=n_a[:4]; p_c[:4]=p_a[:4]
    psi_v[4]=psi_a[4]; n_v[4]=n_a[4]; p_v[4]=p_a[4]
    psi_c[5],n_c[5],p_c[5]=right; psi_c[7],n_c[7],p_c[7]=left
    for j in (0,3): psi_v[j],n_v[j],p_v[j]=left
    for j in (1,2): psi_v[j],n_v[j],p_v[j]=right

    geometry=construct_diamond_geometry(v,centers,d)
    active_rows=np.array([i for i in range(8) if i not in set(neu.tolist())],dtype=np.int64)
    da=d[active_rows]; ga=geometry[active_rows]
    pop=build_ddfv_poisson_flux_operator(ga)
    q=np.column_stack((psi_c[da[:,0]],psi_c[da[:,1]],psi_v[da[:,2]],psi_v[da[:,3]]))
    pflux=np.einsum('mij,mj->mi',pop,q)
    cjets=compute_ddfv_ha_flux_jets(psi_c,psi_v,n_c,n_v,p_c,p_v,da,ga,Dn,Dp)
    cflux=cjets[:,:,0]

    divp=np.zeros(5); divn=np.zeros(5); divh=np.zeros(5)
    amap={int(row):j for j,row in enumerate(active_rows)}
    for row,(K,L,Ks,Ls) in enumerate(d):
        if row in amap:
            j=amap[row]; fp0,fp1=pflux[j]; fn0,fn1,fh0,fh1=cflux[j]
        else:
            fp0=fp1=fn0=fn1=fh0=fh1=0.0
        if K<4: divp[K]+=fp0; divn[K]+=fn0; divh[K]+=fh0
        if L<4: divp[L]-=fp0; divn[L]-=fn0; divh[L]-=fh0
        if Ks==4: divp[4]+=fp1; divn[4]+=fn1; divh[4]+=fh1
        if Ls==4: divp[4]-=fp1; divn[4]-=fn1; divh[4]-=fh1
    divp/=volumes; divn/=volumes; divh/=volumes
    rpsi=divp+gamma*(p_a-n_a+dop)
    r=np.concatenate((rpsi,divn,divh))
    if not np.all(np.isfinite(r)): raise ValueError("assembled residual is nonfinite")
    return r

import numpy as np
from numpy.typing import NDArray


def _step06_fields(state,left,right):
    x=np.asarray(state,dtype=np.float64); psi_a=x[:5]; n_a=x[5:10]; p_a=x[10:15]
    psi_c=np.full(8,np.nan); n_c=np.full(8,np.nan); p_c=np.full(8,np.nan)
    psi_v=np.full(5,np.nan); n_v=np.full(5,np.nan); p_v=np.full(5,np.nan)
    psi_c[:4]=psi_a[:4]; n_c[:4]=n_a[:4]; p_c[:4]=p_a[:4]
    psi_v[4]=psi_a[4]; n_v[4]=n_a[4]; p_v[4]=p_a[4]
    psi_c[5],n_c[5],p_c[5]=right; psi_c[7],n_c[7],p_c[7]=left
    for j in (0,3): psi_v[j],n_v[j],p_v[j]=left
    for j in (1,2): psi_v[j],n_v[j],p_v[j]=right
    return psi_c,psi_v,n_c,n_v,p_c,p_v


def _step06_col(kind,is_vertex,index):
    if is_vertex:
        if index!=4: return None
        return {'psi':0,'n':5,'p':10}[kind]+4
    if index>=4: return None
    return {'psi':0,'n':5,'p':10}[kind]+int(index)


def assemble_newton_system(
    state: NDArray[np.float64], vertices: NDArray[np.float64], cell_centers: NDArray[np.float64],
    diamonds: NDArray[np.int64], control_volumes: NDArray[np.float64], doping: NDArray[np.float64],
    left_state: NDArray[np.float64], right_state: NDArray[np.float64], neumann_diamonds: NDArray[np.int64],
    D_n: float, D_p: float, gamma_p: float, jacobian_check_step: float=2.0e-6,
) -> NDArray[np.float64]:
    x=np.asarray(state,dtype=np.float64); v=np.asarray(vertices,dtype=np.float64); centers=np.asarray(cell_centers,dtype=np.float64)
    d=np.asarray(diamonds); vols=np.asarray(control_volumes,dtype=np.float64); dop=np.asarray(doping,dtype=np.float64)
    left=np.asarray(left_state,dtype=np.float64); right=np.asarray(right_state,dtype=np.float64); neu=np.asarray(neumann_diamonds)
    r0=assemble_stationary_residual(x,v,centers,d,vols,dop,left,right,neu,D_n,D_p,gamma_p)
    check=float(jacobian_check_step)
    if not np.isfinite(check) or check<=0: raise ValueError("jacobian_check_step must be finite and positive")
    d=d.astype(np.int64,copy=False); neu=neu.astype(np.int64,copy=False); gamma=float(gamma_p)
    geom=construct_diamond_geometry(v,centers,d); pop=build_ddfv_poisson_flux_operator(geom)
    psi_c,psi_v,n_c,n_v,p_c,p_v=_step06_fields(x,left,right)
    neus=set(neu.tolist())
    active_rows=np.array([i for i in range(d.shape[0]) if i not in neus],dtype=np.int64)
    jets_active=compute_ddfv_ha_flux_jets(psi_c,psi_v,n_c,n_v,p_c,p_v,d[active_rows],geom[active_rows],D_n,D_p)
    jet_index={int(row):j for j,row in enumerate(active_rows)}
    J=np.zeros((15,15),dtype=np.float64)
    def receivers(K,L,Ks,Ls,direction):
        if direction==0:
            ans=[]
            if K<4: ans.append((int(K),+1.0/vols[K]))
            if L<4: ans.append((int(L),-1.0/vols[L]))
            return ans
        ans=[]
        if Ks==4: ans.append((4,+1.0/vols[4]))
        if Ls==4: ans.append((4,-1.0/vols[4]))
        return ans
    for row,(K,L,Ks,Ls) in enumerate(d):
        if row in neus: continue
        pcols=[_step06_col('psi',False,K),_step06_col('psi',False,L),_step06_col('psi',True,Ks),_step06_col('psi',True,Ls)]
        for direction in (0,1):
            for rr,scale in receivers(K,L,Ks,Ls,direction):
                for j,col in enumerate(pcols):
                    if col is not None: J[rr,col]+=scale*pop[row,direction,j]
        # Each carrier jet has channels dpsiK,dpsiL,dpsiKs,dpsiLs,dcarrierK,dcarrierL,dcarrierKs,dcarrierLs.
        local_psi=pcols
        local_n=[_step06_col('n',False,K),_step06_col('n',False,L),_step06_col('n',True,Ks),_step06_col('n',True,Ls)]
        local_p=[_step06_col('p',False,K),_step06_col('p',False,L),_step06_col('p',True,Ks),_step06_col('p',True,Ls)]
        for direction,fluxrow in ((0,0),(1,1)):
            deriv=jets_active[jet_index[row],fluxrow]
            vec=np.zeros(15)
            for j,col in enumerate(local_psi):
                if col is not None: vec[col]+=deriv[1+j]
            for j,col in enumerate(local_n):
                if col is not None: vec[col]+=deriv[5+j]
            for rr,scale in receivers(K,L,Ks,Ls,direction): J[5+rr]+=scale*vec
        for direction,fluxrow in ((0,2),(1,3)):
            deriv=jets_active[jet_index[row],fluxrow]
            vec=np.zeros(15)
            for j,col in enumerate(local_psi):
                if col is not None: vec[col]+=deriv[1+j]
            for j,col in enumerate(local_p):
                if col is not None: vec[col]+=deriv[5+j]
            for rr,scale in receivers(K,L,Ks,Ls,direction): J[10+rr]+=scale*vec
    for i in range(5): J[i,5+i]-=gamma; J[i,10+i]+=gamma
    if not np.all(np.isfinite(J)): raise ValueError("analytic Jacobian is nonfinite")
    direction=np.linspace(-0.9,1.1,15); direction/=np.linalg.norm(direction)
    h=check*max(1.0,float(np.max(np.abs(x))))
    rp=assemble_stationary_residual(x+h*direction,v,centers,d,vols,dop,left,right,neu,D_n,D_p,gamma_p)
    rm=assemble_stationary_residual(x-h*direction,v,centers,d,vols,dop,left,right,neu,D_n,D_p,gamma_p)
    fd=(rp-rm)/(2*h); an=J@direction
    if float(np.max(np.abs(fd-an)))>2e-7*max(1.0,float(np.max(np.abs(fd)))):
        raise ValueError("analytic Jacobian failed directional consistency check")
    packet=np.empty((16,15)); packet[:15]=J; packet[15]=-r0
    return packet

import numpy as np
from numpy.typing import NDArray


def _step07_contact_state(doping_value,voltage):
    N=float(doping_value); V=float(voltage)
    if not np.isfinite(N) or not np.isfinite(V): raise ValueError("contact doping and voltage must be finite")
    n=(N+np.sqrt(N*N+4.0))/2.0; p=(-N+np.sqrt(N*N+4.0))/2.0
    return np.array([V+np.log(n),n,p],dtype=np.float64)


def run_voltage_continuation(
    vertices: NDArray[np.float64], cell_centers: NDArray[np.float64], diamonds: NDArray[np.int64],
    control_volumes: NDArray[np.float64], doping: NDArray[np.float64], left_doping: float,
    right_doping: float, left_voltage: float, voltages: NDArray[np.float64],
    neumann_diamonds: NDArray[np.int64], D_n: float, D_p: float, gamma_p: float,
    tolerance: float=1.0e-12, max_iterations: int=20, jacobian_check_step: float=2.0e-6,
) -> NDArray[np.float64]:
    dop=np.asarray(doping,dtype=np.float64); seq=np.asarray(voltages,dtype=np.float64)
    if dop.shape!=(5,) or not np.all(np.isfinite(dop)): raise ValueError("doping must be finite length-5")
    if seq.ndim!=1 or seq.size==0 or not np.all(np.isfinite(seq)): raise ValueError("voltages must be nonempty finite 1D")
    if abs(float(seq[0]))>1e-15 or (seq.size>1 and np.any(np.diff(seq)<=0)): raise ValueError("voltages must start at zero and increase strictly")
    tol=float(tolerance); chk=float(jacobian_check_step)
    if not np.isfinite(tol) or tol<=0 or not np.isfinite(chk) or chk<=0: raise ValueError("tolerance and jacobian_check_step must be positive")
    if isinstance(max_iterations,(bool,np.bool_)) or not isinstance(max_iterations,(int,np.integer)) or int(max_iterations)<1: raise ValueError("max_iterations must be a positive integer")
    n0=(dop+np.sqrt(dop*dop+4.0))/2.0; p0=(-dop+np.sqrt(dop*dop+4.0))/2.0
    state=np.concatenate((np.log(n0),n0,p0)).astype(np.float64)
    left=_step07_contact_state(left_doping,left_voltage)
    for voltage in seq:
        right=_step07_contact_state(right_doping,float(voltage)); converged=False
        for _ in range(int(max_iterations)):
            r=assemble_stationary_residual(state,vertices,cell_centers,diamonds,control_volumes,dop,left,right,neumann_diamonds,D_n,D_p,gamma_p)
            if float(np.max(np.abs(r)))<tol: converged=True; break
            sys=assemble_newton_system(state,vertices,cell_centers,diamonds,control_volumes,dop,left,right,neumann_diamonds,D_n,D_p,gamma_p,chk)
            try: delta=np.linalg.solve(sys[:15],sys[15])
            except np.linalg.LinAlgError as exc: raise ValueError("Newton Jacobian is singular") from exc
            if not np.all(np.isfinite(delta)): raise ValueError("Newton increment is nonfinite")
            trial=state+delta
            if np.any(trial[5:]<=0): raise ValueError("full Newton step left the positive-density domain")
            state=trial
        if not converged:
            r=assemble_stationary_residual(state,vertices,cell_centers,diamonds,control_volumes,dop,left,right,neumann_diamonds,D_n,D_p,gamma_p)
            if float(np.max(np.abs(r)))>=tol: raise ValueError("Newton continuation failed to converge")
    return np.asarray(state,dtype=np.float64)

import numpy as np
from numpy.typing import NDArray


def solve_stationary_ddfv_ha(
    vertices: NDArray[np.float64],
    cell_centers: NDArray[np.float64],
    diamonds: NDArray[np.int64],
    control_volumes: NDArray[np.float64],
    doping: NDArray[np.float64],
    left_doping: float,
    right_doping: float,
    left_voltage: float,
    voltages: NDArray[np.float64],
    neumann_diamonds: NDArray[np.int64],
    D_n: float,
    D_p: float,
    gamma_p: float,
    tolerance: float = 1.0e-12,
    max_iterations: int = 20,
    jacobian_check_step: float = 2.0e-6,
    conservation_tolerance: float = 1.0e-10,
) -> float:
    state = run_voltage_continuation(
        vertices, cell_centers, diamonds, control_volumes, doping,
        left_doping, right_doping, left_voltage, voltages,
        neumann_diamonds, D_n, D_p, gamma_p, tolerance,
        max_iterations, jacobian_check_step,
    )
    seq = np.asarray(voltages, dtype=np.float64)
    if seq.ndim != 1 or seq.size == 0:
        raise ValueError("voltages must be a nonempty one-dimensional array")
    final_voltage = float(seq[-1])
    cons_tol = float(conservation_tolerance)
    if not np.isfinite(cons_tol) or cons_tol <= 0.0:
        raise ValueError("conservation_tolerance must be finite and positive")

    left = _step07_contact_state(left_doping, left_voltage)
    right = _step07_contact_state(right_doping, final_voltage)
    psi_active = state[:5]; n_active = state[5:10]; p_active = state[10:15]
    psi_c = np.full(8, np.nan, dtype=np.float64); n_c = np.full(8, np.nan, dtype=np.float64); p_c = np.full(8, np.nan, dtype=np.float64)
    psi_v = np.full(5, np.nan, dtype=np.float64); n_v = np.full(5, np.nan, dtype=np.float64); p_v = np.full(5, np.nan, dtype=np.float64)
    psi_c[:4] = psi_active[:4]; n_c[:4] = n_active[:4]; p_c[:4] = p_active[:4]
    psi_v[4] = psi_active[4]; n_v[4] = n_active[4]; p_v[4] = p_active[4]
    psi_c[5], n_c[5], p_c[5] = right
    psi_c[7], n_c[7], p_c[7] = left
    for idx in (0, 3):
        psi_v[idx], n_v[idx], p_v[idx] = left
    for idx in (1, 2):
        psi_v[idx], n_v[idx], p_v[idx] = right

    d = np.asarray(diamonds, dtype=np.int64)
    geometry = construct_diamond_geometry(np.asarray(vertices, dtype=np.float64), np.asarray(cell_centers, dtype=np.float64), d)
    rows = [tuple(map(int, row)) for row in d]
    try:
        right_row = rows.index((1,5,1,2))
        left_row = rows.index((3,7,3,0))
    except ValueError as exc:
        raise ValueError("required physical contact diamonds are missing") from exc
    contact_rows = np.array([right_row, left_row], dtype=np.int64)
    flux = compute_ddfv_ha_flux_jets(
        psi_c, psi_v, n_c, n_v, p_c, p_v,
        d[contact_rows], geometry[contact_rows], D_n, D_p,
    )
    right_current = float(flux[0, 0, 0] + flux[0, 2, 0])
    left_current = float(flux[1, 0, 0] + flux[1, 2, 0])
    if not np.isfinite(right_current) or not np.isfinite(left_current):
        raise ValueError("terminal current is nonfinite")
    if abs(right_current + left_current) > cons_tol:
        raise ValueError("terminal-current conservation check failed")
    return right_current
SCICODE_GOLD_EOF
