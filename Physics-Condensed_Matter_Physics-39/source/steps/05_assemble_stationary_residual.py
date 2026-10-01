"""
Assemble the stationary residual for the prescribed benchmark topology.

The active locations are the four primal cells ``K0`` through ``K3`` and the
dual cell at ``v4``.  Contact data occupy ``K5``, ``K7``, and ``v0`` through
``v3``; ``K4`` and ``K6`` are insulating boundary placeholders with no ghost
values.  Diamond rows may be supplied in arbitrary order, and
``neumann_diamonds`` refers to row indices in that supplied order.  State and
residual blocks are ordered as five potentials, five electron densities, then
five hole densities, with location order ``K0,K1,K2,K3,v4`` in every block.

Returns
-------
np.ndarray of shape (15,), the stationary residual in the potential, electron, hole block order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_stationary_residual(
    state: "NDArray[np.float64]",
    vertices: "NDArray[np.float64]",
    cell_centers: "NDArray[np.float64]",
    diamonds: "NDArray[np.int64]",
    control_volumes: "NDArray[np.float64]",
    doping: "NDArray[np.float64]",
    left_state: "NDArray[np.float64]",
    right_state: "NDArray[np.float64]",
    neumann_diamonds: "NDArray[np.int64]",
    D_n: float,
    D_p: float,
    gamma_p: float,
) -> "NDArray[np.float64]":
    """Assemble the stationary DDFV residual for the prescribed topology.

    Parameters
    ----------
    state : np.ndarray
        Finite length-15 active state: five potentials, five electron
        densities, then five hole densities; each block is K0,K1,K2,K3,v4.
    vertices : np.ndarray
        Finite array with shape (5,2).
    cell_centers : np.ndarray
        Finite array with shape (8,2), ordered K0..K7.
    diamonds : np.ndarray
        Integer array with shape (8,4), rows (K,L,Kstar,Lstar), in arbitrary
        row order but with the prescribed eight connectivities.
    control_volumes : np.ndarray
        Positive length-5 array for K0..K3 and v4.
    doping : np.ndarray
        Finite length-5 active doping array in K0,K1,K2,K3,v4 order.
    left_state, right_state : np.ndarray
        Finite [psi,n,p] contact states with positive carrier densities.
    neumann_diamonds : np.ndarray
        Distinct integer row indices identifying exactly the two insulating
        connectivities (0,4,0,1) and (3,6,2,3) in the supplied row order.
    D_n, D_p : float
        Finite strictly positive diffusion coefficients.
    gamma_p : float
        Finite Poisson charge-coupling coefficient.

    Returns
    -------
    residual : np.ndarray
        Finite length-15 residual in the same block ordering as ``state``.

    Raises
    ------
    ValueError
        If state, topology, boundary labels, areas, contacts, or coefficients
        violate this contract.
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import NDArray


def _step05_validate_benchmark_topology(vertices, cell_centers, diamonds):
    if vertices.shape != (5,2) or cell_centers.shape != (8,2) or diamonds.shape != (8,4):
        raise ValueError("this residual contract requires the 5-vertex, 8-cell, 8-diamond topology")
    required={(0,2,0,4),(0,1,1,4),(1,2,2,4),(2,3,0,2),(0,4,0,1),(1,5,1,2),(3,6,2,3),(3,7,3,0)}
    rows=[tuple(map(int,r)) for r in diamonds]
    if len(set(rows))!=8 or set(rows)!=required:
        raise ValueError("diamonds do not match the prescribed benchmark connectivity")


def _oracle_assemble_stationary_residual(
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

    geometry=_oracle_construct_diamond_geometry(v,centers,d)
    active_rows=np.array([i for i in range(8) if i not in set(neu.tolist())],dtype=np.int64)
    da=d[active_rows]; ga=geometry[active_rows]
    pop=_oracle_build_ddfv_poisson_flux_operator(ga)
    q=np.column_stack((psi_c[da[:,0]],psi_c[da[:,1]],psi_v[da[:,2]],psi_v[da[:,3]]))
    pflux=np.einsum('mij,mj->mi',pop,q)
    cjets=_oracle_compute_ddfv_ha_flux_jets(psi_c,psi_v,n_c,n_v,p_c,p_v,da,ga,Dn,Dp)
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return equilibrium, biased, converged, and true row-permutation residual cases."""
    common='import numpy as np\nv=np.array([[0.,0.],[2.,0.1],[1.7,1.3],[-0.2,0.9],[1.27,0.53]])\nc=np.array([[1.09,0.21],[1.6566666666666667,0.6433333333333333],[0.99,0.61],[0.5,0.7333333333333333],[1.,0.05],[1.85,0.7],[0.75,1.1],[-0.1,0.45]])\nd=np.array([[0,2,0,4],[0,1,1,4],[1,2,2,4],[2,3,0,2],[0,4,0,1],[1,5,1,2],[3,6,2,3],[3,7,3,0]],dtype=int)\nvol=np.array([0.4665,0.3735,0.375,0.895,0.135])\ndop=np.array([-0.6,0.6,-0.6,-0.6,0.6])\nneu=np.array([4,6],dtype=int)\ndef contact(N,V):\n n=(N+np.sqrt(N*N+4.0))/2.0; p=(-N+np.sqrt(N*N+4.0))/2.0; return np.array([V+np.log(n),n,p])\nleft=contact(-0.6,0.0)'
    return [
        {"setup":common+"\nright=contact(0.6,0.0)\nn=(dop+np.sqrt(dop*dop+4.0))/2.0; p=(-dop+np.sqrt(dop*dop+4.0))/2.0; x=np.concatenate((np.log(n),n,p))","call":"np.round(assemble_stationary_residual(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35),11)","gold_call":"np.round(_oracle_assemble_stationary_residual(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35),11)"},
        {"setup":common+"\nright=contact(0.6,0.25)\nx=np.array([0.15,0.43,0.12,-0.08,0.26,1.01,1.20,0.94,0.87,1.06,0.95,0.82,1.03,1.11,0.91])","call":"np.round(assemble_stationary_residual(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),0.9,0.8,0.25),11)","gold_call":"np.round(_oracle_assemble_stationary_residual(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),0.9,0.8,0.25),11)"},
        {"setup":common+"\nright=contact(0.6,0.25)\nx=np.array([0.1606991679809223,0.4523310902170290,0.1366615751509087,-0.0646317464529461,0.2815199273379000,0.9825669232826038,1.2428301150824772,0.9668920571132257,0.8430014155349345,1.0919614339285924,0.9824905091602737,0.7937601427833058,0.9965801881583750,1.1515111160479945,0.8859439732720107])","call":"np.round(assemble_stationary_residual(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35),13)","gold_call":"np.round(_oracle_assemble_stationary_residual(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35),13)"},
        {"setup":common+"\nperm=np.array([5,2,7,0,6,1,4,3]); d=d[perm]; neu=np.flatnonzero(np.isin(perm,[4,6])).astype(int); right=contact(0.6,0.25)\nx=np.array([0.15,0.43,0.12,-0.08,0.26,1.01,1.20,0.94,0.87,1.06,0.95,0.82,1.03,1.11,0.91])","call":"np.round(assemble_stationary_residual(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35),11)","gold_call":"np.round(_oracle_assemble_stationary_residual(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35),11)"},
    ]
