"""
Evaluate carrier fluxes and their local first-derivative data on each diamond.

The four output rows are ordered as electron-primal, electron-dual,
hole-primal, and hole-dual.  Their channels are ordered as ``[flux, dpsi_K,
dpsi_L, dpsi_Kstar, dpsi_Lstar, dcarrier_K, dcarrier_L, dcarrier_Kstar,
dcarrier_Lstar]``.  Here ``carrier`` means electron density in the electron rows
and hole density in the hole rows.  Diamond rows are independent, geometric
orientation remains signed, and only field entries referenced by ``diamonds``
are required to be finite.

Returns
-------
np.ndarray of shape (m, 4, 9), the four carrier fluxes and their local derivatives for each diamond.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_ddfv_ha_flux_jets(
    psi_cells: "NDArray[np.float64]", psi_vertices: "NDArray[np.float64]",
    n_cells: "NDArray[np.float64]", n_vertices: "NDArray[np.float64]",
    p_cells: "NDArray[np.float64]", p_vertices: "NDArray[np.float64]",
    diamonds: "NDArray[np.int64]", geometry: "NDArray[np.float64]",
    D_n: float, D_p: float,
) -> "NDArray[np.float64]":
    """Evaluate harmonic-average carrier fluxes and exact local sensitivities.
    
    Parameters
    ----------
    psi_cells : np.ndarray
        One-dimensional primal-cell potential array. It must have the same
        length as ``n_cells`` and ``p_cells``. The first two columns of each
        ``diamonds`` row index this array.
    psi_vertices : np.ndarray
        One-dimensional dual-vertex potential array. It must have the same
        length as ``n_vertices`` and ``p_vertices``. The last two columns of
        each ``diamonds`` row index this array.
    n_cells : np.ndarray
        One-dimensional primal-cell electron-density array, with the same
        length as ``psi_cells`` and ``p_cells``. Referenced values must be finite.
    n_vertices : np.ndarray
        One-dimensional dual-vertex electron-density array, with the same
        length as ``psi_vertices`` and ``p_vertices``. Referenced values must be finite.
    p_cells : np.ndarray
        One-dimensional primal-cell hole-density array, with the same length
        as ``psi_cells`` and ``n_cells``. Referenced values must be finite.
    p_vertices : np.ndarray
        One-dimensional dual-vertex hole-density array, with the same length
        as ``psi_vertices`` and ``n_vertices``. Referenced values must be finite.
    diamonds : np.ndarray
        Integer array of shape ``(m,4)``. Each row is ``(K,L,Kstar,Lstar)``
        and indexes the primal-cell arrays with ``K,L`` and the dual-vertex
        arrays with ``Kstar,Lstar``.
    geometry : np.ndarray
        Finite float array of shape ``(m,11)`` returned by
        ``construct_diamond_geometry`` for the same diamond rows.
    D_n : float
        Finite strictly positive electron diffusion coefficient.
    D_p : float
        Finite strictly positive hole diffusion coefficient.
    
    Returns
    -------
    jets : np.ndarray
        Float array with shape ``(m,4,9)``. Flux rows are
        ``[Fn_primal, Fn_dual, Fp_primal, Fp_dual]``. Channel 0 is the flux.
        Channels 1:9 are derivatives with respect to
        ``[psi_K,psi_L,psi_Kstar,psi_Lstar,carrier_K,carrier_L,``
        ``carrier_Kstar,carrier_Lstar]``, where ``carrier`` is ``n`` for
        electron rows and ``p`` for hole rows.
    
    Raises
    ------
    ValueError
        If a field array is not one-dimensional, the three primal-cell field
        arrays do not have equal lengths, the three dual-vertex field arrays
        do not have equal lengths, connectivity or geometry is inconsistent,
        a referenced field value is nonfinite, or either diffusion coefficient
        is not finite and strictly positive.
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_compute_ddfv_ha_flux_jets(
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
    Bc=_oracle_evaluate_bernoulli(dc); Bcm=_oracle_evaluate_bernoulli(-dc)
    Bv=_oracle_evaluate_bernoulli(dv); Bvm=_oracle_evaluate_bernoulli(-dv)
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return orthogonal, skew, near-equilibrium, reversed, and strong-field jet cases."""
    return [
        {"setup":"import numpy as np\nd=np.array([[0,1,0,1]],dtype=int)\ng=np.array([[1.,2.,1.,1.,0.,0.,1.,0.,0.5,1.,2.]])\npc=np.array([0.2,-0.3]); pv=np.array([0.1,-0.4]); nc=np.array([1.1,0.8]); nv=np.array([0.9,1.2]); hc=np.array([0.7,1.3]); hv=np.array([1.4,0.6])","call":"np.round(compute_ddfv_ha_flux_jets(pc.copy(),pv.copy(),nc.copy(),nv.copy(),hc.copy(),hv.copy(),d.copy(),g.copy(),1.2,0.7),11)","gold_call":"np.round(_oracle_compute_ddfv_ha_flux_jets(pc.copy(),pv.copy(),nc.copy(),nv.copy(),hc.copy(),hv.copy(),d.copy(),g.copy(),1.2,0.7),11)"},
        {"setup":"import numpy as np\nv=np.array([[0.,0.],[2.,0.1],[1.7,1.3],[-0.2,0.9],[1.27,0.53]])\nx=np.array([[1.09,0.21],[1.6566666666666667,0.6433333333333333],[0.99,0.61],[0.5,0.7333333333333333],[1.,0.05],[1.85,0.7],[0.75,1.1],[-0.1,0.45]])\nd=np.array([[0,2,0,4],[2,3,0,2],[1,5,1,2]],dtype=int)\ng_model=construct_diamond_geometry(v.copy(),x.copy(),d.copy())\ng_gold=_oracle_construct_diamond_geometry(v.copy(),x.copy(),d.copy())\npc=np.array([0.15,0.43,0.12,-0.08,np.nan,0.545,np.nan,-0.296]); pv=np.array([-0.296,0.545,0.545,-0.296,0.26]); nc=np.array([1.01,1.20,0.94,0.87,np.nan,1.344,np.nan,0.744]); nv=np.array([0.744,1.344,1.344,0.744,1.06]); hc=np.array([0.95,0.82,1.03,1.11,np.nan,0.744,np.nan,1.344]); hv=np.array([1.344,0.744,0.744,1.344,0.91])","call":"np.round(compute_ddfv_ha_flux_jets(pc.copy(),pv.copy(),nc.copy(),nv.copy(),hc.copy(),hv.copy(),d.copy(),g_model.copy(),1.0,0.65),11)","gold_call":"np.round(_oracle_compute_ddfv_ha_flux_jets(pc.copy(),pv.copy(),nc.copy(),nv.copy(),hc.copy(),hv.copy(),d.copy(),g_gold.copy(),1.0,0.65),11)"},
        {"setup":"import numpy as np\nd=np.array([[0,1,0,1]],dtype=int)\nls,lss,eta=1.4,0.9,-0.35\ns=np.sqrt(1.0-eta*eta); A=0.5*ls*lss*s\ng=np.array([[ls,lss,A,1.,0.,eta,s,eta,ls*ls/(2*A),ls*lss/(2*A),lss*lss/(2*A)]])\npc=np.array([0.200000000001,0.2]); pv=np.array([-0.1,-0.100000000002]); nc=np.array([1.2,1.2]); nv=np.array([0.8,0.8]); hc=np.array([0.7,0.7]); hv=np.array([1.4,1.4])","call":"np.round(compute_ddfv_ha_flux_jets(pc.copy(),pv.copy(),nc.copy(),nv.copy(),hc.copy(),hv.copy(),d.copy(),g.copy(),0.8,1.1),12)","gold_call":"np.round(_oracle_compute_ddfv_ha_flux_jets(pc.copy(),pv.copy(),nc.copy(),nv.copy(),hc.copy(),hv.copy(),d.copy(),g.copy(),0.8,1.1),12)"},
        {"setup":"import numpy as np\nv=np.array([[0.,0.],[1.7,1.3]])\nx=np.array([[0.99,0.61],[0.5,0.7333333333333333]])\nd=np.array([[0,1,0,1],[1,0,1,0]],dtype=int)\ng_model=construct_diamond_geometry(v.copy(),x.copy(),d.copy())\ng_gold=_oracle_construct_diamond_geometry(v.copy(),x.copy(),d.copy())\npc=np.array([0.7,-0.2]); pv=np.array([0.4,-0.6]); nc=np.array([1.3,0.75]); nv=np.array([0.85,1.25]); hc=np.array([0.65,1.4]); hv=np.array([1.5,0.7])","call":"np.round(compute_ddfv_ha_flux_jets(pc.copy(),pv.copy(),nc.copy(),nv.copy(),hc.copy(),hv.copy(),d.copy(),g_model.copy(),1.1,0.6),11)","gold_call":"np.round(_oracle_compute_ddfv_ha_flux_jets(pc.copy(),pv.copy(),nc.copy(),nv.copy(),hc.copy(),hv.copy(),d.copy(),g_gold.copy(),1.1,0.6),11)"},
        {"setup":"import numpy as np\nd=np.array([[0,1,0,1]],dtype=int)\nls,lss,eta=1.8,1.1,0.62\ns=np.sqrt(1.0-eta*eta); A=0.5*ls*lss*s\ng=np.array([[ls,lss,A,1.,0.,eta,s,eta,ls*ls/(2*A),ls*lss/(2*A),lss*lss/(2*A)]])\npc=np.array([6.,-4.]); pv=np.array([-3.,5.]); nc=np.array([1.8,0.55]); nv=np.array([0.62,1.7]); hc=np.array([0.58,1.9]); hv=np.array([1.6,0.66])","call":"np.round(compute_ddfv_ha_flux_jets(pc.copy(),pv.copy(),nc.copy(),nv.copy(),hc.copy(),hv.copy(),d.copy(),g.copy(),1.3,0.45),9)","gold_call":"np.round(_oracle_compute_ddfv_ha_flux_jets(pc.copy(),pv.copy(),nc.copy(),nv.copy(),hc.copy(),hv.copy(),d.copy(),g.copy(),1.3,0.45),9)"},
    ]
