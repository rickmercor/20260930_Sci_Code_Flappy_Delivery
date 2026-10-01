"""
Assemble the Newton linear system associated with the stationary residual.

The active state ordering is the same fifteen-component ordering used by
``assemble_stationary_residual``.  The returned packet has shape ``(16,15)``:
rows ``0`` through ``14`` contain the residual Jacobian and row ``15`` contains
the Newton right-hand side.  ``jacobian_check_step`` controls a deterministic
consistency validation.  Diamond rows may be permuted without changing the
physical system.

Returns
-------
np.ndarray of shape (16, 15): Jacobian rows 0 to 14 and the Newton right-hand side in row 15.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_newton_system(
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
    jacobian_check_step: float = 2.0e-6,
) -> "NDArray[np.float64]":
    """Assemble the Newton matrix and right-hand side.
    
    Parameters
    ----------
    state : np.ndarray
        Finite length-15 active state ordered as the five potentials
        ``[psi_K0,psi_K1,psi_K2,psi_K3,psi_v4]``, followed by the five
        electron densities in the same control-volume order and then the five
        hole densities. Active carrier densities must be strictly positive.
    vertices : np.ndarray
        Finite vertex-coordinate array of shape ``(5,2)`` for ``v0,...,v4``.
    cell_centers : np.ndarray
        Finite primal/boundary cell-point array of shape ``(8,2)`` for
        ``x0,...,x7``.
    diamonds : np.ndarray
        Integer array of shape ``(8,4)`` containing the prescribed benchmark
        connectivity ``(K,L,Kstar,Lstar)``. Rows may be permuted.
    control_volumes : np.ndarray
        Finite strictly positive length-5 array ordered as
        ``[|K0|,|K1|,|K2|,|K3|,|v4*|]``.
    doping : np.ndarray
        Finite length-5 active-cell doping array in the same control-volume order.
    left_state : np.ndarray
        Finite length-3 left Dirichlet state ``[psi,n,p]`` with positive carrier densities.
    right_state : np.ndarray
        Finite length-3 right Dirichlet state ``[psi,n,p]`` with positive carrier densities.
    neumann_diamonds : np.ndarray
        One-dimensional integer array containing the current row indices of the
        two insulating boundary diamonds. Indices must follow any row permutation
        of ``diamonds`` and must identify connectivities ``(0,4,0,1)`` and
        ``(3,6,2,3)``.
    D_n : float
        Finite strictly positive electron diffusion coefficient.
    D_p : float
        Finite strictly positive hole diffusion coefficient.
    gamma_p : float
        Finite Poisson charge-coupling coefficient.
    jacobian_check_step : float, optional
        Finite strictly positive scale controlling the deterministic Jacobian
        consistency validation.
    
    Returns
    -------
    system : np.ndarray
        Float array with shape ``(16,15)``. Rows 0:15 contain the
        Jacobian ``J=dR/dx``. Row 15 contains the Newton right-hand side
        ``-R(state)``.
    
    Raises
    ------
    ValueError
        If any input violates the stated benchmark/residual contract, the
        Jacobian is nonfinite, or its deterministic
        consistency check fails.
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_assemble_newton_system(
    state: NDArray[np.float64], vertices: NDArray[np.float64], cell_centers: NDArray[np.float64],
    diamonds: NDArray[np.int64], control_volumes: NDArray[np.float64], doping: NDArray[np.float64],
    left_state: NDArray[np.float64], right_state: NDArray[np.float64], neumann_diamonds: NDArray[np.int64],
    D_n: float, D_p: float, gamma_p: float, jacobian_check_step: float=2.0e-6,
) -> NDArray[np.float64]:
    x=np.asarray(state,dtype=np.float64); v=np.asarray(vertices,dtype=np.float64); centers=np.asarray(cell_centers,dtype=np.float64)
    d=np.asarray(diamonds); vols=np.asarray(control_volumes,dtype=np.float64); dop=np.asarray(doping,dtype=np.float64)
    left=np.asarray(left_state,dtype=np.float64); right=np.asarray(right_state,dtype=np.float64); neu=np.asarray(neumann_diamonds)
    r0=_oracle_assemble_stationary_residual(x,v,centers,d,vols,dop,left,right,neu,D_n,D_p,gamma_p)
    check=float(jacobian_check_step)
    if not np.isfinite(check) or check<=0: raise ValueError("jacobian_check_step must be finite and positive")
    d=d.astype(np.int64,copy=False); neu=neu.astype(np.int64,copy=False); gamma=float(gamma_p)
    geom=_oracle_construct_diamond_geometry(v,centers,d); pop=_oracle_build_ddfv_poisson_flux_operator(geom)
    psi_c,psi_v,n_c,n_v,p_c,p_v=_step06_fields(x,left,right)
    neus=set(neu.tolist())
    active_rows=np.array([i for i in range(d.shape[0]) if i not in neus],dtype=np.int64)
    jets_active=_oracle_compute_ddfv_ha_flux_jets(psi_c,psi_v,n_c,n_v,p_c,p_v,d[active_rows],geom[active_rows],D_n,D_p)
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
    rp=_oracle_assemble_stationary_residual(x+h*direction,v,centers,d,vols,dop,left,right,neu,D_n,D_p,gamma_p)
    rm=_oracle_assemble_stationary_residual(x-h*direction,v,centers,d,vols,dop,left,right,neu,D_n,D_p,gamma_p)
    fd=(rp-rm)/(2*h); an=J@direction
    if float(np.max(np.abs(fd-an)))>2e-7*max(1.0,float(np.max(np.abs(fd)))):
        raise ValueError("analytic Jacobian failed directional consistency check")
    packet=np.empty((16,15)); packet[:15]=J; packet[15]=-r0
    return packet

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return equilibrium, biased, near-solution, permuted, and high-curvature analytic-Jacobian cases."""
    common='import numpy as np\nv=np.array([[0.,0.],[2.,0.1],[1.7,1.3],[-0.2,0.9],[1.27,0.53]])\nc=np.array([[1.09,0.21],[1.6566666666666667,0.6433333333333333],[0.99,0.61],[0.5,0.7333333333333333],[1.,0.05],[1.85,0.7],[0.75,1.1],[-0.1,0.45]])\nd=np.array([[0,2,0,4],[0,1,1,4],[1,2,2,4],[2,3,0,2],[0,4,0,1],[1,5,1,2],[3,6,2,3],[3,7,3,0]],dtype=int)\nvol=np.array([0.4665,0.3735,0.375,0.895,0.135])\ndop=np.array([-0.6,0.6,-0.6,-0.6,0.6])\nneu=np.array([4,6],dtype=int)\ndef contact(N,V):\n n=(N+np.sqrt(N*N+4.0))/2.0; p=(-N+np.sqrt(N*N+4.0))/2.0; return np.array([V+np.log(n),n,p])\nleft=contact(-0.6,0.0)'
    return [
        {"setup":common+"\nright=contact(0.6,0.0)\nn=(dop+np.sqrt(dop*dop+4.0))/2.0; p=(-dop+np.sqrt(dop*dop+4.0))/2.0; x=np.concatenate((np.log(n),n,p))","call":"np.round(assemble_newton_system(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35,2e-6),11)","gold_call":"np.round(_oracle_assemble_newton_system(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35,2e-6),11)"},
        {"setup":common+"\nright=contact(0.6,0.25)\nx=np.array([0.15,0.43,0.12,-0.08,0.26,1.01,1.20,0.94,0.87,1.06,0.95,0.82,1.03,1.11,0.91])","call":"np.round(assemble_newton_system(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35,1.5e-6),11)","gold_call":"np.round(_oracle_assemble_newton_system(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35,1.5e-6),11)"},
        {"setup":common+"\nright=contact(0.6,0.25)\nx=np.array([0.16069,0.45232,0.13665,-0.06464,0.28151,0.98257,1.24282,0.96690,0.84300,1.09196,0.98249,0.79377,0.99658,1.15150,0.88595])","call":"np.round(assemble_newton_system(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35,2.5e-6),12)","gold_call":"np.round(_oracle_assemble_newton_system(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.0,0.65,0.35,2.5e-6),12)"},
        {"setup":common+"\nperm=np.array([5,2,7,0,6,1,4,3]); d=d[perm]; neu=np.flatnonzero(np.isin(perm,[4,6])).astype(int); right=contact(0.6,0.10)\nx=np.array([0.03,0.34,0.01,-0.16,0.16,0.99,1.24,0.97,0.84,1.08,0.99,0.80,1.00,1.15,0.89])","call":"np.round(assemble_newton_system(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),0.8,0.9,0.2,2e-6),11)","gold_call":"np.round(_oracle_assemble_newton_system(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),0.8,0.9,0.2,2e-6),11)"},
        {"setup":common+"\nright=contact(0.6,0.25)\nx=np.array([4.0,-3.0,2.0,-5.0,1.0,1.15,0.72,1.38,0.83,1.04,0.91,1.41,0.76,1.26,0.88])","call":"np.round(assemble_newton_system(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.15,0.55,0.42,8e-7),10)","gold_call":"np.round(_oracle_assemble_newton_system(x.copy(),v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),left.copy(),right.copy(),neu.copy(),1.15,0.55,0.42,8e-7),10)"},
    ]
