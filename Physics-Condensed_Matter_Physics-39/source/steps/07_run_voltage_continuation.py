"""
Compute the final stationary state along a supplied contact-voltage path.

The voltage sequence begins at zero and increases strictly.  The left contact
remains fixed, while the right-contact potential follows the supplied voltage
and its carrier densities remain fixed by ``right_doping``.  The returned state
uses five potentials, five electron densities, and five hole densities, with
location order ``K0,K1,K2,K3,v4`` in every block.  Every continuation level
must satisfy the requested residual tolerance.

Returns
-------
np.ndarray of shape (15,), the converged active state at the final continuation voltage.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_voltage_continuation(
    vertices: "NDArray[np.float64]", cell_centers: "NDArray[np.float64]", diamonds: "NDArray[np.int64]",
    control_volumes: "NDArray[np.float64]", doping: "NDArray[np.float64]", left_doping: float,
    right_doping: float, left_voltage: float, voltages: "NDArray[np.float64]",
    neumann_diamonds: "NDArray[np.int64]", D_n: float, D_p: float, gamma_p: float,
    tolerance: float=1.0e-12, max_iterations: int=20, jacobian_check_step: float=2.0e-6,
) -> "NDArray[np.float64]":
    """Solve the stationary system along a strictly increasing voltage path.
    
    Parameters
    ----------
    vertices : np.ndarray
        Finite vertex-coordinate array of shape ``(5,2)`` for ``v0,...,v4``.
    cell_centers : np.ndarray
        Finite primal/boundary cell-point array of shape ``(8,2)`` for
        ``x0,...,x7``.
    diamonds : np.ndarray
        Integer array of shape ``(8,4)`` containing the prescribed benchmark
        ``(K,L,Kstar,Lstar)`` connectivities; rows may be permuted.
    control_volumes : np.ndarray
        Finite strictly positive length-5 control-volume array ordered as
        ``[K0,K1,K2,K3,v4]``.
    doping : np.ndarray
        Finite length-5 active-cell doping array in that same order.
    left_doping : float
        Finite left-contact doping used to construct its Maxwell-Boltzmann
        Dirichlet state.
    right_doping : float
        Finite right-contact doping. Its carrier densities remain fixed along
        the continuation path while its potential changes with voltage.
    left_voltage : float
        Finite fixed left-contact voltage.
    voltages : np.ndarray
        Nonempty finite one-dimensional right-contact voltage sequence. The first
        entry must be zero and all later entries must increase strictly.
    neumann_diamonds : np.ndarray
        One-dimensional integer array containing the current row indices of the
        two insulating boundary diamonds, remapped consistently if ``diamonds``
        has been permuted.
    D_n : float
        Finite strictly positive electron diffusion coefficient.
    D_p : float
        Finite strictly positive hole diffusion coefficient.
    gamma_p : float
        Finite Poisson charge-coupling coefficient.
    tolerance : float, optional
        Finite strictly positive infinity-norm residual tolerance required at
        every continuation voltage.
    max_iterations : int, optional
        Positive integer maximum number of nonlinear iterations allowed at each
        continuation voltage.
    jacobian_check_step : float, optional
        Finite strictly positive consistency-check scale passed to
        ``assemble_newton_system``.
    
    Returns
    -------
    state : np.ndarray
        Final converged length-15 active state ordered as five potentials, five
        electron densities, and five hole densities for ``K0,K1,K2,K3,v4``.
    
    Raises
    ------
    ValueError
        If any input is invalid, an update system is singular or nonfinite, a
        candidate update leaves the positive-density domain, or a continuation level
        does not reach ``tolerance`` within ``max_iterations``.
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import NDArray


def _step07_contact_state(doping_value,voltage):
    N=float(doping_value); V=float(voltage)
    if not np.isfinite(N) or not np.isfinite(V): raise ValueError("contact doping and voltage must be finite")
    n=(N+np.sqrt(N*N+4.0))/2.0; p=(-N+np.sqrt(N*N+4.0))/2.0
    return np.array([V+np.log(n),n,p],dtype=np.float64)


def _oracle_run_voltage_continuation(
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
            r=_oracle_assemble_stationary_residual(state,vertices,cell_centers,diamonds,control_volumes,dop,left,right,neumann_diamonds,D_n,D_p,gamma_p)
            if float(np.max(np.abs(r)))<tol: converged=True; break
            sys=_oracle_assemble_newton_system(state,vertices,cell_centers,diamonds,control_volumes,dop,left,right,neumann_diamonds,D_n,D_p,gamma_p,chk)
            try: delta=np.linalg.solve(sys[:15],sys[15])
            except np.linalg.LinAlgError as exc: raise ValueError("Newton Jacobian is singular") from exc
            if not np.all(np.isfinite(delta)): raise ValueError("Newton increment is nonfinite")
            trial=state+delta
            if np.any(trial[5:]<=0): raise ValueError("full Newton step left the positive-density domain")
            state=trial
        if not converged:
            r=_oracle_assemble_stationary_residual(state,vertices,cell_centers,diamonds,control_volumes,dop,left,right,neumann_diamonds,D_n,D_p,gamma_p)
            if float(np.max(np.abs(r)))>=tol: raise ValueError("Newton continuation failed to converge")
    return np.asarray(state,dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return canonical, alternate-path, true-permutation, and zero-bias continuation cases."""
    setup='import numpy as np\nv=np.array([[0.,0.],[2.,0.1],[1.7,1.3],[-0.2,0.9],[1.27,0.53]])\nc=np.array([[1.09,0.21],[1.6566666666666667,0.6433333333333333],[0.99,0.61],[0.5,0.7333333333333333],[1.,0.05],[1.85,0.7],[0.75,1.1],[-0.1,0.45]])\nd=np.array([[0,2,0,4],[0,1,1,4],[1,2,2,4],[2,3,0,2],[0,4,0,1],[1,5,1,2],[3,6,2,3],[3,7,3,0]],dtype=int)\nvol=np.array([0.4665,0.3735,0.375,0.895,0.135])\ndop=np.array([-0.6,0.6,-0.6,-0.6,0.6])\nneu=np.array([4,6],dtype=int)'
    return [
        {"setup":setup+"\nseq=np.array([0.0,0.125,0.25])","call":"np.round(run_voltage_continuation(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6),11)","gold_call":"np.round(_oracle_run_voltage_continuation(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6),11)"},
        {"setup":setup+"\nseq=np.array([0.0,0.05,0.13,0.19,0.25])","call":"np.round(run_voltage_continuation(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6),11)","gold_call":"np.round(_oracle_run_voltage_continuation(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6),11)"},
        {"setup":setup+"\nperm=np.array([5,2,7,0,6,1,4,3]); d=d[perm]; neu=np.flatnonzero(np.isin(perm,[4,6])).astype(int); seq=np.array([0.0,0.125,0.25])","call":"np.round(run_voltage_continuation(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6),11)","gold_call":"np.round(_oracle_run_voltage_continuation(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),1.0,0.65,0.35,1e-12,20,2e-6),11)"},
        {"setup":setup+"\nseq=np.array([0.0])","call":"np.round(run_voltage_continuation(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),0.8,0.9,0.2,1e-12,20,2e-6),11)","gold_call":"np.round(_oracle_run_voltage_continuation(v.copy(),c.copy(),d.copy(),vol.copy(),dop.copy(),-0.6,0.6,0.0,seq.copy(),neu.copy(),0.8,0.9,0.2,1e-12,20,2e-6),11)"},
    ]
