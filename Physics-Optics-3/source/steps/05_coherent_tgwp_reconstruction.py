"""
Reconstruct the complex meshfree TGWP wavefunction by coherent superposition.

Reconstruct the paper's coherent meshfree wavefunction on the declared Cartesian grid. For each packet form `C=P*Q^(-1)` and add `c_j*gamma_j*exp{i[(r-q_j)^T C_j(r-q_j)/2+p_j^T(r-q_j)+S_j]/hbar}` before taking any modulus. Grid storage is `(y,x)`. No global phase alignment is applied here.

Returns
-------
One packed complex TGWP field on the y-major grid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def coherent_tgwp_reconstruction(grid_x, grid_y, packet_rows,
                                  final_centers_actions, final_width_rows, hbar=1.0):
    """Return a float array of shape (len(grid_y),len(grid_x),2).

    The last axis is [real(psi),imag(psi)].

    Raises
    ------
    ValueError
        If grids or packet, center, width, or hbar contracts are invalid.
    """
    return np.empty((len(grid_y), len(grid_x), 2), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_coherent_tgwp_reconstruction(grid_x, grid_y, packet_rows, final_centers_actions, final_width_rows, hbar=1.0):
    import numpy as np
    final_centers = final_centers_actions
    final_widths = final_width_rows
    x=np.asarray(grid_x,float); y=np.asarray(grid_y,float)
    packet_rows=np.asarray(packet_rows,float); final_centers=np.asarray(final_centers,float); final_widths=np.asarray(final_widths,float)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 1 or len(y) < 1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('grids must be nonempty finite vectors')
    if packet_rows.ndim != 2 or packet_rows.shape[1] != 6 or final_centers.shape != (len(packet_rows),5) or final_widths.shape != (len(packet_rows),18):
        raise ValueError('packet, center, and width shapes are inconsistent')
    if not np.all(np.isfinite(np.r_[packet_rows.ravel(),final_centers.ravel(),final_widths.ravel(),hbar])) or hbar <= 0:
        raise ValueError('reconstruction inputs must be finite and hbar positive')
    X,Y=np.meshgrid(x,y,indexing='xy')
    pts=np.column_stack((X.ravel(),Y.ravel()))
    packets_=np.asarray(packet_rows,float); c=packets_[:,4]+1j*packets_[:,5]
    centers_=np.asarray(final_centers,float)
    widths_=np.asarray(final_widths,float)
    out=np.zeros(len(pts),complex)
    for j in range(len(packets_)):
        Q=widths_[j,0:4].reshape(2,2)+1j*widths_[j,4:8].reshape(2,2)
        P=widths_[j,8:12].reshape(2,2)+1j*widths_[j,12:16].reshape(2,2)
        gamma=widths_[j,16]+1j*widths_[j,17]
        C=P@np.linalg.inv(Q)
        dr=pts-centers_[j,:2]
        quad=0.5*np.einsum('ni,ij,nj->n',dr,C,dr)
        phase=quad+dr@centers_[j,2:4]+centers_[j,4]
        out += c[j]*gamma*np.exp(1j*phase/hbar)
    field = out.reshape(len(y),len(x)); return np.stack((field.real, field.imag), axis=-1).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nx=np.linspace(-3,3,12,endpoint=False);y=np.linspace(-2,2,8,endpoint=False);p=_oracle_phase_space_sobol_packets(2,3,np.array([-1.,0.]),np.array([1.,0.]),np.ones(2),2*np.ones(2));v=np.array([.3,0.,1.,1.,1.2,.4,2.1,.2]);tr=_oracle_stormer_verlet_centers_actions(p,1.,8,v);w=_oracle_hagedorn_width_evolution(tr,1.,v,2*np.ones(2))", "call":"coherent_tgwp_reconstruction(x,y,p,tr[-1],w)", "gold_call":"_oracle_coherent_tgwp_reconstruction(x,y,p,tr[-1],w)"},
        {"setup":"import numpy as np\nx=np.linspace(-2,2,7);y=np.linspace(-1,1,5);p=_oracle_phase_space_sobol_packets(1,1,np.zeros(2),np.zeros(2),np.ones(2),np.ones(2));v=np.array([0.,0.,1.,1.,1.,1.,1.,0.]);tr=_oracle_stormer_verlet_centers_actions(p,.2,2,v);w=_oracle_hagedorn_width_evolution(tr,.2,v,np.ones(2))", "call":"coherent_tgwp_reconstruction(x,y,p,tr[-1],w)", "gold_call":"_oracle_coherent_tgwp_reconstruction(x,y,p,tr[-1],w)"},
        {"setup":"import numpy as np\nx=np.array([-1.5,-.2,.9]);y=np.array([-.7,.4]);p=_oracle_phase_space_sobol_packets(3,9,np.array([-.5,.1]),np.array([1.3,-.2]),np.array([.6,1.5]),np.array([1.7,.8]));v=np.array([.9,-.2,.7,1.8,-1.,.5,2.7,.6]);tr=_oracle_stormer_verlet_centers_actions(p,.8,9,v);w=_oracle_hagedorn_width_evolution(tr,.8,v,np.array([1.7,.8]))", "call":"coherent_tgwp_reconstruction(x,y,p,tr[-1],w)", "gold_call":"_oracle_coherent_tgwp_reconstruction(x,y,p,tr[-1],w)"}
    ]
