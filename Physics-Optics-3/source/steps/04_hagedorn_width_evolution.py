"""
Propagate matrix-valued Hagedorn widths and continuous complex normalizations.

Along the stored packet centers, evolve the paper's Hagedorn matrices with `Qdot=P/m` and `Pdot=-Hess(V)Q` using the same endpoint kick-drift-kick ordering. Initialize `Q=sqrt(hbar)*Gamma^(-1/2)` and `P=i*sqrt(hbar)*Gamma^(1/2)`. Recover the complex normalization from the continuous square-root branch of `gamma=gamma0*sqrt(det(Q0)/det(Qfinal))`: unwrap the phase of `det(Q)` chronologically, taking each phase increment in `[-pi,pi)`. Solver implementations call the public derivative function; the private oracle chains its `_oracle_` twin.

Returns
-------
One packed final Hagedorn width/normalization row per packet.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def hagedorn_width_evolution(center_trajectory, final_time, potential_parameters,
                              basis_gamma_diag, mass=1.0, hbar=1.0):
    """Return a float array of shape (N,18).

    Columns are row-major Re(Q)[4], Im(Q)[4], Re(P)[4], Im(P)[4],
    Re(gamma), Im(gamma).

    Raises
    ------
    ValueError
        If the trajectory, widths, time, mass, hbar, or potential is invalid.
    """
    return np.empty((center_trajectory.shape[1], 18), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _width_potential_tuple(points, time, potential_parameters):
    import numpy as np
    packed = _oracle_interference_potential_derivatives(points, time, potential_parameters)
    value = packed[:, 0]
    gradient = packed[:, 1:3]
    hessian = np.empty((len(packed), 2, 2), dtype=float)
    hessian[:, 0, 0] = packed[:, 3]
    hessian[:, 0, 1] = packed[:, 4]
    hessian[:, 1, 0] = packed[:, 4]
    hessian[:, 1, 1] = packed[:, 5]
    return value, gradient, hessian

def _oracle_hagedorn_width_evolution(center_trajectory, final_time, potential_parameters, basis_gamma_diag, mass=1.0, hbar=1.0):
    import numpy as np
    params = potential_parameters
    basis_gamma = basis_gamma_diag
    traj=np.asarray(center_trajectory,float)
    basis_gamma=np.asarray(basis_gamma,float)
    if traj.ndim != 3 or traj.shape[2] != 5 or traj.shape[0] < 2 or traj.shape[1] < 1 or not np.all(np.isfinite(traj)):
        raise ValueError('center_trajectory must be finite with shape (steps+1,N,5)')
    if basis_gamma.shape != (2,) or np.any(basis_gamma <= 0) or not np.all(np.isfinite(basis_gamma)):
        raise ValueError('basis_gamma_diag must contain two finite positive values')
    if not np.all(np.isfinite([final_time,mass,hbar])) or final_time <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError('time, mass, and hbar must be finite and positive')
    _oracle_interference_potential_derivatives(traj[0,:,:2],0.0,params)
    steps=traj.shape[0]-1
    n=traj.shape[1]
    dt=final_time/steps
    G=np.diag(np.asarray(basis_gamma,float))
    Q0=np.sqrt(hbar)*np.diag(1/np.sqrt(np.diag(G))).astype(complex)
    P0=1j*np.sqrt(hbar)*np.diag(np.sqrt(np.diag(G))).astype(complex)
    Q=np.repeat(Q0[None,:,:],n,axis=0)
    P=np.repeat(P0[None,:,:],n,axis=0)
    determinant_phase=np.zeros(n,dtype=float)
    previous_principal_phase=np.angle(np.linalg.det(Q))
    for s in range(steps):
        _,_,H0=_width_potential_tuple(traj[s,:,:2],s*dt,params)
        Ph=P-0.5*dt*np.einsum('nij,njk->nik',H0,Q)
        Qn=Q+dt*Ph/mass
        _,_,H1=_width_potential_tuple(traj[s+1,:,:2],(s+1)*dt,params)
        Pn=Ph-0.5*dt*np.einsum('nij,njk->nik',H1,Qn)
        Q,P=Qn,Pn
        principal_phase=np.angle(np.linalg.det(Q))
        phase_increment=(principal_phase-previous_principal_phase+np.pi)%(2*np.pi)-np.pi
        determinant_phase+=phase_increment
        previous_principal_phase=principal_phase
    det_q=np.linalg.det(Q)
    det_q0=np.linalg.det(Q0)
    gamma0=(np.linalg.det(G)/(np.pi*hbar)**2)**0.25
    gammas=gamma0*np.sqrt(abs(det_q0)/np.abs(det_q))*np.exp(-0.5j*determinant_phase)
    return np.column_stack((Q.real.reshape(n,-1),Q.imag.reshape(n,-1),P.real.reshape(n,-1),P.imag.reshape(n,-1),gammas.real,gammas.imag))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\np=_oracle_phase_space_sobol_packets(2,3,np.array([-1.,0.]),np.array([1.,0.]),np.ones(2),2*np.ones(2));v=np.array([.3,0.,1.,1.,1.2,.4,2.1,.2]);tr=_oracle_stormer_verlet_centers_actions(p,1.,8,v)", "call":"hagedorn_width_evolution(tr,1.,v,2*np.ones(2))", "gold_call":"_oracle_hagedorn_width_evolution(tr,1.,v,2*np.ones(2))"},
        {"setup":"import numpy as np\np=_oracle_phase_space_sobol_packets(1,1,np.zeros(2),np.array([.5,-.2]),np.ones(2),np.ones(2));v=np.array([0.,0.,1.,1.,1.,1.,1.,0.]);tr=_oracle_stormer_verlet_centers_actions(p,.5,4,v)", "call":"hagedorn_width_evolution(tr,.5,v,np.ones(2))", "gold_call":"_oracle_hagedorn_width_evolution(tr,.5,v,np.ones(2))"},
        {"setup":"import numpy as np\np=_oracle_phase_space_sobol_packets(3,11,np.array([-2.,.4]),np.array([2.,.1]),np.array([.7,1.2]),np.array([1.4,2.4]));v=np.array([.8,.2,.6,1.7,-1.1,.9,3.,-.4]);tr=_oracle_stormer_verlet_centers_actions(p,1.4,17,v,.8)", "call":"hagedorn_width_evolution(tr,1.4,v,np.array([1.4,2.4]),.8)", "gold_call":"_oracle_hagedorn_width_evolution(tr,1.4,v,np.array([1.4,2.4]),.8)"}
    ]
