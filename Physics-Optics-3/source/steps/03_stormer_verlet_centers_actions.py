"""
Propagate packet centers and classical actions with the paper's second-order scheme.

Evolve every packet center with the paper's second-order Störmer-Verlet scheme under the supplied time-dependent potential. Use the left gradient for the first half momentum kick, drift with that half-step momentum, use the right gradient for the second half kick, and accumulate `Sdot=p^T p/(2m)-V` with half-step momentum and the time/position midpoint. Solver implementations obtain derivatives by calling the earlier public derivative function; the private oracle chains its `_oracle_` twin.

Returns
-------
One full center/action trajectory in chronological order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def stormer_verlet_centers_actions(packet_rows, final_time, step_count,
                                    potential_parameters, mass=1.0, hbar=1.0):
    """Return a float array of shape (step_count+1,N,5).

    The last axis is [qx,qy,px,py,action] at every stored time.

    Raises
    ------
    ValueError
        If packet shape, time, step count, mass, hbar, or potential is invalid.
    """
    return np.empty((int(step_count) + 1, len(packet_rows), 5), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _potential_tuple(points, time, potential_parameters):
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

def _oracle_stormer_verlet_centers_actions(packet_rows, final_time, step_count, potential_parameters, mass=1.0, hbar=1.0):
    import numpy as np
    params = potential_parameters
    packet_rows = np.asarray(packet_rows, float)
    if packet_rows.ndim != 2 or packet_rows.shape[1] != 6 or len(packet_rows) == 0 or not np.all(np.isfinite(packet_rows)):
        raise ValueError('packet_rows must be a nonempty finite (N,6) array')
    if isinstance(step_count, bool) or int(step_count) != step_count or int(step_count) < 1:
        raise ValueError('step_count must be a positive integer')
    if not np.all(np.isfinite([final_time,mass,hbar])) or final_time <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError('time, mass, and hbar must be finite and positive')
    _oracle_interference_potential_derivatives(packet_rows[:,:2],0.0,params)
    n = len(packet_rows)
    dt = final_time / step_count
    out = np.empty((step_count+1, n, 5))
    q = packet_rows[:,:2].copy()
    p = packet_rows[:,2:4].copy()
    action = np.zeros(n)
    out[0] = np.column_stack((q,p,action))
    for s in range(step_count):
        t = s*dt
        _, grad0, _ = _potential_tuple(q,t,params)
        ph = p - 0.5*dt*grad0
        qn = q + dt*ph/mass
        _, grad1, _ = _potential_tuple(qn,t+dt,params)
        pn = ph - 0.5*dt*grad1
        qm = 0.5*(q+qn)
        vm,_,_ = _potential_tuple(qm,t+0.5*dt,params)
        action += dt*(np.sum(ph*ph,axis=1)/(2*mass)-vm)
        q,p=qn,pn
        out[s+1]=np.column_stack((q,p,action))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\np=_oracle_phase_space_sobol_packets(2,3,np.array([-1.,0.]),np.array([1.,0.]),np.ones(2),2*np.ones(2));v=np.array([.3,0.,1.,1.,1.2,.4,2.1,.2])", "call":"stormer_verlet_centers_actions(p,1.,8,v)", "gold_call":"_oracle_stormer_verlet_centers_actions(p,1.,8,v)"},
        {"setup":"import numpy as np\np=_oracle_phase_space_sobol_packets(1,1,np.zeros(2),np.array([.5,-.2]),np.ones(2),np.ones(2));v=np.array([0.,0.,1.,1.,1.,1.,1.,0.])", "call":"stormer_verlet_centers_actions(p,.5,4,v)", "gold_call":"_oracle_stormer_verlet_centers_actions(p,.5,4,v)"},
        {"setup":"import numpy as np\np=_oracle_phase_space_sobol_packets(3,11,np.array([-2.,.4]),np.array([2.,.1]),np.array([.7,1.2]),np.array([1.4,2.4]));v=np.array([.8,.2,.6,1.7,-1.1,.9,3.,-.4])", "call":"stormer_verlet_centers_actions(p,1.4,17,v,.8)", "gold_call":"_oracle_stormer_verlet_centers_actions(p,1.4,17,v,.8)"}
    ]
