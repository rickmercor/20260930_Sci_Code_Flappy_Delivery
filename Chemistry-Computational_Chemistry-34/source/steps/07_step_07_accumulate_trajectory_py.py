"""
Calculate the final canonical state, accumulated trajectory Jacobian, and spin-norm diagnostic over the supplied number of propagation steps.

Measure the spin-norm diagnostic on each propagated Cartesian spin before conversion to canonical spin coordinates, as specified in the public return contract.

Returns
-------
return trajectory_summary
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def accumulate_trajectory(initial_state: "np.ndarray", timestep: float, n_steps: int, parameters: "np.ndarray") -> "np.ndarray":
    """Return final state, accumulated monodromy, and maximum actual spin-norm drift.

    n_steps must be a positive integer. The output contains 4 final-state values,
    16 row-major monodromy entries, and one maximum absolute spin-norm drift. At each
    step the drift is measured from the norm of the propagated Cartesian spin vector
    returned by the coupled electronic subflow, before conversion back to (phi,s).
    """
    return trajectory_summary

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_accumulate_trajectory(initial_state: "np.ndarray", timestep: float, n_steps: int, parameters: "np.ndarray") -> "np.ndarray":
    z = np.asarray(initial_state, dtype=np.float64).copy()
    p = np.asarray(parameters, dtype=np.float64)
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or int(n_steps) <= 0:
        raise ValueError("n_steps must be a positive integer")
    n_steps = int(n_steps)
    rs = p[5]
    mtotal = np.eye(4)
    max_drift = 0.0
    for _ in range(n_steps):
        r, phi, mom, s = z
        if abs(s) >= rs:
            raise ValueError("canonical polar momentum must lie strictly inside the spin sphere")
        rho = np.sqrt(rs*rs-s*s)
        spin = 2.0*np.array([rho*np.cos(phi), rho*np.sin(phi), s])
        rmid = r + 0.5*float(timestep)*mom/p[0]
        propagated_spin = _oracle_coupled_flow(rmid, mom, spin, timestep, p)[1:]
        max_drift = max(max_drift, abs(np.linalg.norm(propagated_spin)-2.0*rs))
        out = _oracle_canonical_tangent_step(z, timestep, p)
        z = out[:4]
        mtotal = out[4:].reshape(4,4) @ mtotal
    return np.concatenate((z, mtotal.ravel(), [max_drift])).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [{'setup': 'import numpy as np\n'
               'def angle_key(x):\n'
               ' x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\n'
               'p_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy();z_sol=np.array([-.63,.91,.77,-.18]);z_ref=z_sol.copy()',
      'call': 'angle_key(accumulate_trajectory(z_sol,.0375,3,p_sol))',
      'gold_call': 'angle_key(_oracle_accumulate_trajectory(z_ref,.0375,3,p_ref))'},
     {'setup': 'import numpy as np\n'
               'def angle_key(x):\n'
               ' x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\n'
               'p_sol=np.array([2.,1.1,-.2,-.4,.3,.8]);p_ref=p_sol.copy();z_sol=np.array([0.,-1.2,-.4,.15]);z_ref=z_sol.copy()',
      'call': 'angle_key(accumulate_trajectory(z_sol,.02,7,p_sol))',
      'gold_call': 'angle_key(_oracle_accumulate_trajectory(z_ref,.02,7,p_ref))'},
     {'setup': 'import numpy as np\n'
               'def angle_key(x):\n'
               ' x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\n'
               'p_sol=np.array([1.,.7,.1,.5,.2,1.]);p_ref=p_sol.copy();z_sol=np.array([1.2,2.8,.3,-.4]);z_ref=z_sol.copy()',
      'call': 'angle_key(accumulate_trajectory(z_sol,.01,11,p_sol))',
      'gold_call': 'angle_key(_oracle_accumulate_trajectory(z_ref,.01,11,p_ref))'},
     {'setup': 'import numpy as np\n'
               'def angle_key(x):\n'
               ' x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\n'
               'p_sol=np.array([1.3,.85,.35,.72,.41,1e16]);p_ref=p_sol.copy();z_sol=np.array([-.63,.91,.77,-4e15]);z_ref=z_sol.copy()',
      'call': 'angle_key(accumulate_trajectory(z_sol,.0375,3,p_sol))',
      'gold_call': 'angle_key(_oracle_accumulate_trajectory(z_ref,.0375,3,p_ref))'},
     {'setup': 'import numpy as np\n'
               'def angle_key(x):\n'
               ' x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\n'
               'p_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy();z_sol=np.array([-.63,.91,.77,-.18]);z_ref=z_sol.copy()',
      'call': 'angle_key(accumulate_trajectory(z_sol,.0375,1,p_sol))',
      'gold_call': 'angle_key(_oracle_accumulate_trajectory(z_ref,.0375,1,p_ref))'},
     {'setup': 'import numpy as np\n'
               'def angle_key(x):\n'
               ' x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\n'
               'p_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy();z_sol=np.array([-.63,.91,.77,-.18]);z_ref=z_sol.copy()',
      'call': 'angle_key(accumulate_trajectory(z_sol,-.0375,23,p_sol))',
      'gold_call': 'angle_key(_oracle_accumulate_trajectory(z_ref,-.0375,23,p_ref))'}]
