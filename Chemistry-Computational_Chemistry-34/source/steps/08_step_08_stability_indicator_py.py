"""
Compute stability, spin-norm, symplecticity, and Liouville diagnostics for the complete trajectory.

Compute the finite-time stability indicator and geometric diagnostics.

The largest monodromy singular value measures trajectory stretching, while the canonical
symplectic and determinant defects diagnose preservation of Hamiltonian geometry.

Returns
-------
return diagnostics
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stability_indicator(initial_state: "np.ndarray", timestep: float, n_steps: int, parameters: "np.ndarray") -> "np.ndarray":
    """Return the final state and five end-to-end trajectory diagnostics.

    Returns nine float64 values: final [R,phi,P,s], maximum actual spin-norm drift,
    largest monodromy singular value, symplectic Frobenius defect, absolute
    determinant defect, and finite-time stability indicator. This is the final
    orchestrator step. The final azimuth is interpreted modulo 2*pi.
    """
    return diagnostics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_stability_indicator(initial_state: "np.ndarray", timestep: float, n_steps: int, parameters: "np.ndarray") -> "np.ndarray":
    summary = _oracle_accumulate_trajectory(initial_state, timestep, n_steps, parameters)
    zfinal = summary[:4]
    mtotal = summary[4:20].reshape(4,4)
    max_drift = summary[20]
    ident = np.eye(2)
    zero = np.zeros((2,2))
    jmat = np.block([[zero,ident],[-ident,zero]])
    sigma = np.linalg.svd(mtotal, compute_uv=False)[0]
    symplectic_defect = np.linalg.norm(mtotal.T@jmat@mtotal-jmat, ord='fro')
    determinant_defect = abs(np.linalg.det(mtotal)-1.0)
    total_time = abs(float(timestep))*int(n_steps)
    indicator = np.log(sigma)/total_time
    return np.concatenate((zfinal,[max_drift,sigma,symplectic_defect,determinant_defect,indicator])).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [{'setup': 'import numpy as np\n'
               'def angle_key(x):\n'
               ' x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\n'
               'p_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy();z_sol=np.array([-.63,.91,.77,-.18]);z_ref=z_sol.copy()',
      'call': 'angle_key(stability_indicator(z_sol,.0375,320,p_sol))',
      'gold_call': 'angle_key(_oracle_stability_indicator(z_ref,.0375,320,p_ref))'},
     {'setup': 'import numpy as np\n'
               'def angle_key(x):\n'
               ' x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\n'
               'p_sol=np.array([2.,1.1,-.2,-.4,.3,.8]);p_ref=p_sol.copy();z_sol=np.array([0.,-1.2,-.4,.15]);z_ref=z_sol.copy()',
      'call': 'angle_key(stability_indicator(z_sol,.02,7,p_sol))',
      'gold_call': 'angle_key(_oracle_stability_indicator(z_ref,.02,7,p_ref))'},
     {'setup': 'import numpy as np\n'
               'def angle_key(x):\n'
               ' x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\n'
               'p_sol=np.array([1.,.7,.1,.5,.2,1.]);p_ref=p_sol.copy();z_sol=np.array([1.2,2.8,.3,-.4]);z_ref=z_sol.copy()',
      'call': 'angle_key(stability_indicator(z_sol,.01,11,p_sol))',
      'gold_call': 'angle_key(_oracle_stability_indicator(z_ref,.01,11,p_ref))'},
     {'setup': 'import numpy as np\n'
               'def angle_key(x):\n'
               ' x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\n'
               'p_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy();z_sol=np.array([-.63,.91,.77,-.18]);z_ref=z_sol.copy()',
      'call': 'angle_key(stability_indicator(z_sol,.0375,1,p_sol))',
      'gold_call': 'angle_key(_oracle_stability_indicator(z_ref,.0375,1,p_ref))'},
     {'setup': 'import numpy as np\n'
               'def angle_key(x):\n'
               ' x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\n'
               'p_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy();z_sol=np.array([-.63,.91,.77,-.18]);z_ref=z_sol.copy()',
      'call': 'angle_key(stability_indicator(z_sol,-.0375,23,p_sol))',
      'gold_call': 'angle_key(_oracle_stability_indicator(z_ref,-.0375,23,p_ref))'}]
