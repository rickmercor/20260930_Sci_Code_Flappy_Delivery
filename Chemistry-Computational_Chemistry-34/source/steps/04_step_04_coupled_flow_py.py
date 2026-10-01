"""
Advance nuclear momentum and Cartesian spin under the selected fixed-position coupled flow.

Use the supplied position, momentum, Cartesian spin, propagation interval, and spin-boson parameters. Return the updated momentum and spin in the declared order.

Returns
-------
return state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def coupled_flow(position: float, momentum: float, spin: "np.ndarray", timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    """Propagate momentum and spin at fixed nuclear position.

    Parameters
    ----------
    position, momentum, timestep : float
        Finite scalars; timestep must be nonzero.
    spin : np.ndarray
        Three finite real spin components.
    parameters : np.ndarray
        Spin-boson parameter vector defined in Step 1.

    Returns
    -------
    state : np.ndarray
        Four float64 values [updated_momentum, updated_u_x, updated_u_y, updated_u_z].

    Raises
    ------
    ValueError
        If scalar or spin validation fails.
    """
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_coupled_flow(position: float, momentum: float, spin: "np.ndarray", timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    position, momentum = float(position), float(momentum)
    spin = np.asarray(spin, dtype=np.float64)
    if not np.isfinite(position) or not np.isfinite(momentum) or spin.shape != (3,) or not np.all(np.isfinite(spin)):
        raise ValueError("position, momentum, and three spin components must be finite")
    vals = _oracle_spin_boson_potential(position, parameters)
    ops = _oracle_electronic_operators(position, timestep, parameters)
    q = ops[3:12].reshape(3,3)
    b = ops[12:21].reshape(3,3)
    hr = _oracle_spin_generator(position, parameters)[21:24].real
    pnew = momentum - timestep*(vals[4] + 0.5*(vals[5]+vals[6])) - 0.5*np.dot(hr, b @ spin).real
    unew = (q @ spin).real
    return np.concatenate(([pnew], unew)).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [
        {'setup':'import numpy as np\np_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy();u_sol=np.array([.7,-.2,1.1]);u_ref=u_sol.copy()', 'call':'coupled_flow(-.63,.77,u_sol,.0375,p_sol)', 'gold_call':'_oracle_coupled_flow(-.63,.77,u_ref,.0375,p_ref)'},
        {'setup':'import numpy as np\np_sol=np.array([2.,1.1,-.2,-.4,.3,.8]);p_ref=p_sol.copy();u_sol=np.array([.1,.2,.3]);u_ref=u_sol.copy()', 'call':'coupled_flow(0.,-.4,u_sol,-.08,p_sol)', 'gold_call':'_oracle_coupled_flow(0.,-.4,u_ref,-.08,p_ref)'},
        {'setup':'import numpy as np\np_sol=np.array([1.,.7,.1,.5,.2,1.]);p_ref=p_sol.copy();u_sol=np.array([-.2,.9,.1]);u_ref=u_sol.copy()', 'call':'coupled_flow(1.2,.3,u_sol,.2,p_sol)', 'gold_call':'_oracle_coupled_flow(1.2,.3,u_ref,.2,p_ref)'},
    ]
