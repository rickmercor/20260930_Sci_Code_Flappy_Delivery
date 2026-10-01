"""
Advance the supplied canonical state by one timestep of the selected spin-mapping propagator.

The canonical state order is [R, phi, P, s]. Return the locally continuous azimuth branch nearest the input azimuth, treating full turns as physically equivalent.

Returns
-------
return updated_state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spin_mint_step(canonical_state: "np.ndarray", timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    """Advance [R, phi, P, s] by one timestep of the selected propagator.

    The supplied state must satisfy abs(s) < spin_radius. Azimuths that differ by an
    integer multiple of 2*pi represent the same state. Return the locally continuous
    branch nearest the input azimuth; this same local branch is used for derivatives.
    """
    return updated_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_spin_mint_step(canonical_state: "np.ndarray", timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    z = np.asarray(canonical_state, dtype=np.float64)
    p = np.asarray(parameters, dtype=np.float64)
    if z.shape != (4,) or not np.all(np.isfinite(z)):
        raise ValueError("canonical_state must contain four finite values")
    _oracle_spin_boson_potential(z[0], p)
    dt = float(timestep)
    if not np.isfinite(dt) or dt == 0.0:
        raise ValueError("timestep must be finite and nonzero")
    mass, *_, rs = p
    r, phi, mom, s = z
    if abs(s) >= rs:
        raise ValueError("canonical polar momentum must lie strictly inside the spin sphere")
    rho = np.sqrt(rs*rs-s*s)
    u = 2.0*np.array([rho*np.cos(phi), rho*np.sin(phi), s])
    rmid = r + 0.5*dt*mom/mass
    flow = _oracle_coupled_flow(rmid, mom, u, dt, p)
    pnew, unew = flow[0], flow[1:]
    rnew = rmid + 0.5*dt*pnew/mass
    phiraw = np.arctan2(unew[1], unew[0])
    phinew = phi + np.arctan2(np.sin(phiraw-phi), np.cos(phiraw-phi))
    return np.array([rnew, phinew, pnew, 0.5*unew[2]], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [
        {'setup':'import numpy as np\ndef angle_key(x):\n x=np.asarray(x);return np.array([x[0],np.cos(x[1]),np.sin(x[1]),x[2],x[3]])\np_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy();z_sol=np.array([-.63,.91,.77,-.18]);z_ref=z_sol.copy()', 'call':'angle_key(spin_mint_step(z_sol,.0375,p_sol))', 'gold_call':'angle_key(_oracle_spin_mint_step(z_ref,.0375,p_ref))'},
        {'setup':'import numpy as np\ndef angle_key(x):\n x=np.asarray(x);return np.array([x[0],np.cos(x[1]),np.sin(x[1]),x[2],x[3]])\np_sol=np.array([2.,1.1,-.2,-.4,.3,.8]);p_ref=p_sol.copy();z_sol=np.array([0.,-1.2,-.4,.15]);z_ref=z_sol.copy()', 'call':'angle_key(spin_mint_step(z_sol,-.08,p_sol))', 'gold_call':'angle_key(_oracle_spin_mint_step(z_ref,-.08,p_ref))'},
        {'setup':'import numpy as np\ndef angle_key(x):\n x=np.asarray(x);return np.array([x[0],np.cos(x[1]),np.sin(x[1]),x[2],x[3]])\np_sol=np.array([1.,.7,.1,.5,.2,1.]);p_ref=p_sol.copy();z_sol=np.array([1.2,2.8,.3,-.4]);z_ref=z_sol.copy()', 'call':'angle_key(spin_mint_step(z_sol,.2,p_sol))', 'gold_call':'angle_key(_oracle_spin_mint_step(z_ref,.2,p_ref))'},
    ]
