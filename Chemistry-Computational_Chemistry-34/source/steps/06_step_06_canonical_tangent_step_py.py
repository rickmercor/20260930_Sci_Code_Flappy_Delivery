"""
Calculate the canonical state update and its analytical one-step Jacobian for the selected propagator.

Return the updated canonical state and all row-major Jacobian entries in the declared order. Use analytical differentiation on the stated local azimuth branch; finite differences, numerical trajectory perturbations, and automatic differentiation are excluded.

Returns
-------
return state_and_monodromy
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def canonical_tangent_step(canonical_state: "np.ndarray", timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    """Return the updated state and its exact 4 by 4 canonical Jacobian.

    Parameters and the local continuous azimuth convention follow spin_mint_step.
    The return is a float64 vector containing the four updated state values followed
    by the row-major 4 by 4 derivative of that local branch.
    """
    return state_and_monodromy

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm, expm_frechet

def _oracle_canonical_tangent_step(canonical_state: "np.ndarray", timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    z = np.asarray(canonical_state, dtype=np.float64)
    p = np.asarray(parameters, dtype=np.float64)
    znew = _oracle_spin_mint_step(z, timestep, p)
    dt = float(timestep)
    mass, omega, _, _, _, rs = p
    r, phi, mom, s = z
    rho = np.sqrt(rs*rs-s*s)
    u = 2.0*np.array([rho*np.cos(phi), rho*np.sin(phi), s])
    du_dphi = 2.0*np.array([-rho*np.sin(phi), rho*np.cos(phi), 0.0])
    du_ds = 2.0*np.array([-s*np.cos(phi)/rho, -s*np.sin(phi)/rho, 1.0])

    # First half drift: x=[R,P,u_x,u_y,u_z].
    d1 = np.eye(5)
    d1[0,1] = 0.5*dt/mass
    rmid = r + 0.5*dt*mom/mass

    gen = _oracle_spin_generator(rmid, p)
    w = gen[:9].reshape(3,3)
    wr = gen[9:18].reshape(3,3)
    a = -1j*w
    ar = -1j*wr
    q, qr = expm_frechet(a*dt, ar*dt, compute_expm=True)
    # Augmented exponential gives B=integral exp(A t)dt; its Frechet derivative gives B_R.
    k = np.zeros((6,6), dtype=np.complex128)
    kr = np.zeros((6,6), dtype=np.complex128)
    k[:3,:3] = a
    k[:3,3:] = np.eye(3)
    kr[:3,:3] = ar
    ek, ekr = expm_frechet(k*dt, kr*dt, compute_expm=True)
    b = ek[:3,3:]
    br = ekr[:3,3:]
    vals = _oracle_spin_boson_potential(rmid, p)
    hr = gen[21:24].real
    hrr = np.zeros(3)
    u2 = (q @ u).real
    force_u = -0.5*(hr @ b).real
    force_r = -dt*vals[8] - 0.5*np.dot(hrr, b@u).real - 0.5*np.dot(hr, br@u).real
    c = np.eye(5)
    c[1,0] = force_r
    c[1,2:] = force_u
    c[2:,0] = (qr@u).real
    c[2:,2:] = q.real

    d2 = np.eye(5)
    d2[0,1] = 0.5*dt/mass
    tangent_x = d2 @ c @ d1

    # Canonical-to-spin and spin-to-canonical coordinate Jacobians.
    cin = np.zeros((5,4))
    cin[0,0] = 1.0
    cin[1,2] = 1.0
    cin[2:,1] = du_dphi
    cin[2:,3] = du_ds
    ux, uy = u2[0], u2[1]
    denom = ux*ux+uy*uy
    if denom <= 64.0*np.finfo(float).eps:
        raise ValueError("azimuth is singular at a spin pole")
    cout = np.zeros((4,5))
    cout[0,0] = 1.0
    cout[1,2] = -uy/denom
    cout[1,3] = ux/denom
    cout[2,1] = 1.0
    cout[3,4] = 0.5
    mstep = cout @ tangent_x @ cin
    return np.concatenate((znew, mstep.ravel())).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [
        {'setup':'import numpy as np\ndef angle_key(x):\n x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\np_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy();z_sol=np.array([-.63,.91,.77,-.18]);z_ref=z_sol.copy()', 'call':'angle_key(canonical_tangent_step(z_sol,.0375,p_sol))', 'gold_call':'angle_key(_oracle_canonical_tangent_step(z_ref,.0375,p_ref))'},
        {'setup':'import numpy as np\ndef angle_key(x):\n x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\np_sol=np.array([2.,1.1,-.2,-.4,.3,.8]);p_ref=p_sol.copy();z_sol=np.array([0.,-1.2,-.4,.15]);z_ref=z_sol.copy()', 'call':'angle_key(canonical_tangent_step(z_sol,-.08,p_sol))', 'gold_call':'angle_key(_oracle_canonical_tangent_step(z_ref,-.08,p_ref))'},
        {'setup':'import numpy as np\ndef angle_key(x):\n x=np.asarray(x);return np.concatenate((x[:1],[np.cos(x[1]),np.sin(x[1])],x[2:]))\np_sol=np.array([1.,.7,.1,.5,.2,1.]);p_ref=p_sol.copy();z_sol=np.array([1.2,2.8,.3,-.4]);z_ref=z_sol.copy()', 'call':'angle_key(canonical_tangent_step(z_sol,.2,p_sol))', 'gold_call':'angle_key(_oracle_canonical_tangent_step(z_ref,.2,p_ref))'},
    ]
