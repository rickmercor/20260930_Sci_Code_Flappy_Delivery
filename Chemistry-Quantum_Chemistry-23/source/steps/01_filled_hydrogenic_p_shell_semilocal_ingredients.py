"""
For a completely filled, spin-unpolarized hydrogenic p shell with principal quantum number 2, 3 or 4 and nuclear charge Z, evaluate the normalized radial function and derivative, the spherical density and derivative, the positive kinetic-energy density, and the semilocal ingredients needed by the later ePC steps.

A filled p shell contains six electrons. The addition theorem makes the total density spherical, while the angular part of the orbital gradients contributes to the positive kinetic-energy density. The reduced gradient s and iso-orbital indicator z=tau_W/tau provide the standard local variables used by the later meta-GGA steps.

Returns
-------
a numpy array of shape (8, len(r)) with rows R, dR/dr, n, dn/dr, tau, tau_W, s and z. Raise ``ValueError`` for invalid principal quantum number, nonpositive Z or radii.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def p_shell_semilocal_ingredients(n_principal: int, Z: float, r: "np.ndarray") -> "np.ndarray":
    """Return R, dR/dr, n, dn/dr, tau, tau_W, s and z on positive radii.

    Atomic units. ``n_principal`` must be 2, 3 or 4, ``Z`` must be positive,
    and every radius must be strictly positive. Evaluate normalized hydrogenic
    radial functions and their analytic radial derivatives for the requested shell.
    Where n>=1e-30 compute the requested standard semilocal ingredients and cap z at 1; otherwise set
    tau_W=s=z=0. Return rows [R,dR,n,dn,tau,tau_W,s,z].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _p_radial_polynomial(n_principal):
    if n_principal == 2:
        return 1.0/(2.0*np.sqrt(6.0)), [0.0,1.0]
    if n_principal == 3:
        return 8.0/(27.0*np.sqrt(6.0)), [0.0,1.0,-1.0/6.0]
    return np.sqrt(5.0)/(16.0*np.sqrt(3.0)), [0.0,1.0,-0.25,1.0/80.0]


def _oracle_p_shell_semilocal_ingredients(n_principal: int, Z: float, r: np.ndarray) -> np.ndarray:
    if n_principal not in (2,3,4):
        raise ValueError("n_principal must be 2, 3 or 4.")
    if Z <= 0.0:
        raise ValueError("Z must be positive.")
    r=np.asarray(r,dtype=float)
    if np.any(r<=0.0):
        raise ValueError("Radii must be positive.")
    c,coeffs=_p_radial_polynomial(n_principal)
    poly=np.polynomial.Polynomial(coeffs)
    x=Z*r; decay=np.exp(-x/n_principal)
    R=c*Z**1.5*poly(x)*decay
    dR=c*Z**2.5*(poly.deriv()(x)-poly(x)/n_principal)*decay
    n=3.0/(2.0*np.pi)*R**2
    dn=3.0/np.pi*R*dR
    tau=3.0/(4.0*np.pi)*(dR**2+2.0*R**2/r**2)
    keep=n>=1e-30
    tau_w=np.zeros_like(n); grad=np.zeros_like(n); z=np.zeros_like(n)
    nk,dnk,tauk=n[keep],dn[keep],tau[keep]
    tau_w[keep]=dnk**2/(8.0*nk)
    grad[keep]=np.abs(dnk)/(2.0*(3.0*np.pi**2)**(1.0/3.0)*nk**(4.0/3.0))
    z[keep]=np.minimum(tau_w[keep]/tauk,1.0)
    return np.array([R,dR,n,dn,tau,tau_w,grad,z])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    return [
        {"setup":"import numpy as np","call":"p_shell_semilocal_ingredients(2,7.0,np.array([0.05,0.2,0.4,0.8,1.5]))","gold_call":"_oracle_p_shell_semilocal_ingredients(2,7.0,np.array([0.05,0.2,0.4,0.8,1.5]))"},
        {"setup":"import numpy as np","call":"p_shell_semilocal_ingredients(3,5.0,np.array([0.1,0.6,1.1,1.19,1.3,3.0]))","gold_call":"_oracle_p_shell_semilocal_ingredients(3,5.0,np.array([0.1,0.6,1.1,1.19,1.3,3.0]))"},
        {"setup":"import numpy as np","call":"p_shell_semilocal_ingredients(4,8.0,np.array([0.1,0.5,1.0,1.9,4.0]))","gold_call":"_oracle_p_shell_semilocal_ingredients(4,8.0,np.array([0.1,0.5,1.0,1.9,4.0]))"},
        {"setup":"import numpy as np","call":"p_shell_semilocal_ingredients(2,9.5,np.array([1e-4,0.01,0.3,2.0,5.0]))","gold_call":"_oracle_p_shell_semilocal_ingredients(2,9.5,np.array([1e-4,0.01,0.3,2.0,5.0]))"},
        {"setup":"import numpy as np","call":"p_shell_semilocal_ingredients(4,4.5,np.array([0.3,1.2,3.5,7.0,12.0]))","gold_call":"_oracle_p_shell_semilocal_ingredients(4,4.5,np.array([0.3,1.2,3.5,7.0,12.0]))"},
    ]
