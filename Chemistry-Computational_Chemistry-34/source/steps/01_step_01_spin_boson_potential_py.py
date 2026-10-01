"""
Evaluate the fixed two-state diabatic potential and its first and second nuclear derivatives.

Evaluate the one-dimensional two-state spin-boson potential and derivatives.

The paper's benchmark separates a harmonic state-independent potential from a traceless
two-state diabatic matrix with constant real coupling.

Returns
-------
return values
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spin_boson_potential(position: float, parameters: "np.ndarray") -> "np.ndarray":
    """Return potential values and their first two nuclear derivatives.

    Parameters
    ----------
    position : float
        Finite nuclear position R.
    parameters : np.ndarray
        Six values [mass, omega, alpha, kappa, delta, spin_radius], with positive
        mass, omega, delta, and spin_radius and finite alpha and kappa.

    Returns
    -------
    values : np.ndarray
        Twelve float64 values [V0,V1,V2,Delta,V0_R,V1_R,V2_R,Delta_R,
        V0_RR,V1_RR,V2_RR,Delta_RR].

    Raises
    ------
    ValueError
        If inputs violate the stated shape, finiteness, or positivity contract.
    """
    return values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_spin_boson_potential(position: float, parameters: "np.ndarray") -> "np.ndarray":
    position = float(position)
    p = np.asarray(parameters, dtype=np.float64)
    if not np.isfinite(position) or p.shape != (6,) or not np.all(np.isfinite(p)):
        raise ValueError("position and six parameters must be finite")
    mass, omega, alpha, kappa, delta, spin_radius = p
    if mass <= 0.0 or omega <= 0.0 or delta <= 0.0 or spin_radius <= 0.0:
        raise ValueError("mass, omega, delta, and spin_radius must be positive")
    v0 = 0.5 * mass * omega * omega * position * position
    v1 = alpha + kappa * position
    v2 = -v1
    return np.array([v0, v1, v2, delta,
                     mass*omega*omega*position, kappa, -kappa, 0.0,
                     mass*omega*omega, 0.0, 0.0, 0.0], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [
        {'setup':'import numpy as np\np_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy()', 'call':'spin_boson_potential(-.63,p_sol)', 'gold_call':'_oracle_spin_boson_potential(-.63,p_ref)'},
        {'setup':'import numpy as np\np_sol=np.array([2.,1.1,-.2,-.4,.3,.8]);p_ref=p_sol.copy()', 'call':'spin_boson_potential(0.,p_sol)', 'gold_call':'_oracle_spin_boson_potential(0.,p_ref)'},
        {'setup':'import numpy as np\np_sol=np.array([1.,.7,.1,.5,.2,1.]);p_ref=p_sol.copy()', 'call':'spin_boson_potential(1.2,p_sol)', 'gold_call':'_oracle_spin_boson_potential(1.2,p_ref)'},
    ]
