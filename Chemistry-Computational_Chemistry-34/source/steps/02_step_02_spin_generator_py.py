"""
Construct the two-state electronic generator and its nuclear derivative for the supplied spin-boson instance.

The electronic generator represents the instantaneous evolution of the Cartesian spin. Return its matrix, derivative, and associated Hamiltonian data in the declared order, recovering their source-defined conventions from the selected method.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def spin_generator(position: float, parameters: "np.ndarray") -> "np.ndarray":
    """Return W, its first derivative, and the spin Hamiltonian data.

    Parameters and validation are as in spin_boson_potential.

    Returns
    -------
    result : np.ndarray
        A complex128 vector containing W.ravel(), W_R.ravel(), H, and H_R,
        in that order (24 entries total).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_spin_generator(position: float, parameters: "np.ndarray") -> "np.ndarray":
    vals = _oracle_spin_boson_potential(position, parameters)
    _, v1, v2, delta, _, v1r, v2r, deltar, *_ = vals
    h = np.array([2.0*delta, 0.0, v1-v2], dtype=np.float64)
    hr = np.array([2.0*deltar, 0.0, v1r-v2r], dtype=np.float64)
    hx, hy, hz = h
    w = 1j*np.array([[0.0,-hz,hy],[hz,0.0,-hx],[-hy,hx,0.0]], dtype=np.complex128)
    hxr, hyr, hzr = hr
    wr = 1j*np.array([[0.0,-hzr,hyr],[hzr,0.0,-hxr],[-hyr,hxr,0.0]], dtype=np.complex128)
    return np.concatenate((w.ravel(), wr.ravel(), h.astype(np.complex128), hr.astype(np.complex128)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [
        {'setup':'import numpy as np\np_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy()', 'call':'spin_generator(-.63,p_sol)', 'gold_call':'_oracle_spin_generator(-.63,p_ref)'},
        {'setup':'import numpy as np\np_sol=np.array([2.,1.1,-.2,-.4,.3,.8]);p_ref=p_sol.copy()', 'call':'spin_generator(0.,p_sol)', 'gold_call':'_oracle_spin_generator(0.,p_ref)'},
        {'setup':'import numpy as np\np_sol=np.array([1.,.7,.1,.5,.2,1.]);p_ref=p_sol.copy()', 'call':'spin_generator(1.2,p_sol)', 'gold_call':'_oracle_spin_generator(1.2,p_ref)'},
    ]
