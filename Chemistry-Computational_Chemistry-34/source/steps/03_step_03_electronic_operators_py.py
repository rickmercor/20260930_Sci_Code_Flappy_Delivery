"""
Calculate the electronic propagation operators for a fixed nuclear position and nonzero propagation interval.

Return the ordered electronic-generator spectrum, Cartesian spin propagation operator, and time-integrated operator in the declared order. Use a numerically stable evaluation across the source method's zero-mode limit.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def electronic_operators(position: float, timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    """Return ordered eigenvalues, electronic Q, and integrated B operators.

    Parameters
    ----------
    position : float
        Finite nuclear position.
    timestep : float
        Finite nonzero propagation time.
    parameters : np.ndarray
        Spin-boson parameter vector defined in Step 1.

    Returns
    -------
    result : np.ndarray
        Complex128 vector [eigenvalues(3), Q.ravel()(9), B.ravel()(9)], where
        Q=exp(-i W timestep) and B=integral_0^timestep exp(-i W t) dt. Here B is
        the time-integrated electronic propagator used in the momentum update.

    Raises
    ------
    ValueError
        If timestep is zero or nonfinite, or upstream validation fails.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_electronic_operators(position: float, timestep: float, parameters: "np.ndarray") -> "np.ndarray":
    timestep = float(timestep)
    if not np.isfinite(timestep) or timestep == 0.0:
        raise ValueError("timestep must be finite and nonzero")
    data = _oracle_spin_generator(position, parameters)
    w = data[:9].reshape(3,3)
    eigvals, sw = np.linalg.eigh(w)
    phase = np.exp(-1j*eigvals*timestep)
    ups = np.empty(3, dtype=np.complex128)
    for i, lam in enumerate(eigvals):
        if lam == 0.0:
            ups[i] = timestep
        else:
            ups[i] = 1j*np.expm1(-1j*lam*timestep)/lam
    q = (sw*phase) @ sw.conj().T
    b = (sw*ups) @ sw.conj().T
    q = np.real_if_close(q, tol=1000).astype(np.complex128)
    b = np.real_if_close(b, tol=1000).astype(np.complex128)
    return np.concatenate((eigvals.astype(np.complex128), q.ravel(), b.ravel()))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    return [
        {'setup':'import numpy as np\np_sol=np.array([1.3,.85,.35,.72,.41,np.sqrt(3)/2]);p_ref=p_sol.copy()', 'call':'electronic_operators(-.63,.0375,p_sol)', 'gold_call':'_oracle_electronic_operators(-.63,.0375,p_ref)'},
        {'setup':'import numpy as np\np_sol=np.array([2.,1.1,-.2,-.4,.3,.8]);p_ref=p_sol.copy()', 'call':'electronic_operators(0.,-.08,p_sol)', 'gold_call':'_oracle_electronic_operators(0.,-.08,p_ref)'},
        {'setup':'import numpy as np\np_sol=np.array([1.,.7,.1,.5,.2,1.]);p_ref=p_sol.copy()', 'call':'electronic_operators(1.2,.2,p_sol)', 'gold_call':'_oracle_electronic_operators(1.2,.2,p_ref)'},
        {'setup':'import numpy as np\np_sol=np.array([1.,1.,0.,0.,1e-16,1.]);p_ref=p_sol.copy()', 'call':'electronic_operators(0.,1e16,p_sol)', 'gold_call':'_oracle_electronic_operators(0.,1e16,p_ref)'},
    ]
