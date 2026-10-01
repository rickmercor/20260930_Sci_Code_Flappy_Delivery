"""
Combine carved angle and gamma time-speed log emissions.

At each real turn, the observation consists of a signed turning angle \(\phi_t\), duration \(\tau_t\), and straight-step speed \(s_t\). Conditional on hidden behavioural state \(j\), the source factorizes their joint emission:

\[

b_j(\phi_t,\tau_t,s_t)

=

f_j(\phi_t)\,

g_j(\tau_t)\,

g_j(s_t),

\qquad

\log b_j

=

\log f_j(\phi_t)+\log g_j(\tau_t)+\log g_j(s_t).

\]

This conditional independence concerns the three coordinates at one event; neighbouring events remain dependent after hidden states are marginalized.



An exactly straight turn has zero carved angular density and therefore log density \(-\infty\). Adding component log densities avoids directly multiplying small density values. The density uses observed speed, rather than length, as its third coordinate.

Returns
-------
np.ndarray of shape (n, 2), the joint angle-duration-speed emission log densities for each event and state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def joint_event_emissions(angle_log: "np.ndarray", gamma_log: "np.ndarray") -> "np.ndarray":
    """Form the conditional independent joint event emission in log coordinates.
 
    Parameters
    ----------
    angle_log, gamma_log : np.ndarray
        Matching (n,2) state-specific log densities, n >= 2.
 
    Returns
    -------
    joint_log : np.ndarray
        (n,2) elementwise log density of angle, duration and speed.
    Raises
    ------
    ValueError
        If the specified input domain, shape or finite-value constraints are violated.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
def _oracle_joint_event_emissions(angle_log: "np.ndarray", gamma_log: "np.ndarray") -> "np.ndarray":
    a=np.asarray(angle_log,float); b=np.asarray(gamma_log,float)
    if a.ndim!=2 or a.shape[1]!=2 or a.shape[0]<2 or b.shape!=a.shape or np.any(np.isnan(a)) or not np.all(np.isfinite(b)):
        raise ValueError("emissions require matching n by two arrays")
    return a+b

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three distinct event model cases."""
    return [
        {"setup": 'import numpy as np\na=np.array([[-1.,-3.],[-2.,-1.]]);b=np.array([[-.4,-.6],[-1.2,-3.]])', "call": 'joint_event_emissions(a,b)', "gold_call": '_oracle_joint_event_emissions(a,b)', "tol": 1e-09},
        {"setup": 'import numpy as np\na=np.array([[-np.inf,-2.],[-3.,-np.inf]]);b=np.zeros((2,2))', "call": 'joint_event_emissions(a,b)', "gold_call": '_oracle_joint_event_emissions(a,b)', "tol": 1e-09},
        {"setup": 'import numpy as np\na=np.array([[-1000.,-1005.],[-1200.,-1203.],[-40.,-50.]]);b=np.array([[-.3,-1.],[-2.,-5.],[-1.,-.2]])', "call": 'joint_event_emissions(a,b)', "gold_call": '_oracle_joint_event_emissions(a,b)', "tol": 1e-09},
    ]
