"""
Convert supplied turning points into irregular event-indexed observations

The locations are already classified as real turning points. A movement step is the straight displacement between successive points, so intervals between events need not be equal. For successive points \(P_t\) and \(P_{t+1}\), its duration and speed are

\[

\tau_t=T_{t+1}-T_t,

\qquad

s_t=\frac{\lVert P_{t+1}-P_t\rVert}{\tau_t}.

\]

The turning angle compares this displacement with the incoming direction. The supplied initial heading defines that comparison for the first step.



The hidden Markov model is indexed by arrivals at turning points rather than fixed clock samples. Distance to the reference centre and clock hour are evaluated at each arrival, because those covariates govern the behavioural transition at that turn. Durations use elapsed hours; cyclic time-of-day covariates use arrival time modulo \(24\). Signed angles are wrapped to \([-\pi,\pi)\), which fixes their representation at the branch cut.

Returns
-------
Five np.ndarray vectors of shape (n,): durations in hours, speeds, signed angles, arrival distances, and arrival clock hours, in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def construct_event_steps(points: "np.ndarray", times: "np.ndarray", initial_heading: "np.ndarray", centre: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Construct the known straight steps and their arrival-event covariates.
 
    Parameters
    ----------
    points : np.ndarray
        Finite (n+1,2) turning-point coordinates, with n >= 2 and positive segment lengths.
    times : np.ndarray
        Strictly increasing (n+1,) elapsed hours, in the same coordinate system as clock hour.
    initial_heading : np.ndarray
        Nonzero (2,) vector for the incoming direction at the first segment.
    centre : np.ndarray
        Finite (2,) reference location.
 
    Returns
    -------
    durations, speeds, angles, distances, hours : tuple[np.ndarray, ...]
        Five (n,) arrays. Angle is signed from incoming to current segment in [-pi,pi),
        distances are measured from each segment's arrival point to centre, and
        clock hours are arrival times modulo 24. Durations remain in hours.
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
 
def _oracle_construct_event_steps(points: "np.ndarray", times: "np.ndarray", initial_heading: "np.ndarray", centre: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    p=np.asarray(points,float); t=np.asarray(times,float)
    h=np.asarray(initial_heading,float); c=np.asarray(centre,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or t.shape!=(len(p),) or h.shape!=(2,) or c.shape!=(2,) or not all(np.all(np.isfinite(a)) for a in (p,t,h,c)):
        raise ValueError("invalid event input shapes or nonfinite values")
    v=np.diff(p,axis=0); lengths=np.linalg.norm(v,axis=1); dt=np.diff(t)
    if np.any(lengths<=0) or np.linalg.norm(h)==0 or np.any(dt<=0):
        raise ValueError("segments and heading must be nonzero; times increase")
    prev=np.vstack((h,v[:-1]))
    phi=np.arctan2(prev[:,0]*v[:,1]-prev[:,1]*v[:,0],np.sum(prev*v,axis=1))
    phi=(phi+np.pi)%(2*np.pi)-np.pi
    return dt,lengths/dt,phi,np.linalg.norm(p[1:]-c,axis=1),np.mod(t[1:],24.)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three distinct event model cases."""
    return [
        {"setup": 'import numpy as np\np=np.array([[0.,0.],[1.,0.],[1.5,1.],[2.5,.4]]);t=np.array([23.7,23.9,24.5,25.1]);h=np.array([1.,.2]);c=np.array([1.,0.])', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(construct_event_steps(p,t,h,c))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_construct_event_steps(p,t,h,c))', "tol": 1e-09},
        {"setup": 'import numpy as np\np=np.array([[0.,0.],[-1.,0.],[-2.,0.]]);t=np.array([0.,.5,1.]);h=np.array([1.,0.]);c=np.array([0.,0.])', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(construct_event_steps(p,t,h,c))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_construct_event_steps(p,t,h,c))', "tol": 1e-09},
        {"setup": 'import numpy as np\np=np.array([[0.,0.],[.2,.8],[.3,1.4],[.8,1.6]]);t=np.array([22.,23.,24.,26.]);h=np.array([0.,1.]);c=np.array([.1,.9])', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(construct_event_steps(p,t,h,c))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_construct_event_steps(p,t,h,c))', "tol": 1e-09},
    ]
