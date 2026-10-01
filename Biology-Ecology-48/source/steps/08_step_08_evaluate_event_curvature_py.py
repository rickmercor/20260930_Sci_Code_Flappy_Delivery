"""
Orchestrate all model components for the benchmark mixed curvature.

This calculation applies one event-indexed model to the supplied trajectory and fixed state parameters. Known turns determine unequal durations, speeds, signed angles, arrival distances, and cyclic clock hours. The carved angular law and two gamma laws determine state-dependent emissions; distance and time of day determine transitions between successive turns.



The forward likelihood marginalizes over behavioural states. The final sensitivity couples state 1's angular concentration with the cyclic state-1-to-state-2 transition coefficient, so emissions, transitions, normalization factors, and derivatives must all use the same event order. The initial distribution is fixed and independent of both differentiated parameters. The requested result is the scalar

\[

\left.

\frac{\partial^2\log L}

{\partial\kappa_1\,\partial a_2^{(12)}}

\right|_{\text{given trajectory and parameters}}.

\]

Returns
-------
float, the final benchmark mixed marginal log-likelihood derivative at the supplied trajectory and fixed parameters.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_event_curvature(points: "np.ndarray", times: "np.ndarray", heading: "np.ndarray", centre: "np.ndarray", kappa: "np.ndarray", width: "np.ndarray", time_shape: "np.ndarray", time_rate: "np.ndarray", speed_shape: "np.ndarray", speed_rate: "np.ndarray", coefficients: "np.ndarray", initial: "np.ndarray") -> float:
    """Return the fixed-parameter mixed curvature of the event-indexed HMM.
 
    Parameters
    ----------
    points, times, heading, centre : np.ndarray
        Geometric event inputs as specified in construct_event_steps.
    kappa, width : np.ndarray
        Carved turn parameters as specified in carved_turn_log_density.
    time_shape, time_rate, speed_shape, speed_rate : np.ndarray
        Gamma emission shape-rate parameters for durations and speeds.
    coefficients : np.ndarray
        (2,4) off-diagonal transition coefficients.
    initial : np.ndarray
        Fixed (2,) state distribution, independent of coefficients.
 
    Returns
    -------
    mixed_derivative : float
        d^2 log L / d kappa[0] d coefficients[0,2], within 1e-6.
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
 
def _oracle_evaluate_event_curvature(points: "np.ndarray", times: "np.ndarray", heading: "np.ndarray", centre: "np.ndarray", kappa: "np.ndarray", width: "np.ndarray", time_shape: "np.ndarray", time_rate: "np.ndarray", speed_shape: "np.ndarray", speed_rate: "np.ndarray", coefficients: "np.ndarray", initial: "np.ndarray") -> float:
    dt,s,phi,d,h=_oracle_construct_event_steps(points,times,heading,centre)
    angle,score=_oracle_carved_turn_log_density(phi,kappa,width)
    gamma=_oracle_gamma_step_log_density(dt,s,time_shape,time_rate,speed_shape,speed_rate)
    G,D=_oracle_event_transition_matrices(d,h,coefficients)
    joint=_oracle_joint_event_emissions(angle,gamma)
    _oracle_scaled_event_forward(joint,G,initial)
    return _oracle_mixed_event_curvature(joint,score,G,D,initial)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three distinct event model cases."""
    return [
        {"setup": 'import numpy as np\np=np.array([[0.,0.],[.5,.1],[1.,.7],[1.4,1.],[2.2,.3]]);t=np.array([22.8,23.1,23.55,24.1,24.6]);h=np.array([1.,0.]);c=np.array([.5,.1]);k=np.array([2.4,5.7]);w=np.array([1.3,.42]);at=np.array([2.2,4.1]);bt=np.array([3.2,4.7]);as_=np.array([3.1,5.3]);bs=np.array([2.1,3.7]);b=np.array([[-.9,.23,1.1,-.55],[.2,-.18,-.4,.35]]);delta=np.array([.55,.45])', "call": 'evaluate_event_curvature(p,t,h,c,k,w,at,bt,as_,bs,b,delta)', "gold_call": '_oracle_evaluate_event_curvature(p,t,h,c,k,w,at,bt,as_,bs,b,delta)', "tol": 1e-06},
        {"setup": 'import numpy as np\np=np.array([[0.,0.],[1.,0.],[2.,.05],[3.,.3]]);t=np.array([0.,.2,.7,1.2]);h=np.array([1.,.2]);c=np.array([0.,0.]);k=np.array([1.,3.]);w=np.array([.1,.4]);at=np.array([1.,3.]);bt=np.array([2.,4.]);as_=np.array([2.,4.]);bs=np.array([1.,3.]);b=np.zeros((2,4));delta=np.array([.5,.5])', "call": 'evaluate_event_curvature(p,t,h,c,k,w,at,bt,as_,bs,b,delta)', "gold_call": '_oracle_evaluate_event_curvature(p,t,h,c,k,w,at,bt,as_,bs,b,delta)', "tol": 1e-06},
        {"setup": 'import numpy as np\np=np.array([[0.,0.],[.4,.2],[.7,.8],[1.5,.5],[1.7,1.2],[2.,1.3]]);t=np.array([5.,5.3,5.65,6.1,6.8,7.]);h=np.array([1.,.2]);c=np.array([.2,1.]);k=np.array([2.,6.]);w=np.array([1.,.01]);at=np.array([3.,4.]);bt=np.array([4.,5.]);as_=np.array([5.,2.]);bs=np.array([4.,2.]);b=np.array([[-1.,.2,1.,.4],[.2,-.1,-1.,.2]]);delta=np.array([.2,.8])', "call": 'evaluate_event_curvature(p,t,h,c,k,w,at,bt,as_,bs,b,delta)', "gold_call": '_oracle_evaluate_event_curvature(p,t,h,c,k,w,at,bt,as_,bs,b,delta)', "tol": 1e-06},
    ]
