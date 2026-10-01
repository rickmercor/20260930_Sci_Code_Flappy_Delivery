"""
Evaluate arrival-event multinomial-logit transitions and a cyclic coefficient derivative.

Behavioural switching occurs at turning decisions. Its probability can depend on distance \(d_t\) from the centre and arrival clock hour \(H_t\). The source uses a reference-category logit: within each originating-state row, the self-transition predictor is zero. For origin state \(i\) and the other state \(j\),

\[

u_t^{(ij)}

=

a_0^{(ij)}+a_1^{(ij)}d_t

+a_2^{(ij)}\cos\!\left(\frac{\pi H_t}{12}\right)

+a_3^{(ij)}\sin\!\left(\frac{\pi H_t}{12}\right),

\qquad

\Gamma_t(i,j)=\frac{e^{u_t^{(ij)}}}{1+e^{u_t^{(ij)}}}.

\]

The corresponding self-transition probability is

\(\Gamma_t(i,i)=1-\Gamma_t(i,j)\).



The matrix \(\Gamma_t\) maps the state at event \(t-1\) to the state at event \(t\), using the arrival event's covariates. The fixed initial distribution applies to the first emission without a preceding transition. Differentiation with respect to the state-1-to-state-2 cosine coefficient changes only the first transition row; the derivatives of its two entries sum to zero. Arrival hour modulo \(24\) supplies the cyclic clock covariate across midnight.

Returns
-------
Tuple of two np.ndarray tensors, each shape (n, 2, 2): row-stochastic arrival-event transition matrices and their derivatives with respect to coefficients[0, 2].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def event_transition_matrices(distances: "np.ndarray", hours: "np.ndarray", coefficients: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    """Compute event-indexed transitions and derivative for coefficient beta[0,2].
 
    The two off-diagonal logits are beta[i,0]+beta[i,1]*distance+
    beta[i,2]*cos(pi*hour/12)+beta[i,3]*sin(pi*hour/12).
    The diagonal logit in each row is zero. Transition at index t takes
    state t-1 to state t and uses the arrival event's covariates; index 0
    is returned but is not applied before the first emission.
 
    Parameters
    ----------
    distances, hours : np.ndarray
        Finite (n,) arrival-event covariates, with nonnegative distance.
    coefficients : np.ndarray
        Finite (2,4) logits, rows from origin states 1 and 2.
 
    Returns
    -------
    transitions, derivative : tuple[np.ndarray, np.ndarray]
        Each (n,2,2), row stochastic for transitions. Derivative is with
        respect to the state-1-to-state-2 cosine coefficient only.
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
from scipy.special import expit
 
def _oracle_event_transition_matrices(distances: "np.ndarray", hours: "np.ndarray", coefficients: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    d=np.asarray(distances,float); h=np.asarray(hours,float); beta=np.asarray(coefficients,float)
    if d.ndim!=1 or h.shape!=d.shape or beta.shape!=(2,4) or not all(np.all(np.isfinite(a)) for a in (d,h,beta)) or np.any(d<0):
        raise ValueError("invalid event covariates or coefficients")
    cs=np.cos(np.pi*h/12); sn=np.sin(np.pi*h/12)
    u=beta[:,0]+d[:,None]*beta[:,1]+cs[:,None]*beta[:,2]+sn[:,None]*beta[:,3]
    p=expit(u); G=np.empty((len(d),2,2));G[:,0,0]=1-p[:,0];G[:,0,1]=p[:,0];G[:,1,0]=p[:,1];G[:,1,1]=1-p[:,1]
    D=np.zeros_like(G); q=p[:,0]*(1-p[:,0])*cs;D[:,0,0]=-q;D[:,0,1]=q
    return G,D

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three distinct event model cases."""
    return [
        {"setup": 'import numpy as np\nd=np.array([.2,2.,.7]);h=np.array([0.,6.,23.5]);b=np.array([[-1.,.5,1.,-.5],[.3,-.4,.2,.1]])', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(event_transition_matrices(d,h,b))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_event_transition_matrices(d,h,b))', "tol": 1e-09},
        {"setup": 'import numpy as np\nd=np.array([0.,1.]);h=np.array([6.,18.]);b=np.zeros((2,4))', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(event_transition_matrices(d,h,b))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_event_transition_matrices(d,h,b))', "tol": 1e-09},
        {"setup": 'import numpy as np\nd=np.array([1.,3.,.4]);h=np.array([23.9,0.,12.]);b=np.array([[5.,-.8,1.2,3.],[-4.,2.,-.2,-.5]])', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(event_transition_matrices(d,h,b))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_event_transition_matrices(d,h,b))', "tol": 1e-09},
    ]
