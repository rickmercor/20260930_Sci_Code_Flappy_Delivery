"""
Compute the stable event HMM log likelihood and filtering probabilities.

The event-indexed hidden Markov likelihood sums over all behavioural-state sequences. The fixed initial distribution \(\delta\) supplies the state probabilities for the first emission. Every later event contributes one transition evaluated at its arrival, followed by its emission. With \(b_t\) denoting the vector of state-specific emission densities, the unscaled forward quantities satisfy

\[

\alpha_0=\delta\odot b_0,

\qquad

\alpha_t=(\alpha_{t-1}\Gamma_t)\odot b_t

\quad (t\geq 1),

\qquad

L=\sum_j\alpha_{n-1,j}.

\]

Here \(\odot\) denotes elementwise multiplication. Unequal durations enter through their emission densities; transitions occur once per turn and are not multiplied by elapsed time.



Repeated unscaled multiplication can underflow. Normalizing the forward vector after each event produces filtered state probabilities; the logarithms of its successive normalization factors sum to \(\log L\). Filtering retains uncertainty among states, whereas a Viterbi path selects one state sequence.

Returns
-------
Tuple of float and np.ndarray of shape (n, 2): the marginal log likelihood and normalized filtered state probabilities.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scaled_event_forward(log_emissions: "np.ndarray", transitions: "np.ndarray", initial: "np.ndarray") -> "tuple[float, np.ndarray]":
    """Evaluate the event HMM forward likelihood at fixed parameters.
 
    Initial is fixed independent of model parameters; transition[t]
    connects events t-1 and t only for t >= 1. A row with at least one
    finite log emission is required. Return a finite log likelihood and
    normalized (n,2) filtering probabilities.
 
    Parameters
    ----------
    log_emissions : np.ndarray
        (n,2) finite or -inf log densities, n >= 2, each row having a finite entry.
    transitions : np.ndarray
        (n,2,2) nonnegative row-stochastic matrices.
    initial : np.ndarray
        Positive (2,) fixed probabilities summing to one.
 
    Returns
    -------
    log_likelihood, filters : tuple[float, np.ndarray]
        Log density and (n,2) state-filter distributions.
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
 
def _oracle_scaled_event_forward(log_emissions: "np.ndarray", transitions: "np.ndarray", initial: "np.ndarray") -> "tuple[float, np.ndarray]":
    L=np.asarray(log_emissions,float);G=np.asarray(transitions,float);delta=np.asarray(initial,float)
    if L.ndim!=2 or L.shape[1]!=2 or len(L)<2 or G.shape!=(len(L),2,2) or delta.shape!=(2,) or np.any(np.isnan(L)) or np.any(np.isposinf(L)) or not np.all(np.any(np.isfinite(L),axis=1)) or not np.all(np.isfinite(G)) or np.any(G<0) or not np.allclose(G.sum(axis=2),1,atol=1e-10) or np.any(delta<=0) or not np.allclose(delta.sum(),1):
        raise ValueError("invalid HMM inputs")
    out=np.empty_like(L); ll=0.;f=delta
    for t in range(len(L)):
        if t: f=f@G[t]
        m=np.max(L[t]);raw=f*np.exp(L[t]-m);scale=raw.sum()
        if scale<=0:raise ValueError("zero event likelihood")
        f=raw/scale;out[t]=f;ll+=m+np.log(scale)
    return float(ll),out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three distinct event model cases."""
    return [
        {"setup": 'import numpy as np\nL=np.array([[-1.,-2.],[-2.,-1.],[-.8,-2.]]);G=np.tile(np.array([[.8,.2],[.3,.7]]),(3,1,1));delta=np.array([.6,.4])', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(scaled_event_forward(L,G,delta))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_scaled_event_forward(L,G,delta))', "tol": 1e-09},
        {"setup": 'import numpy as np\nL=np.array([[-np.inf,-.4],[-2.,-np.inf]]);G=np.tile(np.array([[.9,.1],[.2,.8]]),(2,1,1));delta=np.array([.4,.6])', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(scaled_event_forward(L,G,delta))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_scaled_event_forward(L,G,delta))', "tol": 1e-09},
        {"setup": 'import numpy as np\nL=np.array([[-1000.,-1001.],[-1200.,-1205.],[-1300.,-1290.]]);G=np.tile(np.array([[.03,.97],[.95,.05]]),(3,1,1));delta=np.array([.2,.8])', "call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(scaled_event_forward(L,G,delta))', "gold_call": '(lambda r: np.concatenate([np.asarray(x,dtype=float).ravel() for x in r]))(_oracle_scaled_event_forward(L,G,delta))', "tol": 1e-09},
    ]
