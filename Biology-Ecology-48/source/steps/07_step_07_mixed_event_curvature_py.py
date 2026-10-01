"""
Differentiate the scaled hidden state forward recursion jointly in two parameters.

The benchmark varies two parameters with different local roles. State 1's carved-angle concentration \(\kappa_1\) changes that state's emission density. The state-1-to-state-2 cosine coefficient \(a_2^{(12)}\) changes transition probabilities at later turning events. Their local factors have no direct mixed derivative, but marginalization couples their effects through uncertainty about the hidden states.



The requested quantity is

\[

\frac{\partial^2\log L}

{\partial\kappa_1\,\partial a_2^{(12)}}.

\]

For \(x=\kappa_1\) and \(y=a_2^{(12)}\), its likelihood-level identity is

\[

\frac{\partial^2\log L}{\partial x\,\partial y}

=

\frac{L_{xy}}{L}

-

\frac{L_xL_y}{L^2}.

\]

The concentration score includes the derivative of the carved density normalizer. The transition derivative changes opposite entries in the first transition row, preserving a zero derivative for that row sum. Differentiating the scaled forward recursion propagates both sensitivities through normalization and hidden-state uncertainty.

Returns
-------
float, the mixed derivative of the marginal log likelihood with respect to kappa[0] and coefficients[0, 2].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mixed_event_curvature(log_emissions: "np.ndarray", concentration_score: "np.ndarray", transitions: "np.ndarray", transition_derivative: "np.ndarray", initial: "np.ndarray") -> float:
    """Return mixed log-likelihood derivative in kappa[0] and beta[0,2].
 
    Kappa[0] changes only the state-1 carved turning emission; beta[0,2]
    changes only the state-1-to-state-2 cyclic cosine transition logit.
    State labels, initial distribution and all other parameters are fixed.
    The derivative is of the marginal log likelihood, including uncertainty
    in hidden states. A result within 1e-6 absolute accuracy is required.
 
    Parameters
    ----------
    log_emissions, concentration_score : np.ndarray
        Matching (n,2) log emission and concentration-score arrays, n >= 2.
    transitions, transition_derivative : np.ndarray
        Matching (n,2,2) transition and derivative arrays.
    initial : np.ndarray
        Positive fixed (2,) probability vector summing to one.
 
    Returns
    -------
    mixed_derivative : float
        d^2 log L / d kappa[0] d beta[0,2].
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
 
def _oracle_mixed_event_curvature(log_emissions: "np.ndarray", concentration_score: "np.ndarray", transitions: "np.ndarray", transition_derivative: "np.ndarray", initial: "np.ndarray") -> float:
    L=np.asarray(log_emissions,float);S=np.asarray(concentration_score,float)
    G=np.asarray(transitions,float);D=np.asarray(transition_derivative,float);delta=np.asarray(initial,float)
    if L.ndim!=2 or L.shape[1]!=2 or len(L)<2 or S.shape!=L.shape or G.shape!=(len(L),2,2) or D.shape!=G.shape or not np.all(np.isfinite(S)) or not np.all(np.isfinite(D)) or delta.shape!=(2,) or not np.all(delta>0):
        raise ValueError("invalid mixed curvature inputs")
    # Only the first state's emission depends on concentration kappa[0].
    S=S.copy(); S[:,1]=0.
    f=delta.copy(); fk=np.zeros(2);fb=np.zeros(2);fkb=np.zeros(2);ans=0.
    for t in range(len(L)):
        if t:
            oldf,oldk,oldb,oldkb=f,fk,fb,fkb
            f=oldf@G[t];fk=oldk@G[t]
            fb=oldb@G[t]+oldf@D[t]
            fkb=oldkb@G[t]+oldk@D[t]
        m=np.max(L[t]);e=np.exp(L[t]-m);ek=e*S[t]
        r=f*e;rk=fk*e+f*ek;rb=fb*e;rkb=fkb*e+fb*ek
        c=r.sum();ck=rk.sum();cb=rb.sum();ckb=rkb.sum()
        if c<=0:raise ValueError("zero event likelihood")
        ans+=ckb/c-ck*cb/c**2
        fn=r/c;fkn=(rk-fn*ck)/c;fbn=(rb-fn*cb)/c
        fkbn=(rkb-fbn*ck-fkn*cb-fn*ckb)/c
        f,fk,fb,fkb=fn,fkn,fbn,fkbn
    return float(ans)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three distinct event model cases."""
    return [
        {"setup": 'import numpy as np\nL=np.array([[-1.,-2.],[-.5,-1.],[-1.3,-.2]]);S=np.array([[.2,.3],[-.1,.5],[.7,-.1]]);G=np.tile(np.array([[.8,.2],[.3,.7]]),(3,1,1));D=np.zeros_like(G);D[:,0,0]=[-.02,-.1,.03];D[:,0,1]=-D[:,0,0];delta=np.array([.5,.5])', "call": 'mixed_event_curvature(L,S,G,D,delta)', "gold_call": '_oracle_mixed_event_curvature(L,S,G,D,delta)', "tol": 1e-06},
        {"setup": 'import numpy as np\nL=np.array([[-.1,-1.],[-.5,-2.]]);S=np.array([[.7,.2],[.2,.1]]);G=np.tile(np.array([[.5,.5],[.4,.6]]),(2,1,1));D=np.zeros_like(G);delta=np.array([.7,.3])', "call": 'mixed_event_curvature(L,S,G,D,delta)', "gold_call": '_oracle_mixed_event_curvature(L,S,G,D,delta)', "tol": 1e-06},
        {"setup": 'import numpy as np\nL=np.array([[-500.,-501.],[-600.,-590.],[-200.,-210.],[-300.,-299.]]);S=np.array([[.2,.1],[.4,.7],[-.1,.2],[.5,.1]]);G=np.tile(np.array([[.1,.9],[.8,.2]]),(4,1,1));D=np.zeros_like(G);D[:,0,1]=[.03,-.06,.08,-.02];D[:,0,0]=-D[:,0,1];delta=np.array([.35,.65])', "call": 'mixed_event_curvature(L,S,G,D,delta)', "gold_call": '_oracle_mixed_event_curvature(L,S,G,D,delta)', "tol": 1e-06},
    ]
