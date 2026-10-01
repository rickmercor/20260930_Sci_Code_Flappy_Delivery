"""
Evaluate independent gamma time and speed emissions in event coordinates.

A movement event has an elapsed duration \(\tau\) and a travel speed \(s\) during its straight step. In the source's event-based formulation, each has a state-dependent gamma distribution. Shape and rate specify each gamma law; the rate is the inverse of scale. For positive \(x\),

\[

g(x\mid q,r)

=

\frac{r^q}{\Gamma(q)}x^{q-1}e^{-rx},

\]

where \(q\) is shape and \(r\) is rate.



The observation is \((\tau,s)\), with speed computed as segment length divided by duration. Conditional on hidden state \(j\), the duration and speed factors are independent:

\[

f_j(\tau,s)=g(\tau\mid q_{\tau,j},r_{\tau,j})

             g(s\mid q_{s,j},r_{s,j}).

\]

The density is evaluated in observed speed coordinates, so no length-to-speed Jacobian enters. If length were the density coordinate instead, transforming the speed density would introduce a factor \(1/\tau\).

Returns
-------
np.ndarray of shape (n, 2), the state-specific sum of gamma duration and gamma speed log densities.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def gamma_step_log_density(durations: "np.ndarray", speeds: "np.ndarray", time_shape: "np.ndarray", time_rate: "np.ndarray", speed_shape: "np.ndarray", speed_rate: "np.ndarray") -> "np.ndarray":
    """Return state-dependent log gamma densities for observed durations and speeds.
 
    Shape-rate gamma density at x>0 is rate**shape*x**(shape-1)*exp(-rate*x)/Gamma(shape).
    The observed variable is speed, not length; no length-to-speed Jacobian is required.
 
    Parameters
    ----------
    durations, speeds : np.ndarray
        Positive finite (n,) values in hours and coordinate units per hour.
    time_shape, time_rate, speed_shape, speed_rate : np.ndarray
        Positive finite (2,) state-specific shape and rate parameters.
 
    Returns
    -------
    log_density : np.ndarray
        (n,2) sum of independent time and speed log densities.
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
from scipy.special import gammaln
 
def _oracle_gamma_step_log_density(durations: "np.ndarray", speeds: "np.ndarray", time_shape: "np.ndarray", time_rate: "np.ndarray", speed_shape: "np.ndarray", speed_rate: "np.ndarray") -> "np.ndarray":
    dt=np.asarray(durations,float); s=np.asarray(speeds,float)
    arrays=[np.asarray(a,float) for a in (time_shape,time_rate,speed_shape,speed_rate)]
    if dt.ndim!=1 or s.shape!=dt.shape or not all(np.all(np.isfinite(a)) and np.all(a>0) for a in [dt,s]+arrays) or any(a.shape!=(2,) for a in arrays):
        raise ValueError("gamma observations and parameters must be positive")
    at,bt,as_,bs=arrays
    return (at*np.log(bt)-gammaln(at)+(at-1)*np.log(dt[:,None])-bt*dt[:,None]
            +as_*np.log(bs)-gammaln(as_)+(as_-1)*np.log(s[:,None])-bs*s[:,None])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three distinct event model cases."""
    return [
        {"setup": 'import numpy as np\ndt=np.array([.2,.8,2.]);s=np.array([.4,3.,2.]);at=np.array([2.,4.]);bt=np.array([3.,5.]);as_=np.array([3.,6.]);bs=np.array([2.,4.])', "call": 'gamma_step_log_density(dt,s,at,bt,as_,bs)', "gold_call": '_oracle_gamma_step_log_density(dt,s,at,bt,as_,bs)', "tol": 1e-09},
        {"setup": 'import numpy as np\ndt=np.array([.0001,.3]);s=np.array([.1,10.]);at=np.ones(2);bt=np.ones(2);as_=np.ones(2);bs=np.ones(2)', "call": 'gamma_step_log_density(dt,s,at,bt,as_,bs)', "gold_call": '_oracle_gamma_step_log_density(dt,s,at,bt,as_,bs)', "tol": 1e-09},
        {"setup": 'import numpy as np\ndt=np.array([3.,4.]);s=np.array([5.,.05]);at=np.array([.4,9.]);bt=np.array([1.,2.]);as_=np.array([.5,4.]);bs=np.array([.1,10.])', "call": 'gamma_step_log_density(dt,s,at,bt,as_,bs)', "gold_call": '_oracle_gamma_step_log_density(dt,s,at,bt,as_,bs)', "tol": 1e-09},
    ]
