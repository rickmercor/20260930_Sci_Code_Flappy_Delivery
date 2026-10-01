"""
Compute clustering curvature of a staged investment right at fixed jump count. At times[0],...,times[d-1] the firm may pay costs[i] to continue. At times[d] it receives max(S-Kterm,0). The continuous log increment over dt is normal with mean (mu-sigma**2/2)*dt and variance sigma**2*dt. The independent marked Queue-Hawkes component of step 2 runs over the entire horizon, without resetting activation at decision dates. Marks are uncompensated. Discount every time interval at rho. 

For each alpha near its nominal value, choose a single baseline arrival rate using matched_arrival_rate(times[-1],target,alpha,beta,q0). This global calibration is shared by all increments. Define V(alpha) using the finite analytic COS recursion, conditioned on every activation state 0,...,max_state at each date. Exact transition coefficients include excursions above the cutoff between dates. At any retained date, including terminal maturity, continuation/payoff is set to zero for states above the cutoff, without renormalization of the remaining states. Return d2 V/d alpha2. All other scalar parameters, date intervals, frequency counts and activation cutoff are held fixed. Differentiate the transition kernels, calibrated baseline and every moving exercise boundary analytically. Do not finite-difference prices or integrate intermediate payoffs numerically.

Use all seven preceding functions: matched_arrival_rate, queue_kernel_curvature, diffusion_discount_factor, call_payoff_cosine_coeffs, state_continuation_jet, queue_exercise_boundaries and queue_payoff_jet. Frequencies on each date are k*pi/(b-a), k=0,...,N-1. The terminal call coefficients are identical in every retained queue state and have zero alpha derivatives. At the initial state x=log(S0), use the usual half weight on frequency zero.

The final curvature combines global expected-count calibration, persistent activation, analytic payoff projections and boundary sensitivities.

Returns
-------
return 0.0
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def persistent_clustering_curvature(times=(.75,1.5,2.5),costs=(6.,10.),
        intervals=((1.2,8.4),(-.6,10.2),(-2.8,12.4)),N=(192,256,384),
        max_state=32,S0=100.,Kterm=105.,mu=.04,sigma=.22,rho=.08,
        target=2.4,alpha=.35,beta=.9,mu_J=.06,sigma_J=.18,q0=2):
    """Return one finite float: the second ordinary clustering derivative.

    times is a finite strictly increasing positive vector of length d+1,
    d>=1. costs is positive finite shape (d,). intervals is finite shape
    (d+1,2), each b>a. N is an integer vector (d+1,), every entry >=16.
    max_state is a nonnegative integer, with 0<=q0<=max_state. S0,Kterm
    and sigma are positive; mu,rho are finite. Other model inputs satisfy
    matched_arrival_rate and queue_kernel_curvature. log(S0) lies inside
    the first interval. All nominal continuation curves must satisfy the
    strict endpoint bracket and single positive-slope crossing of step 6.
    Raise ValueError for invalid inputs or a failed exercise bracket.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_persistent_clustering_curvature(times=(.75,1.5,2.5),costs=(6.,10.),
        intervals=((1.2,8.4),(-.6,10.2),(-2.8,12.4)),N=(192,256,384),
        max_state=32,S0=100.,Kterm=105.,mu=.04,sigma=.22,rho=.08,
        target=2.4,alpha=.35,beta=.9,mu_J=.06,sigma_J=.18,q0=2):
    np=__import__('numpy')
    t=np.asarray(times,float);c=np.asarray(costs,float);I=np.asarray(intervals,float);ns=np.asarray(N)
    if t.ndim!=1 or len(t)<2 or c.shape!=(len(t)-1,) or I.shape!=(len(t),2) or ns.shape!=t.shape:
        raise ValueError('configuration shapes')
    if not all(np.all(np.isfinite(z)) for z in [t,c,I,ns]) or np.any(np.diff(np.r_[0.,t])<=0) or np.any(c<=0) or np.any(I[:,1]<=I[:,0]) or np.any(ns<16) or np.any(ns!=np.floor(ns)):
        raise ValueError('configuration ranges')
    if not np.all(np.isfinite([S0,Kterm,mu,sigma,rho])) or min(S0,Kterm,sigma)<=0:
        raise ValueError('project inputs')
    if isinstance(max_state,(bool,np.bool_)) or not np.isscalar(max_state) or not np.isfinite(max_state) or int(max_state)!=max_state or max_state<0:
        raise ValueError('max_state')
    if isinstance(q0,(bool,np.bool_)) or not np.isscalar(q0) or not np.isfinite(q0) or int(q0)!=q0 or not 0<=q0<=max_state:
        raise ValueError('q0')
    if not I[0,0]<np.log(S0)<I[0,1]:raise ValueError('initial log state')
    ns=ns.astype(int);states=int(max_state)+1;q0=int(q0);dt=np.diff(np.r_[0.,t])
    lj=_oracle_matched_arrival_rate(t[-1],target,alpha,beta,q0)
    frequencies=[np.arange(n)*np.pi/(b-a) for n,(a,b) in zip(ns,I)]
    H=np.zeros((3,states,ns[-1]))
    H[0]=_oracle_call_payoff_cosine_coeffs(Kterm,*I[-1],ns[-1])
    for stage in range(len(c)-1,-1,-1):
        w=frequencies[stage+1]
        K=_oracle_queue_kernel_curvature(w,dt[stage+1],lj,alpha,beta,mu_J,sigma_J,max_state)
        D=_oracle_diffusion_discount_factor(w,dt[stage+1],mu,sigma,rho)
        C=_oracle_state_continuation_jet(K,H,D)
        roots=_oracle_queue_exercise_boundaries(C[0],w,I[stage+1,0],*I[stage],c[stage])
        H=_oracle_queue_payoff_jet(C,w,frequencies[stage],I[stage+1,0],*I[stage],roots,c[stage])
    w=frequencies[0]
    K=_oracle_queue_kernel_curvature(w,dt[0],lj,alpha,beta,mu_J,sigma_J,max_state)
    D=_oracle_diffusion_discount_factor(w,dt[0],mu,sigma,rho)
    C=_oracle_state_continuation_jet(K,H,D)
    prime=np.ones(len(w));prime[0]=.5
    return float(np.real(C[2,q0]*np.exp(1j*w*(np.log(S0)-I[0,0])))@prime)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit dictionaries for Studio differential-case preflight."""
    return [{'setup': 'kw={}',
      'call': 'persistent_clustering_curvature(**kw)',
      'gold_call': '_oracle_persistent_clustering_curvature(**kw)'},
     {'setup': "kw={'times': (0.8, 1.7), 'costs': (9.0,), 'intervals': ((1.0, 8.5), (-1.0, 11.5)), "
               "'N': (160, 256), 'max_state': 20, 'q0': 1, 'target': 1.6, 'alpha': 0.3, 'beta': 1.1, "
               "'mu_J': -0.04, 'sigma_J': 0.2}",
      'call': 'persistent_clustering_curvature(**kw)',
      'gold_call': '_oracle_persistent_clustering_curvature(**kw)'},
     {'setup': "kw={'times': (0.5, 1.2, 2.0, 3.0), 'costs': (3.0, 6.0, 10.0), 'intervals': ((1.8, "
               "8.8), (0.0, 10.5), (-1.8, 12.0), (-3.5, 13.5)), 'N': (128, 160, 224, 320), "
               "'max_state': 20, 'S0': 110.0, 'Kterm': 115.0, 'sigma': 0.25, 'rho': 0.06, 'alpha': "
               "0.2, 'beta': 0.8, 'q0': 0, 'target': 1.5, 'mu_J': 0.1}",
      'call': 'persistent_clustering_curvature(**kw)',
      'gold_call': '_oracle_persistent_clustering_curvature(**kw)'},
     {'setup': 'def check(f):\n'
               '    try:\n'
               '        f(times=(2.,1.),costs=(9.,),intervals=((1.,8.),(0.,10.)),N=(64,64))\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n',
      'call': 'check(persistent_clustering_curvature)',
      'gold_call': 'check(_oracle_persistent_clustering_curvature)'}]
