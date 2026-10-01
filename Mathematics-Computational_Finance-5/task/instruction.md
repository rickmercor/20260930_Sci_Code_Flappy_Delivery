# Clustering curvature of a staged investment right at fixed expected jump count

## Background

Compound investment rights combine several decisions to continue or abandon a project. When information arrivals are self-exciting, the current activation state affects future jump activity and remains relevant at later decisions.

An analytic COS representation allows the payoff at each decision to be projected without numerical quadrature. Holding expected jump count fixed while changing clustering also changes the baseline arrival rate. The sensitivity calculation must account for that calibration and for movement of the exercise boundaries.

## Problem

A project starts at S(0)=100. Its log value is the sum of an independent continuous component with increment mean (mu-sigma²/2) dt and variance sigma² dt, and an uncompensated sum of jump marks. Use mu=0.04 and sigma=0.22. Each jump mark is normal with mean 0.06 and standard deviation 0.18. Arrivals have rate lambda+alpha Q, increase the integer activation state Q by one, and add a mark. Expirations have rate beta Q and decrease Q by one without changing the project value. Q(0)=2, beta=0.9, and the nominal clustering rate is alpha=0.35. Activation persists throughout the entire horizon, including across investment decisions.

At times 0.75 and 1.5, the firm may pay 6 and 10, respectively, to continue. At time 2.5, the remaining right pays max(S(2.5)-105,0). Discount all cash flows at the separate project rate rho=0.08. For each alpha in a neighborhood of 0.35, recalibrate one constant baseline lambda(alpha) so that the expected total number of arrivals over [0,2.5] is exactly 2.4. Use that same baseline over every increment.

Determine the second ordinary derivative of the time-zero investment value with respect to alpha along this fixed-expected-count calibration curve. This measures the local curvature of investment value as clustering changes while expected jump activity is held fixed.

Use the analytic COS payoff projections. The task extends the scalar independent-increment recursion by conditioning on the activation state at each date. Use the Queue-Hawkes joint marked-queue transform to retain the dependence between terminal activation and accumulated jump marks. Derive the state-conditioned recursion and its parameter derivatives; the extension and sensitivity formulas are not claimed to be stated in the literature.

Fix the numerical calculation as follows:

- At times (0.75,1.5,2.5), use log-value intervals [(1.2,8.4),(-0.6,10.2),(-2.8,12.4)] and respectively (192,256,384) cosine terms. Frequencies on [a,b] are k*pi/(b-a), starting at k=0 with the usual half weight.
- Retain activation states 0 through 32 at each of these dates, including terminal maturity. Set the payoff or continuation for larger endpoint states to zero without renormalization. Use exact infinite-state transition coefficients between retained endpoints: paths that exceed 32 and return between dates must contribute.
- Find a separate numerical exercise boundary for every retained state and decision date. Check the strict endpoint straddle and solve the finite continuation curve's crossing with absolute and relative tolerances 1e-12. The nominal curves have one positive-slope crossing per state.
- Compute intermediate payoff projections and all parameter derivatives analytically. Account for the derivatives of lambda(alpha), the transition coefficients and the moving exercise boundaries. Do not use finite-difference prices or numerical quadrature of the intermediate payoffs. Hold all specified intervals, term counts and state cutoffs fixed when differentiating.

Report one finite scalar, the second derivative at alpha=0.35, to at least eight significant figures. In the reasoning, give lambda and its first two derivatives, the nominal price and first derivative, the exercise boundaries for activation state 2 at both decision dates, and the boundary-motion contribution to the second derivative of a payoff projection.

## Output format

Give the single numeric answer in <final_answer>...</final_answer>, immediately followed by <reasoning>...</reasoning> containing the scientific justification and numerical checkpoints.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_matched_arrival_rate

Goal
----
Calibrate immigration while preserving the expected total number of jumps.



The activation process starts at q0. Arrivals have intensity lambda+alpha*Q

and increase Q by one; expirations have intensity beta*Q and decrease Q by

one. Choose the constant baseline lambda(alpha) so E[N(T)-N(0)]=target,

where N counts arrivals and alpha is the clustering rate. Return lambda

and its first two ordinary derivatives with respect to alpha. Hold T,

target, beta and q0 fixed. Derive the expectation from these birth and

death dynamics, and evaluate the derivatives analytically.

```python
import numpy as np

def matched_arrival_rate(T,target,alpha,beta,q0):
    """Return float shape (3,): lambda, lambda prime, lambda double-prime.

    T and target are positive finite scalars; beta > alpha > 0. q0 is a
    nonnegative integer. The calibrated baseline must be strictly positive.
    Raise ValueError for nonfinite inputs, invalid ranges or infeasible target.
    """
    return np.zeros(3)
```

### Step 2

02_queue_kernel_curvature

Goal
----
Differentiate the persistent marked-queue transition law twice.



Q increases by one at rate lambda(alpha)+alpha*Q and decreases by one at

rate beta*Q. Every increase adds an independent N(mu_J,sigma_J**2) mark to

the uncompensated sum M. The initial state q is fixed. lambda_jet contains

lambda, d lambda/d alpha, and d2 lambda/d alpha2 at the nominal alpha.

The remaining model parameters and the time interval are held fixed.



For q,j=0,...,max_state, and each frequency v[k], compute the value and

first two ordinary alpha derivatives of

E[exp(1j*v[k]*(M(tau)-M(0))) * 1{Q(tau)=j} | Q(0)=q].

The last axis enumerates the real frequencies, the middle two enumerate

initial and terminal queue states, and the leading axis contains derivative

orders 0,1,2. Order two is the second derivative, not half of it.



The state process is infinite; trajectories above max_state that return

before tau contribute. Do not renormalize the retained endpoint states.

Obtain the derivatives from the exact joint generating function through

analytic differentiation. Finite differences, numerical Fourier inversion,

simulation, time integration and finite-state generator exponentiation are

not permitted in this function. No diffusion factor or discount is included.

```python
import numpy as np

def queue_kernel_curvature(v,tau,lambda_jet,alpha,beta,mu_J,sigma_J,max_state):
    """Return a complex array with shape (3,max_state+1,max_state+1,len(v)).

    v: nonempty finite 1-D real array. lambda_jet: finite real shape (3,)
    with positive entry zero; its derivative entries may have either sign.
    Scalars: tau >= 0, beta > alpha > 0, sigma_J >= 0, finite mu_J.
    max_state: nonnegative integer. All scalar inputs must be finite.
    At tau=0, order zero is the identity and both derivatives are zero.
    Raise ValueError for invalid shapes, nonfinite inputs or invalid ranges.
    """
    return np.zeros((3,max_state+1,max_state+1,np.size(v)),dtype=complex)
```

### Step 3

03_diffusion_discount_factor

Goal
----
Transform and discount the independent continuous log-value increment. For a window of length tau, the continuous increment is normal with mean (mu-sigma**2/2)*tau and variance sigma**2*tau. Compute its characteristic function at the supplied real frequencies, multiplied by exp(-rho*tau). The drift mu and discount rate rho are separate project parameters.

```python
import numpy as np

def diffusion_discount_factor(v,tau,mu,sigma,rho):
    """Return a complex array with the same shape as finite real array v.

    tau and sigma are nonnegative finite scalars. mu and rho are finite.
    Raise ValueError for nonfinite inputs, negative tau or negative sigma.
    """
    return np.zeros(np.shape(v),dtype=complex)
```

### Step 4

04_call_payoff_cosine_coeffs

Goal
----
Implement $call_payoff_cosine_coeffs$ for a terminal call payoff. On a finite interval $[a_p, b_p]$ of the terminal log-value $y$, define the frequencies $w_k = k * pi / (b_p - a_p)$ for $k = 0, ..., N_in - 1$ and the projections of the payoff $max(exp(y) - K, 0)$ onto the shifted cosine basis, V_k = (2 / (b_p - a_p)) * integral over [a_p, b_p] of max(exp(y) - K, 0) * cos(w_k * (y - a_p)) dy .

Return the vector of those projections. They must be evaluated exactly rather than by numerical quadrature: the interval reaches log-values near 20 in the intended use, the integrand therefore reaches magnitudes near `1e8`, and a quadrature rule loses the leading digits of the alternating sum that the later steps form from these coefficients.

```python
import numpy as np

def call_payoff_cosine_coeffs(K, a_p, b_p, N_in):
    """Exact cosine projections of ``max(exp(y) - K, 0)`` on ``[a_p, b_p]``.

    Parameters
    ----------
    K : float
        Strictly positive strike of the terminal payoff.
    a_p : float
        Left endpoint of the interval.
    b_p : float
        Right endpoint of the interval, ``b_p > a_p``.
    N_in : int
        Number of coefficients, ``N_in >= 1``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(N_in,)`` holding ``V_0, ..., V_{N_in - 1}``. If
        ``log(K) >= b_p`` the payoff vanishes on the interval and the returned
        array is exactly zero.

    Raises
    ------
    ValueError
        If ``K <= 0``, ``b_p <= a_p``, or ``N_in`` is not an integer with
        ``N_in >= 1``.
    """
    return np.zeros(int(N_in), dtype=float)
```

### Step 5

05_state_continuation_jet

Goal
----
Propagate value coefficients and clustering derivatives through queue states. kernel_jet[d,q,j,k] contains the d-th ordinary alpha derivative, d=0,1,2, of the marked transition transform from state q to state j at frequency k. payoff_jet[d,j,k] contains the corresponding derivatives of the real cosine payoff coefficient in terminal state j. The complex diffusion vector is independent of alpha and already includes the window's discount factor.

Return the value and first two derivatives of the complex coefficients C[q,k] = diffusion[k] * sum_j kernel[0,q,j,k]*payoff[0,j,k] These are unprimed coefficients: the half weight of frequency zero is applied when evaluating the trigonometric expansion, not in this step.

```python
import numpy as np

def state_continuation_jet(kernel_jet,payoff_jet,diffusion):
    """Return complex shape (3,Q_initial,N).

    kernel_jet: finite complex shape (3,Q_initial,Q_terminal,N).
    payoff_jet: finite real shape (3,Q_terminal,N).
    diffusion: finite complex shape (N,). All dimensions are positive.
    Entries at leading index two are second derivatives, not Taylor
    coefficients divided by two. Raise ValueError for inconsistent shapes
    or nonfinite values.
    """
    return np.zeros((3,np.shape(kernel_jet)[1],np.size(diffusion)),dtype=complex)
```

### Step 6

06_queue_exercise_boundaries

Goal
----
Find the numerical exercise boundary separately for every activation state. For row q of C define V_q(x)=sum_k' Re(C[q,k]*exp(1j*omega[k]*(x-a_next))). The prime halves k=0. Solve V_q(x_q)=cost for every row on the supplied outer interval [a,b], using absolute and relative root tolerances 1e-12. Require V_q(a)<cost<V_q(b). Valid inputs have exactly one crossing and positive slope there; these properties are guaranteed apart from the explicitly checked endpoint bracket.

```python
import numpy as np

def queue_exercise_boundaries(C,omega,a_next,a,b,cost):
    """Return a float vector containing one root per row of C.

    C: finite complex shape (Q,N); omega: finite real shape (N,), starts
    at zero and strictly increases. Q,N are positive. a_next,a,b,cost are
    finite, b>a and cost>0. Raise ValueError for invalid shapes/ranges,
    nonfinite inputs, or failure of any strict endpoint bracket.
    """
    return np.zeros(np.shape(C)[0])
```

### Step 7

07_queue_payoff_jet

Goal
----
Project the compound payoff and its clustering derivatives analytically. For activation state q, C[d,q,k] is the d-th ordinary parameter derivative of the complex coefficient of a finite continuation curve V_q(x), d=0,1,2. The curve uses Re(C*exp(1j*omega*(x-a_next))), with k=0 half weighted. At nominal parameter alpha the supplied root r_q solves V_q(r_q)=cost, with positive x derivative. The cost, intervals and frequencies are fixed as alpha varies; the coefficients and exercise boundaries vary.

Return the value and first two alpha derivatives of the normalized outer projection 2/(b-a) integral from r_q(alpha) to b of (V_q(x,alpha)-cost)*cos(nu[n]*(x-a)) dx, for every q and outer mode n. Evaluate the integrals and boundary-motion terms analytically, without numerical quadrature or finite differences. Equal inner and outer frequencies, and both zero frequencies, are admissible.

```python
import numpy as np

def queue_payoff_jet(C,omega,nu,a_next,a,b,roots,cost):
    """Return finite float shape (3,Q,len(nu)), in derivative order 0,1,2.

    C: finite complex (3,Q,N), Q,N>0. omega: finite (N,) and nu: finite
    nonempty 1-D arrays, both starting at zero and strictly increasing.
    roots: finite (Q,), with a<roots[q]<b. Scalars a_next,a,b,cost are
    finite, b>a, cost>0. The nominal root identity is guaranteed for valid
    input. Raise ValueError for invalid shapes/ranges, nonfinite inputs,
    or nonpositive continuation slope at a supplied root.
    """
    return np.zeros((3,np.shape(C)[1],np.size(nu)))
```

### Step 8

08_persistent_clustering_curvature

Goal
----
Compute clustering curvature of a staged investment right at fixed jump count. At times[0],...,times[d-1] the firm may pay costs[i] to continue. At times[d] it receives max(S-Kterm,0). The continuous log increment over dt is normal with mean (mu-sigma**2/2)*dt and variance sigma**2*dt. The independent marked Queue-Hawkes component of step 2 runs over the entire horizon, without resetting activation at decision dates. Marks are uncompensated. Discount every time interval at rho. 

For each alpha near its nominal value, choose a single baseline arrival rate using matched_arrival_rate(times[-1],target,alpha,beta,q0). This global calibration is shared by all increments. Define V(alpha) using the finite analytic COS recursion, conditioned on every activation state 0,...,max_state at each date. Exact transition coefficients include excursions above the cutoff between dates. At any retained date, including terminal maturity, continuation/payoff is set to zero for states above the cutoff, without renormalization of the remaining states. Return d2 V/d alpha2. All other scalar parameters, date intervals, frequency counts and activation cutoff are held fixed. Differentiate the transition kernels, calibrated baseline and every moving exercise boundary analytically. Do not finite-difference prices or integrate intermediate payoffs numerically.

Use all seven preceding functions: matched_arrival_rate, queue_kernel_curvature, diffusion_discount_factor, call_payoff_cosine_coeffs, state_continuation_jet, queue_exercise_boundaries and queue_payoff_jet. Frequencies on each date are k*pi/(b-a), k=0,...,N-1. The terminal call coefficients are identical in every retained queue state and have zero alpha derivatives. At the initial state x=log(S0), use the usual half weight on frequency zero.

```python
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
```
