# Material_Science-Semiconductor_Materials-30

## Background

Charge capture and emission couple the conserved charge-state populations through an electrostatic barrier. The task combines this defect model with a linear-noise approximation and an occupancy-triggered readout. The Gaussian population dynamics and the centered quadratic capacitance detector define the numerical model. The latter is evaluated exactly within that model; it is not a complete second-order expansion of the underlying jump process. The coefficients, feedback rule and detector settings below are benchmark inputs, not measurements reported in the source paper.

## Problem

Identify family A assigned to EH1/EH3 before high-temperature annealing and family B assigned to S1/S2 after annealing at 623 K in the uploaded paper. Calculate the signed cross-spectral coherence of their DLTS fluctuations in the following benchmark.

Each family has charge states 0, -1, -2. The reduced state is ordered by family, with
$$p_{i0}=1-p_{i1}-p_{i2},\qquad Q=\sum_iw_i(p_{i1}+2p_{i2}).$$
Use the parameters below at 220 K. In this benchmark index 0 is the EH1/EH3 family, index 2 is the S1/S2 family, and index 1 is an unselected third family. Annealing determines the paper's defect assignments; it does not change these inputs.

| Index | w | sigma_01 (cm²) | sigma_12 (cm²) | E_01 (eV) | E_12 (eV) | g_01 | g_12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.46 | 2.0e-15 | 1.3e-15 | 0.40 | 0.46 | 2.0 | 0.5 |
| 1 | 0.31 | 1.1e-15 | 2.4e-15 | 0.42 | 0.48 | 1.0 | 2.0 |
| 2 | 0.23 | 3.2e-15 | 0.8e-15 | 0.44 | 0.45 | 0.5 | 1.0 |

The barrier and transition rates are
$$\psi+0.12\psi^3=1.7Q,$$
$$c=\sigma\,10^7\sqrt{T/300},\qquad e=c\,[2.8\times10^{19}(T/300)^{3/2}]\,g\exp[-E/(k_BT)],$$
where k_B=8.617333262145e-5 eV/K. Capture is suppressed by the barrier; emission is not:
$$u=cn\exp(-\psi),$$
$$\dot p_{i1}=u_{i0}p_{i0}-(e_{i0}+u_{i1})p_{i1}+e_{i1}p_{i2},\qquad
\dot p_{i2}=u_{i1}p_{i1}-e_{i1}p_{i2}.$$

The total cycle period is 0.016 s. Fill at n=2e10 cm⁻³ until the first upward crossing of psi=1, then read at n=1e8 cm⁻³ until the fixed cycle end. The population is continuous at switching. Select the attracting periodic branch reached by cycling from empty populations. Let tau be its fill duration. Read at fractions r_1=1/12 and r_2=3/4 of the remaining interval:
$$t_j=\tau+r_j(0.016-\tau).$$
The threshold, period and fractions are fixed. The switching time and both observation times fluctuate with occupancy.

There are N=500 traps, with N_i=Nw_i. On each smooth segment use the Gaussian linear-noise model
$$\xi=\sqrt N(p-\bar p),\qquad d\xi=A(\bar p)\xi\,dt+B(\bar p)dW,\qquad BB^T=D,\qquad A=\partial F/\partial p.$$
For family i, the four reduced increments are (1,0), (-1,0), (-1,1), (1,-1), with per-trap rates
$$a_i=(u_{i0}p_{i0},\ e_{i0}p_{i1},\ u_{i1}p_{i1},\ e_{i1}p_{i2}).$$
Each channel contributes a_i times the outer product of its increment divided by w_i to D. Reaction channels are independent. Linearize the threshold and the moving gates along the fluctuating trajectory. There is no extra switch noise. Preserve the correlations shared by the two gates and the next cycle start.

Let z_n contain all first-gate raw-charge fluctuations followed by all second-gate fluctuations, where raw family charge is p_i1+2p_i2 and fluctuations are scaled by sqrt(N). Write the resulting stationary Gaussian cycle model as
$$X_{n+1}=MX_n+\epsilon_{x,n},\qquad z_n=CX_n+\epsilon_{z,n},$$
with same-cycle innovation blocks Q_xx, Q_xz, Q_zz. Innovation pairs are independent between cycles and independent of X_n. Let R_0=Cov(z_n).

Use c_d(q)=(1+2q)^(-1/2) for the detector. Let L_i and sqrt(N)H_i be the gradient and ordinary Hessian, in gate-major raw-charge coordinates, of w_i[c_d(q_{i,1})-c_d(q_{i,2})] at the mean gates. The measured centered detector is the specified quadratic polynomial
$$Y_{i,n}=L_i z_n+\tfrac12[z_n^TH_i z_n-\operatorname{tr}(H_iR_0)].$$
Use this polynomial exactly on the Gaussian process. Do not add higher-order population corrections.

At omega=0.35 radians/cycle define
$$S(\omega)=\sum_{k=-\infty}^{\infty}e^{-i\omega k}\,\mathbb E[Y_{n+k}Y_n^T],\qquad
Z=\frac{\operatorname{Im}S_{BA}}{\sqrt{\operatorname{Re}S_{AA}\operatorname{Re}S_{BB}}}.$$
Use the infinite two-sided spectrum with no period or 1/(2pi) prefactor. Index order in S_BA matters.

Return Z. In the reasoning give the paper assignments, the role of concentration alongside level position and capture cross section, and the mechanism connected to incomplete divacancy filling in the paper's silicon example; p*, its unrounded cycle residual, tau and rho(M); the six mean gate charges; tr(Q_xx), Q_xz[0,0] and Q_xz[0,3]; the diagonal of stationary Cov(X); L[0,0], H[0,0,0]; Re(S_AA), Re(S_BB), Re(S_BA), Im(S_BA). Also give Z with H set to zero while retaining the same Gaussian cycle. Use zero-based indices and at least seven significant digits. Evaluate the cycle residual before rounding p*. State the detector lag covariance Gamma_k and the two-sided spectral construction used.

Output format

<final_answer>one finite decimal number</final_answer>
<reasoning>paper assignments, calculation and requested numerical diagnostics</reasoning>

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_emission_coefficients

Goal
----
Calculate transition-resolved capture and emission coefficients.

```python
def emission_coefficients(T, sigma, depth, degeneracy):
    """Return c,e of shape(m,2), in cm^3/s and s^-1 respectively.
    T>0 is in kelvin. sigma>0 is in cm^2, depth>=0 in eV, degeneracy>0 dimensionless;
    the three arrays have shape(m,2), m>=1. Use v=1e7*sqrt(T/300) cm/s,
    Nc=2.8e19*(T/300)^1.5 cm^-3 and kB=8.617333262145e-5 eV/K.
    c=sigma*v; e=c*Nc*degeneracy*exp(-depth/(kB*T)).
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return capture, emission
```

### Step 2

02_barrier_state

Goal
----
Solve the electrostatic barrier and its population gradient.

```python
def barrier_state(p, weights, alpha, eta):
    """Return float psi and its population gradient normal of shape(d,).
    p has d=2*m entries; weights has m positive entries summing to1. alpha>=0,
    eta>=0. Finite off-simplex p is allowed in this step. Q=sum_i weights[i]*
    (p[2*i]+2*p[2*i+1]); psi is the unique real root psi+eta*psi^3=alpha*Q.
    normal[j]=dpsi/dp[j] in ambient coordinates. Include alpha=0 and eta=0 limits.
    Only population derivatives are returned; there are no x,y coordinates.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return psi, normal
```

### Step 3

03_kinetic_jet

Goal
----
Construct the drift, coupled Jacobian and intrinsic diffusion matrix.

```python
def kinetic_lna(p, cfg, density):
    """cfg has T>0 (K); sigma, depth, degeneracy of shape(m,2), with sigma>0 (cm^2),
    depth>=0 (eV), degeneracy>0; weights shape(m,), strictly positive with sum 1;
    alpha>=0 and eta>=0, dimensionless. p has d=2*m entries, ordered
    (p_01,p_02,p_11,p_12,...); p_i0=1-p_i1-p_i2. p_i1,p_i2>=0 and p_i1+p_i2<=1.

    density>=0 is the electron density in cm^-3. Let c,e be Step 1 coefficients,
    psi Step 2 barrier and u=c*density*exp(-psi). For each family,
    F_i1=u_i0*p_i0-(e_i0+u_i1)*p_i1+e_i1*p_i2;
    F_i2=u_i1*p_i1-e_i1*p_i2. A=dF/dp includes the barrier dependence of u.
    The fluctuation coordinate is xi=sqrt(N)*(p_random-p_mean), with N_i=N*weights[i].
    The four channels have reduced stoichiometries (1,0),(-1,0),(-1,1),(1,-1)
    and per-trap rates u_i0*p_i0,e_i0*p_i1,u_i1*p_i1,e_i1*p_i2 respectively.
    D is the sum of rate*outer(stoichiometry,stoichiometry)/weights[i] embedded
    in each family's block. Distinct reaction channels have independent noise.
    Return F(d,), A(d,d), D(d,d). D is for xi, so it contains no factor 1/N.
    Emission is not suppressed by the barrier. No clipping or renormalization.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return drift, jacobian, diffusion
```

### Step 4

04_gaussian_flow

Goal
----
Propagate the mean, transition matrix and accumulated process covariance.

```python
def gaussian_flow(p, cfg, density, duration):
    """cfg has T>0 (K); sigma, depth, degeneracy of shape(m,2), with sigma>0 (cm^2),
    depth>=0 (eV), degeneracy>0; weights shape(m,), strictly positive with sum 1;
    alpha>=0 and eta>=0, dimensionless. p has d=2*m entries, ordered
    (p_01,p_02,p_11,p_12,...); p_i0=1-p_i1-p_i2. p_i1,p_i2>=0 and p_i1+p_i2<=1.

    density>=0; duration>=0 seconds. Along q'=F(q), use A(q),D(q) from kinetic_lna.
    The Gaussian fluctuation solves dxi=A(q)xi dt+B(q)dW, with B B^T=D(q).
    Return q_end(d,), Phi(d,d), V(d,d) such that xi_end=Phi*xi_start+epsilon,
    Cov(epsilon)=V and epsilon is independent of xi_start. Coefficients vary along q.
    At duration 0 return p.copy(), identity, zeros. Phi initially identity; V initially 0.
    Raises
    ------
    ValueError: if numerical integration fails.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return endpoint, transition, process_covariance
```

### Step 5

05_hybrid_cycle

Goal
----
Form one cycle with joint fluctuations at two moving read gates.

```python
def hybrid_cycle(p, cfg, protocol, gates):
    """cfg has T>0 (K); sigma, depth, degeneracy of shape(m,2), with sigma>0 (cm^2),
    depth>=0 (eV), degeneracy>0; weights shape(m,), strictly positive with sum 1;
    alpha>=0 and eta>=0, dimensionless. p has d=2*m entries, ordered
    (p_01,p_02,p_11,p_12,...); p_i0=1-p_i1-p_i2. p_i1,p_i2>=0 and p_i1+p_i2<=1.

    Here alpha>0. protocol has period>0 seconds, level>0 dimensionless,
    fill_density>0 and read_density>=0 in cm^-3. gates is exactly two fractions
    0<=r1<=r2<=1. Fill until the first upward crossing psi=level at tau; read until
    the fixed end time period. The mean state is continuous at the switch.
    Observe each family's raw charge p_i1+2*p_i2 at
     t_j=tau+r_j*(period-tau).
    These observation times follow the perturbed event, not the nominal clock.
    Use the leading Gaussian linearization of this hybrid process: on each smooth
    segment xi obeys the SDE of gaussian_flow; linearize the threshold crossing
    and the observation times with respect to initial fluctuations and all earlier
    noise. The threshold, total period and fractions are fixed. There is no state
    reset, added switch noise or independent timing-noise source. Noise increments
    on disjoint nominal smooth intervals are independent. Retain correlations
    caused by their shared history and shared event-time fluctuation.
    For arbitrary deterministic initial p, define
     xi_next=M*xi_start+epsilon_x,
     z=C*xi_start+epsilon_z,
    where z has 2*m entries: all first-gate charge fluctuations followed by all
    second-gate charge fluctuations, scaled by sqrt(N); no weights in raw charges.
    The joint innovation is independent of xi_start. Qxx=Cov(epsilon_x),
    Qxz=Cov(epsilon_x,epsilon_z), Qzz=Cov(epsilon_z). Gate-state noise and endpoint
    noise are generally correlated. Return exactly eight values in this order:
     endpoint(d,), M(d,d), C(2*m,d), Qxx(d,d), Qxz(d,2*m), Qzz(2*m,2*m),
     mean_gates(2*m,), tau(float seconds).
    Coincident gates and gate endpoints 0,1 are supported; covariance may be singular.
    For normal inputs the first crossing is unique, transversal and interior.
    Raises
    ------
    ValueError: if initial psi>=level, no upward event occurs strictly before period,
    the event is not transversal/upward, or a required integration fails.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return endpoint, M, C, Qxx, Qxz, Qzz, mean_gates, tau
```

### Step 6

06_periodic_state

Goal
----
Find the attracting mean state at the start of the fill phase.

```python
def periodic_state(cfg, protocol, gates):
    """cfg has T>0 (K); sigma, depth, degeneracy of shape(m,2), with sigma>0 (cm^2),
    depth>=0 (eV), degeneracy>0; weights shape(m,), strictly positive with sum 1;
    alpha>=0 and eta>=0, dimensionless. p has d=2*m entries, ordered
    (p_01,p_02,p_11,p_12,...); p_i0=1-p_i1-p_i2. p_i1,p_i2>=0 and p_i1+p_i2<=1.

    protocol and gates have the meanings and domain from hybrid_cycle. alpha>0.
    Return pstar(d,) satisfying hybrid_cycle(pstar,...)[0]=pstar. Select the unique
    attracting branch reached by repeated cycles from empty populations. All startup
    cycles and a neighbourhood of this fixed point have unique interior transversal
    events; the cycle Jacobian has spectral radius<=.98. Return the fill-start phase.
    The infinity-norm cycle residual must be <=2e-9 on the unrounded state.
    Raises
    ------
    ValueError: if a required hybrid cycle fails, a physical periodic state cannot
    be obtained, or the fixed-point computation does not converge.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return periodic_population
```

### Step 7

07_stationary_covariance

Goal
----
Solve the stationary covariance of the cycle-start Gaussian state.

```python
def stationary_covariance(M, Qxx):
    """M is a real square matrix of dimension d>=1, with spectral radius<=.98.
    Qxx is real symmetric positive semidefinite with shape(d,d). Return symmetric
    P(d,d) satisfying P=M P M.T+Qxx. Zero and singular Qxx are supported. M may be
    nonnormal and may have negative eigenvalues. P is the covariance of the scaled
    cycle-start population fluctuation, with no extra population-size factor.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return covariance
```

### Step 8

08_detector_coefficients

Goal
----
Expand the family-resolved capacitance difference to quadratic order.

```python
def detector_coefficients(mean_gates, weights, beta, population):
    """mean_gates is length 2*m in gate-major order, weights lengthm positive summing1.
    beta>=0 is a scalar; population>0 is N. Every 1+beta*mean_gates entry is positive.
    For c(q)=(1+beta*q)^(-1/2), the measured family difference is
    weights[i]*(c(q_first_i)-c(q_second_i)). Let z be the sqrt(N)-scaled raw
    gate-charge fluctuation. Return L(m,2*m), H(m,2*m,2*m) defining
     y_i=L[i]@z+0.5*(z.T@H[i]@z-tr(H[i]@Cov(z))).
    L is the gradient of that difference at mean_gates. H is its Hessian divided
    by sqrt(population). Derivatives are ordinary derivatives, so do not put the
    Taylor factor 1/2 inside H. This specified quadratic model is used exactly;
    it is not a request for higher-order corrections to population dynamics.
    beta=0 gives zero coefficients. H[i] is symmetric.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return linear, quadratic
```

### Step 9

09_quadratic_spectrum

Goal
----
Compute the full discrete-frequency spectrum of a quadratic Gaussian readout

```python
def quadratic_spectrum(M, C, Qxx, Qxz, Qzz, L, H, omega):
    """All matrices are real. M(d,d) has spectral radius<=.98; C(g,d),
    Qxx(d,d), Qxz(d,g), Qzz(g,g), L(o,g), H(o,g,g), with d,g,o>=1.
    The joint innovation covariance [[Qxx,Qxz],[Qxz.T,Qzz]] is symmetric PSD.
    Each H[a] is symmetric; it may be dense and indefinite. omega is any finite
    real number in radians/cycle. Consider the stationary zero-mean Gaussian model
     x_(n+1)=M*x_n+epsilon_x,n, z_n=C*x_n+epsilon_z,n.
    Innovation pairs have the stated covariance, are independent between cycles,
    and independent of x_n. Define R0=Cov(z_n) and the centered detector
     y_a,n=L[a]@z_n+0.5*(z_n.T@H[a]@z_n-tr(H[a]@R0)).
    Return real_part(o,o), imag_part(o,o) of the infinite two-sided spectrum
     S(omega)=sum over integer k of exp(-1j*omega*k)*Cov(y_(n+k),y_n).
    Cov(y_(n+k),y_n) means E[y_(n+k) y_n.T]. No 1/(2*pi) or sampling-period
    factor is applied. Keep the complex cross-spectrum, including its sign.
    Correlated process/measurement innovations, M=0, L=0, H=0, singular noise,
    omega=0 and omega=pi are supported. There is no finite lag cut-off in the
    specified answer. The earlier stationary_covariance function is available.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return real_part, imaginary_part
```

### Step 10

10_dlts_noise

Goal
----
Calculate the signed cross-spectral coherence between two defect families.

```python
def dlts_noise(cfg, protocol, gates, beta, population, omega, numerator, denominator):
    """cfg has T>0 (K); sigma, depth, degeneracy of shape(m,2), with sigma>0 (cm^2),
    depth>=0 (eV), degeneracy>0; weights shape(m,), strictly positive with sum 1;
    alpha>=0 and eta>=0, dimensionless. p has d=2*m entries, ordered
    (p_01,p_02,p_11,p_12,...); p_i0=1-p_i1-p_i2. p_i1,p_i2>=0 and p_i1+p_i2<=1.

    Use periodic_state, hybrid_cycle, detector_coefficients and quadratic_spectrum
    with their stated conventions. protocol/gates obey the stable event-controlled
    domain; beta>=0 scalar, population>0, omega finite radians/cycle.
    numerator and denominator are valid zero-based family indices, possibly equal.
    Let S be the stationary spectrum of the specified centered quadratic detector.
    Return float Im(S[numerator,denominator])/
    sqrt(Re(S[numerator,numerator])*Re(S[denominator,denominator])).
    Raises
    ------
    ValueError: if a prerequisite fails or either selected auto-spectrum is nonpositive,
    including zero-signal choices such as identical gates or beta=0.
    All inputs are finite; numerical arrays may be array-like. Do not mutate inputs.
    Import dependencies inside the function; NumPy and SciPy are available.
    Earlier public functions are available. Candidate code calls their public names;
    reference functions are supplied separately and are not candidate dependencies.
    """
    return signed_coherence
```
